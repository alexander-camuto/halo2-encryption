use crate::add_chip::{AddChip, AddConfig, AddInstruction};
use ark_std::rand::rngs::OsRng;
use ark_std::rand::{CryptoRng, RngCore};
use ecc::integer::rns::{Common, Integer, Rns};
use ecc::maingate::{
    MainGate, MainGateConfig, RangeChip, RangeConfig, RangeInstructions, RegionCtx,
};
use ecc::{AssignedPoint, BaseFieldEccChip, EccConfig};
use ezkl_lib::circuit::modules::poseidon::spec::PoseidonSpec;
use halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
};
use halo2_proofs::arithmetic::Field;
use halo2_proofs::circuit::{AssignedCell, Chip, Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk;
use halo2_proofs::plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance};
use halo2curves::bn256::{Fq, Fr, G1Affine, G1};
use halo2curves::group::{Curve, Group};
use halo2curves::CurveAffine;
use std::ops::{Mul, MulAssign};
use std::rc::Rc;
use std::vec;

// Absolute offsets for public inputs.
const C1_X: usize = 0;
const C1_Y: usize = 1;

///
const NUMBER_OF_LIMBS: usize = 4;
const BIT_LEN_LIMB: usize = 64;

type CircuitCipher = (
    AssignedPoint<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    AssignedCell<Fr, Fr>,
);

type CircuitHash = PoseidonHash<Fr, PoseidonChip<Fr, 2, 1>, PoseidonSpec, ConstantLength<2>, 2, 1>;

pub struct ElGamalChip {
    config: ElGamalConfig,
    ecc: BaseFieldEccChip<G1Affine, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    poseidon: PoseidonChip<Fr, 2, 1>,
    add: AddChip,
}

#[derive(Debug, Clone)]
pub struct ElGamalConfig {
    main_gate_config: MainGateConfig,
    range_config: RangeConfig,
    poseidon_config: PoseidonConfig<Fr, 2, 1>,
    add_config: AddConfig,
    plaintext_col: Column<Advice>,
    ciphertext_c1_exp_col: Column<Instance>,
    ciphertext_c2_exp_col: Column<Instance>,
    sk_commitment_col: Column<Instance>,
}

impl ElGamalConfig {
    fn config_range(&self, layouter: &mut impl Layouter<Fr>) -> Result<(), Error> {
        let range_chip = RangeChip::<Fr>::new(self.range_config.clone());
        range_chip.load_table(layouter)?;

        Ok(())
    }

    fn ecc_chip_config(&self) -> EccConfig {
        EccConfig::new(self.range_config.clone(), self.main_gate_config.clone())
    }
}

impl Chip<Fq> for ElGamalChip {
    type Config = ElGamalConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl ElGamalChip {
    pub fn new(p: ElGamalConfig) -> ElGamalChip {
        ElGamalChip {
            ecc: BaseFieldEccChip::new(p.ecc_chip_config().clone()),
            poseidon: PoseidonChip::construct(p.poseidon_config.clone()),
            add: AddChip::construct(p.add_config.clone()),
            config: p,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> ElGamalConfig {
        let advices = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // let table_idx = meta.lookup_table_column();

        // Poseidon requires four advice columns, while ECC incomplete addition requires
        // six, so we could choose to configure them in parallel. However, we only use a
        // single Poseidon invocation, and we have the rows to accommodate it serially.
        // Instead, we reduce the proof size by sharing fixed columns between the ECC and
        // Poseidon chips.
        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        let rc_a = lagrange_coeffs[0..2].try_into().unwrap();
        let rc_b = lagrange_coeffs[2..4].try_into().unwrap();

        // Also use the first Lagrange coefficient column for loading global constants.
        // It's free real estate :)
        meta.enable_constant(lagrange_coeffs[3]);

        let rns = Rns::<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>::construct();

        let main_gate_config = MainGate::<Fr>::configure(meta);
        let overflow_bit_lens = rns.overflow_lengths();
        let composition_bit_lens = vec![BIT_LEN_LIMB / NUMBER_OF_LIMBS];

        let range_config = RangeChip::<Fr>::configure(
            meta,
            &main_gate_config,
            composition_bit_lens,
            overflow_bit_lens,
        );

        let poseidon_config = PoseidonChip::configure::<PoseidonSpec>(
            meta,
            advices[1..3].try_into().unwrap(),
            advices[0],
            rc_a,
            rc_b,
        );

        let dh_col = meta.advice_column();
        meta.enable_equality(dh_col);
        let plaintext_col = meta.advice_column();
        meta.enable_equality(plaintext_col);
        let ciphertext_res_col = meta.advice_column();
        meta.enable_equality(ciphertext_res_col);

        let add_config = AddChip::configure(meta, dh_col, plaintext_col, ciphertext_res_col);

        let ciphertext_c1_exp_col = meta.instance_column();
        meta.enable_equality(ciphertext_c1_exp_col);

        let ciphertext_c2_exp_col = meta.instance_column();
        meta.enable_equality(ciphertext_c2_exp_col);

        let sk_commitment_col = meta.instance_column();
        meta.enable_equality(sk_commitment_col);

        ElGamalConfig {
            poseidon_config,
            main_gate_config,
            range_config,
            add_config,
            plaintext_col,
            ciphertext_c1_exp_col,
            ciphertext_c2_exp_col,
            sk_commitment_col,
        }
    }
}

#[derive(Default, Clone)]
pub struct ElGamalGadget {
    r: Fr,
    msg: Vec<Fr>,
    pk: G1Affine,
    sk: Fr,
    pub resulted_ciphertext: (G1, Vec<Fr>),
    pub sk_hash: Fr,
    aux_generator: G1Affine,
    window_size: usize,
}

impl ElGamalGadget {
    pub fn new(r: Fr, msg: Vec<Fr>, pk: G1Affine, sk: Fr) -> ElGamalGadget {
        let resulted_ciphertext = Self::encrypt(pk.clone(), msg.clone(), r.clone());
        let sk_hash = Self::hash_sk(sk.clone());
        let aux_generator = <G1Affine as CurveAffine>::CurveExt::random(OsRng).to_affine();
        return Self {
            r,
            msg,
            pk,
            sk,
            resulted_ciphertext,
            sk_hash,
            aux_generator,
            window_size: 1,
        };
    }

    fn rns() -> Rc<Rns<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>> {
        let rns = Rns::<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>::construct();
        Rc::new(rns)
    }

    pub fn keygen<R: CryptoRng + RngCore>(mut rng: &mut R) -> anyhow::Result<(Fr, G1)> {
        // get a random element from the scalar field
        let secret_key = Fr::random(&mut rng);

        // compute secret_key*generator to derive the public key
        // With BN256, we create the private key from a random number. This is a private key value (sk
        //  and a public key mapped to the G2 curve:: pk=sk.G2
        let mut public_key = G1::generator();
        public_key.mul_assign(secret_key.clone());

        Ok((secret_key, public_key))
    }

    pub fn encrypt(pk: G1Affine, msg: Vec<Fr>, r: Fr) -> (G1, Vec<Fr>) {
        let g = G1Affine::generator();
        let c1 = g.mul(&r);

        let coords = pk.mul(&r).to_affine().coordinates().unwrap();

        let x = Integer::from_fe(*coords.x(), Self::rns());
        let y = Integer::from_fe(*coords.y(), Self::rns());

        let hasher = poseidon::Hash::<Fr, PoseidonSpec, ConstantLength<2>, 2, 1>::init();
        let dh = hasher.hash([x.native().clone(), y.native().clone()]); // this is Fq now :( (we need Fr)

        let mut c2 = vec![];

        for i in 0..msg.len() {
            c2.push(msg[i] + dh);
        }

        return (c1, c2);
    }

    pub fn hash_sk(sk: Fr) -> Fr {
        let hasher = poseidon::Hash::<Fr, PoseidonSpec, ConstantLength<2>, 2, 1>::init();
        let dh = hasher.hash([sk.clone(), sk.clone()]); // this is Fq now :( (we need Fr)
        dh
    }

    pub fn decrypt(cipher: &(G1, Vec<Fr>), sk: Fr) -> Vec<Fr> {
        let c1 = cipher.0.clone();
        let c2 = cipher.1.clone();

        let s = c1.mul(sk).to_affine().coordinates().unwrap();

        let x = Integer::from_fe(*s.x(), Self::rns());
        let y = Integer::from_fe(*s.y(), Self::rns());

        let hasher = poseidon::Hash::<Fr, PoseidonSpec, ConstantLength<2>, 2, 1>::init();
        let dh = hasher.hash([x.native().clone(), y.native().clone()]); // this is Fq now :( (we need Fr)

        let mut msg = vec![];
        for i in 0..c2.len() {
            msg.push(c2[i] - dh);
        }

        return msg;
    }

    pub fn get_instances(cipher: &(G1, Vec<Fr>), sk_hash: Fr) -> Vec<Vec<Fr>> {
        let c1_coordinates = cipher
            .0
            .to_affine()
            .coordinates()
            .map(|c| {
                let x = Integer::from_fe(*c.x(), Self::rns());
                let y = Integer::from_fe(*c.y(), Self::rns());

                vec![x.native().clone(), y.native().clone()]
            })
            .unwrap();

        vec![c1_coordinates, cipher.1.clone(), vec![sk_hash]]
    }

    pub(crate) fn verify_sk_hash(
        &self,
        mut layouter: impl Layouter<Fr>,
        config: ElGamalConfig,
        sk: &AssignedCell<Fr, Fr>,
    ) -> Result<AssignedCell<Fr, Fr>, plonk::Error> {
        let chip = ElGamalChip::new(config.clone());

        // compute dh = poseidon_hash(randomness*pk)
        let sk_hash = {
            let poseidon_hasher =
                CircuitHash::init(chip.poseidon, layouter.namespace(|| "Poseidon hasher"))?;
            poseidon_hasher.hash(
                layouter.namespace(|| "Poseidon hash (sk)"),
                [sk.clone(), sk.clone()],
            )?
        };

        Ok(sk_hash)
    }

    pub(crate) fn verify_encryption(
        &self,
        mut layouter: impl Layouter<Fr>,
        config: &ElGamalConfig,
        m: &AssignedCell<Fr, Fr>,
        sk: &AssignedCell<Fr, Fr>,
    ) -> Result<CircuitCipher, plonk::Error> {
        let mut chip = ElGamalChip::new(config.clone());

        let g = G1Affine::generator();
        let c1 = g.mul(self.r).to_affine();

        // compute s = randomness*pk
        let s = self.pk.clone().mul(self.r).to_affine();

        let (s, c1) = layouter.assign_region(
            || "region 0",
            |region| {
                let offset = 0;
                let ctx = &mut RegionCtx::new(region, offset);

                chip.ecc
                    .assign_aux_generator(ctx, Value::known(self.aux_generator))?;
                chip.ecc.assign_aux(ctx, self.window_size, 1)?;

                let s = chip.ecc.assign_point(ctx, Value::known(s)).unwrap();

                // compute c1 = randomness*generator
                let c1 = chip.ecc.assign_point(ctx, Value::known(c1)).unwrap();

                let s_from_sk = chip.ecc.mul(ctx, &c1, sk, self.window_size).unwrap();

                chip.ecc.assert_equal(ctx, &s, &s_from_sk)?;

                Ok((s, c1))
            },
        )?;

        // compute dh = poseidon_hash(randomness*pk)
        let dh = {
            let poseidon_message = [s.x().native().clone(), s.y().native().clone()];
            let poseidon_hasher =
                CircuitHash::init(chip.poseidon, layouter.namespace(|| "Poseidon hasher"))?;
            poseidon_hasher.hash(
                layouter.namespace(|| "Poseidon hash (randomness*pk)"),
                poseidon_message,
            )?
        };

        // compute c2 = poseidon_hash(nk, rho) + psi.
        let c2 = chip.add.add(
            layouter.namespace(|| "c2 = poseidon_hash(randomness*pk) + m"),
            &dh,
            m,
        )?;

        Ok((c1, c2))
    }

    fn layout(&self, config: Self::Config, mut layouter: impl Layouter<Fr>) -> Result<(), Error> {
        config.config_range(&mut layouter)?;

        let (msg_var, sk_var) = layouter.assign_region(
            || "plaintext",
            |mut region| {
                let msg_var: Result<Vec<_>, _> = self
                    .msg
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        region.assign_advice(
                            || format!("plaintext {}", i),
                            config.plaintext_col,
                            i,
                            || Value::known(*m),
                        )
                    })
                    .collect();

                let sk_var = region.assign_advice(
                    || "sk",
                    config.plaintext_col,
                    self.msg.len(),
                    || Value::known(self.sk),
                )?;

                Ok((msg_var?, sk_var))
            },
        )?;

        // Force the public input to be the hash of the secret key so that we can ascertain decryption can happen
        let sk_hash = self.verify_sk_hash(
            layouter.namespace(|| "verify_sk_hash"),
            config.clone(),
            &sk_var,
        )?;
        layouter.constrain_instance(sk_hash.cell(), config.sk_commitment_col, 0)?;

        for i in 0..msg_var.len() {
            let cipher = self.verify_encryption(
                layouter.namespace(|| "verify_encryption"),
                &config,
                &msg_var[i],
                &sk_var,
            )?;

            let c1 = cipher.0;
            let c2 = cipher.1;

            layouter
                .constrain_instance(c1.x().native().cell(), config.ciphertext_c1_exp_col, C1_X)
                .and(layouter.constrain_instance(
                    c1.y().native().cell(),
                    config.ciphertext_c1_exp_col,
                    C1_Y,
                ))
                .and(layouter.constrain_instance(c2.cell(), config.ciphertext_c2_exp_col, i))?;
        }
        Ok(())
    }
}

impl Circuit<Fr> for ElGamalGadget {
    type Config = ElGamalConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();
    fn without_witnesses(&self) -> Self {
        Self::default()
    }
    //type Config = EccConfig;
    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        ElGamalChip::configure(cs)
    }
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        self.layout(config, layouter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;

    #[test]
    pub fn test_encrypt_decrypt() {
        let mut rng = test_rng();

        let (sk, pk) = ElGamalGadget::keygen(&mut rng).unwrap();

        let r = Fr::random(&mut rng);

        let mut msg = vec![];
        //
        for _ in 0..32 {
            msg.push(Fr::random(&mut rng));
        }

        let circuit = ElGamalGadget::new(r, msg, pk.to_affine(), sk);
        let cipher_text = circuit.resulted_ciphertext.clone();

        let decrypted_msg = ElGamalGadget::decrypt(&cipher_text, sk);

        assert_eq!(decrypted_msg, circuit.msg);
    }
}
