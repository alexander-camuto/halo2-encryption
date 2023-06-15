use crate::add_chip::{AddChip, AddConfig, AddInstruction};
use ark_std::rand::rngs::OsRng;
use ark_std::rand::{CryptoRng, RngCore};
use ecc::integer::rns::{Common, Integer, Rns};
use ecc::integer::NUMBER_OF_LOOKUP_LIMBS;
use ecc::maingate::{
    MainGate, MainGateConfig, RangeChip, RangeConfig, RangeInstructions, RegionCtx,
};
use ecc::{AssignedPoint, BaseFieldEccChip, EccConfig, GeneralEccChip, Point};
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
use halo2curves::group::Curve;
use halo2curves::group::Group;
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

fn rns<C: CurveAffine>() -> Rns<C::Base, C::Scalar, NUMBER_OF_LIMBS, BIT_LEN_LIMB> {
    Rns::construct()
}

fn setup<C: CurveAffine>(
    k_override: u32,
) -> (Rns<C::Base, C::Scalar, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, u32) {
    let rns = rns::<C>();
    let bit_len_lookup = BIT_LEN_LIMB / NUMBER_OF_LOOKUP_LIMBS;
    let mut k: u32 = (bit_len_lookup + 1) as u32;
    if k_override != 0 {
        k = k_override;
    }
    (rns, k)
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
            meta.fixed_column(),
        ];
        let rc_a = lagrange_coeffs[0..2].try_into().unwrap();
        let rc_b = lagrange_coeffs[2..4].try_into().unwrap();

        // Also use the first Lagrange coefficient column for loading global constants.
        // It's free real estate :)
        meta.enable_constant(lagrange_coeffs[4]);

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

        ElGamalConfig {
            poseidon_config,
            main_gate_config,
            range_config,
            add_config,
            plaintext_col,
            ciphertext_c1_exp_col,
            ciphertext_c2_exp_col,
        }
    }
}

#[derive(Default, Clone)]
pub struct ElGamalGadget {
    r: Fr,
    msg: Vec<Fr>,
    pk: G1Affine,
    pub resulted_ciphertext: (G1, Vec<Fr>),
}

impl ElGamalGadget {
    pub fn new(r: Fr, msg: Vec<Fr>, pk: G1Affine) -> ElGamalGadget {
        let resulted_ciphertext = Self::encrypt(pk.clone(), msg.clone(), r.clone());
        return Self {
            r,
            msg,
            pk,
            resulted_ciphertext,
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

    pub fn get_instances(cipher: &(G1, Vec<Fr>)) -> Vec<Vec<Fr>> {
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

        vec![c1_coordinates, cipher.1.clone()]
    }

    pub(crate) fn verify_encryption(
        &self,
        mut layouter: impl Layouter<Fr>,
        config: ElGamalConfig,
        m: &AssignedCell<Fr, Fr>,
    ) -> Result<
        (
            AssignedPoint<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
            AssignedCell<Fr, Fr>,
        ),
        plonk::Error,
    > {
        let chip = ElGamalChip::new(config.clone());

        let g = G1Affine::generator();
        let c1 = g.mul(self.r).to_affine();

        // compute s = randomness*pk
        let s = self.pk.clone().mul(self.r).to_affine();

        let (s, c1) = layouter.assign_region(
            || "region 0",
            |region| {
                let offset = 0;
                let ctx = &mut RegionCtx::new(region, offset);

                let s = chip.ecc.assign_point(ctx, Value::known(s)).unwrap();

                // compute c1 = randomness*generator
                let c1 = chip.ecc.assign_point(ctx, Value::known(c1)).unwrap();

                Ok((s, c1))
            },
        )?;

        // compute dh = poseidon_hash(randomness*pk)
        let dh = {
            let poseidon_message = [s.x().native().clone(), s.y().native().clone()];
            let poseidon_hasher = PoseidonHash::<
                Fr,
                PoseidonChip<Fr, 2, 1>,
                PoseidonSpec,
                ConstantLength<2>,
                2,
                1,
            >::init(
                chip.poseidon, layouter.namespace(|| "Poseidon hasher")
            )?;
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

        config.config_range(&mut layouter)?;

        Ok((c1, c2))
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
        let msg_var = layouter.assign_region(
            || "plaintext",
            |mut region| {
                let res: Result<Vec<_>, _> = self
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
                res
            },
        )?;

        for i in 0..msg_var.len() {
            let (c1, c2) = self.verify_encryption(
                layouter.namespace(|| "verify_encryption"),
                config.clone(),
                &msg_var[i],
            )?;
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
