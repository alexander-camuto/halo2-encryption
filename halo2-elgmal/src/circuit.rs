use crate::add_chip::{AddChip, AddConfig, AddInstruction};
use crate::constants::TestFixedBases;
use ark_std::rand::{CryptoRng, RngCore};
use halo2_gadgets::ecc::chip::{EccChip, EccConfig, EccPoint};
use halo2_gadgets::ecc::EccInstructions;
use halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
};
use halo2_gadgets::utilities::lookup_range_check::LookupRangeCheckConfig;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::circuit::{AssignedCell, Chip, Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk;
use halo2_proofs::plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance};
use halo2curves::group::{Curve, Group};
use halo2curves::pasta::pallas;
use halo2curves::CurveAffine;
use std::ops::{Mul, MulAssign};
use std::vec;

// Absolute offsets for public inputs.
const C1_X: usize = 0;
const C1_Y: usize = 0;

pub struct ElGamalChip {
    config: ElGamalConfig,
    ecc: EccChip<TestFixedBases>,
    poseidon: PoseidonChip<pallas::Base, 3, 2>,
    add: AddChip,
}

#[derive(Debug, Clone)]
pub struct ElGamalConfig {
    ecc_config: EccConfig<TestFixedBases>,
    poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    add_config: AddConfig,
    plaintext_col: Column<Advice>,
    ciphertext_c1x_exp_col: Column<Instance>,
    ciphertext_c1y_exp_col: Column<Instance>,
    ciphertext_c2_exp_col: Column<Instance>,
}

impl Chip<pallas::Base> for ElGamalChip {
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
            ecc: EccChip::construct(p.ecc_config.clone()),
            poseidon: PoseidonChip::construct(p.poseidon_config.clone()),
            add: AddChip::construct(p.add_config.clone()),
            config: p,
        }
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> ElGamalConfig {
        let advices = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        let table_idx = meta.lookup_table_column();

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
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        let rc_a = lagrange_coeffs[2..5].try_into().unwrap();
        let rc_b = lagrange_coeffs[5..8].try_into().unwrap();

        // Also use the first Lagrange coefficient column for loading global constants.
        // It's free real estate :)
        meta.enable_constant(lagrange_coeffs[0]);

        // Shared fixed column for loading constants
        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], table_idx);
        let ecc_config =
            EccChip::<TestFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);

        let poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
            meta,
            advices[6..9].try_into().unwrap(),
            advices[5],
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

        let ciphertext_c1x_exp_col = meta.instance_column();
        let ciphertext_c1y_exp_col = meta.instance_column();
        meta.enable_equality(ciphertext_c1x_exp_col);
        meta.enable_equality(ciphertext_c1y_exp_col);

        let ciphertext_c2_exp_col = meta.instance_column();
        meta.enable_equality(ciphertext_c2_exp_col);

        ElGamalConfig {
            poseidon_config,
            ecc_config,
            add_config,
            plaintext_col,
            ciphertext_c1x_exp_col,
            ciphertext_c1y_exp_col,
            ciphertext_c2_exp_col,
        }
    }
}

#[derive(Default)]
pub struct ElGamalGadget {
    r: pallas::Scalar,
    msg: Vec<pallas::Base>,
    pk: pallas::Point,
    pub resulted_ciphertext: (pallas::Point, Vec<pallas::Base>),
}

impl ElGamalGadget {
    pub fn new(r: pallas::Scalar, msg: Vec<pallas::Base>, pk: pallas::Point) -> ElGamalGadget {
        let resulted_ciphertext = Self::encrypt(pk.clone(), msg.clone(), r.clone());
        return Self {
            r,
            msg,
            pk,
            resulted_ciphertext,
        };
    }

    pub fn keygen<R: CryptoRng + RngCore>(
        mut rng: &mut R,
    ) -> anyhow::Result<(pallas::Scalar, pallas::Point)> {
        // get a random element from the scalar field
        let secret_key = pallas::Scalar::random(&mut rng);

        // compute secret_key*generator to derive the public key
        let mut public_key = pallas::Point::generator();
        public_key.mul_assign(secret_key.clone());

        Ok((secret_key, public_key))
    }

    pub fn encrypt(
        pk: pallas::Point,
        msg: Vec<pallas::Base>,
        r: pallas::Scalar,
    ) -> (pallas::Point, Vec<pallas::Base>) {
        let c1 = pallas::Point::generator().mul(&r);
        let p_ra = pk.mul(&r).to_affine().coordinates().unwrap();

        let hasher =
            poseidon::Hash::<pallas::Base, poseidon::P128Pow5T3, ConstantLength<2>, 3, 2>::init();
        let dh = hasher.hash([p_ra.x().clone(), p_ra.y().clone()]);

        let mut c2 = vec![];

        for i in 0..msg.len() {
            c2.push(msg[i] + dh);
        }

        return (c1, c2);
    }

    pub fn get_instances(cipher: &(pallas::Point, Vec<pallas::Base>)) -> Vec<Vec<pallas::Base>> {
        let c1_coordinates = cipher
            .0
            .to_affine()
            .coordinates()
            .map(|c| vec![c.x().clone(), c.y().clone()])
            .unwrap();

        vec![
            vec![c1_coordinates[0]],
            vec![c1_coordinates[1]],
            cipher.1.clone(),
        ]
    }

    pub(crate) fn verify_encryption(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        config: ElGamalConfig,
        m: &AssignedCell<pallas::Base, pallas::Base>,
    ) -> Result<(EccPoint, AssignedCell<pallas::Base, pallas::Base>), plonk::Error> {
        let chip = ElGamalChip::new(config.clone());

        let g = pallas::Point::generator();

        // compute s = randomness*pk
        let s = self.pk.clone().mul(self.r).to_affine();
        let s = chip
            .ecc
            .witness_point(&mut layouter, Value::known(s))
            .unwrap();

        // compute c1 = randomness*generator
        let c1 = g.mul(self.r).to_affine();
        let c1 = chip
            .ecc
            .witness_point(&mut layouter, Value::known(c1))
            .unwrap();

        // compute dh = poseidon_hash(randomness*pk)
        let dh = {
            let poseidon_message = [s.x(), s.y()];
            let poseidon_hasher =
                PoseidonHash::<
                    pallas::Base,
                    PoseidonChip<pallas::Base, 3, 2>,
                    poseidon::P128Pow5T3,
                    ConstantLength<2>,
                    3,
                    2,
                >::init(chip.poseidon, layouter.namespace(|| "Poseidon hasher"))?;
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
}

impl Circuit<pallas::Base> for ElGamalGadget {
    type Config = ElGamalConfig;
    type FloorPlanner = SimpleFloorPlanner;
    fn without_witnesses(&self) -> Self {
        Self::default()
    }
    //type Config = EccConfig;
    fn configure(cs: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        ElGamalChip::configure(cs)
    }
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
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
                .constrain_instance(c1.x().cell(), config.ciphertext_c1x_exp_col, C1_X)
                .and(layouter.constrain_instance(
                    c1.y().cell(),
                    config.ciphertext_c1y_exp_col,
                    C1_Y,
                ))
                .and(layouter.constrain_instance(c2.cell(), config.ciphertext_c2_exp_col, i))?;
        }
        Ok(())
    }
}
