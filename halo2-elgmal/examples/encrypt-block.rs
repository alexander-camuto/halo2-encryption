use ark_std::test_rng;
use halo2_elgmal::ElGamalGadget;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::dev::MockProver;
use halo2curves::{bn256::Fr, group::Curve};

fn main() {
    let mut rng = test_rng();

    let (sk, pk) = ElGamalGadget::keygen(&mut rng).unwrap();

    let r = Fr::random(&mut rng);

    let mut msg = vec![];
    //
    for _ in 0..1 {
        msg.push(Fr::random(&mut rng));
    }

    let circuit = ElGamalGadget::new(r, msg, pk.to_affine(), sk);

    let mut public_inputs: Vec<Vec<Fr>> = vec![vec![]];
    public_inputs.extend(ElGamalGadget::get_instances(
        &circuit.resulted_ciphertext,
        circuit.sk_hash,
    ));

    let res = MockProver::run(17, &circuit, public_inputs).unwrap();
    res.assert_satisfied_par();
}
