use ark_std::test_rng;
use halo2_elgmal::ElGamalGadget;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::dev::MockProver;
use halo2curves::bn256::{Fq, Fr};

fn main() {
    let mut rng = test_rng();

    let (_sk, pk) = ElGamalGadget::keygen(&mut rng).unwrap();

    let r = Fr::random(&mut rng);

    let mut msg = vec![];
    //
    for _ in 0..32 {
        msg.push(Fq::random(&mut rng));
    }

    let circuit = ElGamalGadget::new(r, msg, pk);

    let mut public_inputs: Vec<Vec<Fq>> = vec![vec![]];
    public_inputs.extend(ElGamalGadget::get_instances(&circuit.resulted_ciphertext));

    let res = MockProver::run(17, &circuit, public_inputs).unwrap();
    res.assert_satisfied_par();
}
