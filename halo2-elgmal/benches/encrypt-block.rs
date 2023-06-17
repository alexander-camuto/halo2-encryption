use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::pfsys::create_keys;
use halo2_proofs::plonk::verify_proof;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverGWC, VerifierGWC};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{arithmetic::Field, poly::commitment::CommitmentScheme};

use ark_std::test_rng;
use halo2_elgmal::ElGamalGadget;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, TranscriptReadBuffer, TranscriptWriterBuffer,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::group::Curve;
use std::ops::Deref;

fn rundot(c: &mut Criterion) {
    let mut group = c.benchmark_group("encrypt-block");
    let params = <KZGCommitmentScheme<Bn256> as CommitmentScheme>::ParamsProver::new(23);

    for &len in [
        // 4 * 4,
        // 28 * 28,
        // 58 * 58,
        // 128 * 128,
        2046 * 2046,
        // 10000 * 10000,
    ]
    .iter()
    {
        let mut rng = test_rng();

        let (sk, pk) = ElGamalGadget::keygen(&mut rng).unwrap();

        let r = Fr::random(&mut rng);

        let mut msg = vec![];
        //
        for _ in 0..len {
            msg.push(Fr::random(&mut rng));
        }

        let circuit = ElGamalGadget::new(r, msg, pk.to_affine(), sk);

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("pk", len), &len, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, Fr, ElGamalGadget>(&circuit, &params)
                    .unwrap();
            });
        });

        let pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr, ElGamalGadget>(&circuit, &params)
            .unwrap();

        let mut instances: Vec<Vec<Fr>> = vec![vec![]];
        instances.extend(ElGamalGadget::get_instances(
            &circuit.resulted_ciphertext,
            circuit.sk_hash,
        ));
        let pi_inner = instances.iter().map(|e| e.deref()).collect::<Vec<&[Fr]>>();
        let pi_inner: &[&[&[Fr]]] = &[&pi_inner];

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("prove", len), &len, |b, &_| {
            b.iter(|| {
                let mut transcript: Blake2bWrite<_, _, _> =
                    TranscriptWriterBuffer::<_, G1Affine, _>::init(vec![]);
                let mut rng = ark_std::rand::rngs::OsRng;

                halo2_proofs::plonk::create_proof::<KZGCommitmentScheme<_>, ProverGWC<_>, _, _, _, _>(
                    &params,
                    &pk,
                    &[circuit.clone()],
                    pi_inner,
                    &mut rng,
                    &mut transcript,
                )
                .unwrap();
                let _proof = transcript.finalize();
            });
        });

        let mut transcript: Blake2bWrite<_, _, _> =
            TranscriptWriterBuffer::<_, G1Affine, _>::init(vec![]);
        let mut rng = ark_std::rand::rngs::OsRng;

        halo2_proofs::plonk::create_proof::<KZGCommitmentScheme<_>, ProverGWC<_>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            pi_inner,
            &mut rng,
            &mut transcript,
        )
        .unwrap();
        let proof = transcript.finalize();

        group.bench_with_input(BenchmarkId::new("verify", len), &len, |b, &_| {
            b.iter(|| {
                let mut transcript: Blake2bRead<_, _, _> =
                    TranscriptReadBuffer::init(std::io::Cursor::new(proof.clone()));
                let strategy = SingleStrategy::new(&params);
                verify_proof::<KZGCommitmentScheme<_>, VerifierGWC<_>, _, _, _>(
                    &params,
                    &pk.get_vk(),
                    strategy,
                    pi_inner,
                    &mut transcript,
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots().sample_size(10);
  targets = rundot
}
criterion_main!(benches);
