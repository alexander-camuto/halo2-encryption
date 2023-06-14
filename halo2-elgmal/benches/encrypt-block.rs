use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::pfsys::create_keys;
use halo2_proofs::plonk::verify_proof;
use halo2_proofs::poly::ipa::commitment::IPACommitmentScheme;
use halo2_proofs::poly::ipa::multiopen::ProverIPA;
use halo2_proofs::poly::ipa::strategy::SingleStrategy;
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::{arithmetic::Field, poly::commitment::CommitmentScheme};

use ark_std::test_rng;
use halo2_elgmal::ElGamalGadget;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, TranscriptReadBuffer, TranscriptWriterBuffer,
};
use halo2curves::pasta::{pallas, EqAffine};
use std::ops::Deref;

fn rundot(c: &mut Criterion) {
    let mut group = c.benchmark_group("encrypt-block");
    let params = <IPACommitmentScheme<EqAffine> as CommitmentScheme>::ParamsProver::new(17);

    for &len in [
        // 4 * 4,
        28 * 28,
        58 * 58,
        128 * 128,
        2046 * 2046,
        10000 * 10000,
    ]
    .iter()
    {
        let mut rng = test_rng();

        let (_sk, pk) = ElGamalGadget::keygen(&mut rng).unwrap();

        let r = pallas::Scalar::random(&mut rng);

        let mut msg = vec![];
        //
        for _ in 0..len {
            msg.push(pallas::Base::random(&mut rng));
        }

        let circuit = ElGamalGadget::new(r, msg, pk);

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("pk", len), &len, |b, &_| {
            b.iter(|| {
                create_keys::<IPACommitmentScheme<EqAffine>, pallas::Base, ElGamalGadget>(
                    &circuit, &params,
                )
                .unwrap();
            });
        });

        let pk = create_keys::<IPACommitmentScheme<EqAffine>, pallas::Base, ElGamalGadget>(
            &circuit, &params,
        )
        .unwrap();

        let instances = ElGamalGadget::get_instances(&circuit.resulted_ciphertext);
        let pi_inner = instances
            .iter()
            .map(|e| e.deref())
            .collect::<Vec<&[pallas::Base]>>();
        let pi_inner: &[&[&[pallas::Base]]] = &[&pi_inner];

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("prove", len), &len, |b, &_| {
            b.iter(|| {
                let mut transcript: Blake2bWrite<_, _, _> =
                    TranscriptWriterBuffer::<_, EqAffine, _>::init(vec![]);
                let mut rng = ark_std::rand::rngs::OsRng;

                halo2_proofs::plonk::create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
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
            TranscriptWriterBuffer::<_, EqAffine, _>::init(vec![]);
        let mut rng = ark_std::rand::rngs::OsRng;

        halo2_proofs::plonk::create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
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
                verify_proof(&params, &pk.get_vk(), strategy, pi_inner, &mut transcript).unwrap();
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
