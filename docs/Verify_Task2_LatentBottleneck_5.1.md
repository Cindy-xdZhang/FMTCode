# Verify_Task2_LatentBottleneck_5.1

## Question

Does a narrower common Variational Autoencoder (VAE) latent bottleneck increase
the 3D Task2 advantage of `FMT(pathline) -> VAE -> KMeans` over
`Raw pathline -> VAE -> KMeans`?

## Frozen factors

- The seven physical-family assignments, ten datasets, FMT feature block,
  hidden layers, KL weight, learning rate, optimizer-step budget, clustering
  protocol, and development split are frozen from
  `Verify_Task2_FMTVAEFamilySearch_4.1`.
- The exact source stage-2 selection SHA-256 is
  `439d9e5dcf72adc78d12c38dda55443d87ffed2b8aeb061067b3cda8a8d97795`.
- Raw and FMT receive the same latent dimension, hidden layers, optimizer
  settings, training seed, split, and KMeans settings in every paired cell.
- Only latent dimension changes: `{1,2,3,4,6,8,12,16,24,32,48,64}`.
- New development training seeds are `83,84,85`.
- Only ordinals 0--7 are opened. Ordinals 8--9 and every confirmation cache
  remain closed.

This gives 12 candidates x 10 datasets x 3 seeds x 2 arms = 720 trainings.
No model checkpoint is written.

## Selection rule

For each physical family, maximize paired `FMT+VAE - Raw+VAE` validation F1.
Ties are broken by worst-seed gain, worst-dataset gain, then absolute FMT F1.
This rule intentionally measures whether the same bottleneck preserves FMT
input structure better than Raw input structure; it does not optimize a
separate stronger VAE for Raw.

The development target is dataset-macro F1 gain at least `+0.22`, slightly
above the `4.1` development result `+0.21956`. Any selected recipe remains a
development result and requires a newly frozen population before replacing the
current Task2 paper result.

## Traceability

- Main code: `Search_Task2_LatentBottleneck_5_1.py`
- Config: `config/Verify_Task2_LatentBottleneck_5.1.yaml`
- Contracts: `tests/test_task2_latent_bottleneck_5_1.py`
- Ibex scripts: `ibex_bash/verify_task2_latent_bottleneck_5.1_*.sh`
- Deployment commit, archive SHA, preflight SHA, job IDs, devices, and final
  metrics are recorded only after they exist.
