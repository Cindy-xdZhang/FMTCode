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
- Initial implementation commit: `156d8ef`.
- Initial deployment archive SHA-256:
  `8c00205074d7acef4c1a3ce572fa2635e37223ee1a4e16212dcc6e5c87ef1013`.
- Preflight job `51029073` completed in 18 s on `cn604-06`, exit 0,
  empty stderr. File/content manifest SHA-256 values are
  `48ab4e7468313781749e173159db7035bec793960c24f7095ac92b6ad4c8a32c`
  and `f45c4ade1a868bd5818330b001bc8bf5c548185a02c90e6abac3334655018ba3`.
- The first GPU array `51029104` exposed a runtime-only cache-path bug in
  child `51029104_0` (Slurm job `51031587`) at
  `2026-08-30T14:57:01+03:00`: preflight checked
  `development_cache/<dataset>`, while candidate loading passed only
  `development_cache`. It failed after 9 s with exit 1 before training or
  writing any candidate result. Children 1--119 were held immediately. The
  old array and its now-unsatisfiable selector `51029108` were cancelled at
  `15:06:42+03:00`; they provide no performance evidence.
- Commits `32af30b` and `2303aba` make runtime and preflight use the same
  dataset-leaf resolver and restore Ibex's permitted default GPU routing.
  The final correction archive SHA-256 is
  `e4d79a124c2df80bb49c485a57bfcfb0ac4bba84ebc7f2518bd2488bfb02a68a`,
  identical locally and remotely. Local and remote Python compilation, all
  eight contracts, and remote `bash -n` succeeded.
- Corrected preflight job `51032357` completed on `cn604-13` from
  `15:05:45` to `15:06:07`, exit 0 and empty stderr. Its file/content
  manifests remain exactly
  `48ab4e7468313781749e173159db7035bec793960c24f7095ac92b6ad4c8a32c`
  and `f45c4ade1a868bd5818330b001bc8bf5c548185a02c90e6abac3334655018ba3`,
  confirming that the scientific grid did not change.
- Replacement GPU array `51032757` and strict selector `51032779` were
  submitted at `15:08:39` and `15:08:53`. Performance remains pending;
  partial metrics are not read.
- At `2026-08-30T14:51+03:00`, before any child started, the scheduler-only
  `v100` feature constraint was removed and the compatible four-hour
  partition was added. The replacement uses Ibex's default GPU routing, which
  resolved to `gpu,gpu24,gpu72` at submission and can use GTX 1080 Ti, P100,
  V100, or A100 nodes. This changes no data, model, latent grid,
  seed, training code, budget, pairing, or selection rule; each Raw/FMT pair
  still runs within the same child on the same GPU.  The change is solely to
  reduce queue delay and is recorded before any performance output exists.
- At `14:55+03:00`, all still-pending children of Task3 arrays 38.1, 39.1,
  40.1, 41.1, 42.1, 43.1, 45.1, and 47.1 were temporarily held so the newer
  Task2 array could receive the next compatible free GPUs. Already-running
  Task3 children were verified to remain running.  The hold is scheduler-only
  and changed neither experiment. All holds were released immediately after
  the first Task2 child exposed the cache-path failure; no running Task3 child
  was interrupted.
