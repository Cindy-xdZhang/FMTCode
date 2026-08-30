# mainExp_Task2_3D_5.2

> Status: **DEPLOYMENT READY; confirmation not yet opened.**

## Question

Does the family-specific latent dimension selected by
`Verify_Task2_LatentBottleneck_5.1` preserve a substantial same-VAE
`FMT+VAE - Raw+VAE` F1 advantage on a previously unseen spatial primitive
population?

## Frozen primary comparison

- Ten 3D datasets and whole-field IVD-p95 evaluation labels.
- `Raw pathline -> VAE -> latent -> KMeans` versus
  `FMT(pathline) -> the same VAE -> latent -> KMeans`.
- Within every dataset, recipe and training seed, Raw and FMT share hidden
  layers, latent dimension, KL weight, optimizer, learning rate, optimizer
  steps, split and KMeans settings.
- Primary latent dimensions frozen by selection SHA-256
  `cf1c546fb833cacc3b6b582544db6269ea5b2e5d5ad5af04f44223f343e3e3dc`:
  channel 8, half-cylinder 64, Tangaroa 1, delta-wing 12, F-22 1,
  Boeing 747 6, smoke buoyancy 24.
- The Task2 4.1 latent dimensions are evaluated as a diagnostic control on
  the same samples and seeds. The control cannot replace the selected primary
  method after confirmation is opened.
- VAE training uses development ordinals 0--7. Ordinals 8--9 calibrate only
  which anonymous KMeans cluster denotes vortex. No confirmation label is
  used in training, normalization, KMeans fitting or cluster-ID calibration.
- New paired training seeds are 9090--9094. No checkpoint is written.

## New population

Physical time windows, pathline integration, encoder and IVD-p95 definition
remain unchanged. Before any new primitive exists, the spatial phase is
derived from SHA-256 of
`mainExp_Task2_3D_5.2|fifth-spatial-population-v1`. The first eight hex
digits give Halton index 544 and phase
`[-.4833984375, -.0281207133, .4344]`. It differs from all four exposed
spatial populations. Existing temporal source packs are reused only as
phase-independent velocity windows; their parent manifest SHA-256 is
`020b48ffa9ee6d59b5c333a1b6513e015523c1e0db984884f1b9ade20a3e2a61`.

## Decision and reporting

- Primary target: selected dataset-macro F1 gain at least `+.15`.
- Aspirational target: selected dataset-macro F1 gain at least `+.22`.
- Always report selected and control absolute Raw/FMT F1, paired gains,
  10-dataset and 7-family positive counts, seed-macro robustness, and the
  worst dataset.
- Results are read only after all ten dataset shards and 200 paired recipe
  trainings complete. Partial metrics cannot alter the method.

## Code and execution order

- Config: `config/mainExp_Task2_3D_5.2.yaml`.
- Source derivation: `Prepare_Task2_LatentConfirmation_SourceManifest_5_2.py`.
- Fifth-population builder: `Build_Task2_LatentConfirmation_5_2.py`.
- Frozen evaluation: `Confirm_Task2_LatentBottleneck_5_2.py`.
- Contracts: `tests/test_mainexp_task2_3d_5_2.py`.
- Ibex scripts: `ibex_bash/mainexp_task2_latent_5.2_*.sh`.

Required order: derive temporal-source manifest, static preflight, freeze
recipe, source preflight, build 10 caches, evaluation preflight, evaluate ten
datasets, then summarize. A failed stage is recorded and repaired without
changing the frozen method.
