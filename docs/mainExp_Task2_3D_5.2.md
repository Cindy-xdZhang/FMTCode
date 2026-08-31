# mainExp_Task2_3D_5.2

> Status: **COMPLETED on Ibex; all 200 trainings independently verified.**

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
  method after confirmation is opened. Its metrics are a 5.2-population
  rerun of the 4.1 recipe, not the historical Task2-4.1 result.
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

## Completed result

Ibex jobs `51036980`--`51036987` completed on 2026-08-30. All ten cache
children, all ten evaluation children and the final summary exited 0. The
downloaded `per_run.csv` was independently recomputed rather than trusting
the generated summary: it contains 200/200 unique
`(recipe, dataset, seed, arm)` rows, all requested optimizer steps completed,
ten datasets, two recipes, five seeds and both Raw/FMT arms.

| Frozen recipe | Raw+VAE F1 | FMT+VAE F1 | Dataset-macro gain | Family-macro gain | Positive datasets/families | Worst dataset gain |
|---|---:|---:|---:|---:|---:|---:|
| **Selected latent (primary)** | `.39227` | `.63143` | **`+.23916`** | **`+.26598`** | **10/10; 7/7** | **`+.05812`** |
| 4.1 recipe rerun on 5.2 population (diagnostic control) | `.48594` | `.65216` | `+.16622` | `+.15676` | 9/10; 6/7 | `-.07432` |

The primary result exceeds both the preregistered `+.15` target and the
aspirational `+.22` target. All five seed-macro gains are positive
(`+.21912` to `+.25625`). The smallest dataset gain is half-cylinder Re160;
the selected latent-1 F-22 recipe changes the former control counterexample
to `+.19507` on this new population.

Relative to the diagnostic control on exactly the same population and seeds,
latent selection increases the measured gain by `+.07293`, but it does not
improve absolute FMT F1: selected FMT is `.02074` lower than control FMT.
Relative to the previous Task2-4.1 paper result, selected FMT stays essentially
unchanged (`.63077 -> .63143`) while Raw decreases (`.46165 -> .39227`). Thus
the supported claim is specifically that the frozen shared bottleneck exposes
a larger FMT input advantage; this experiment does not claim that the selected
latent dimensions make the FMT representation itself more accurate.

No checkpoint was written. Final evidence SHA-256 values are:

- `summary.json`: `739b48ea0aedc715f7dea429d79f1cf27a8964530f6778d351ced6539d2c3095`
- `per_run.csv`: `20d9de155b7e1b94669eedda0bef95a6a3ce5691237af559cfbe00c923c7111f`
- `paper_table.csv`: `fe68a9877b897ab1af7f7feb8ebd9430b2e2ef690121527515754112636fcd45`
- frozen recipe manifest: `2d0c0cf133ba068bcf9f17d5b82bafbad4205a43735bbbc928e513aa2a766307`
- evaluation preflight: `4b832b6ff2cc30e6b5828101d439197fc90142c0970609e0592780dd1fd2be0f`

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
