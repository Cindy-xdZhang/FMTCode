# Verify_Task3_CosineMinLR_43.1

## Question

Can cosine learning-rate annealing improve absolute FMT vortex classification
while increasing its paired advantage over the same-width train-only Raw-PCA
arm?

Cosine annealing gradually reduces the learning rate from its frozen base value
to a configured terminal fraction over the unchanged 100-epoch budget. It does
not alter features, labels, loss, network capacity, batches, or parameter count.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: exact constant-learning-rate control and cosine terminal ratios
  `{0,0.001,0.01,0.025,0.05,0.10,0.25,0.50}`.
- Within every candidate, FMT and train-only Raw-PCA use the same schedule,
  initialization, batches, optimizer, budget, split, and head.
- The control has no training override and follows the historical no-scheduler
  path exactly.
- Confirmation remains closed.

Nine candidates across ten datasets produce 90 array mappings and 540 paired
trainings.

The broad 7.1 experiment included only one cosine ratio, `0.05`, with the older
5.2 representations. This search resolves the schedule on the current 22.1
family-specific anchored features. It was declared without reading partial
metrics from 35.1--42.1.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the exact constant-learning-rate control.
Eligible candidates are ordered by paired F1 gain, then absolute FMT F1 and
the registered robustness tie-breakers.

The joint development target remains F1 gain at least `0.195` and absolute FMT
F1 at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Local Python compilation and 67 relevant unit tests passed. Full local
preflight confirmed 10 datasets, 9 candidates, 90 array mappings, 540 paired
trainings, all parameter budgets, and closed confirmation state. The local
preflight manifest SHA-256 is `9e96a17d...a204`.

Implementation commit `6679080` was pushed before deployment. The immutable
archive SHA-256 is `d7380899...0618`; local raw and remote canonical config
SHA-256 values are `7d17d75b...4e81` and `fd810922...c7e0`, respectively.
Remote Python compilation, the same 67 tests, and all three `bash -n` checks
passed.

Submitted at `2026-08-30T10:47:54+03:00`: CPU preflight job `51017332`, GPU
array `51017334[0-89%24]`, selector `51017338`, and evidence archive/cleanup
job `51058184`. All four jobs completed with exit code 0 and empty stderr. The
GPU array ran from the first child at 21:19:43 until 23:26:42 and completed all
90 mappings and 540 paired trainings. Its allocations comprised 9 A100, 41
GTX 1080 Ti, 16 P100, and 24 V100 child runs. The selector finished at
23:26:55, and the archive/cleanup job finished at 23:27:38.

## Final development result

The per-family guarded selection chose terminal ratios `0.50` for Channel,
`0.05` for halfcylinder, and `0.01` for Tangaroa; Boeing 747, Delta Wing,
F-22, and Smoke retained the exact constant-learning-rate control. Across ten
datasets, train-only Raw-PCA and FMT obtained F1 `0.697864` and `0.887322`,
respectively, for a paired gain of `+0.189459`. Their Average Precision values
were `0.739669` and `0.946137`, for a gain of `+0.206467`. All 10 datasets had
positive F1 gain, and the worst dataset gain was `+0.047216`.

The exact constant-schedule control already had F1 gain `+0.189125` and FMT
F1 `0.887076`. Selection therefore improved gain by only `+0.000333` and
absolute FMT F1 by only `+0.000247`. It failed both registered joint targets:
gain `>=0.195` and FMT F1 `>=0.893`, and it remains weaker than the completed
Dropout search 38.1 (`+0.196479`). Cosine annealing is therefore not promoted
as a standalone confirmation candidate, although its selected family recipes
remain eligible inputs to the preregistered combination searches.

An independent implementation reconstructed all 540 archived records, all
guards, all seven family choices, and every macro metric. Its maximum absolute
difference from the selector was `2.22e-16`; all source hashes were consistent.
The independent audit SHA-256 is `2d15f307...b4473`. Selection, leaderboard,
preflight, and stable per-run archive SHA-256 values are
`403d5207...d0e6`, `b6c2a0c8...0920`, `5dcb69fb...9929`, and
`4d02dfd9...0e24`. The cleanup verified 540 archived CSV files, deleted 540
temporary checkpoints, and left zero checkpoints.
