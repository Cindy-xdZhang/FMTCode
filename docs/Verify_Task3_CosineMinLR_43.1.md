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
array `51017334[0-89%24]`, and selector `51017338`. The array has strict
`afterok:51017332`; the selector has strict `afterok:51017334_*`. Preflight
ran from 10:47:56 to 10:49:02 on `cn604-13`, exited 0, and produced an empty
stderr. The remote manifest SHA-256 is `5dcb69fb...9929`. The GPU array is now
pending only on `QOSMaxGRESPerUser`; performance results have not been read.
