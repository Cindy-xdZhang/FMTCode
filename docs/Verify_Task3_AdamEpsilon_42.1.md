# Verify_Task3_AdamEpsilon_42.1

## Question

Can the AdamW denominator epsilon improve absolute FMT vortex classification
while increasing its paired advantage over the same-width train-only Raw-PCA
arm?

AdamW divides the first-moment estimate by the square root of the second moment
plus epsilon. Epsilon therefore limits effective steps when gradients and
second moments are small. This can matter for low-dimensional residual inputs
without changing features, labels, loss, model capacity, or parameter count.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: exact PyTorch-default control and epsilon
  `{1e-12,1e-10,1e-9,1e-7,1e-6,1e-5,1e-4}`.
- Within every candidate, FMT and train-only Raw-PCA use the same epsilon,
  initialization, batches, AdamW settings, budget, split, and head.
- The control does not pass an epsilon override and therefore retains the
  PyTorch default `1e-8` code path.
- Confirmation remains closed.

Eight candidates across ten datasets produce 80 array mappings and 480 paired
trainings. Epsilon is recorded in checkpoints and per-run output.

This search was declared without reading partial metrics from 35.1--41.1.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the exact default-epsilon control.
Eligible candidates are ordered by paired F1 gain, then absolute FMT F1 and
the registered robustness tie-breakers.

The joint development target remains F1 gain at least `0.195` and absolute FMT
F1 at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Local Python compilation and 61 relevant unit tests passed. Full local
preflight confirmed 10 datasets, 8 candidates, 80 array mappings, 480 paired
trainings, all parameter budgets, and closed confirmation state. The local
preflight manifest SHA-256 is `ce17319b...f3e7`.

Implementation commit `03594f7` was pushed before deployment. The immutable
archive SHA-256 is `4c17851e...72d8`; local raw and remote canonical config
SHA-256 values are `9a48c35e...c14d` and `0f851490...773c`, respectively.
Remote Python compilation, the same 61 tests, and all three `bash -n` checks
passed.

Submitted at `2026-08-30T10:39:02+03:00`: CPU preflight job `51016835`, GPU
array `51016836[0-79%24]`, and selector `51016837`. The array has strict
`afterok:51016835`; the selector has strict `afterok:51016836_*`. Preflight
ran on `cn604-07` from 10:39:03 to 10:40:10, exited zero with empty stderr,
and produced remote manifest SHA-256 `e63b1141...8b45`. It confirmed 10
datasets, 8 candidates, 80 mappings, 480 paired trainings, all capacity
guards, and closed confirmation state. At `2026-08-30T21:09+03:00`, the GPU
array had 12/80 children complete, 6 running, and 62 waiting, with no failed
child. Performance results have not been read. Evidence archive/cleanup job
`51057984` was submitted with strict `afterok:51016837`; its committed script
SHA-256 is `97c3f337...a46f74`, identical locally and remotely. It will require
all 480 per-run CSV files, create a byte-stable archive, and delete only this
experiment's temporary checkpoints after successful selection.
