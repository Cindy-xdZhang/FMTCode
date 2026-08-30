# Verify_Task3_LabelSmoothing_41.1

## Question

Can binary label smoothing improve absolute FMT vortex classification while
increasing its paired advantage over the same-width train-only Raw-PCA arm?

For training only, smoothing epsilon maps a hard label `y` to
`(1-epsilon)y + epsilon/2`. Validation and evaluation retain the original hard
IVD labels. This regularizes overconfident residual logits without changing
features, capacity, label generation, validation thresholds, or fusion-alpha
selection.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: exact hard-label control and epsilon
  `{0.001,0.0025,0.005,0.01,0.02,0.05,0.10,0.20}`.
- Within every candidate, FMT and train-only Raw-PCA use the same epsilon,
  initialization, batches, class weights, optimizer, budget, split, and head.
- Epsilon zero follows the previous weighted binary-cross-entropy computation
  exactly.
- Confirmation remains closed.

Nine candidates across ten datasets produce 90 array mappings and 540 paired
trainings. Smoothing is recorded in histories, checkpoints, and per-run output.

This search was declared after the complete 34.1 selector and without reading
partial metrics from 35.1--40.1.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the exact hard-label control. Eligible
candidates are ordered by paired F1 gain, then absolute FMT F1 and the
registered robustness tie-breakers.

The joint development target remains F1 gain at least `0.195` and absolute FMT
F1 at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Local Python compilation and 55 relevant unit tests passed. Full local
preflight confirmed 10 datasets, 9 candidates, 90 array mappings, 540 paired
trainings, all parameter budgets, and closed confirmation state. The local
preflight manifest SHA-256 is `fdbd889b...6c83`.

Implementation commit `352497c` was pushed before deployment. The immutable
archive SHA-256 is `ea93e2de...105c`; local raw and remote canonical config
SHA-256 values are `caea707d...6e7b` and `b60da2ac...646d`, respectively.
Remote Python compilation, the same 55 tests, and all three `bash -n` checks
passed.

Submitted at `2026-08-30T10:29:35+03:00`: CPU preflight job `51016609`, GPU
array `51016612[0-89%24]`, and selector `51016613`. The array has strict
`afterok:51016609`; the selector has strict `afterok:51016612_*`. Preflight
ran on `cn604-04` from 10:29:38 to 10:30:52, exited zero with empty stderr,
and produced remote manifest SHA-256 `b22f0a28...9f41`. It confirmed 10
datasets, 9 candidates, 90 mappings, 540 paired trainings, all capacity
guards, and closed confirmation state. At `2026-08-30T21:09+03:00`, the GPU
array had 81/90 children complete and 9 running, with no failed child.
Performance results have not been read. Evidence archive/cleanup job
`51057902` was submitted with strict `afterok:51016613`; its committed script
SHA-256 is `67d45e7d...b040e`, identical locally and remotely. It will require
all 540 per-run CSV files, create a byte-stable archive, and delete only this
experiment's temporary checkpoints after successful selection.
