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

Remote validation and deployment are pending.
