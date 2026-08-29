# Verify_Task3_TrainingResidualScale_32.1

## Question

Can the multiplier applied to the trainable residual logits during training
improve absolute FMT classification while increasing the paired gain over the
same-width train-only Raw-PCA control?

The training logits are `raw_logits + training_alpha * residual_logits`.
Therefore `training_alpha` directly scales the classification gradient entering
the residual branch. It does not freeze or replace the independently selected
validation/inference fusion alpha. This is an optimization-conditioning test,
not a new classifier or a post-hoc score rescaling.

## Why this is not a duplicate

`Verify_Task3_LossOptimization_7.1` was a broad one-factor screen and included
only training alpha 2 and 3 around the implicit alpha-1 control. Smoke selected
alpha 2. The present experiment systematically covers
`{0.125,0.25,0.5,0.75,1,1.5,2,3,4}` using the stronger completed 22.1
family-specific anchored feature. It was specified without reading any partial
or final performance from 27.1--31.2.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Within every candidate, FMT and train-only Raw-PCA use the same training
  alpha, initialization, batches, optimizer, budget, split, and head.
- Confirmation remains closed.

Nine candidates across ten datasets produce 90 array mappings and 540 paired
trainings. The alpha-1 candidate is the exact behavioral control.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to alpha 1. Eligible candidates are ordered
by paired F1 gain, then absolute FMT F1 and the registered robustness
tie-breakers.

The joint development target is F1 gain at least `0.195` and absolute FMT F1
at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.
