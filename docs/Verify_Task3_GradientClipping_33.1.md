# Verify_Task3_GradientClipping_33.1

## Question

Can global gradient-norm clipping improve absolute FMT classification while
increasing its paired advantage over the same-width train-only Raw-PCA arm?

Gradient clipping rescales all trainable residual-head gradients only when
their joint L2 norm exceeds a fixed threshold. It is applied after the
finite-gradient check and immediately before the optimizer step. It does not
change labels, features, model capacity, loss terms, validation thresholds, or
fusion-alpha selection.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: no clipping and global norm limits
  `{0.1,0.25,0.5,1,2,5,10}`.
- Within every candidate, FMT and train-only Raw-PCA use the same threshold,
  initialization, batches, optimizer, budget, split, and head.
- The no-clipping control is a strict code-level no-op.
- Confirmation remains closed.

Eight candidates across ten datasets produce 80 array mappings and 480 paired
trainings. The maximum pre-clipping gradient norm is recorded per epoch for
auditing whether each threshold actually activates.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to no clipping. Eligible candidates are
ordered by paired F1 gain, then absolute FMT F1 and the registered robustness
tie-breakers.

The joint development target is F1 gain at least `0.195` and absolute FMT F1
at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

This experiment was specified without reading partial or final performance
from Task3 searches 27.1--32.1.
