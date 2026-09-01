# Verify_Task3_AuxiliaryNormScale_65.1

## Question

Can changing only the initial trainable affine scale of normalization layers
inside the auxiliary projection increase the paired Task3 advantage of FMT
without reducing absolute FMT F1 or Average Precision?

## Motivation and scope

The current development portfolio feeds both fixed FMT and train-only Raw-PCA
features through the same `Linear -> LayerNorm -> GELU` projection.  Earlier
experiments searched projection architecture, width, Dropout, learning rate,
weight decay, Adam betas, and Adam epsilon, but not the initial LayerNorm affine
scale.  That scale directly controls how strongly the fixed auxiliary
representation enters the residual head at the start of training.

This is a development-only single-factor search.  It cannot replace the
audited spatial-population confirmations 6.1, 7.2, or 8.1.

## Frozen comparison

- The independently audited 64.1 portfolio supplies one frozen recipe per
  physical family.  IVD-p95 labels, development population and split, feature,
  auxiliary width and architecture, residual head, optimizer, scheduler,
  training budget, and paired seeds remain unchanged.
- Only trainable LayerNorm/RMSNorm affine weights inside the auxiliary
  projection receive the candidate initial scale.  Linear layers and every
  downstream residual-head parameter keep byte-identical initialization under
  the paired seed.
- FMT and train-only Raw-PCA use the same candidate, trainable parameter count,
  initialization procedure, split, and budget in every cell.
- The control omits the override and exactly preserves PyTorch's historical
  scale of one.  Confirmation remains closed.

## Registered grid and selection

The candidate scales are `{0,.01,.03,.10,.25,.50,.75,1.50,2,4}` plus the exact
source control.  Eleven candidates across ten datasets and three paired seeds
yield 110 array mappings and 660 paired trainings.

A non-control candidate is eligible only if its FMT F1 and FMT Average
Precision are both no lower than the exact control.  Eligible candidates are
selected per physical family by paired dataset-macro F1 gain, absolute FMT F1,
and the registered robustness tie-breakers.  The joint target is F1 gain at
least `+0.211` and absolute FMT F1 at least `0.893`.  Failure of either target
is retained as a negative result.

## Status

Completed and independently re-audited on 2026-09-01. Dataset-macro Raw-PCA/FMT
F1 is `0.683970/0.890863` (gain `+0.206893`); Average Precision gain is
`+0.223263`. The search does not satisfy the zero-tolerance portfolio guard or
the joint target. All 660 paired results were archived (SHA-256
`df542f59...05a2`) and checkpoints were deleted only after 66.1 audit.
