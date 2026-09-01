# Verify_Task3_AuxiliaryNormBias_67.1

## Question

Can changing only the initial trainable LayerNorm bias inside the auxiliary
projection increase the paired Task3 advantage of FMT without reducing
absolute FMT F1 or Average Precision?

## Motivation and scope

The auxiliary projection applies GELU immediately after LayerNorm.  Its
trainable bias therefore changes the initial fraction and magnitude of
auxiliary activations passed to the residual head.  Projection architecture,
width, Dropout, optimizer parameters, and normalization scale have separate
registered searches; the normalization bias has not been varied.

This is a development-only single-factor search and cannot replace audited
spatial-population confirmations 6.1, 7.2, or 8.1.

## Frozen comparison

- The independently audited 66.1 portfolio supplies one frozen recipe per
  physical family.  Any normalization-scale choice made by 65.1 is preserved.
- Only LayerNorm affine bias inside the auxiliary projection receives the
  candidate initial value.  Linear layers, normalization scale, and every
  residual-head parameter keep byte-identical initialization under the paired
  seed.
- FMT and train-only Raw-PCA use the same candidate, trainable parameter count,
  initialization procedure, split, and training budget.
- The control omits the override and exactly preserves PyTorch's historical
  bias of zero.  Confirmation remains closed.

## Registered grid and selection

The candidate biases are `{-2,-1,-.5,-.25,-.1,.05,.1,.25,.5,1}` plus the exact
source control.  Eleven candidates across ten datasets and three paired seeds
yield 110 array mappings and 660 paired trainings.

A non-control candidate is eligible only if FMT F1 and FMT Average Precision
are both no lower than the exact control.  Eligible candidates are selected per
physical family by paired dataset-macro F1 gain, absolute FMT F1, and the
registered robustness tie-breakers.  The joint target is F1 gain at least
`+0.212` and absolute FMT F1 at least `0.893`.  Failure is retained.

## Status

Completed and independently re-audited on 2026-09-01. Dataset-macro Raw-PCA/FMT
F1 is `0.683964/0.890836` (gain `+0.206872`); Average Precision gain is
`+0.223047`. The search does not satisfy the zero-tolerance portfolio guard or
the joint target. All 660 paired results were archived (SHA-256
`2ab072a2...3490`) and checkpoints were deleted only after 68.1 audit.
