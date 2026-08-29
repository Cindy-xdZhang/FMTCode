# Verify_Task3_AuxiliaryProjection_21.1

## Question

Can a non-degenerate, identically paired auxiliary projection increase the
Task3 FMT gain over train-only Raw-PCA after the auxiliary width is frozen per
physical family by 20.1?

## Motivation

The historical projection is `Linear -> LayerNorm -> GELU`. At width one,
LayerNorm subtracts the only coordinate from itself, so every sample becomes
zero before GELU. At width two it also removes the mean and scale of each
sample, retaining very little amplitude information. Therefore, the width-one
candidate in 20.1 is a fair paired control but not an informative one-dimensional
representation.

## Frozen protocol

- Preflight waits for the complete 20.1 selector and freezes its SHA-256 hash
  and family-specific auxiliary width. No partial 19.1 or 20.1 result is read.
- FMT and train-only Raw-PCA receive the same projection architecture, output
  width, residual head, optimizer, batches, random seeds, and training budget.
- Eight projections compare the historical control with linear-only,
  GELU/SiLU, RMS normalization, and a 64-wide nonlinear pre-projection before
  the frozen bottleneck.
- Raw-PCA has exactly the FMT input width and is fitted on training data only,
  so paired trainable parameter counts remain equal.
- Every candidate must remain below the frozen Raw-wide parameter ceiling.
- Only exposed development populations are read; confirmation remains closed.

## Scale and decision

Eight projections x 10 datasets x 3 seeds x 2 arms give 480 trainings in 80
GPU array children. Family-specific selection maximizes dataset-macro F1 gain,
then Average Precision gain, absolute FMT F1, positive-dataset count, worst
dataset F1 gain, and worst paired-seed gain. The preregistered target is
dataset-macro F1 gain `>= +0.170`.

Absolute Raw-PCA and FMT F1/Average Precision must be reported beside their
difference. A wider difference caused by both arms degrading is not an
absolute classifier improvement. Any development winner still requires a new
unseen spatial population before supporting a paper-level generalization.
