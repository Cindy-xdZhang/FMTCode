# Verify_Task3_AuxiliaryFeatureScale_73.1

## Question

Can a fixed rescaling of the projected auxiliary representation increase the
paired Task3 advantage of FMT without reducing absolute FMT F1 or Average
Precision?

## Motivation and scope

The frozen Raw geometry and the auxiliary representation enter the residual
head at potentially different numerical amplitudes.  Earlier experiment 32.1
scaled the final residual logit, while 65.1 changed the trainable normalization
weight at initialization.  Neither tested a fixed multiplier after the full
auxiliary projection, activation, Dropout, and optional Gaussian noise.  This
multiplier directly controls the auxiliary contribution seen by the residual
head and remains fixed throughout training.

This is a development-only single-factor search and cannot replace audited
spatial-population confirmations 6.1, 7.2, or 8.1.

## Frozen comparison

- The independently audited 72.1 portfolio supplies one frozen recipe per
  physical family.
- Scaling is applied after auxiliary projection, Dropout, and optional Gaussian
  noise but before residual fusion.
- The multiplier has no parameters or checkpoint state.  `scale=1` bypasses
  multiplication and exactly preserves the source computation graph.
- FMT and train-only Raw-PCA use the same scale, trainable parameter count,
  random seed, split, and training budget.
- All upstream normalization, optimizer, residual-head, feature, and noise
  choices are preserved.  Confirmation remains closed.

## Registered grid and selection

Candidate scales are `{0,.01,.03,.10,.25,.50,.75,1.50,2,4}` plus the exact
`1.0` control.  Eleven candidates across ten datasets and three paired seeds
yield 110 array mappings and 660 paired trainings.

A non-control candidate is eligible only if FMT F1 and FMT Average Precision
are both no lower than the exact control.  Eligible candidates are selected per
physical family by paired dataset-macro F1 gain, absolute FMT F1, and the
registered robustness tie-breakers.  The joint target is F1 gain at least
`+0.215` and absolute FMT F1 at least `0.893`.  Failure is retained.

## Status

Preregistered while 57.1 was incomplete and before 58.1--72.1 produced
performance results.  No 73.1 performance artifact has been generated or read.
