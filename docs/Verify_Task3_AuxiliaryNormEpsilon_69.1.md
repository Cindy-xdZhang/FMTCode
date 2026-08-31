# Verify_Task3_AuxiliaryNormEpsilon_69.1

## Question

Can changing only the denominator epsilon of LayerNorm/RMSNorm inside the
auxiliary projection increase the paired Task3 advantage of FMT without
reducing absolute FMT F1 or Average Precision?

## Motivation and scope

The auxiliary projection can be very narrow.  Its normalization epsilon sets
the variance floor and therefore controls how strongly small projected
differences are amplified before the nonlinearity.  Projection architecture,
width, affine scale/bias, Dropout, and optimizer epsilon have separate
registered searches; the normalization operation's own epsilon has not been
varied.  This is distinct from Adam's denominator epsilon.

This is a development-only single-factor search and cannot replace audited
spatial-population confirmations 6.1, 7.2, or 8.1.

## Frozen comparison

- The independently audited 68.1 portfolio supplies one frozen recipe per
  physical family.  Any normalization scale and bias selected upstream are
  preserved.
- Only the `eps` value of LayerNorm/RMSNorm inside the auxiliary projection is
  changed.  It is not checkpoint state, consumes no random numbers, and leaves
  every trainable tensor byte-identical at initialization under the paired
  seed.
- FMT and train-only Raw-PCA use the same candidate, trainable parameter count,
  initialization, split, and training budget.
- The control omits the override and exactly preserves each source module's
  native epsilon.  Confirmation remains closed.

## Registered grid and selection

Candidate epsilons are `{1e-12,1e-10,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1}`
plus the exact source control.  Eleven candidates across ten datasets and
three paired seeds yield 110 array mappings and 660 paired trainings.

A non-control candidate is eligible only if FMT F1 and FMT Average Precision
are both no lower than the exact control.  Eligible candidates are selected per
physical family by paired dataset-macro F1 gain, absolute FMT F1, and the
registered robustness tie-breakers.  The joint target is F1 gain at least
`+0.213` and absolute FMT F1 at least `0.893`.  Failure is retained.

## Status

Preregistered while 57.1 was incomplete and before 58.1--68.1 produced
performance results.  No 69.1 performance artifact has been generated or read.
