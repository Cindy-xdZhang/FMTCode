# Verify_Task3_AuxiliaryEpsilon_63.1

## Question

Does assigning only the trainable auxiliary projection a different Adam-family
denominator epsilon increase the paired Task3 advantage of FMT without reducing
absolute FMT F1 or Average Precision?

## Motivation and scope

Experiment 42.1 changed epsilon for the complete residual optimizer. Searches
55.1, 57.1, 59.1, and 61.1 separate the optimization of the auxiliary
projection from the downstream residual head, but none changes the auxiliary
epsilon alone. This experiment tests whether the fixed FMT and train-only
Raw-PCA inputs require a different adaptive-denominator floor while preserving
identical treatment of the two paired arms.

This is a development-only single-factor search. It cannot replace audited
spatial-population confirmations 6.1, 7.2, or 8.1.

## Frozen comparison

- The independently audited 62.1 portfolio supplies one frozen recipe per
  physical family. IVD-p95 labels, development population and split, feature,
  auxiliary width and projection, residual head, learning rate, weight decay,
  betas, epoch budget, early stopping, and paired seeds remain unchanged.
- Only parameters in the trainable auxiliary projection receive the candidate
  epsilon. The downstream residual head keeps the source optimizer epsilon.
- FMT and train-only Raw-PCA use the same candidate epsilon, trainable parameter
  count, initialization, split, and training budget in every cell.
- The control has no local override and exactly reproduces the source recipe.
  Confirmation remains closed.

## Registered grid and selection

The auxiliary epsilon grid is `{1e-12,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,
1e-4,1e-3,1e-2}` plus the exact source control. Eleven candidates across ten
datasets and three paired seeds yield 110 array mappings and 660 paired
trainings.

Selection is physical-family specific. A non-control candidate is eligible
only if FMT F1 and FMT Average Precision are both no lower than the exact
control. Eligible candidates are ranked by paired dataset-macro F1 gain,
absolute FMT F1, and the registered robustness tie-breakers. The joint target
is F1 gain at least `+0.210` and absolute FMT F1 at least `0.893`. Failure of
either target is retained as a negative result.

## Status

Preregistered before 57.1--62.1 produced performance results. No 63.1
performance artifact has been generated or read.
