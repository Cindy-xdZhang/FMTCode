# Verify_Task3_AuxiliaryGaussianNoise_71.1

## Question

Can training-time Gaussian perturbation of the projected auxiliary
representation increase the paired Task3 advantage of FMT without reducing
absolute FMT F1 or Average Precision?

## Motivation and scope

Elementwise Dropout was tested in 53.1 and produced only a small improvement.
Additive zero-mean Gaussian noise is a distinct regularizer: it preserves every
coordinate while penalizing local sensitivity of the downstream classifier.
It has not been tested in this project.  Because FMT features encode structured
geometric statistics whereas Raw-PCA features are fitted linear coordinates,
their robustness to the same perturbation may differ.

This is a development-only single-factor search and cannot replace audited
spatial-population confirmations 6.1, 7.2, or 8.1.

## Frozen comparison

- The independently audited 70.1 portfolio supplies one frozen recipe per
  physical family.
- Candidate noise is added after auxiliary projection and Dropout, only in
  training mode.  Evaluation is deterministic and noise-free.
- The noise module has no parameters or checkpoint state.  `std=0` returns the
  original tensor without drawing random numbers and is the exact control.
- FMT and train-only Raw-PCA use the same standard deviation, trainable
  parameter count, random seed, split, and training budget.
- All upstream normalization, optimizer, residual-head, and feature choices are
  preserved.  Confirmation remains closed.

## Registered grid and selection

Candidate standard deviations are
`{.005,.010,.025,.050,.100,.200,.300,.500,.750,1.000}` plus the exact zero-noise
control.  Eleven candidates across ten datasets and three paired seeds yield
110 array mappings and 660 paired trainings.

A non-control candidate is eligible only if FMT F1 and FMT Average Precision
are both no lower than the exact control.  Eligible candidates are selected per
physical family by paired dataset-macro F1 gain, absolute FMT F1, and the
registered robustness tie-breakers.  The joint target is F1 gain at least
`+0.214` and absolute FMT F1 at least `0.893`.  Failure is retained.

## Status

Preregistered while 57.1 was incomplete and before 58.1--70.1 produced
performance results.  No 71.1 performance artifact has been generated or read.
