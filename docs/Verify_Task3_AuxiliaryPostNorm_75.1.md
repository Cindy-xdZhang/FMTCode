# Verify_Task3_AuxiliaryPostNorm_75.1

## Question

Does parameter-free normalization after the auxiliary projection activation
increase the paired Task3 advantage of FMT without reducing absolute FMT F1 or
Average Precision?

## Motivation and scope

Most selected projections normalize before GELU.  GELU then reintroduces a
positive mean and sample-dependent magnitude before the representation enters
the residual head.  Earlier projection searches compared pre-activation
LayerNorm and RMSNorm but did not isolate this post-activation distribution.
The registered fixed transforms test whether centering or normalizing that
distribution exposes relative FMT channel structure more effectively.

This is a development-only single-factor search and cannot replace audited
spatial-population confirmations 6.1, 7.2, or 8.1.

## Frozen comparison

- The independently audited 74.1 portfolio supplies one frozen recipe per
  physical family.
- The transform is applied after projection and activation, but before
  Dropout, Gaussian noise, fixed feature scaling, and residual fusion.
- `none` bypasses the module exactly. `center`, `rms`, and `layer` use fixed
  per-sample statistics over the auxiliary dimension with epsilon `1e-5`.
- The transform has no trainable parameters, buffers, checkpoint state, or
  random-number consumption.
- FMT and train-only Raw-PCA use the same mode, parameter count, seed, split,
  and training budget. Confirmation remains closed.

## Registered grid and selection

Four modes across ten datasets and three paired seeds yield 40 array mappings
and 240 paired trainings.

A non-control candidate is eligible only if FMT F1 and FMT Average Precision
are both no lower than the exact control.  Eligible candidates are selected per
physical family by paired dataset-macro F1 gain, absolute FMT F1, and the
registered robustness tie-breakers.  The joint target is F1 gain at least
`+0.216` and absolute FMT F1 at least `0.893`.  Failure is retained.

## Status

Preregistered while 57.1 was incomplete and before 58.1--74.1 produced
performance results.  No 75.1 performance artifact has been generated or read.
Because Slurm no longer accepted a new dependency on completed historical job
`51096759`, a committed read-only upstream gate verifies that job's terminal
state plus the passed 74.1 audit and all 80 frozen-file identities. The 75.1
preflight must depend strictly on this gate; the scientific protocol is
unchanged.
