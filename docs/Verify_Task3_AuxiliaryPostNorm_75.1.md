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

Deployed on Ibex from commit `84ec1c3`: upstream gate `51139428` completed;
preflight/array/selector/evidence are `51139429/51139431/51139432/51139434`.
The array completed before any candidate metric was read.  All 40 array
children and all 240 paired trainings exited zero.  The selector, independent
evidence job, and downstream cleanup also exited zero; every recorded stderr
file is empty.  The GPU children used 31 GTX 1080 Ti, 6 P100, 2 V100, and 1
RTX 2080 Ti allocation.

## Results

The exact `none` control produced dataset-macro Raw-PCA/FMT F1
`0.684115/0.890836`, F1 gain `+0.206721`, and Average Precision gain
`+0.222986`.  The registered family-wise selector chose `rms` for DeltaWing
and `layer` for F22; Boeing 747, channel, halfcylinder, SmokeBuoyancy, and
Tangaroa retained `none`.  The resulting Raw-PCA/FMT F1 was
`0.683970/0.890863`, F1 gain was `+0.206893`, and Average Precision gain was
`+0.223263`.  All ten dataset gains were positive and the worst dataset F1
gain was `+0.055787`.

The selected transform therefore changed the control by only `+0.000172` F1
gain and `+0.000027` absolute FMT F1.  It did not reach either the registered
F1-gain target `+0.216` or the absolute FMT-F1 target `0.893`.  This is a valid
negative development result: post-activation fixed normalization is not a new
overall Task3 winner.

## Evidence

- selection, leaderboard, per-run archive, and remote independent-audit
  SHA-256: `2f9d7df3...58077`, `d0b69383...ba14a`,
  `e5b76cf3...8fe49`, and `8fdddcf7...590`;
- the remote independent audit reconstructed 240 rows, four candidates, ten
  datasets, seven families, three seeds, and both arms, with maximum selector
  difference `4.44e-16`;
- a separately downloaded checkpoint-free archive passed every published
  SHA-256 check.  A second local invocation of
  `Audit_Task3_ParameterSearch.py` reproduced the same seven family choices
  and all dataset-macro metrics within `2.22e-16` (local audit SHA-256
  `83f994ca...b3fc`);
- evidence archive metadata SHA-256 is `a32f605d...351b`;
- cleanup job `51139440` ran only after the 76.1 independent audit, deleting
  exactly 240 temporary checkpoints and confirming zero remained.
