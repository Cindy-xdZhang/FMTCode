# Verify_Task3_ResidualHeadDepthWidth_31.2

## Revision from 31.1

The 31.1 local full preflight rejected `smokeBuoyancy/c14_w96_d3` because it
exceeded the frozen Raw-wide parameter cap. No training or performance result
was produced. Version 31.2 removes the complete width-96 level and preserves
all other scientific choices.

## Frozen protocol and question

The question remains whether residual-head hidden width and depth can improve
both absolute FMT Task3 classification and its paired advantage over a
same-width train-only Raw-PCA arm. Development populations, IVD-p95 labels,
splits, frozen Raw checkpoints, 22.1 family-specific anchored FMT features,
Layer Normalization, Gaussian Error Linear Unit activation, zero dropout,
optimizer, loss, fusion selection, training budget, and paired seeds 40--42
remain frozen. Confirmation remains closed.

## Capacity-safe factorial

Widths are `32, 48, 64, 80`; depths are `1, 2, 3`. Their Cartesian product
gives 12 candidates. Width 64 and depth 2 is the exact historical control.
The experiment contains 12 candidates, 10 data entries, 3 paired seeds, and
2 arms: 720 trainings in 120 GPU array jobs.

Every candidate is applied unchanged to FMT and train-only Raw-PCA. Full
preflight must verify that every candidate remains below the 148,225-parameter
Raw-wide cap before Ibex GPU submission.

## Selection and targets

Family-specific selection first requires absolute FMT F1 and Average Precision
to be no lower than the exact control. Eligible candidates are ranked by paired
F1 gain, then absolute FMT F1 and the registered robustness tie-breakers.

Joint development target: F1 gain over Raw-PCA at least `0.195` and absolute
FMT F1 at least `0.893`. Failure of either requirement is retained as a
negative result. Any winner still requires a fresh spatial population.

## Ibex deployment

Implementation commit `b7b7a9b` was pushed before deployment. The archive
SHA-256 is `b28b1d62...f819`; the remote canonical config SHA-256 is
`2de75e79...7a44`. Local and remote runs both passed the 11 relevant unit
tests; remote Python compilation and all three `bash -n` checks also passed.

Submitted at `2026-08-30T02:05:34+03:00`: preflight job `51003042`, GPU array
`51003043[0-119%24]`, and selector `51003044`. The GPU array depends on the
preflight, and the selector depends on every array child. No partial metric is
used before the selector completes.

The remote full preflight completed at `02:06:53+03:00` in 78 seconds with
exit code zero and empty stderr. It confirmed 10 datasets, 12 candidates, 120
array mappings, 720 paired trainings, no capacity violation, and closed
confirmation state. Its manifest SHA-256 is `fc7ef7ba...19ca`. The GPU array
dependency is released and currently waits only for the user GPU quota.
