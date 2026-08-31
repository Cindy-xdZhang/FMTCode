# Verify_Task3_FocalGammaLow_50.1

## Question

Does resolving focal gamma below `0.10` increase the paired F1 advantage of
FMT without lowering absolute FMT F1 or Average Precision?

## Motivation and evidence boundary

Completed development search 39.1 selected gamma `0.10` for Channel and the
half-cylinder family, exactly at that grid's lower focal boundary. All other
families retained weighted binary cross entropy, and larger gamma values
generally failed the zero-regression absolute-FMT guard. The frozen parent
selection SHA-256 is `1e99c432...9e994`.

This is an adaptive development follow-up. It uses no fresh confirmation
population and cannot establish a paper-level claim by itself.

## Frozen comparison

- The development populations, IVD-p95 labels, split, Raw checkpoints,
  family-specific 22.1 FMT features, network, optimizer, batch size, epoch
  budget, early stopping, and seeds remain identical to 39.1.
- Both FMT and train-only Raw-PCA receive the same loss and gamma in every
  candidate.
- The control is the exact no-override weighted-binary-cross-entropy path.
- Paired seeds are `40`, `41`, and `42`; confirmation remains closed.

## Registered grid and selection

The focal grid is `{0.01,0.025,0.05,0.075,0.10,0.15,0.20}` plus the exact
control. Eight candidates across ten datasets yield 80 array mappings and
480 paired trainings.

Selection remains physical-family specific. A candidate must first keep both
FMT F1 and FMT Average Precision at least equal to the same-family exact
control. Eligible candidates are ranked by paired dataset-macro F1 gain,
absolute FMT F1, and the same registered robustness tie-breakers as 39.1.

The joint target remains F1 gain at least `+0.195` and absolute FMT F1 at
least `0.893`. Failure of either target is a negative result. Any development
winner must enter a separately frozen fresh spatial confirmation before use
as paper-level evidence.

## Main files

- `config/Verify_Task3_FocalGammaLow_50.1.yaml`
- `tests/test_task3_focal_gamma_low_50_1.py`
- `Search_Task3_LossOptimization_7_1.py`
- `ibex_bash/verify_task3_focal_gamma_low_50.1_*.sh`

## Status

Local Python compilation and 13 relevant contract/audit tests pass. Static
preflight confirms 10 datasets, 7 physical families, 8 candidates, 3 paired
seeds, 2 arms, and 480 trainings. Full local preflight also passes all cache,
label, feature-width, parameter-capacity, and frozen-checkpoint checks; its
manifest SHA-256 is `b9bf269a...5ec3d`. No performance result has been read,
and confirmation is closed.

Implementation commit `cf85a72` was pushed before deployment. The immutable
archive SHA-256 is `e23cf300...1c8b1`, identical locally and remotely; the
remote canonical config SHA-256 is `a0b706e8...b5a58`. Remote Python
compilation, the same 13 tests, static preflight, and all four `bash -n`
checks pass. Jobs submitted at `2026-08-30T21:00:51+03:00` are full preflight
`51056257`, GPU array `51056260[0-79%24]`, selector `51056263`, and artifact
job `51056264`. Their dependencies are respectively none,
`afterok:51056257`, `afterok:51056260_*`, and `afterok:51056263`. Performance
results have not been read and confirmation remains closed. Full preflight
`51056257` ran on `cn604-07` from `21:00:53` to `21:02:02`, completed with
exit code 0 and empty stderr, and produced remote manifest SHA-256
`a83e431a...f5c0`. The GPU array is eligible and waiting for Slurm priority.

To avoid delaying independent audit until 52.1 has copied the selected models,
no-delete evidence job `51072242` was submitted with strict
`afterok:51056263`. It archives all 480 per-run CSV files, requires all 480
temporary checkpoints both before and after the archive, excludes model files
from the archive, and publishes stable SHA-256 values. It does not train or
select a candidate. Local and remote script SHA-256 is
`94a448d82d922938dc0b881bb7829c7718d1f03631d850a1c0552fdeb5d52e7f`;
remote `bash -n` passed.

The destructive archive/cleanup job `51056264` now also has strict
`afterok:51072242`, in addition to its selector and 52.1 dependencies. This
prevents cleanup from racing the evidence job's 30-second archive-stability
check.
