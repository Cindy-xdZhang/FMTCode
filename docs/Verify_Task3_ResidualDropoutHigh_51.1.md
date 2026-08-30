# Verify_Task3_ResidualDropoutHigh_51.1

## Question

Does extending residual-head dropout above `0.50` increase the paired Task3
F1 advantage of FMT without lowering absolute FMT F1 or Average Precision?

## Motivation and evidence boundary

Completed development search 38.1 selected dropout `0.50` for both F22 and
Boeing, exactly at the registered grid's upper boundary. Its frozen selection
SHA-256 is `f94c51b7...5c7616`. This leaves the high-dropout region unresolved.

This is an adaptive development-only follow-up. It reads no incomplete search
and no fresh confirmation population. A winner cannot be used as paper-level
evidence until it is frozen and tested on a new spatial population.

## Frozen comparison

- The development populations, IVD-p95 labels, split, Raw checkpoints,
  family-specific 22.1 FMT features, network width/depth, optimizer, batch
  size, epoch budget, early stopping, and seeds remain identical to 38.1.
- FMT and train-only Raw-PCA receive the same dropout rate in each cell.
- The control is the exact zero-dropout historical path with no local model
  override.
- Paired seeds are `40`, `41`, and `42`; confirmation remains closed.

## Registered grid and selection

The grid is the exact control plus dropout
`{0.50,0.55,0.60,0.65,0.70,0.75,0.80}`. Repeating `0.50` makes the parent
boundary directly comparable inside this execution. Eight candidates across
ten datasets yield 80 array mappings and 480 paired trainings.

Selection is physical-family specific. A candidate is eligible only if both
FMT F1 and FMT Average Precision are no lower than the exact control for that
family. Eligible candidates are ranked by paired dataset-macro F1 gain,
absolute FMT F1, and the registered robustness tie-breakers.

The joint target remains F1 gain at least `+0.195` and absolute FMT F1 at
least `0.893`. Failure of either target is a negative result.

## Main files

- `config/Verify_Task3_ResidualDropoutHigh_51.1.yaml`
- `tests/test_task3_residual_dropout_high_51_1.py`
- `Search_Task3_LossOptimization_7_1.py`
- `ibex_bash/verify_task3_residual_dropout_high_51.1_*.sh`

## Status

Local Python compilation, 45 related `unittest` contract/audit tests, static preflight,
and full preflight pass. The full-preflight manifest SHA-256 is
`d19a9415...8b293a`; it confirms ten datasets, seven physical families,
eight candidates, three paired seeds, two arms, and 480 trainings. No
performance result has been read, and confirmation is closed. The experiment
is ready for immutable deployment.
