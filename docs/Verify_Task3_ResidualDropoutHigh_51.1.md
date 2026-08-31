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
was fixed in commits `7603658` and `5e70e2d`. The immutable deployment archive
SHA-256 is `2c59c46b4de1baba08141e49d41b7d1af47f1b70ad6f812f084404db68dfe54a`,
identical locally and on Ibex. Local raw and remote canonical config SHA-256
values are `c57331663bf32011077706d0444c2aca1a6d76ca331c2a7b0c37cf28750153f3`
and `4f67ee6fbbe9baf779b990eab6515ec18e472f6a3026aa4dad0a5a0aad0713ae`;
their difference is the committed Windows/Unix line ending only. Remote Python
compilation, the same 45 `unittest` tests, and all four `bash -n` checks pass.
The optional `pytest` runner is absent in both environments and is not counted
as passed evidence.

Submitted at `2026-08-30T22:29:49+03:00`: preflight job `51058832`, GPU array
`51058833[0-79%24]`, selector `51058835`, and archive/cleanup job `51058838`.
The dependency chain was verified with `scontrol`: preflight -> complete GPU
array -> selector -> archive. Preflight ran on `cn604-10` from 22:29:50 to
22:31:05, exited zero with empty stderr, and produced remote manifest SHA-256
`ec613eec5b4d41be3a4f2fb02bc082d448d65188c6c0db4dee1d12946ed38c5b`.
It confirmed ten datasets, eight candidates, 80 mappings, 480 paired
trainings, all capacity guards, and closed confirmation. The GPU array is
eligible and waiting for priority; no performance result has been read.

No-delete evidence job `51072260` has strict `afterok:51058835`. It will make
the 480 per-run CSV files independently auditable immediately after selection,
while requiring all 480 temporary checkpoints to remain available for 52.1
and excluding every model file from the archive. It does not train or select a
candidate. Local and remote script SHA-256 is
`3fc03e2d30c2c4c3f47fbd0891bd391deec9ce8d48922f61b503c8d4e4b6e0dc`;
remote `bash -n` passed.
