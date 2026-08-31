# Verify_Task3_AuxiliaryDropout_53.1

## Question

Does dropout applied only to the projected auxiliary representation increase
the paired Task3 F1 advantage of FMT without lowering absolute FMT F1 or
Average Precision?

## Why this is a distinct experiment

Completed search 38.1 regularized hidden layers after Raw geometry and the
auxiliary representation had already been fused. Experiment 53.1 instead
applies dropout immediately after the matched FMT or train-only Raw-PCA
projection and before fusion with frozen Raw geometry. It tests whether FMT's
redundant Fourier/kinematic channels are more robust to auxiliary-channel
masking than globally mixed Raw-PCA components.

This hypothesis and grid were registered without reading any partial metric
from running searches 44.1, 48.1, 50.1, or 51.1. It is a development-only
experiment and cannot replace independent spatial-population confirmation.

## Frozen comparison

- IVD-p95 labels, exposed development populations, split, frozen Raw models,
  family-specific 22.1 FMT features, auxiliary width, residual head, optimizer,
  epoch budget, early stopping, and seeds are unchanged.
- FMT and train-only Raw-PCA receive the same auxiliary dropout probability in
  each paired cell.
- The regularizer has no parameters and acts at the same network location and
  width in both arms.
- The exact control is the historical `p=0` route with no candidate override.
- Confirmation remains closed.

## Registered grid and selection

The grid is `p={0,.025,.05,.10,.15,.20,.30,.40,.50,.60,.70}`. Eleven
candidates across ten datasets and three paired seeds give 110 array mappings
and 660 paired trainings.

Selection is physical-family specific. A candidate is eligible only if both
FMT F1 and FMT Average Precision are no lower than its exact control. Eligible
candidates are ranked by paired dataset-macro F1 gain, then absolute FMT F1 and
the registered robustness tie-breakers. The joint target is F1 gain at least
`+0.200` and absolute FMT F1 at least `0.893`; failure of either is retained as
a negative result.

## Main files

- `FMT_Utils/PathlineClassifier_3D.py`
- `Search_Task3_FMTResidual_3D.py`
- `config/Verify_Task3_AuxiliaryDropout_53.1.yaml`
- `tests/test_task3_auxiliary_dropout_53_1.py`
- `ibex_bash/verify_task3_auxiliary_dropout_53.1_*.sh`

## Final result

Jobs `51069690`, `51069691`, `51069692`, and `51072617` all completed with
exit code zero. The array produced all 660 paired trainings; selector and
evidence stderr were empty. The stable per-run archive contains 660 CSV files,
no model file, and has SHA-256
`92dadd082eaac7433bb4c27b0d5d0c6b67e4143e53725d833a6ee317fe62f75e`.
The selection and leaderboard hashes are `9fddfe44…d3ce` and
`0ea3afc0…313`.

`Audit_Task3_ParameterSearch.py` independently reconstructed all 77
family-candidate rows and the seven family winners directly from the archive.
Its maximum difference from the selector was `1.11e-16`; all source hashes and
paired parameter counts agree. Dataset-macro Raw-PCA/FMT F1 is
`0.69642/0.88775` (gain `+0.19133`), and Average Precision is
`0.73957/0.94615` (gain `+0.20658`). All ten datasets have positive F1 gain;
the worst is `+0.05409`.

The exact no-dropout control has F1 `0.69738/0.88712` (gain `+0.18975`).
Family-specific dropout therefore improves absolute FMT F1 by only `+0.00062`
and paired gain by `+0.00158`. The registered `+0.20` F1-gain and `0.893`
absolute-FMT targets both fail. The result is valid development evidence but
does not support auxiliary dropout as the next main improvement.
