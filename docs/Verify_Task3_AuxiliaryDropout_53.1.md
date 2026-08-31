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

## Status

Implementation is complete. Local Python compilation and 56 directly related
`unittest` contracts pass. The full local preflight also passes and records ten
datasets, eleven candidates, three paired seeds, two arms, 110 array mappings,
660 trainings, all capacity guards, and closed confirmation. Its manifest
SHA-256 is `2a17ac826b302f616ae04e7f2840f1e344eef77c8bd781147b06521643e43ad8`.
The three Slurm files are LF-encoded; their local and remote SHA-256 values
match exactly. Ibex preflight job `51069690` completed with exit code 0 and an
empty stderr, producing a manifest with SHA-256
`bc4b60c644559be1b65208f55c1e5c4ae777652cab1a7eaf502903fed6cf8bc2`.
GPU array `51069691` and selector `51069692` are deployed. The array is held by
`afterany:51060469`, so it starts only after the already frozen 7.2 independent
confirmation chain reaches a terminal state; the selector requires every
array child to succeed. No 53.1 performance result has been produced or read.
