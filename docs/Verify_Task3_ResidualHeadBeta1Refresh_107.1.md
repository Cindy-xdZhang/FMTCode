# Verify_Task3_ResidualHeadBeta1Refresh_107.1

## Question

Does independently changing Adam's first-moment decay coefficient for the
downstream residual correction head improve the paired FMT-versus-Raw-PCA
Task3 result? Experiment 34.1 changed global Adam coefficients and 61.1 changed
only the auxiliary projection's beta1. Neither isolated residual-head momentum.

## Frozen protocol

- Source: independently audited portfolio 106.1, one complete recipe per
  physical family.
- Population: unchanged 5.2 development split, 10 datasets, paired seeds
  40--42, IVD p95 labels.
- Exact source control plus downstream-head Adam beta1 values
  `{0,.25,.5,.7,.8,.85,.9,.95,.98,.99}`.
- The override applies to every trainable parameter outside `fmt_encoder`.
  The head inherits source beta2; the auxiliary projection keeps both source
  beta values and all other optimizer settings. The Raw backbone is frozen.
- Both FMT and Raw-PCA arms use the same candidate value, architecture,
  initialization, split, seed and budget.
- 11 candidates × 10 datasets × 3 seeds × 2 arms = 660 trainings.
- Selection uses dataset-macro paired F1 gain, the registered tie breakers,
  and zero-tolerance FMT F1/Average Precision guards against the exact source.
- Targets: paired F1 gain at least `+.246` and absolute FMT F1 at least `.906`.
- Confirmation remains closed.

## Evidence and cleanup

The selector runs only after all 110 array children succeed. The evidence job
independently recomputes selection and macro metrics, archives all 660
`per_run.csv` files without checkpoints, verifies SHA-256 stability, and keeps
all 660 checkpoints until portfolio 108.1 freezes and independently audits its
40 models plus 40 results. Cleanup is restricted to the exact resolved 107.1
candidate directory.

## Status

Preregistered before 105.1 produced any performance metric and while 77.1 was
incomplete. No result is available, and no conclusion is claimed.
