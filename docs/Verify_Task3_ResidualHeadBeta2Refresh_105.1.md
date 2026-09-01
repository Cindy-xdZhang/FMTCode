# Verify_Task3_ResidualHeadBeta2Refresh_105.1

## Question

Does independently changing Adam's second-moment decay coefficient for the
downstream residual correction head improve the paired FMT-versus-Raw-PCA
Task3 result? Experiment 34.1 changed global Adam coefficients and 59.1/61.1
changed only auxiliary-projection coefficients. None isolated the residual
head's second-moment timescale.

## Frozen protocol

- Source: independently audited portfolio 104.1, one complete recipe per
  physical family.
- Population: unchanged 5.2 development split, 10 datasets, paired seeds
  40--42, IVD p95 labels.
- Exact source control plus downstream-head Adam beta2 values
  `{0,.5,.8,.9,.95,.98,.99,.995,.999,.9999}`.
- The override applies to every trainable parameter outside `fmt_encoder`.
  The auxiliary projection keeps its source beta values and other optimizer
  settings; the frozen Raw backbone is never optimized.
- Both FMT and Raw-PCA arms use the same candidate value, architecture,
  initialization, split, seed and budget.
- 11 candidates × 10 datasets × 3 seeds × 2 arms = 660 trainings.
- Selection uses dataset-macro paired F1 gain, the registered tie breakers,
  and zero-tolerance FMT F1/Average Precision guards against the exact source.
- Targets: paired F1 gain at least `+.244` and absolute FMT F1 at least `.905`.
- Confirmation remains closed.

## Evidence and cleanup

The selector runs only after all 110 array children succeed. The evidence job
independently recomputes selection and macro metrics, archives all 660
`per_run.csv` files without checkpoints, verifies SHA-256 stability, and keeps
all 660 checkpoints until portfolio 106.1 freezes and independently audits its
40 models plus 40 results. Cleanup is restricted to the exact resolved 105.1
candidate directory.

## Status

Preregistered before 103.1 produced any performance metric and while 77.1 was
incomplete. No result is available, and no conclusion is claimed.
