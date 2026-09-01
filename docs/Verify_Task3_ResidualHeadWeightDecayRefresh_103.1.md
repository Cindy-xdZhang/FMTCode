# Verify_Task3_ResidualHeadWeightDecayRefresh_103.1

## Question

Does independently scaling weight decay on the downstream residual correction
head improve the paired FMT-versus-Raw-PCA Task3 result? Earlier experiment
27.1 changed global weight decay and 57.1 changed only auxiliary-projection
weight decay; neither separated downstream-head regularization from the
projection.

## Frozen protocol

- Source: independently audited portfolio 102.1, one complete recipe per
  physical family.
- Population: unchanged 5.2 development split, 10 datasets, paired seeds
  40--42, IVD p95 labels.
- Exact source control plus downstream-head weight-decay multipliers
  `{0,.01,.05,.10,.25,.50,2,4,10,100}`.
- The multiplier applies to every trainable parameter outside `fmt_encoder`.
  The auxiliary projection keeps its source optimizer settings; the frozen Raw
  backbone is never optimized.
- Both FMT and Raw-PCA arms use the same candidate multiplier, architecture,
  initialization, split, seed and budget.
- 11 candidates × 10 datasets × 3 seeds × 2 arms = 660 trainings.
- Selection uses dataset-macro paired F1 gain, the registered tie breakers,
  and zero-tolerance FMT F1/Average Precision guards against the exact source.
- Targets: paired F1 gain at least `+.242` and absolute FMT F1 at least `.904`.
- Confirmation remains closed.

## Evidence and cleanup

The selector runs only after all 110 array children succeed. The evidence job
independently recomputes selection and macro metrics, archives all 660
`per_run.csv` files without checkpoints, verifies SHA-256 stability, and keeps
all 660 checkpoints until portfolio 104.1 freezes and independently audits its
40 models plus 40 results. Cleanup is then restricted to the exact resolved
103.1 candidate directory.

## Status

Preregistered before 101.1 produced any performance metric. No result is yet
available, and no conclusion is claimed.
