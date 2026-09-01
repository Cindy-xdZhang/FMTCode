# Verify_Task3_ResidualHeadEpsilonRefresh_109.1

## Question

Does independently changing Adam's denominator epsilon for the downstream
residual correction head improve the paired FMT-versus-Raw-PCA Task3 result?
Experiment 42.1 changed epsilon globally and 63.1 changed only the auxiliary
projection. Neither isolated the residual head's adaptive-step denominator.

## Frozen protocol

- Source: independently audited portfolio 108.1, one complete recipe per
  physical family.
- Population: unchanged 5.2 development split, 10 datasets, paired seeds
  40--42, IVD p95 labels.
- Exact source control plus downstream-head Adam epsilon values
  `{1e-12,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}`.
- The override applies to every trainable parameter outside `fmt_encoder`.
  Both head beta coefficients and all auxiliary optimizer settings remain at
  their source values. The Raw backbone is frozen.
- Both FMT and Raw-PCA arms use the same candidate epsilon, architecture,
  initialization, split, seed and budget.
- 11 candidates × 10 datasets × 3 seeds × 2 arms = 660 trainings.
- Selection uses dataset-macro paired F1 gain, the registered tie breakers,
  and zero-tolerance FMT F1/Average Precision guards against the exact source.
- Targets: paired F1 gain at least `+.248` and absolute FMT F1 at least `.907`.
- Confirmation remains closed.

## Evidence and cleanup

The selector runs only after all 110 array children succeed. The evidence job
independently recomputes selection and macro metrics, archives all 660
`per_run.csv` files without checkpoints, verifies SHA-256 stability, and keeps
all 660 checkpoints until portfolio 110.1 freezes and independently audits its
40 models plus 40 results. Cleanup is restricted to the exact resolved 109.1
candidate directory.

## Status

Preregistered before 107.1 produced any performance metric and while 77.1 was
incomplete. No result is available, and no conclusion is claimed.
