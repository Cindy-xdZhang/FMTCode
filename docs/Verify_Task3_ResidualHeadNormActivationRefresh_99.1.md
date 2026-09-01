# Verify_Task3_ResidualHeadNormActivationRefresh_99.1

## Question

Do normalization and activation inside the residual head interact with the
new auxiliary-projection and head-capacity portfolio strongly enough to improve
supervised 3D Task3 FMT gain without reducing absolute FMT F1 or Average
Precision?

## Evidence-based motivation

The only complete head normalization/activation search, 29.1, used the old
22.1 anchored representation. It did not improve the overall result, but it
selected RMSNorm for Channel and Smoke and ReLU for the half-cylinder family,
showing a real family-specific interaction. Experiments 30.1--98.1 subsequently
changed the representation, projection, regularization and head capacity. This
factorial has never been repeated on that newer stack.

This development experiment was frozen while 77.1 remained incomplete and
before 98.1 could produce any portfolio metric. It does not use confirmation
data or partial metrics from unfinished arrays.

## Frozen protocol

- source: one complete recipe per physical family from independently audited
  98.1;
- datasets: the ten frozen Task3 development entries with IVD-p95 labels;
- paired seeds: 40, 41 and 42;
- arms: FMT and same-width train-only Raw-PCA residuals;
- control: the exact 98.1 source recipe, with no head replacement;
- factorial: LayerNorm, RMSNorm or no normalization crossed with Gaussian Error
  Linear Unit, Sigmoid Linear Unit or Rectified Linear Unit activation;
- 10 candidates × 10 datasets × 3 seeds × 2 arms = 600 trainings;
- only `head_normalization` and `head_activation` may replace source values;
  source features, projections, capacity, optimizer, loss, split,
  initialization and training budget remain unchanged;
- both paired arms use exactly the same head design in each run;
- full preflight must reject any cell at or above the frozen Raw-wide parameter
  cap before GPU submission.

## Selection and evidence

A candidate is eligible within a physical family only when its FMT F1 and FMT
Average Precision are both no lower than the exact source control. Eligible
rows use paired F1 gain and the frozen tie-breakers. Targets are dataset-macro
F1 gain at least `+0.238` and absolute FMT F1 at least `0.902`; missing either
target is retained as a valid negative result, and meeting them does not open a
new confirmation population.

`Audit_Task3_ParameterSearch.py` independently reconstructs all 600 paired run
rows and the guarded selection. Per-run CSV files are archived without model
checkpoints; 600 temporary checkpoints remain only until 100.1 freezes and
independently audits 40 models and 40 result files.

Canonical config SHA-256:
`5c3fd8f7bc82b477ff456eff651d879f2247fc429c2ca86f7345025dfa656ad5`.
