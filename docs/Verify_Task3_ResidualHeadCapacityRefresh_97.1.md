# Verify_Task3_ResidualHeadCapacityRefresh_97.1

## Question

Does residual-head width and depth interact with the substantially newer
auxiliary-projection portfolio strongly enough to improve supervised 3D Task3
FMT gain without reducing absolute FMT F1 or Average Precision?

## Why this is a new test rather than a repeat chosen after seeing results

The only complete head-capacity search, 31.2, was conducted on the much older
22.1 anchored representation. It found family-specific width/depth choices but
did not meet its joint target. Experiments 32.1--96.1 subsequently changed the
auxiliary feature construction, projection and training recipe. Head capacity
has never been re-optimized after those changes.

This development experiment was frozen while 77.1 remained incomplete and
before 96.1 could produce any portfolio metric. It does not use confirmation
data or partial metrics from unfinished arrays.

## Frozen protocol

- source: one complete recipe per physical family from independently audited
  96.1;
- datasets: the ten frozen Task3 development entries with IVD-p95 labels;
- paired seeds: 40, 41 and 42;
- arms: FMT and same-width train-only Raw-PCA residuals;
- control: the exact 96.1 source recipe, with no head replacement;
- capacity grid: hidden widths 32, 48, 64 and 80 crossed with hidden depths 1,
  2 and 3;
- 13 candidates × 10 datasets × 3 seeds × 2 arms = 780 trainings;
- only `head_hidden_dim` and `head_depth` may replace source values; source
  features, projections, activation operations, optimizer, loss, split,
  initialization and training budget remain unchanged;
- both paired arms use exactly the same head capacity in each run;
- full preflight must reject any cell at or above the frozen Raw-wide parameter
  cap before GPU submission.

## Selection and evidence

A candidate is eligible within a physical family only when its FMT F1 and FMT
Average Precision are both no lower than the exact source control. Eligible
rows use paired F1 gain and the frozen tie-breakers. Targets are dataset-macro
F1 gain at least `+0.236` and absolute FMT F1 at least `0.901`; missing either
target is retained as a valid negative result, and meeting them does not open a
new confirmation population.

`Audit_Task3_ParameterSearch.py` independently reconstructs all 780 paired run
rows and the guarded selection. Per-run CSV files are archived without model
checkpoints; 780 temporary checkpoints remain only until 98.1 freezes and
independently audits 40 models and 40 result files.

Canonical config SHA-256:
`d1111b1674501fca373cefcf77ae7c11f4595a2c8049f7ed4095662b4d634e1f`.
