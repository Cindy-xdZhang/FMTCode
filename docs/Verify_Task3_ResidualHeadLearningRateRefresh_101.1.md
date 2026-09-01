# Verify_Task3_ResidualHeadLearningRateRefresh_101.1

## Question

Can a learning rate specific to the downstream residual head improve
supervised 3D Task3 FMT gain after the auxiliary projection and head design
have been selected, without reducing absolute FMT F1 or Average Precision?

## Evidence-based motivation

Experiment 27.1 searched one global learning rate, while 55.1 searched only the
trainable auxiliary projection's learning-rate multiplier. No completed
experiment has separated the residual head's optimization rate from both the
auxiliary projection and the frozen Raw backbone. This distinction matters
because the projection must learn an FMT representation while the downstream
head learns how to correct frozen Raw logits; their useful update scales need
not match.

This development experiment was frozen while 77.1 remained incomplete and
before 100.1 could produce any portfolio metric. It does not use confirmation
data or partial metrics from unfinished arrays.

## Frozen protocol

- source: one complete recipe per physical family from independently audited
  100.1;
- datasets: the ten frozen Task3 development entries with IVD-p95 labels;
- paired seeds: 40, 41 and 42;
- arms: FMT and same-width train-only Raw-PCA residuals;
- control: the exact 100.1 source recipe, with no optimizer override;
- downstream head multiplier: `0.05`, `0.10`, `0.25`, `0.50`, `2`, `4`, `8`
  or `16` relative to the source base learning rate;
- 9 candidates × 10 datasets × 3 seeds × 2 arms = 540 trainings;
- the multiplier applies to every trainable parameter outside `fmt_encoder`;
  the Raw backbone remains frozen, and the source auxiliary-projection
  optimizer settings are unchanged;
- both paired arms use exactly the same multiplier, architecture, split, seed
  and training budget;
- a multiplier of one is represented only by the exact source control and
  preserves the historical flat optimizer parameter list when no existing
  auxiliary-group override is active;
- full preflight must reject any cell at or above the frozen Raw-wide parameter
  cap before GPU submission.

## Selection and evidence

A candidate is eligible within a physical family only when its FMT F1 and FMT
Average Precision are both no lower than the exact source control. Eligible
rows use paired F1 gain and the frozen tie-breakers. Targets are dataset-macro
F1 gain at least `+0.240` and absolute FMT F1 at least `0.903`; missing either
target is retained as a valid negative result, and meeting them does not open a
new confirmation population.

`Audit_Task3_ParameterSearch.py` independently reconstructs all 540 paired run
rows and the guarded selection. Per-run CSV files are archived without model
checkpoints; 540 temporary checkpoints remain only until 102.1 freezes and
independently audits 40 models and 40 result files.

Canonical config SHA-256:
`d1b07663c2450cdc1f382c02efc1c1cf7373823952051ac7dcb1852827195ec8`.
