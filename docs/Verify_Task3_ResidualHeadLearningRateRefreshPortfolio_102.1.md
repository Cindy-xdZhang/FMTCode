# Verify_Task3_ResidualHeadLearningRateRefreshPortfolio_102.1

## Question

After 101.1 is independently audited, should any physical family replace its
100.1 recipe with the guarded residual-head learning-rate winner?

## Frozen protocol

This training-free portfolio rule was frozen before 101.1 produced any
performance metric.

- source A: independently audited 100.1 portfolio;
- source B: preregistered and independently audited 101.1 search;
- datasets: the ten frozen Task3 development entries;
- source paired seeds: 40, 41 and 42;
- frozen artifacts: seeds 40 and 41 for both arms, giving 40 models and 40
  per-run result files;
- confirmation data remain closed and this experiment performs no training.

For each physical family, source B is eligible only when its FMT F1 and FMT
Average Precision do not fall below the exact 101.1 source control. Eligible
rows use the same paired-gain ranking and tie-breakers; source A is the
fallback.

## Independent evidence

The selector verifies both source identities and freezes selected files.
`Audit_Task3_ResidualHeadLearningRateRefreshPortfolio_102_1.py` does not import
the 102.1 selector; it independently rebuilds choices and macro statistics and
verifies the SHA-256 of all 80 frozen files. Only after that audit succeeds may
the 540 temporary 101.1 checkpoints be deleted.

Targets are dataset-macro F1 gain at least `+0.240` and absolute FMT F1 at
least `0.903`. Reaching them still does not open a confirmation population.

Canonical config SHA-256:
`a8fa82d2be226ac2e2610914a0d8491597b5737ede0ecccfd6b5f6f26c5a53a0`.
