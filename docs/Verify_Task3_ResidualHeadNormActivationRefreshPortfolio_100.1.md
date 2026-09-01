# Verify_Task3_ResidualHeadNormActivationRefreshPortfolio_100.1

## Question

After 99.1 is independently audited, should any physical family replace its
98.1 recipe with the guarded residual-head normalization/activation winner?

## Frozen protocol

This training-free portfolio rule was frozen before 99.1 produced any
performance metric.

- source A: independently audited 98.1 portfolio;
- source B: preregistered and independently audited 99.1 factorial;
- datasets: the ten frozen Task3 development entries;
- source paired seeds: 40, 41 and 42;
- frozen artifacts: seeds 40 and 41 for both arms, giving 40 models and 40
  per-run result files;
- confirmation data remain closed and this experiment performs no training.

For each physical family, source B is eligible only when its FMT F1 and FMT
Average Precision do not fall below the exact 99.1 source control. Eligible
rows use the same paired-gain ranking and tie-breakers; source A is the
fallback.

## Independent evidence

The selector verifies both source identities and freezes selected files.
`Audit_Task3_ResidualHeadNormActivationRefreshPortfolio_100_1.py` does not
import the 100.1 selector; it independently rebuilds choices and macro
statistics and verifies the SHA-256 of all 80 frozen files. Only after that
audit succeeds may the 600 temporary 99.1 checkpoints be deleted.

Targets are dataset-macro F1 gain at least `+0.238` and absolute FMT F1 at
least `0.902`. Reaching them still does not open a confirmation population.

Canonical config SHA-256:
`f4d4dd2dff8b2601058da13fc9b2874a559119a82b0f681242d92e3482ef540e`.
