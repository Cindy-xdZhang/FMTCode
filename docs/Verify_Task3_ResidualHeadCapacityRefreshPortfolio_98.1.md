# Verify_Task3_ResidualHeadCapacityRefreshPortfolio_98.1

## Question

After 97.1 is independently audited, should any physical family replace its
96.1 recipe with the guarded residual-head capacity winner?

## Frozen protocol

This training-free portfolio rule was frozen before 97.1 produced any
performance metric.

- source A: independently audited 96.1 portfolio;
- source B: preregistered and independently audited 97.1 capacity search;
- datasets: the ten frozen Task3 development entries;
- source paired seeds: 40, 41 and 42;
- frozen artifacts: seeds 40 and 41 for both arms, giving 40 models and 40
  per-run result files;
- confirmation data remain closed and this experiment performs no training.

For each physical family, source B is eligible only when its FMT F1 and FMT
Average Precision do not fall below the exact 97.1 source control. Eligible
rows use the same paired-gain ranking and tie-breakers; source A is the
fallback.

## Independent evidence

The selector verifies both source identities and freezes selected files.
`Audit_Task3_ResidualHeadCapacityRefreshPortfolio_98_1.py` does not import the
98.1 selector; it independently rebuilds all choices and macro statistics and
verifies the SHA-256 of all 80 frozen files. Only after that audit succeeds may
the 780 temporary 97.1 checkpoints be deleted.

Targets are dataset-macro F1 gain at least `+0.236` and absolute FMT F1 at
least `0.901`. Reaching them still does not open a confirmation population.

Canonical config SHA-256:
`b8687fc41ed3314dc27a40ec509082113c5059cce32a211298efcb8d0b79f40d`.
