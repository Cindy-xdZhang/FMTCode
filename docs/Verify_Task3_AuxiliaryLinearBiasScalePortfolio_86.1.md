# Verify_Task3_AuxiliaryLinearBiasScalePortfolio_86.1

## Question

After 85.1 is independently audited, should any physical family replace its
84.1 recipe with the guarded auxiliary Linear-bias scale winner?

## Frozen protocol

This training-free portfolio rule was preregistered before 85.1 produced any
performance metric.

- source A: independently audited 84.1 portfolio;
- source B: preregistered and independently audited 85.1 search winner;
- datasets: the ten frozen Task3 development entries;
- source paired seeds: 40, 41 and 42;
- frozen artifacts: seeds 40 and 41 for both arms, giving 40 models and 40
  per-run result files;
- confirmation data remain closed and this experiment performs no training.

For each physical family, source B is eligible only when its FMT F1 and FMT
Average Precision do not fall below its exact source control. Eligible rows use
the same paired-gain ranking and tie-breakers as 85.1; source A is the fallback.

## Independent evidence

The selector verifies both source identities and freezes selected files.
`Audit_Task3_AuxiliaryLinearBiasScalePortfolio_86_1.py` does not import the
86.1 selector; it independently rebuilds the choices and macro statistics and
verifies the SHA-256 of all 80 frozen files. Only after that audit succeeds may
the 480 temporary 85.1 checkpoints be deleted.

Targets are dataset-macro F1 gain at least `+0.224` and absolute FMT F1 at
least `0.895`. Reaching them still does not open a confirmation population.

Canonical config SHA-256:
`50f66f43984bd48716038b2b29a96d5e75d6af97b81a82c55bddb4546f8fabb3`.
