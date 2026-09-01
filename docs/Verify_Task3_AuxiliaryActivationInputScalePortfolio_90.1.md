# Verify_Task3_AuxiliaryActivationInputScalePortfolio_90.1

## Question

After 89.1 is independently audited, should any physical family replace its
88.1 recipe with the guarded activation-input-scale winner?

## Frozen protocol

This training-free portfolio rule was preregistered before 89.1 produced any
performance metric.

- source A: independently audited 88.1 portfolio;
- source B: preregistered and independently audited 89.1 search winner;
- datasets: the ten frozen Task3 development entries;
- source paired seeds: 40, 41 and 42;
- frozen artifacts: seeds 40 and 41 for both arms, giving 40 models and 40
  per-run result files;
- confirmation data remain closed and this experiment performs no training.

For each physical family, source B is eligible only when its FMT F1 and FMT
Average Precision do not fall below its exact source control. Eligible rows use
the same paired-gain ranking and tie-breakers as 89.1; source A is the fallback.

## Independent evidence

The selector verifies both source identities and freezes selected files.
`Audit_Task3_AuxiliaryActivationInputScalePortfolio_90_1.py` does not import
the 90.1 selector; it independently rebuilds the choices and macro statistics
and verifies the SHA-256 of all 80 frozen files. Only after that audit succeeds
may the 480 temporary 89.1 checkpoints be deleted.

Targets are dataset-macro F1 gain at least `+0.228` and absolute FMT F1 at
least `0.897`. Reaching them still does not open a confirmation population.

Canonical config SHA-256:
`0c4c2a81dce39e3b86e7672b14ed3d00322fc90e4d6cac1e87a53c27e15f4d2b`.
