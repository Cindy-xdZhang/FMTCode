# Verify_Task3_AuxiliaryActivationInputShiftPortfolio_92.1

## Question

After 91.1 is independently audited, should any physical family replace its
90.1 recipe with the guarded activation-input-shift winner?

## Frozen protocol

This training-free portfolio rule was preregistered before 91.1 produced any
performance metric.

- source A: independently audited 90.1 portfolio;
- source B: preregistered and independently audited 91.1 search winner;
- datasets: the ten frozen Task3 development entries;
- source paired seeds: 40, 41 and 42;
- frozen artifacts: seeds 40 and 41 for both arms, giving 40 models and 40
  per-run result files;
- confirmation data remain closed and this experiment performs no training.

For each physical family, source B is eligible only when its FMT F1 and FMT
Average Precision do not fall below its exact source control. Eligible rows use
the same paired-gain ranking and tie-breakers as 91.1; source A is the fallback.

## Independent evidence

The selector verifies both source identities and freezes selected files.
`Audit_Task3_AuxiliaryActivationInputShiftPortfolio_92_1.py` does not import
the 92.1 selector; it independently rebuilds the choices and macro statistics
and verifies the SHA-256 of all 80 frozen files. Only after that audit succeeds
may the 540 temporary 91.1 checkpoints be deleted.

Targets are dataset-macro F1 gain at least `+0.230` and absolute FMT F1 at
least `0.898`. Reaching them still does not open a confirmation population.

Canonical config SHA-256:
`7417ddc3b429dc30ccbbe8ff0ea8c9685ca7fc38ebb83fa5956a26f2418cf1c3`.
