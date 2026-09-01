# Verify_Task3_AuxiliaryLinearWeightInitializationPortfolio_84.1

## Question

After 83.1 is independently audited, should any physical family replace its
82.1 recipe with the guarded auxiliary linear-weight initialization winner?

## Frozen protocol

This training-free portfolio rule was preregistered before 83.1 produced any
performance metric.

- source A: independently audited 82.1 portfolio;
- source B: preregistered and independently audited 83.1 search winner;
- datasets: the ten frozen Task3 development entries;
- source paired seeds: 40, 41 and 42;
- frozen artifacts: seeds 40 and 41 for both arms, giving 40 models and 40
  per-run result files;
- confirmation data remain closed and this experiment performs no training.

For each physical family, a source-B candidate is eligible only when its FMT F1
and FMT Average Precision do not fall below the source control. Eligible rows
use the same paired-gain ranking and tie-breakers as 83.1. Source A remains the
fallback.

## Independent evidence

The selector verifies both source identities and freezes the selected files.
`Audit_Task3_AuxiliaryLinearWeightInitializationPortfolio_84_1.py` does not
import the 84.1 selector; it independently rebuilds the choice and macro
statistics and verifies the SHA-256 of all 80 frozen files. Only after that
audit succeeds may the 540 temporary 83.1 checkpoints be deleted.

Targets are dataset-macro F1 gain at least `+0.222` and absolute FMT F1 at
least `0.894`. Reaching them still does not open a new confirmation population.

Canonical config SHA-256:
`a761c073a89c90d323eb8cdbc01c01f947b1e66f4aa393fa9591507e246bd4eb`.
