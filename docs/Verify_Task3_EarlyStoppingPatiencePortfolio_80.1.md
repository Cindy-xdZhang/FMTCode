# Verify_Task3_EarlyStoppingPatiencePortfolio_80.1

## Purpose

80.1 is a training-free, per-physical-family portfolio between the independently
audited 78.1 method and the guarded 79.1 early-stopping-patience winner. It
prevents one global stopping rule from discarding a stronger source recipe for
a particular physical family.

## Frozen rule

The rule was frozen before 79.1 produced any performance metric. For each of
the seven physical families, compare only the audited 78.1 and 79.1 sources.
Apply the same zero-tolerance FMT F1/Average Precision guard and frozen ranking
used by the preceding portfolios. No model is trained and confirmation remains
closed.

After selection, copy both arms for seeds 40 and 41 across ten datasets:

- 40 model checkpoints;
- 40 matching `per_run.csv` files;
- 80 files in total, each verified by SHA-256.

`Audit_Task3_EarlyStoppingPatiencePortfolio_80_1.py` must independently rebuild
the seven family choices and all macro metrics without importing the 80.1
selector. Only after this audit passes may the registered cleanup delete
exactly 540 temporary 79.1 checkpoints.

Targets remain paired dataset-macro F1 gain at least `+0.219` and absolute FMT
F1 at least `0.893`. Audit success proves evidence consistency, not that either
target was reached.

Canonical config SHA-256:
`9d950e04b5c1821d2213559c04afc907b86590ac29bbe3c7f112e6c09d8e5532`.
