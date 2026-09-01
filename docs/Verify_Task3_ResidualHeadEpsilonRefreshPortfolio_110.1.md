# Verify_Task3_ResidualHeadEpsilonRefreshPortfolio_110.1

## Purpose

Make a training-free, preregistered per-family choice between independently
audited portfolio 108.1 and the 109.1 residual-head Adam epsilon winner.

## Frozen protocol

- Sources: 108.1 and 109.1 only; both source config hashes are frozen.
- Source paired seeds: 40--42. Frozen artifact seeds: 40--41.
- No training and no confirmation-population access.
- Require each source's zero-tolerance absolute-FMT guard before comparison.
- Select per physical family using dataset-macro paired F1 gain followed by
  absolute FMT F1, Average Precision gain, absolute FMT Average Precision,
  positive-dataset count, worst-dataset gain and worst-seed gain.
- Targets: paired F1 gain at least `+.248` and absolute FMT F1 at least `.907`.

`Audit_Task3_ResidualHeadEpsilonRefreshPortfolio_110_1.py` does not import the
110.1 selector. It independently reconstructs all seven family choices, macro
metrics and hashes for 40 result files plus 40 checkpoints. Audit success means
evidence consistency, not automatic scientific success, and does not open a
new confirmation population.

## Status

Preregistered before 109.1 produced any performance metric. No result is yet
available, and no conclusion is claimed.
