# Verify_Task3_ResidualHeadBeta1RefreshPortfolio_108.1

## Purpose

Make a training-free, preregistered per-family choice between independently
audited portfolio 106.1 and the 107.1 residual-head Adam beta1 winner.

## Frozen protocol

- Sources: 106.1 and 107.1 only; both source config hashes are frozen.
- Source paired seeds: 40--42. Frozen artifact seeds: 40--41.
- No training and no confirmation-population access.
- Require each source's zero-tolerance absolute-FMT guard before comparison.
- Select per physical family using dataset-macro paired F1 gain followed by
  absolute FMT F1, Average Precision gain, absolute FMT Average Precision,
  positive-dataset count, worst-dataset gain and worst-seed gain.
- Targets: paired F1 gain at least `+.246` and absolute FMT F1 at least `.906`.

`Audit_Task3_ResidualHeadBeta1RefreshPortfolio_108_1.py` does not import the
108.1 selector. It independently reconstructs all seven family choices, macro
metrics and hashes for 40 result files plus 40 checkpoints. Audit success means
evidence consistency, not automatic scientific success, and does not open a
new confirmation population.

## Status

Preregistered before 107.1 produced any performance metric. No result is yet
available, and no conclusion is claimed.
