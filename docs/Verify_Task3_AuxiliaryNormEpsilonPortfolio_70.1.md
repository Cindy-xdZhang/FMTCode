# Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1

## Purpose

This training-free development selector is frozen before 69.1 produces any
performance metric.  For each physical family it compares the independently
audited 68.1 portfolio winner with the independently audited 69.1 auxiliary
normalization-epsilon winner under the same registered metric order.

## Frozen rules

- Selection uses development populations only; confirmation remains closed.
- Both sources must pass independent audit and all source hashes must match.
- A 69.1 family winner is eligible only after its zero-tolerance absolute FMT
  F1 and Average Precision guard passes.
- Ranking is paired dataset-macro F1 gain, absolute FMT F1, Average Precision
  gain, absolute FMT Average Precision, positive-dataset count, worst-dataset
  gain, then worst-seed gain.
- The selector freezes exactly 40 models and 40 result rows for paired seeds
  40--41 before temporary 69.1 checkpoints are deleted.

The targets are F1 gain `>= +0.213` and absolute FMT F1 `>= 0.893`.  Missing
either target is retained as a negative result and reaching both does not open
a new confirmation population.

## Status

Preregistered before 69.1 produced any performance artifact.  No 69.1 or 70.1
performance result has been read.
