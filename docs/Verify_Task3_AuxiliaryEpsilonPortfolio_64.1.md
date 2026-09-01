# Verify_Task3_AuxiliaryEpsilonPortfolio_64.1

## Purpose

This training-free development selector is frozen before 63.1 produces any
performance metric. For each physical family it compares the independently
audited 62.1 portfolio winner with the independently audited 63.1
auxiliary-epsilon winner under the same registered metric order.

## Frozen rules

- Selection uses development populations only; confirmation remains closed.
- Both sources must pass independent audit and all source hashes must match the
  preregistered configuration.
- A 63.1 family winner is eligible only after its zero-tolerance absolute FMT
  F1 and Average Precision guard passes. The 62.1 source is accepted only after
  its 80 frozen artifacts pass the existing independent audit.
- Ranking is paired dataset-macro F1 gain, absolute FMT F1, Average Precision
  gain, absolute FMT Average Precision, positive-dataset count, worst-dataset
  gain, then worst-seed gain.
- The selector freezes exactly 40 models and 40 result rows for paired seeds
  40--41 before temporary 63.1 checkpoints are deleted.

The targets are F1 gain `>= +0.210` and absolute FMT F1 `>= 0.893`. Missing
either target is retained as a negative result. Reaching them does not open a
new confirmation population.

## Status

Completed and independently audited on 2026-09-01. The guarded portfolio gives
dataset-macro Raw-PCA/FMT F1 `0.683459/0.890572` (gain `+0.207113`) and Average
Precision gain `+0.223384`. It is the best audited development portfolio in
58.1--74.1. All 40 models and 40 result files pass SHA-256 verification. The
joint target is not reached because absolute FMT F1 remains below `0.893`.
