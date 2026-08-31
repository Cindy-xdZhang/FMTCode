# Verify_Task3_AuxiliaryLearningRatePortfolio_56.1

## Purpose

This training-free development selector was frozen before 55.1 produced any
performance metric. For each physical family it compares the audited 54.1
portfolio winner with the independently selected 55.1 auxiliary-projection
learning-rate winner under the same registered metric order.

## Frozen rules

- Selection uses development populations only; confirmation remains closed.
- Both sources must have passed their independent audit and all source hashes
  must match the preregistered configuration.
- A 55.1 family winner is eligible only after its zero-tolerance absolute FMT
  F1 and Average Precision guard passes. The 54.1 source is accepted only after
  its 80 frozen artifacts pass the existing independent audit.
- Ranking is paired dataset-macro F1 gain, absolute FMT F1, Average Precision
  gain, absolute FMT Average Precision, positive-dataset count, worst-dataset
  gain, then worst-seed gain.
- The selector freezes exactly 40 models and 40 result rows for paired seeds
  40--41 before temporary 55.1 checkpoints are deleted.

The targets are gain `>= +0.20552` (the completed 54.1 portfolio value) and
absolute FMT F1 `>= 0.893`. Missing either target is retained as a negative
result. Reaching them does not open a new confirmation population.

## Status

Preregistered. No 55.1 or 56.1 performance artifact has been read.
