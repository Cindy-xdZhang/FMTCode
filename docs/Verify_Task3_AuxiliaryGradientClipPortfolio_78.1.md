# Verify_Task3_AuxiliaryGradientClipPortfolio_78.1

## Purpose

This training-free selector is frozen before 77.1 produces any performance
metric. For each physical family it compares the independently audited 76.1
portfolio winner with the independently audited 77.1 projection-gradient-cap
winner under the same registered metric order.

## Frozen rules

- Development populations only; confirmation remains closed.
- Both sources must pass independent audit and every preregistered SHA-256
  value must match.
- A 77.1 family winner is eligible only after its zero-tolerance absolute FMT
  F1 and Average Precision guard passes.
- Ranking is paired dataset-macro F1 gain, absolute FMT F1, Average Precision
  gain, absolute FMT Average Precision, positive-dataset count, worst-dataset
  gain, then worst-seed gain.
- Exactly 40 models and 40 result rows for paired seeds 40--41 are frozen before
  temporary 77.1 checkpoints are deleted.

The targets are F1 gain `>= +0.218` and absolute FMT F1 `>= 0.893`. Missing
either target remains a negative result and does not open confirmation.

## Status

Preregistered only; no source performance artifact has been read.
