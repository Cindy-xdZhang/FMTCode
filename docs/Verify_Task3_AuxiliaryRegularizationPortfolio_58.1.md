# Verify_Task3_AuxiliaryRegularizationPortfolio_58.1

## Purpose

This is a training-free development selector frozen before 57.1 produces any
performance metric. For each physical family it compares the independently
audited 56.1 portfolio winner with the independently selected and audited 57.1
auxiliary-projection weight-decay winner.

## Frozen rules

- Selection uses development populations only; confirmation remains closed.
- Both sources must match their preregistered canonical config hashes and pass
  independent evidence audit before selection starts.
- A 57.1 family winner remains eligible only after its zero-tolerance absolute
  FMT F1 and Average Precision guard passes. The 56.1 source is accepted only
  after all 80 frozen artifacts pass its existing independent audit.
- Ranking is paired dataset-macro F1 gain, absolute FMT F1, Average Precision
  gain, absolute FMT Average Precision, positive-dataset count, worst-dataset
  gain, then worst-seed gain.
- The selector freezes exactly 40 models and 40 result rows for paired seeds
  40--41. A separate auditor rebuilds all seven family choices and verifies all
  80 frozen files before the 660 temporary 57.1 checkpoints can be deleted.

The targets are paired F1 gain `>= +0.207` and absolute FMT F1 `>= 0.893`.
Missing either target is retained as a negative result. Reaching them does not
open a new confirmation population.

## Status

Preregistered. No 57.1 or 58.1 performance artifact has been read.
