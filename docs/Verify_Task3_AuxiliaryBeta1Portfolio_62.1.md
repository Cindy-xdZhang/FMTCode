# Verify_Task3_AuxiliaryBeta1Portfolio_62.1

## Purpose

This training-free development selector is frozen before 61.1 produces any
performance metric. It compares the independently audited 60.1 portfolio with
the independently selected and audited 61.1 auxiliary-`beta1` winner for each
physical family.

## Frozen rules

- Both sources must match their canonical config hashes and pass independent
  audits; confirmation remains closed.
- A 61.1 winner must pass its zero-tolerance FMT F1 and Average Precision guard.
- Ranking uses paired dataset-macro F1 gain, absolute FMT F1, Average Precision
  gain, absolute FMT Average Precision, positive-dataset count, worst-dataset
  gain, and worst-seed gain.
- Exactly 40 seed40--41 models and 40 result rows are frozen. An independent
  auditor rebuilds all choices and verifies all 80 files before 61.1 temporary
  checkpoints may be deleted.

The targets are F1 gain `>= +0.209` and absolute FMT F1 `>= 0.893`. Failure is
retained, and success does not automatically open another confirmation set.

## Status

Preregistered. No 61.1 or 62.1 performance artifact has been read.
