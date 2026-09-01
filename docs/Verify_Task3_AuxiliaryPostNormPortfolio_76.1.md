# Verify_Task3_AuxiliaryPostNormPortfolio_76.1

## Purpose

This training-free development selector is frozen before 75.1 produces any
performance metric.  For each physical family it compares the independently
audited 74.1 portfolio winner with the independently audited 75.1 auxiliary
post-normalization winner under the same registered metric order.

## Frozen rules

- Selection uses development populations only; confirmation remains closed.
- Both sources must pass independent audit and all source hashes must match.
- A 75.1 family winner is eligible only after its zero-tolerance absolute FMT
  F1 and Average Precision guard passes.
- Ranking is paired dataset-macro F1 gain, absolute FMT F1, Average Precision
  gain, absolute FMT Average Precision, positive-dataset count, worst-dataset
  gain, then worst-seed gain.
- The selector freezes exactly 40 models and 40 result rows for paired seeds
  40--41 before temporary 75.1 checkpoints are deleted.

The targets are F1 gain `>= +0.216` and absolute FMT F1 `>= 0.893`. Missing
either target is retained as a negative result and reaching both does not open
a new confirmation population.

## Status

Preregistered before 75.1 produced any performance artifact. No 75.1 or 76.1
performance result has been read.

Deployed on Ibex from commit `84ec1c3`. Source identity job `51139435`
completed with `performance_artifacts_read=false`.  Selector `51139436`,
independent audit `51139437`, and the downstream 75.1 cleanup `51139440` all
completed with exit code zero and empty stderr after their strict dependencies
were satisfied.

## Results

The portfolio selector retained the existing 74.1 source for all seven
physical families; no 75.1 post-normalization source entered the portfolio.
The resulting dataset-macro Raw-PCA/FMT F1 was `0.683459/0.890572`, F1 gain
was `+0.207113`, and Average Precision gain was `+0.223384`.  These are the
existing 74.1 portfolio metrics rather than a new improvement.  The portfolio
missed both registered targets (`+0.216` F1 gain and `0.893` absolute FMT F1),
so no confirmation population was opened.

## Evidence

- portfolio-selection, independent-audit, source-identity, and evidence-list
  SHA-256: `f8da77b7...65a4`, `a2fd86c0...31ad`,
  `4515ffca...d1c7`, and `6313fa6f...daf2`;
- the independent implementation reconstructed all seven choices with maximum
  numerical difference `0`, verified equal paired parameter counts, and
  checked all 40 frozen models plus 40 frozen result files by content hash;
- the source selector chose `current_portfolio` for Boeing 747, channel,
  DeltaWing, F22, halfcylinder, SmokeBuoyancy, and Tangaroa;
- model checkpoints were not downloaded locally.  Their identities were
  verified by the Ibex independent audit before the source checkpoints were
  removed under the retention protocol.
