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

Completed and independently audited on Ibex.

- Selector `51091889` and independent audit `51091892` completed with exit
  code zero and empty stderr.
- All seven physical families retain the 54.1 `current_portfolio`; no 55.1
  auxiliary-learning-rate winner survives the registered comparison.
- The resulting development macro remains Raw-PCA/FMT F1
  `0.684701/0.890218`, gain `+0.205517`, with Average Precision gain
  `+0.220142`.
- Exactly 40 result files and 40 paired checkpoints are frozen. The independent
  auditor verifies every content hash and paired parameter count and has zero
  metric difference from the selector.
- Selection and audit SHA-256 values are
  `e9acac4133fe01327be466fc76aa69c218f9df1ddec75b2df10711a98174c40d`
  and `d1eb81649572a77f69849c8d9983e97235ac29824162382a14448b7df41643e9`.

The exact F1 gain is slightly below the preregistered rounded threshold
`0.20552`, and absolute FMT F1 remains below `0.893`; therefore both the
absolute and joint targets are false. Confirmation remains closed.
