# Verify_Task3_ExtendedPortfolio_54.1

## Purpose

This training-free development stage was frozen before 53.1 produced any
performance metric. It extends the 52.1 source set with the preregistered
auxiliary-representation Dropout search and selects one guarded recipe per
physical family across 44.1, 45.1, 48.1, 50.1, 51.1, and 53.1.

The ranking remains dataset-macro paired F1 gain, absolute FMT F1, Average
Precision gain, absolute FMT Average Precision, positive-dataset count,
worst-dataset gain, then worst-seed gain. Every source winner must have passed
its own zero-tolerance absolute-FMT F1 and Average Precision guard.

## Boundaries

- It trains no model and reads no confirmation data.
- Source seeds 40--42 participate in development selection; only paired seeds
  40--41 are copied for confirmation.
- Ten datasets form seven physical families. The selector freezes exactly 40
  models, 40 result rows, and all content hashes in its own output directory.
- A source-identity job first checks the six deployed config experiment names
  and normalized-text SHA-256 values without reading any performance artifact.
- `Audit_Task3_AdaptivePortfolio.py` independently reconstructs all family
  choices and macro metrics and verifies every copied result/checkpoint hash.
- A 54.1 result remains development evidence. Any selected recipe must be
  evaluated on a new spatial primitive population before becoming a paper
  result.

## Frozen targets

- Dataset-macro F1 gain target: `+0.20`.
- Absolute FMT F1 target: `0.893`.
- Confirmation remains closed regardless of whether either target is reached.

## Main artifacts

- `config/Verify_Task3_ExtendedPortfolio_54.1.yaml`
- `Select_Task3_ExtendedPortfolio_54_1.py`
- `Audit_Task3_AdaptivePortfolio.py`
- `tests/test_task3_extended_portfolio_54_1.py`
- `ibex_bash/verify_task3_extended_portfolio_54.1_*.sh`

## Final development result

Source gate `51074578`, selector `51074606`, and independent audit `51074616`
all completed with exit code zero and empty stderr. The portfolio selection
SHA-256 is `9dee26470e959b6c47d5c27140bd305c2682b6a9ee67d5c589358853a5c91428`.
The independent auditor rebuilt all family choices, verified 80 frozen files
covering 40 models and 40 results, and differed from the selector by at most
`2.22e-16`.

Development dataset-macro Raw-PCA/FMT F1 is `0.68470/0.89022`, giving
`+0.20552`; Average Precision is `0.72858/0.94872`, giving `+0.22014`. All ten
datasets have positive F1 gain. Boeing and Smoke use the 48.1 full-stack
source, Tangaroa uses the 45.1 head/alpha/clip source, and the remaining four
families use the 44.1 safe-factor source. No family selects 53.1 auxiliary
dropout.

The F1-gain target is reached, but the absolute FMT F1 target is not
(`0.89022 < 0.893`), so `joint_target_reached=false`. This remains development
evidence and cannot replace the independent 8.1 spatial confirmation.
