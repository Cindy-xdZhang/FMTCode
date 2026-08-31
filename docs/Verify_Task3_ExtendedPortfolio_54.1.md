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

## Deployment status

Implementation commit `71313b0` and immutable deployment archive SHA-256
`6f3fdc02d9762460991a600a5057270a533dd21e700d57b843f32d193ec65f13`
are deployed at `/home/zhanx0o/FMT_Task3_ExtendedPortfolio_54_1`. Seven tests,
Python compilation, all shell syntax checks, and static preflight pass.
Source-identity job `51074578` completed with exit code zero and report SHA-256
`890fac3110e21a46b32f4aae57c2a41f57b399f2b20d1b1f8f7d3a6ad88f38e9`;
it records `performance_artifacts_read=false`. Selector `51074606` waits for
the completed 53.1 selector and evidence job, and independent audit `51074616`
waits for 54.1. No 54.1 performance result exists yet.
