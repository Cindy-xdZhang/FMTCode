# Verify_Task3_FinalPortfolio_49.1

## Question

Which already guarded family winner from 44.1, 45.1, or 48.1 gives the
largest development-set FMT-over-Raw-PCA gain before a fresh confirmation?

## Frozen protocol

- No training and no confirmation access.
- Sources are exactly `Verify_Task3_SafeFactorCombination_44.1`,
  `Verify_Task3_HeadAlphaClipCombination_45.1`, and
  `Verify_Task3_HeadFullStackCombination_48.1`.
- Per physical family, first require the source-selected row to have passed
  its preregistered absolute FMT F1 and Average Precision guard; then maximize
  paired dataset-macro F1 gain with the fixed tie breakers in the config.
- Freeze the winner's seed-40/41 result and checkpoint SHA-256 values.  Seed 42
  remains part of development selection but is not used in final confirmation.
- Target development gain is `+0.20`; target absolute FMT F1 is `.893`.

The selector may run only after all three source selectors finish.  Its output
is `outputs/Verify_Task3_FinalPortfolio_49.1/portfolio_selection.json`, which
is the sole model source for `mainExp_Task3_3D_7.1`.

## Completed development result

Job `51043495` ran only after all three source selectors completed and exited
successfully at `2026-08-31T07:25:29+03:00`; stderr was empty. It performed no
training and froze 40 seed-40/41 model/result/checkpoint identities.

- Raw-PCA/FMT dataset-macro F1: `0.68470116 / 0.89021797`.
- Paired F1 gain: `+0.20551681`.
- Raw-PCA/FMT dataset-macro Average Precision: `0.72857737 / 0.94871932`.
- Paired Average Precision gain: `+0.22014195`.
- Positive datasets: `10/10`.
- Portfolio SHA-256:
  `c20586d803d9dc2586d0958d6a8be3f1d0a5006bebfa8b9874129b6b389c498f`.

The `+0.20` gain target passed, while absolute FMT F1 remained below `.893`;
the joint target therefore failed. The selected sources were 44.1 for Channel,
Delta wing, F-22, and Half-cylinder; 48.1 for Boeing 747 and Smoke buoyancy;
and 45.1 for Tangaroa.

This is a completed development result, not confirmation evidence. The older
7.1 entry remains held. The broader five-source 52.1 portfolio, which also
includes 50.1 and 51.1, supersedes 49.1 as the input to the planned fresh
spatial-population confirmation 7.2.
