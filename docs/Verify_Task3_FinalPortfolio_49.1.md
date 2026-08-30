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
