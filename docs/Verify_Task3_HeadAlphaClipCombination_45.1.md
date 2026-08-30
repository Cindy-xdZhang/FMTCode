# Verify_Task3_HeadAlphaClipCombination_45.1

## Question

Do the completed family-specific winners from residual-head capacity (31.2),
training residual scale (32.1), and gradient clipping (33.1) combine to improve
both paired F1 gain and absolute FMT F1?

## Evidence boundary

This adaptive development experiment was declared after all three source
selectors completed. It fills a specific omission in 44.1: 44.1 combines
training and optimizer/loss factors but does not include the 31.2 head winner.
No 36.1--44.1 partial metric was read when this factorial was frozen.
Confirmation remains closed.

## Frozen comparison

- Population and split: completed 5.2 development protocol.
- Feature: completed 22.1 family-specific anchored FMT representation.
- Paired seeds: 40, 41, 42.
- Exact control head: two hidden layers, width 64, LayerNorm, GELU.
- The FMT and train-only Raw-PCA arms use the same merged head, training alpha,
  clipping threshold, initialization, optimizer, batches, split, and budget.

## Candidate grid and selection

The complete binary factorial over `head`, `alpha`, and `clipping` has eight
candidates. Across ten datasets this is 80 array mappings and 480 paired
trainings. Preflight freezes all source selector SHA-256 values and rejects
hidden recipe conflicts.

Every candidate must preserve each family's FMT F1 and Average Precision
relative to the exact anchored-feature control with zero tolerance. The joint
development target remains F1 gain `>= +0.195` and absolute FMT F1 `>= 0.893`.
Any winner still requires a fresh spatial-population confirmation.
