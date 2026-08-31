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

## Deployment

- Implementation commit: `3fe1aee`.
- Immutable archive SHA-256: `32bb4313d8bc4af394caeadd4d4a8ba3c69aca90386bfaa6a037ceb19d71a9de`.
- Preflight: job `51020702`.
- GPU array: job `51020733[0-79%24]`, strict `afterok:51020702`.
- Selector: job `51020778`, strict `afterok:51020733_*`.
- Evidence archive: job `51071279`, strict `afterok:51020778`. It will
  archive and hash all 480 per-run CSV files while requiring all 480 temporary
  checkpoints to remain present for 52.1; it never deletes a model. The local
  and remote script SHA-256 is
  `da5d84d733ae54f506c170f4bcc4cc95f0b93ce876008ab22027893968780b39`.
- Local and remote 45.1/44.1/19.1 combination contracts: 10/10 passed.
  Remote Python compilation and all three Slurm-script syntax checks passed.
- Preflight completed at `2026-08-30T11:43:58+03:00` with exit code 0 and
  empty stderr: 10 datasets, 8 candidates, 80 mappings, 480 paired trainings,
  and zero ineligible recipes. Manifest SHA-256:
  `169bf1ede723953b60b829ca15d3d534ab1df68974e12143b32120494a460b57`.
