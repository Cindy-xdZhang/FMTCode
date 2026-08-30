# Verify_Task3_ParameterEMA_40.1

## Question

Can an exponential moving average (EMA) of trainable residual-head parameters
improve absolute FMT classification while increasing its paired advantage over
the same-width train-only Raw-PCA arm?

EMA keeps a non-trainable running average of each trainable residual parameter
after every optimizer step. Validation, early stopping, checkpoint selection,
and inference use the averaged parameters; optimizer updates continue from the
unsmoothed live parameters. The frozen Raw backbone is never averaged.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: exact no-EMA control and decay
  `{0.5,0.8,0.9,0.95,0.98,0.99,0.995,0.999}`.
- Within every candidate, FMT and train-only Raw-PCA use the same decay,
  initialization, batches, optimizer, epoch budget, split, and head.
- The no-EMA control does not construct, update, swap, or checkpoint averaged
  parameters and is therefore a strict code-level no-op.
- Confirmation remains closed.

Nine candidates across ten datasets produce 90 array mappings and 540 paired
trainings. The decay is recorded in histories, checkpoints, and per-run output.

## Why this search follows 33.1

The completed 33.1 logs contain 480 trainings, of which only 21 (4.4%) reached
epoch 100. The earlier 7.1 160-epoch candidate was not selected for any
physical family. A larger epoch budget therefore has limited support. EMA
instead tests whether noisy late residual-head updates obscure a better FMT
solution without changing features, labels, capacity, or the paired baseline.

This search was declared after the complete 33.1 selector result and without
reading partial metrics from 34.1--39.1.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the exact no-EMA control. Eligible
candidates are ordered by paired F1 gain, then absolute FMT F1 and the
registered robustness tie-breakers.

The joint development target remains F1 gain at least `0.195` and absolute FMT
F1 at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Local Python compilation and 48 relevant unit tests passed. Full local
preflight confirmed 10 datasets, 9 candidates, 90 array mappings, 540 paired
trainings, all parameter budgets, and closed confirmation state. The local
preflight manifest SHA-256 is `edbbe882...a8f9`.

Implementation commit `749b74a` was pushed before deployment. The immutable
archive SHA-256 is `93b0aabf...c3ce`; local raw and remote canonical config
SHA-256 values are `7e565a03...5152` and `96e25b06...7ceb`, respectively.
Remote Python compilation, the same 48 tests, and all three `bash -n` checks
passed.

Submitted at `2026-08-30T10:15:01+03:00`: CPU preflight job `51016376`, GPU
array `51016379[0-89%24]`, and selector `51016380`. The array had strict
`afterok:51016376`; the selector had strict `afterok:51016379_*`. Preflight
ran on `cn511-09` from 10:15:03 to 10:16:16, exited zero with empty stderr,
and produced remote manifest SHA-256 `fd3bbc3f...ff06`. It confirmed 10
datasets, 9 candidates, 90 mappings, 540 paired trainings, all capacity
guards, and closed confirmation state.

## Completed result

All 90 GPU children and all 540 paired trainings completed successfully by
19:32:23; total GPU stderr was zero bytes. The array used 4 A100-SXM4-80GB,
2 GTX 1080 Ti, 3 P100-PCIE-16GB, and 81 V100-SXM2-32GB allocations. Selector
`51016380` then completed in 7 seconds with empty stderr.

The family-specific choices are decay 0.99 for channel, 0.80 for
halfcylinder and Tangaroa, 0.95 for deltaWing, and exact no-EMA controls for
Boeing747, F22, and SmokeBuoyancy. Selected dataset-macro results are:

- Raw-PCA F1: `0.69734194`
- FMT F1: `0.88742737`
- paired F1 gain: `+0.19008543`
- Raw-PCA Average Precision: `0.74035309`
- FMT Average Precision: `0.94608995`
- paired Average Precision gain: `+0.20573686`

All ten dataset-level F1 gains are positive. Relative to the exact no-EMA
control (`0.69781588 -> 0.88707552`, gain `+0.18925964`), selection raises
the FMT F1 by only `+0.00035186` and the paired gain by `+0.00082580`.
Consequently, it misses both preregistered targets: F1 gain `>=0.195` and
absolute FMT F1 `>=0.893`. It is also weaker than the completed dropout search
38.1 (`+0.19648` F1 gain), so EMA does not independently open confirmation;
its guarded family winners remain valid inputs to preregistered combination
searches 44.1 and 48.1.

The downloaded selection, leaderboard, and per-run archive SHA-256 values are
`f5d4952f...1c1d`, `4c8f3c94...abe5`, and `0fc2bb50...f243`. An independent
audit reconstructed the complete 9 x 10 x 3 x 2 Cartesian product, paired
parameter counts, source hashes, absolute-FMT guards, family ranking, and all
macros directly from the 540 CSV files without importing the selector. Its
maximum discrepancy was `5.6e-17`; audit SHA-256 is `0fd76781...e29`.
Temporary `.pt` files were verified absent after archival; 540 empty
checkpoint directories remain and are not experiment evidence.
