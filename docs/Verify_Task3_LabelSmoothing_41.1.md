# Verify_Task3_LabelSmoothing_41.1

## Question

Can binary label smoothing improve absolute FMT vortex classification while
increasing its paired advantage over the same-width train-only Raw-PCA arm?

For training only, smoothing epsilon maps a hard label `y` to
`(1-epsilon)y + epsilon/2`. Validation and evaluation retain the original hard
IVD labels. This regularizes overconfident residual logits without changing
features, capacity, label generation, validation thresholds, or fusion-alpha
selection.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: exact hard-label control and epsilon
  `{0.001,0.0025,0.005,0.01,0.02,0.05,0.10,0.20}`.
- Within every candidate, FMT and train-only Raw-PCA use the same epsilon,
  initialization, batches, class weights, optimizer, budget, split, and head.
- Epsilon zero follows the previous weighted binary-cross-entropy computation
  exactly.
- Confirmation remains closed.

Nine candidates across ten datasets produce 90 array mappings and 540 paired
trainings. Smoothing is recorded in histories, checkpoints, and per-run output.

This search was declared after the complete 34.1 selector and without reading
partial metrics from 35.1--40.1.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the exact hard-label control. Eligible
candidates are ordered by paired F1 gain, then absolute FMT F1 and the
registered robustness tie-breakers.

The joint development target remains F1 gain at least `0.195` and absolute FMT
F1 at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Local Python compilation and 55 relevant unit tests passed. Full local
preflight confirmed 10 datasets, 9 candidates, 90 array mappings, 540 paired
trainings, all parameter budgets, and closed confirmation state. The local
preflight manifest SHA-256 is `fdbd889b...6c83`.

Implementation commit `352497c` was pushed before deployment. The immutable
archive SHA-256 is `ea93e2de...105c`; local raw and remote canonical config
SHA-256 values are `caea707d...6e7b` and `b60da2ac...646d`, respectively.
Remote Python compilation, the same 55 tests, and all three `bash -n` checks
passed.

Submitted at `2026-08-30T10:29:35+03:00`: CPU preflight job `51016609`, GPU
array `51016612[0-89%24]`, and selector `51016613`. The array has strict
`afterok:51016609`; the selector has strict `afterok:51016612_*`. Preflight
ran on `cn604-04` from 10:29:38 to 10:30:52, exited zero with empty stderr,
and produced remote manifest SHA-256 `b22f0a28...9f41`. It confirmed 10
datasets, 9 candidates, 90 mappings, 540 paired trainings, all capacity
guards, and closed confirmation state.

The GPU array ran from `19:05:21` to `21:25:22+03:00`. All 90 children exited
zero, all 540 paired trainings produced a per-run CSV, and the 90 GPU stderr
files total zero bytes. The actual device distribution was 4 A100-SXM4-80GB,
44 GTX 1080 Ti, 4 RTX 2080 Ti, 21 P100-PCIE-16GB, and
17 V100-SXM2-32GB. Selector `51016613` ran on `cn604-15` from `21:25:23` to
`21:25:30`, exited zero, and had empty stderr.

## Completed development result

Only Channel selected nonzero smoothing, with epsilon `0.001`. The
half-cylinder, Tangaroa, DeltaWing, F22, Boeing, and Smoke families all
retained the exact hard-label control.

The selected dataset-macro metrics are:

- Raw-PCA F1: `0.6978320228`;
- FMT F1: `0.8872801689`;
- paired F1 gain: `+0.1894481461`;
- Average Precision gain: `+0.2061758805`;
- positive datasets: `10/10`;
- worst dataset F1 gain: `+0.0472160712` on DeltaWing-resampled.

The all-control reference is Raw-PCA/FMT F1
`0.6977764373/0.8871118008`, paired F1 gain `+0.1893353635`, and Average
Precision gain `+0.2059043255`. Selection therefore changed paired F1 gain by
only `+0.0001127826` and absolute FMT F1 by `+0.0001683681`.

The experiment missed both registered targets (`0.195` paired F1 gain and
`0.893` absolute FMT F1) and is weaker than the completed Dropout search
(`+0.1964790`). Label smoothing does not open an independent confirmation.
Its selector remains a frozen input to the pre-registered combination search.

## Independent audit and retention

The stable artifact SHA-256 values are:

- selection: `db7f3e865ca9cd063c35bc2d3796f7368a47df4f4720f77500295192953f2e07`;
- leaderboard: `49f2f20e0a5f8f8792ac5710e5abadc3c89c0ad176ebc5c4fafc9f4002f7b34b`;
- preflight manifest: `b22f0a28fba6e691a32d7f866ec6fde4c32e8f88981173329b4d2d7fc3d39f41`;
- 540-file per-run archive: `891aa774d61f060cdb2a005351e10b251cd76e6876070e3c475bc59e5db4d339`.

`Audit_Task3_ParameterSearch.py` independently reconstructed the full
candidate/dataset/seed/arm Cartesian product, absolute-FMT guards,
family-specific selection, exact control, and macro metrics from the archive.
The maximum discrepancy from the selector was `2.22e-16`; the audit SHA-256
is `ebfb529920da5e407cdf63618ab68eb518c5925bca0b16f4c396feb738eabe0b`.

Evidence archive/cleanup job `51057902` ran on `cn604-15` from `21:25:31` to
`21:26:10`, exited zero, and had empty stderr. It archived all 540 per-run CSV
files, deleted 540 temporary checkpoints, and verified that zero checkpoint
files remain. Its committed script SHA-256 is `67d45e7d...b040e`, identical
locally and remotely. Confirmation remains closed.
