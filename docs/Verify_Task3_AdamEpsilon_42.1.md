# Verify_Task3_AdamEpsilon_42.1

## Question

Can the AdamW denominator epsilon improve absolute FMT vortex classification
while increasing its paired advantage over the same-width train-only Raw-PCA
arm?

AdamW divides the first-moment estimate by the square root of the second moment
plus epsilon. Epsilon therefore limits effective steps when gradients and
second moments are small. This can matter for low-dimensional residual inputs
without changing features, labels, loss, model capacity, or parameter count.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: exact PyTorch-default control and epsilon
  `{1e-12,1e-10,1e-9,1e-7,1e-6,1e-5,1e-4}`.
- Within every candidate, FMT and train-only Raw-PCA use the same epsilon,
  initialization, batches, AdamW settings, budget, split, and head.
- The control does not pass an epsilon override and therefore retains the
  PyTorch default `1e-8` code path.
- Confirmation remains closed.

Eight candidates across ten datasets produce 80 array mappings and 480 paired
trainings. Epsilon is recorded in checkpoints and per-run output.

This search was declared without reading partial metrics from 35.1--41.1.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the exact default-epsilon control.
Eligible candidates are ordered by paired F1 gain, then absolute FMT F1 and
the registered robustness tie-breakers.

The joint development target remains F1 gain at least `0.195` and absolute FMT
F1 at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Local Python compilation and 61 relevant unit tests passed. Full local
preflight confirmed 10 datasets, 8 candidates, 80 array mappings, 480 paired
trainings, all parameter budgets, and closed confirmation state. The local
preflight manifest SHA-256 is `ce17319b...f3e7`.

Implementation commit `03594f7` was pushed before deployment. The immutable
archive SHA-256 is `4c17851e...72d8`; local raw and remote canonical config
SHA-256 values are `9a48c35e...c14d` and `0f851490...773c`, respectively.
Remote Python compilation, the same 61 tests, and all three `bash -n` checks
passed.

Submitted at `2026-08-30T10:39:02+03:00`: CPU preflight job `51016835`, GPU
array `51016836[0-79%24]`, and selector `51016837`. The array had strict
`afterok:51016835`; the selector had strict `afterok:51016836_*`. Preflight
ran on `cn604-07` from 10:39:03 to 10:40:10, exited zero with empty stderr,
and produced remote manifest SHA-256 `e63b1141...8b45`. It confirmed 10
datasets, 8 candidates, 80 mappings, 480 paired trainings, all capacity
guards, and closed confirmation state.

The GPU array completed at `2026-08-30T21:58:47+03:00`. All 80 children
exited zero, all 480 paired trainings produced a per-run CSV, and the GPU
stderr files total zero bytes. The actual device distribution was 7
A100-SXM4-80GB, 37 GTX 1080 Ti, 3 RTX 2080 Ti, 15 P100-PCIE-16GB, and 18
V100-SXM2-32GB. Selector `51016837` ran on `cn113-35-l` from 22:18:10 to
22:18:21, exited zero, and had empty stderr.

## Completed development result

The selected epsilon is family-specific: Boeing and Smoke retained the exact
default-epsilon control; Channel and F22 selected `1e-4`; DeltaWing and
Tangaroa selected `1e-10`; and half-cylinder selected `1e-7`.

The selected dataset-macro metrics are:

- Raw-PCA F1: `0.6963002440`;
- FMT F1: `0.8872018792`;
- paired F1 gain: `+0.1909016352`;
- Average Precision gain: `+0.2067538460`;
- positive datasets: `10/10`;
- worst dataset F1 gain: `+0.0529384509`.

The exact default-epsilon control is Raw-PCA/FMT F1
`0.6972389364/0.8870755155`, paired F1 gain `+0.1898365791`, and Average
Precision gain `+0.2057573052`. Selection therefore changed paired F1 gain by
`+0.0010650561` and absolute FMT F1 by only `+0.0001263637`.

The experiment missed both registered targets (`0.195` paired F1 gain and
`0.893` absolute FMT F1) and is weaker than the completed Dropout search
(`+0.1964790`). Adam epsilon does not open an independent confirmation. Its
selector remains a frozen input to the pre-registered combination searches.

## Independent audit and retention

The stable artifact SHA-256 values are:

- selection: `c2d7d2f468b0281c043c8a100479db79067646bcd3077a38c90c4c910fa3ecbc`;
- leaderboard: `10b9af895448e648492a4ea4a3557d9c83589c1d52767550f5452280373bff54`;
- preflight manifest: `e63b1141f8408e2735d84f825f422aeda59a2d9d19366894d370fbefaafa8b45`;
- 480-file per-run archive: `73d563ef49be8851668534d5971982e55d1a0e7711d1388e74d8f3179a7d232f`;
- artifact manifest: `0a7a73c2274fe7bf25e1607782445e1523e7bbe23a86f3d8edacefc66162f305`;
- checkpoint-cleanup record: `c88178feeccdb68ec89c4f01cbba03c2d1ee20a27006e9d7d3f487da804e3eef`.

`Audit_Task3_ParameterSearch.py` independently reconstructed the complete
candidate/dataset/seed/arm Cartesian product, family-specific selection,
absolute-FMT guards, exact control, and macro metrics from all 480 CSV files.
The maximum discrepancy from the selector was `2.22e-16`; the audit SHA-256
is `dac2f527dfea957c2a3005c2a56b18832af6a0f56f9b093e8d057df8f51adbd5`.

Evidence archive/cleanup job `51057984` ran on `cn113-35-l` from 22:18:22 to
22:19:04, exited zero, and had empty stderr. It archived all 480 per-run CSV
files, deleted 480 temporary checkpoints, and verified that zero checkpoint
files remain. Its committed script SHA-256 is `97c3f337...a46f74`, identical
locally and remotely. Confirmation remains closed.
