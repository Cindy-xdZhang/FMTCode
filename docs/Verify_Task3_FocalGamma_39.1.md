# Verify_Task3_FocalGamma_39.1

## Question

Can focal loss improve absolute FMT vortex classification while increasing
its paired F1 advantage over the same-width train-only Raw-PCA arm?

Focal loss multiplies weighted binary cross entropy by a confidence-dependent
factor. Gamma zero is equivalent to weighted binary cross entropy; increasing
gamma progressively emphasizes currently difficult observations.

## Evidence motivating this focused search

The completed broad 7.1 development search selected focal gamma 3 for Channel,
gamma 1 for the half-cylinder family, and gamma 2 for Boeing. That experiment
used the older 5.2 FMT representations and tested only gamma
`{0.5,1,2,3}`. The present experiment retests this specific signal on the
stronger completed 22.1 family-specific FMT representation, adds low-gamma
resolution, and closes the upper range through gamma 5. It is independent of
all unfinished 33.1--38.1 searches.

## Frozen comparison

- Development populations, labels, split, frozen Raw checkpoints, optimizer,
  learning rate, batch size, epoch budget, early stopping, and fusion search
  follow the completed 5.2 protocol.
- The completed 22.1 selector supplies one anchored FMT feature per physical
  family.
- Both arms use the same two-hidden-layer width-64 residual multilayer
  perceptron with LayerNorm, GELU, and zero dropout.
- Paired seeds are `40, 41, 42`.
- Within each candidate, FMT and train-only Raw-PCA receive the same loss,
  gamma, labels, ordering seed, initialization, network, and training budget.
- Confirmation data remain closed.

## Search grid and exact control

The registered focal gamma values are
`{0.10,0.25,0.50,0.75,1.00,1.50,2.00,2.50,3.00,4.00,5.00}`.
The additional control has no loss override and therefore executes the exact
historical weighted-binary-cross-entropy path.

Twelve candidates across ten datasets produce 120 array mappings and 720
paired trainings.

## Selection and targets

Selection is physical-family specific on exposed development data. A candidate
is eligible only if absolute FMT F1 and FMT Average Precision are both no lower
than the exact weighted-binary-cross-entropy control. Eligible candidates are
ranked by paired F1 gain, then absolute FMT F1 and the registered robustness
tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.195`; and
- absolute dataset-macro FMT F1 at least `0.893`.

Failure of either target remains a negative result. Any development winner
still requires evaluation on a fresh spatial population before a paper-level
claim.

## Main files

- `config/Verify_Task3_FocalGamma_39.1.yaml`
- `Verify_Task3_FMTResidual.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_focal_gamma_39_1.py`
- `ibex_bash/verify_task3_focal_gamma_39.1_*.sh`

## Ibex deployment

Implementation commit `9e824d3` was pushed before deployment. The immutable
archive SHA-256 is `105853f7...4991`; the remote canonical config SHA-256 is
`a7487120...d041`. Local and Ibex environments both passed the 10 relevant
tests; remote Python compilation and all three `bash -n` checks also passed.
The local full-preflight manifest SHA-256 is `cb20b0aa...4111`.

Submitted at `2026-08-30T08:42:12+03:00`: CPU preflight job `51012669`, GPU
array `51012681[0-119%24]`, and selector `51012699`. The GPU array has
`afterok:51012669`; the selector has `afterok:51012681_*`. The preflight ran
on `cn604-13` from `08:42:13` to `08:43:23+03:00`, exited zero with empty
stderr, and produced remote manifest SHA-256 `e33cd17c...f702`.

The GPU array ran from `18:01:55` to `20:39:47+03:00`. All 120 children
completed with exit code zero, all 720 paired trainings produced a
`per_run.csv`, and aggregate GPU stderr was zero bytes. The actual devices
were 19 A100-SXM4-80GB, 53 GTX 1080 Ti, 6 RTX 2080 Ti,
23 P100-PCIE-16GB, and 19 V100-SXM2-32GB. Selector `51012699` then ran on
`cn604-14` from `20:39:49` to `20:39:57`, exited zero, and had empty stderr.

## Completed development result

Family-specific selection gave:

- Channel: focal gamma `0.10`;
- half-cylinder: focal gamma `0.10`;
- Tangaroa, DeltaWing, F22, Boeing, and Smoke: exact weighted-binary-cross-
  entropy control.

The selected dataset-macro result is:

- Raw-PCA F1: `0.6973445419`;
- FMT F1: `0.8875032322`;
- paired F1 gain: `+0.1901586904`;
- Average Precision gain: `+0.2065562721`;
- positive datasets: `10/10`;
- worst dataset F1 gain: `+0.0527172582` on DeltaWing-resampled.

The all-control reference is Raw-PCA/FMT F1
`0.6972769385/0.8871154510`, paired F1 gain `+0.1898385125`, and Average
Precision gain `+0.2057710602`. Selection therefore improved paired F1 gain
by only `+0.0003201779` and absolute FMT F1 by `+0.0003877812`.

The experiment missed both registered targets: paired F1 gain is below
`0.195`, and absolute FMT F1 is below `0.893`. It is also weaker than the
completed Dropout search (`+0.1964790` F1 gain). Focal gamma therefore does
not open an independent confirmation; it remains only a pre-registered input
to the downstream combination searches.

## Independent audit and retention

The selector artifacts have SHA-256 values:

- selection: `1e99c4328b7271cf4526333c5e38a9415911208219b3cea4da0692a86f59e994`;
- leaderboard: `b76870ab5d882fa6c80f1f6dbd29b9e264d013e7bdac9d085bdadff82c6a6ddc`;
- preflight manifest: `e33cd17cecbad2bf2bdd70a1135a7322c054e5688934a7ed3e4a3a74a8d0f702`;
- 720-file per-run archive: `58a100061cb6cce4b7c078af4ea34a9ec31125f999d9d55f41ee1f169e6a6af9`.

`Audit_Task3_ParameterSearch.py` independently reconstructed the full
candidate/dataset/seed/arm Cartesian product, eligibility guards,
family-specific selection, exact control, and dataset-macro metrics from the
720 per-run files. The maximum discrepancy from the selector was
`1.11e-16`; the local audit SHA-256 is
`bff22e749cf8eab262eb3f736fd8b92da6e051408ba2c3667274278110ec48c3`.

Archive-and-cleanup job `51053640` completed on `cn604-14` from `20:39:59`
to `20:40:11`, with empty stderr. It archived all 720 per-run CSV files,
deleted 720 temporary checkpoints, and verified that zero checkpoint files
remain. Confirmation remains closed.

The archive job initially wrote `2b5a4482...2115e` for the tar archive, but
the archive modification time subsequently advanced by about 59 seconds and
its stable hash became `58a10006...6a6af`. No second Slurm job touched this
experiment, so the filesystem-level cause is not proven. The initial manifest
is retained as `artifact_sha256.initial.txt`; after repeated stability checks,
the current remote archive matched the downloaded bytes, all 720 CSVs passed
the independent audit, and `artifact_sha256.txt` was regenerated with the
stable hash. This correction changes no metric, guard, or family selection.
