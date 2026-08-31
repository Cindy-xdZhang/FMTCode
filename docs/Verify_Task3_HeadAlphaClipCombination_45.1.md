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

## Completed development result

All 80 array children and all 480 paired trainings completed successfully. The
selector ran at `2026-08-31T07:25:12+03:00` and finished with empty stderr.

- Raw-PCA/FMT dataset-macro F1: `0.68794798 / 0.88973008`.
- Paired F1 gain: `+0.20178210`.
- Raw-PCA/FMT dataset-macro Average Precision: `0.73148483 / 0.94815251`.
- Paired Average Precision gain: `+0.21666768`.
- Positive datasets: `10/10`; worst per-dataset F1 gain: `+0.05271726`.
- Exact-control Raw-PCA/FMT F1: `0.69723722 / 0.88706248`; gain:
  `+0.18982525`.

The gain target (`>= +0.195`) passed. The absolute FMT target (`>= 0.893`)
missed by `0.00326992`, so the preregistered joint target did not pass. This
experiment is therefore a valid development ablation, not a replacement for
44.1 and not a paper-level confirmation.

Selected recipe by physical family:

| Family | Recipe |
|---|---|
| Boeing 747 | `k02_alpha` |
| Channel | `k03_clipping` |
| Delta wing | `k00_feature_control` |
| F-22 | `k07_head_alpha_clipping` |
| Half-cylinder | `k07_head_alpha_clipping` |
| Smoke buoyancy | `k03_clipping` |
| Tangaroa | `k04_head_alpha` |

## Evidence and independent audit

Evidence job `51071279` completed at `2026-08-31T07:26:01+03:00` with exit
code 0 and empty stderr. It archived all 480 per-run CSV files while retaining
all 480 temporary checkpoints for 52.1.

- Selection SHA-256: `cf80d0d62563da50a5b76b0b909079aa51af42fa3cf8b95bcd4fac4c9c1d43d9`.
- Leaderboard SHA-256: `19606af4ef28316a2bdde416f238d2afec07b800c761cc6165a8ee75baf4afa8`.
- Per-run archive SHA-256: `08e3788f91a338413b403d67080f3ea4d8ecb53d37713f7c4a229516fda51bcd`.
- Independent-audit SHA-256: `58947d24cd306146b7f836fc45705f0f564aed768262ea774a013c9fd54280b2`.

`Audit_Task3_ParameterSearch.py` independently reconstructed all family guards,
candidate choices, and macro metrics from the 480 archived CSV files. It found
equal paired parameter counts, consistent source hashes, and a maximum absolute
difference of exactly `0.0` versus the selector. `confirmation_opened=false`.
