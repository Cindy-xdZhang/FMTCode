# Verify_Task3_TrainingResidualScale_32.1

## Question

Can the multiplier applied to the trainable residual logits during training
improve absolute FMT classification while increasing the paired gain over the
same-width train-only Raw-PCA control?

The training logits are `raw_logits + training_alpha * residual_logits`.
Therefore `training_alpha` directly scales the classification gradient entering
the residual branch. It does not freeze or replace the independently selected
validation/inference fusion alpha. This is an optimization-conditioning test,
not a new classifier or a post-hoc score rescaling.

## Why this is not a duplicate

`Verify_Task3_LossOptimization_7.1` was a broad one-factor screen and included
only training alpha 2 and 3 around the implicit alpha-1 control. Smoke selected
alpha 2. The present experiment systematically covers
`{0.125,0.25,0.5,0.75,1,1.5,2,3,4}` using the stronger completed 22.1
family-specific anchored feature. It was specified without reading any partial
or final performance from 27.1--31.2.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Within every candidate, FMT and train-only Raw-PCA use the same training
  alpha, initialization, batches, optimizer, budget, split, and head.
- Confirmation remains closed.

Nine candidates across ten datasets produce 90 array mappings and 540 paired
trainings. The alpha-1 candidate is the exact behavioral control.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to alpha 1. Eligible candidates are ordered
by paired F1 gain, then absolute FMT F1 and the registered robustness
tie-breakers.

The joint development target is F1 gain at least `0.195` and absolute FMT F1
at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Implementation commit `eb00dc6` was pushed before deployment. The immutable
archive SHA-256 is `7f8658a1...2270`; local raw and remote canonical config
SHA-256 values are `3da738b6...5525` and `24f4c507...049f`, respectively (the
difference is Windows CRLF versus Git-archive LF line endings). Local and
remote runs passed the eight relevant unit tests; remote Python compilation
and all three `bash -n` checks also passed.

Submitted on 2026-08-30: preflight job `51003518`, GPU array
`51003520[0-89%24]`, and selector `51003522`. The remote full preflight ran on
`cn511-17` from 02:31:07 to 02:32:14, exited zero with empty stderr, and
confirmed 10 datasets, 9 candidates, 540 paired trainings, and closed
confirmation state. The remote manifest SHA-256 is `b8e560e3...2897`.

## Completed development result

All 90 GPU children completed successfully between `06:51:51` and
`08:16:46+03:00`; all stderr files are empty. The selector completed at
`08:18:18+03:00`. Actual devices were 52 P100 16 GB, 37 GTX 1080 Ti, and one
RTX 2080 Ti job. The selection and leaderboard SHA-256 values are
`07d46203...0248` and `9aa46d2e...b4e`.

The selected development dataset-macro results are:

- Raw-PCA F1/AP: `0.690615 / 0.733421`;
- FMT F1/AP: `0.889486 / 0.947132`;
- paired F1/AP gain: `+0.198870 / +0.213712`.

Selected training alpha is `0.5` for Channel, `0.25` for Tangaroa, `0.75`
for F22 and Boeing, `2.0` for the half-cylinder family, and the exact alpha-1
control for Delta Wing and Smoke. This is the first focused search to exceed
the preregistered `0.195` relative-gain target, but absolute FMT F1 remains
`0.003514` below the `0.893` target. The joint target therefore failed and
confirmation remains closed; the larger difference is not reported as a
standalone success.
