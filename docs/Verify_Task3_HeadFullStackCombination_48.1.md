# Verify_Task3_HeadFullStackCombination_48.1

## Question

Can the selected residual-head capacity and common training horizon improve
the optimizer, loss, regularization, and stability stacks while preserving
absolute FMT F1 and Average Precision?

## Evidence boundary

This adaptive development experiment was declared after the complete 38.1
Dropout result, but before reading any 39.1--47.1 result. All candidate recipes
are fixed in the config before those selectors finish. A source that later
selects its exact control remains in every declared candidate that references
it; candidates are never removed after viewing results. Confirmation remains
closed.

## Frozen comparison

- Development population and split: completed Task3 5.2 protocol.
- Feature: completed 22.1 family-specific anchored FMT representation.
- Paired seeds: 40, 41, 42.
- Within every cell, FMT and train-only Raw-PCA use identical head, optimizer,
  loss, initialization, data order, split, and training budget.
- Every upstream selector must cover the same seven physical families and
  report `confirmation_opened=false`.
- Preflight hashes every source selector before any GPU child runs.

## Fixed grid

The sources comprise the selected head, training residual scale, gradient
clipping, AdamW betas, batch size, positive-class weight, Dropout, focal loss,
parameter exponential moving average, label smoothing, AdamW epsilon, cosine
learning-rate schedule, and training horizon. The 16 candidates contain the
exact feature control, head and horizon controls, focused head interactions,
optimizer/loss/stability stacks, and full stacks with and without alpha,
cosine, or horizon.

Sixteen candidates across ten datasets give 160 array children and 960 paired
trainings. Selection first requires FMT F1 and Average Precision to be no lower
than the exact feature control, then maximizes paired F1 gain using the frozen
tie-breakers.

The development targets are F1 gain `>=0.20` and absolute FMT F1 `>=0.893`.
Any winner remains development-only and must pass a fresh spatial-population
confirmation before replacing the current Task3 paper result.

## Main files

- `config/Verify_Task3_HeadFullStackCombination_48.1.yaml`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_head_full_stack_combination_48_1.py`
- `ibex_bash/verify_task3_head_full_stack_combination_48.1_*.sh`

## Ibex deployment

- Implementation commit: `eb2cd44`.
- Immutable archive SHA-256:
  `158adf0fd203d7b3e86346ae29c31e1eb5254bec4ad90750b44023686b562372`.
- Local and remote 48.1/44.1/45.1 contracts: 11/11 passed.
- Remote Python compilation, static preflight, and all three `bash -n` checks
  passed. Static preflight confirms 10 datasets, 7 physical families,
  16 candidates, 14 selector sources, 960 paired trainings, and closed
  confirmation.
- Preflight job `51041336` has strict `afterok` dependencies on pending
  selectors 39.1, 40.1, 41.1, 42.1, 43.1, and 47.1.
- GPU array `51041338[0-159%24]` has strict `afterok:51041336`.
- Selector `51041339` has strict `afterok:51041338_*`.
- Evidence archive job `51071147` has strict `afterok:51041339`. It archives
  and hashes all 960 per-run CSV files while requiring all 960 temporary
  checkpoints to remain present for the downstream 52.1 portfolio. It never
  deletes a model. The local and remote script SHA-256 is
  `244567198cb48197275dcf804664d23168ef12d53edc2717923b7770decd3812`.

The preflight, GPU array, and selector were submitted before any 39.1--47.1
final metric was read. The later evidence job cannot change candidates,
training, or selection and was added only to preserve auditable CSV evidence.

## Final development result

All 160 GPU children completed with exit code zero, producing all 960 declared
paired trainings. Selector `51041339` completed in 11 seconds with empty
stderr. The family-specific selected Raw-PCA/FMT dataset-macro F1 is
`0.6861681/0.8901764`, hence the paired F1 gain is `+0.2040084`. Raw-PCA/FMT
Average Precision is `0.7284376/0.9490412`, hence the Average Precision gain is
`+0.2206036`. All 10 dataset entries have positive F1 gain; the worst entry is
deltaWing-resampled at `+0.0538466`.

The exact feature control has Raw-PCA/FMT F1 `0.6974124/0.8870755`, giving
`+0.1896631`. Selection therefore raises absolute FMT F1 by `+0.0031009` and
paired F1 gain by `+0.0143453`, while Raw-PCA F1 falls by `0.0112443`. This is
not solely a weaker-Raw result because absolute FMT also improves. Relative to
the earlier 38.1 Dropout winner, absolute FMT F1 rises from `0.8883567` to
`0.8901764` and paired gain rises from `+0.1964790` to `+0.2040084`.

The preregistered `+0.20` gain target is reached. The absolute FMT target
`0.893` is missed by `0.0028236`, so the joint development target is not
reached and confirmation remains closed at this stage. The selected recipes
are:

| Physical family | Selected combination |
|---|---|
| Boeing 747 | `u06_head_optimizer_stack` |
| channel | `u10_head_all` |
| delta wing | `u07_head_loss_regularization_stack` |
| F-22 Raptor | `u09_head_all_non_scheduler` |
| half-cylinder | `u11_head_all_without_alpha` |
| Smoke Buoyancy | `u02_horizon` |
| Tangaroa | `u05_head_alpha_clipping_dropout` |

## Independent audit and retention

Evidence job `51071147` completed with empty stderr. It archived 960/960
per-run CSV files, retained all 960 temporary checkpoints for 52.1, and
published archive SHA-256
`aebbaa7a9f7dd31d4d543143c7f1bb63b34eded12b245da7c7ab918476f591c0`.
The independent audit does not import the selector implementation. It rebuilt
all seven family decisions from the archived CSV files, confirmed equal
paired parameter counts and all source hashes, and matched the selector to a
maximum absolute difference of `3.33e-16`. Its SHA-256 is
`0ec09b9a2dd4f7cc81841d4e8dab5928e27d6e57da9d561a7d57675333d39c0e`.

Final selector artifact SHA-256 values are:

- selection: `5693e802d2ec7c74656b750be30c04a5fb4603e803f1e8eebca72fa3edbea983`;
- leaderboard: `0660706b6340e94889e9676e69b2fc1f72a253f3b6e5f3c5fa31c1aa1d1cac2e`;
- preflight manifest: `ab4b4de684e730798345bcb16fc807804a1eedc6caa50cc17dc1929021984e32`.

This remains a development result. Its winner may enter the preregistered
52.1 portfolio, but only the later fresh spatial-population confirmation can
replace the current paper result.
