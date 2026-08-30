# Verify_Task3_ResidualDropout_38.1

## Question

Can residual-head dropout improve absolute FMT vortex classification while
increasing its paired F1 advantage over the same-head train-only Raw-PCA arm?

Dropout randomly suppresses hidden activations during training and is disabled
at evaluation. It is applied identically to the paired FMT and Raw-PCA arms.

## Evidence motivating this focused search

The completed broad 7.1 development search included only dropout `0.10` and
`0.20`. Relative to its zero-dropout control, one of those rates improved
paired F1 gain in half-cylinder, delta-wing, F-22, Boeing, and Smoke, while
channel and Tangaroa did not improve. The effects were small and the grid was
too sparse to identify whether weaker regularization works better. This is
therefore a justified independent one-factor search, not a combination with
any unfinished 31.2--37.1 result.

## Frozen comparison

- Development populations, labels, split, frozen Raw checkpoints, optimizer,
  learning rate, batch size, epoch budget, early stopping, and fusion search
  follow the completed 5.2 protocol.
- The completed 22.1 selector supplies one anchored FMT feature per physical
  family.
- Both arms use the same two-hidden-layer width-64 residual multilayer
  perceptron with LayerNorm and GELU.
- Paired seeds are `40, 41, 42`.
- For each candidate, FMT and train-only Raw-PCA receive the same dropout rate,
  labels, ordering seed, initialization, network, and training budget.
- Confirmation data remain closed.

## Search grid and exact control

The registered dropout rates are
`{0,.025,.05,.075,.10,.15,.20,.30,.40,.50}`. The zero-dropout candidate has
no local model override and is the exact historical control.

Ten candidates across ten datasets produce 100 array mappings and 600 paired
trainings.

## Selection and targets

Selection is physical-family specific on exposed development data. A candidate
is eligible only if absolute FMT F1 and FMT Average Precision are both no lower
than the zero-dropout control. Eligible candidates are ranked by paired F1
gain, then absolute FMT F1 and the registered robustness tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.195`; and
- absolute dataset-macro FMT F1 at least `0.893`.

Failure of either target remains a negative result. Any development winner
still requires a fresh spatial population before a paper-level claim.

## Main files

- `config/Verify_Task3_ResidualDropout_38.1.yaml`
- `FMT_Utils/PathlineClassifier_3D.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_residual_dropout_38_1.py`
- `ibex_bash/verify_task3_residual_dropout_38.1_*.sh`

## Ibex deployment

Implementation commit `274718f` was pushed before deployment. The immutable
archive SHA-256 is `bae16519...e2c4`; the remote canonical config SHA-256 is
`cee33cc2...c79`. Local and Ibex environments both passed the 11 relevant
tests; remote Python compilation and all three `bash -n` checks also passed.
The local full-preflight manifest SHA-256 is `efdb2161...fb28`.

Submitted at `2026-08-30T08:28:27+03:00`: CPU preflight job `51012519`, GPU
array `51012521[0-99%24]`, and selector `51012532`. The GPU array has an
`afterok:51012519` dependency and the selector has
`afterok:51012521_*`, verified with `scontrol`. The CPU preflight ran on
`cn604-08` from `08:28:28` to `08:29:45+03:00`, exited zero with empty stderr,
and produced remote manifest SHA-256 `053c8e8a...df11`. The GPU array initially
waited only for the per-user GPU quota.

## Completed development result

The 100 GPU children completed successfully from `14:32:59` to `18:20:03`
on GTX 1080 Ti, P100, V100, and A100 nodes. Selector `51012532` then completed
in eight seconds with empty stderr. All 600 paired trainings are present, no
checkpoint was retained, and confirmation remained closed.

The family-specific selected rates are:

| Physical family | Dropout |
|---|---:|
| channel | 0.10 |
| halfcylinder | 0.40 |
| tangaroa | 0.30 |
| deltaWing | 0.00 |
| f22raptor | 0.50 |
| boeing747 | 0.50 |
| smokeBuoyancy | 0.00 |

Across the ten dataset entries, selected Raw-PCA/FMT F1 is
`0.6918777/0.8883567`, giving a paired gain of `+0.1964790`. Average Precision
gain is `+0.2118812`. The exact zero-dropout control has Raw-PCA/FMT F1
`0.6976753/0.8870755`, hence gain `+0.1894002`. Dropout therefore adds
`+0.0070788` paired F1 gain while also raising absolute FMT F1 by `+0.0012812`.

The registered gain target `>=0.195` is reached, but the absolute FMT F1 target
`>=0.893` is missed by `0.0046433`; consequently the joint target is not
reached. This is a positive development result for combination search, not a
paper-level replacement for `mainExp_Task3_3D_6.1`. Any use of these rates in
a final claim still requires a fresh spatial population.

Final SHA-256 values are:

- `optimization_selection.json`:
  `f94c51b7414107629a202bc29f5f92c4610758bee936081dfb0b6c91e25c7616`
- `optimization_leaderboard.csv`:
  `a2abbb547fd3b8debfb07bf2d82f536c89e69573e9e7944bdab9c9ebcc33a5f8`
- `preflight_manifest.json`:
  `053c8e8a957bc07c7fcb0f03fec2a933f57ba967f8491831fb196a7db00adf11`

Local independent recomputation recovered 10 unique datasets and reproduced
the selected and control macro values exactly from the stored per-family
records. Local and remote hashes match byte for byte.
