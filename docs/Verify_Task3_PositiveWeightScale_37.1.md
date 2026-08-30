# Verify_Task3_PositiveWeightScale_37.1

## Question

Can the positive-class weight improve absolute FMT vortex classification while
increasing its paired F1 advantage over the same-width train-only Raw-PCA arm?

The positive-class weight multiplies only the positive term of weighted binary
cross entropy. A scale of one is the existing class-balanced objective; values
below one reduce positive-class emphasis and values above one increase it.

## Why this focused search is justified

The completed broad 7.1 development search tested scales
`{0.35,0.50,0.75,1.25,1.50,2.00}`. Tangaroa selected the lower boundary
`0.35`, F-22 selected the upper boundary `2.00`, and the delta-wing family
selected `0.50`. Those boundary selections show that the useful range was not
closed for every physical family. 37.1 therefore expands both tails and adds
finer cells near the low-scale winners.

This is an independent one-factor search on the later, completed 22.1 anchored
FMT representation. It does not combine positive-weight changes with another
unresolved Task3 search.

## Frozen comparison

- Development populations, labels, split, frozen Raw checkpoints, optimizer,
  learning rate, batch size, epoch budget, early stopping, and fusion search
  follow the completed 5.2 protocol.
- The completed 22.1 selector supplies one anchored FMT feature per physical
  family.
- Both arms use the same two-hidden-layer width-64 residual multilayer
  perceptron with LayerNorm, GELU, and zero dropout.
- Paired seeds are `40, 41, 42`.
- For each candidate, FMT and train-only Raw-PCA receive the same class-weight
  scale, labels, ordering seed, initialization, network, and training budget.
- Confirmation data remain closed.

## Search grid and exact control

The registered scales are
`{0.10,0.20,0.30,0.35,0.40,0.50,0.60,0.75,1.00,1.25,1.50,2.00,2.50,3.00,4.00}`.
The `1.00` candidate carries no training override and is therefore an exact
historical control rather than an approximate reimplementation.

Fifteen candidates across ten datasets produce 150 array mappings and 900
paired trainings.

## Selection and targets

Selection is physical-family specific on exposed development data. A candidate
is eligible only if its absolute FMT F1 and FMT Average Precision are both no
lower than the scale-one control. Eligible candidates are ranked by paired F1
gain, then absolute FMT F1 and the registered robustness tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.195`; and
- absolute dataset-macro FMT F1 at least `0.893`.

Failure of either target remains a negative result. Any development winner
still requires evaluation on a fresh spatial population before it supports a
paper-level generalization claim.

## Main files

- `config/Verify_Task3_PositiveWeightScale_37.1.yaml`
- `Verify_Task3_FMTResidual.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_positive_weight_scale_37_1.py`
- `ibex_bash/verify_task3_positive_weight_scale_37.1_*.sh`

## Status

The experiment is complete. The local and remote full preflights record 10
datasets, 15 candidates, 150 array mappings, three paired seeds, two arms, and
900 expected trainings with `confirmation_opened=false`. Their SHA-256 values
are `976e01a1e50b43816edecfe274eed3000eab64fcd5e3c1cb3a2be504edc20ceb`
and `d4ec39fe9207ee3f98eaad330f58422ceca28609e9f5a283e124f5e3fb0eee34`.
The 22.1/37.1 contract suites pass 10/10 tests locally and on Ibex; the three
execution modules compile and all three Slurm scripts pass `bash -n` on Ibex.

Implementation commit `372a56c` was pushed and archived byte-for-byte. The
archive SHA-256 is
`d6db2b282c5f8729662ad8b0a8e32ddf93c343699a39bf305829a0d0d3911b2f`.
Submitted to Ibex on `2026-08-30`:

- preflight `51010975`, completed at `06:55:52+03:00` on `cn604-08`, exit
  code 0 with empty stderr;
- GPU array `51010980[0-149%24]`, completed 150/150 children from
  `11:54:47` through `14:40:28`, with no failed child; and
- selector `51010981`, completed at `14:40:44` on `cn604-18`, exit code 0.

The family-specific selection yields Raw-PCA/FMT dataset-macro F1
`.69716/.88778`, a paired gain of `+.19062`; Average Precision is
`.74053/.94799`, a gain of `+.20746`. All 10 datasets remain positive. The
selected scales differ from the exact scale-one control only for Channel
(`.10`), Delta-wing (`.50`), and Smoke (`.10`); the other four physical
families retain scale `1.00`.

The result does **not** meet the registered joint target: F1 gain is below
`+.195` and absolute FMT F1 is below `.893`. Relative to the 22.1 development
control, the gain increases by about `+.00371`, but absolute FMT F1 decreases
by about `.00144`. Therefore 37.1 is a useful negative ablation and is not
promoted to a fresh confirmation. Confirmation remains closed.

Leaderboard and selection SHA-256 values are
`85786d390129bc7809e922b507aa8064b7eb8d7dd91b4bf5458086799f5a01a9`
and `25b2f8f305dac07b1860c37c009519c04c59f46df3c8d6efb85a58aafaae30be`;
the downloaded copies match the remote hashes exactly. No checkpoint was
downloaded.
