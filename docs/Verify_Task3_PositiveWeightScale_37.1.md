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

Implementation and local validation are complete. The local full preflight
records 10 datasets, 15 candidates, 150 array mappings, three paired seeds,
two arms, and 900 expected trainings with `confirmation_opened=false`.
Its manifest SHA-256 is
`976e01a1e50b43816edecfe274eed3000eab64fcd5e3c1cb3a2be504edc20ceb`.
The 22.1/37.1 contract suites pass 10/10 tests and the three execution modules
compile. The local environment has no `pytest` package, so the older
pytest-only loss suite will be rerun in the Ibex environment before any GPU
submission. No performance result has been read, and confirmation remains
closed.
