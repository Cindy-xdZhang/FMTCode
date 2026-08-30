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

## Status

Implementation is complete. Performance results have not been read and
confirmation remains closed. Deployment identifiers and hashes are added only
after their artifacts exist.
