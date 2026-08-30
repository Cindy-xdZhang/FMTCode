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

## Status

Implementation is complete. Performance results have not been read and
confirmation remains closed. Deployment identifiers and validation hashes are
added only after the corresponding artifacts exist.
