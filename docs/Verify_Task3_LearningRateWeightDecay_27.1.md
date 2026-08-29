# Verify_Task3_LearningRateWeightDecay_27.1

## Question

Can a complete learning-rate and weight-decay search improve both the absolute
FMT classifier and its paired advantage over train-only Raw-PCA when the
anchored FMT representation is frozen?

## Why this is a distinct experiment

`Verify_Task3_LossOptimization_7.1` tested weight decay as isolated one-factor
changes around its original feature and recipe. It did not evaluate a complete
learning-rate by weight-decay interaction. Earlier 5.2 candidates that changed
learning rate also changed another residual-input recipe, so they cannot answer
this optimizer-only question.

`Verify_Task3_AnchoredFeatureDecomposition_22.1` is the completed development
experiment with the highest absolute FMT F1 so far (`0.88922`) and a paired F1
gain of `0.18691`. 27.1 freezes its family-specific anchored feature and varies
only the two AdamW optimizer hyperparameters.

## Frozen protocol

- Development populations, labels, splits, frozen Raw checkpoints, batch size,
  epoch budget, early stopping, and fusion search come from
  `Verify_Task3_SpatialRobust_5.2`.
- Each physical family uses the completed 22.1 anchored-feature winner.
- Both arms use the same two-layer, width-64 residual MLP with zero dropout.
- Paired training seeds are `40, 41, 42`.
- Confirmation data remain closed.

## Paired comparison

Every optimizer cell trains both:

1. the frozen family-specific FMT feature; and
2. train-only Raw-PCA with the same auxiliary input width.

Learning rate, weight decay, initialization seed, mini-batches, labels,
network, class weighting, training budget, early stopping, and fusion search
are identical between the two arms. Raw-PCA is fitted only on training data.

## Candidate grid

- Learning rate: `0.0002`, `0.0005`, `0.001`, `0.002`, `0.005`.
- Weight decay: `0`, `0.0001`, `0.001`.
- Registered reference cell: learning rate `0.001`, weight decay `0.0001`,
  matching the base config's AdamW defaults.

Some 5.2 Stage2 family recipes embedded a learning rate of `0.0003`. Every
27.1 cell intentionally replaces that embedded value, including the reference
cell, so all seven families are evaluated on the same complete optimizer grid.
The absolute guard is therefore internal to 27.1; it is not described as an
exact reproduction of every family-specific 5.2 Stage2 recipe.

This is a complete `5 x 3` factorial: 15 candidates, 10 data entries, 3 paired
seeds, and 2 arms, or 900 training runs in 150 array jobs.

## Selection and targets

Selection is family-specific on exposed development data only. A candidate is
eligible only if its FMT F1 and FMT Average Precision are each no lower than
the exact optimizer control. Eligible candidates are ranked first by
FMT-minus-Raw-PCA dataset-macro F1 gain, then by absolute FMT F1 and the
registered robustness tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.190`; and
- absolute dataset-macro FMT F1 at least `0.892`.

Failure to reach either value is recorded as a negative result. A development
winner cannot establish a paper-level generalization claim without evaluation
on a fresh, previously unseen spatial population.

## Main files

- `config/Verify_Task3_LearningRateWeightDecay_27.1.yaml`
- `Search_Task3_LossOptimization_7_1.py`
- `Verify_Task3_FMTResidual.py`
- `tests/test_task3_learning_rate_weight_decay_27_1.py`
- `ibex_bash/verify_task3_learning_rate_weight_decay_27.1_*.sh`
