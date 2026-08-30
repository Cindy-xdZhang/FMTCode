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

## Final development result

Ibex array `50997723[0-149%24]` completed all 150 children and all 900
paired trainings. The final child ended at `2026-08-30T03:32:20+03:00`;
selector `50997724` then completed at `03:32:31`, exit code 0, with empty
stderr. The 150 children ran on 4 A100, 31 GTX 1080 Ti, 19 RTX 2080 Ti,
70 P100, and 26 V100 GPUs.

The selector produced:

- Raw-PCA dataset-macro F1 `0.6969338144`;
- FMT dataset-macro F1 `0.8884536452`;
- paired F1 gain `+0.1915198308`;
- Raw-PCA Average Precision `0.7390788689`;
- FMT Average Precision `0.9469567928`;
- paired Average Precision gain `+0.2078779239`.

All ten dataset-level F1 gains are positive. The family winners are Boeing
`lr=.001, wd=0`; channel `lr=.005, wd=.0001`; delta-wing `lr=.002,
wd=.001`; half-cylinder `lr=.005, wd=0`; Tangaroa `lr=.002, wd=.0001`;
F-22 and Smoke retain the exact `lr=.001, wd=.0001` control.

The relative-gain target is reached, but the absolute FMT target is not:
`0.88845 < 0.892`, hence the joint target is false and confirmation remains
closed. Relative to 22.1, F1 gain improves by `+0.004612`, while absolute FMT
F1 and Average Precision decrease by `0.000770` and `0.003483`. The larger
gap therefore does not establish a stronger absolute FMT classifier.

Artifact SHA-256 values are `5f340246…acdbd` for
`optimization_selection.json`, `ff8542d8…1573` for the leaderboard, and
`76fcf949…6ea6` for the preflight manifest.
