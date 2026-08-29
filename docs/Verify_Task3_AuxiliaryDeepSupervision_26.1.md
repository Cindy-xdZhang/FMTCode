# Verify_Task3_AuxiliaryDeepSupervision_26.1

## Question

Can direct training-only supervision of the projected auxiliary representation
increase both absolute FMT classification quality and the paired FMT advantage
over train-only Raw-PCA?

## Motivation

`Verify_Task3_AnchoredFeatureDecomposition_22.1` increased absolute FMT F1 by
only `0.00190`, while reaching a dataset-macro F1 advantage of `0.18691` over
Raw-PCA.  The residual objective supervises only the final
`frozen Raw logit + residual` output.  Its auxiliary projection can therefore
discard class-relevant FMT information whenever the frozen Raw geometry branch
already explains the training sample.

This experiment adds a small classifier to the projected auxiliary vector and
optimizes a second weighted binary cross-entropy loss.  The classifier is used
only during training.  Inference remains exactly the original frozen-Raw plus
residual path.

## Frozen inputs

- Development populations, labels, splits, frozen Raw checkpoints, and
  training budgets come from `Verify_Task3_SpatialRobust_5.2`.
- The completed `Verify_Task3_CombinedOptimization_11.1` selector supplies the
  family-specific optimizer/network/loss recipe.
- The completed `Verify_Task3_AnchoredFeatureDecomposition_22.1` selector
  supplies the family-specific FMT feature.
- Confirmation data remain closed.

## Paired comparison

Every candidate trains both:

1. fixed FMT feature -> auxiliary projection -> residual model; and
2. train-only Raw-PCA with exactly the same input width -> the same auxiliary
   projection -> the same residual model.

The two arms share auxiliary-classifier architecture, auxiliary-loss weight,
class weights, mini-batches, labels, optimizer, epochs, early stopping, fusion
search, and paired seeds `40, 41, 42`.  Raw-PCA is fitted only on training data.

The auxiliary classifier is constructed after all inference modules.  Enabling
it therefore cannot alter the random initialization of the shared inference
path.  Its parameters count toward the capacity ceiling in both arms.

## Candidate grid

- Exact control: no auxiliary classifier and zero auxiliary loss.
- Auxiliary classifier: `linear` or one-hidden-layer `MLP(32)`.
- Auxiliary-loss weight: `0.01`, `0.03`, `0.10`, `0.30`, or `1.00`.

This is one exact control plus a complete `2 x 5` grid: 11 candidates, 10 data
entries, 3 seeds, and 2 paired arms, or 660 training runs in 110 array jobs.

## Selection and targets

Selection is family-specific on exposed development data only.  A candidate is
eligible only if its FMT F1 and FMT Average Precision are each no lower than
the exact control; no tolerance for absolute regression is allowed.  Eligible
candidates are ranked first by FMT-minus-Raw-PCA dataset-macro F1 gain, then by
absolute FMT F1 and the registered robustness tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.185`; and
- absolute dataset-macro FMT F1 at least `0.892`.

Failure to reach either value is recorded as a negative result.  A development
winner cannot support a paper-level generalization claim until it is evaluated
on a fresh, previously unseen spatial population.

## Main files

- `config/Verify_Task3_AuxiliaryDeepSupervision_26.1.yaml`
- `FMT_Utils/PathlineClassifier_3D.py`
- `Verify_Task3_FMTResidual.py`
- `Search_Task3_FMTResidual_3D.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_auxiliary_deep_supervision_26_1.py`
- `ibex_bash/verify_task3_auxiliary_deep_supervision_26.1_*.sh`

## Result

Ibex jobs `50997130` and `50997133` completed all 660 paired trainings and the
registered selector. Dataset-macro Raw-PCA/FMT F1 was `0.69289/0.88490`, a
gain of `+0.19201`; Average Precision was `0.73530/0.94578`, a gain of
`+0.21049`. All ten data entries had positive F1 gain.

The gain target passed, but the joint target failed because absolute FMT F1
was below `0.892`. Relative to 22.1, FMT F1 and Average Precision fell by
`0.00432/0.00466`; Raw-PCA fell more. Direct auxiliary supervision therefore
did not improve the absolute FMT representation under this protocol.
Confirmation remains closed. Selection and leaderboard SHA-256 values are
`1693a9f9...257f` and `d49b2932...7ccd`.
