# Verify_Task3_OptimizerFamily_28.1

## Question

After learning rate and weight decay are selected, can the optimizer family
improve both the absolute FMT classifier and its paired advantage over
train-only Raw-PCA?

## Why this is distinct

Task3 has used AdamW throughout. `Verify_Task3_LossOptimization_7.1` tried
isolated weight-decay, scheduler, batch-size, and training-length changes, and
`Verify_Task3_LearningRateWeightDecay_27.1` performs a complete AdamW learning
rate by weight-decay search. Neither experiment compares optimizer families.

28.1 is preregistered before any 27.1 performance result exists. It waits for
the complete 27.1 selector, freezes that family-specific anchored feature,
learning rate, and weight decay, and changes only the optimizer algorithm.

## Frozen protocol

- Development populations, labels, splits, frozen Raw checkpoints, batch size,
  epoch budget, early stopping, and fusion search come from 5.2.
- The completed 27.1 selector supplies one learning-rate/weight-decay cell and
  anchored FMT feature per physical family.
- Both arms use the same two-layer, width-64 residual multilayer perceptron
  with zero dropout and paired seeds `40, 41, 42`.
- Confirmation data remain closed.

## Candidates

1. AdamW exact control;
2. AdamW with AMSGrad;
3. Adam;
4. Adam with AMSGrad;
5. Rectified Adam (RAdam); and
6. Nesterov-accelerated Adam (NAdam).

Each candidate uses the exact family-specific learning rate and weight decay
frozen by 27.1. The FMT and train-only Raw-PCA arms use the same optimizer,
initialization seed, mini-batches, labels, network, and training budget. The
grid contains 6 candidates, 10 data entries, 3 paired seeds, and 2 arms: 360
training runs in 60 array jobs.

## Selection and targets

Selection is family-specific on exposed development data. A candidate is
eligible only if its absolute FMT F1 and FMT Average Precision are each no
lower than the AdamW control. Eligible candidates are ranked by paired F1 gain,
then absolute FMT F1 and the registered robustness tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.195`; and
- absolute dataset-macro FMT F1 at least `0.893`.

Failure to reach either target is a negative result. A development winner must
still be evaluated on a fresh spatial population before supporting a paper-level
generalization claim.

## Main files

- `config/Verify_Task3_OptimizerFamily_28.1.yaml`
- `Verify_Task3_FMTResidual.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_optimizer_family_28_1.py`
- `ibex_bash/verify_task3_optimizer_family_28.1_*.sh`
