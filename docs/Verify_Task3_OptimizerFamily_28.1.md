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

## Deployment status

The upstream 27.1 selector completed successfully. Preflight job `50999751`
ran from `2026-08-30T03:32:34+03:00` to `03:33:37` on `cn604-14`, exit code
0 with empty stderr, and froze the seven family-specific 27.1 winners above.
Its manifest SHA-256 is `bdf473ee…5823d`.

Array `50999752[0-59%24]` started at `03:35:48` and completed all 60/60
children with exit code 0. All 60 GPU stderr files are empty. The array used
2 A100-SXM4-80GB, 21 GTX 1080 Ti, 4 RTX 2080 Ti, 28 P100, and 5 V100 GPUs.
Selector `50999753` ran from `04:40:00` to `04:40:07` on `cn113-35-l`, exit
code 0 with empty stderr.

## Complete development result

The family-specific selector returned Raw-PCA/FMT dataset-macro F1
`0.696930/0.888538`, a paired gain of `+0.191608`. Dataset-macro Average
Precision was `0.739405/0.947280`, a gain of `+0.207875`; all ten datasets had
positive F1 gain. Channel selected RAdam, F-22 selected AdamW with AMSGrad,
and the other five physical families retained exact-control AdamW.

Neither preregistered target was reached: F1 gain remained below `0.195` and
absolute FMT F1 remained below `0.893`. Optimizer family is therefore a
negative/neutral development result rather than a new Task3 method.
Confirmation stayed closed. Selection, leaderboard, and preflight SHA-256 are
`9a8b392c…e5ebb`, `2f86dc11…6ff38`, and `bdf473ee…5823d`.
