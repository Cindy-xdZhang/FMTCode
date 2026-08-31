# Verify_Task3_AuxiliaryBeta2_59.1

## Question

After the independently audited 58.1 portfolio freezes the current best
development recipe per physical family, does assigning Adam's second-moment
coefficient (`beta2`) separately to the trainable auxiliary projection improve
both absolute FMT classification and the paired FMT-minus-Raw-PCA advantage?

## Why this is not the previous Adam search

Experiment 34.1 changed Adam `beta1` and `beta2` globally. The auxiliary
projection and downstream residual head therefore always used the same moment
timescales. Experiments 55.1 and 57.1 separate the projection's learning rate
and weight decay, but still inherit one global Adam `beta2`. The deterministic
FMT projection and the downstream residual head may have different gradient
second-moment timescales, so this experiment changes only the projection's
`beta2` while preserving the downstream optimizer.

## Frozen sequential protocol

- The source is the complete, independently audited 58.1 development
  portfolio. Its feature, projection, residual head, optimizer, auxiliary
  learning-rate/weight-decay settings, loss, and all other settings are frozen
  per physical family.
- Only trainable parameters in the auxiliary projection receive the candidate
  `beta2`. They inherit `beta1` from the source recipe. The downstream residual
  head retains both source optimizer betas; frozen Raw parameters remain outside
  the optimizer.
- FMT and train-only Raw-PCA use the same candidate, parameter groups,
  initialization, split, seed, and training budget in every paired cell.
- The control has no local override and preserves the exact 58.1 source recipe,
  including its historical single optimizer group when no earlier auxiliary
  learning-rate or weight-decay override is active. Confirmation remains closed.

## Grid, scale, and decision

The absolute auxiliary `beta2` grid is
`{0,.5,.8,.9,.95,.98,.99,.995,.999,.9999}`, plus one exact source-recipe
control. Eleven candidates across ten datasets and three paired seeds give 110
array mappings and 660 paired trainings.

Selection is physical-family specific. A non-control candidate is eligible only
if FMT F1 and FMT Average Precision are both no lower than the exact control.
Eligible candidates are ranked by paired dataset-macro F1 gain, absolute FMT F1,
and the registered robustness tie-breakers. The joint target is F1 gain at least
`+0.208` and absolute FMT F1 at least `0.893`. Failure of either target is
retained. No unfinished-array result may be inspected, and this development
experiment cannot open a new confirmation population.

## Main files

- `Verify_Task3_FMTResidual.py`
- `Search_Task3_LossOptimization_7_1.py`
- `config/Verify_Task3_AuxiliaryBeta2_59.1.yaml`
- `tests/test_task3_auxiliary_beta2_59_1.py`
- `ibex_bash/verify_task3_auxiliary_beta2_59.1_*.sh`

## Status

Preregistered before 57.1 or 58.1 produced any performance result. There is no
59.1 performance conclusion yet.
