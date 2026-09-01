# Verify_Task3_AuxiliaryWeightDecay_57.1

## Question

After the independently audited 56.1 portfolio freezes the current best
development recipe per physical family, does assigning the trainable auxiliary
projection a weight decay distinct from the downstream residual head improve
both absolute FMT classification and the paired FMT-minus-Raw-PCA advantage?

## Why this factor is still untested

Experiment 27.1 changed learning rate and weight decay globally, so the
auxiliary projection and downstream residual head always received the same
regularization. Experiment 55.1 separates their learning rates but preserves a
common weight decay. Fixed FMT coordinates and train-only Raw-PCA coordinates
can require different projection regularization even when both arms use the
same trainable architecture and parameter count.

## Frozen sequential protocol

- The source is the complete, independently audited 56.1 development
  portfolio. Its selected feature, projection, residual head, optimizer,
  auxiliary learning-rate multiplier, loss, and all other settings are frozen
  per physical family.
- Only parameters in the trainable auxiliary projection receive the candidate
  weight-decay multiplier. The downstream residual head retains the source
  recipe's weight decay. Frozen Raw parameters remain outside the optimizer.
- FMT and train-only Raw-PCA use the same multiplier, parameter groups,
  initialization, split, seed, and training budget in every cell.
- Multiplier one has no local override and preserves the exact source recipe's
  historical optimizer grouping when its auxiliary learning-rate multiplier is
  also one. Confirmation remains closed.

## Grid, scale, and decision

The multiplier grid is `{0,.01,.05,.10,.25,.50,1,2,4,10,100}`. Eleven
candidates across ten datasets and three paired seeds give 110 array mappings
and 660 paired trainings.

Selection is physical-family specific. A non-control candidate is eligible
only if FMT F1 and FMT Average Precision are both no lower than the exact
source-recipe control. Eligible candidates are ranked by paired dataset-macro
F1 gain, absolute FMT F1, and the registered robustness tie-breakers. The
joint target is F1 gain at least `+0.207` and absolute FMT F1 at least `0.893`.
Failure of either target is retained. No result from an unfinished array may
be inspected, and this development experiment cannot open a new confirmation
population.

## Main files

- `Verify_Task3_FMTResidual.py`
- `Search_Task3_LossOptimization_7_1.py`
- `config/Verify_Task3_AuxiliaryWeightDecay_57.1.yaml`
- `tests/test_task3_auxiliary_weight_decay_57_1.py`
- `ibex_bash/verify_task3_auxiliary_weight_decay_57.1_*.sh`

## Status

Completed and independently re-audited on 2026-09-01. Dataset-macro Raw-PCA/FMT
F1 is `0.685623/0.890239` (gain `+0.204616`); Average Precision gain is
`+0.220076`. Only Boeing selects non-control `c09_auxwd1000`; all other
families retain the exact control. The joint target is not reached. All 660
paired results were archived (SHA-256 `41621e2a...97a`) and 660 temporary
checkpoints were deleted only after the 58.1 independent audit passed.
