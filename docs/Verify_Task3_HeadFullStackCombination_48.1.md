# Verify_Task3_HeadFullStackCombination_48.1

## Question

Can the selected residual-head capacity and common training horizon improve
the optimizer, loss, regularization, and stability stacks while preserving
absolute FMT F1 and Average Precision?

## Evidence boundary

This adaptive development experiment was declared after the complete 38.1
Dropout result, but before reading any 39.1--47.1 result. All candidate recipes
are fixed in the config before those selectors finish. A source that later
selects its exact control remains in every declared candidate that references
it; candidates are never removed after viewing results. Confirmation remains
closed.

## Frozen comparison

- Development population and split: completed Task3 5.2 protocol.
- Feature: completed 22.1 family-specific anchored FMT representation.
- Paired seeds: 40, 41, 42.
- Within every cell, FMT and train-only Raw-PCA use identical head, optimizer,
  loss, initialization, data order, split, and training budget.
- Every upstream selector must cover the same seven physical families and
  report `confirmation_opened=false`.
- Preflight hashes every source selector before any GPU child runs.

## Fixed grid

The sources comprise the selected head, training residual scale, gradient
clipping, AdamW betas, batch size, positive-class weight, Dropout, focal loss,
parameter exponential moving average, label smoothing, AdamW epsilon, cosine
learning-rate schedule, and training horizon. The 16 candidates contain the
exact feature control, head and horizon controls, focused head interactions,
optimizer/loss/stability stacks, and full stacks with and without alpha,
cosine, or horizon.

Sixteen candidates across ten datasets give 160 array children and 960 paired
trainings. Selection first requires FMT F1 and Average Precision to be no lower
than the exact feature control, then maximizes paired F1 gain using the frozen
tie-breakers.

The development targets are F1 gain `>=0.20` and absolute FMT F1 `>=0.893`.
Any winner remains development-only and must pass a fresh spatial-population
confirmation before replacing the current Task3 paper result.

## Main files

- `config/Verify_Task3_HeadFullStackCombination_48.1.yaml`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_head_full_stack_combination_48_1.py`
- `ibex_bash/verify_task3_head_full_stack_combination_48.1_*.sh`
