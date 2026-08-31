# Verify_Task3_RawDependencyRebuild_8.1

## Purpose

This is an infrastructure repair for `mainExp_Task3_3D_8.1`, not a model or
hyperparameter experiment. All ten original evaluation children failed before
model inference because their frozen residual checkpoints referenced Raw 3.2
checkpoints that had already been removed by cleanup. No confirmation metric
was produced or read.

## Frozen reconstruction contract

- Use the original 3.2 training script and model module at their registered
  SHA-256 values.
- Use the original development caches, IVD-p95 labels, split, architecture,
  optimizer, seeds 40–41, and preserved baseline validation tables.
- Train only the missing Raw dependency; do not train or alter a residual
  model and do not access the seventh spatial population.
- Run two independent V100 reconstructions.
- Require tensor-identical Raw `state_dict` values between replicas.
- Require every preserved 3.2 validation metric to agree within `1e-12`.
- Require the reconstructed Raw normalization to equal all 40 frozen residual
  checkpoint normalizations exactly.
- Only after every gate passes, install 20 Raw checkpoints at the historical
  paths and relaunch the unchanged 8.1 frozen evaluation.

Any failed check stops the chain. A passed repair establishes dependency
reproducibility only; it does not support FMT performance.

## Main files

- `Repair_Task3_RawDependencies_8_1.py`
- `config/Verify_Task3_RawDependencyRebuild_8.1.yaml`
- `tests/test_task3_raw_dependency_rebuild_8_1.py`
- `ibex_bash/verify_task3_raw_dependency_rebuild_8.1_*.sh`

## Status

Implementation and four focused unit tests pass locally. Ibex deployment and
the strict reconstruction gate are pending.
