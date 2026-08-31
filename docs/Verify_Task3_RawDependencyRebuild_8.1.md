# Verify_Task3_RawDependencyRebuild_8.1

## Purpose

This was a proposed infrastructure fallback for `mainExp_Task3_3D_8.1`, not a
model or hyperparameter experiment. All ten original evaluation children
failed before model inference because the portable 54.1 package omitted the
Raw 3.2 checkpoints referenced by its residual checkpoints. The initial
diagnosis incorrectly concluded that cleanup had removed the original files;
read-only inspection later showed that all 20 originals were still intact in
the canonical 3.2 output.

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

Rejected. The safer exact-copy dependency closure in commit `a38a102` completed
first and preserved every original checkpoint SHA-256 without retraining. Its
evaluation, summary, and independent audit jobs `51088068 -> 51088083 ->
51088109 -> 51088124` all completed successfully.

The stale reconstruction preflight `51088368` was nevertheless submitted by a
concurrent process. Its own strict gate detected the already complete 8.1
shards and summary and stopped with exit code 1 before any V100 job was
submitted. No model, metric, or confirmation artifact was changed. The code is
retained only to document the rejected fallback; it must not be used to replace
the exact original checkpoints or the audited 8.1 result.
