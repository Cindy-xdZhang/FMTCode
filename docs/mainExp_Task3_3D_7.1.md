# mainExp_Task3_3D_7.1

> Status: **PRE-REGISTERED; waiting for the development-only 44.1, 45.1,
> and 48.1 selectors, followed by the training-free 49.1 portfolio selector.**
> No fifth-population primitive, IVD label, or metric exists.

## Scientific question

Does the family-specific training stack selected by the training-free
`Verify_Task3_FinalPortfolio_49.1` retain at least `+0.15`
dataset-macro F1 advantage over its same-recipe, same-capacity train-only
Raw-PCA arm on a previously unseen fifth spatial primitive population?

## Frozen comparison

- Ten 3D datasets and whole-field IVD-p95 binary labels.
- The family-specific portfolio winner among guarded 44.1, 45.1, and 48.1
  winners is used without post-selection edits.
- FMT and Raw-PCA use the same selected head, optimizer, loss, scheduler,
  training horizon, split, initialization seed, and trainable parameter count.
- Paired seeds 40 and 41 are evaluated: 10 datasets × 2 seeds × 2 arms = 40
  frozen evaluations.
- The selected portfolio result rows, checkpoints, thresholds, residual scales,
  Raw normalization, and train-only Raw-PCA transforms are hashed before the
  fifth population is generated.
- Confirmation performs no training, threshold selection, residual-scale
  selection, feature selection, or hyperparameter selection.

The source searches used seeds 40–42.  Confirmation evaluates the pre-existing
seed-40/41 pair so its uncertainty and compute budget remain directly
comparable to `mainExp_Task3_3D_6.1`; seed 42 was part of development
selection only.

## Fifth spatial population

Physical times, temporal velocity windows, pathline integration parameters,
and IVD-p95 definition remain exactly those used in 6.1.  Only the spatial
seed-grid phase changes.  Before any pending search result was read, the phase
was derived by:

1. SHA-256 of `mainExp_Task3_3D_7.1|fifth-spatial-population-v1`.
2. First eight hexadecimal digits modulo 1024, plus one: Halton index 187.
3. Centered radical inverses in bases 2, 3, and 5.

The resulting phase is
`[0.36328125, 0.13786008230452673, -0.0023999999999999577]`.  It differs from
the four exposed spatial populations.  Its key SHA-256 is
`73f9e8bae962803e24231f1c4e68142be348e7c8f37d7e64b02b1e4a626d0ba3`.

## Decision rule

- Primary: dataset-macro F1 gain at least `+0.15`.
- Aspirational: dataset-macro F1 gain at least `+0.20`.
- Always report Raw-PCA and FMT absolute F1 and Average Precision,
  dataset-macro and family-macro gains, positive-dataset counts, and the worst
  dataset.
- Results may be read only after all ten paired dataset shards finish.

## Execution boundary

The strict dependency order is:

1. The 44.1, 45.1, and 48.1 selectors complete on exposed development data;
   the training-free 49.1 portfolio selector chooses and hashes one winner per
   family.
2. Derive the phase-independent temporal-source manifest and run static
   preflight while fifth-population artifact counts are zero.
3. Freeze the 49.1 portfolio-selection SHA-256 plus all 40 selected
   result/checkpoint hashes.
4. Run source preflight, then generate the fifth-population caches and labels.
5. Run evaluation preflight, ten paired evaluations, and one final summary.
6. After summary artifacts are independently verified, delete temporary
   checkpoints according to the project retention protocol.

Any failed or partial stage is recorded and cannot alter the frozen recipe.

## Main artifacts

- `config/mainExp_Task3_3D_7.1.yaml`
- `Build_Task3_FinalTuned_Confirmation_7_1.py`
- `Prepare_Task3_FinalTuned_SourceManifest_7_1.py`
- `Confirm_Task3_FinalTuned_7_1.py`
- `tests/test_mainexp_task3_3d_7_1.py`
- `ibex_bash/mainexp_task3_3d_7.1_*.sh`
