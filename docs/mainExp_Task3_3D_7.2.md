# mainExp_Task3_3D_7.2

> Status: **SUBMITTED; waiting for the five development selectors, the
> training-free 52.1 adaptive portfolio, and its independent audit.** No
> sixth-population primitive, IVD label, or performance metric exists.

## Scientific question

Does the family-specific recipe selected by
`Verify_Task3_AdaptivePortfolio_52.1` retain at least `+0.15` dataset-macro F1
advantage over its same-recipe, same-capacity train-only Raw-PCA arm on a new
sixth spatial primitive population?

## Frozen comparison

- Ten 3D datasets and whole-field IVD-p95 binary labels.
- The guarded family winner across 44.1, 45.1, 48.1, 50.1, and 51.1 is used
  without post-selection edits.
- FMT and Raw-PCA use the same selected head, optimizer, loss, scheduler,
  training horizon, split, initialization seed, and trainable parameter count.
- Existing paired seeds 40 and 41 are evaluated: 10 datasets × 2 seeds × 2
  arms = 40 frozen evaluations. Seed 42 participates only in development
  selection.
- The 52.1 selection, copied result rows, checkpoints, thresholds, residual
  scales, Raw normalization, and train-only Raw-PCA transforms are hashed
  before the sixth population is generated.
- Confirmation performs no training or model, feature, threshold,
  residual-scale, or hyperparameter selection.

## Sixth spatial population

Physical times, temporal velocity windows, pathline integration parameters,
and the IVD-p95 definition remain unchanged. Only the spatial seed-grid phase
changes. Before any pending search result was read, it was derived from:

1. SHA-256 of `mainExp_Task3_3D_7.2|sixth-spatial-population-v1`.
2. First eight hexadecimal digits modulo 1024, plus one: Halton index 678.
3. Centered radical inverses in bases 2, 3, and 5.

The phase is `[-0.1044921875, -0.3655692729766804,
0.11632000000000009]`; key SHA-256 is
`ff3312a55c80504295453f30f21340683392157b0e1feabc05a798442ab0581a`.
It differs from all five previously declared spatial populations. The 7.1
entry remains held and unchanged; 7.2 does not reuse its phase.

## Decision rule

- Primary: dataset-macro F1 gain at least `+0.15`.
- Aspirational: dataset-macro F1 gain at least `+0.20`.
- Always report Raw-PCA and FMT absolute F1 and Average Precision,
  dataset-macro and family-macro gains, positive-dataset counts, and the worst
  dataset.
- Results may be read only after all ten paired dataset shards finish.

## Execution boundary

1. All five source selectors finish on exposed development data; 52.1 selects
   one guarded recipe per family and copies all 40 evaluation artifacts. An
   implementation-independent audit must reconstruct every family choice and
   verify all 80 frozen files before this confirmation can start.
2. Derive the phase-independent temporal-source manifest and run static
   preflight while sixth-population artifact counts are zero.
3. Freeze the 52.1 selection SHA-256 plus all result/checkpoint hashes.
4. Run source preflight, then generate caches and IVD-p95 labels.
5. Run evaluation preflight, ten paired evaluations, and one final summary.
6. Independently audit the complete summary before deleting temporary
   checkpoints under the project retention protocol. Job `51073430` is a
   strict `afterok` successor of the summary and exits successfully whenever
   the evidence is internally consistent, regardless of whether the measured
   gain reaches either scientific target.

Any failed or partial stage is recorded and cannot alter the frozen recipe.

## Main artifacts

- `config/mainExp_Task3_3D_7.2.yaml`
- `Build_Task3_AdaptiveTuned_Confirmation_7_2.py`
- `Prepare_Task3_AdaptiveTuned_SourceManifest_7_2.py`
- `Confirm_Task3_AdaptiveTuned_7_2.py`
- `Audit_Task3_AdaptiveTuned_7_2.py`
- `tests/test_mainexp_task3_3d_7_2.py`
- `tests/test_audit_task3_adaptive_tuned_7_2.py`
- `ibex_bash/mainexp_task3_3d_7.2_*.sh`
- `ibex_bash/verify_task3_adaptive_tuned_7.2_evidence.sh`
