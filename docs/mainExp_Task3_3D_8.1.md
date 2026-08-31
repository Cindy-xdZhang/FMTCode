# mainExp_Task3_3D_8.1

> Status: preregistered before 53.1 produced any performance metric. The job
> chain may start only after 54.1 selection and its independent audit succeed.

## Scientific question

Does the guarded recipe selected by the six-source
`Verify_Task3_ExtendedPortfolio_54.1` retain at least `+0.15` dataset-macro F1
advantage over its paired same-recipe, same-capacity train-only Raw-PCA arm on
a previously unused seventh spatial primitive population?

## Frozen comparison

- Ten 3D datasets with whole-field IVD-p95 binary labels.
- One guarded recipe per physical family across 44.1, 45.1, 48.1, 50.1,
  51.1, and 53.1; no post-selection edit is allowed.
- FMT and Raw-PCA share the selected architecture, optimizer, loss, training
  horizon, split, initialization seed, trainable parameter count, threshold,
  and residual decision protocol.
- Paired seeds 40 and 41 yield 40 frozen model evaluations. Seed 42 remains
  development-only.
- Confirmation performs no training or feature, threshold, scale, model, or
  hyperparameter selection.

## Seventh spatial population

Physical times, temporal velocity windows, pathline integration, and IVD-p95
remain fixed. Only the seed-grid phase changes. Before 53.1 produced metrics:

1. Hash `mainExp_Task3_3D_8.1|seventh-spatial-population-v1` with SHA-256.
2. First eight hexadecimal digits modulo 1024, plus one: Halton index `798`.
3. Centered radical inverses in bases 2, 3, and 5.

This gives phase `[-0.0283203125, -0.21559213534522176,
0.26992000000000016]` and key SHA-256
`2927271d352ad675da727a104e40cb4236e3550506d9cf5f7f348bf63a2fcdb1`.
The phase differs from all six previously declared populations.

## Decision and evidence

- Primary target: dataset-macro F1 gain at least `+0.15`.
- Aspirational target: at least `+0.20`.
- Always report absolute Raw-PCA/FMT F1 and Average Precision, dataset- and
  family-macro gains, positive-dataset/family counts, both seed gains, and the
  worst dataset.
- Results may be read only after all ten dataset shards finish.
- `Audit_Task3_ExtendedTuned_8_1.py` imports neither confirmation nor summary
  code; it independently rebuilds all aggregates and verifies every model and
  evidence hash. Audit success checks integrity and does not require either
  scientific target to pass.

## Main artifacts

- `config/mainExp_Task3_3D_8.1*.yaml`
- `Build_Task3_ExtendedTuned_Confirmation_8_1.py`
- `Prepare_Task3_ExtendedTuned_SourceManifest_8_1.py`
- `Confirm_Task3_ExtendedTuned_8_1.py`
- `Audit_Task3_ExtendedTuned_8_1.py`
- `tests/test_mainexp_task3_3d_8_1.py`
- `tests/test_audit_task3_extended_tuned_8_1.py`
- `ibex_bash/mainexp_task3_3d_8.1_*.sh`
- `ibex_bash/verify_task3_extended_tuned_8.1_evidence.sh`
