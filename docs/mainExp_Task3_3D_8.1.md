# mainExp_Task3_3D_8.1

> Status: completed and independently audited. The experiment was
> preregistered before 53.1 produced any performance metric and started only
> after 54.1 selection and its independent audit succeeded.

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

## Dependency repair

Jobs `51074813` through `51074841` completed successfully: the 54.1 recipe was
frozen before data generation, all ten seventh-population caches and both IVD
label groups were generated, and evaluation preflight verified the frozen
recipe and data hashes. Evaluation array `51074843` then failed in all ten
children before model inference because the 54.1 frozen package copied each
residual checkpoint but not the Raw checkpoint it references. The original 20
Raw checkpoints were still intact in the canonical 3.2 source output; the
failure was a missing transitive dependency in the portable package, not a
deleted model or a numerical result. No confirmation shard or metric was
written, and the blocked summary/audit jobs were cancelled.

Operational repair commit `a38a102` adds an exact dependency closure. It reads
the literal `raw_checkpoint` field by pickle opcode disassembly without
unpickling code, copies the original 20 files, verifies their SHA-256 values,
and records which paired FMT/Raw-PCA residuals share each Raw model. The closure
contains 40 residual references, 20 Raw files, zero training run, no read
confirmation metric, and `scientific_configuration_changed=false`. Its SHA-256
is `2e5881bff9726c78ca1d6f4a2a1a4d4390294812c1266a8a2447e45f685e59b5`.
Local and Ibex runs of 35 related tests passed. This exact-copy route was used
instead of retraining because all original Raw checkpoints were available.

## Final result

Repair/evaluation/summary/audit jobs `51088068 -> 51088083 -> 51088109 ->
51088124` all completed with exit code zero. Dataset-macro Raw-PCA/FMT F1 is
`0.69131/0.87136`, an improvement of `+0.18005`. Average Precision is
`0.74392/0.94383`, an improvement of `+0.19991`. Family-macro F1 and Average
Precision gains are `+0.19440` and `+0.21523`.

All ten datasets, all seven physical families, and both paired seeds have
positive F1 gain. The minimum dataset is DeltaWing-LBM at `+0.03633`; seed-40
and seed-41 gains are `+0.18192` and `+0.17818`. The preregistered `+0.15`
primary target passes; the `+0.20` aspirational F1 target does not.

The independent auditor rebuilt all aggregates from 40 rows, checked the 40
frozen model identities, and differed from the summary by at most `1.11e-16`.
Summary, per-run CSV, and audit SHA-256 values are respectively
`0066299c…07bd`, `78d19ec3…bc52`, and `eabc384a…edb3`. The evidence bundle in
`output/mainExp_Task3_3D_8.1_ibex/` contains no model weights and matches every
registered hash.

A concurrently prepared double-V100 reconstruction was not used. Its
preflight job `51088368` started after this exact-copy evaluation had already
completed and correctly failed because confirmation artifacts existed; no GPU
reconstruction job was submitted and no 8.1 artifact was modified.

After the independent audit and local evidence download, artifact job
`51089166` verified the summary, per-run CSV, audit, closure manifest, all 20
temporary checkpoint copies, and their original sources by SHA-256. It then
deleted only those 20 temporary copies and confirmed that none remained. The
original source checkpoints were not touched. The cleanup report SHA-256 is
`a5e1f6e4b6485a737623d98cff9486d00dcde44e6ec41295620d4840bc5b1d35`;
the report is included in `output/mainExp_Task3_3D_8.1_ibex/` and contains no
model weights.
