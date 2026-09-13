# Verify_AIVDTransfer_1.1 — unchanged Task3 AIVD transfer to Task1/Task2

Registered 2026-09-07 before this experiment's performance evaluation. The user explicitly requests Task3's `aivd1w3_dft` in Task1/Task2, with all cylinder data restricted to physical start time t>=7.5 and the old baselines retrained. This is a diagnostic comparison on previously used benchmark data, not a fresh confirmation and not a replacement of the frozen paper main tables.

## Fixed comparisons

| Arm | Exact input | Dimensions | Role |
|---|---|---:|---|
| old_fmt | fmt_all+kin4 | 189 | Original task-level unified FMT recipe, retrained |
| aivd | aivd1w3_dft | 1 | Primary standalone transfer of Task3's scalar feature |
| aivd_kin4 | aivd1w3_dft+kin4 | 29 | Replace only fmt_all, retain the original kin4 addon |
| raw | centre-relative raw pathlines | 672 | Task2 frozen Raw VAE protocol, retrained |

The primary comparison replaces the full original FMT input with the one-dimensional Task3 representation; it also removes kin4. The separate predeclared control keeps kin4, so these two questions are not conflated. All outcomes are reported; neither arm is chosen using test scores.

Task1: training features -> training-only StandardScaler -> PCA8 for multidimensional arms -> KMeans, two clusters, n_init20. The scalar arm has no PCA because its feature dimension is one; this is a deterministic dimensional requirement, not a searched hyperparameter. Seeds7080–7084. The encoder has no learned parameters, while normalization, PCA and KMeans are fitted. Validation labels only assign the two anonymous clusters to vortex/non-vortex; no supervised decision-threshold search is added.

Task2: same frozen FeatureVAE3D hidden layers [512,256], GELU, latent64, KL weight1e-6, AdamW learning rate3e-4, weight decay1e-5, batch256, exactly7000 optimizer updates, seeds100–104. No validation early stopping or new VAE search. Latent means are clustered with KMeans seed7068/n_init20. Raw uses the existing pre_group_rms normalization; FMT inputs use the existing training-only StandardScaler. Input/output widths change with representation, as in the frozen Raw/FMT protocol; total parameter counts must be reported, not claimed equal. Within each dataset/seed, all four arms train on the same allocated GPU.

## Time and split policy

All ten 3D data entries from the previous replay are included. Only cylinder/half-cylinder are time-filtered. The original simulation interval is [0,15], so the latter half starts at7.5. Select cached slice ordinals solely from recorded source-time metadata, preserving their original roles. Re640 already meets the cutoff; do not halve its [7.5,15] source file again.

| Dataset | Task1 train / validation / test ordinals | Task2 train / validation / test ordinals |
|---|---|---|
| Re160, Re6400 | 4–7 / 8–9 / 2–3 of separate confirmation cache | 4–5 / 6–7 / 8–9 |
| Re640 | 0–7 / 8–9 / 0–3 of separate confirmation cache | 0–5 / 6–7 / 8–9 |
| Other seven entries | Original splits, unchanged | Original splits, unchanged |

Consequently Re160/Re6400 Task2 train on two eligible cached source slices. Keep7000 updates fixed, record the resulting epochs, and do not alter the split based on the outcome. This is not a newly generated uniformly sampled late-time dataset. All arms use identical valid primitives and frozen whole-field IVD p95 references within each run. Old full-time metrics are not the primary comparator.

## Exact implementation boundary

Use the original `FMT_Utils/Task12Data_3D.py` dispatcher and `FMT_Utils/DFT_FMT_3D.py` implementation. No changes to the old encoder, pseudoinverse tolerance, float32 arithmetic, index-time differences, derivative endpoint scheme or Fourier normalization. Derivatives are computed before selecting the first three scalar samples. Each source slice is processed as one complete mean-vorticity context before VAE minibatching. Do not compute features independently per minibatch.

Translation and constant-rotation invariance apply to the scalar in exact arithmetic with a fixed sample context. Strict finite-step invariance under time-dependent rotations is not guaranteed. The retained kin4 block and Raw input are not covered by a scalar objectivity claim. Prediction quality on p95 references is not an objectivity test.

## Validation and retention

Five local synthetic tests verify independent NumPy equivalence, scalar/DFT interpretation, centre independence, translation/fixed-rotation invariance, a time-dependent-rotation counterexample with temporal refinement, and mean-context/fourth-coordinate-frame dependence. Preflight reads only the first eligible training slice of each dataset and checks exact Task3-call equivalence, widths1/29/189, and unchanged kin4; two-update synthetic VAE checks validate widths1 and29 without scientific model selection.

All models and validation cluster mappings in a shard are persisted as metadata before any test cache is opened. No model checkpoint is written. Retain source snapshot/manifest, complete config, per-slice input hashes/time/validity certificates, software/device/timing records, independent prediction-based audit, all350 per-run rows, per-dataset mean/sample-standard-deviation and paired deltas. Macro weights all10 entries equally; no significance or equivalence claim without a separate justified test. Report all100 scientific shards and all failures/retries in the Ibex registry.

Method-level conclusions go only in `docs/experiment_log.md`; the result report can reproduce its recorded evidence with the same scope.
