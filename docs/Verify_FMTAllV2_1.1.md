# Verify_FMTAllV2_1.1 — protocol fixed before performance evaluation

2026-09-07. User-authorized replacement of the nonobjective centre/temporal-coordinate construction, followed by paired reruns of 3D Task1, Task2 and Task5 on all ten existing entries. Implementation conclusions and performance conclusions belong in `docs/experiment_log.md`; this file specifies the experiment.

## Encoder

For the same seven material particles, define at every sampled time
`d_i(t) = x_i(t) - x_0(t)`, i=1,...,6. A time-dependent rigid observer gives
`d_i*(t) = Q(t)d_i(t)`: the vector is covariant, not componentwise invariant.
Before temporal analysis form 21 scalar sequences `G_ij(t)=d_i(t)^T d_j(t)`, i<=j.
Then `G_ij*(t)=G_ij(t)` for every orthogonal Q(t); translation cancels separately at each time.

Normalize by the mean of the six initial squared offsets, subtract the normalized initial Gram matrix, and apply a real-input discrete Fourier transform (DFT) along the sample index. Retain six real coefficients and five nonzero-frequency imaginary coefficients per scalar: 231 features. There is no centre-trajectory feature, temporal point-coordinate difference, or old feature appendage. Normalization/subtraction and bin count use existing Gram-feature conventions, without a search.

The centre is only the origin of same-time relative offsets, not a standalone descriptor. The representation retains relative lengths and angles, loses handedness, and is not invariant to arbitrary changes of neighbour identity/order. Rotated observers must follow the same material particles, not reseed a new axis cross. Temporal Fourier frequencies are cycles per sampled window, not physical Hz; the frozen 32-point caches can have rounded nonuniform physical sample intervals. This does not affect frame objectivity, but limits frequency interpretation.

This is a redesigned objective neighbour encoder, not simply deletion of 23 centre dimensions from the old 161-dimensional feature. No claim of equal information or performance is made in advance.

## Frozen comparison

| Task | Old task recipe | New recipe | Unchanged downstream procedure |
|---|---|---|---|
| Task1 | fmt_all + kin4, 189 dimensions | fmt_all_v2, 231 | train-only StandardScaler, PCA to 8 components, two-cluster KMeans; 5 original seeds; held-out calibration of cluster identity |
| Task2 | fmt_all + kin4, 189 | fmt_all_v2, 231; also rerun Raw | same VAE hidden widths 512/256, latent width 64, KL weight 1e-6, AdamW learning rate 3e-4, 7000 updates, batch 256, decay 1e-5; 5 original seeds; KMeans and calibration unchanged |
| Task5 | fmt_all + gram2 + kin6, 268 | fmt_all_v2 zero padded to 268 | identical residual network and shared frozen Raw backbone, same training/validation scales, 100-epoch maximum with original validation stopping, 5 original seeds; held-out scales evaluated after model/threshold freeze |

Task2 input/output layers necessarily follow the representation width; hidden architecture, objective and budget are identical. Record actual parameter counts. Task5 padding equalizes the entire network parameter count; its Raw branch still uses coordinates, so encoder objectivity does **not** imply full Task5-network objectivity.

Use all ten entries, existing masks, sample identities, splits, IVD p95 labels and seeds. This is a predeclared paired replay on previously used benchmarks, not a new unseen confirmation. Preserve the frozen cylinder times in both arms to isolate the encoder; do not regenerate early-time cylinder data or apply a late-only filter to just one arm. New sampling continues to obey the original-time cutoff t>=7.5. No model, threshold, frequency bin count or feature selection uses test data. Each shard freezes all trained arms and calibrated decisions before loading confirmation data.

Report pooled-per-dataset F1, IoU, precision, recall, and cluster ARI/NMI for Tasks1/2; also Average Precision (AP) for Task5, plus per-unseen-scale results. Macro averages give each dataset equal weight. Primary performance changes are paired new minus old under the same seed and dataset. Do not substitute development-selection values for test performance.

## Evidence and retention

Config: `config/Verify_FMTAllV2_1.1.json`; encoder: `FMT_Utils/FMTAllV2_3D.py`; runner: `experiments/Verify_FMTAllV2_3D.py`; tests: `tests/test_fmt_all_v2_3d.py`. Record code/config SHA256, base git commit, input file hashes, material-row certificates, device and actual scheduler timestamps. Synthetic and training-fixture objectivity checks precede fitting. Fail on mismatched rows, nonfinite features or incomplete primitives; do not filter differently per arm.

Task2 models remain in memory. Task5 writes temporary residual checkpoints only within the shard until final predictions and CSV/JSON metrics are durable, then deletes them. Existing shared Raw backbones are read-only dependencies. No checkpoint download. Preserve all failures and register all submitted jobs.

## Relation to prior code

The same-time scalar construction follows the existing `time_local_gram_dft_features_3d` in `FMT_Utils/DFT_FMT_3D.py`. The new independent NumPy implementation computes geometry and Fourier coefficients in float64 before returning float32, validates finite/positive initial scale, and exposes an explicit `fmt_all_v2` entry without any centre-spectrum or old add-on block. For initial scales above the old epsilon clamp, its mathematical descriptor is the six-bin version of that existing time-local Gram construction. Old Task5 already included a two-bin Gram block alongside nonobjective components; the new encoder uses this objective construction alone. This is an implementation/protocol revision, not a claim that the same-time Gram identity was newly discovered.
