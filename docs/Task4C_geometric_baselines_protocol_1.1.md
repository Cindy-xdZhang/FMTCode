# Task4-c: three geometry baselines, 1.1

Version: `Ablation_Task4C_GeometricBaselines_1.1`. User authorization: add tangent/curvature pooling, whole-bundle Point-NN-style point-cloud encoding, and a parameter-matched bidirectional long short-term memory network (BiLSTM). These are binary Hairpin / Non-hairpin classifiers. This request authorizes these methods for Task4-c only; frozen Task6 and the old paper four-class implementation remain unchanged.

## Frozen data and comparison

Reuse every geometry, seed and metadata file from `Ablation_Task4C_BottomDensity_1.2`, scientific commit `a7643890d22c7faf63c1f716bb6222ba5271c5ff`. Source output: `/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/outputs/Ablation_Task4C_BottomDensity_1.2`. The base configuration SHA256 is `c92b892ed1db9e6e01d64f4eea7792e616e5ab6f0b4a7436b3b456739be44363`.

There are 193,000 training, 3,000 validation and 10,000 test bundles from the same Channel and TBL snapshots. Training counts are 98,500 and 94,500 respectively. No new sampling, labels, class balancing, length variation, negatives or instance exclusions. Original spatial-block splits and shared physical instances remain; this is the previously used benchmark, not a fresh independent-instance confirmation set. Validate all physical file hashes before using a read-only source symlink.

All inputs are the same bundle-centroid / maximum-radius normalized geometry, at most 27 valid lines, at least 10, and 32 uniform-arc-length points per line. `counts` excludes padded lines. No velocity, vorticity, labels, instance identifiers or seed metadata enters the models. The present reference is p35 FMT on this 193,000-bundle dataset: 76,738 learned parameters and test F1 0.888404 +/- 0.011497 over seeds 96611–96613. Older approximately 0.72 results used different training datasets and are not the current paired reference. Existing Conv8/12/16/24 results are preserved.

## Prespecified methods

| Method | Geometry encoder and aggregation | Learned classifier | Total learned parameters |
|---|---|---|---:|
| baseline0 | Unit tangent xyz + scalar curvature at each point; concatenate mean and max over valid points, giving 8 values | 8→256→228→64→2 | 76,782 |
| baseline1 | Fixed whole-cloud Point-NN adaptation, 1152 values | 1152→62→76→2 | 76,704 |
| baseline2 | Shared one-layer BiLSTM along each line, hidden 81 per direction; final two hidden states give 162 per line; concatenate masked mean and max over lines, giving 324 | 324→64→2 | 76,786 |

Widths are chosen arithmetically to match the FMT method's **total learned parameter count**, within 0.1%, without performance search. baseline0 and baseline1 fixed feature extractors have zero trainable parameters. baseline2 has 55,728 recurrent and 21,058 classifier parameters. FMT's fixed encoder has zero trainable parameters; its 76,738 parameters belong to its learned per-line and pooled classifier. Equal parameter count does not imply equal computational cost.

Each hidden classifier layer uses Linear, LayerNorm, GELU and dropout 0.15. There is no recurrent dropout between layers in the single-layer BiLSTM; dropout is in its classifier. Baselines 0/1 use per-feature population mean/std fitted only on the combined training features. Std below 1e-8 is replaced by one. Baseline2 consumes the existing normalized xyz without additional feature standardization. There is no learned transformation before baseline0 pooling.

### baseline0 geometry

For an interior point, let `u = x_i-x_(i-1)` and `v = x_(i+1)-x_i`. The oriented unit tangent is `(u+v)/||u+v||`. Endpoints use their one-sided chord. Unsigned curvature is `2||u cross v|| / (||u|| ||v|| ||u+v||)`, the reciprocal circumcircle radius. Copy the adjacent interior curvature to each endpoint. Compute geometry in float64 and store the pooled vector in float32. Numerical floors are 1e-12 for tangent norm and 1e-24 for the product of the curvature denominator. This gives zero curvature for a straight line and 1/r for samples on a circle. Curvature is in the existing normalized coordinate system. Pool all valid points equally; padded values never participate.

### baseline1 Point-NN adaptation

Primary sources: [CVPR 2023 paper](https://arxiv.org/html/2303.08134v2) and [official source at the pinned commit](https://github.com/ZrrSkywalker/Point-NN/tree/a85bfc365258a2c65a5f6ac289537b4d7a7cec0f). License and attribution: `third_party/point_nn/`.

Flatten all valid points (320–864) and discard line and sequence identity. Initial sine/cosine encoding has width 72. Four farthest-point-sampling stages halve the current point count and double the feature width, ending at 1152. Group up to 90 nearest available neighbors. Use relative coordinates and relative features, each normalized by its per-cloud scalar standard deviation; concatenate normalized neighbor features and center features. Apply the upstream additive/multiplicative geometry weighting, neighbor max-plus-mean, fixed evaluation-mode BatchNorm equivalent and GELU. Final point max-plus-mean yields 1152 features. Positional phase uses the official code defaults `1000*x / 100**(j/F)`; its alpha/beta names follow code, not differing notation in the paper.

Explicit adaptations: deterministic native PyTorch sampling rather than the CUDA extension; canonical lexicographic xyz ordering before the first sample to remove input-order dependence; stable distance ties; per-cloud rather than cross-batch standard deviation; fixed normalization without learned affine parameters; variable valid cloud sizes with capped neighbor count; and the user-requested MLP in place of the paper's label-memory classifier. No point duplication, truncation or learned point encoder. This is a Point-NN-style feature baseline, not a claim of reproducing the paper's full classifier.

### baseline2 sequence interpretation

Each vortex line is a **spatial sequence at one snapshot**, not a temporal trajectory. A shared BiLSTM receives only its 32 xyz coordinates. Pooling across lines is symmetric, so line order is irrelevant. The two sequence directions are retained in the final hidden states. Unrelated lines are never concatenated into one artificial continuous sequence. The old two-level, six-channel, four-class paper network and its autoencoder are neither changed nor run.

## Training and evaluation

Use seeds 96611–96613 for every new method, batch 128, AdamW learning rate 0.001 and weight decay 0.0001, at most 500 epochs, early stopping after 50 epochs without improved validation fixed-0.5 F1 (Average Precision breaks ties). The validation-loss plateau scheduler halves learning rate after 15 epochs, minimum 1e-6; clip gradient norm at 5. Every training row appears once per shuffled epoch, without replacement. No search or synthetic augmentation. Use V100 with the same deterministic settings as the reference.

The training loop is copied without algorithm changes from `experiments/Task4C_InstanceCoverage_9_15_v2.py::train`, changing only model construction and which inputs receive train-only standardization; an inline source comparison verifies those two edits. Feature encoders are fixed and may cache unlabeled test geometry; test labels and predictions are loaded by training only after writing `selection.lock.json`. Select on validation, restore the best in-memory state, then evaluate the test once at threshold 0.5. Save full predictions, F1/precision/recall/accuracy/Average Precision, per-flow results, history, parameter counts and training time. Feature encoding time is reported separately. No checkpoint files.

## Checks, implementation and execution

Main code: `FMT_Utils/Task4C_GeometricBaselines_1_1.py`; runner: `experiments/Task4C_GeometricBaselines_1_1.py`; config: `config/Ablation_Task4C_GeometricBaselines_1.1.json`; launcher: `ibex_bash/task4c_geometric_baselines_1p1.sh`.

Reusable preflight verifies analytic straight/circular curvature, padded-line exclusion, point permutation invariance, point batch independence, BiLSTM line permutation invariance and within-line sequence sensitivity, exact parameter counts, and finite gradients. A real-data pilot reads 32 training rows only (eight per class per flow), takes 100 fixed optimizer steps for each method, checks finite gradients and reduced loss, and tests a full batch-128 backward pass. It does not select methods, hyperparameters or seeds. No standalone temporary verification files are retained.

Submission chain: preflight → source-file verification/reuse → real-data pilot → two flow-wise feature encoders → nine training jobs → independent prediction merge and viewer export. Every process is registered, including failures or cancellations. Scientific commit and job IDs are recorded in the experiment log and Ibex registry at submission. After completion, independently recompute saved predictions and verify scheduler/runtime records. Method-level conclusions belong only in `docs/experiment_log.md`.
