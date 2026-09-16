# Task4-c PointNet++ single-scale grouping, 1.1

Version: `Ablation_Task4C_PointNetPlusPlus_1.1`. User authorization on 2026-09-16: implement PointNet++ as another baseline and submit it to Ibex. Continue the existing matched-parameter comparison on the same frozen data. No changes to the other models or running experiments.

## Primary method and frozen data

PointNet++ learns hierarchical local point features by farthest point sampling (FPS), ball grouping, shared pointwise multilayer perceptrons and maximum pooling. FPS selects the next point farthest from the already selected set; ball grouping gathers points inside a fixed Euclidean radius. Use the official single-scale grouping (SSG) classification topology, with one radius at each local level.

Sources: [NeurIPS 2017 paper](https://arxiv.org/abs/1706.02413), [official classification code](https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/models/pointnet2_cls_ssg.py), [set abstraction](https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/utils/pointnet_util.py), and [ball query](https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/tf_ops/grouping/tf_grouping_g.cu). Pin upstream commit `42926632a3c33461aebfbee2d829098b30a23aaa`; preserve its MIT license in `third_party/pointnet2/`.

Reuse every geometry, seed and metadata file from `Ablation_Task4C_BottomDensity_1.2`, scientific commit `a7643890d22c7faf63c1f716bb6222ba5271c5ff`. Base configuration SHA256: `c92b892ed1db9e6e01d64f4eea7792e616e5ab6f0b4a7436b3b456739be44363`. Source root: `/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/outputs/Ablation_Task4C_BottomDensity_1.2`.

Exactly 193,000 training, 3,000 validation and 10,000 test bundles from Channel and TBL; training counts 98,500 and 94,500. Verify all 18 source file hashes before using a read-only physical-data symlink. Same labels, bottom-biased training density, integration lengths, spatial blocks and shared instances as the other baselines. The test set is an already used benchmark, not a new independent-instance confirmation.

Each sample is the same centroid / maximum-radius normalized bundle geometry. Flatten its 10–27 valid lines, each containing 32 points, into a cloud of 320–864 xyz points. Do not input line identity, sequence identity, velocity, vorticity, seeds, labels or instance identifiers to the model. Padding is excluded. The method does not voxelize the geometry.

## Architecture and explicit adaptations

| Level | Centers / grouping | Shared learned layers |
|---|---|---|
| Local 1 | FPS up to 512 centers, radius 0.2, up to 32 neighbors | Relative xyz 3→16→16→32, then neighbor maximum |
| Local 2 | FPS 128 centers, radius 0.4, up to 64 neighbors | Relative xyz + previous features, 35→32→32→64, then neighbor maximum |
| Global | All 128 centers, origin-centered group as in official code | xyz + features, 67→64→128→256, then global maximum |
| Classifier | Two hidden layers and binary logits | 256→80→45→2 |

Total learned parameters: **76,723**, compared with FMT **76,738** (15 fewer, 0.01955% difference). Shared abstraction widths are one quarter of the official 64/64/128, 128/128/256 and 256/512/1024 widths. Classifier widths80/45 are chosen by parameter arithmetic, not performance search. Keep the original 512/128 hierarchy, radii0.2/0.4 and neighbor caps32/64. This is a reduced-width SSG baseline, not the original full-width model or a multi-scale grouping model.

Every hidden layer uses a shared affine transformation, BatchNorm with eps0.001 and momentum0.1, and ReLU. Invalid padded centers do not enter BatchNorm or pooling. The classifier applies dropout0.15 after each hidden layer. PointNet++ SSG has no input/feature transformation networks or their orthogonality regularizer. Loss is binary two-logit cross entropy. Weights use the same native PyTorch initialization convention as the existing baselines.

Variable clouds smaller than512 points contribute all their valid points as level1 centers, with remaining center slots masked. Never synthesize extra valid centers by repeating exhausted FPS indices. Lexicographically sort valid xyz before sampling; start FPS at the first canonical point and break distance ties by this deterministic point order. FPS and grouping use float64 geometric arithmetic for batch-independent indices. Geometry supplied to learned layers remains float32. Exact duplicates remain ordinary distinct input points and are not jittered or removed.

Within each ball, preserve the official rule: take the first up-to-K points in current input order with distance strictly smaller than the radius. If fewer exist, repeat the first valid neighbor to fill K, as in the official code. These repeated entries do participate in local BatchNorm, matching that grouping rule; padded nonexistent centers do not. Local coordinates subtract the corresponding center, without division by the radius. The global group uses the origin rather than recentering the sampled points. No learned or label-dependent sampling and no random data augmentation are introduced.

Canonical ordering, float64 geometric decisions, count capping/masking and native PyTorch kernels are explicit adaptations. They are validated but are not claimed to reproduce every numerical detail of the original TensorFlow CUDA implementation. This comparison uses the frozen Task4-c optimizer rather than reproducing ModelNet40 training or benchmark scores.

## Caching, training and evaluation

Sampling and grouping depend only on frozen xyz, so compute their index arrays once per bundle for every split using the same routine. Cache canonical order, first/second center indices and both ball-group indices as lossless uint16. No features are precomputed: every shared point network remains trainable. The network gathers geometry and intermediate learned features using these indices. This saves repeated FPS work and cannot make grouping depend on training labels or changing weights. Cache size is approximately10.0GiB for206,000 bundles; source geometry is not duplicated.

Use seeds96611–96613, V100, deterministic GPU settings, batch128, AdamW learning rate0.001, weight decay0.0001, gradient clipping5, maximum500 epochs and early stopping after50 epochs without improvement in validation fixed-0.5 F1. Average Precision breaks validation-F1 ties. The validation-loss scheduler halves learning rate after15 epochs, minimum1e-6. All training rows appear once per shuffled epoch. Do not fit additional xyz standardization. Training enables BatchNorm updates/dropout; validation/test use evaluation mode.

Validation alone selects the in-memory state; write the selection lock before reading test for model evaluation. Test does not select radii, widths, epochs or thresholds. No weight checkpoint files are created or downloaded. Preserve per-seed predictions and metrics, training time, actual device and source identity. Report all three seeds and do not change settings in response to test F1.

## Validation, paths and deployment

Implementation: `FMT_Utils/Task4C_PointNetPlusPlus_1_1.py`; runner: `experiments/Task4C_PointNetPlusPlus_1_1.py`; config: `config/Ablation_Task4C_PointNetPlusPlus_1.1.json`; launcher: `ibex_bash/task4c_pointnetplusplus_1p1.sh`.

CPU/V100 preflight independently checks scalar FPS and official-style ball grouping, point permutation, padding exclusion in sampling and training BatchNorm, lossless cached indices, finite forward/backward and parameter count. A training-only pilot uses32 real bundles, tests batch-independent geometry indices, performs100 fitting steps, and verifies actual batch128 gradients, GPU memory and approximate throughput. These engineering checks do not select hyperparameters; a failure blocks dependent jobs.

Slurm chain: preflight → data reuse/hash checks → real-data pilot → two flow index encoders → three training seeds → independent merge, **nine processes**. Reserve32GiB host memory and one V100 per GPU process. The two-day training walltime is a scheduling limit, not a promised completion time. Register every job and any failure/cancellation in `docs/ibex_run_registry.md`. The merge recomputes F1 from saved predictions and checks all row IDs and labels against the frozen metadata before producing final tables and the viewer package. Method conclusions belong in `docs/experiment_log.md`.

Initial local CPU preflight passed. Formal V100 checks and training are pending submission; no PointNet++ performance is available at protocol creation.
