# Task4-c classic PointNet, small width, 1.1

Experiment: `Ablation_Task4C_PointNet_1.1`. User request: add the classic learned PointNet point-cloud classifier to the current baseline comparison. Continue the previous matched-parameter comparison, preserving both transformation networks while reducing channel widths. Method ID is explicitly `pointnet_small`; this is not the original-width network. The other three baselines continue in their frozen independent deployment.

## Frozen dataset and training

Reuse the complete 193,000 training / 3,000 validation / 10,000 test bundles from `Ablation_Task4C_BottomDensity_1.2`, scientific commit `a7643890d22c7faf63c1f716bb6222ba5271c5ff`. Source output `/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/outputs/Ablation_Task4C_BottomDensity_1.2`. Base config SHA256 `c92b892ed1db9e6e01d64f4eea7792e616e5ab6f0b4a7436b3b456739be44363`. Verify all 18 geometry/seed/metadata files before using a source symlink; no resampling or edits to the source data. Labels, negatives, integration lengths, bottom bias, shared instances and spatial split rules remain fixed. The evaluation files have been used previously, so this is not a fresh confirmation dataset.

Input consists only of the existing bundle-centroid/maximum-radius normalized xyz. Each valid line contributes all 32 points; flatten the 10–27 valid lines into a 320–864-point cloud. Do not supply line identity, arc-length order, tangents, velocity, labels or instance identifiers to the network. No voxelization or fixed Fourier encoding is used.

Three seeds 96611–96613, V100 deterministic computation, batch 128, dropout 0.15, AdamW learning rate 0.001 and weight decay 0.0001, gradient clipping at 5, at most 500 epochs, validation-loss plateau factor 0.5/patience 15/minimum lr 1e-6, and early stopping patience 50 are the same as the FMT comparison. Choose the epoch by validation fixed-0.5 F1, breaking ties by Average Precision. Restore the selected in-memory state, write `selection.lock.json`, then load and evaluate test once at 0.5. The scheduler continues to use classification cross-entropy on validation. The model-specific orthogonality term described below is added only to the training objective. No augmentation, sampling changes or parameter search.

The direct comparison is FMT on the same data: 76,738 parameters and test F1 0.888404 +/- 0.011497. The other three geometric baselines use this same split and training protocol. All historical results remain unchanged.

## Classic structure and capacity adaptation

Primary references: [PointNet paper, CVPR 2017](https://arxiv.org/abs/1612.00593), [official classification source](https://github.com/charlesq34/pointnet/blob/2618f72bc1a0fd21b074096e748016960d44ef55/models/pointnet_cls.py), and [official T-Net source](https://github.com/charlesq34/pointnet/blob/2618f72bc1a0fd21b074096e748016960d44ef55/models/transform_nets.py). Preserve the MIT attribution in `third_party/pointnet/`.

Both transformation networks (T-Nets) learn a matrix from the entire unordered cloud. Each uses shared pointwise layers → max pooling → two hidden dense layers → matrix output. The matrix output has zero initial weights and identity bias, exactly preserving the input at initialization.

| Component | Widths / operation | Parameters |
|---|---|---:|
| Input T-Net | 3→11→22→168; max; 168→88→44→9; reshape 3×3 | 24,031 |
| Initial point layers | 3→11→11 | 220 |
| Feature T-Net | 11→11→22→168; max; 168→88→44→121; reshape 11×11 | 29,159 |
| Remaining point layers | 11→11→22→168 | 4,662 |
| Global classification | max over valid points; 168→88→39→2 | 18,677 |
| Total | Includes both T-Nets and all BatchNorm affine parameters | **76,749** |

This is 11 parameters (0.0143%) above FMT. Widths were selected arithmetically before training, solely to match capacity. The original channels 64/128/1024 and classifier 512/256 are reduced, while the layer sequence, both T-Nets, and max-only aggregation are retained. The original-width binary version of this implementation would have 3,470,283 parameters and is not submitted in this experiment.

Hidden layers use affine maps, BatchNorm and ReLU; classifier dropout follows each hidden layer. Shared pointwise linear layers implement the same per-point operation as a 1×1 convolution. BatchNorm uses epsilon 0.001 and momentum 0.1. Valid points are packed before shared layers and scattered back for pooling, so padding cannot change BatchNorm statistics or maxima. The dense T-Net layers normalize across bundles. PyTorch normalization, running variance and initializers are used, rather than claiming bitwise equivalence to the TensorFlow implementation. Dropout follows the current common training protocol (0.15), not the upstream classifier's 0.3.

## Feature-transform regularization

For the learned feature transform `A` of shape `[batch,11,11]`, add `0.001 * 0.5 * sum((A A^T - I)^2)` to mean classification cross-entropy. The half-sum is over both the batch and matrix dimensions, following the exact `tf.nn.l2_loss` reduction in the official classifier, without an extra division by batch size. Do not regularize the 3×3 input transform. This term encourages near-orthogonal feature transforms; it is part of the classic architecture's training objective. Training loss includes the regularizer; reported validation loss uses classification cross-entropy for the unchanged scheduler. F1 is computed from class predictions as before.

## Verification and execution

Model: `FMT_Utils/Task4C_PointNet_1_1.py`. Runner: `experiments/Task4C_PointNet_1_1.py`. Config: `config/Ablation_Task4C_PointNet_1.1.json`. Launcher: `ibex_bash/task4c_pointnet_1p1.sh`.

Reuse the frozen geometry loader, split identities, prediction code and merge checks. The frozen three-baseline training loop changes only the model factory, absence of fixed-feature z-score, and addition of the official feature-transform loss; verify those source changes before submission. No dependency on Point-NN feature encoding is needed. All input features are learned end to end from xyz.

CPU/V100 preflight checks: point-permutation invariance before and after learning, evaluation-batch independence, identical training outputs and BatchNorm state when padding changes, extra padded-line exclusion, identity initialization and learning of both T-Nets, analytic regularizer value/gradient, exact parameter count, finite forward/backward. A training-only pilot selects the first eight samples of each class in each flow (32 total), takes 100 fixed optimizer steps, checks reduced cross-entropy and a full batch-128 backward pass. It does not select hyperparameters or inspect validation/test performance.

Job chain: preflight → source-file verification/reuse → real-data pilot → three seed jobs → independent prediction merge/viewer export (seven processes). Record every process, including failures. Store histories, selected epochs, full train/validation/test predictions, per-flow metrics, parameter counts, actual GPU, config/code hashes, and training time; retain no weight files. Scientific commit and job IDs go into the experiment log and Ibex registry. Final method-level conclusions belong only in `docs/experiment_log.md`.
