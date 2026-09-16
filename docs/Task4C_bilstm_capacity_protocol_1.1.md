# Task4-c smaller BiLSTM plus MLP, 1.1

Version: `Ablation_Task4C_BiLSTMCapacity_1.1`. The user explicitly corrected the requested total from a mistaken 5.6M to **approximately 56,786 parameters**, shrinking the existing bidirectional long short-term memory network (BiLSTM) plus multilayer perceptron (MLP). The budget includes both modules, biases and normalization parameters. No larger model was submitted.

## Fixed comparison and capacity

Keep the original shared, single-layer, per-line BiLSTM and symmetric mean/max pooling across valid lines. Each vortex line is a spatial sequence of 32 xyz coordinates at one snapshot. Do not concatenate unrelated lines into one sequence. No velocity, vorticity, labels or instance identifiers enter the model.

| Component | Frozen original | New smaller model |
|---|---:|---:|
| Hidden units in each LSTM direction | 81 | 71 |
| Per-line output dimension | 162 | 142 |
| Mean/max concatenation dimension | 324 | 284 |
| MLP widths | 324→64→2 | 284→47→2 |
| BiLSTM parameters | 55,728 | 43,168 |
| MLP parameters, including LayerNorm | 21,058 | 13,585 |
| **Total learned parameters** | **76,786** | **56,753** |

The new total is33 below56,786 (0.0581%). Both recurrent and classifier widths decrease; choose these widths by parameter arithmetic rather than performance search. Recurrent count for input width3 and hidden width h is `8*h*(h+5)` for the two directions, including both bias vectors. With one MLP hidden width m, its count is `(4*h+5)*m+2`, including LayerNorm affine terms. Sum is43,168+13,585=56,753. Assert all three counts against the instantiated PyTorch modules in CPU/GPU checks and before every training run.

The MLP retains Linear→LayerNorm→GELU→Dropout0.15→Linear. The single recurrent layer has no between-layer recurrent dropout. Pooling masks padded lines exactly as before. The forward function is the original with the per-line reshape dimension changed from162 to142; no new features or pooling changes. The larger original result remains frozen: test F1 **0.919499 ± 0.010500** over three seeds, scientific commit `d20c2458d3672558997ade7697dd810bc832c00b`.

## Data and training unchanged

Reuse the complete `Ablation_Task4C_BottomDensity_1.2` data:193,000 training,3,000 validation and10,000 test bundles. Channel/TBL training counts98,500/94,500. Source scientific commit `a7643890d22c7faf63c1f716bb6222ba5271c5ff`; source config SHA256 `c92b892ed1db9e6e01d64f4eea7792e616e5ab6f0b4a7436b3b456739be44363`; source output `/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/outputs/Ablation_Task4C_BottomDensity_1.2`.

Verify all18 source geometry/seed/metadata file hashes and create a read-only physical-data symlink. No resampling, integration changes, label changes, new split or altered bottom-region density. Geometry retains the original bundle-centroid/maximum-radius normalization,10–27 valid lines and32 points per line. Preserve shared-instance spatial blocks and the already used test benchmark; no independent-instance confirmation claim.

Directly reuse the code object of `experiments/Task4C_GeometricBaselines_1_1.py::train`, replacing only the model factory and experiment identity. The internal method key remains `baseline2`, with a new experiment/output directory; this keeps the same raw-geometry loader and no additional feature standardization. Training and prediction routines, per-epoch sample order, batching and stopping rules remain original.

Three seeds96611–96613; V100 with deterministic settings; batch128; AdamW learning rate0.001, weight decay0.0001; gradient clipping5; at most500 epochs. Select by validation fixed-0.5 F1, breaking ties by Average Precision; stop after50 epochs without improved selection score. Validation-loss plateau halves learning rate after15 epochs, minimum1e-6. Write selection lock before loading test for evaluation. Save predictions, metrics, actual total parameters, timing and device. Keep weights only in memory; no checkpoint files.

The original result is a capacity reference, not an equal-parameter comparison with this smaller network. This experiment changes only the two model widths. Other baselines, FMT neighbor comparisons and the ongoing PointNet++ jobs remain unaffected. No tuning based on this test result.

## Checks and deployment

Code: `FMT_Utils/Task4C_BiLSTMCapacity_1_1.py`; runner: `experiments/Task4C_BiLSTMCapacity_1_1.py`; config: `config/Ablation_Task4C_BiLSTMCapacity_1.1.json`; launcher: `ibex_bash/task4c_bilstm_capacity_1p1.sh`.

Preflight verifies exact recurrent/classifier/total counts, padded-line exclusion, line-order invariance, sensitivity to order within each line, batch-independent evaluation and finite gradients. Repeat on V100. The training-only pilot takes32 real bundles, runs100 fixed fitting steps, checks reduced loss, and verifies batch128 forward/backward, memory and throughput. These are engineering checks and do not select settings.

Submission chain: preflight → verify/reuse source files → real-data pilot → three training seeds → independent merge, **seven processes**. No encoding stage is needed. The final merge checks saved prediction hashes, source labels/full row coverage and recomputed F1 before writing the summary and viewer package. Register every job and failure/cancellation in `docs/ibex_run_registry.md`; record method-level conclusions only in `docs/experiment_log.md`. Preserve the original76,786-parameter results and code.
