# Task4-c FPS, augmentation and Fourier network search 1.1

Version: `Ablation_Task4C_FPSAugmentSearch_1.1`. The user authorizes a parallel Ibex search combining farthest-point sampling (FPS: greedily choose the line seed farthest from the selected seed set), tangent jitter, rigid rotations, sequence lengths and curvature-based sampling. Desired test F1 is approximately 0.93. The frozen Conv3D 24³ mean is **0.933198**, and the original BiLSTM plus MLP mean is **0.919499**; reaching 0.930000 alone does not exceed both.

## Frozen data and references

Use every existing BottomDensity 1.2 bundle: 193,000 training, 3,000 validation and 10,000 test. Source commit `a7643890d22c7faf63c1f716bb6222ba5271c5ff`; config SHA256 `c92b892ed1db9e6e01d64f4eea7792e616e5ab6f0b4a7436b3b456739be44363`. Check all 18 physical file hashes, then link the read-only source. Do not change physical centers, integration lengths, valid-line counts, labels, negative sampling, instance membership or splits. The test set is the previously used shared-instance benchmark, not an independent-instance confirmation set.

All valid lines remain anchors. Each anchor uses six FPS neighbors among the other valid lines. Reuse the frozen float32 seed-distance/tie rule from NeighborSelection 1.1. Calculate these indices on the original seed coordinates in every split. Tangent sliding changes sample positions on a curve, not its physical seeding point; a common rigid rotation preserves seed distances. Reusing the indices also avoids new roundoff-dependent tie choices after rotation. No training/test difference in line selection is introduced.

The existing 59-configuration ranking predates FPS. By default, take its **Task4-c validation F1** top five: p13/h0, p10/h0, p14/h0, p17/h0, p21/h0. Their old scores are 0.736591, 0.731765, 0.731361, 0.729634, 0.729332 on the old 4.14 development data. These are selection provenance, not predictions of new performance. Include p35/h0 as a sixth frozen-structure control. The ranking source hash and exact old scores are stored in the new config. The five operations are, respectively: globally largest/smallest 8+8, 2+2, 12+12, 32+32 signed neighbor descriptor values plus their mean, and one largest-absolute descriptor kept in its original column. p35 retains descriptor-wise mean and numerical maximum. Preserve each recipe's actual code and six Fourier bins; use current maximum-radius normalization and neighbor scale/weight 1/1 throughout.

## Geometry before Fourier encoding

Default source is the frozen **32-point polyline**. Resampling to 48 positions does not recover geometric detail removed by the original 32-point resampling. No new integration is performed. This source choice and old-ranking choice were presented as optional clarification; the stated defaults apply unless the user changes them before submission.

Four deterministic sampling forms apply equally to training, validation and test:

- 32-point uniform: exact original array, preserving the reference bit-for-bit.
- 48-point uniform: interpolate cumulative arclength on the original polyline.
- 32-point curvature importance and 48-point curvature importance: estimate the norm of the change of unit tangent divided by local arclength; average adjacent vertex values onto segments. Cap curvature at four times the arclength-weighted line mean. Allocate half the sampling mass uniformly by arclength and half by curvature-weighted arclength. A straight line falls back to uniform mass. Invert this cumulative mass; keep both endpoints.

The ordinary discrete Fourier transform (DFT) acts on the resulting point-index sequence. In the importance-sampling arms, frequency refers to this reparameterized sequence, not uniform physical arclength. This is an explicit representation change; the algorithm is not relabeled a nonuniform Fourier transform.

Six train-only augmentation states cross each sampling form: none, jitter, small rotation, arbitrary rotation, jitter + small rotation, jitter + arbitrary rotation.

- Jitter: each interior point moves forward/backward along its local polyline by a uniform random arclength offset at most 15% of the shorter adjacent segment. Endpoints remain fixed; the bound prevents sample-order reversal. This is tangential sliding, not independent xyz noise. Fresh offsets are generated each training batch/epoch.
- Small rotation: one random axis and an angle uniformly sampled within ±15° per entire bundle.
- Arbitrary rotation: one Haar-uniform proper three-dimensional rotation per entire bundle, obtained from a normalized four-dimensional Gaussian quaternion. No reflections or translations. Rotating every line together preserves relative shape and handedness. This is SO(3) augmentation, not a claim that the original direction spectrum is rotation invariant or that all E(3) transformations are tested.
- Compose sampling → tangent jitter → common rotation → whole-bundle centroid/maximum-radius normalization → six-bin Fourier encoding → training-only feature standardization → learned network. Validation/test have deterministic sampling and **no random augmentation or test-time augmentation**.

Fit feature means/standard deviations to all clean training tokens and, for augmented arms, one fixed augmented view of the same training rows. Use a separate seeded random stream for this fit, separate from model initialization, sample ordering and online augmentation. Validation/test never enter normalization. The extra augmented view prevents rotated coordinate channels from being scaled by a nearly constant unrotated channel. Cache clean tokens only as an implementation optimization; augmented tokens are computed from transformed geometry online.

## Five new network structures

All five use the current p35 Fourier input; no raw-coordinate bypass, estimated velocity gradient or label input. This holds the representation fixed while changing the learnable network. They are custom architectures, not claims of faithful reproductions of named papers.

| Architecture | Definition | Total trainable parameters |
|---|---|---:|
| Wide residual | 256-wide shared line map, three residual feed-forward blocks, masked mean/max, classifier | 992,386 |
| Attention pooling | 192-wide shared line map; four learned weighted line pools plus mean/max | 517,766 |
| Set attention | 192-wide shared map and two six-head line-to-line attention blocks; mean/max | 902,594 |
| FPS graph | 128-wide map; two message-passing blocks on the same six FPS neighbors; mean/max | 217,986 |
| Separate feature branches | Center23, direction72, neighbor46 projected separately to96, fused at256; mean/max | 629,602 |

Each classifier ends with 256→128→2. Dropout remains 0.15; masked aggregation excludes padded lines. New parameter counts are explicitly larger than the 76,738-parameter p35 and 72,192-parameter Conv3D references. A gain cannot automatically be attributed only to augmentation or Fourier encoding. Original-pooling model totals are p13 73,026; p10 71,490; p14 74,050; p17 79,170; p21 88,514; p35 76,738.

## Parallel search and locked evaluation

Eleven model recipes × four sampling forms × six augmentation states = **264 screening fits**. Same training rows, seed96611, batch128, at most80 epochs, patience20 and validation-loss scheduler patience10. This reduced budget is only screening, not the final training budget. Order configurations by validation fixed-0.5 F1, then Average Precision, then stable candidate ID.

The top12 complete candidates each receive **three fresh full-budget training runs** with seeds96612,96613,96711: at most500 epochs, patience50, original scheduler patience15. Select one candidate by three-seed validation mean F1, then mean Average Precision, then ID. Write a global selection lock before any final test prediction.

Finally train that selected recipe and a paired p35/32-uniform/no-augmentation/FPS control with seeds96721–96723: **six final fits**. Report all six outcomes, including failures or failure to reach the objective. No test-driven reranking, threshold selection, additional search or early stopping. Original Conv/BiLSTM and FMT scores remain frozen references with their original seeds; the fresh FPS control provides a paired implementation comparison.

Total **306 fits, 312 Slurm processes** including preflight, reuse, real-data pilot, shortlist, selection and merge. Arrays allow **24 concurrent V100 GPUs** during screening/refinement; actual allocation depends on Ibex scheduling. Batch128, AdamW learning rate0.001, weight decay0.0001, gradient clipping5, deterministic GPU operations. Preserve an old profile's explicit optimizer settings if it is included. Every epoch traverses all193,000 training rows once without replacement; no synthetic class balancing. Record wall time and GPU model per process.

Weights remain in memory; no checkpoint files. Record validation predictions for every screen/refinement fit, and validation/test predictions for final fits. Independent merge checks source row coverage, labels/instances, prediction hashes, confusion matrices/F1, selection epoch, mean/standard deviation and execution identity. Registry includes every failed/cancelled task as well as successful tasks. No automatic further search after the prescribed final comparison.

## Code and engineering gates

Model/geometry: `FMT_Utils/Task4C_FPSAugmentSearch_1_1.py`; runner: `experiments/Task4C_FPSAugmentSearch_1_1.py`; config: `config/Ablation_Task4C_FPSAugmentSearch_1.1.json`; launcher: `ibex_bash/task4c_fps_augment_search_1p1.sh`. Old files are unchanged.

CPU and V100 gates check exact p35/FPS/32-point features; uniform resampling against independent NumPy interpolation; endpoints/order; proper rotations and distance preservation; finite curved/jittered features; parameter counts; padding, batch and line-order behavior; deterministic gradients. A real training-only pilot compares cached frozen FPS features for Channel and TBL, fits32 balanced bundles for100 steps with all six architecture families, and measures batch128 forward/backward including the most expensive geometry transformation. Formal search depends on these gates. Results and conclusions go to experiment_log, metrics to the main table, and all submissions to ibex_run_registry.
