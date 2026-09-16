# Task4-c: six-neighbor selection, 1.1

Version: `Ablation_Task4C_NeighborSelection_1.1`. The user asked how seeds become bundles, whether training and test use different neighbor rules, and requested comparisons including farthest point sampling (FPS). FPS greedily selects the next seed with the greatest distance to its closest already selected seed. This experiment changes only the six neighbor lines inside each p35 token.

## Current seed-to-bundle construction

The paired dataset is `Ablation_Task4C_BottomDensity_1.2`, scientific commit `a7643890d22c7faf63c1f716bb6222ba5271c5ff`. Its configuration SHA256 is `c92b892ed1db9e6e01d64f4eea7792e616e5ab6f0b4a7436b3b456739be44363`. It has 193,000 training, 3,000 validation and 10,000 test bundles; Channel/TBL training counts are 98,500/94,500. Source output is `/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/outputs/Ablation_Task4C_BottomDensity_1.2`.

1. `FMT_Utils/Task4C_PaperBundles_3_1.py::load_flow` constructs candidate head components from lambda2-qualified cells with positive mean spanwise vorticity fluctuation. Thresholds are Channel -13.395 and TBL -0.0272. Original whole-head ground-truth labels and exclusions are frozen.
2. `FMT_Utils/Task4C_Multiscale_4_1.py::center_and_neighbors` chooses an eligible center cell and a point in its middle half on each axis. Define `h=(dx*dy*dz)^(1/3)` and `r` as 0.25h, 0.5h or h. Candidate seeds are `center+r*(i,j,k)`, for every combination of i,j,k in {-1,0,1}: **27 candidates including the center**. Face, edge and corner offsets are included, with distances r, sqrt(2)r and sqrt(3)r. Seeds must stay in the domain, in the same candidate-head component, and have positive interpolated spanwise vorticity fluctuation. The original center must pass this seed filter.
3. `experiments/Task4C_PhysicalLength_4_14.py::trace_lines` integrates every accepted seed in both directions through the full native vorticity field using Runge–Kutta–Fehlberg (RK45). It does not integrate the velocity field. Current frozen integration tuples `(ds, maximum steps)` are Channel (0.001,80)/(0.002,50)/(0.004,30) and TBL (0.025,160)/(0.05,100)/(0.04,150), giving single-direction targets 0.08/0.10/0.12 and 4/5/6. These are the restored 4.14 tuples, not the abandoned variable-length v2 proposal; this ablation changes none of them.
4. Reject nonfinite, too-short or coordinate-degenerate curves and curves whose halves fail the frozen actual-length/step limits. Concatenate the reversed backward half and forward half, remove repeated points, and resample the full curve at 32 equal arc-length positions. `trace_batch` keeps **all surviving curves**; it does not choose only the nearest ones. Reject the bundle if fewer than 10 survive. The original center curve can fail this integration cleaning while the remaining bundle still passes.
5. Subtract the centroid of all valid curve points and divide by their maximum radius. Save up to 27 curves and their corresponding seed coordinates, with valid counts excluding padding. Physical geometry has shape `(bundles,27,32,3)`.

## A separate selection occurs inside FMT

`FMT_Utils/FMT_P35_NormFrequency_3_1.py::encode` treats **every valid line as an anchor**, not only the originally seeded center. For each anchor it picks the six other valid lines with nearest seed coordinates. The anchor and six neighbors make the seven-line primitive passed to the fixed Fourier encoder. At six frequencies, each token contains center features (23), signed coordinate/tangent features (72), and per-feature mean/max of neighbor descriptors (23+23): 141 values.

The learned shared line network acts on each token; masked mean and maximum pooling over all valid anchors feeds the classifier. Therefore faraway valid lines are not entirely omitted: they appear as anchors. The limitation being tested is missing relations between more distant lines in each local token, not a claim that the whole bundle was truncated to seven curves.

Seed distance measures starting-point separation, not complete-curve similarity. FPS can improve seed coverage but cannot recover curves never seeded or removed during integration, and is not guaranteed to retain every unusual shape. This experiment does not change the seed stencil or use a curve-shape distance.

## Training and test consistency

All splits use the same stencil, nine radius/integration combinations, seed filters, integrator, cleaning, resampling, normalization and FMT neighbor-selection function. Bundles and feature caches are generated before optimization and remain fixed across epochs. Training shuffles rows and enables dropout; evaluation disables dropout. Feature standardization is fitted on training tokens only and applied unchanged to validation/test.

The sampling distributions are not identical: the current training set includes added bottom-biased sampling, while validation/test remain the original fixed files. Original eligible head cells were partitioned into five spatial blocks (three training, one validation, one test). Evaluation centers must be at least 1h from training centers and within 4h of a same-head training center; test centers must also be at least 0.5h from validation centers. Added training centers preserve the frozen distance checks.

The block rule constrains center cells. Neighbor seeds need only stay in the same head component, and integration uses the full native field. Instances and trajectory support can therefore overlap across splits. These are shared-instance local spatial generalization results, not unseen-instance generalization. This is the already used benchmark, not a fresh confirmation set.

## Prespecified comparison

| Variant | Six neighbors per anchor | Training runs |
|---|---|---|
| nearest6 | Six nearest other seed points | Reuse the three frozen p35 runs after exact cache reproduction |
| fps6 | Initialize with anchor; greedily add six farthest-point samples | Seeds 96611–96613 |
| nearest3_fps3 | Initialize with anchor and three nearest neighbors; greedily add three FPS neighbors | Seeds 96611–96613 |

Candidates always comprise all other valid lines of that same cached bundle. Distances use the same float32 `torch.cdist` on normalized seed coordinates as the frozen reference. Equal-distance ties use the first cached line index; permutation invariance under tied distances is not claimed. Selected neighbors must be distinct and exclude the anchor and padded lines. Train/validation/test use the same selected strategy, with no resampling during training.

All models have exactly **76,738 learned parameters**, 141 features per line, six Fourier frequencies, whole-bundle maximum-radius normalization and training-only feature standardization. Use the unchanged shared p35 `h0` network, AdamW learning rate 0.001, weight decay 0.0001, dropout 0.15, batch 128, at most 500 epochs, early stopping after 50 epochs without improved validation F1, and the frozen validation-loss learning-rate schedule. F1 uses threshold 0.5; validation F1 then Average Precision selects the checkpoint in memory. Test is evaluated only after selection. No checkpoint files are saved. Existing Conv3D and other baseline jobs are unaffected.

The frozen paired nearest6 reference is test F1 **0.888404 ± 0.011497** (three optimization seeds); the older approximately 0.72 experiments used different training data and are not the comparison here. Report every new variant without selecting one by test performance. No advantage for FPS is presumed.

## Validation and execution

Implementation: `FMT_Utils/Task4C_NeighborSelection_1_1.py`, runner `experiments/Task4C_NeighborSelection_1_1.py`, configuration `config/Ablation_Task4C_NeighborSelection_1.1.json`, launcher `ibex_bash/task4c_neighbor_selection_1p1.sh`.

The reusable preflight checks an independent scalar greedy implementation, padded/self exclusion, unique choices, repeatability, distant corner coverage, unchanged first 95 center features, exact nearest6 equivalence and finite gradients. It runs on CPU and V100. The trainer loop is copied from the frozen engine with only dispatch changes to use p35 normalization and model for both new method names; no optimizer or stopping changes.

Before encoding, verify all 18 original physical-file hashes and reconstruct all valid world-space seeds to check that every train/validation/test seed is a unique member of its stated 3×3×3 stencil. Record line-count distributions and original-center survival instead of assuming all center curves survive. A training-only V100 pilot must reproduce the first 32 cached nearest6 bundles of each flow exactly. Encode both new choices with identical batch size 32. Report average selected distance, furthest selected distance relative to available extent, and uncovered-seed radius for all three choices; these are diagnostics, not selection criteria.

Slurm chain: preflight → reuse/audit → pilot → two flow encoders → six training runs → independent merge (12 processes total). Every job, failure and cancellation is recorded in `docs/ibex_run_registry.md`; conclusions go in `docs/experiment_log.md`. Independent merging recomputes F1 and checks prediction row IDs against the frozen metadata, including the reused nearest6 reference.

Initial local status: CPU preflight passed. GPU pilot and formal comparison are pending deployment; no new F1 is available at protocol creation.
