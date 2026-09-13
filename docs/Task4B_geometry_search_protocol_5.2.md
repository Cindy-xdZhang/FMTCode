# Other_Task4B_GeometrySearch_5.2

Status: preregistered before training. This exploratory version follows the user's request to change the network, learning rate, and geometry differences. The frozen baseline is mainExp_Task4B_PatchSegmentation_5.1; no baseline source or result is replaced.

## Data and selection boundary

Use the exact 5.1 manifest and candidate points, with channel IVD > 5.785 and TBL IVD > 0.08. Labels remain ordinary streamwise (0), ordinary spanwise (1), hairpin head (2), hairpin leg (3). Non-vortex is excluded; invalid primitives remain missed predictions in final evaluation. Channel and TBL always share one classifier. Neither labels, instance IDs, IVD, nor field-computed vorticity enters its input.

Only original training patches participate in tuning. Seed 9051 selects 90 internal validation patches per flow, including approximately one tenth of original training instances. A one-voxel gap separates all fit patches from validation patches. Intersecting original training windows become a temporary buffer, not fit data. Channel: 420 fit, 90 validation, 390 buffer patches; 59 fit and 7 validation instances. TBL: 433 fit, 90 validation, 377 buffer patches; 47 fit and 5 validation instances. All original 900 training patches per flow are restored for the final refit. Original test patches do not participate in this split or tuning.

## Geometry features

Seven streamlines use the center and positive/negative x, y, z offsets, with 33 points each. Coordinates are q=(p-seed)/h. New variants use offset h; old cached FMT retains its frozen offset 0.1501458h. Step h is limited by native interpolation bounds of the patch, accounting for all Runge-Kutta intermediate queries. No field sample outside the patch is used. Interpolation and geometric differences are float64; final features are float32.

Two parameterizations are compared: unit-speed dq/ds=v/|v| and equal-time dq/ds=v/M, where M is the maximum native-node speed inside this patch and is shared by all seven lines. The physical time step is h/M. A uniform positive rescaling of all velocities leaves both geometries unchanged. Unit-speed geometry discards speed variation, so its derivative estimates curl of the direction field, not generally velocity curl.

Let T be the central along-line difference of q, A the 3x3 matrix of positive-minus-negative neighbor displacements, and B the corresponding differences of T. Compute J=B pinv(A) and curl from its antisymmetric entries. In the equal-time case, J approximates (h/M) times the physical velocity gradient. These are geometry estimates, not the stored finite-difference curl used to construct proxy labels. Piecewise trilinear interpolation derivatives can differ from interpolated native-grid finite-difference curl.

Each augmented input concatenates frozen-formula FMT(q), FMT(T), six-frequency Gram/chirality invariants of the curl sequence, and 54 local descriptors (tangents, gradient, normalized gradient, curl, normalized curl, speed, curl magnitude, absolute velocity-curl cosine, helicity, divergence, strain magnitude): 161+161+23+54=399 dimensions. Local directional components intentionally retain the fixed x/y/z flow frame. No claim of rotation invariance is made for this complete input. The new unit/time comparison shares all settings; comparison against old FMT also changes the neighbor offset and numerical precision and is not a single-factor ablation.

Normalization mean and standard deviation use fit rows only during search, and all original training rows during refit. Standard deviations are floored at 1e-6. FMT neighbor blocks have weight 0.5 after standardization. No absolute position, patch ID, flow ID, true curl, or true label is provided to the classifier.

## Frozen search

| Candidate | Input | Network | Initial learning rate |
|---|---|---|---|
| baseline_fmt_mlp | cached FMT 161 | 1024x2 multilayer perceptron | 0.001 |
| wide_fmt_mlp | cached FMT 161 | 2048x2 multilayer perceptron | 0.001 |
| residual_fmt_lr1e3 | cached FMT 161 | width 512, four residual blocks | 0.001 |
| residual_fmt_lr3e4 | cached FMT 161 | width 512, four residual blocks | 0.0003 |
| unit_geometry_curl | unit-speed geometry 399 | width 512, four residual blocks | 0.001 |
| time_geometry_curl_lr1e3 | equal-time geometry 399 | width 512, four residual blocks | 0.001 |
| time_geometry_curl_lr3e4 | equal-time geometry 399 | width 512, four residual blocks | 0.0003 |
| factorized_time_geometry | equal-time geometry 399 | residual network, factorized membership/orientation outputs | 0.0003 |

The factorized classifier predicts two binary distributions internally and multiplies them into the same four supervised class probabilities. It does not receive binary ground-truth inputs. Residual blocks add their learned output to their input; this is a custom classifier, not a reproduction of the image ResNet architecture.

First run a fixed balanced memorization diagnostic: up to 512 rows per flow/class, drawn only from fit, seed 9053, maximum 500 epochs, batch 512, no dropout or weight decay. Pass requires zero classification errors for three consecutive epochs. This tests a finite subset, not perfect fitting of the complete training set. Each candidate then starts a fresh model for internal validation search, seed 9052, 300 epochs, batch 1024, inverse-square-root class weights, Adam, cosine schedule to 1e-6. Baseline candidate retains dropout 0.1 and weight decay 1e-4; others use zero. Validation is evaluated at epoch 1 and every 10 epochs; overlapping validation occurrences are averaged per unique voxel. Selection uses the unweighted mean of the two flows' four-class mean intersection over union (mIoU). Ties choose the first candidate/earliest epoch.

Refit the selected candidate from scratch on all original training rows for the selected epoch count, following the same 300-epoch cosine schedule prefix. Then run the original test once, averaging overlapping occurrences per voxel and counting invalid primitives as errors. This test was reported in 5.1, so it is a reused benchmark, not fresh confirmation. No test metric selects candidates, epochs, or settings. Save predictions, logs, complete config, source hashes, and split; never save a model checkpoint.

## Implementation and checks

- Config: config/Other_Task4B_GeometrySearch_5.2.json.
- Split, integration and geometry: FMT_Utils/Task4B_GeometricCurl_3D.py.
- Build, search, selection, refit and audit: experiments/Task4B_GeometrySearch_5_2.py.
- Tests: tests/test_task4b_geometric_curl_5_2.py (analytic shear curl and velocity-scale invariance).
- Full audit checks original train membership, zero fit/validation source overlap, frozen baseline features, fixed final test support, independent metric recomputation, and no checkpoints.
- Every Ibex process is recorded in docs/ibex_run_registry.md; method findings only in docs/experiment_log.md.

Reference definitions: [MIT curl definition](https://math.mit.edu/~djk/18_022/chapter08/section01.html), [residual learning](https://arxiv.org/abs/1512.03385), [PyTorch Adam documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html). These inform the derivative and optimizer implementations; the frozen project classifier is the experimental baseline.
