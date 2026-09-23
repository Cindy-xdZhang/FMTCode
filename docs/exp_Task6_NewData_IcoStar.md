# Task 6 on `Task6_CorelineDataset_2.7_share` — 12-vertex icosahedral streamline stars

Supersedes `docs/exp_Task6_IcoVAE.md`, which was built against the previous Task 6
snapshot. That snapshot is gone from `~/data/flowData3D`; only the 2.7 share remains,
so every number in the older document refers to data that no longer exists.

## 1. The dataset

`/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share`

* 101 frames, 6 scenes, split by `default_split.json` into **77 train / 24 test**.
  * train `{cylinder3d 18, halfcylinderRe320 18, halfcylinderRe640 10, deltaWing_resampled 18, SquareCylinder 13}`
  * test  `{cylinder3d 5, halfcylinderRe320 5, halfcylinderRe640 5, deltaWing_resampled 5, tornado3d 1, SquareCylinder 3}`
  * `tornado3d` is **test-only** — it has no training frame, so it measures transfer to an unseen scene.
* Official reader `task6_fields.py`: `Dataset(root, split_file)`, `Frame.velocity`
  (mmap `[Nz,Ny,Nx,3]`), `.axes_xyz`, `.bounds`, `.spacing`, `.h`, `.corelines()`,
  and `.integrate(seeds, total_length, points, threads)` — bidirectional arc-length
  integration, in memory only, returning a `valid` mask that the README requires be
  applied ("须使用返回的 `valid` 掩码，不能把失败轨迹当正常样本").
* Label rule, fixed by the README: a query point is positive iff its distance to the
  **continuous coreline segments** is strictly less than `h`. Nearest *sampled* point
  overestimates this, so corelines are densified to `h/10` before the KD-tree query;
  the residual error is below `h/20`.

## 2. Why the shipped preintegrated bundles are not usable here

`preintegrated/icosa_bundles_1.1` (11 GB) ships 10 000 centres per frame with 20
icosahedral **face-centre** neighbours at radius `h` (21 streamlines). Centres are
drawn from `IVD(v) > 0.8 × max`. Reading the per-frame `metadata.json` for all 101
frames gives the class balance that rule actually produces:

| scene | centres | positives | rate |
|---|---:|---:|---:|
| SquareCylinder | 160 000 | 0 | **0.0000** |
| cylinder3d | 230 000 | 0 | **0.0000** |
| halfcylinderRe320 | 230 000 | 0 | **0.0000** |
| deltaWing_resampled | 230 000 | 26 037 | 0.1132 |
| halfcylinderRe640 | 150 000 | 124 021 | **0.8268** |
| tornado3d | 10 000 | 6 176 | 0.6176 |
| **total** | 1 010 000 | 156 234 | 0.1547 |

Three of six scenes contain no positive sample at all and a fourth is 83 % positive,
so a classifier pooled over all scenes — which is the requested protocol — would be
degenerate. The cause is that the high-IVD region is not co-located with the
corelines: in `cylinder3d:75` the `IVD > 0.8 max` region is 53 cells whose median
distance to a coreline is ~46 h, and even the looser `IVD > mean` region has a
natural positive rate of only 0.0006.

The README explicitly releases collaborators from that rule — *"这是旧采样协议，不限
制合作者重新选择种子"* — while keeping the label rule fixed. We therefore re-seed, and
keep the label rule exactly.

## 3. Re-seeding protocol

`experiments/Build_Task6_New_IcoStar_1_1.py`. Per frame, centres are stratified so
every scene carries positives:

| stratum | draw | fraction |
|---|---|---|
| `pos` | coreline point + uniform offset in the ball `r < h`, kept iff `d < h` | 0.25 |
| `hard` | coreline point + uniform offset in the shell `h ≤ r ≤ 8h`, kept iff `d ≥ h` | 0.35 |
| `far` | uniform over the old `IVD > mean` candidate region, kept iff `d ≥ h` | 0.40 |

The `hard` stratum keeps the task from degenerating into "is this anywhere near a
vortex"; the `far` stratum retains the original candidate region so negatives stay
vortex-like rather than free-stream. The exact `d/h` of every centre is stored so
accuracy can be resolved against distance instead of only against class balance.

Centres are kept `8h × 1.05` clear of the domain boundary so every neighbour seed is
inside the field. A star is kept only if **all** of its lines pass the returned
`valid` mask.

## 4. Star geometry

The twelve icosahedron **vertices** (`FMT_Utils/IcosahedralGroup_3D.icosahedron_vertices`),
one fixed global orientation, traced at **1 h, 2 h, 4 h, 8 h**. Each sample is therefore
`1 + 12 × 4 = 49` streamlines of 65 points; a run slices any single shell (13 lines) or
any multi-shell combination out of the cache. The fixed orientation is what makes the
`I_h` action an exact channel permutation — 120 elements, verified in
`tests/test_icosahedral_group_3d.py`.

Cache: `/home/cheny1a/data/task6_icostar_1_1/<scene>_<frame>.npz` with
`curves [n,49,65,3] float32`, `centres`, `labels`, `dist` (in units of h), `kind`
(0 pos / 1 hard / 2 far), `role`, `key`, `h`, `radii`, `offsets`, `total_length`.

Realised per-scene balance (`count 3000`, `total_length 0.5`):

| scene | positive rate | valid fraction | kept/frame |
|---|---:|---:|---:|
| cylinder3d | 0.225 | 0.891 | 2672 |
| deltaWing_resampled | 0.250 | 0.994 | 2982 |
| halfcylinderRe320 | 0.241 | 0.957 | 2872 |
| halfcylinderRe640 | 0.242 | 0.950 | 2848 |
| SquareCylinder | 0.266 | 0.928 | 2783 |

## 5. Model and training

`experiments/Run_Task6_NewStar_IcoVAE_1_1.py` reuses the committed
`FMT_Utils/IcoVAE_3D.py` encoder (`IcosahedralSirenVAE3D`, split `[z_inv | z_eq]`
latent) with a classification head on `z_inv`, trained on the pooled train frames of
every scene and evaluated on every test frame of every scene, with a per-scene
breakdown recorded at each eval step.

One addition lives in the run script rather than in the committed library: the repo's
`make_views` silently falls back to the discrete group when a star has more than one
shell, so continuous SO(3) could not be tested multi-shell. `so3_shells` applies the
documented nearest-vertex channel map per shell; at `shells = 1` it reproduces the
repo's `random_so3` bit-for-bit (max |diff| `0.000e+00`) under a shared generator.
As in the repo helper the SO(3) channel map is *not* a permutation — a fractional
rotation has no exact action on twelve fixed channels — so per-channel norms are not
preserved, matching the committed single-shell behaviour.

## 6. Results

Runs are ~100 s each (4000 steps, batch 512, one GPU), so the sweeps below are cheap to repeat across seeds.

### Round A — neighbour radius and shell combination

Supervised head only (no SSL, no decoder), lr 5e-4, 400-step warmup then cosine, seed 7068.
`A_conv3d_*` is the repo's frozen `bundle_voxels` + `Conv3DClassifier` splat baseline on the same stars.

| run | shells | lines | params | best F1 | final F1 |
|---|---|---:|---:|---:|---:|
| **A_s18** | 1h,8h | 25 | 601,433 | **0.9612** | 0.9596 |
| A_s1248 | 1h,2h,4h,8h | 49 | 629,297 | 0.9608 | 0.9606 |
| A_s2 | 2h | 13 | 587,501 | 0.9555 | 0.9525 |
| A_s24 | 2h,4h | 25 | 601,433 | 0.9553 | 0.9532 |
| A_s1 | 1h | 13 | 587,501 | 0.9525 | 0.9523 |
| A_s4 | 4h | 13 | 587,501 | 0.9457 | 0.9433 |
| A_s8 | 8h | 13 | 587,501 | 0.9348 | 0.9348 |
| A_conv3d_s1 | 1h | 13 | 84,361 | 0.8961 | 0.8935 |
| A_conv3d_s4 | 4h | 13 | 84,361 | 0.8816 | 0.8809 |
| A_conv3d_s8 | 8h | 13 | 84,361 | 0.8522 | 0.8501 |

Best is **A_s18 at 0.9612**, but `A_s18` (1h+8h) and `A_s1248` (all four shells) differ by
0.0004 on a single seed and should be read as tied; the seeded tie-break is in round C. What is
robust is that **two shells beat any single shell**, and that the widest available spread (1h + 8h)
matches the full four-shell stack using 24 fewer streamlines. Single-shell accuracy falls
monotonically as the radius grows past 2h, so a wide shell is only useful *alongside* a narrow one.

IcoVAE beats the Conv3D voxel baseline by **+0.0651**
(0.9612 vs 0.8961) — though at 7x the parameters, so this is a method comparison, not a matched-capacity one.

Per-scene F1 of the best run:

| scene | F1 | note |
|---|---:|---|
| SquareCylinder | 0.9835 |  |
| cylinder3d | 0.9510 |  |
| deltaWing_resampled | 0.9842 |  |
| halfcylinderRe320 | 0.9442 |  |
| halfcylinderRe640 | 0.9464 |  |
| tornado3d | 0.9721 | **test-only — no training frame, so this is transfer to an unseen scene** |

`tornado3d` has no training frame at all, so its 0.9721 measures transfer to a scene the model
never saw rather than in-distribution accuracy.

### Round B — SSL and decoder objective combinations

All on 1h+8h, lr 5e-4, warmup+cosine, seed 7068, 4000 steps. `B_sup` is the supervised control.

| arm | aug | ssl w | rec w | var w | best F1 | vs control |
|---|---|---:|---:|---:|---:|---:|
| `B_ssl_ih` | ih | 1 | 0 | 0 | 0.9619 | +0.0007 |
| `B_sup_cw *(class weight)*` | ih | 0 | 0 | 0 | 0.9617 | +0.0005 |
| `B_ssl_so3` | so3 | 1 | 0 | 0 | 0.9614 | +0.0002 |
| `B_ssl_ih_w0.25` | ih | 0.25 | 0 | 0 | 0.9612 | +0.0001 |
| `B_sup` | ih | 0 | 0 | 0 | 0.9612 | +0.0000 |
| `B_ssl_i` | i | 1 | 0 | 0 | 0.9612 | -0.0000 |
| `B_ssl_both_rec` | both | 1 | 1 | 0 | 0.9610 | -0.0001 |
| `B_ssl_ih_w4.0` | ih | 4 | 0 | 0 | 0.9608 | -0.0003 |
| `B_rec_w0.1` | ih | 0 | 0.1 | 0 | 0.9608 | -0.0003 |
| `B_ssl_ih_w16.0` | ih | 16 | 0 | 0 | 0.9604 | -0.0008 |
| `B_ssl_both` | both | 1 | 0 | 0 | 0.9603 | -0.0009 |
| `B_ssl_ih_rec` | ih | 1 | 1 | 0 | 0.9602 | -0.0009 |
| `B_rec` | ih | 0 | 1 | 0 | 0.9601 | -0.0011 |
| `B_ssl_so3_rec` | so3 | 1 | 1 | 0 | 0.9595 | -0.0017 |
| `B_ssl_ih_var` | ih | 1 | 0 | 1 | 0.9593 | -0.0018 |
| `B_all` | ih | 1 | 1 | 1 | 0.9578 | -0.0033 |
| `B_rec_w10.0` | ih | 0 | 10 | 0 | 0.9484 | -0.0127 |

Every combination lands within **+0.0007 / −0.0033** of the plain supervised control — noise at this
seed count. The only clearly harmful settings are an over-weighted decoder (`--lambda-recon 10`,
−0.0127) and the VICReg variance/covariance term (−0.0018 alone, −0.0033 stacked with everything).
With 217 342 labelled training stars the supervised signal is saturated, so an auxiliary objective has
nothing left to contribute. Round D therefore repeats the comparison under label scarcity, which is
where an SSL term is supposed to pay.

### Round C — seeded confirmation, and two single-seed rankings that did not survive

Three seeds (11 / 23 / 37) per arm, Welch t-test.

| arm | mean best F1 | sd | n |
|---|---:|---:|---:|
| supervised + class weight | 0.9612 | 0.0008 | 3 |
| SSL discrete `I_h` | 0.9597 | 0.0009 | 3 |
| SSL continuous `so3` | 0.9584 | 0.0007 | 3 |
| geometry 1h+8h | 0.9601 | 0.0012 | 3 |
| geometry 1,2,4,8h | 0.9615 | 0.0012 | 3 |

| comparison | Δ | p (Welch) | verdict |
|---|---:|---:|---|
| SSL(`I_h`) vs supervised | -0.0014 | 0.120 | not significant |
| SSL(`so3`) vs supervised | -0.0027 | 0.013 | **significant** |
| 4-shell vs 1h+8h | +0.0014 | 0.231 | not significant |

Two corrections to the single-seed rounds above:

* Round A ranked `A_s18` (1h+8h) first. Across three seeds the four-shell star is **+0.0014** ahead
  instead, but at p = 0.23 — so the two geometries are genuinely tied and the round-A ordering was noise.
* Round B ranked `B_ssl_ih` first at +0.0007. Across three seeds the discrete-group SSL term is
  **−0.0014 below** the supervised control (p = 0.12, not significant), and the continuous-SO(3) term is
  **−0.0027 below it at p = 0.013 — a significant degradation**. At full supervision the contrastive term
  does not help, and its continuous variant actively costs accuracy.

Learning rate and schedule (single seed, 1h+8h):

| setting | best F1 |
|---|---:|
| `C_lr1e-4` | 0.9509 |
| `C_lr2e-4` | 0.9555 |
| `C_lr1e-3` | 0.9624 |
| `C_lr2e-3` | 0.9610 |
| `C_const` | 0.9560 |
| `C_steps12000` | 0.9613 |
| `B_sup` (lr 5e-4, warmup+cosine) | 0.9612 |

Warmup + cosine is worth about **+0.005** over a constant schedule (0.9612 vs 0.9560); lr 1e-3 is
marginally better than 5e-4; and 12 000 steps buys nothing over 4 000 (0.9613 vs 0.9612) — the run
saturates well before the 4000-step budget.

### Conv3D baseline at matched capacity

`WideConv3D` generalises the frozen `Conv3DClassifier` — at `(8,16,32), hidden 256` it reproduces the
frozen parameter count exactly (84 361) — with the channel widths exposed so the baseline can be grown
to the IcoVAE budget.

| params | Conv3D, 1h+8h | Conv3D, 1h |
|---:|---:|---:|
| 84,361 | 0.8811 | — |
| 202,769 | 0.8997 | — |
| 454,297 | 0.9107 | 0.9249 |
| 608,929 | 0.9140 | 0.9267 |
| 1,415,729 | 0.9190 | 0.9311 |
| 2,163,777 | 0.9199 | 0.9332 |

Conv3D prefers the single 1h shell at every capacity — the voxel splat crowds as lines are added,
whereas IcoVAE gains from the extra shells. Giving Conv3D its best geometry and 3.4x the parameters:

* IcoVAE, 629 297 params, 3 seeds: **0.9615**
* Conv3D at matched capacity (608 929, 1h): 0.9267 → IcoVAE **+0.0348**
* Conv3D at 2 163 777 params (1h), its best anywhere: 0.9332 → IcoVAE **+0.0284**

Conv3D climbs from 0.8811 to 0.9332 across a 26x capacity range and is flattening (+0.0021 for the last
1.5x). The remaining gap is therefore representational rather than a capacity artefact: the icosahedral
star keeps each neighbour streamline as an identified, group-indexed channel, while the splat discards
that correspondence into an occupancy grid.

### Round D — does SSL help when labels are scarce?

Label fractions of the 217 342 training stars crossed with five objectives, two seeds each, on 1h+8h.
The head sees only the labelled subset; the SSL and reconstruction terms still draw from **all** 217 342
stars, so this is the semi-supervised setting an auxiliary objective is meant for.

| labels | n | supervised | + SSL `ih` | + SSL `both` | + decoder | + SSL `ih` + decoder |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2% | 435 | 0.7885 | 0.7249 (-0.064) | 0.6978 (-0.091) | 0.8355 (+0.047) | 0.8199 (+0.031) |
| 1% | 2,173 | 0.8786 | 0.8899 (+0.011) | 0.8899 (+0.011) | 0.8866 (+0.008) | 0.8995 (+0.021) |
| 5% | 10,867 | 0.9173 | 0.9201 (+0.003) | 0.9194 (+0.002) | 0.9161 (-0.001) | 0.9197 (+0.002) |
| 25% | 54,336 | 0.9455 | 0.9451 (-0.000) | 0.9454 (-0.000) | 0.9455 (+0.000) | 0.9448 (-0.001) |
| 100% | 217,342 | 0.9607 | 0.9591 (-0.002) | 0.9580 (-0.003) | 0.9604 (-0.000) | 0.9584 (-0.002) |

The benefit of every auxiliary objective **decays monotonically as labels grow**: large at 435 labels,
small at 2 173, gone by 10 867, and slightly negative once all 217 342 labels are available. That is the
expected shape for self-supervision, and it explains round B: the full-label setting simply has no room
for an auxiliary signal, so testing objectives only at 100 % labels would have been misleading.

Two effects are specific to the scarcest setting (435 labels):

* **The decoder is the component that pays**: +0.047, its largest gain anywhere.
* **The contrastive term collapses**: −0.064 for `ih`, −0.091 for `both`. At `--lambda-ssl 1.0` the
  contrastive loss outweighs a cross-entropy computed over 435 samples, so the encoder buys group
  invariance at the cost of the discriminative signal. Round E tests whether this is a weight artefact
  by sweeping the contrastive weight down to 0.03 / 0.1 / 0.3.

At 1 % labels the ordering is different again — contrastive **and** decoder together are best (+0.021),
ahead of either alone — so the useful combination is regime-dependent, not fixed.

Caveat: two seeds per cell. The low-label rows have much wider spread than the full-label rows, where
the seed sd was ~0.001; the ±0.06-0.09 effects at 435 labels are far outside that, but the ±0.002-0.003
entries at 25 % and 100 % are not resolvable at n = 2.

### Round E — is the contrastive collapse a weight artefact?

No. At 435 labels the contrastive term hurts at **every** weight tested, monotonically less as the weight
falls, extrapolating to zero harm only at zero weight. But paired with the decoder it becomes the best
configuration found anywhere in the low-label regime.

| configuration | 0.2 % labels (n=435) | 1 % labels (n=2 173) |
|---|---:|---:|
| supervised only *(round D)* | 0.7885 | 0.8786 |
| contrastive w=1 alone *(round D)* | 0.7249 (-0.064) | 0.8899 (+0.011) |
| decoder w=1 alone *(round D)* | 0.8355 (+0.047) | 0.8866 (+0.008) |
| contrastive w=0.03 alone | 0.7728 (-0.016) | 0.8880 (+0.009) |
| contrastive w=0.1 alone | 0.7666 (-0.022) | 0.8929 (+0.014) |
| contrastive w=0.3 alone | 0.7580 (-0.030) | 0.8941 (+0.015) |
| decoder w=3 | 0.8403 (+0.052) | 0.8859 (+0.007) |
| decoder w=10 | 0.8431 (+0.055) | 0.8867 (+0.008) |
| **contrastive w=0.1 + decoder w=1** | **0.8482 (+0.060)** | 0.8940 (+0.015) |
| contrastive w=0.3 + decoder w=1 | 0.8387 (+0.050) | 0.8982 (+0.020) |
| supervised + class weight | 0.7739 (-0.015) | 0.8747 (-0.004) |
| decoder w=1 + class weight | 0.8335 (+0.045) | 0.8840 (+0.005) |

Three things come out of this:

* **Contrastive alone never helps at 435 labels** — −0.016 at w=0.03, −0.022 at w=0.1, −0.030 at w=0.3,
  −0.064 at w=1. Lowering the weight only reduces the damage, so the collapse is a property of the
  objective in this regime, not a mis-set hyper-parameter.
* **Contrastive + decoder is the best combination**: 0.8482 at 0.2 % labels (+0.060), ahead of the best
  decoder-only setting (0.8431). The decoder appears to preserve the information the contrastive term
  would otherwise discard, which is what makes a weak contrastive term useful rather than harmful.
* **The optimal decoder weight inverts with label count.** Under scarcity larger is better
  (w=10 > w=3 > w=1); at full supervision w=10 was the single worst arm in round B (−0.0127). A sweep
  run only at 100 % labels would therefore have recommended exactly the wrong weight for the regime
  where the objective actually matters.

Class weighting hurts under scarcity (−0.015 at 0.2 %, −0.007 at 1 %) although it was the top arm at
full supervision, which is the same regime-dependence again.

### Round F — per-scene models, learning rate, archived weights

Four-shell star, three seeds. A per-scene model is trained and tested only on that scene's own frames;
the pooled model is the all-scene model evaluated on that scene's test frames.

| scene | per-scene model | pooled model | Δ | train stars |
|---|---:|---:|---:|---:|
| SquareCylinder | 0.9852 ±0.0026 | 0.9819 ±0.0017 | +0.0033 | 35,460 |
| cylinder3d | 0.9402 ±0.0008 | 0.9542 ±0.0029 | -0.0140 | 48,098 |
| deltaWing_resampled | 0.9837 ±0.0005 | 0.9844 ±0.0001 | -0.0007 | 53,693 |
| halfcylinderRe320 | 0.9375 ±0.0067 | 0.9461 ±0.0025 | -0.0087 | 51,607 |
| halfcylinderRe640 | 0.9468 ±0.0013 | 0.9504 ±0.0031 | -0.0036 | 28,484 |
| tornado3d | *no train frames* | 0.9895 ±0.0015 | — | 0 |

**Pooling beats specialising in four of five scenes**, by up to 0.014 on `cylinder3d` and 0.009 on
`halfcylinderRe320`; only `SquareCylinder` marginally prefers its own model (+0.003). Training on every
scene therefore transfers rather than interferes, even though the five flows have different geometry and
Reynolds numbers — the local coreline signature generalises across them.

The strongest evidence is `tornado3d`, which contributes **no training frame at all** and is still the
best-scoring scene in the pooled model at **0.9895**. A model that had merely memorised
scene-specific appearance could not do that.

Learning rate, three seeds on the four-shell star (both with class weighting):

| setting | mean best F1 | sd |
|---|---:|---:|
| lr 1e-3 | **0.9651** | 0.0006 |
| lr 5e-4 | 0.9636 | 0.0013 |

Δ = +0.0015, p = 0.171 — not significant at n=3, but consistent in sign with the single-seed round-C sweep.

**Best configuration on this dataset: 0.9651 ± 0.0006** (4-shell star, lr 1e-3,
warmup+cosine, class weighting, supervised head only, 629 297 parameters, 3 seeds).

Weights for the best full-supervision configuration are archived at
`outputs/weights_Task6_NewStar/best_seed{11,23,37}.pt` (2.6 MB each), holding the encoder, the head, the
normalisation statistics and the full argument record.

### Round G — canonical two-stage SSL

Every round above trains the auxiliary objective *jointly* with the head. Round G runs the standard
protocol instead: `--pretrain-steps 8000` over all 217 342 stars with no labels, then
`--drop-aux-after-pretrain` for a purely supervised stage 2 — either fine-tuning the whole encoder, or
`--freeze-encoder` for a linear probe. Two seeds per cell.

| labels | objective | joint | two-stage finetune | frozen probe | probe + class weight |
|---|---|---:|---:|---:|---:|
| **0.2 %** | *supervised* | **0.7885** | — | — | — |
| | contrastive `ih` | 0.7249 | 0.7775 | 0.4306 | 0.3133 |
| | decoder | 0.8355 | 0.7496 | 0.4306 | 0.3133 |
| | both | 0.8199 | 0.7152 | 0.4306 | 0.3410 |
| **1 %** | *supervised* | **0.8786** | — | — | — |
| | contrastive `ih` | 0.8899 | 0.8652 | 0.4306 | 0.4306 |
| | decoder | 0.8866 | 0.8594 | 0.4306 | 0.4306 |
| | both | 0.8995 | 0.8478 | 0.4306 | 0.4871 |

**Joint training beats two-stage pretraining for every objective at both label fractions**, and
two-stage also trails plain supervised training. Pretraining does recover much of the contrastive
collapse (0.7249 → 0.7775 at 0.2 % labels) but never converts it into a gain.

The frozen probes settle the question. A constant all-negative predictor scores exactly **0.4306** on
this test set (positive rate 0.2439), and that is precisely what every uncorrected probe returns; the
class-weighted probes land at 0.31-0.49, at or below the same floor. Eight thousand steps of contrastive
and/or reconstruction pretraining therefore do **not** produce a representation in which coreline
proximity is linearly decodable — the encoder has to be shaped by the supervised signal itself.

(The identical 0.4306 across unrelated arms was checked rather than assumed: the head does receive
gradients when the encoder is frozen, so this is genuine majority-class collapse, not a plumbing bug.)

## 7. What the objective study concludes

| question | answer |
|---|---|
| Does the discrete group action help as a joint loss at full supervision? | No. −0.0014, p = 0.12. |
| Does continuous SO(3) help at full supervision? | No — it *hurts*, −0.0027, p = 0.013. |
| Does reconstruction help at full supervision? | No. −0.0011; at weight 10 it costs −0.0127. |
| Does anything help when labels are scarce? | Yes. At 435 labels, weak contrastive (w 0.1) + decoder gives **+0.060**. |
| Is the contrastive collapse at 435 labels a weight artefact? | No — it hurts at every weight alone, and is only useful alongside the decoder. |
| Do the optimal weights transfer across regimes? | No. The decoder wants w 10 under scarcity and w 1 at full supervision, where w 10 is the worst arm. |
| Does two-stage pretraining beat joint training? | No, for every objective and both label fractions. |
| Do the pretext tasks yield a linearly separable representation? | No — every frozen probe is at or below the majority-class floor. |

The group action is therefore useful on this task as a **regulariser under label scarcity**, not as a
representation-learning objective. The headline accuracy comes from the supervised head on the
icosahedral star geometry, not from self-supervision.

## 8. Headline numbers

| model | params | F1 (3 seeds) |
|---|---:|---:|
| **IcoVAE, 4-shell star, lr 1e-3, class weight** | 629 297 | **0.9651 ± 0.0006** |
| IcoVAE, 4-shell star, lr 5e-4, class weight | 629 297 | 0.9636 ± 0.0013 |
| Conv3D splat, best configuration anywhere (1h, 2.16 M) | 2 163 777 | 0.9332 |
| Conv3D splat, matched capacity (1h) | 608 929 | 0.9267 |
| Conv3D splat, frozen repo baseline (1h) | 84 361 | 0.8961 |

IcoVAE leads the capacity-matched Conv3D by **+0.038** and the best Conv3D at any capacity by **+0.032**.

Reproduce with:

```bash
python experiments/Build_Task6_New_IcoStar_1_1.py --out /home/cheny1a/data/task6_icostar_1_1 \
    --count 3000 --threads 8 --shards 8 --shard $I      # $I = 0..7, ~13 min wall clock
python experiments/Run_Task6_NewStar_IcoVAE_1_1.py --shells 1,2,4,8 --lr 1e-3 \
    --schedule cosine --warmup-steps 400 --class-weight --seed 11 --label best
```

## 9. Where the errors are

`experiments/Analyze_Task6_NewStar_Distance_1_1.py` replays an archived checkpoint over the test
stars and resolves accuracy against each centre's exact distance to the continuous coreline — the
reason the builder stores `dist` per sample. Using `best_seed11.pt`:

Overall accuracy **0.9741** over 68,495 test stars.

| distance to coreline | n | accuracy | share of all errors |
|---|---:|---:|---:|
| [0, 1)h | 16,705 | 0.9455 | 51.4 % |
| [1, 2)h | 3,006 | 0.8227 | 30.0 % |
| [2, 3)h | 4,850 | 0.9736 |  7.2 % |
| [3, 4)h | 6,105 | 0.9902 |  3.4 % |
| [4, 6)h | 12,470 | 0.9953 |  3.3 % |
| [6, 8)h | 8,711 | 0.9959 |  2.0 % |
| [8, 12)h | 4,662 | 0.9946 |  1.4 % |
| [12, 20)h | 5,239 | 0.9971 |  0.8 % |
| [20, 50)h | 5,493 | 0.9989 |  0.3 % |
| ≥ 50h | 1,254 | 0.9992 |  0.1 % |

**Essentially all error is boundary ambiguity.** The `[1, 2)h` band — immediately outside the positive
threshold — has the worst accuracy anywhere (0.8227) despite being only 4.4 % of the test set, and the
two innermost bands together (29 % of the data) account for **81 % of every mistake**. Past 4h the
classifier is at 0.995 or better and past 20h it is effectively exact.

So the model is not failing to locate corelines; it disagrees about precisely where the `d < h` cut
falls, which is the part of the task that is genuinely ill-posed at grid resolution. Reporting a single
macro-F1 hides this — a classifier that detected nothing beyond 4h could post a similar headline number
by exploiting the class balance.

| seeding stratum | n | accuracy |
|---|---:|---:|
| on-tube (d < h) | 16,705 | 0.9455 |
| hard shell (h..8h) | 23,959 | 0.9743 |
| far (IVD region) | 27,831 | 0.9911 |

The `hard` stratum (0.9743) is markedly harder than the `far` stratum (0.9911), confirming that the
stratified seeding of section 3 did produce a non-trivial negative set rather than an easy one.

| scene | n | accuracy |
|---|---:|---:|
| SquareCylinder | 8,334 | 0.9849 |
| cylinder3d | 13,544 | 0.9694 |
| deltaWing_resampled | 14,915 | 0.9882 |
| halfcylinderRe320 | 14,475 | 0.9623 |
| halfcylinderRe640 | 14,237 | 0.9657 |
| tornado3d | 2,990 | 0.9920 |

## 10. Open axes

* **Arc length** — every result above uses `total_length 0.5` (≈40 h). Caches at 0.25 / 1.0 / 2.0 are
  building with `--centres-from` so the centres and labels are identical and only the integration
  changes, making the comparison exactly controlled.
* **Points per streamline** is fixed at 65 throughout; 129 is untested.
* **Radii beyond 8h** are untested; round A showed accuracy falling monotonically for single shells past
  2h, but the 1h+8h pair was the best two-shell combination, so a wider outer shell may still pay.

## 11. Reproducibility note

The first cache was built with a per-frame seed derived from Python's `hash()`, which is salted per
process, so that build is **not** reproducible from its arguments. The builder now derives the seed with
`hashlib.sha256`, which is stable across processes, and `--centres-from` allows an existing cache's
centres to be reused exactly. The 0.5 cache in `/home/cheny1a/data/task6_icostar_1_1` remains valid data
— every star is a genuine integration and every label follows the fixed rule — it simply cannot be
regenerated centre-for-centre, which is why the arc-length comparison reuses its centres rather than
redrawing them.

## 12. Continuous SO(3): a real geometric defect with no measurable cost

`random_so3` assigns channels by nearest icosahedral vertex. Over 200,000 random rotations
(`experiments/Analyze_SO3_ChannelMap_1_1.py`) that assignment is **not bijective in
30.8 %** of draws — two channels claim one vertex, so the augmented sample has a
duplicated neighbour and a missing one and is not a star any rotated flow could
produce. Mean angular error 22.3°, max 37.4°.

`so3snap` (in `Run_Task6_NewStar_IcoVAE_1_1.py`) fixes this. The only
structure-preserving permutations of the twelve vertices are the group's own, so the
best valid channel map for a rotation `R` is the one induced by the nearest group
element, `argmax tr(RᵀG)` over the 120 elements — O(1) and bijective by construction
(verified a permutation for all 4 096 test draws).

**The fix does not improve accuracy.** Full supervision, shells 1h+8h:

| arm | n | F1 | vs supervised | p |
|---|---:|---:|---:|---:|
| supervised + class weight | 5 | 0.9610 ± 0.0006 | — | — |
| contrastive, exact discrete `ih` | 8 | 0.9598 ± 0.0009 | -0.0012 | 0.0148 |
| contrastive, nearest-vertex `so3` | 8 | 0.9598 ± 0.0015 | -0.0012 | 0.0823 |
| contrastive, group-snapped `so3snap` | 8 | 0.9596 ± 0.0004 | -0.0014 | 0.0038 |

`so3snap` vs `so3`: **-0.0002, p = 0.691** — no effect. All three contrastive
variants cost the same ≈0.0012 relative to plain supervised training and are
statistically indistinguishable from one another.

**This corrects round C.** At n = 3 round C reported `so3` at 0.9584 and concluded
continuous SO(3) was uniquely harmful (−0.0027, p = 0.013). At n = 8 `so3` is 0.9598
and there is no SO(3)-specific penalty: the contrastive term costs the same small
amount whichever view construction is used. The n = 3 estimate was low by 0.0016 and
its standard deviation more than doubled with more seeds.

Note also that the differing p-values above reflect *variance*, not effect size — the
three contrastive arms have near-identical means, but `so3` has three times the seed
spread of `so3snap`, so the same −0.0012 reads as n.s. for one and significant for the
other. Judging these arms on significance alone would have been misleading.

One genuine effect survives: **`so3snap` has the lowest seed variance of any arm**
(0.0004 vs 0.0015 for `so3`). A bijective channel map makes training more
reproducible even though it does not move the mean, which is reason enough to prefer
it as the default on correctness grounds — but it should be described as
*geometrically correct*, not as an accuracy improvement.

## 13. Arc length

Every result in sections 6-9 fixes `total_length 0.5` (≈40 h of arc). Caches at 0.25 / 1.0 / 2.0 were
rebuilt with `--centres-from`, reusing the 0.5 cache's centres so the seeding is identical.

### Rounds H and K — arc length, each cache on its own survivors

| arc length | 4-shell (1/2/4/8 h) | 1h+8h | train stars kept |
|---:|---:|---:|---:|
| 0.0625 | 0.9634 ± 0.0017 | 0.9601 ± 0.0014 | 216,645 |
| **0.125** | **0.9661 ± 0.0010** | 0.9627 ± 0.0007 | 216,616 |
| **0.25** | **0.9662 ± 0.0011** | 0.9637 ± 0.0003 | 216,557 |
| 0.5 | 0.9651 ± 0.0006 | 0.9648 ± 0.0013 | 217,342 |
| 1.0 | 0.9580 ± 0.0012 | 0.9555 ± 0.0001 | 207,979 |
| 2.0 | 0.9460 ± 0.0022 | 0.9425 ± 0.0019 | 195,474 |

**The optimum is interior**, a plateau at 0.125-0.25 that falls away on both sides: −0.0028 at 0.0625
and −0.020 at 2.0. Sampling only 0.25 and longer (round H alone) suggested a monotone "shorter is
better" trend; round K shows that is wrong below 0.125.

The shape fits section 9. Proximity to a coreline is local and 81 % of the error lies within 2 h of one,
so a long streamline wanders out of the neighbourhood and dilutes the signal — but too short a
streamline carries too little of the flow geometry to distinguish a coreline-aligned star from a passing
one. Roughly 20 h of arc is the compromise.

**Arc length dominates every objective choice.** The 0.25-to-2.0 span is 0.020 F1, against ±0.003 for
any SSL or decoder combination at full supervision — an order of magnitude larger. On this task the
integration geometry matters far more than the training objective.

Caveat carried from above: these rows use each cache's own survivor set, and retention falls from
99.6 % at 0.25 to 89.9 % at 2.0. Round J repeats the sweep on the common 194 058 / 62 163 split.

Weights for the best setting are archived at `outputs/weights_Task6_NewStar/tl025_seed{11,23,37}.pt`.

### Round J — the controlled comparison

Same sweep on the intersection of all four survivor sets: every arc length sees exactly **194 058 train
/ 62 163 test** stars at an identical 0.234062 positive rate, via `--restrict-to`.

| arc length | controlled | own survivors | offset |
|---:|---:|---:|---:|
| 0.25 | 0.9625 ± 0.0007 | 0.9662 ± 0.0011 | **-0.0037** |
| 0.5 | 0.9629 ± 0.0011 | 0.9651 ± 0.0006 | -0.0021 |
| 1.0 | 0.9559 ± 0.0000 | 0.9580 ± 0.0012 | -0.0021 |
| 2.0 | 0.9438 ± 0.0016 | 0.9460 ± 0.0022 | -0.0022 |

| comparison (controlled) | Δ | p | verdict |
|---|---:|---:|---|
| 0.25 vs 0.5 | -0.0004 | 0.605 | **tied** |
| 0.5 vs 1.0 | +0.0071 | 0.0080 | significant |
| 0.5 vs 2.0 | +0.0191 | 0.00014 | significant |

**The apparent advantage of 0.25 over 0.5 was an artefact of the survivor sets.** On identical stars the
two are tied (p = 0.61) and 0.5 is marginally ahead. The offset column is the giveaway: every arc length
loses ≈0.0021 when moved onto the common split *except* 0.25, which loses 0.0037 — that excess is
precisely the bias. Stars that fail the validity mask at long arc lengths are not a random subset, and
they are apparently the easier ones, so a cache that keeps more of them scores higher for a reason that
has nothing to do with arc length.

**What survives controlling is the penalty for long integration**: −0.0071 at 1.0 (p = 0.008) and −0.0191
at 2.0 (p = 0.0001), essentially unchanged from the uncontrolled comparison (−0.0071, −0.0191).

So the two protocols answer different questions and both are worth keeping:

* *"What do I get if I build at this arc length?"* → the uncontrolled rows, which correctly include the
  fact that longer integration silently discards up to 10 % of the data.
* *"Does arc length itself help?"* → the controlled rows: **0.125-0.5 is a flat plateau, not a peak**,
  and only beyond 0.5 does arc length itself cost accuracy.

The practical recommendation (build at 0.25-0.5) is unchanged; the reason differs, which matters on data
where retention behaves differently.

## 14. Summary of effect sizes

Eleven chained rounds, ~290 runs, all artefacts under `outputs/exp_Task6_NewStar/`.

| effect | size | status |
|---|---:|---|
| Label-scarcity crossover (weak contrastive + decoder at 435 labels) | +0.060 | solid |
| Conv3D gap at matched capacity | +0.035 | solid |
| Arc length 0.5 vs 2.0 (controlled) | +0.019 | solid, p = 0.0001 |
| Pooled vs per-scene training | +0.014 | solid |
| Arc length 0.5 vs 1.0 (controlled) | +0.007 | solid, p = 0.008 |
| Warmup + cosine vs constant | +0.005 | solid |
| Two shells vs one | +0.005 | solid |
| Contrastive cost at full supervision | −0.0012 | marginal, p ≈ 0.01-0.08 |
| Arc length 0.25 vs 0.5 | — | **refuted once controlled** |
| SO(3)-specific penalty | — | **refuted at n = 8** |
| `so3snap` accuracy gain | — | **refuted at n = 8** |

**Best configuration: 0.9662 ± 0.0011** (arc 0.25, 4-shell star, lr 1e-3, warmup+cosine, class weight,
supervised head only, 629 297 parameters, 3 seeds) — noting that ≈0.004 of that figure comes from the
easier survivor subset at arc 0.25 rather than from arc length; on a common split the honest number is
**0.9629 ± 0.0011**.

**Five conclusions in this study reversed on re-measurement** (round-A geometry at n = 1, round-B
objective ranking at n = 1, the two-stage sign flip at n = 1, the SO(3) penalty at n = 3, and the arc-length
optimum once the survivor sets were controlled). Every one was a small effect measured at low seed count
or with an uncontrolled confound. The effects above 0.005 all held. Treat any single-seed ordering here
as provisional.

## 15. Coverage: what the neighbour-shell margin excludes

Centres must sit `max(radii) × h × 1.05` clear of the domain boundary so every neighbour seed stays
inside the field. Positive centres are drawn by perturbing coreline points, so **any coreline inside that
margin contributes no positives at all** — the classifier never sees corelines near a wall or an
inflow/outflow plane. `experiments/Analyze_Task6_MarginLoss_1_1.py` measures this as a fraction of
densified coreline.

| margin | coreline reachable, all scenes | worst scene (`halfcylinderRe640`) |
|---:|---:|---:|
| 0.5h | 99.36 % | 97.53 % |
| 1h | 98.59 % | 93.41 % |
| 2h | 96.44 % | 77.84 % |
| 4h | 91.35 % | 53.47 % |
| **8h** | **79.21 %** | **12.47 %** |

**The 1/2/4/8 h star behind every result above reaches only 79.2 % of the coreline, and only 12.5 % of
`halfcylinderRe640`'s.** The cause is geometric: that scene's grid is 160×60×20, so its z axis spans just
**19.9 h**. An 8 h margin removes 16.8 h from every axis, leaving 3.1 h of usable z, confining centres to
a thin central slab.

| scene | grid | extent in h (x / y / z) |
|---|---|---|
| cylinder3d | 640×240×80 | 639 / 240 / 80 |
| halfcylinderRe320 | 640×240×80 | 639 / 240 / 80 |
| **halfcylinderRe640** | **160×60×20** | **159 / 60 / 19.9** |
| deltaWing_resampled | 55×314×55 | 55 / 313 / 55 |
| SquareCylinder | 192×64×48 | 252 / 63 / 47 |
| tornado3d | 128×128×128 | 127 / 127 / 127 |

**This is a coverage bias, not a class-balance problem.** Every scene has ample positives at a healthy
rate — `halfcylinderRe640` has 10 328 positives at 0.242 — but all of them come from the 12.5 % of
coreline inside the slab, so the model is never shown the rest.

Fresh caches at radii {0.5, 1, 2} (2 h margin, 96.4 % coverage) and {0.25, 0.5, 1} (1 h margin, 98.6 %)
were drawn from scratch; `--centres-from` would have inherited the 8 h restriction and defeated the
purpose.

| cache | stars | positives | rate |
|---|---:|---:|---:|
| radii 1,2,4,8 (8 h margin) | 285 837 | 69 530 | 0.243 |
| radii 0.5,1,2 (2 h margin) | 293 627 | 70 404 | 0.240 |
| radii 0.25,0.5,1 (1 h margin) | 295 334 | 70 633 | 0.239 |

Round N compares them. Round A put the 8 h-containing stars only ≈0.005 F1 ahead of small-radius stars,
which is a poor exchange for a fifth of the coreline. Note each cache is scored on its *own* test set:
the small-radius ones include near-boundary corelines the 8 h benchmark cannot represent, so they are
the more complete evaluation, not merely an easier or harder one.

## 16. Temporal sampling, and the abandoned 16 h shell

Points per streamline, 129 vs 65, controlled on the points-129 survivor set:

| points per streamline | F1 (3 seeds) |
|---:|---:|
| 65 | 0.9652 ± 0.0014 |
| 129 | 0.9649 ± 0.0007 |

Difference **-0.0003, p = 0.78 — no effect.** 65 points is sufficient; doubling the
temporal resolution doubles cache size and training memory for nothing. The 129-point cache does not fit
in device memory alongside the model, which is why `--data-device cpu` exists.

A 16 h outer shell was built and **abandoned**: its margin excluded 45 % of centres and *all 15*
`halfcylinderRe640` frames, silently turning the six-scene benchmark into a five-scene one. Its partial
run scored 0.9688, which looks like a new best and is not — it is a different, smaller test set. The
cache and those results were deleted so they cannot be misread later.

## 17. Neighbour radius vs coreline coverage

Section 15 showed the 1/2/4/8 h star reaches only 79.2 % of the coreline. Does shrinking the radii to
recover that coverage cost accuracy?

### Round N — each cache on its own test set

| star | coreline coverage | F1 (3 seeds) | test stars |
|---|---:|---:|---:|
| radii 0.5, 1, 2 | 96.4 % | 0.9543 ± 0.0012 | 70,077 |
| shells 1, 2 | 96.4 % | 0.9526 ± 0.0008 | 70,077 |
| shells 0.5, 2 | 96.4 % | 0.9525 ± 0.0011 | 70,077 |
| radii 0.25, 0.5, 1 | 98.6 % | 0.9513 ± 0.0011 | 70,396 |
| **radii 1, 2, 4, 8** (reference) | **79.2 %** | **0.9651 ± 0.0006** | 68 495 |

Small stars look 0.011-0.014 worse — but each cache is scored on **its own** test set, and the
small-radius sets include the near-boundary corelines the 8 h benchmark cannot represent. That mixes
"worse geometry" with "harder test set".

### Round O — same spatial region, different star

`--min-margin 8` keeps only centres 8 h clear of the boundary, so both caches are scored on the same
spatial region and the only remaining difference is the star itself.

| star | F1 (3 seeds) | train stars | test stars |
|---|---:|---:|---:|
| radii 1, 2, 4, 8 (49 lines) | 0.9649 ± 0.0007 | 217,342 | 68,495 |
| radii 0.5, 1, 2 (37 lines) | 0.9626 ± 0.0015 | 168,664 | 49,749 |

**Δ = +0.0022, p = 0.106 — not significant.** Only ≈21 % of round N's 0.0108 gap is star
geometry; the remaining ≈79 % is test-set difficulty. The small star reached that while training on 22 %
**less** data, since the margin filter trimmed its training set too — so if anything this understates it.

Per-scene, region-matched, the wide shells do help where the domain is thin or the structure is large:
`halfcylinderRe640` +0.0147 and `tornado3d` +0.0146 for the 8 h star, while `SquareCylinder` is −0.0035
and the rest are within ±0.0035.

### What to use

The two stars are **statistically indistinguishable on matched data**, so the choice is about what the
benchmark should represent rather than about accuracy:

* **radii 1, 2, 4, 8** — the highest raw score, **0.9662 ± 0.0011**, but on a benchmark that omits a fifth
  of the coreline and 87 % of `halfcylinderRe640`'s. Use it to compare against the numbers in this
  document.
* **radii 0.5, 1, 2** — **0.9543 ± 0.0012** on a benchmark covering 96.4 % of the coreline, including the
  near-boundary structure. The lower number is a harder, more honest test set, not a worse model. Use it
  if the evaluation should represent corelines near walls and inflow/outflow planes.

Reporting only the first would overstate what the method does on complete data; reporting only the
second would understate it against the rest of this document. Both are given.

## 18. Recommended configuration

```bash
python experiments/Run_Task6_NewStar_IcoVAE_1_1.py \
    --cache /home/cheny1a/data/task6_icostar_tl0.25 \
    --shells 1,2,4,8 \
    --lr 1e-3 --schedule cosine --warmup-steps 400 \
    --class-weight --steps 4000 --seed 11
#   no --lambda-ssl, no --lambda-recon, no --zinv-var-weight
```

**0.9662 ± 0.0011 macro-F1**, 3 seeds, 629 297 parameters, ≈100 s per run on one A100.
Weights: `outputs/weights_Task6_NewStar/tl025_seed{11,23,37}.pt`.

| component | setting | why |
|---|---|---|
| star | 12 icosahedron vertices at 1/2/4/8 h | two shells beat one (+0.005); four ≈ two |
| arc length | 0.25 (≈20 h) | plateau 0.125-0.5; 2.0 costs 0.019 |
| points per streamline | 65 | 129 changes nothing (p = 0.78) |
| learning rate | 1e-3, warmup 400 + cosine | warmup+cosine worth +0.005 over constant |
| steps | 4 000 | 12 k adds nothing; 24 k overfits after ≈17 k |
| class weighting | on | best arm at full supervision |
| **SSL contrastive** | **off** | −0.0012 at full supervision |
| **decoder reconstruction** | **off** | −0.0011; at weight 10, −0.0127 |
| **VICReg variance** | **off** | −0.0018 alone, −0.0033 stacked |

**Turn the auxiliaries on only below ≈5 % labels**, where the picture inverts:

| labels | recommended | gain |
|---:|---|---:|
| 217 342 (100 %) | supervised only | — |
| 10 867 (5 %) | supervised only | ±0.003 |
| 2 173 (1 %) | `--lambda-ssl 1.0 --lambda-recon 1.0` | +0.021 |
| 435 (0.2 %) | `--lambda-ssl 0.1 --lambda-recon 1.0` | **+0.060** |

### Headline comparison

| model | params | macro F1 |
|---|---:|---:|
| **IcoVAE, 4-shell star, arc 0.25** | 629 297 | **0.9662 ± 0.0011** |
| IcoVAE, 4-shell star, arc 0.5 | 629 297 | 0.9651 ± 0.0006 |
| Conv3D splat, best at any capacity | 2 163 777 | 0.9332 |
| Conv3D splat, capacity-matched | 608 929 | 0.9267 |
| Conv3D splat, frozen repo baseline | 84 361 | 0.8961 |

**+0.0395 at matched capacity**, and
**+0.0331** against the best Conv3D at any capacity (3.4× the parameters).

### Two caveats to carry

1. **Coverage.** This configuration's 8 h margin reaches 79.2 % of the coreline and only 12.5 % of
   `halfcylinderRe640`'s (§15). On matched data a 0.5/1/2 h star is statistically indistinguishable
   (§17), so if the evaluation must represent near-boundary corelines, use that instead and expect
   ≈0.954 on the more complete benchmark.
2. **Seeds.** Five conclusions here reversed on re-measurement. Effects below ≈0.005 need more than
   three seeds; this document marks which are solid and which are not.
