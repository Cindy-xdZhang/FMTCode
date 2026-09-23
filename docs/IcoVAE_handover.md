# IcoVAE — handover

A 3D icosahedral equivariant SIREN-VAE over pathline/streamline stars, applied to
two tasks in this repo. This document is a map: what was built, where it lives,
how the setup differs from the original repo, and what is measured vs still open.
The detailed results are in two companion documents.

> **To reproduce Task 6, follow
> [`Task6_Reproduction_Guide.md`](Task6_Reproduction_Guide.md).** It has the environment, the
> dataset build, every hyper-parameter and the expected numbers. This document is the method
> overview; the guide is the runbook.

| | detail document |
|---|---|
| **Task 6 — how to run and reproduce** | **[`Task6_Reproduction_Guide.md`](Task6_Reproduction_Guide.md)** |
| **Task 1** — unsupervised vortex clustering | [`3d_kmeans_ico.md`](3d_kmeans_ico.md) |
| **Task 6** — supervised coreline classification | [`exp_Task6_NewData_IcoStar.md`](exp_Task6_NewData_IcoStar.md) |
| Task 6, previous data snapshot (superseded) | [`exp_Task6_IcoVAE.md`](exp_Task6_IcoVAE.md) |
| 2D original this ports from | [`2d_kmeans_d8.md`](2d_kmeans_d8.md) |

All code described here is in **new files**; no pre-existing repo source file was
modified. This document and its companions are the only edited files, and they were
authored as part of this work.

---

## 1. The method in one page

A primitive is a **13-line star**: a centre pathline plus twelve neighbours
seeded at the vertices of a regular icosahedron around it. The relevant symmetry
group is the full icosahedral group

* `I` — 60 proper rotations (≅ A₅)
* `I_h = I × {±E}` — **120 elements** with reflections

Every element permutes the twelve vertices, so a group element acts on a star by
rotating all vectors and permuting the neighbour channels. That action is
**exactly equivalent to having traced the star in a rotated flow** — verified to
`8.88e-16` over all 120 elements (`tests/test_icosahedral_group_3d.py`).

Why icosahedral rather than the octahedral (6-neighbour, 48-element) star used
elsewhere in the repo: over 20,000 random rotations the mean angle to the nearest
group element is **29.5° (icosahedral, 60)** against **40.8° (octahedral, 24)**,
and twelve points sample the sphere far more evenly than six.

The model is `[N,13,T,3] → encoder → z = [z_inv | z_eq] → SIREN decoder`, trained
with a contrastive term over group views of `z_inv`, a VICReg variance/covariance
floor, KL, and reconstruction. Only `z_inv` is used downstream.

**The star is the method.** On Task 6, replacing geometrically-arbitrary
neighbours with correctly traced ones is worth **+0.093 F1** — more than every
hyper-parameter in the study combined. Any reuse of this code should treat star
construction as the first-order concern.

---

## 2. Code map

### Shared
| file | what |
|---|---|
| `FMT_Utils/IcosahedralGroup_3D.py` | 12 vertices, 120 matrices + channel permutations, `apply_group`, `apply_group_shells`, continuous `random_so3` |
| `FMT_Utils/IcoVAE_3D.py` | model, exactly-equivariant normaliser, objective terms, view construction; line count configurable (13 / 25 / 37) |
| `FMT_Utils/ClusterReadout_3D.py` | 7 clusterers, 4 internal criteria, the macro-F1 metric |
| `FMT_Utils/StreamlineStar_3D.py` | RK4 arclength streamline tracer; `icosahedral_star`, `multi_shell_star` |
| `FMT_Utils/IcosahedralPrimitives_3D.py` | 13-line pathline integrator (Task 1) |
| `tests/test_icosahedral_group_3d.py` | 7 checks: group order, closure, vertex permutation, action-vs-retrace, SO(3) properness, covering |

### Task 1 (unsupervised)
| file | what |
|---|---|
| `experiments/Build_Task1_Ico_Cache_1_1.py` | cache builder, **no split** |
| `experiments/Run_Task1_IcoVAE_1_1.py` | training, full F1-curve recording, latent saving |
| `experiments/Analyze_Task1_IcoVAE_Clusterer_1_1.py` | offline clusterer × restarts × epoch-rule sweep |
| `experiments/Analyze_Task1_IcoVAE_Readout_1_1.py` | partition × epoch rule study |
| `experiments/Run_Task1_Baselines_Full_1_1.py` | FMT and Raw baselines on the same full data |

### Task 6 (supervised)
| file | what |
|---|---|
| `experiments/Build_Task6_New_IcoStar_1_1.py` | **current builder** — stratified centres, 12 vertices at 1/2/4/8 h, official `task6_fields` API |
| `experiments/Run_Task6_NewStar_IcoVAE_1_1.py` | **current trainer** — shell slicing, SSL/decoder objectives, two-stage pretraining, label-fraction |
| `experiments/Run_Task6_NewStar_Conv3D_1_1.py` | Conv3D voxel baseline, width-configurable for capacity matching |
| `experiments/Analyze_Task6_NewStar_Distance_1_1.py` | accuracy resolved against distance-to-coreline and seeding stratum |
| `experiments/Build_Task6_ExactStar_1_1.py` | previous-snapshot builder — **superseded**, that data no longer exists |
| `experiments/Build_Task6_MultiShell_1_1.py` | previous-snapshot multi-shell builder — superseded |
| `experiments/Build_Task6_Ico_Cache_1_1.py` | borrowed-neighbour stars — **withdrawn, see §4.2** |
| `experiments/Run_Task6_IcoVAE_1_1.py` | previous-snapshot trainer |

---

## 3. Task 1 — setup differences from the original repo

These are deliberate and they change what the numbers mean, so they matter for
anyone comparing against the original.

**1. No split.** The original divides timeslices into development (VAE train
0-5, checkpoint 6-7, k-means 0-7, calibration 8-9) and four confirmation slices.
Here **every timeslice is training data and every timeslice is scored**. The task
is unsupervised, so a split buys nothing and costs a third of the data — and the
earlier octahedral study failed precisely at the development→confirmation
boundary. Practical consequence: **published split-protocol numbers are not a
valid reference.** Recomputed on full data, FMT scores **0.8007**, not the
published 0.5247 — the split alone was costing it **+0.234**.

**2. Five datasets, not four.** halfcylinderRe160 / Re640 / Re6400 / tangaroa /
deltaWing, 14 timeslices each, ~50,000 primitives per flow.

**3. Icosahedral, not octahedral.** 13-line star, `I_h` (120) rather than 7-line,
`O_h` (48).

**4. GMM-tied read-out, not k-means.** Following the updated 2D study. Worth
**+0.051**, and it *changed conclusions* — see §4.1.

**5. The IVD reference is a metric only** and never enters training, checkpoint
selection or read-out selection.

### Result

| arm (5 flows, mean) | F1 |
|---|---|
| best IcoVAE: all three SSL tasks, variance floor **off** | **0.8555** |
| IcoVAE, discrete views only | 0.8430–0.8470 |
| **FMT, same full-data setup** | **0.8007** |
| untrained encoder (lr = 0) | 0.6594 |

**+0.055 over FMT**, winning on all five flows. Training is worth **+0.196**.

---

## 4. Two results that reversed on re-measurement

Both are recorded in the detail docs; flagging them here because they are the
kind of error that is easy to repeat.

### 4.1 The clusterer changed the science, not just the score

Measured through k-means, the untrained baseline read 0.4603 and training looked
worth +0.36. Re-read through GMM-tied on the *same saved latents*, the untrained
encoder scores **0.6594** and training is worth **+0.196**. Half of what was
attributed to the objective was the read-out mishandling an untrained latent.
Two specific consequences:

* On deltaWing an **untrained** encoder scores 0.8787 — above FMT's 0.8637. Most
  of that flow's score is the primitive and the metric, not learning.
* tangaroa looked like a structural loss (−0.032, reproducible across five
  configs). Under GMM-tied it ties FMT, with **no retraining**. The binding
  constraint was the clusterer's metric.

**Lesson for further work: fix the read-out before interpreting any ablation.**

### 4.2 Neighbours must be geometrically correct

Task 6 ships precomputed streamlines at scattered seeds, so an early version
*borrowed* the twelve neighbours from nearby seeds. That leaves a 12–15° angular
error and, below 2h, **half the stars reuse the same streamline in several
channels** — the channel permutation carries no geometric meaning and the group
augmentation acts on an arbitrary ordering.

| construction | supervised F1 |
|---|---|
| **traced, r = 8h** | **0.7995** |
| borrowed (12 of 64 nearest, Hungarian-matched) | 0.6856 |

All borrowed-neighbour numbers are withdrawn. **An IcoVAE without a genuine
icosahedral star is not testing the method.**

---

## 5. Task 6 — the 2.7 dataset, and why the shipped bundles were not used

The previous snapshot is gone; only `Task6_CorelineDataset_2.7_share` remains, so
every number in `exp_Task6_IcoVAE.md` refers to data that no longer exists. The
provenance hunt described in earlier versions of this section is **obsolete** — the
new package ships the velocity field for all six scenes directly, which was listed
here as "the single highest-value thing to add". It is done.

**101 frames, 6 scenes, 77 train / 24 test.** `tornado3d` appears only in test, so it
measures transfer to a scene with no training data at all.

The package also ships `preintegrated/icosa_bundles_1.1` (11 GB, 1.01 M centres, 20
face-centre neighbours at 1 h). **It is not usable for a pooled all-scene
classifier**: its `IVD > 0.8 x max` centre rule yields *zero* positives in three of
six scenes (SquareCylinder, cylinder3d, halfcylinderRe320 — 620 000 centres, no
positives) and 82.7 % positives in halfcylinderRe640. The high-IVD region is not
co-located with the corelines. The README explicitly frees collaborators to re-seed
while fixing the label rule, so we re-seed and keep the label rule exactly (`d < h`
to the continuous coreline, corelines densified to `h/10` first).

Stratified seeding — 25 % on-tube, 35 % hard shell `h..8h`, 40 % from the old IVD
region — gives every scene a 0.22-0.27 positive rate.

### Result (all 6 scenes, 217 342 train / 68 495 test stars)

| model | params | macro F1 (3 seeds) |
|---|---:|---:|
| **IcoVAE, 4-shell star (1/2/4/8 h)** | 629 297 | **0.9651 ± 0.0006** |
| Conv3D splat, best anywhere (1 h) | 2 163 777 | 0.9332 |
| Conv3D splat, capacity-matched (1 h) | 608 929 | 0.9267 |
| Conv3D splat, frozen repo baseline (1 h) | 84 361 | 0.8961 |

**+0.035 at matched capacity, +0.032 against Conv3D at 3.4x the parameters**, and
Conv3D gains only +0.002 over its last 1.5x capacity step — the gap is
representational, not a capacity artefact.

Two shells beat any single shell; accuracy falls monotonically for single shells past
2 h, so a wide shell only helps *alongside* a narrow one.

**Pooling beats per-scene specialisation in four of five scenes**, and `tornado3d` —
with no training frame at all — is the best-scoring scene at **0.9895**. The model
learned a transferable coreline signature, not scene appearance.

### The SSL question, answered across label regimes

The previous snapshot's conclusion ("reconstruction helps, contrastive does nothing")
turns out to be regime-specific rather than general:

| labels | what helps |
|---|---|
| 217 342 (100 %) | nothing; continuous SO(3) *hurts*, −0.0027 at p = 0.013 |
| 10 867 (5 %) | nothing, ±0.003 |
| 2 173 (1 %) | contrastive + decoder, +0.021 |
| 435 (0.2 %) | decoder +0.047; contrastive **−0.064**; weak contrastive (w 0.1) + decoder **+0.060** |

* The contrastive collapse at 435 labels is **not** a weight artefact — it hurts at
  every weight alone, and is only useful alongside the decoder.
* **Optimal weights do not transfer across regimes**: the decoder wants w = 10 under
  scarcity, but w = 10 is the worst arm at full supervision (−0.0127). A sweep run only
  at 100 % labels recommends the wrong setting for the regime that matters.
* **Joint training beats two-stage pretraining** for every objective at both label
  fractions, and every frozen linear probe sits at or below the majority-class floor
  (0.4306) — the pretext tasks do not yield a linearly separable representation.

So on Task 6 the group action is a **regulariser under label scarcity**, not a
representation-learning objective. The headline accuracy comes from the supervised
head on the icosahedral star geometry.

### Where the errors are

Accuracy 0.9741 overall, but the `[1, 2)h` band — immediately outside the positive
threshold — sits at **0.8227**, and the innermost 2 h (29 % of the data) produces
**81 % of every mistake**. Past 4 h the classifier is at 0.995+. The residual error is
boundary ambiguity at grid resolution, not failure to detect corelines.

---

## 6. What is tested, and what is open

### Tested
| | Task 1 | Task 6 (2.7 data) |
|---|---|---|
| views: discrete `I_h` / `I` / continuous SO(3) / mixed | ✓ | ✓ |
| reconstruction on/off, weight sweep | ✓ | ✓ (optimum **inverts** with label count) |
| VICReg variance floor | ✓ (interacts with everything) | ✓ (harmful, −0.002 to −0.003) |
| learning rate | ✓ (5e-4 interior optimum) | ✓ (1e-3; 1e-4 clearly worse) |
| warmup / cosine | — | ✓ (**+0.005** over constant — larger than any objective effect) |
| latent size, batch, head type, weight decay | ✓ | partially |
| clusterer & read-out rules (7 × 4 × 5) | ✓ | n/a |
| neighbour radius / multi-shell | n/a | ✓ (1 / 2 / 4 / 8 h and combinations; two shells beat one) |
| per-scene vs joint models | n/a | ✓ (**joint wins 4 of 5 scenes**) |
| capacity-matched Conv3D baseline | — | ✓ (6-point ladder, 84 k → 2.16 M) |
| label-scarcity sweep (0.2 % → 100 %) | — | ✓ (**the decisive experiment for SSL**) |
| two-stage pretraining + linear probe | — | ✓ (loses to joint everywhere) |
| training length | ✓ | ✓ (4 k suffices; 24 k overfits after ~17 k) |
| error vs distance-to-coreline | — | ✓ |

Task 6 totals ~200 runs across eight chained rounds; artefacts in
`outputs/exp_Task6_NewStar/*.json` (each holds the full F1-vs-step curve and a
per-scene breakdown) and weights in `outputs/weights_Task6_NewStar/`.

### Open / suggested next
1. **Seeds.** Most Task 1 single-factor cells are one seed. On Task 6 three
   single-seed rankings reversed once seeded (round-A geometry, round-B objective,
   the two-stage sign flip), so treat any unseeded ordering as provisional.
2. **Arc length on Task 6** — running. Every Task 6 result fixes
   `total_length 0.5` (≈40 h); caches at 0.25 / 1.0 / 2.0 reuse the same centres via
   `--centres-from`, so the comparison is exactly controlled.
3. **Points per streamline** is fixed at 65 on Task 6; 129 is untested.
4. **Exact continuous SO(3) views — now measured.** `random_so3` assigns channels by nearest
   vertex, and over 200,000 rotations that map is **not bijective in 31 % of draws**
   (two channels claim one vertex, so a neighbour is duplicated and another dropped).
   A corrected view (`so3snap`, snap to the nearest group element) is implemented and
   verified bijective — but at n = 8 it changes accuracy by -0.0002 (p = 0.69), i.e. **not
   at all**. The defect is real geometrically and costless downstream. `so3snap` is
   still the better default: it has the lowest seed variance of any arm
   (0.0004 vs 0.0015). See `experiments/Analyze_SO3_ChannelMap_1_1.py`.

5. **Task 1 radius.** Task 6 found radius worth ±0.03; Task 1 uses a single fixed
   offset (`0.5 × min grid spacing`) and has never swept it.
6. **Task 1 under the Task 6 lens.** Task 1 is evaluated only in the fully
   unsupervised regime. The Task 6 label-scarcity sweep shows the contrastive term's
   value depends sharply on how much supervision is present — worth checking whether
   Task 1's conclusions are similarly regime-bound.

---

## 7. Reproducing

For Task 6 use [`Task6_Reproduction_Guide.md`](Task6_Reproduction_Guide.md) instead of the
snippet below — it is the maintained runbook, uses the recommended 1 h-margin configuration,
and lists every hyper-parameter with its justification.

```bash
# Task 1 — cache (no split), then one run per dataset
python experiments/Build_Task1_Ico_Cache_1_1.py \
  --datasets halfcylinderRe160_eth,halfcylinderRe640_eth,halfcylinderRe6400_eth
python experiments/Run_Task1_IcoVAE_1_1.py --dataset halfcylinderRe160_eth \
  --aug both --lambda-recon 1 --zinv-var-weight 0 --clusterer gmm_tied \
  --save-latents outputs/exp_Task1_IcoVAE_latents
python experiments/Analyze_Task1_IcoVAE_Clusterer_1_1.py --clusterers kmeans,gmm_tied

# Task 6 (2.7 data) — build the star cache, ~13 min wall clock over 8 shards
for I in 0 1 2 3 4 5 6 7; do
  python experiments/Build_Task6_New_IcoStar_1_1.py \
    --out /home/cheny1a/data/task6_icostar_1_1 --count 3000 --threads 8 --shards 8 --shard $I &
done; wait

# Task 6 — the best recipe (0.9651 ± 0.0006 macro F1, 3 seeds)
python experiments/Run_Task6_NewStar_IcoVAE_1_1.py --shells 1,2,4,8 --lr 1e-3 \
  --schedule cosine --warmup-steps 400 --class-weight --seed 11 --label best \
  --save-weights outputs/weights_Task6_NewStar/best_seed11.pt

# Task 6 — capacity-matched Conv3D baseline, and the error breakdown
python experiments/Run_Task6_NewStar_Conv3D_1_1.py --shells 1 --arch wide \
  --channels 32,64,128 --hidden 320 --label conv3d_609k
python experiments/Analyze_Task6_NewStar_Distance_1_1.py \
  --weights outputs/weights_Task6_NewStar/best_seed11.pt

python tests/test_icosahedral_group_3d.py     # 7 checks, all should pass
```

Environment: `conda activate pyflowvis` (torch 2.12 / cu126, sklearn 1.8, netCDF4,
numba, typeguard — the last three were installed for this work).
Data: `/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share/`; the derived
star cache is ~11 GB at `/home/cheny1a/data/task6_icostar_1_1/`.

**Reproducibility caveat.** The first Task 6 star cache was built with a per-frame
seed derived from Python's `hash()`, which is salted per process, so that particular
build cannot be regenerated centre-for-centre. The builder now uses `hashlib.sha256`
and is stable; `--centres-from` reuses an existing cache's centres exactly.
