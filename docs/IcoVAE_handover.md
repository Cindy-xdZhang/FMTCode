# IcoVAE — handover

A 3D icosahedral equivariant SIREN-VAE over pathline/streamline stars, applied to
two tasks in this repo. This document is a map: what was built, where it lives,
how the setup differs from the original repo, and what is measured vs still open.
The detailed results are in two companion documents.

| | detail document |
|---|---|
| **Task 1** — unsupervised vortex clustering | [`3d_kmeans_ico.md`](3d_kmeans_ico.md) |
| **Task 6** — supervised coreline classification | [`exp_Task6_IcoVAE.md`](exp_Task6_IcoVAE.md) |
| 2D original this ports from | [`2d_kmeans_d8.md`](2d_kmeans_d8.md) |

Everything here is in **new files**; no pre-existing repo file was modified.

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
| `experiments/Build_Task6_ExactStar_1_1.py` | **traced** stars (use this) |
| `experiments/Build_Task6_MultiShell_1_1.py` | multi-shell traced stars |
| `experiments/Build_Task6_Ico_Cache_1_1.py` | borrowed-neighbour stars — **superseded, see §4.2** |
| `experiments/Run_Task6_IcoVAE_1_1.py` | supervised training, SSL auxiliaries, warmup/cosine, per-scene mode |

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

## 5. Task 6 — what is usable, and why only one scene

Tracing neighbours needs the velocity field. Each candidate was verified by
interpolating it along the stored streamlines and measuring the cosine against
the tangent (true match ≈ 1.0, unrelated field ≈ 0.0):

| scene | best candidate | median cos | usable |
|---|---|---|---|
| **deltaWing_resampled** | `deltaWing_mag0_3reesampled` | **0.9985** | **yes** |
| cylinder3d | `halfcylinderRe160` | 0.8653 | no |
| halfcylinderRe640 | `halfcylinderRe640` | 0.8657 | no |
| halfcylinderRe320 | `halfcylinderRe6400` | 0.7662 | no |

`halfcylinderRe160Resampled` and `halfcylinderRe640resampled` were also tested
with correct **time-value** lookup and reach only 0.863 / 0.862 — right
simulation family, wrong variant. `SquareCylinder` and `tornado3d` ship no
`sample_streamlines.npy` at all, and `tornado3d` was the split's only genuinely
held-out scene.

**If you have the true fields for the other scenes, that is the single highest-value
thing to add** — it would take Task 6 from one scene to four and restore the
cross-scene generalisation the split was designed to measure.

### Result (deltaWing, 200k train / 100k test, 7.8% positive)

| arm (5 seeds) | pos F1 | macro F1 | AP |
|---|---|---|---|
| **reconstruction auxiliary** | **0.8014 ± 0.0026** | 0.8924 | 0.7941 |
| contrastive + reconstruction | 0.8001 ± 0.0058 | 0.8918 | 0.7729 |
| supervised only | 0.7888 ± 0.0053 | 0.8856 | **0.8113** |
| contrastive only | 0.7868 ± 0.0042 | 0.8844 | 0.8110 |

Reconstruction: **+0.0126, p = 0.003**. Contrastive: **−0.0020, p = 0.52**.

**The contrastive term — which is the whole method on Task 1 — does nothing
here.** Plausibly because invariance to `I_h` deliberately discards orientation,
and orientation is what distinguishes a coreline-aligned star from a passing one;
whereas reconstruction forces the latent to keep geometry the label does not
supply.

**Caveat worth carrying:** AP *disagrees* with F1 — the supervised arm has the
best average precision (0.8113) and the worst F1. Reconstruction improves
calibration near the 0.5 threshold rather than the ranking. If the downstream use
ranks candidates, the gain may not transfer.

---

## 6. What is tested, and what is open

### Tested
| | Task 1 | Task 6 |
|---|---|---|
| views: discrete `I_h` / continuous SO(3) / both | ✓ | ✓ |
| reconstruction on/off, weight sweep | ✓ | ✓ (peak at λ=30; 100+ collapses) |
| VICReg variance floor | ✓ (interacts with everything) | ✓ (flat) |
| learning rate | ✓ (5e-4 interior optimum) | ✓ (warmup **essential** at 1e-3: without it, divergence) |
| warmup / cosine | — | ✓ (warmup +0.63, cosine +0.011) |
| latent size, batch, head type, weight decay | ✓ | ✓ — **all flat** |
| clusterer & read-out rules (7 × 4 × 5) | ✓ | n/a |
| neighbour radius | n/a (traced at fixed offset) | ✓ **peak at 8–9h** |
| per-scene vs joint models | n/a | ✓ |

### Running / queued
* **Round E** — 3× training data (60k seeds/frame), streamline arclength 0.25 / 1.0 vs 0.5, 129 samples vs 65.
* **Round F** — **multi-shell stars**: centre + 12 at *two* radii (4+8h, 2+8h, 6+10h, three-shell 4+8+12h). Still exactly icosahedrally symmetric — the group permutes both shells by the same permutation, verified to `1.33e-15`. `apply_group_shells` and a configurable line count are implemented and tested.

### Open / suggested next
1. **Get the remaining Task 6 fields** (§5) — highest value by far.
2. **Seeds.** Most Task 1 single-factor cells are one seed, and two conclusions
   in this study reversed on re-measurement. Task 6's headline is 5 seeds/arm.
3. **Exact continuous SO(3) views.** The current `random_so3` rotates vectors
   exactly but assigns channels by *nearest vertex*; below ~29.5° that leaves
   channels unchanged while vectors rotate, which is not a valid star. A graph
   formulation carrying each arm's direction as a node feature would fix it.
4. **Task 1 radius.** Task 6 found radius worth ±0.14; Task 1 uses a single fixed
   offset (`0.5 × min grid spacing`) and has never swept it.
5. **AP-vs-F1 divergence on Task 6** (§5) — worth resolving before the
   reconstruction gain is relied on.

---

## 7. Reproducing

```bash
# Task 1 — cache (no split), then one run per dataset
python experiments/Build_Task1_Ico_Cache_1_1.py \
  --datasets halfcylinderRe160_eth,halfcylinderRe640_eth,halfcylinderRe6400_eth
python experiments/Run_Task1_IcoVAE_1_1.py --dataset halfcylinderRe160_eth \
  --aug both --lambda-recon 1 --zinv-var-weight 0 --clusterer gmm_tied \
  --save-latents outputs/exp_Task1_IcoVAE_latents
python experiments/Analyze_Task1_IcoVAE_Clusterer_1_1.py --clusterers kmeans,gmm_tied

# Task 6 — traced stars at the best radius, then the winning recipe
python experiments/Build_Task6_ExactStar_1_1.py --radius-scale 8 --count 20000
python experiments/Run_Task6_IcoVAE_1_1.py --cache outputs/exp_Task6_Exact_r8 \
  --schedule cosine --warmup-steps 2000 --lr 1e-3 --lambda-recon 30 --lambda-ssl 0

python tests/test_icosahedral_group_3d.py     # 7 checks, all should pass
```

Environment: `conda activate pyflowvis` (torch 2.12 / cu126, sklearn 1.8,
netCDF4, numba, typeguard — the last three were installed for this work).
Data: `/home/cheny1a/data/flowData3D/`, Task 6 under
`Task6_CorelineDataset_2.7_share/`.
