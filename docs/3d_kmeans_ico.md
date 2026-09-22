# The icosahedral SIREN-VAE: method, ablations, and selection strategies

Pathline clustering on five 3D flows — `halfcylinderRe160`, `halfcylinderRe640`,
`halfcylinderRe6400`, `tangaroa`, `deltaWing`. The 3D counterpart of
[`2d_kmeans_d8.md`](2d_kmeans_d8.md), and the two disagree more than they agree.

Macro-F1 against IVD is a **metric only**: training, checkpoint selection and
read-out selection never touch a label.

**There is no split.** Every timeslice of every flow is training data and every
timeslice is scored. The task is unsupervised, so a development/confirmation
division buys nothing and costs data — and the earlier octahedral study
([`exp_Task1_OctVAE_3D.md`](exp_Task1_OctVAE_3D.md)) failed precisely at the
boundary between its two sets.

> **Read §9 before quoting any number.** Two results in this study reversed once
> the read-out was fixed, including the size of the effect training has. Numbers
> measured through k-means are marked as such.

---

## 1. The method

### 1.1 Signal

`build_signal_ico` → `[N, 13, T-1, 3]` per seed point:

| channel | content |
|---|---|
| 0 | centre pathline time-derivative |
| 1..12 | the twelve neighbours' displacement **relative to the centre**, time-differenced |

Neighbours sit at the vertices of a regular icosahedron at radius
`(dx·dy·dz)^⅓ × 0.5`. Both quantities are translation invariant and rotate as
vectors, which is what the group action requires.

`normalize_signal_ico` rescales with **scalar** statistics — one RMS for the
centre, a second shared across all twelve arms — so normalisation commutes with
rotation and with the channel permutation. Two deliberate differences from 2D,
both forced by 3D geometry:

* **no mean subtraction** — dividing by a scalar commutes with a rotation,
  subtracting one does not, and these flows carry a strong mean velocity;
* **magnitude clipping, not componentwise** — clamping components independently
  does not commute with a rotation; rescaling an over-long 3-vector does.

### 1.2 The group

The icosahedron is the largest vertex-transitive arrangement whose symmetry group
permutes its vertices. Verified in `tests/test_icosahedral_group_3d.py`:

| group | elements | det |
|---|---|---|
| `I` — proper rotations (≅ A₅) | **60** | +1 |
| `I_h = I × {±E}` — with reflections | **120** | ±1 |

Every element permutes the twelve vertices, the set is closed under
multiplication, and **applying a group element to a cached primitive is exactly
equivalent to having traced it in a rotated flow — max |diff| 8.88e-16 across all
120 elements.**

Why icosahedral rather than octahedral, over 20,000 random rotations:

| group | elements | mean gap to nearest | max gap |
|---|---|---|---|
| octahedral `O` | 24 | 40.8° | 62.4° |
| **icosahedral `I`** | **60** | **29.5°** | **44.1°** |

### 1.3 Architecture and objective

```
signal ─► ResConv+Transformer encoder ─► z = [ z_inv (16) | z_eq (12) ]
                                           └─► SIREN decoder ─► reconstruction
```

`L = λ_recon·L_recon + β·KL + λ_contrast·L_contrast + λ_var·V + λ_cov·C`, with
`L_contrast` the mean pairwise `1 − cos` over `[z_inv(x), z_inv(g₁x), z_inv(g₂x)]`
and `V, C` the VICReg variance hinge and covariance penalty that stop the
positive-only contrastive term collapsing to `z_inv = const`. Only `z_inv` is
clustered; `z_eq` absorbs orientation so the decoder can still reconstruct a
rotated star.

### 1.4 Views

Each batch row draws its own view. `--aug ih` uses one of the 120 discrete
elements (**exact**). `--aug so3` draws a continuous rotation: exact on the
vectors, **nearest-vertex on the channel index** — unavoidable, since a star
indexed by twelve fixed channels cannot represent a fractional rotation.
`--aug both` alternates.

---

## 2. Protocol

Clustering `k=2` on L2-normalised `z_inv`, 10 restarts, every 100 steps for 6000
steps. At each evaluation **every** restart is scored by inertia,
Davies–Bouldin, silhouette and Calinski–Harabasz, and the F1 of the partition
each rule would select is recorded.

`z_inv` is **saved at every evaluation** (float16), so clusterer, restart count
and epoch rule are swept offline from one training pass rather than by
retraining — 30 config × flow cells for the cost of 25 runs
(`experiments/Analyze_Task1_IcoVAE_Clusterer_1_1.py`).

**The untrained baseline is `lr = 0`, not zero steps** — the conv stem contains
BatchNorm whose running statistics must still calibrate. §9 shows this baseline
is the single most important control in the study.

One seed per cell. This is the largest remaining gap.

---

## 3. The read-out: a tied-covariance GMM, not k-means

Following the updated 2D study (§7.1 there), k-means is replaced by a
2-component Gaussian mixture with **tied** covariance — one pooled within-cluster
covariance estimated jointly with the partition, i.e. k-means under a
Mahalanobis metric. Five flows, 10 restarts, silhouette pick among restarts,
epoch by min Davies–Bouldin:

| clusterer | final | median | **min DB** | max silhouette | oracle | drift |
|---|---|---|---|---|---|---|
| k-means | 0.6913 | 0.7459 | 0.7940 | 0.8027 | 0.8312 | 0.0773 |
| **GMM tied** | 0.7228 | 0.7838 | **0.8447** | 0.8400 | **0.8557** | 0.0452 |
| GMM full | 0.5966 | 0.6376 | 0.6612 | 0.6813 | 0.6861 | 0.0105 |

**+0.051 over k-means at the selected epoch** and +0.025 on the oracle — it both
finds better partitions and reveals a better ceiling. `tied` beating `full` by
0.18 reproduces the 2D signature exactly: what matters is the **pooled metric**,
not model flexibility. The vortex split is well separated without being tight,
which is where the Euclidean metric prefers the wrong partition.

**This is the single most consequential change in the study**, and it is free —
no retraining, applied to latents already on disk.

### 3.1 Two 2D read-out claims that do not transfer

**The restart count is free here.** 2D found an interior optimum at 8–14 with a
cliff at 20 worth 0.083, from the winner's curse of maximising silhouette over a
larger candidate set. On Re160, 5 / 10 / 20 restarts give *identical* selected
values for both clusterers (k-means 0.8123, GMM-tied 0.8337); only the median
moves, by 0.002. The likely reason is N: 2D clustered 512–675 points, this
clusters ~50,000, so the silhouette estimate is far less noisy.

**Davies–Bouldin cannot be retired — except for one config.** 2D found that with
GMM-tied the curve converges, making the final epoch best and DB harmful
(−0.085). In 3D the curve settles onto a **suboptimal** partition for most
configs: `final` delivers 0.7228 against 0.8447 selected. The exception is
`var0` (§6), whose curve genuinely converges — drift 0.0036, and `final` 0.8253
slightly *beats* `min_db` 0.8249. Convergence to the wrong answer is not the same
phenomenon as convergence to the right one.

---

## 4. Ablation: the full objective matrix

Every cell is GMM-tied with 10 restarts, at the best label-free rule for that
config. `views` is the augmentation (`ih` = the 120 discrete elements, `so3` =
continuous rotation, `both` = alternating), `rec` the SIREN reconstruction term,
`var` the VICReg variance hinge.

| config | views | rec | var | rule | Re160 | Re640 | Re6400 | tangaroa | deltaWing | **mean** | oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ALL THREE, var off** | both | ✓ | **off** | max sil | 0.8638 | 0.8381 | **0.8326** | **0.8811** | 0.8620 | **0.8555** | 0.8657 |
| continuous SO(3) | so3 | — | on | min DB | 0.8495 | 0.8166 | 0.7907 | 0.8515 | 0.9267 | 0.8470 | 0.8537 |
| discrete + recon | ih | ✓ | on | max sil | 0.8576 | 0.8296 | 0.7884 | 0.8295 | 0.9234 | 0.8457 | 0.8547 |
| discrete + SO(3) | both | — | on | min DB | 0.8337 | 0.8192 | 0.7823 | 0.8619 | 0.9266 | 0.8447 | 0.8557 |
| discrete I_h | ih | — | on | max sil | 0.8257 | 0.8264 | 0.7918 | 0.8441 | **0.9270** | 0.8430 | 0.8552 |
| SO(3) + recon | so3 | ✓ | on | max sil | 0.8556 | 0.8177 | 0.7800 | 0.8346 | 0.9269 | 0.8430 | 0.8542 |
| ALL THREE | both | ✓ | on | max sil | 0.8562 | 0.8222 | 0.7831 | 0.8075 | 0.9247 | 0.8387 | 0.8505 |
| discrete + recon, var off | ih | ✓ | **off** | max sil | 0.8579 | **0.8456** | 0.7385 | 0.8664 | 0.8481 | 0.8313 | **0.8671** |
| **UNTRAINED (lr = 0)** | — | — | — | any | 0.4345 | 0.7598 | 0.7528 | 0.4714 | 0.8787 | **0.6594** | 0.6594 |
| *FMT 13-line, same data* | | | | — | *0.7547* | *0.7649* | *0.7613* | *0.8636* | *0.8637* | *0.8016* | |

### 4.1 Combining all three SSL tasks wins, but only with the variance floor off

`both + recon + var off` is the best configuration at **0.8555**, and the
interaction is the whole story: the same three tasks *with* the floor on score
**0.8387**, the worst of the trained arms. The floor is worth **−0.017** on its
own (§6) but **+0.017** once all three tasks are present.

It is also by far the most stable configuration measured:

| | final | median | min DB | max silhouette | drift |
|---|---|---|---|---|---|
| ALL THREE, var off | 0.8503 | 0.8508 | 0.8511 | 0.8555 | **0.0021** |
| ALL THREE (floor on) | 0.7247 | 0.8069 | 0.8316 | 0.8387 | 0.0231 |

All four read-outs land within **0.005** of each other. **This is the one
configuration where the 2D claim transfers** — the curve converges, so no epoch
rule is needed and the final epoch is as good as any (§3.1).

### 4.2 What each task contributes

Holding the floor on, against `discrete I_h` (0.8430):

| added to discrete views | mean | Δ |
|---|---|---|
| nothing (`ih`) | 0.8430 | — |
| + reconstruction | 0.8457 | +0.003 |
| + SO(3) views (`both`) | 0.8447 | +0.002 |
| + both | 0.8387 | **−0.004** |

Individually each auxiliary task is worth ~nothing, and stacking them *hurts* —
until the variance floor comes off, at which point the same stack is the best
config by +0.009. No single-factor reading of this table is correct; the terms
only make sense jointly.

### 4.3 The per-flow structure matters more than the mean

The mean hides a consistent split. `ALL THREE, var off` wins on **Re6400
(0.8326 vs 0.7918 next best) and tangaroa (0.8811)** — the two flows every other
config struggles with — but is the *worst* trained config on deltaWing (0.8620
against 0.9270). The configs with the variance floor on are uniformly strong on
deltaWing and weak on Re6400.

Since deltaWing is also the flow where an untrained encoder already scores 0.8787
(§9), the reading is that **deltaWing rewards whatever the initialisation already
provides, while Re6400 and tangaroa require the representation to actually
learn** — and the floor-off stack is what learns them.

## 5. Strategy 1 — learning rate: no transfer

*(measured through k-means; the ordering is stable but the absolute values are
low by ~0.05, see §9)*

2D found lr 1e-4 flattened the oscillation and was the second-largest win. Here
5e-4 is an interior optimum, bracketed from both sides on Re160:

| lr | 3e-5 | 1e-4 | 2e-4 | **5e-4** | 1e-3 |
|---|---|---|---|---|---|
| selected | 0.7508 | 0.7439 | 0.7504 | **0.8388** | 0.8246 |
| oracle | 0.7537 | 0.8308 | 0.8320 | 0.8391 | 0.8355 |

Below 2e-4 the *oracle* is unchanged (~0.83) while the selected value drops ~0.09
— the representation is fine, the epoch rule is failing. At 3e-5 the oracle
collapses too: undertrained. Confirmed on Re640 and Re6400.

## 6. Strategy 2 — the variance floor

2D found `zinv_var_weight 0` worth +0.257 on boussinesq. In 3D it is a **trade
between peak and convergence**, and the trade is visible only under GMM-tied:

| `var0` vs the others (GMM-tied, 5-flow means) | value |
|---|---|
| mean at the best label-free rule | 0.8253 (lowest of the trained configs) |
| **oracle** | **0.8671 (highest)** |
| **drift over the last quarter** | **0.0036 (lowest; next is 0.0171)** |
| `final` epoch usable? | **yes — 0.8253, the only config where it is** |

So the VICReg variance hinge buys peak accuracy at the cost of a curve that never
settles; removing it buys a converged curve at a lower peak. The 2D reading —
"turning it off removes the good state rather than stabilising it" — holds, but
in 3D the converged state is only 0.02 below the oscillating one, which makes it
a reasonable default when no epoch rule is trusted.

Under k-means this looked completely different (0.7469 at DB, 0.8609 at
silhouette on Re160) and produced a conclusion about DB-vs-silhouette that the
fixed read-out does not support.

## 7. Strategy 3 — partition rule and epoch rule

Under **GMM-tied**, min-DB and max-silhouette agree closely (0.8447 vs 0.8400
mean) and either is usable; the partition rule within an evaluation barely
matters because the metric already isolates the right split.

Under **k-means**, where most of this study was first run, the picture was
different and is kept for the record: the *epoch* rule was the lever (+0.16
between min-DB and final) while all four partition rules agreed to 0.0001 at the
selected epoch — the reverse of 2D, where the partition rule was worth +0.20.
That inversion was an artefact: k-means discards the good partition at most
epochs, so min-DB was doing the work of finding the rare epochs where it
survives.

**Calinski–Harabasz is not an independent criterion** — identical to inertia to
four decimals in every cell, exactly as in 2D.

## 8. Strategy 4 — capacity and schedule

Doubling the latent (`z_inv` 16→32, `z_eq` 12→24) collapses DB-based epoch
selection, reproducibly:

| | Re160 | Re6400 |
|---|---|---|
| `zinv32` @ min-DB | **0.4521** | **0.4538** |
| `zinv32` @ max-silhouette | 0.7362 | 0.7560 |
| 16/12 baseline @ min-DB | 0.8388 | 0.7652 |

The two collapse values agree to 0.002, and the oracle falls on both flows, so
the representation is worse as well. **Capacity is not the constraint.**

**The schedule is saturated.** 12,000 steps reproduces 6,000 *exactly* to four
decimals — Re160 0.8388, Re6400 0.7652 — for twice the wall clock.

`lr 1e-3` on Re6400 scores 0.7662 against the shipped rate's 0.7652, confirming
5e-4 as an interior optimum on a second flow.

---

## 9. The untrained baseline, and two corrections it forced

This is the most important section in the document.

The `lr = 0` control was originally read through k-means and scored **0.4603**,
which made training look worth **+0.36**. Re-read through GMM-tied on the same
saved latents, the untrained encoder scores **0.6594**:

| flow | init @ k-means | **init @ GMM-tied** | FMT | best trained |
|---|---|---|---|---|
| deltaWing | 0.4231 | **0.8787** | 0.8637 | 0.9267 |
| Re640 | 0.5310 | **0.7598** | 0.7717 | 0.8296 |
| Re6400 | 0.4852 | **0.7528** | 0.7689 | 0.7918 |
| Re160 | 0.4448 | 0.4345 | 0.7356 | 0.8576 |
| tangaroa | 0.4173 | 0.4714 | 0.8636 | 0.8619 |
| **mean** | **0.4603** | **0.6594** | **0.8007** | **0.8470** |

**Correction 1 — the effect of training is +0.188, not +0.36.** About half of
what was attributed to the contrastive objective was k-means mishandling the
untrained latent. The objective is still the largest single factor and the
ranking of trained configs is unchanged, but the effect size was inflated.

**Correction 2 — on deltaWing an untrained encoder beats FMT.** 0.8787 against
0.8637, with no training whatsoever, and the trained model only reaches 0.9267.
So the deltaWing "+0.457 from training" reported earlier is really **+0.048**.
A random 13-line icosahedral encoder plus a tied-covariance GMM is already a
strong vortex detector on that flow; most of the score is the *primitive and the
metric*, not the learning.

A third correction, from §3: **tangaroa was never a structural loss.** Under
k-means every config landed 0.815–0.841 against FMT's 0.8636 and this was
diagnosed as epoch selection failing to capture a good representation. Under
GMM-tied tangaroa reaches 0.8619–0.8664, a tie with FMT, with no retraining. The
binding constraint was the clusterer's metric.

---

## 10. Against the right baseline

The published Task-1 FMT numbers (Re160 .5225, Re640 .5177, Re6400 .5340) come
from the **old split protocol** and are not a valid reference here. Recomputed on
identical full data through the identical read-out machinery
(`experiments/Run_Task1_Baselines_Full_1_1.py`):

| arm | Re160 | Re640 | Re6400 | tangaroa | deltaWing | mean |
|---|---|---|---|---|---|---|
| FMT, published (old split) | 0.5225 | 0.5177 | 0.5340 | — | — | 0.5247 |
| FMT, 7-line star, full data | 0.7356 | 0.7717 | 0.7689 | — | — | 0.7587 |
| **FMT, 13-line star, full data** | 0.7547 | 0.7649 | 0.7613 | **0.8636** | **0.8637** | **0.8007** |
| Raw coordinates, full data | 0.6709 | 0.6704 | 0.6721 | 0.3912 | 0.4356 | — |
| IcoVAE untrained (lr = 0) | 0.4345 | 0.7598 | 0.7528 | 0.4714 | 0.8787 | 0.6594 |
| **IcoVAE all-three-tasks, var off** | 0.8638 | 0.8381 | 0.8326 | 0.8811 | 0.8620 | **0.8555** |
| IcoVAE `ssl_so3`, GMM-tied | 0.8495 | 0.8166 | 0.7907 | 0.8515 | 0.9267 | 0.8470 |
| IcoVAE `ssl_ih_rec`, GMM-tied | 0.8576 | 0.8296 | 0.7884 | 0.8295 | 0.9234 | 0.8457 |

**The split alone was costing FMT +0.234.** Measured properly FMT scores 0.8007,
not 0.5247.

**The best configuration delivers +0.054** (0.8555 against 0.8007) and wins on
**all five flows**: +0.109 Re160, +0.073 Re640, +0.071 Re6400, +0.018 tangaroa,
−0.002 deltaWing. Every trained config beats FMT on the mean (0.8313–0.8555), so
the conclusion is robust to recipe choice; the *size* is not, ranging +0.030 to
+0.054.

## 11. What the representation can reach

| flow | best delivered | best oracle | selection gap |
|---|---|---|---|
| Re160 | 0.8576 | 0.8704 | 0.013 |
| Re640 | 0.8461 | 0.8461 | 0.000 |
| Re6400 | 0.7918 | 0.8437 | **0.052** |
| tangaroa | 0.8664 | 0.8813 | 0.015 |
| deltaWing | 0.9270 | 0.9297 | 0.003 |

Selection is near-optimal on four flows. Re6400 is the one place where a better
label-free rule is still worth ~0.05.

## 12. What transfers from 2D, and what does not

| 2D finding | 3D outcome |
|---|---|
| **GMM-tied replaces k-means** | **holds, largest single gain** — +0.051, `tied` > `full` by 0.18 |
| Calinski–Harabasz ≡ inertia | **holds** — identical to four decimals |
| the contrastive term is the method | **holds, but smaller** — +0.188, not the +0.36 k-means suggested |
| reconstruction is nearly free | **holds** — 0.8457 vs 0.8430, +0.003 |
| continuous views add nothing | **wrong in 3D** — `ssl_so3` is the *best* config under GMM-tied |
| lr 1e-4 flattens the curve | **does not transfer** — 5e-4 is an interior optimum |
| silhouette is the single best read-out change | **superseded** — the clusterer is, and under GMM-tied the partition rule barely matters |
| restart count 8–14, cliff at 20 | **does not transfer** — flat 5→20 at N ≈ 50,000 |
| curve converges → use `final`, retire DB | **only for the floor-off configs** — `all3_var0` converges to 0.8503 final vs 0.8555 selected (drift 0.0021); floor-on configs converge to a *suboptimal* partition |
| combining SSL tasks should compound | **only with the floor off** — all three tasks score 0.8387 with the floor on, 0.8555 with it off |
| `zinv_var_weight 0` is a large win | **a trade** — lowest peak, highest oracle, only converged curve |

The pattern: **read-out choices transfer, training-hyper-parameter choices do
not.** The clusterer and the criteria behave the same way in both dimensions; the
learning rate, view type and variance floor all reverse.

## 13. Caveats

* **One seed per cell.** Given that two conclusions in this study reversed on
  re-measurement, this is the first thing a follow-up should fix.
* The `var0` Re160 cell comes from an identically-configured earlier run.
* §5, §6 and §8 were measured through k-means; the orderings are believed stable
  but the absolute values run ~0.05 low.
* The IVD reference is a p95 threshold on a derived quantity, not human labels.
* Silhouette costs an O(N²) pass; at N ≈ 50,000 it is subsampled to 4,096 points
  with a fixed seed. GMM-tied at this N is ~10× slower than k-means per fit,
  which is why the read-out sweep is done offline.

## 14. Files

| file | what |
|---|---|
| `FMT_Utils/IcosahedralGroup_3D.py` | 12 vertices, 120 elements, channel permutations, discrete + SO(3) views |
| `FMT_Utils/IcosahedralPrimitives_3D.py` | 13-line star integrator |
| `FMT_Utils/IcoVAE_3D.py` | model, equivariant normaliser, objective |
| `FMT_Utils/ClusterReadout_3D.py` | 7 clusterers, 4 criteria, the metric |
| `experiments/Build_Task1_Ico_Cache_1_1.py` | cache builder (no split) |
| `experiments/Run_Task1_IcoVAE_1_1.py` | training, F1-curve recording, latent saving |
| `experiments/Analyze_Task1_IcoVAE_Clusterer_1_1.py` | offline clusterer × restarts × epoch sweep |
| `experiments/Analyze_Task1_IcoVAE_Readout_1_1.py` | partition × epoch rule study |
| `experiments/Run_Task1_Baselines_Full_1_1.py` | FMT and Raw on the same full data |
| `tests/test_icosahedral_group_3d.py` | 7 checks incl. action-vs-retrace 8.88e-16 |
