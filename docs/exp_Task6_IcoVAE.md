# Task 6 — coreline classification with the icosahedral SIREN-VAE

Supervised counterpart to [`3d_kmeans_ico.md`](3d_kmeans_ico.md): does a seed lie
on a vortex coreline, given a 13-line icosahedral streamline star? The question
is whether the self-supervised objectives that carry the *unsupervised* Task 1
still earn their place when a label is available.

**Answer: partly. The SIREN reconstruction earns +0.0126 (p = 0.003). The
icosahedral contrastive term — which is the whole method on Task 1 — contributes
nothing here (−0.0020, p = 0.52).**

> Everything below is measured on **deltaWing only**, with **exactly traced**
> icosahedral stars. Earlier revisions of this document reported numbers from
> other scenes using neighbours *borrowed from nearby seeds*; those stars are not
> icosahedral in any meaningful sense and the numbers are withdrawn (§2.3).

---

## 1. Which scenes are usable

Task 6 ships 400,000 seeds per frame with a precomputed 65-step streamline each,
a binary coreline label, and a `default_split.json` of 45 train / 24 test frames.
Tracing a *new* streamline from `centre + r·d_j` — which is what an icosahedral
star requires — needs the underlying velocity field.

Each candidate field was tested by interpolating it along the stored streamlines
and measuring the cosine between the field direction and the streamline tangent.
A true match reads ≈ 1.0; an unrelated field reads ≈ 0.0.

| Task 6 scene | best candidate | median cos | usable |
|---|---|---|---|
| **deltaWing_resampled** | `deltaWing_mag0_3reesampled` @ idx 7 | **0.9985** | **yes** |
| cylinder3d | `halfcylinderRe160` | 0.8653 | no |
| halfcylinderRe640 | `halfcylinderRe640` | 0.8657 | no |
| halfcylinderRe320 | `halfcylinderRe6400` | 0.7662 | no |

`halfcylinderRe160Resampled` (160×60×20) and `halfcylinderRe640resampled`
(160×60×20, t 7.5–15) were tested with correct *time-value* lookup and reach
only 0.863 / 0.862 — the same simulation family, a different variant. The
calibration is the cross-check: deltaWing against a half-cylinder field gives
cos ≈ 0.000, against its own field 0.9985.

`SquareCylinder` and `tornado3d` ship no `sample_streamlines.npy` at all;
`tornado3d` was the split's only genuinely held-out scene.

**So the study is one scene**: 10 train / 5 test frames, 20,000 seeds each —
200,000 train / 100,000 test, positive rate ≈ 12%.

## 2. The star

`FMT_Utils/StreamlineStar_3D.py` traces RK4 streamlines in arclength, matching
the dataset's convention (65 samples spanning 0.5 of arclength, centred on the
seed, truncated at the domain boundary). The star is the centre streamline plus
twelve traced from `centre + r·d_j`.

### 2.1 Radius is the single largest lever

| radius | reconstruction+contrastive | supervised |
|---|---|---|
| 1h | 0.7530 | 0.7419 |
| 2h | 0.7708 | 0.7714 |
| 4h | 0.7846 | 0.7859 |
| 6h | 0.7954 | 0.7809 |
| **8h** | **0.7995** | 0.7872 |
| 9h | 0.8019 | — |
| 10h | 0.7995 | — |
| 12h | 0.7887 | 0.7839 |

A genuine peak with a plateau at **8–9h**, falling away on both sides. 12h is
where the star starts to exceed the streamline's own 13.7h of arclength.

### 2.2 Exact tracing beats borrowing neighbours by +0.093

On the identical scene and split, supervised-only:

| star construction | best F1 |
|---|---|
| **traced, r = 8h** | **0.7995** |
| traced, r = 4h | 0.7251 |
| borrowed: 12 of the 64 nearest seeds, Hungarian-matched to the directions | 0.6856 |
| traced, r = 1h | 0.6716 |

### 2.3 Why the borrowed-neighbour numbers are withdrawn

Assigning existing seeds to the twelve directions leaves a 12–15° angular error,
and below 2h the seed cloud is so sparse that **half the stars reuse the same
streamline in several channels**. That is not an icosahedral star: the channel
permutation the group action depends on carries no geometric meaning, so the
`I_h` augmentation is acting on an arbitrary ordering. Tracing removes both
problems and makes any radius available.

| | angular error | rows reusing a seed |
|---|---|---|
| borrowed, radius 0.5h | 49.1° | 53% |
| borrowed, radius 2h | 18.0° | 42% |
| borrowed, k-nearest + Hungarian | 14.8° | 0% |
| **traced** | **0°** | **0%** |

## 3. What the objective terms are worth

Five seeds per arm, r = 8h, warmup 2000 + cosine, lr 1e-3, λ_recon 30.

| arm | n | mean | std | vs supervised | Welch |
|---|---|---|---|---|---|
| **reconstruction only** | 5 | **0.8014** | 0.0026 | **+0.0126** | t = 4.79, **p = 0.003** |
| contrastive + reconstruction | 5 | 0.8001 | 0.0058 | +0.0113 | t = 3.23, p = 0.012 |
| supervised only | 5 | 0.7888 | 0.0053 | — | — |
| contrastive only | 5 | 0.7868 | 0.0042 | −0.0020 | t = −0.68, **p = 0.517** |

**Reconstruction is the whole effect.** The contrastive term is indistinguishable
from nothing on its own and slightly *degrades* the combination. This is the
exact inverse of Task 1, where the contrastive term is worth +0.19 over an
untrained encoder and reconstruction ≈ +0.003.

A plausible reading: the label already supplies the discriminative signal, so an
auxiliary task can only add what the label does not carry. Reconstruction forces
the latent to retain the *geometry* of the star; contrastive invariance to `I_h`
removes orientation information — and orientation is exactly what distinguishes
a coreline-aligned star from a passing one.

λ_recon has an interior optimum: 0 → 0.7819, 1 → 0.7892, 3 → 0.7920,
10 → 0.7987, **30 → 0.8004**, 100 → 0.7880, 300 → 0.7258, 1000 → 0.6616.

## 4. Learning rate and schedule

At lr 1e-3 the schedule is not optional:

| schedule | best F1 |
|---|---|
| constant, no warmup | **0.1564** *(diverges; final 0.0000)* |
| cosine, no warmup | 0.4968 |
| constant + warmup 1000 | 0.7878 |
| **warmup + cosine** | **0.7987 – 0.8037** |

**Warmup is worth +0.63; cosine adds a further +0.011.** Warmup length is flat
over 1000–6000. Raising the rate further (1.5e-3, 2e-3 with warmup 3000) gives
0.7962 / 0.8036 — no reliable gain. Below, 3e-4 gives 0.7972: also flat.

## 5. Everything else is flat

Across 41 configurations at r = 8h the top 22 span **0.7943 – 0.8018**:

| factor | range tested | spread |
|---|---|---|
| latent size | 8/6 → 64/48 | 0.7958 – 0.7990 |
| head | linear vs MLP | 0.7960 vs 0.7987 |
| views | `ih` / `so3` / `both` | 0.7956 – 0.7987 |
| batch | 256 – 2048 | 0.7902 – 0.7982 |
| λ_ssl | 0 / 5 / 15 / 45 / 100 | 0.7952 – 0.8015 |
| weight decay | 0 / 1e-4 / 1e-3 | 0.7964 – 0.7991 |

Two settings do cost: `encoder_lr_scale 0.1` (0.7513 — the encoder needs the full
rate) and schedules longer than 16k steps (32k → 0.7912, 48k → 0.7919).

**Once the star is geometrically correct the method is insensitive to almost
every hyper-parameter.** The ranking of levers is: star radius and exactness
(+0.093 and +0.14 across the range) ≫ warmup (+0.63 at lr 1e-3, i.e. the
difference between working and diverging) ≫ reconstruction (+0.013) ≫ everything
else (< 0.008 combined).

## 6. Files

| file | what |
|---|---|
| `FMT_Utils/StreamlineStar_3D.py` | RK4 arclength tracer, exact icosahedral star |
| `experiments/Build_Task6_ExactStar_1_1.py` | traced-star cache (verified fields only) |
| `experiments/Build_Task6_Ico_Cache_1_1.py` | borrowed-neighbour cache (superseded, §2.3) |
| `experiments/Run_Task6_IcoVAE_1_1.py` | training, SSL auxiliaries, warmup/cosine, per-scene mode |

## 7. Caveats

* **One scene.** deltaWing is the only Task 6 scene whose field is available, so
  nothing here is known to generalise across flows.
* The split is temporal within that scene, so it measures generalisation to
  unseen time steps, not unseen flows.
* F1 is the positive (coreline) class at threshold 0.5; AP and macro F1 are also
  recorded, and AP sometimes favours the supervised arm where F1 does not.
* Best single run 0.8092; best 5-seed mean 0.8014 ± 0.0026.
