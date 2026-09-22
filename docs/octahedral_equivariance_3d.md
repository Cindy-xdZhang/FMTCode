# 3D octahedral-equivariant SIREN-VAE — design strategy

Status: **strategy accepted; Phase 0 passed; Phase 1 run through three versions —
the method is competitive but does not yet beat FMT.** Macro F1 went .3276
(v1.2) -> .5110 (v1.3) against FMT's .5751 once the k-means boundary is fitted on
the development slices closest in time to confirmation. The group machinery is
verified correct; the remaining gap is a representation/selection problem on the
two higher-Reynolds half-cylinders, not a group one. Full result and diagnosis in
[`exp_Task1_OctVAE_3D.md`](exp_Task1_OctVAE_3D.md) §10. This document is
the design argument — the group, the primitive, the signal and the protocol were
fixed here before any code was written.  The experiment record, including the
measured engineering findings, is in
[`exp_Task1_OctVAE_3D.md`](exp_Task1_OctVAE_3D.md).

Target: a second Task1 method alongside the training-free FMT encoder, ported
from the 2D `D8-equivariant SIREN-VAE + VICReg` described in
`/home/cheny1a/git/PyflowVis/README_pathline_clustering.md`, including its four
default refinements.

---

## 1. What the 3D primitive actually is

The 3D Task1 primitive is **7 lines: a centre plus 6 axis-aligned neighbours**
`x±, y±, z±` — an *octahedral* star, not a 12-point star.

Evidence:

- `FMT_Utils/FMT_3D_pipeline.py:187-191` — the literal offset table
  `[0,0,0], [±o,0,0], [0,±o,0], [0,0,±o]`, reshaped to `[-1, 7, L, 4]`.
- `FMT_Utils/RawPathline_3D.py:11` — `reshape(len(flat), 7, -1, 3)`.
- `FMT_Utils/PrimitiveVAE_3D.py:14` — asserts geometry shape `(7, 32, 3)`.
- `docs/research_tasks_and_protocol.md:7` — "3D primitive 为中心线和 `x±、y±、z±` 共 7 条线".
- Cached data, `outputs/exp_Task1_Tangaroa_0.1/development_cache/tangaroa_eth/`:
  `raw_features` is `(3840, 672)` and `672 = 7 x 32 x 3`; `line_lengths` is `(4096, 7)`.

A 12-neighbour variant does not exist in this repo. The only larger stencil is
`FMT_Utils/LargeNeighbor_3D.py`, which is 25 lines (centre + 8 cube-vertex
directions on each of 3 radial shells) — a different, radially-stratified object.

**The 12-evenly-distributed-neighbour intuition is a good design instinct and is
addressed in §4 — but it describes a primitive we would have to build, not the one
the caches hold.**

---

## 2. The group: the 3D counterpart of D8

In 2D the 8 neighbours sit on a circle, so a 45-degree rotation acts as a cyclic
shift `perm[k] = (k - n) mod 8` (`FMT_Utils/siren_vae.py:74-84`). The group is
D8, order 16 = 8 rotations x 2 reflections.

The 3D counterpart of "the group that permutes the neighbour channels" is the
**full octahedral group `O_h`, order 48** — the symmetry group of the 6-point set
`{±e_x, ±e_y, ±e_z}`. Its rotation subgroup `O` has order 24.

The decisive structural fact, verified numerically:

> **`O_h` is exactly the set of 3x3 signed permutation matrices.**
> 3! axis permutations x 2^3 sign choices = 48; the 24 with `det = +1` are `O`.

So the group action has a closed form with no interpolation and no trigonometry.
Write a group element as `(sigma, eps)` with `sigma` a permutation of `{0,1,2}` and
`eps in {+1,-1}^3`, meaning `M e_a = eps_a e_sigma(a)`. Then:

- **Vector components** transform as `v -> M v` — a permutation of the 3
  components with signs. Exact in float, no rounding.
- **Neighbour channels** permute by where their direction goes. Index the 6
  neighbour channels in the pipeline's own order `(x+, x-, y+, y-, z+, z-)`, i.e.
  channel `2a + [s < 0]` holds direction `s e_a`. Then
  `M (s e_a) = (s eps_a) e_sigma(a)`, so the content of channel `2a + [s<0]`
  moves to channel `2 sigma(a) + [s eps_a < 0]`.
- **Centre channel** is fixed in position; only its components rotate.

This is *simpler* than the 2D implementation, not harder. The difficulty the
request anticipated — "rotating the whole is hard" — dissolves once the group is
chosen to be the stabiliser of the neighbour set rather than a generic rotation.

**Nothing about this requires 12 neighbours.** It requires the neighbour set to be
closed under the group, which the octahedral 6 already is.

---

## 3. Where 3D is genuinely harder: group coverage

The real cost of 3D is not implementing the action — it is that a finite group
covers `SO(3)` far more coarsely than it covers `SO(2)`. Measured as the rotation
angle from a uniformly random rotation to its nearest group element (Monte Carlo,
200k samples, `scipy` `Rotation.random`):

| group | elements | manifold | mean angle | 95th pct | max (covering radius) |
|---|---|---|---|---|---|
| 2D rotations of D8 | 8 | SO(2), dim 1 | 11.2 deg | 21.4 deg | 22.5 deg (analytic) |
| octahedral `O` | 24 | SO(3), dim 3 | **40.7 deg** | 57.0 deg | ~62.6 deg |
| icosahedral `I` | 60 | SO(3), dim 3 | 29.5 deg | 39.7 deg | ~44.3 deg |

So the 3D augmentation is about **3.6x coarser in mean angle** than the 2D one it
is ported from, despite having 3x more elements (48 vs 16). This is the honest
statement of the 3D difficulty, and it should be written into the results doc
rather than discovered later as an unexplained weakness.

Consequences accepted up front:

1. The invariance the contrastive loss can impose is *exactly* `O_h`-invariance,
   not approximate `O(3)`-invariance. We should not claim the latter.
2. Because coverage is coarser, `z_eq` has more work to do — see §8.
3. It motivates the optional extra positive-pair source in §8 (temporal
   truncation), which is group-independent.

---

## 4. The 12-neighbour question, answered

"12 evenly distributed neighbours" is ambiguous between two sets, and they behave
very differently:

| stencil | #nbrs | min neighbour angle | symmetry group | rotations | needs re-integration | comparable to published Task1 |
|---|---|---|---|---|---|---|
| octahedron `±e_a` (**current**) | 6 | 90.0 deg | `O_h` | 24 | **no — caches exist** | **yes** |
| cuboctahedron `(±1,±1,0)/sqrt2` | 12 | 60.0 deg | `O_h` | 24 | yes | no |
| icosahedron | 12 | 63.4 deg | `I_h` | 60 | yes | no |
| cube vertices `(±1,±1,±1)/sqrt3` | 8 | 70.5 deg | `O_h` | 24 | yes (exists in `LargeNeighbor_3D`) | no |

Three findings that decide the matter:

1. **The cuboctahedral 12 buys nothing group-theoretically.** Its symmetry group is
   the same `O_h`, order 48. Going from 6 to 12 neighbours would double the
   integration cost and change the primitive without enlarging the augmentation
   group by a single element.
2. **Only the icosahedral 12 enlarges the group** (60 rotations instead of 24), and
   by §3 that improves mean coverage from 40.7 to 29.5 degrees — a factor of 1.38.
   It is the maximally even 12-point set (min neighbour angle 63.4 deg beats the
   cuboctahedron's 60.0 deg), so the instinct behind the question is correct.
3. **`I_h` and `O_h` are incompatible.** Neither contains the other, and icosahedral
   rotations are not signed permutations, so an icosahedral primitive cannot reuse
   the axis-aligned caches, cannot reuse the frozen FMT 161-D encoder without
   redefinition, and cannot be compared line-for-line against the published Task1
   Raw/FMT table.

**Decision: keep the 6-neighbour octahedral primitive.** A 1.38x improvement in
angular coverage does not justify re-integrating every slice, invalidating the
comparison to the published Task1 numbers, and confounding "new method" with "new
primitive" — which the protocol's capacity-control clause
(`docs/research_tasks_and_protocol.md:35`) exists precisely to prevent.

If a future experiment ever does re-integrate, the note to carry forward is:
**use the icosahedral 13-line primitive, not the cuboctahedral one** — same cost,
60 rotations instead of 24.

---

## 5. Is the augmentation legitimate? (label invariance)

An augmentation is only valid if it preserves the label. The Task1 label is
`IVD(seed) >= percentile95(IVD volume)` with
`IVD = || omega - <omega> ||` (`FLowUtils/ScalarField3d.py:80-84`).

Under `x -> M x`, `v -> M v` for any `M` in `O(3)`, vorticity is a pseudovector:
`omega -> det(M) M omega`, and the spatial mean transforms identically, so

```
omega - <omega>  ->  det(M) M (omega - <omega>)
|| . ||          ->  unchanged,   because ||M u|| = ||u|| and |det M| = 1
```

**IVD is invariant under the full `O(3)`, reflections included.** All 48 `O_h`
elements are therefore label-exact augmentations. Reflection augmentation needs no
apology, unlike the case for a handedness-sensitive label.

### The chirality tension — a real trade-off to measure, not assume

The frozen FMT encoder deliberately includes a **chirality block** built from
triple products, documented as "preserve handedness under proper rotations while
changing sign under reflections" (`FMT_Utils/DFT_FMT_3D.py:381-382`). It is
`num_freq - 1 = 5` dims per line x 7 lines = **35 of the 161 FMT dimensions (22%)**.

Full `O_h` invariance discards exactly that information. Since the *label* is
reflection-invariant (above), chirality can only help through dataset-specific
handedness correlations — plausibly a shortcut rather than a transferable feature,
but this is an empirical question, not a settled one.

**Therefore: ship a `--group {oh,o}` flag.** `oh` (48 elements, default, faithful to
the 2D method) and `o` (24 proper rotations only, chirality-preserving). One flag,
one extra run, and the question is answered instead of argued.

---

## 6. Three things that must change from the 2D code

Porting is mostly mechanical, but three points are not, and getting them wrong
would silently break equivariance.

### 6.1 Normalisation must drop the mean subtraction

`normalize_signal_d8` (`FMT_Utils/siren_vae.py:211-241`) subtracts a **scalar**
mean and divides by a scalar RMS. Division by a scalar is exactly equivariant
(rotations preserve norms). Subtraction of a scalar `c` from a vector is **not**:

```
normalize(M s) = (M s - c*1) / sigma
M normalize(s) = (M s - c*M 1) / sigma        differ by  c (M 1 - 1) / sigma
```

In 2D this is a small approximation. In 3D it is worse, because the half-cylinder
and Tangaroa flows have a strong mean streamwise velocity, so `c` is far from zero
and the axis-permuting elements of `O_h` make `M 1 != 1` often.

**Fix: scalar RMS about zero, no mean subtraction**, computed per group (one for
the centre channel, one shared by all 6 neighbours — the same grouping the channel
permutation acts within). This is exactly `O(3)`-equivariant.

This is checkable, and will be a unit test:
`max |normalize(M s) - M normalize(s)| < 1e-6` over all 48 elements.

### 6.2 Channel-balance grouping

`channel_balance_weights` uses one weight for the centre and one shared across all
neighbours, precisely so the weighting commutes with channel permutation. The same
rule holds in 3D with 6 neighbours instead of 8 — **one centre weight, one shared
neighbour weight**. Per-channel weights would break equivariance.

### 6.3 Epoch budget must be re-derived, not copied

The 2D pipeline is transductive over ~512 pathlines (`README` §7). Here:

| | 2D reference | 3D, per dataset |
|---|---|---|
| training primitives | ~512 | ~28,700 (8 dev slices x ~3,590) |
| steps/epoch at batch 256 | 2 | ~112 |
| 3000 epochs | 6,000 steps | 336,000 steps |

Copying `--epochs 3000` would multiply the optimisation budget by 56x. **Budget in
steps.** Proposal: ~28,000 steps (~250 epochs), i.e. 4.7x the 2D step budget,
justified by the larger dataset and coarser group; decoder fine-tune ~4,000 steps
(2D used 2,000 epochs = 4,000 steps); evaluate every ~560 steps (5 epochs) to keep
the Davies-Bouldin selection grid comparable in density to the 2D one.

These are starting values to be fixed in config before the first scored run, not
tuned against confirmation slices.

---

## 7. The signal

3D analogue of the 2D `[N, 9, L-1, 2]`:

```
signal : [N, 7, 31, 3]
  channel 0    : centre pathline time-derivative       d/dt x_0(t)
  channels 1-6 : neighbour displacement-rate rel. centre  d/dt ( x_i(t) - x_0(t) )
                 ordered (x+, x-, y+, y-, z+, z-)
```

Both are translation-invariant by construction and rotate as vectors, which is
what the group action requires. `31 = 32 - 1` from `sampled_steps: 32`.

**It is recoverable from the existing caches with no re-integration.**
`_raw_local_features` (`experiments/Build_Task2_Universality_Cache.py:63-65`)
stores `x_i(t) - x_0(0)` as `[N, 7, 32, 3]` flattened to 672, and

```
centre_rate_t   = diff_t ( r[:,0] )
relative_rate_t = diff_t ( r[:,i] - r[:,0] )      the common anchor x_0(0) cancels
```

The 2D `x100` neighbour factor is a documented no-op under group-wise
normalisation and is **not** carried over; the 3D normaliser is group-wise by
construction, so introducing the factor would only add a trap.

Available immediately: **145,878 primitives across 4 datasets**
(107,519 in `exp_Task1_ReSweep_0.1` over Re160/Re640/Re6400, 38,359 in
`exp_Task1_Tangaroa_0.1`), plus 4 frozen confirmation slices each.

---

## 8. Model and the four refinements

The model is essentially dimension-agnostic; only the channel count changes.

| | 2D | 3D |
|---|---|---|
| encoder input channels `K x C` | 9 x 2 = 18 | 7 x 3 = **21** |
| sequence length | `L-1` | **31** |
| `z_inv` | 16 | 16 (unchanged) |
| `z_eq` | 8 | **12** |

`z_eq` grows because it absorbs orientation, and orientation in 3D is a
3-dimensional manifold with a 48-element group acting on it, versus 1-dimensional
with 16 elements in 2D. 12 is a proposal, not a measured optimum; it is fixed
before the first scored run and recorded.

**The four refinements all port unchanged in spirit:**

1. `--encoder_lr_scale 0.1` — unchanged, no dimensional dependence.
2. `--recon_channel_balance` — unchanged, with the 1-centre/6-neighbour grouping of §6.2.
3. `--checkpoint_select davies_bouldin` — unchanged, and doubly important here
   because it is label-free, which is what keeps the confirmation slices frozen
   under `docs/research_tasks_and_protocol.md:33` (test freeze).
4. `--decoder_finetune_epochs` — unchanged, rescaled per §6.3.

**Two 3D-specific additions:**

- **Group-averaged evaluation (diagnostic, free).** Because the group is finite and
  the encoder is tiny, `z_inv_avg = (1/|G|) sum_g f(g . x)` is exactly
  `O_h`-invariant by construction and costs 48 forward passes over ~38k samples —
  seconds. Reporting clustering on both `z_inv` and `z_inv_avg` separates "the
  method works" from "the contrastive loss converged", which the 2D setup cannot
  distinguish. Default on at eval, off at train.
- **Optional temporal-truncation views** (`--temporal_views`), default **off**.
  A prefix crop keeps the seed time and therefore the IVD label exact, so it is a
  legitimate extra positive pair that is independent of the group's coarse SO(3)
  coverage. Off by default because it is a deviation from the ported method;
  available because §3 says we may need it.

k-means everywhere goes through one helper with `n_init=10` and a recorded
`random_state`, per the 2D README's bistability warning (§7) and the protocol's
clause 6 (`docs/research_tasks_and_protocol.md:36`).

---

## 9. Protocol compliance

Mapped against `docs/research_tasks_and_protocol.md`:

| clause | how it is met |
|---|---|
| §4.1 time-slice splits, never random spatial seeds | reuse the existing `exp_Task1_*` split: train `[0-7]`, cluster-calibration `[8,9]`, confirmation = 4 later slices, windows disjoint |
| §4.2 test frozen | checkpoint chosen by Davies-Bouldin (label-free) on development; cluster-to-class map frozen on calibration slices; confirmation touched once |
| §4.3 label freeze | unchanged IVD p95 definition, already in the cached `reference` |
| §4.4 train-only normalisation | RMS stats fitted on train slices only, reused verbatim at eval |
| §4.5 capacity control | report total and trainable parameters; Raw+k-means and FMT+k-means baselines run from the same caches |
| §4.6 >=3 seeds | 3 training seeds minimum (proposal: 5, to match `final_kmeans_seeds`) |
| §4.7 failures preserved | new output tree, nothing overwritten |
| §4.8 reproducible evidence | config + git commit + per-run CSV + summary JSON |
| §5 Task1 extra | k-means fitted on train features only; ARI/NMI reported as primary (mapping-free); both Raw direct and FMT direct reported |

### One framing point that must be stated, not glossed

Structurally this method is `primitive -> learned encoder -> latent -> k-means`,
which is the shape of **Task2's `Raw+VAE` arm**, not of Task1's training-free
encoders. It is being *evaluated* under Task1 (same slices, same labels, same
metrics, unsupervised, label-free throughout), and Task1's acceptance criterion is
clustering performance only (`docs/research_tasks_and_protocol.md:47`). But
`docs/research_tasks_and_protocol.md:26-27` explicitly warns against blending the
task narratives.

So the results doc will say plainly: **this is a self-supervised Task1 arm, it is
not training-free, and its natural second home is as an upgraded Task2 `Raw+VAE`
baseline.** It must never be reported as "FMT beaten at its own training-free game"
without that sentence attached.

---

## 10. Files to add

| path | contents |
|---|---|
| `FMT_Utils/OctahedralGroup_3D.py` | the 48 signed-permutation elements, channel permutation table, batched group action, equivariant normaliser, signal builder from cached `raw_features` |
| `FMT_Utils/SirenVAE_3D.py` | encoder, SIREN decoder, `z_inv`/`z_eq` split, contrastive + VICReg + channel-balanced recon losses |
| `experiments/Run_Task1_OctVAE_3D_1_1.py` | training/eval driver, Raw and FMT k-means baselines from the same caches, CSV + JSON output |
| `config/exp_Task1_OctVAE_3D_1.1.yaml` | datasets, splits, frozen hyperparameters, seeds |
| `tests/test_octahedral_group_3d.py` | group closure, channel-permutation correctness, exact normalisation equivariance, signal reconstruction from cache |
| `docs/exp_Task1_OctVAE_3D.md` | method doc and results (results appended only after runs complete) |
| `docs/octahedral_equivariance_3d.md` | this document |

Nothing existing is modified.

---

## 11. Phasing

**Phase 0 — group correctness (fast, no training). DONE.** Built
`OctahedralGroup_3D.py` + `tests/test_octahedral_group_3d.py`. All gates pass:
48 distinct elements forming a closed group; the induced channel permutation
matches the geometric action on the 6 directions for all 48; normaliser
equivariance **3.55e-15 on real cached primitives**; signal rebuilt from cache
matches a direct recomputation.

**Phase 1 — the method, on the existing caches. DONE.** Trains per dataset on the
four cached flows, 3 training seeds x 5 k-means seeds, reporting four invariant
readouts against the Raw and FMT arms recomputed on the identical slices.
Measured cost: 33.3 ms/step at batch 256 after batching the three contrastive
views into one encoder pass, so ~16 min of pure training per seed, with the four
datasets run concurrently because the workload is launch-bound. Group pooling
was extended from the mean to **mean, max and their concatenation**, all three
exactly `O_h`-invariant.

**Phase 2 — the two ablations that answer open questions.**
`--group o` (chirality, §5) and `--temporal_views` (coverage, §3). Not yet run.
A third ablation arrived unplanned and is already in the results: `octvae_scaled`
scores the same latent with per-dimension standardising, documenting the
clustering failure of risk 3 rather than hiding it.

Phase 3 (icosahedral re-integration) is **not** proposed; §4 argues against it.

---

## 12. Risks

1. **Coarse coverage (§3) may cap the achievable invariance.** Mitigations are
   already designed in: group-pooled eval, temporal views. Measured outcome: the
   coarse coverage did **not** bite. The `plain` readout — `z_inv` with no
   pooling — is the best readout on 3 of 4 flows, so the contrastive loss makes
   `z_inv` near-invariant on its own and explicit group pooling adds nothing.
   Reporting `plain` beside the pooled readouts is what makes that statement
   checkable rather than assumed.
2. **VICReg collapse.** Measured: did not occur. `z_inv` per-dimension std stays
   around .50-.57 and the latent is **full rank 16** at every probe on all three
   datasets examined. An apparent collapse in version 1.1 was entirely a k-means
   artefact (risk 3), which is why the rank and std are probed directly rather
   than inferred from a downstream metric.
3. **k-means bistability** (2D README §7) — this turned out to be the single
   biggest practical risk, and `n_init` does **not** handle it. Standardising the
   L2-normalised latent (which is what the repo's `fit_kmeans_transform` does,
   correctly, for Raw and FMT) drives k-means onto a degenerate "everything is a
   vortex" split whose inertia is a genuine optimum — `n_init=100` returns the
   same answer as `n_init=20`. The latent must be clustered L2-normalised and
   unstandardised, as the 2D method does. Full measurements in
   `exp_Task1_OctVAE_3D.md` §9.
4. **Re160 confirmation slice 134 is a known outlier** (2.93% positives against
   ~8% elsewhere, FMT F1 0.2197). It will distort this method's Re160 aggregate
   exactly as it distorts FMT's. Report Re160 with and without it, and do not quote
   the aggregate alone.
5. **Flow anisotropy.** The stencil is isotropic in physical units
   (`primitive_offset_mode: min`), but the grid is not (`spatial_strides`
   `{x:4, y:2, z:2}` for Tangaroa), so axis-swapping elements of `O_h` mix
   directions with different effective resolution. Second-order, and arguably a
   useful regulariser since the label is genuinely isotropic — but it is a reason
   the reflection-only subgroup might behave differently from the axis-permuting
   one, and worth a line in the results if `--group o` diverges.
