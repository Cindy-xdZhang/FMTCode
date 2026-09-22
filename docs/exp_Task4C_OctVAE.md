# exp_Task4C_OctVAE — the octahedral SIREN-VAE transferred to Task4-c

Version 1.1 copied c156's FPS neighbour selection and is kept as the record of
why that was the wrong primitive. **Version 2.1 (§7) is the correct construction**
and is what further work should build on.

Transfers the Task1 encoder of [`exp_Task1_OctVAE_3D.md`](exp_Task1_OctVAE_3D.md)
to Task4-c Hairpin / Non-hairpin bundle classification, keeping c156's pooling
and classification head so the change is localised to the representation.

## 1. Data provenance — read this before quoting any number

The frozen `Ablation_Task4C_BottomDensity_1.2` cache (193,000 / 3,000 / 10,000)
lives on Ibex and is **not reachable from this machine**; `/ibex` is not mounted
and no `geometry.npy` exists locally. The raw VTK sources *are* local, so the
cache was rebuilt from them.

**The four source VTK SHA256 hashes were verified against the frozen config
before building**, and all four match:

| file | sha256 (first 16) | matches frozen config |
|---|---|---|
| `channel_flow/channel.vtk` | `c1e3c18d0e67b6eb` | yes |
| `channel_flow/channel_GTs.vtk` | `b9fdd93a1d286178` | yes |
| `tbl_flow/tbl.vtk` | `cc82db1d3ca4e196` | yes |
| `tbl_flow/tbl_GTs.vtk` | `4fcd5efcfe13e107` | yes |

The rebuild runs `Task4C_PhysicalLength_4_14.py prepare` with sampling seed
96601 and all original rules, into
`outputs/exp_Task4C_LocalRebuild_1.0/` (a separate namespace, so nothing in the
official `mainExp_`/`Ablation_` trees is touched).

**The rebuilt evaluation sets reproduce the frozen ones exactly.** The 1.1 and
1.2 expansions grew only the training set and preserved validation/test
byte-for-byte, so the 4.14 rebuild should regenerate them — and the label counts
confirm it:

| flow | split | N | positive / negative | frozen (handoff §3) | match |
|---|---|---:|---|---|---|
| channel | validation | 1,500 | 267 / 1,233 | 267 / 1,233 | **yes** |
| channel | test | 5,000 | 1,075 / 3,925 | 1,075 / 3,925 | **yes** |
| tbl | validation | 1,500 | 182 / 1,318 | 182 / 1,318 | **yes** |
| tbl | test | 5,000 | 692 / 4,308 | 692 / 4,308 | **yes** |

**What is and is not comparable.** Test and validation are the frozen sets, so
scores are measured on the same benchmark. **Training is 27,000 bundles (4.14),
against 193,000 in the frozen 1.2 cache** — a 7x smaller training set, and the
bottom-density plus five-fold expansion was introduced precisely to raise
accuracy. So **nothing here may be compared against c156's .920390 or any row of
the handoff §6 baseline table**, all of which train on 193,000. The only valid
control is an FMT+MLP model retrained on this same 27,000 rebuild, which is what
`config/exp_Task4C_LocalRebuild_1.0_fmtbase.json` runs.

## 2. Why the two tasks line up

c156 already forms, for **each valid line as centre**, a group of centre plus six
farthest-point-sampled neighbours. That is exactly the 7-line shape the Task1
encoder consumes, so the transfer swaps the fixed 141-D FMT and its shared
per-line network for the Task1 encoder at the same position:

```
bundle [B,27,32,3]  ->  per valid centre: centre + FPS-6 neighbours  [B,27,7,32,3]
                    ->  Task1 ResConv+Transformer encoder -> mu [B,27,256]
                    ->  masked mean + max over valid lines -> 512
                    ->  c156 head: mlp(512,256), mlp(256,128), Linear(128,2)
```

The latent is sized 256 (`z_inv` 144 + `z_eq` 112, the 4:3 split of Task1's
16/12) precisely so mean+max pooling yields 512 and c156's head is reused
unchanged.

Signal per group: channel 0 is the centre line about its own centroid, channels
1-6 are neighbour minus centre at the same arclength index. Both are
translation-invariant and rotate as vectors.

## 3. Two things that do not transfer, and one that saves it

**The channel permutation disappears.** Task1's six neighbours sit at signed axis
directions, so a group element permutes them. Here they are FPS-selected at
arbitrary positions — but FPS depends only on pairwise distances, which are
rotation-invariant, so a global rotation selects the same neighbours in the same
order. The induced permutation is the identity and the action is a pure
component rotation. A side effect worth noting: the group is no longer confined
to `O_h`, so the coarse `SO(3)` coverage that limits Task1 (40.7 degrees mean,
`octahedral_equivariance_3d.md` §3) does not bind here.

**The label is not rotation-invariant.** Task1's IVD is an `O(3)` invariant, so
all 48 augmentations are label-exact. A hairpin is defined in a wall-bounded flow
with a canonical orientation — legs downstream along `+x`, head lifted from the
wall at low `z` — so rotating a bundle does not preserve "is a hairpin". c156
accordingly uses **no rotation augmentation** (handoff §5).

**The split latent is what makes it safe anyway.** The contrastive and VICReg
terms push `z_inv` toward invariance while `z_eq` stays free and absorbs
orientation, and the classifier reads the *whole* latent. Shape is regularised,
orientation is retained, nothing label-relevant is discarded. `--group reflect`
restricts augmentation to the spanwise reflection `y -> -y`, the one transform
that is a genuine statistical symmetry of both flows, as a comparison point;
`--group none` disables it entirely, matching c156.

## 4. Setup

| | value |
|---|---|
| encoder | Task1 `ResTransformerEncoder3D`, 21 input channels, 32 steps |
| latent | `z_inv` 144 + `z_eq` 112 = 256; pooled mean+max = 512 |
| head A | c156's exact head — `mlp(512,256)`, `mlp(256,128)`, `Linear(128,2)` |
| head B | single `Linear(512,2)` — a linear probe on the same pooled code |
| parameters | **747,500** (MLP head) / **583,276** (linear head), against c156's 992,386 |
| optimiser | AdamW, lr 1e-3 (c156's value), weight decay 1e-4, grad-clip 5, batch 128 |
| schedule | 200-step linear LR warmup then cosine (carried from Task1 v1.5) |
| augmentation | `oh` (48 rotations) on the contrastive branch only |
| selection | best validation F1, average precision as tiebreak — c156's rule |
| decision | Hairpin probability >= .5; F1 on the two flows' test bundles merged |
| seeds | 96721, 96722 (two, as a screen) |

## 5. Files

| path | contents |
|---|---|
| `FMT_Utils/Task4C_OctVAE_1_1.py` | group construction, rotation action, classifier |
| `experiments/Run_Task4C_OctVAE_1_1.py` | training and evaluation driver |
| `config/exp_Task4C_LocalRebuild_1.0.json` | local 4.14 cache rebuild |
| `config/exp_Task4C_LocalRebuild_1.0_fmtbase.json` | same-data FMT+MLP control |

No existing file is modified.

## 6. Version 1.1 results (partial, stopped)

Two seeds x two heads were launched at c156's lr 1e-3 and stopped partway once
the primitive problem in §7 was identified. The completed runs:

| head | seed | best validation F1 | epoch | test F1 | channel / tbl | parameters |
|---|---|---:|---:|---:|---|---:|
| MLP | 96721 | .6329 | 50 | **.6241** | .7093 / .4951 | 747,500 |
| linear | 96721 | .6464 | 73 | **.6325** | .7191 / .4828 | 583,276 |

Seed 96722 reached validation .6250 (MLP) and .6533 (linear) before being stopped.
Three observations survive the cancellation:

- **Severe overfitting.** Training loss falls to ~.002-.005 by epoch 50 while
  validation F1 stalls near .63. 582k encoder parameters against 27,000 training
  bundles, where c156 has 992k against 193,000.
- **The linear head matched or beat the MLP head** (validation .6464/.6533 vs
  .6329/.6250; AP .72-.73 vs .65-.68). The bottleneck was the representation, not
  head capacity.
- **TBL lagged channel badly** (.48-.50 vs .71), where c156 is balanced
  (.927 / .910).

No same-data FMT control was obtained before the stop, so these numbers have no
valid reference point and should not be quoted as a comparison.

## 7. Version 2.1 — the primitive was wrong, and the data says exactly how

### 7.1 The bundles are a lattice, not a cloud

`FMT_Utils/Task4C_Multiscale_4_1.center_and_neighbors` seeds every bundle on a
regular **3x3x3 cubic lattice**:

```python
offsets = [[x, y, z] for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1)]
offsets[[0, centre_index]] = offsets[[centre_index, 0]]      # centre into slot 0
distance = (prod(spacing)**(1/3)) * neighbor_grid_scale       # 0.25h / 0.5h / 1h
seeds = centre + offsets * distance
```

The six **face-adjacent** sites are therefore exactly Task1's octahedral star
`+-h e_x, +-h e_y, +-h e_z`, all at one common distance. Version 1.1 selected six
neighbours by farthest-point sampling instead, which discards that ordering.

The difference is not cosmetic. Under FPS the neighbour order is itself
rotation-invariant, so a group element induces the **identity** channel
permutation and only the component rotation survives — a much weaker constraint.
On the lattice the full Task1 action applies, rotation **and** channel
permutation, because `O_h` is the exact symmetry group of the seeding template.

**Verified on real bundles:** rotating the geometry, seeds, centroid and seeding
centre in space, then rebuilding the primitive from scratch, agrees with
`OctahedralGroup_3D.apply_group` to **max |diff| 0.00e+00 across all 48
elements** (`tests/test_task4c_octahedral_primitives.py`). Every retained line
lies exactly on the lattice — 0 off-lattice rows out of 40,000 bundles — so the
site of each surviving line is recoverable even though the cache compacts them
and does not store `offset_grid_ids`.

### 7.2 What it costs: 69.6% of bundles are dropped

A primitive is usable only when the centre and all six faces survive both the
seeding validity test (inside the same candidate head, positive `oyf`) and the
post-integration curve cleaning. Incomplete primitives are excluded from training
and prediction rather than back-filled.

| flow | split | bundles | complete | ratio | mean valid lines | mean faces | pos-rate all | pos-rate complete |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| channel | train | 13,500 | 4,143 | .307 | 18.28 | 4.789 | .1987 | .2061 |
| channel | validation | 1,500 | 464 | .309 | 18.37 | 4.803 | .1780 | .1746 |
| channel | test | 5,000 | 1,416 | .283 | 18.01 | 4.722 | .2150 | .2154 |
| tbl | train | 13,500 | 4,136 | .306 | 18.56 | 4.863 | .1435 | .1330 |
| tbl | validation | 1,500 | 432 | .288 | 18.18 | 4.779 | .1213 | .0972 |
| tbl | test | 5,000 | 1,564 | .313 | 18.62 | 4.875 | .1384 | .1349 |
| **total** | | **40,000** | **12,155** | **.3039** | | | | |

**12,155 of 40,000 cluster centres are complete — 30.4%. 69.6% are dropped.**
The rate is remarkably uniform across flows and splits (.283 to .313), and bundles
keep 18.3 of 27 lines on average but only 4.80 of the 6 faces.

Missing-face histogram (0 to 6 missing): **12,161 / 12,792 / 10,840 / 3,962 /
240 / 5 / 0**. Losing one or two faces is the common case. In 97 bundles the
centre line itself was cleaned away.

### 7.3 The offset scale explains almost all of it

| offset | complete | bundles | ratio |
|---|---:|---:|---:|
| **0.25 h** | 8,223 | 13,338 | **.6165** |
| 0.50 h | 3,244 | 13,336 | .2433 |
| 1.00 h | 688 | 13,326 | .0516 |

Wider offsets push neighbour seeds out of the candidate head region, where the
validity test rejects them. Restricting to the 0.25h scale would retain 61.7% of
that third; keeping all three scales and requiring completeness retains more
bundles in absolute terms (12,155 against 8,223), so the implementation keeps all
scales by default and exposes a scale filter.

### 7.4 Two consequences to carry forward

**The exclusion is class-neutral.** Positive rate moves by -.0015 (train), -.0124
(validation) and -.0035 (test), so dropping incomplete primitives does not
meaningfully rebalance the labels.

**The evaluation set changes, and that breaks comparability.** Test falls from
10,000 to 2,980 bundles. A score computed on the complete-only subset is **not**
comparable to c156's .920390 or to any handoff §6 baseline row, all of which are
measured on all 10,000 — this is on top of the training-set caveat in §1. Any
future comparison needs a same-subset control.

**A closed-orbit alternative, not implemented.** The full 27-site cube is also
`O_h`-closed (1 centre + 6 faces + 12 edges + 8 corners, each orbit closed), so a
27-channel primitive with a 27-way permutation and padding-aware action would
recover the dropped 70% at the cost of a wider input.

### 7.5 Files

| path | contents |
|---|---|
| `FMT_Utils/Task4C_OctVAE_2_1.py` | lattice recovery, octahedral selection, primitive builder |
| `experiments/Report_Task4C_OctahedralCoverage_1_1.py` | the coverage report above |
| `tests/test_task4c_octahedral_primitives.py` | lattice and group-action gate |
| `outputs/exp_Task4C_OctahedralCoverage_1.1/coverage.json` | machine-readable artifact |

No training has been run on version 2.1.



---

## 8. What the excluded lines actually look like

The 69.6% loss in §7.2 only matters if the discarded curves are defective. They
are not. The cache stores survivors only, so the excluded lines were regenerated
from the raw VTK: the full 27-site lattice is re-seeded, every site is integrated
with the frozen RK45 settings, and each missing face is attributed to the rule
that actually rejected it.

Figures: `outputs/exp_Task4C_ExcludedLines_1.1/excluded_{channel,tbl}_validation.png`,
12 bundles each, drawn in units of the lattice offset `h` with the centre line in
black, surviving faces in blue and excluded faces dashed by cause.

### 8.1 Almost every exclusion is region bookkeeping, not curve quality

200 bundles per flow, 1,200 face-neighbour slots each:

| cause | channel | share of exclusions | tbl | share of exclusions |
|---|---:|---:|---:|---:|
| retained | 957 (.7975) | — | 982 (.8183) | — |
| **head region** | 209 (.1742) | **.8601** | 175 (.1458) | **.8028** |
| `oyf` / domain | 28 (.0233) | .1152 | 37 (.0308) | .1697 |
| arc length | 6 (.0050) | .0247 | 6 (.0050) | .0275 |
| short / degenerate | 0 | .0000 | 0 | .0000 |

**80-86% of all exclusions are the head-component membership test.** Those seeds
passed `oyf > 0`, sat inside the domain, integrated cleanly and produced
full-length curves — they were dropped only because their cell belonged to a
*different* connected lambda2 component than the bundle's centre. The curve
cleaning rules that the handoff documents at length (points <= 2, non-finite,
axis-degenerate) fire **exactly zero times** across 2,400 slots, and the arc
length window fires on 0.5%.

### 8.2 The excluded curves are geometrically ordinary

Arc fraction is the one scale-free statistic, and it is decisive:

| arc fraction (measured / requested) | retained | head region | `oyf`/domain |
|---|---:|---:|---:|
| channel | .9992 | .9990 | .9984 |
| tbl | .9990 | .9990 | .9991 |

Every group integrates to ~99.9% of the requested length, so **no excluded line
is short, broken or degenerate** — which the zero count in the
`short / degenerate` row already implied.

#### A correction: the "farther from the centre" claim was a scale artefact

An earlier version of this section reported that excluded neighbours sit
"1.7-1.8x farther from the centre line", from a mean pointwise curve separation
in **absolute** units. That comparison is confounded. Separation scales with the
seed offset `h`, and exclusions are heavily concentrated at large `h`
(completeness .6165 at 0.25h against .0516 at 1h, §7.3), so the excluded group is
drawn disproportionately from the widest stencils. The same applies to the raw
`length` and `bounding-box` columns, which track the integration target
`ds * maxiteration` and therefore the scale mix, not the curve's character.

Measured properly — pointwise separation between the neighbour curve and the
centre curve **in units of `h`**, over 120 bundles per flow:

| | | pt0 | pt8 | pt15 | pt16 | pt23 | pt31 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **channel** retained | 592 | 3.68 | 1.89 | **1.00** | **1.01** | 1.65 | 4.08 | 2.11 |
| head region | 110 | 4.62 | 2.23 | **1.02** | **1.01** | 1.82 | 4.63 | 2.38 |
| `oyf`/domain | 14 | 4.71 | 2.58 | **1.00** | **1.02** | 2.36 | 4.80 | 2.65 |
| **tbl** retained | 566 | 17.34 | 7.03 | **1.07** | **1.09** | 5.50 | 14.24 | 7.30 |
| head region | 113 | 15.74 | 7.61 | **1.07** | **1.09** | 7.79 | 16.09 | 8.13 |
| `oyf`/domain | 30 | 12.14 | 5.71 | **1.09** | **1.15** | 4.59 | 12.95 | 6.05 |

Two things follow.

**At the seed every group is at exactly one offset, 1.00-1.15 h**, as the lattice
construction guarantees. The bidirectional curve puts the seed at its midpoint,
which is why points 15 and 16 pin to 1. Nothing about the starting geometry
distinguishes an excluded neighbour.

**The divergence along the integration is nearly the same too.** Head-region
exclusions spread 2.38 h against 2.11 h (channel) and 8.13 h against 7.30 h
(tbl) — about **11-13% more**, not 70-80%. In tbl the `oyf`/domain group is
actually *closer* than retained (6.05 h against 7.30 h), reversing the sign of
the original claim.

So the excluded neighbours are not merely "full length"; they diverge from the
centre line at almost the same rate as the ones that were kept. The visibly
wandering red curves in the tbl figure are real but are a minority within a
minority — `oyf`/domain is 17% of tbl exclusions — and they do not move the
group mean.

### 8.3 Consequence


Dropping incomplete primitives is therefore not filtering out bad geometry — it
is discarding good curves because of where their seed happened to fall relative
to a lambda2 component boundary. Two implications for the next step:

- **The 70% loss is probably recoverable.** Relaxing head-component membership
  for *neighbours* while keeping it for the centre would be well-founded — the
  neighbour curves are full-length and well-behaved, and the centre still anchors
  the bundle to one head. That single change addresses 80-86% of exclusions.
- **The `oyf` test is worth keeping on its own terms**, since it encodes the
  physical sense of spanwise rotation that defines a hairpin head, but §8.2 shows
  it is not removing curves that are geometrically distinguishable on average.

Neither change is made here; this section only establishes what the exclusions
are. Code: `experiments/Visualize_Task4C_ExcludedLines_1_1.py` and
`experiments/Report_Task4C_ExclusionCauses_1_1.py`.


---

## 9. Version 3.1 — relaxed head rule, then two-stage self-supervised training

Three changes, in the order they were made.

### 9.1 Relaxing the neighbour head rule recovers most of the data

`FMT_Utils/Task4C_RelaxedHead_1_1.py` removes the head-component test **for
neighbours only**. The centre still must lie in its own head, so the bundle stays
anchored and the label logic is untouched; neighbours still need `oyf > 0` and
the domain test. The random draws happen in the same order with the same seed
sequence, so the sampled centres are **identical** to the frozen build and only
neighbour survival differs. It is applied by rebinding the seeding function at
runtime, so no existing file is modified.

| | original | relaxed |
|---|---:|---:|
| usable bundles | 12,155 / 40,000 (**.3039**) | 32,054 / 40,000 (**.8014**) |
| 0.25h / 0.50h / 1.00h | .6165 / .2433 / .0516 | **.9104 / .8278 / .6657** |
| mean valid lines | 18.3 | 24.9 |
| mean faces present | 4.80 | 5.70 |
| train / test primitives | 8,279 / 2,980 | **21,576 / 8,041** |

Positive rate is now essentially unchanged by the exclusion (.1903 against .1903
on channel train), where before it shifted by up to .012.

### 9.2 Two-stage training

Stage 1 trains the Task1 SIREN-VAE on the octahedral primitives with **no label
and no head** — channel-balanced reconstruction, KL, contrastive on `z_inv`
across augmented views, VICReg — at **lr 5e-5** (the measured Task1 optimum) with
a 200-step warmup. Augmentation uses the exact lattice action, rotation **and**
channel permutation. Stage 2 freezes the encoder, encodes each split once, and
trains only the head.

Early stopping uses Davies-Bouldin on `z_inv`, carrying the Task1 lesson: the
latent is clustered L2-normalised and **never standardised**, and any checkpoint
whose split leaves a cluster below 2% is rejected outright rather than scored.
That guard behaved correctly — minimum cluster fraction stayed at .26-.47
throughout, so it never fired spuriously.

### 9.3 Result: the frozen self-supervised representation is weak

Seed 96721, `--group oh`, 3000 stage-1 steps.

| flows | checkpoint | head | validation F1 | test F1 |
|---|---|---|---:|---:|
| both | DB @ 400 | MLP | .3720 | .3912 |
| both | DB @ 400 | linear | .1402 | .1659 |
| both | final @ 3000 | MLP | .3680 | .3691 |
| both | final @ 3000 | linear | .1602 | .1836 |
| **channel only** | DB @ 400 | **MLP** | .4461 | **.4770** |
| channel only | DB @ 400 | linear | .1979 | .1724 |
| channel only | final @ 3000 | MLP | .4488 | .4615 |
| channel only | final @ 3000 | linear | .1862 | .1560 |

For reference, version 1.1's **end-to-end supervised** training of the same
encoder reached .6241 (MLP) and .6325 (linear), and the handoff baselines on the
full 193,000-bundle training set reach .85-.93. Test sets differ (§7.4, §1), so
these are orientations, not a ranking.

### 9.4 Three diagnostics that explain it

**Davies-Bouldin cannot work on this task.** Recording what the unsupervised
2-cluster split actually aligns with — a diagnostic, never fed back into
selection:

| | ARI vs Hairpin label | ARI vs flow identity |
|---|---:|---:|
| both flows, step 250 | .0017 | **.3116** |
| both flows, step 3000 | .0263 | **.1447** |
| channel only, step 400 | **.0068** | .0000 |
| channel only, step 3000 | **.0021** | .0000 |

With both flows pooled the dominant split is **flow identity**, not the label —
channel and TBL are different simulations on incomparable scales (x in [0, 3.13]
against [271, 352]), so Davies-Bouldin was optimising a nuisance axis. Training
one flow at a time removes it completely (ARI vs flow = 0 by construction). But
**ARI against the label stays at .002-.011**: there is no label-aligned cluster
structure for the criterion to find in the first place. Davies-Bouldin early
stopping is therefore not merely mis-tuned here, it is uninformative in
principle — unlike Task1, where the task *was* the two-way clustering.

**Checkpoint selection is not the bottleneck either.** The final checkpoint
scores within .02 of the DB-selected one on every arm, so the weakness is in the
representation, not in where training stopped.

**Per-flow training did not help.** Channel-only MLP reaches .4770 against .4739
for the same channel bundles inside the pooled run — within noise. Removing the
flow nuisance fixed the *diagnostic* without changing the *result*, which is
itself informative: the nuisance axis was not what was suppressing accuracy.

**The primitive uses 28% of the bundle.** The octahedral star is 7 lines of the
24.94 valid lines a bundle carries on average. Version 1.1 pooled over every
valid line as a centre, ~25 groups per bundle. Hairpin-ness is a property of the
whole bundle's shape, so a 7-line sub-star is a thin summary of it — the most
likely reason the frozen latent carries so little label information, and why the
linear probe (.16-.19) collapses relative to the MLP (.46-.48).

### 9.5 Where this points

The 27-site cube is `O_h`-closed (1 centre + 6 faces + 12 edges + 8 corners, each
orbit closed), so a 27-channel primitive with a 27-way permutation would keep the
exact equivariance **and** use the whole bundle instead of 28% of it. That is the
change §7.4 already flagged and the one these results argue for. It was not run.

Evidence: `outputs/exp_Task4C_RelaxedHead_1.0`,
`outputs/exp_Task4C_TwoStage_{1.1,ckpt,channel}`,
`outputs/exp_Task4C_OctahedralCoverage_relaxed`. Code:
`FMT_Utils/Task4C_RelaxedHead_1_1.py`,
`experiments/Build_Task4C_RelaxedHead_1_1.py`,
`experiments/Run_Task4C_TwoStage_1_1.py`.


---

## 10. The FMT control: relaxing the head rule is harmful, and §8.3 was wrong

§8.3 argued the 70% loss was "probably recoverable" by relaxing the neighbour
head test, on the grounds that the excluded curves are geometrically ordinary.
The frozen FMT encoder was then run on both caches to check. **It is not
recoverable: the relaxation costs accuracy.**

Identical 13,500 bundles per flow, identical labels and splits, identical FMT
recipe and 88,514-parameter MLP — only which lattice lines survived differs.

| cache | lines/bundle | seed 96611 | seed 96612 | mean test F1 |
|---|---:|---:|---:|---:|
| original | 18.28 | .6913 | .6922 | **.6918** |
| relaxed | 24.93 | .6342 | .6368 | **.6355** |
| | | | | **-.0563** |

Per flow, seed 96611: channel .7393 -> .6832, tbl .6123 -> .5535. The drop is
consistent across both seeds and both flows, and validation agrees with test
(.694 -> .638), so it is not a test-set fluctuation.

**Why §8 was right about the curves and wrong about the conclusion.** The
measurements in §8 stand: the excluded lines are full length (arc fraction .999),
non-degenerate, and diverge from the centre only 11-13% more than retained ones.
What those measurements cannot see is *which object a line belongs to*. A seed in
a neighbouring connected lambda2 component traces a perfectly good vortex line —
of a **different vortex**. Adding ~6.6 such lines per bundle contaminates a
descriptor that is supposed to characterise one candidate head, and the
classifier pays for it.

So the head-component test is not a curve-quality filter, as §8.1 correctly
established; it is a **semantic membership** filter, and it is doing real work.
Geometric indistinguishability was the wrong evidence for the question asked.

### 10.1 The same-data reference this finally provides

The FMT control also supplies the reference point missing since §1, because it
trains on the same 27,000-bundle rebuild as everything else here:

| arm | cache | training bundles | test F1 |
|---|---|---:|---:|
| **FMT + MLP** | original | 27,000 | **.6918** |
| FMT + MLP | relaxed | 27,000 | .6355 |
| OctVAE end-to-end, FPS neighbours (§6) | original | 27,000 | .6241 (MLP) / .6325 (linear) |
| OctVAE two-stage, frozen, octahedral (§9) | relaxed | 21,576 | .3912 both flows / .4770 channel |
| *handoff FMT p35, for scale only* | *frozen 1.2* | *193,000* | *.888* |

Two things follow. The 7x smaller training set costs roughly **.20 F1** on its
own (.888 against .6918), which retrospectively explains most of the gap flagged
in §6. And against a proper same-data reference, **end-to-end OctVAE is .06
below FMT**, while the **frozen two-stage arm is .19-.25 below** — so the
two-stage design, not the relaxation, is what the §9 numbers were really
measuring.

### 10.2 Revised reading of §9

The relaxed cache used in §9 is itself worth about -.056, so the two-stage arm
should be read against FMT's .6355 on that cache, not .6918. That leaves a gap of
roughly .16 (channel) to .24 (both flows) still attributable to the two-stage
design: a frozen self-supervised encoder, a criterion with no label-aligned
structure to select on (ARI .002-.011, §9.4), and a primitive that summarises
each bundle with 7 of its ~25 lines.

**Recommendation, revised.** Keep the original head rule. The route to using more
of the bundle is not to admit lines from neighbouring components but to use more
of the lines that legitimately belong to this one — the 27-site `O_h`-closed cube
(§7.4, §9.5), which stays inside the bundle's own head.

Evidence: `outputs/exp_Task4C_{LocalRebuild,RelaxedHead}_1.0/runs/regularized_fmt_mlp_seed9661{1,2}/result.json`,
configs `config/exp_Task4C_{LocalRebuild,RelaxedHead}_1.0.json`.


---

## 11. Lattice face-6 against the distance-based neighbour rules

`nearest6` is the original rule and the one behind the handoff's **.888404** row
("最近6邻居"): the six *closest* seeds by Euclidean distance. `fps6` instead picks
each next neighbour *farthest* from those already chosen, giving **.893015**, and
is what c156 (.920390) uses. `lattice6` is the new rule: the six **face-adjacent
lattice sites** in canonical `x+, x-, y+, y-, z+, z-` order.

All three run the same training-free **p35 FMT** encoder (141-D token,
`max_radius` normalisation, 6 frequencies) and the same MLP; only the neighbour
rule and the cache change.

One structural asymmetry matters for reading the table. `nearest6` and `fps6`
make **every valid line a centre**, so a bundle yields ~18-25 seven-line groups,
each encoded to 141-D and mean+max pooled into a **282-D** descriptor.
`lattice6` yields exactly **one** group per bundle, because only the lattice
centre has all six faces inside a 3x3x3 cube — a single **141-D** token, no
pooling.

| cache | strategy | train | test | mean test F1 |
|---|---|---:|---:|---:|
| original 4.14 | nearest6 | 27,000 | 10,000 | **.7058** |
| original 4.14 | lattice6 | 8,279 | 2,980 | .4441 |
| relaxed 4.14 | nearest6 | 27,000 | 10,000 | .6480 |
| relaxed 4.14 | lattice6 | 21,576 | 8,041 | .5079 |
| original 1.1 | nearest6 | 38,600 | 10,000 | **.7728** |
| original 1.1 | lattice6 | 12,913 | 2,980 | .5275 |

### 11.1 The single octahedral token is the bottleneck

`lattice6` loses .26 against `nearest6` on the original 4.14 cache (.4441 against
.7058) and .14 on the relaxed one (.5079 against .6480) — and the relaxed
comparison is the fair one, since there `lattice6` has 21,576 training
primitives, close to `nearest6`'s 27,000.

This is the attribution test §9 needed. The frozen self-supervised OctVAE scored
.3912 on the same relaxed cache and the same octahedral primitive; **p35 scores
.5079 there**, and .6480 when allowed the whole bundle. So the §9 shortfall
decomposes roughly as

* **~.14 from the primitive** — one 7-line star instead of ~25 pooled groups,
* **~.12 from the encoder** — frozen self-supervised OctVAE against training-free
  p35 on that identical primitive.

Neither alone explains it; the primitive is the larger term.

### 11.2 Relaxation helps one rule and hurts the other

The head relaxation moves `nearest6` **down** .058 (.7058 -> .6480) and
`lattice6` **up** .064 (.4441 -> .5079). Both directions follow from §10:
admitting neighbours from an adjacent lambda2 component contaminates a
descriptor built from the *whole* bundle, but for the octahedral star the
dominant effect is that 2.6x more bundles have a complete star at all
(8,279 -> 21,576 training primitives). The -.058 also reproduces the -.0563
measured with the generic `fmt_mlp` recipe in §10, so that penalty is robust
across recipes.

### 11.3 Training data scales the way the handoff implies

With `nearest6` held fixed: 27,000 -> **.7058**, 38,600 -> **.7728**, against the
published 193,000-bundle figure of **.888**. The local rebuild is therefore
tracking the benchmark's data-scaling curve, which is what the stage 1.2 rebuild
tests directly.

A note on the 1.1 cache: its training positive rate rises to .4202, because the
densification adds 100 bundles per **positive** training instance. Validation and
test keep the original rates, so the shift is a training-side rebalance, not a
change of task.

Code `experiments/Run_Task4C_LatticeNeighbours_1_1.py`, evidence
`outputs/exp_Task4C_LatticeNeighbours_1.1/summary.json`.
