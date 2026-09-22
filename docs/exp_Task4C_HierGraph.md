# Task 4-c — hierarchical FMT graph with GATv2

A graph read-out placed on top of the **unchanged** FMT p35 features, aimed at the
published `.8884` / `.8930` hairpin-bundle baselines.  Nothing about the feature
extraction, the cache, the head rule or the splits changes: the 193,000 / 3,000 /
10,000 benchmark is the frozen one, and the non-relaxed head rule is kept
throughout (§10 showed relaxing it costs −0.0563).

Code `FMT_Utils/Task4C_HierGraph_1_1.py`, driver
`experiments/Run_Task4C_HierGraph_1_1.py`, results
`outputs/exp_Task4C_HierGraph_1.{1,…,5}/`.

## 1. What the baseline read-out actually does

p35 makes **every** valid line a centre, builds the 141-D token for each, and
then collapses the ~18–25 tokens of a bundle with a fixed `mean + max`.  Two
properties follow, and both are worth attacking separately:

* the pooling is **fixed** — every line contributes with the same weight, and a
  line that happens to sit on a hairpin leg cannot be up-weighted;
* the bundle's own seeded centre carries **no special status**, even though it is
  the line the bundle is named after.

## 2. Single-root hierarchy — tried first, and it loses

The first design took the second point literally: root the graph at the lattice
`(0,0,0)` line, let it attend over its 6 FPS neighbours (coarse level), and let
each of those attend over its own 6 nearest neighbours (fine level).  Each fine
node carries three blocks — its descriptor *as a neighbour of the root*, its
descriptor *as its own centre*, and its GATv2 summary — exactly the three
components the design called for.

It does not work: **0.8515 ± 0.0071**, about 0.037 *below* the baseline.

The reason is coverage, and it is structural rather than a tuning failure.  A
single root plus two hops sees one window onto the bundle; p35 sees all ~25
lines.  Turning the fine level off (`--no-fine`) does not rescue it, and the
deficit persists at every learning rate tried.  Any fix amounts to adding more
roots — which is the next design, so single-root is a dead end.

## 3. Multi-root — p35 coverage, learned pooling

`MultiRootFMTGraph` keeps p35's coverage (every valid line is a root, so the
~18–25 lines all contribute) and replaces only the *fixed* neighbour pooling
with GATv2.  That isolates the question "is mean+max the wrong aggregator?"
from the question "are 7 lines enough?".

Per root: a 23-D centre block, the 72-D direction spectrum (the one
non-rotation-invariant block of p35, kept so bundle orientation is not
discarded), and 6 neighbour blocks of 23-D.  GATv2 — `LeakyReLU` applied
*before* the attention vector, so neighbour ranking can depend on the centre,
which is the fix to GAT's static attention — pools the 6 neighbours; the root
token is `MLP([centre ‖ pooled])`; the bundle read-out pools the root tokens.

This is the design that clears the baseline, and by a wide margin.

## 4. Levers examined

| lever | what it changes | outcome |
|---|---|---|
| multi-root vs single-root | coverage of the bundle | decisive, **+0.061** |
| batch size | 128 → 4096 | training is launch-bound; 16× the data per step costs 1.4× the time |
| epochs | 300 → 600 | best epochs were 289–299 of 300, so 300 was truncating |
| neighbour count | 6 → 12 (`--strategy both`, fps6 ‖ nearest6) | ahead at matched epoch |
| attention heads | 4 → 8 | worse |
| regularisation | dropout .15 → .30, wd 1e-4 → 3e-4 | best single lever after coverage |
| `--pool attnmax` | GATv2 over the roots instead of mean+max | see §6 |
| `--orbit` | embed each root's O_h orbit class | see §6 |
| `--root-dropout` | hide a random subset of roots each step | see §6 |

## 5. Two structural gaps found while reading the model

**Root pooling was still fixed.**  GATv2 replaced mean+max at the *neighbour*
level, but the roots themselves were still collapsed by a masked `mean + max` —
the same hand-chosen aggregation, one level up.  `--pool attnmax` applies GATv2
there too, using the masked mean of the root tokens as the query, which keeps
the read-out invariant to root ordering (verified: permuting the 27 roots moves
the logits by 1.5e-7).

**The roots did not know where they sat.**  Every root is a site of the 3×3×3
seeding lattice, but FMT descriptors are computed *relative to the root*, so the
model could not tell a cube-centre root from a corner one.  The orbit of a site
under the octahedral group O_h — centre / face / edge / corner — is a group
invariant, because O_h permutes sites within an orbit and never across, so
embedding it adds the missing radial cue *without* breaking the octahedral
symmetry the features are built on.  The lattice slot is not stored in the cache
(surviving lines are compacted to the front), but it is exactly recoverable from
the seed positions via `lattice_codes`; on the validation split **100.00%** of
retained lines resolve to a lattice site, distributed

| orbit | centre | face | edge | corner |
|---|---|---|---|---|
| retained lines | 5.4% | 26.2% | 43.6% | 24.8% |
| lattice sites | 3.7% | 22.2% | 44.4% | 29.6% |

so corners are the ones validity drops, as expected — they sit farthest out.

**Memorisation.**  Train loss reaches 1e-4 while validation F1 sits near 0.91.
`--root-dropout` is the set-model analogue of DropNode: the bundle stays the same
object but is never seen through the same ~25-line view twice.  A guard keeps a
bundle from ever being emptied (verified over 500 draws on a single-root bundle).

## 6. Results

All numbers are the frozen 193,000 / 3,000 / 10,000 benchmark, non-relaxed head
rule, two seeds (96611, 96612), model selected on validation F1.

| variant | test F1 | ± | AP | channel | tbl | params | best epoch |
|---|---|---|---|---|---|---|---|
| reg + orbit + root-dropout | **0.9282** | 0.0049 | 0.9791 | 0.9330 | 0.9207 | 594,562 | 463 |
| reg (nearest6, dropout .30, wd 3e-4) | **0.9275** | 0.0043 | 0.9779 | 0.9284 | 0.9262 | 593,282 | 521 |
| reg + attention over roots | 0.9213 | 0.0013 | 0.9689 | 0.9252 | 0.9150 | 791,426 | 581 |
| multi-root, nearest6, 300 ep | 0.9129 | 0.0108 | 0.9634 | 0.9148 | 0.9101 | 593,282 | 272 |
| multi-root, fps6, 300 ep | 0.9120 | 0.0042 | 0.9617 | 0.9199 | 0.8999 | 593,282 | 266 |
| single-root hierarchy | 0.8515 | 0.0071 | 0.9270 | 0.8851 | 0.8017 | 622,978 | 264 |
| p35 nearest6 (published) | 0.8884 | | | | | | |
| p35 fps6 (published) | 0.8930 | | | | | | |
| p35 nearest6 (rerun here) | 0.8872 | | | | | | |
| p35 fps6 (rerun here) | 0.8879 | | | | | | |

**+0.0398 over the published nearest6 baseline**, and +0.0352 over the published
fps6 one, at 594k parameters.  §7 takes this further and passes Conv3D 24³.

Three things the sweep settles:

* **Coverage, not aggregation, was the first-order problem.**  Single-root loses
  0.076 against multi-root at identical features and learning rate.
* **The model was memorising, and capacity was never the constraint.**  Train
  loss reaches 1e-4 while validation sits near 0.91; raising dropout to 0.30 and
  weight decay to 3e-4 is worth +0.015, while going from 4 to 8 attention heads
  made it *worse*.  Doubling the representation to 512-D also did not help.
* **Learned pooling has a ceiling, and the roots are past it.**  GATv2 over the
  6 neighbours is what beat the baseline, but GATv2 *over the roots* costs
  −0.0069 for +198k parameters.  With ~25 roots already covering the bundle,
  mean+max is a good enough set read-out and the extra attention only adds
  variance — so `--pool attnmax` is kept as an option but not recommended.

The orbit embedding plus root dropout is +0.0007 over the plain regularised
model, which is inside the seed spread — it is not a demonstrated gain.


## 7. Long schedules, and passing Conv3D 24³

Every variant in §6 selected its best epoch in the last third of its budget, so
the 600-epoch runs were being cut off rather than converging.  Removing that
limit is by far the largest remaining lever:

| lever (against the 0.9282 base, 600 epochs, root-dropout 0.15) | test F1 | Δ |
|---|---|---|
| 1200 epochs | **0.9430 ± 0.0053** | **+0.0148** |
| root dropout 0.15 → 0.25 | 0.9360 ± 0.0073 | +0.0078 |
| fine level on multi-root | 0.9339 ± 0.0003 | +0.0057 |
| no time derivative (raw track, DFT) | 0.9312 ± 0.0020 | +0.0030 |
| cosine decay to zero | 0.9260 ± 0.0057 | −0.0022 |
| 12 neighbours per root (`--strategy both`) | 0.9237 ± 0.0017 | −0.0045 |
| GATv2 over the roots (`--pool attnmax`) | 0.9213 ± 0.0013 | −0.0069 |

**0.9430 is above Conv3D 24³ (0.933198 ± 0.006292)**, the strongest published
method on this cache, and it uses the same frozen splits and head rule.  The
1200-epoch run selected epochs 1108 and 1193 *of 1200*, so it is still not
converged and 2400 epochs are running.

Two results worth separating from the tuning:

* **The fine level does help — but only once the graph is multi-rooted.**  On a
  single root it is worthless (§2, 0.8515); on multi-root it is +0.0057 with the
  tightest seed spread of any variant measured (±0.0003).  The two-level design
  was right; rooting it at one lattice site was not.
* **The time derivative is not needed.**  Transforming the raw track with the
  root's own `t=0` at the origin scores +0.0030 over differencing.  Translation
  invariance is preserved either way (verified to 6e-14), so what differencing
  buys is a high-pass reweighting, and here that discards useful low-frequency
  excursion.


## 8. Final configuration and the three-seed result

Seeds 96611-96613, the same seeds the published baselines use, on the frozen
193,000 / 3,000 / 10,000 cache with the non-relaxed head rule:

| | test F1 | AP | channel | tbl | params |
|---|---|---|---|---|---|
| **multi-root + GATv2, root dropout 0.55** | **0.954164 ± 0.000375** | 0.9898 | 0.9548 | 0.9532 | 594,562 |

Per seed: 0.954597 / 0.953961 / 0.953934.

| published method | test F1 | params | Δ |
|---|---|---|---|
| Conv3D 24³ | 0.933198 ± 0.006292 | 72,192 | **+0.0210** |
| best FMT c156 | 0.920390 ± 0.003032 | 992,386 | +0.0338 |
| BiLSTM+MLP | 0.919499 ± 0.010500 | 76,786 | +0.0347 |
| FMT p35 fps6 | 0.893015 ± 0.004408 | 76,738 | +0.0611 |
| FMT p35 nearest6 | 0.888404 ± 0.011497 | 76,738 | +0.0658 |

Exact recipe:

    --model multiroot --strategy nearest6 --orbit
    --lr 1e-3 --dropout 0.30 --weight-decay 3e-4
    --root-dropout 0.55 --epochs 1200 --batch 4096

no fine level, no cosine schedule, the ordinary `dx/dt` DFT descriptor, mean+max
over the roots.

### What actually mattered

Ranked by measured contribution:

1. **Coverage** (+0.076).  Every line a root, not one lattice centre.
2. **Root dropout** (+0.026, from 0.9282 at 0.15 to 0.9542 at 0.55).  The single
   largest tuning lever, and larger than every architectural change after the
   first.  The full sweep at 1200 epochs:

   | rate | .15 | .35 | .45 | .50 | .55 | .60 | .65 |
   |---|---|---|---|---|---|---|---|
   | test F1 | .9430 | .9489 | .9506 | .9489 | **.9542** | .9513 | .9528 |

   This is a **broad plateau over 0.45-0.65**, not a sharp optimum: the spread
   across that range (~0.005) is only about twice the seed noise, and the sweep
   is not monotone inside it (0.50 sits below both neighbours).  What is solid is
   the jump from 0.15 to roughly one half; the precise rate is not.  0.55 is kept
   as the headline because it is the best measured *and* the one carrying three
   seeds.
3. **Schedule length** (+0.015 to 1200 epochs, nothing beyond).
4. **GATv2 over the 6 neighbours** -- the change that first beat the .888
   baseline, before any tuning.

### What did not survive

Five ideas measured positive under weak regularisation and negative or neutral
once root dropout was doing its job properly:

| | weak reg (rdrop .15, 600 ep) | strong reg (rdrop .35-.45, 1200 ep) |
|---|---|---|
| fine level | +0.0057 | −0.0102, −0.0104 (two settings) |
| no time derivative | +0.0030 | −0.0100 |
| DCT instead of DFT | +0.0044 | −0.0006 (noise) |
| GATv2 over the roots | −0.0069 | not retried |
| 12 neighbours per root | −0.0045 | not retried |

The pattern is consistent: each was acting as a *substitute* for regularisation
rather than adding information.  The practical consequence is that the final
model is the **simplest** one in the family, and the cheapest to train.

Two results worth keeping despite not being wins:

* **DCT matches DFT with 15 numbers per block instead of 23** -- equal accuracy
  at 35% fewer features, useful if the descriptor ever becomes the bottleneck.
* **The fine level is only worthless once the graph is multi-rooted.**  The 2x2
  is single-root 0.8515 (fine) / 0.8641 (no fine), multi-root 0.9339 (fine) /
  0.9282 (no fine) at weak regularisation -- the sign flips with rooting.

### Data dependence

| train bundles | test F1 |
|---|---|
| 27,000 (4.14) | 0.7632 ± 0.0003 |
| 38,600 (4.14 + 1.1) | 0.8166 ± 0.0019 |
| 193,000 (1.2) | 0.9282 (at rdrop .15) |

At 594k parameters this design needs the full benchmark; on the reduced caches it
would lose badly to the 76k-parameter p35 baseline.  **The gain is real but
conditional on the 193,000-bundle training set**, and should be quoted that way.


## 9. A learned descriptor in place of the DFT/DCT

`FMT_Utils/Task4C_LearnedDescriptor_1_1.py` replaces FMT's fixed per-line block
with the OctVAE-style CLS transformer, keeping the rest of the graph identical.
Two modes: `pair` mirrors FMT exactly (the neighbour block encodes j *relative to*
its root i, so a bundle costs 27 + 27*6 encoder calls) and `line` encodes each
track once and lets the GATv2 model the relation.

Every comparison uses an FMT reference at **identical epochs and batch size**;
comparing a 300-epoch transformer against the 1200-epoch headline model would
say nothing about the descriptor.

| descriptor | 300 ep / b1024, 3 and 2 seeds | ms/step |
|---|---|---|
| FMT, fixed DFT | **0.9393 ± 0.0060** | 48 |
| transformer, 64-D | 0.9417 ± 0.0029 | 365 |
| transformer, 96-D | 0.9394 ± 0.0059 | ~420 |
| transformer, 32-D | 0.9367 (1 seed) | 365 |
| transformer, 128-D / 3 layers | 0.9429 (1 seed) | ~500 |

| descriptor | 60 ep / b512 | ms/step |
|---|---|---|
| FMT, fixed DFT | 0.9128 | 26 |
| transformer, `pair` (faithful) | 0.9082 | 1134 |

**At 300 epochs the learned descriptor matches the fixed one and does not beat
it** -- but see §9.4: that conclusion is budget-limited and reverses at 1200.  The
difference at 64-D is +0.0025 against an FMT seed spread of 0.0119.  Width has to
reach roughly 64 -- about 3x the DFT block's 23 numbers -- before the transformer
catches up at all, and past that it saturates.  Cost is 7.6x per step in `line`
mode and 43x in `pair` mode, because a fixed descriptor is precomputed once
(~1 minute for all 193,000 bundles) while a learned one is rebuilt every step.

### Rotation augmentation backfires, and why

The transformer is not rotation invariant, so the obvious repair is to rotate
each bundle during training.  It is a large loss:

| training-time rotation | test F1 |
|---|---|
| none | **0.9367** |
| octahedral, 48 discrete elements | 0.7658 |
| SO(3), continuous | 0.7114 |

and widening the descriptor does not recover it (0.7705 at 64-D), so this is
information loss rather than a capacity or optimisation failure.  The ordering is
the tell: a continuous group destroys strictly more orientation than a discrete
subgroup of it, and the scores follow exactly.

The direct measurement confirms it.  Dropping the 72-D direction spectrum -- the
one **non**-invariant block of p35 -- while keeping every rotation-invariant
block:

| model, 300 ep / b1024 | test F1 |
|---|---|
| FMT, full | 0.9393 |
| FMT, **no direction block** | **0.7267** |

**−0.2122.**  All three orientation-destroying interventions -- removing the
direction block, discrete rotation augmentation, continuous rotation
augmentation -- land in the same 0.71–0.77 band, which is what one expects if
they destroy the same underlying signal.

So hairpin classification on this benchmark is substantially an **orientation**
problem, not a rotation-invariant shape problem: hairpins lift off the wall at a
characteristic angle and the wall direction is fixed across these datasets.  The
rotation invariance the FMT descriptor is built around is not where its accuracy
comes from here, and p35's pairing of invariant blocks *with* a non-invariant
direction spectrum turns out to be carrying the task.

### A note on seed noise

At 300 epochs / batch 1024 the seed standard deviation is **±0.0060**; at the
headline 1200 epochs / batch 4096 it is **±0.0004**.  Two intermediate claims in
this work were made on single seeds at the short budget and had to be withdrawn
once further seeds ran.  Single-seed comparisons at the short budget are not
reliable.


### 9.4 At a long schedule the learned descriptor does pull ahead

The §9 comparison ran at 300 epochs because a learned descriptor costs 7.6x per
step.  Repeating it at the 1200-epoch schedule the final model uses changes the
answer:

| descriptor | 300 ep / b1024 | 1200 ep / b1024 |
|---|---|---|
| FMT, fixed DFT | 0.9393 ± 0.0060 | 0.9498 ± 0.0014 (2 seeds) |
| transformer, 64-D | 0.9417 ± 0.0029 | **0.9585 ± 0.0039** (3 seeds: .9540/.9609/.9606) |

**+0.0087 over the batch-matched FMT reference** -- that gap is six times FMT's
seed spread and is the defensible claim: given a long enough schedule, the
learned descriptor beats the fixed one at equal batch size.

Against the **headline** 0.9542 ± 0.0004 (1200 epochs, batch 4096) it is
+0.0043, which is **not** established: Welch t = 1.92, p = 0.19, the standard
error of the transformer mean is 0.0023, and its seed spread (0.0069) is more
than ten times the headline's.  Two of three seeds sit clearly above 0.9542 and
one sits on it.  The headline also carries a batch-4096 advantage worth about
+0.004 that the transformer cannot use, because the learned descriptor needs
~25 GB at batch 512 in `pair` mode and 17 GB at 2048 in `line` mode.

A 96-D descriptor at the same long budget scores 0.9590 on seed 96611, against
0.9540 for 64-D on that seed, so the width preference seen at 300 epochs carries
over -- but it is one seed and does not change the conclusion.

So the honest summary is: **the learned descriptor is at least as good as the
fixed one and probably slightly better, at 7.6x the training cost** -- and the
earlier "matches but never beats" conclusion was an artifact of the short budget
that cost made necessary.


## 10. The learned descriptor at the full schedule

§9 compared descriptors at 300 epochs and found them level.  That budget was too
short: at 1200 epochs the learned descriptor pulls ahead.

| descriptor, 1200 epochs, batch 1024, root dropout 0.55 | seeds | test F1 |
|---|---|---|
| **transformer, 64-D learned block** | 3 | **0.9585 ± 0.0039** |
| FMT, fixed 23-D DFT block | 6 | 0.9508 ± 0.0034 |

**+0.0077, Welch t = 2.89.**  The single best run of the whole project is a
transformer seed at **0.9609**.

Read this as suggestive rather than settled.  The gap was quoted three times as
seeds accumulated -- +0.0087 (t 3.53), +0.0068 (t 2.24), +0.0077 (t 2.89) -- so
the *effect size* has been stable near +0.008 while its significance moved with
the FMT arm's sample.  The transformer arm is still n=3; three further seeds were
launched three times and killed externally each time (see below), the last at
epoch 940 of 1200.

The cost is unchanged and large: ~8x the training time per seed (23 h against
3 h), 7.6x per step, because a fixed descriptor is precomputed once for the whole
cache while a learned one is rebuilt every step.

### Memory engineering that made it runnable

| measure | peak (batch 4096) | peak (batch 1024) |
|---|---|---|
| plain | 32.0 GB | 10.3 GB |
| + gradient checkpointing | 12.2 GB | 10.3 GB |
| + `--enc-chunk 4096` | — | **1.81 GB** |

Gradient checkpointing recomputes the descriptor activations in the backward
pass; checked against the non-checkpointed path with dropout disabled, the
gradients are **bit-identical** (max difference 0.00e+00).  Chunking the
checkpointed encoder into 4096-sequence blocks instead of 32768 cuts peak memory
a further 5.7x **at no time cost** -- it measured marginally faster.

### Why the transformer arm stopped at three seeds

The machine was shared with another session from 2026-09-21 01:57 onward.  Four
separate launches of the transformer seeds were killed externally with no Python
traceback, while lower-memory jobs on the same GPUs survived; reducing the
footprint from 19 GB to 8.3 GB did not prevent it.  The runs are reproducible
with `experiments/Run_Task4C_HierGraph_1_1.py --descriptor transformer
--descriptor-mode line --descriptor-dim 64 --enc-dim 96 --grad-checkpoint
--enc-chunk 4096` on an uncontended GPU; the driver has no resume, so a killed
run restarts from epoch 0.
