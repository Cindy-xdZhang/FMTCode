# Task 6 — reproduction guide (IcoVAE coreline classification)

How to rebuild the dataset, train the model, and reproduce the reported numbers on
`Task6_CorelineDataset_2.7_share`. Written for someone who has the official code repo.

Background, the full experiment record and every negative result are in
[`exp_Task6_NewData_IcoStar.md`](exp_Task6_NewData_IcoStar.md); the method overview across
Task 1 and Task 6 is in [`IcoVAE_handover.md`](IcoVAE_handover.md). **This document is the one
to follow to reproduce results.**

---

## 1. Headline

| | |
|---|---|
| Task | per-point vortex coreline classification, 6 scenes, 77 train / 24 test frames |
| Model | icosahedral SIREN-VAE encoder + MLP head, 587,501 parameters |
| **Result** | **macro-F1 0.9549 ± 0.0018** (3 seeds) |
| Coreline coverage | **98.6 %** (1 h boundary margin) |
| Training cost | ~85 s per run on one A100-40GB |
| Auxiliary losses | **none** — no SSL, no decoder reconstruction |

Dataset variants (temporal split, tighter positive rules) are in §11.

The recommended configuration is a **single icosahedral shell at radius 1 h**. It needs only a
1 h boundary margin, which keeps 98.6 % of the coreline available for positive sampling. Larger
shells score higher on their own benchmark but force a larger margin that discards up to a fifth
of the coreline — see §7.

---

## 2. Reproduce in one command

```bash
bash experiments/reproduce_task6.sh /path/to/Task6_CorelineDataset_2.7_share
```

This checks the dataset revision, builds the cache, **verifies it**, trains three seeds and
prints the result against the reference. It fails loudly rather than training on a cache that
would not reproduce. Roughly 30 minutes end to end on one A100.

If you prefer to run the steps by hand, follow §5–§8 — but run the §6 verification either way.

### What this does NOT use

**No `preintegrated/` directory is read at any point** — not `icosa_bundles_1.1`, not
`icosa__p96`, not `icosa_bundles_max80_1.1`. Every streamline is produced by the official
`frame.integrate(...)` in `task6_fields.py` from the raw velocity fields. Training on the
shipped preintegrated curves instead is the single most likely cause of a large shortfall,
because those bundles use the official `IVD > 0.8 x max` centre rule (§13) and a different
neighbour geometry (20 face centres at 1 h, not 12 vertices).

## 3. Environment

```
python 3.11.16
torch  2.12.0+cu126      (CUDA 12.6, one GPU is enough)
numpy  1.26.4
scipy  1.17.1
numba  0.67.0            (required by the official integrator)
```

`conda activate pyflowvis` in this setup. One A100-40GB; the recommended configuration holds all
data on the GPU and peaks near 12 GB. For larger caches add `--data-device cpu`.

## 4. Data

Official package, unmodified:

```
/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share
    task6_fields.py      Dataset / Frame / frame.integrate(...)
    dataset.json         101 frames
    default_split.json   77 train / 24 test / 0 unused
```

Scenes: `cylinder3d`, `halfcylinderRe320`, `halfcylinderRe640`, `deltaWing_resampled`,
`SquareCylinder`, `tornado3d`. **`tornado3d` is test-only** — it has no training frame, so it
measures transfer to an unseen scene.

### The official package is read-only

**Nothing in the dataset folder was modified by this work.** No `.npy`, `.npz`, `.vtp` or `.json`
file under `Task6_CorelineDataset_2.7_share` was written, moved or deleted; the only side effect
of running the code is Python creating `__pycache__/` directories when `task6_fields.py` and
`task6_bundles.py` are imported, which are safe to delete.

Everything this work uses is **derived data**, written to a separate directory
(`/home/cheny1a/data/task6_icostar_*`) and regenerated entirely from the official package by the
command in §4. Nothing needs to be shipped alongside the dataset — the derived caches can be
deleted and rebuilt at any time. The only artefact worth keeping is
`docs/assets/temporal_split.json` (§13.1), and even that is reproducible from the snippet there.

Every read goes through the official API: `Dataset`, `Frame.velocity`, `Frame.corelines()` and
`Frame.integrate(...)` from `task6_fields.py`, plus `core_points.npy` / `core_offsets.npy` /
`ivd.npy` read directly. Streamlines are produced only by the official
`frame.integrate(...)` — no independent integrator is used anywhere.

> **Note on `python task6_fields.py verify`.** This currently fails on a pre-existing packaging
> gap: `provenance/manual_history_before_field_editor.zip` is listed in `dataset.json`'s
> `files_sha256` but is absent from the distribution. The file was already missing before this
> work started (everything under `provenance/` is dated 22 Sep) and is unrelated to the frame
> data, corelines or splits. All per-frame source hashes still match — the preintegrated bundle
> reader checks them on every access and raises otherwise.

Verify the package before starting:

```bash
cd /home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share
python task6_fields.py verify
```

### Label rule

A centre is positive **iff its distance to the continuous coreline segments is strictly less
than `h`**, where `h = min(grid spacing)`. Distance is to the *continuous polyline*, not to its
sampled vertices. This is the official rule, and the implementation here reproduces the official
`labels.npy` **exactly — 1 010 000 / 1 010 000 centres, 0 mismatches** across all 101 frames.
Nearest-vertex distance achieves only 99.8653 % and an `h/10` densification 99.9982 %, so exact
point-to-segment distance is used (`segment_distance` in the builder).

## 5. Build the star cache

```bash
cd <this repo>
ROOT=/path/to/Task6_CorelineDataset_2.7_share     # <-- your dataset package
OUT=$HOME/data/task6_icostar_r0250510             # <-- where the cache goes
for I in 0 1 2 3 4 5 6 7; do
  python experiments/Build_Task6_New_IcoStar_1_1.py \
      --root $ROOT --out $OUT --radii 0.25,0.5,1 --count 3000 \
      --threads 8 --shards 8 --shard $I &
done; wait
```

`--root` is required on any machine other than the one these numbers were measured on;
without it the builder falls back to a hard-coded path. `TASK6_ROOT` works as an environment
variable alternative.

~25 min wall clock over 8 shards, **8.0 GB** on disk, 101 frames. The cache stores radii
0.25 / 0.5 / 1 h; the recommended run slices the 1 h shell out of it with `--shells 1`, so one
build covers every configuration in §8.

Each frame file holds `curves [n, 1+12k, 65, 3] float32`, `centres`, `labels`, `dist` (distance
to coreline in units of h), `kind` (0 positive / 1 hard / 2 far), `role`, `key`, `h`, `radii`,
`offsets`, `total_length`.

### What the builder does

* **Geometry.** Centre streamline plus 12 neighbours at the vertices of a regular icosahedron,
  one fixed global orientation, at each requested radius. The fixed orientation is what makes
  the `I_h` action an exact channel permutation (120 elements; verified in
  `tests/test_icosahedral_group_3d.py`).
* **Integration.** The official `frame.integrate(...)`: bidirectional arc length
  `--total-length` (default 0.5 ≈ 40 h), resampled to 65 points. A star is kept only if **all**
  of its lines pass the returned `valid` mask, as the dataset README requires.
* **Seeding.** Centres are stratified so every scene carries positives — 25 % on the coreline
  tube (`d < h`), 35 % in a hard shell (`h ≤ d ≤ 8h`), 40 % from the old `IVD > mean` region.
  This departs deliberately from the official `IVD > 0.8 × max` rule, which yields zero
  positives in three of six scenes (§11).
* **Margin.** Centres stay `max(radii) × h × 1.05` clear of the boundary so no neighbour seed is
  ever clipped. With `--radii 0.25,0.5,1` that margin is 1 h.

## 6. Verify the cache before training

```bash
python experiments/Check_Task6_Cache_1_1.py $OUT
```

It must end with `CACHE MATCHES THE GUIDE`. The expected values are:

| property | expected |
|---|---|
| frames | 101 (77 train / 24 test) |
| radii | `[0.25, 0.5, 1.0]` → 1 h boundary margin |
| lines per sample | 37 (centre + 12 x 3 shells) |
| points per line | 65 |
| total_length | 0.5 |
| positive_scale | 1.0 |
| split | package default (interleaved) |
| overall positive rate | 0.20 – 0.30 (reference: 0.2392) |
| scenes with zero positives | none |

**If any scene reports zero positives, stop.** That means the centres came from the official
`IVD > 0.8 x max` rule, which contains no positive examples at all in `SquareCylinder`,
`cylinder3d` and `halfcylinderRe320`. Training proceeds without error and produces a
plausible-looking pooled score while three of six scenes sit at the majority-class floor.

## 7. Train

```bash
python experiments/Run_Task6_NewStar_IcoVAE_1_1.py \
    --cache /home/cheny1a/data/task6_icostar_r0250510 \
    --shells 1 \
    --lr 1e-3 --schedule cosine --warmup-steps 400 \
    --class-weight --steps 4000 --batch 512 \
    --seed 11 --label repro_seed11 \
    --save-weights outputs/weights_Task6/repro_seed11.pt
```

Repeat with `--seed 23` and `--seed 37` and average. Each run writes
`outputs/exp_Task6_NewStar/<label>.json` with the full F1-vs-step curve and a per-scene
breakdown at every eval step.

### Every hyper-parameter

| flag | value | why |
|---|---|---|
| `--shells` | `1` | single 1 h shell; see §6 |
| `--lr` | `1e-3` | 1e-4 is clearly worse (−0.014); 2e-3 no better |
| `--schedule` | `cosine` | **+0.005 over constant** — larger than any objective effect |
| `--warmup-steps` | `400` | needed at 1e-3; constant LR without warmup can diverge |
| `--steps` | `4000` | 12 000 adds nothing; 24 000 overfits after ≈17 000 |
| `--batch` | `512` | default |
| `--class-weight` | on | best arm at full supervision |
| `--z-inv` / `--z-eq` | `16` / `12` | split latent; flat in sweeps |
| `--head` | `mlp` | linear is ≈0.003 worse |
| `--clip-sigma` | `5.0` | default |
| `--encoder` | `conv` | conv + transformer; `pure` transformer is no better |
| `--lambda-ssl` | **0** | contrastive costs −0.0012 at full supervision |
| `--lambda-recon` | **0** | decoder costs −0.0011; at weight 10, −0.0127 |
| `--zinv-var-weight` | **0** | VICReg term costs −0.0018 |
| `--total-length` | 0.5 *(build time)* | plateau 0.125–0.5; 2.0 costs 0.019 |
| `--points` | 65 *(build time)* | 129 changes nothing (p = 0.78) |

**The auxiliary losses are off deliberately** — they are measured to be slightly harmful at full
supervision. Turn them on only when labels are scarce (§8).

## 8. Expected numbers

Shell configurations on the 1 h-margin cache (98.6 % coverage), lr 1e-3, 3 seeds unless noted:

| shells | macro-F1 | note |
|---|---:|---|
| **1** | **0.9549 ± 0.0018** | **recommended** |
| 0.25,0.5,1 | 0.9513 ± 0.0011 | all three shells |
| 0.5 | 0.9512 ± 0.0005 |  |
| 0.25,1 | 0.9491 ± 0.0010 |  |
| 0.5,1 | 0.9485 ± 0.0004 |  |
| 0.25 | 0.9484 ± 0.0008 |  |
| 0.25,0.5 | 0.9464 ± 0.0006 |  |

Learning rate, verified on the three-shell configuration (3 seeds each):

| lr | macro-F1 |
|---:|---:|
| 5e-04 | 0.9507 ± 0.0022 |
| **1e-03** | **0.9513 ± 0.0011** |
| 2e-03 | 0.9461 ± 0.0004 |

**Adding shells hurts here.** A single 1 h shell (0.9549) beats all three (0.9513).
Closely spaced shells are largely redundant and only add channels. Widely separated shells *do*
help — 1 h + 8 h beat any single shell in earlier sweeps — but those require an 8 h margin, which
is exactly what §7 rules out.

Per-scene F1 of the recommended configuration:

| scene | F1 | note |
|---|---:|---|
| SquareCylinder | 0.9774 |  |
| cylinder3d | 0.9538 |  |
| deltaWing_resampled | 0.9823 |  |
| halfcylinderRe320 | 0.9492 |  |
| halfcylinderRe640 | 0.9243 |  |
| tornado3d | 0.9862 | **test-only — no training frame, so this is transfer to an unseen scene** |

Dataset sizes for this configuration: **224,938 train / 70,396 test stars**,
test positive rate 0.2412.

### Baseline

```bash
python experiments/Run_Task6_NewStar_Conv3D_1_1.py \
    --cache /home/cheny1a/data/task6_icostar_r0250510 --shells 1 \
    --arch wide --channels 32,64,128 --hidden 320 --label conv3d_609k
```

A capacity-matched Conv3D voxel-splat baseline (608 929 parameters), reusing the repo's frozen
`bundle_voxels` and `Conv3DClassifier`. On the 8 h cache it reaches 0.9267 against IcoVAE's
0.9649 on identical data — **+0.038 for IcoVAE at matched capacity** — and it saturates near
0.933 even at 2.16 M parameters, so the gap is representational rather than a capacity artefact.

## 9. Why a 1 h margin

Centres must sit `max(radii) × h × 1.05` clear of the boundary so every neighbour seed stays in
the domain. Positive centres are perturbed coreline points, so **any coreline inside that margin
contributes no positives at all**. Measured by `experiments/Analyze_Task6_MarginLoss_1_1.py`:

| margin | coreline reachable | worst scene (`halfcylinderRe640`) |
|---:|---:|---:|
| **1 h** | **98.59 %** | **93.41 %** |
| 2 h | 96.44 % | 77.84 % |
| 4 h | 91.35 % | 53.47 % |
| 8 h | 79.21 % | 12.47 % |

`halfcylinderRe640`'s grid is 160×60×20, so its z axis spans only **19.9 h**; an 8 h margin
removes 16.8 h of it, confining centres to a 3.1 h slab.

A 1/2/4/8 h star scores higher (0.9662) — but on a benchmark missing a fifth of the coreline.
Evaluated **on the same spatial region**, the wide star's advantage is **+0.0022, p = 0.106 —
not significant**; roughly 79 % of the apparent gap is test-set difficulty, not model quality.
A 1 h margin is therefore the right default: near-complete coverage at no statistically
detectable cost.

## 10. If labels are scarce

Auxiliary objectives are harmful at full supervision but help sharply below ≈5 % labels, and
which one helps depends on the regime:

| labels | recommended flags | gain |
|---:|---|---:|
| 224,938 (100 %) | none | — |
| ≈11 000 (5 %) | none | ±0.003 |
| ≈2 200 (1 %) | `--lambda-ssl 1.0 --lambda-recon 1.0` | +0.021 |
| ≈440 (0.2 %) | `--lambda-ssl 0.1 --lambda-recon 1.0` | **+0.060** |

Use `--label-fraction` to subsample the labelled pool; SSL and reconstruction still see every
star. The contrastive term **alone** costs −0.064 at 0.2 % labels — it is only useful alongside
the decoder. Two-stage pretraining (`--pretrain-steps`) loses to joint training in every setting
tested, and every frozen linear probe sits at the majority-class floor.

## 11. Code map

| file | role |
|---|---|
| `experiments/reproduce_task6.sh` | **end-to-end reproduction: build, verify, train 3 seeds** |
| `experiments/Check_Task6_Cache_1_1.py` | **cache verifier — run before training** |
| `experiments/Build_Task6_New_IcoStar_1_1.py` | dataset builder (stratified seeding, exact labels) |
| `experiments/Run_Task6_NewStar_IcoVAE_1_1.py` | training / evaluation |
| `experiments/Run_Task6_NewStar_Conv3D_1_1.py` | Conv3D voxel baseline, width-configurable |
| `experiments/Analyze_Task6_MarginLoss_1_1.py` | coreline coverage vs margin |
| `experiments/Analyze_Task6_NewStar_Distance_1_1.py` | accuracy vs distance-to-coreline |
| `experiments/Analyze_SO3_ChannelMap_1_1.py` | continuous-SO(3) view diagnostic |
| `FMT_Utils/IcoVAE_3D.py` | encoder, decoder, objective terms *(pre-existing)* |
| `FMT_Utils/IcosahedralGroup_3D.py` | 12 vertices, 120 group elements, permutations *(pre-existing)* |
| `tests/test_icosahedral_group_3d.py` | 7 group checks; all should pass |

## 12. Other useful options

Builder:

| flag | purpose |
|---|---|
| `--radii` | shell radii in units of h; also sets the boundary margin |
| `--total-length` | streamline arc length (build time) |
| `--points` | samples per streamline (default 65) |
| `--centres-from` | reuse another cache's centres, for controlled comparisons |
| `--official-centres` | take centres from `preintegrated/icosa_bundles_1.1`, reproducing the official positive set exactly |
| `--count` | centres proposed per frame before the validity filter |
| `--pos-frac` / `--hard-frac` / `--hard-max` | seeding strata |

Trainer:

| flag | purpose |
|---|---|
| `--restrict-to` | evaluate only on centres common to several caches |
| `--min-margin` | keep centres this many h from the boundary, to compare caches on one region |
| `--data-device cpu` | stream batches for caches too large for GPU memory |
| `--scenes` | train and test on a single scene |
| `--save-weights` | archive encoder, head and normalisation stats |

## 13. Dataset variants

Sections 4–6 build the dataset with the **package split** and the **official `d < h` positive
rule**. Two variants of the data definition are supported; both are build-time choices.

### 11.1 Split — package default vs temporal

The package split interleaves frames in time: `cylinder3d` trains on frames 75-142 and tests on
81-136, so most test frames are bracketed by training frames only a few timesteps away. A
temporal split removes that leakage by giving each scene its earliest frames for training and its
latest for test, keeping the per-scene 77 / 24 counts.

```bash
# generate the temporal split once (writes outputs/temporal_split.json)
python - <<'EOS'
import sys, json, collections
R='<dataset root>'; sys.path.insert(0, R)
from task6_fields import Dataset
ds = Dataset(R)
by = collections.defaultdict(list); counts = collections.defaultdict(lambda: [0, 0])
for i, role in enumerate(('train', 'test')):
    for k in ds.split[role]:
        s = k.split(':')[0]; by[s].append(int(k.split(':')[1])); counts[s][i] += 1
train, test = [], []
for s in sorted(by):
    idx = sorted(by[s]); ntr, nte = counts[s]
    train += [f"{s}:{i}" for i in idx[:ntr]]
    test  += [f"{s}:{i}" for i in idx[len(idx) - nte:]]
json.dump({'train': train, 'test': test, 'unused': [],
           'dataset_sha256': json.load(open(R + '/default_split.json'))['dataset_sha256']},
          open('outputs/temporal_split.json', 'w'), indent=1)
EOS

# then build with it
python experiments/Build_Task6_New_IcoStar_1_1.py \
    --out $OUT --radii 0.25,0.5,1 --count 3000 \
    --split-file outputs/temporal_split.json \
    --threads 8 --shards 8 --shard $I
```

| split | macro-F1 (3 seeds) | note |
|---|---:|---|
| package default (interleaved) | 0.9549 ± 0.0018 | comparable with §6 |
| **temporal (early → late)** | **0.9395 ± 0.0019** | leak-free; **recommended for new work** |

The temporal split costs **-0.0154**, which measures the temporal leakage in the package
split. Every number in §6 uses the package split and is inflated by roughly that amount. The
temporal split still clears 0.94, so prefer it unless you need to compare against §6 directly.

### 11.2 Positive rule — `d < scale·h`

`--positive-scale` tightens the positive threshold. The on-tube seeding radius follows the same
scale, so the class balance stays near 0.24 instead of collapsing.

```bash
python experiments/Build_Task6_New_IcoStar_1_1.py \
    --out $OUT --radii 0.25,0.5,1 --count 3000 \
    --split-file outputs/temporal_split.json --positive-scale 0.5 \
    --threads 8 --shards 8 --shard $I
```

On the temporal split, single 1 h shell, lr 1e-3, 3 seeds:

| positive rule | macro-F1 |
|---|---:|
| `d < 1.0 h` (official) | 0.9395 ± 0.0019 |
| `d < 0.5 h` | 0.9538 ± 0.0028 |
| `d < 0.2 h` | 0.9593 ± 0.0003 |

**Do not read the upward trend as an improvement.** Each scale defines a *different task*, so the
F1 values are not comparable. A tighter threshold gives a purer positive class — a centre within
0.2 h sits essentially on the coreline — rather than a finer discrimination. Use
`--positive-scale 1.0` for any comparison against published numbers or against §8.

### 11.3 Choosing a setup

| goal | split | positive rule |
|---|---|---|
| compare against §8 / earlier results | package default | `1.0` |
| **new work, leak-free evaluation** | **temporal** | **`1.0`** |
| reproduce the official positive set exactly | package default + `--official-centres` | `1.0` |
| study a tighter coreline definition | temporal | `0.5` or `0.2` |

The cache records `positive_scale` and `split_file` in `meta.json`, and each frame file stores
`positive_scale`, so a cache always identifies which variant it is.

## 14. Troubleshooting: my number does not match

The recipe in §4–§6 was re-run from the committed code against a freshly verified cache and
reproduced **0.9538** against the reference 0.9549 ± 0.0018. If you get something materially
different, the cause is almost always *which stars you trained on*, not the training code.
Run `python experiments/Check_Task6_Cache_1_1.py <cache>` first — it detects most of these.

| what you observe | likely cause | fix |
|---|---|---|
| **≈0.95** | correct | — |
| **≈0.92 pooled, but 3 scenes near 0.50** | centres came from the official `IVD > 0.8 × max` rule (`--official-centres`, or training on a shipped `preintegrated/` bundle). `SquareCylinder`, `cylinder3d`, `halfcylinderRe320` contain **no positives at all**, so they sit at the majority-class floor while the pooled score still looks high | rebuild without `--official-centres` (§5) |
| **≈0.62 if you average per scene** | same cause as above, aggregated differently | as above |
| **≈0.94** | temporal split instead of the package split | drop `--split-file`, or compare against §13.1 instead |
| **≈0.95 but not exactly** | different seed, or a dataset revision with re-annotated corelines | check the `dataset.json` sha printed by the reproduction script |
| **≈0.948** | trained on the 0.25 h shell rather than 1 h | pass `--shells 1`, and confirm the cache's radii with the checker |
| **much lower, e.g. ≈0.80** | trained on shipped preintegrated curves (21 lines, 20 face centres at 1 h) instead of building the cache; or far fewer stars per frame; or a mismatched `--shells` / cache pairing | rebuild with §4 and verify with §6 |
| build fails or silently uses the wrong data | `--root` not passed; the builder falls back to a hard-coded path | pass `--root`, or set `TASK6_ROOT` |

### Sanity values to check against

After §4 the cache should report (from `Check_Task6_Cache_1_1.py`):

```
  frames         101   (train 77 / test 24)
  radii          [0.25, 0.5, 1.0]   -> boundary margin 1.0 h
  lines/sample   37   points/line 65
  TOTAL          295,334 stars    70,633 positives   rate 0.2392
```

And a training run with `--shells 1` should report **224,938 train / 70,396 test stars**,
test positive rate **0.2412**, and **587,501** parameters. If those three numbers match and the
F1 still does not, the difference is in the dataset revision, not the pipeline.

## 15. Known caveats

1. **Seeding is stratified, not the official protocol.** The official `IVD > 0.8 × max` centre
   rule gives zero positives in `SquareCylinder`, `cylinder3d` and `halfcylinderRe320`
   (620 000 centres, no positives) and 82.7 % positives in `halfcylinderRe640`, so it cannot
   train a pooled classifier. The dataset README explicitly permits re-seeding while fixing the
   label rule. To reproduce the official positive set exactly, use `--official-centres`.
2. **The label rule is exact; an earlier `h/10` approximation was not.** Caches built before the
   `segment_distance` fix have ≈0.002 % of labels wrong, all at the threshold.
3. **The first 8 h cache is not bit-reproducible** — its per-frame seed came from Python's salted
   `hash()`. The builder now uses `hashlib.sha256`.
4. **Effects below ≈0.005 need more than three seeds.** Five conclusions in the companion
   document reversed on re-measurement at higher seed counts; treat any single-seed ordering as
   provisional.
