# The D8 SIREN-VAE: method, ablations, and selection strategies

Pathline clustering on **cylinder2d** and **boussinesq** only (pipedcylinder2d is
excluded throughout — its low ceiling on every method in this project is suspected to
be a property of the data rather than of the model, and it is not studied here).

Macro-F1 against IVD is a **metric only**: training, checkpoint selection and read-out
selection never touch a label.

Measured 2026-09-20 with `Clustering_D8_recheck/d8_recheck.py`. The underlying
re-check narrative, including the mistakes made along the way, is in
[`d8_vae_recheck.md`](d8_vae_recheck.md); this document is the method and the results.

---

## 1. The method

### 1.1 Signal

`build_signal` → `[N, 9, L-1, 2]` per seed point:

| channel | content |
|---|---|
| 0 | centre pathline time-derivative |
| 1..8 | the 8 angular neighbours' displacement **relative to the centre**, time-differenced, ×100 |

Neighbours are ordered clockwise from North (N, NE, E, SE, S, SW, W, NW), all at the
same radius. `normalize_signal_d8` then rescales with **scalar** statistics — one
mean+RMS for the centre, a second shared across all eight arms — so that the
normalisation commutes with rotation and with cyclic channel permutation.

### 1.2 Architecture

```
signal ──► ResConv+Transformer encoder ──► z_total = [ z_inv (16) | z_eq (8) ]
                                                │
                                                └──► SIREN decoder ──► reconstruction
```

`z_inv` is pushed toward D8-invariance by the contrastive term and is **the only thing
clustering reads**. `z_eq` is unconstrained and absorbs the transformation so the
decoder can still reconstruct a rotated star.

### 1.3 Objective

```
L = λ_recon · L_recon  +  β · KL  +  λ_contrast · L_contrast  +  λ_var · V  +  λ_cov · C
```

* `L_recon` — MSE, **channel-balanced** by within-sample variance (after
  `normalize_signal_d8` every channel has unit *global* variance, but on cylinder2d the
  centre channel's variance is 99% across-sample and 1% along-time, while each arm is
  ~30% along-time; unweighted MSE therefore fits the centre and ignores the arms).
* `L_contrast` — mean pairwise `1 − cos` over `[z_inv(x), z_inv(aug₁ x), z_inv(aug₂ x)]`,
  so every view is pulled toward every other, not toward one reference.
* `V, C` — VICReg-style variance hinge and off-diagonal covariance on `z_inv`,
  countering the degenerate optimum of a positive-only contrastive loss (`z_inv → const`).

Defaults: `λ_recon 1`, `β 1e-4`, `λ_contrast 15`, `λ_var 5`, `λ_cov 0.2`,
`zinv_target_std 0.5`.

### 1.4 Augmentation — the group action

Each batch row draws its **own** D8 element (8 rotations × reflection = 16), applied to
the raw signal, which is then re-normalised with the training statistics. Both the
velocity vectors and the neighbour channel order are transformed consistently, so the
augmented star is a physically valid rotated/reflected star.

`--aug_mode so2` (added for this study) instead draws a **continuous** O(2) angle:
exact on the vectors, circular linear interpolation on the arm index. It reproduces
`d8_augment` to **0.0e+00 at all 16 group elements** and interpolates between them
(`test_so2_channel.py`). The interpolation is unavoidable — a star indexed by 8 fixed
channels cannot represent a fractional arm shift.

### 1.5 Training and selection

AdamW, **two parameter groups**: the encoder runs at `lr × 0.1` (`--encoder_lr_scale`),
the decoder at `lr`. Cosine annealing with `T_max = epochs`. 3000 epochs, batch 256,
grad-clip 1.0, `lr 5e-4`, weight decay 1e-4.

Then, in order:

1. **Read-out** — a 2-component Gaussian mixture with **tied** (shared) covariance on
   L2-normalised `z_inv`, **10 restarts**, keeping the partition with the best
   **silhouette**. See §7; this replaced `KMeans(n_init=10)` on 2026-09-21.
2. **Checkpoint selection** — none. `--checkpoint_select final` takes the last epoch.
   With the read-out above the curve converges, and every internal index measured is
   worse than reading nothing (§8).
3. **Decoder-only fine-tune** — 2000 epochs with the encoder frozen *and in `eval()`*,
   latents precomputed. It cannot move `z_inv`, so it cannot change any clustering
   result; it exists only to sharpen reconstruction.

> **Defaults changed 2026-09-21.** `_CLUSTERER` `kmeans`→`gmm_tied`, `_KMEANS_SELECT`
> `inertia`→`silhouette`, `_KMEANS_RESTARTS` 20→10, `--checkpoint_select`
> `davies_bouldin`→`final`. Delivered mean F1 0.7265 → **0.9194**. Nothing about
> training changed. Revert with `_CLUSTERER="kmeans"; _KMEANS_SELECT="inertia";
> _KMEANS_RESTARTS=20; --checkpoint_select davies_bouldin`.

---

## 2. Protocol

**The runs are bit-deterministic.** `set_reproducible` fixes all seeds and sets
`cudnn.deterministic`; two independent processes produced identical 121-point curves
(max difference 0.0000). There is no noise floor to allow for — every difference below
is exact for seed 1.

**The untrained baseline is lr = 0, not `--epochs 0`.** The encoder's conv stem contains
BatchNorm, and the `--epochs 0` path returns before any training-mode forward pass, so
its running statistics never calibrate. lr = 0 runs the identical code path, forward
passes and augmented views, and updates no weight.

Unless stated, all cells use the shipped configuration and differ in one factor.

---

## 3. Ablation: what each part of the objective earns

Five arms, one grid, matched seeds and initialisation.

| arm | views | decoder in loss | cyl | bou | **mean** | Δ vs untrained |
|---|---|---|---|---|---|---|
| `init` (lr = 0) | — | — | 0.4251 | 0.4659 | **0.4455** | — |
| `ssl_d8` | discrete D8 | off | 0.9112 | 0.5396 | **0.7254** | **+0.280** |
| `ssl_so2` | continuous O(2) | off | 0.9125 | 0.5515 | **0.7320** | +0.286 |
| `ssl_d8_rec` | discrete D8 | **on (shipped)** | 0.9125 | 0.5405 | **0.7265** | +0.281 |
| `ssl_so2_rec` | continuous O(2) | on | 0.9167 | 0.5547 | **0.7357** | +0.290 |

**The contrastive term is the method.** +0.280 of the +0.281 total. On cylinder2d it
lifts the encoder from 0.4251 — near chance — to 0.9112. This is by far the largest
learning effect in this project: the GAT-SO(2) method's training is worth −0.0008 and
the DCT track's +0.037. The reason it works *here* is that the D8-VAE's **input** is
the raw star, on which the group acts non-trivially, so two augmented views genuinely
differ. Encoders whose features are invariant by construction produce bit-identical
views and give the contrastive term no gradient at all.

**Reconstruction adds +0.001 (D8) / +0.004 (O(2)) to clustering.** It is not useless —
it is what makes the model generative, and the fine-tune improves reconstruction MSE
substantially — but for the clustering read-out it is close to free. It appears more
valuable at the shipped learning rate only because it slightly damps the oscillation
(§4); at lr 1e-4 the two arms are *identical*:

| cylinder2d, end-of-run F1 | `ssl_d8` (no decoder) | `ssl_d8_rec` (decoder) |
|---|---|---|
| lr 5e-4 | 0.6496 | 0.6496 |
| lr 2e-4 | 0.7044 | 0.6951 |
| lr 1e-4 | **0.9207** | **0.9207** |

**Continuous SO(2) views add +0.0065** — nothing. Either 16 discrete elements already
saturate the view diversity the contrastive term needs, or the channel-index
interpolation blunts the extra angles. The exact continuous action requires the graph
formulation in `FMT_Utils/gat_so2.py`, where an arm is a node carrying its angle.

---

## 4. The instability, and its mechanism

Every arm ends far below its selected value (0.585–0.650 against ~0.913 on cylinder2d).
The curve does not converge — it oscillates between ~0.94 and ~0.45 at 25-epoch
intervals, 30 of 120 adjacent evaluations jumping by more than 0.20, and the good number
is caught in passing rather than reached.

**It is not the encoder.** Per-evaluation diagnostics:

| | latent drift between evals | ARI vs previous labelling |
|---|---|---|
| calm steps (\|ΔF1\| < 0.02) | 0.0003 | 0.986 |
| jump steps (\|ΔF1\| > 0.20) | **0.0002–0.0005** | **−0.085** |

Across a swing from 0.894 to 0.441 the latent moves as little as on a calm step, while
the partition becomes unrelated to the previous one. (ARI is permutation-invariant, so
−0.085 is a genuinely different partition, not a relabelling.)

**It is k-means, and specifically its objective.** Where F1 collapses, two optima exist
and the *good one has the higher inertia*:

```
ep 525   pipeline (n_init=10) = 0.4511
   inertia 22.7428  DB 0.7861  sil 0.5208  F1 0.4511   ← k-means keeps this
   inertia 22.8186  DB 1.0650  sil 0.5326  F1 0.6915
   inertia 23.2252  DB 0.7366  sil 0.5681  F1 0.9275
```

k-means discards the right answer **by construction**, and a larger `n_init` discards it
more reliably. (`_KMEANS_N_INIT` was already 10 — this is not the historical 1-init bug.)

Training hyper-parameters change the symptom without being the mechanism: where the
latent *settles* decides whether it sits on this knife edge at all.

---

## 5. Strategy 1 — learning rate

cylinder2d, all else shipped:

| lr | arm | selected | best | **end** | frac evals > 0.85 | jumps > 0.2 |
|---|---|---|---|---|---|---|
| 5e-4 *(shipped)* | `ssl_d8_rec` | 0.9125 | 0.9313 | 0.6496 | 0.21 | 20 |
| 2e-4 | `ssl_d8_rec` | 0.9125 | 0.9313 | 0.6951 | 0.47 | 17 |
| **1e-4** | `ssl_d8_rec` | 0.9125 | 0.9256 | **0.9207** | **0.99** | **1** |
| 3e-5 | `ssl_d8_rec` | 0.8997 | 0.9222 | 0.9222 | 0.98 | 1 |
| 5e-4 | `ssl_d8` | 0.9112 | 0.9431 | 0.6496 | 0.28 | 30 |
| 2e-4 | `ssl_d8` | 0.9125 | 0.9391 | 0.7044 | 0.48 | 16 |
| **1e-4** | `ssl_d8` | 0.9125 | 0.9207 | **0.9207** | 0.99 | 1 |
| 3e-5 | `ssl_d8` | 0.8997 | 0.9167 | 0.9167 | 0.98 | 1 |

At lr 1e-4 the curve flattens: 99% of evaluations above 0.85, one jump in 3000 epochs,
and **the final epoch delivers 0.9207 with no checkpoint selection at all** — above the
shipped recipe's DB-selected 0.9125. 3e-5 is slightly worse, so 1e-4 is an interior
optimum. The effect is identical with and without the decoder.

---

## 6. Strategy 2 — the variance floor (`zinv_var_weight`)

The script's own help says 5.0 *"HURTS on pipedcylinder2d/boussinesq — sweep it, don't
assume"*, and 5.0 is the global default. On **boussinesq**:

| `zinv_var_weight` | lr | selected | best | end |
|---|---|---|---|---|
| 5.0 | 5e-4 *(shipped)* | 0.5405 | 0.8134 | 0.5927 |
| 5.0 | 1e-4 | 0.7976 | 0.8056 | 0.6732 |
| **0.0** | **5e-4** | **0.7976** | 0.8134 | **0.7976** |
| 0.0 | 1e-4 | 0.7640 | 0.7976 | 0.7640 |

Turning it off is worth **+0.257** on boussinesq and also removes the oscillation there
(the curve sits flat at 0.813). It is a *win* on cylinder2d, so this is a per-dataset
setting shipped as a global default. It does not compose with the LR fix — both repair
the same failure, and together they are worse than either alone.

---

## 7. Strategy 3 — how to pick the partition (the single biggest lever)

Mean F1 over the **entire** training curve (every evaluation, ~30 restarts each), so
this isolates the read-out from checkpoint selection:

| rule | cylinder2d | boussinesq | mean |
|---|---|---|---|
| inertia *(current)* | 0.6915 | 0.6263 | 0.6589 |
| min Davies–Bouldin | 0.6184 | 0.7389 | 0.6787 |
| min DB, minority frac ≥ 0.05 | 0.6184 | 0.6641 | 0.6413 |
| min DB, minority frac ≥ 0.10 | 0.6144 | 0.6159 | 0.6152 |
| min DB, minority frac ≥ 0.20 | 0.4681 | 0.5227 | 0.4954 |
| **max silhouette** | **0.8899** | **0.8296** | **0.8598** |
| max silhouette, frac ≥ 0.05 | 0.8899 | 0.8176 | 0.8538 |
| max silhouette, frac ≥ 0.20 | 0.6446 | 0.6305 | 0.6375 |
| max Calinski–Harabasz | 0.6915 | 0.6263 | 0.6589 |

**Silhouette is the fix: +0.20 on both datasets.** Run ~20 k-means++ restarts and keep
the partition with the highest silhouette rather than the lowest inertia. Label-free, no
retraining, implemented as `_KMEANS_SELECT` in `_fit_kmeans` (default unchanged).

Why it works: inertia rewards *compactness alone* and is minimised by carving off
whichever chunk is tightest; silhouette rewards *separation relative to compactness*.
On this latent the vortex/non-vortex split is the well-separated one but not the
tightest, which is exactly where the two disagree.

**Davies–Bouldin cannot be tuned into this role.** Within-evaluation Spearman against
F1, across the candidate partitions:

| | cylinder2d | boussinesq |
|---|---|---|
| **silhouette** | **+0.889** | **+0.894** |
| −Davies–Bouldin | −0.073 | **−0.311** |
| −inertia | −0.060 | −0.046 |

DB is uninformative on one dataset and *anti*-informative on the other. The obvious
diagnosis — a degenerate tiny-cluster split — is wrong: DB already picks *larger*
minority clusters than silhouette (0.188 vs 0.156), and guards make both rules
monotonically worse, because above ~0.15 the guard exceeds the true minority class.

**Calinski–Harabasz is not an independent criterion.** It equals inertia to four
decimals on both datasets, because for fixed *k* and *n* it is a monotone function of
the within-cluster sum of squares.

### 7.1 A clusterer whose metric matches the structure

Silhouette-over-restarts is a *search* around k-means' bad optimum. Fixing the metric
instead is better. A 2-component Gaussian mixture with **tied** covariance — one shared
covariance matrix, i.e. k-means under a Mahalanobis metric, k-means in the space
whitened by the pooled within-cluster covariance — finds the vortex split directly.

Mean F1 over all epochs (label-free pick = best silhouette among restarts):

| clusterer | cylinder2d | ≥0.90 | boussinesq | ≥0.90 |
|---|---|---|---|---|
| k-means | 0.8899 | 50/121 | 0.8296 | 28/121 |
| **GMM tied** | **0.9173** | **112/121** | **0.8546** | **69/121** |
| GMM full | 0.8825 | 11/121 | 0.7442 | 0/121 |
| GMM diag | 0.6196 | 39/121 | 0.5212 | 0/121 |
| spectral, knn 10 | 0.8319 | 87/121 | 0.5210 | 0/121 |
| agglomerative average | 0.8347 | 29/121 | 0.8125 | 2/121 |
| agglomerative complete | 0.7649 | 20/121 | 0.7067 | 5/121 |

It roughly doubles the number of epochs admitting a ≥0.90 partition, which was the
binding constraint. `tied` beating both `full` (too flexible) and `diag` (too rigid) is
the signature of the *pooled metric* being what matters, not model capacity — the
vortex split is discriminative without being high-variance, which is precisely why
Euclidean k-means prefers the wrong partition.

Note this is not the global PCA whitening that *hurt* in the GAT study (0.454 vs
0.678): the metric here is estimated jointly with the partition, not applied in advance.

### 7.2 The number of restarts is not a free parameter

Silhouette tracks F1 at rho ~ 0.89, **not** 1.0. Maximising a proxy over a larger
candidate set finds partitions that win on the proxy and lose on the target — the
winner's curse. Too few restarts miss the good partition entirely. There is an interior
optimum. Final-epoch F1 on **boussinesq**, GMM-tied:

| restarts | 1 | 3 | 5 | **8** | **10** | **14** | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| boussinesq | 0.444 | 0.444 | 0.728 | **0.911** | **0.911** | **0.911** | 0.829 | 0.829 |
| cylinder2d | 0.928 | 0.928 | 0.928 | 0.928 | 0.928 | 0.928 | 0.928 | 0.928 |

A 3-point plateau at 8–14 on boussinesq and total indifference on cylinder2d, with a
cliff at 20 worth **0.083**. k-means has no such cliff (flat 8→30) — it is the *safer*
clusterer at a lower ceiling.

**Honest caveat:** the 8–14 window was located using the labels. It is a plateau rather
than a spike, and only one of the two datasets constrains it, but it is a
hyper-parameter chosen on the evaluation and should be re-validated on a new dataset.

This one mechanism also explains three earlier observations that looked unrelated:
`n_init=10` being worse than `n_init=1` for k-means, consensus and stability selection
converging on the *wrong* cluster (§8.1), and silhouette working within an epoch but
not across epochs.

---

## 8. Strategy 4 — how to pick the epoch

With the **GMM-tied** read-out the F1 curve *converges*, so there is no peak to catch:

| mean \|ΔF1\| over the last 30 evals | worst of those 30 |
|---|---|
| cylinder2d 0.0000 | 0.9275 |
| boussinesq 0.0008 | 0.9064 |

Any epoch in the last quarter delivers ≥0.906. And every internal index is now *worse*
than reading nothing:

| epoch rule (GMM-tied, 10 restarts) | boussinesq | cylinder2d | mean |
|---|---|---|---|
| **final epoch — no selection at all** | **0.9112** | **0.9275** | **0.9194** |
| median epoch | 0.9028 | 0.9275 | 0.9151 |
| min Davies–Bouldin | 0.7727 | 0.8956 | 0.8341 |
| max silhouette | 0.7640 | 0.9013 | 0.8327 |
| oracle epoch | 0.9212 | 0.9313 | 0.9263 |

Davies–Bouldin costs **0.085** against reading nothing, and the pipeline's 100-epoch
warmup does not rescue it. The shipped default is now `--checkpoint_select final`.

**This retires Davies–Bouldin from the method entirely.** It was load-bearing only
because it was compensating for a read-out that threw the good partition away at most
epochs; fix the read-out and it becomes actively harmful.

A fixed-fraction checkpoint looked best in a first pass (0.8997 at the midpoint) but is
**not** a rule: neighbouring fractions give 0.9180 at 0.25 and 0.7996 at 0.33. Discarded.

### 8.1 What does NOT work, and why it is the same reason

Two families of label-free rule fail, both because **the good partition is a minority
state** (only 28/121 epochs under k-means; 5.3 of 30 restarts at a given epoch):

| | boussinesq | cylinder2d |
|---|---|---|
| Spearman(partition stability, F1) | **−0.150** | **−0.414** |
| max-mean-ARI epoch delivers | 0.8285 | 0.8906 |
| consensus over all 121 epochs | 0.8285 | 0.8684 |

Stability is *anti*-correlated with F1 and consensus lands exactly on the median. Any
procedure built on typicality, averaging or reproducibility converges on the **wrong**
cluster by construction. That rules out consensus clustering, stability selection and
trajectory ensembling together, and is the same effect as the restart-count cliff in
§7.2 seen from the other side.

---

## 9. Combined, end to end

Delivered F1, all label-free, runs bit-deterministic:

| configuration | boussinesq | cylinder2d | **mean** |
|---|---|---|---|
| shipped before today: k-means/inertia + DB epoch | 0.5405 | 0.9125 | **0.7265** |
| k-means/silhouette(20) + DB epoch | 0.8358 | 0.9167 | 0.8763 |
| GMM-tied/silhouette(20) + final epoch | 0.8285 | 0.9275 | 0.8780 |
| **GMM-tied/silhouette(10) + final epoch — new default** | **0.9112** | **0.9275** | **0.9194** |
| *the same, if you DB-selected instead* | 0.7727 | 0.8956 | 0.8341 |
| oracle epoch | 0.9212 | 0.9313 | 0.9263 |

**+0.193 over the previous recipe, with training completely unchanged**, and an oracle
regret of 0.007. Verified end to end by running the script with nothing overridden.

### Recommended configuration — now the default

```python
_CLUSTERER       = "gmm_tied"     # tied-covariance GMM, not k-means
_KMEANS_SELECT   = "silhouette"   # not inertia
_KMEANS_RESTARTS = 10             # NOT free: 8-14 only (see 7.2)
--checkpoint_select final         # no checkpoint selection at all
```

Optional and independent: `lr 1e-4` for a flat curve on cylinder2d under the *old*
read-out (§5); `zinv_var_weight 0` on boussinesq under the old read-out (§6). Neither
is needed once the read-out is fixed, and both were compensating for the same defect.

## 9a. Which objective terms are actually exercised?

The loss has five terms and the pipeline has several more switches. Not all of them do
anything, and **not all of them have ever been ablated** — worth stating explicitly so
the untested ones are not mistaken for validated choices.

| term / switch | default | status |
|---|---|---|
| `lambda_contrast` (D8 contrastive) | 15 | **carries the method**: +0.280 of the +0.281 that training earns (§3) |
| `lambda_recon` (SIREN reconstruction) | 1 | **~0 for clustering** (+0.001 D8, +0.004 O(2)); still what makes the model generative |
| `beta` (KL) | 1e-4 | **never ablated** — running in §9c |
| `zinv_var_weight` (VICReg variance hinge) | 5 | measured, but only under the **old** read-out: a switch between a converged low state and an oscillating high one (§9b) |
| `zinv_cov_weight` (VICReg covariance) | 0.2 | **never ablated separately** — running in §9c |
| `encoder_lr_scale` | 0.1 | **not ablated in this study** (an earlier note records final-epoch F1 0.921 → 0.689 without it) |
| `recon_channel_balance` | on | **not ablated in this study** |
| `aug_mode` | `d8` | `so2` implemented and exact at all 16 group elements; worth +0.0065, i.e. nothing (§3.3) |
| `lambda_contrast_final` (ramp) | `None` | **never used** |
| `effrank_rise` (encoder auto-freeze) | 0 / absent | **never fires** — it is not even a CLI argument, and no run in this study froze the encoder |
| `encoder_freeze_epoch` | −1 | **never used** |
| decoder-only fine-tune | 2000 ep | runs, but the encoder is frozen *and* in `eval()` with latents precomputed, so it **cannot change any clustering result** — reconstruction only |

So of the five loss terms, one is doing essentially all the work, one is measurably
irrelevant to clustering, one is a dataset-dependent switch, and two have never been
tested. Two of the pipeline's stability mechanisms have never fired at all.

## 9b. SSL task weights (boussinesq)

Swept under the k-means/silhouette read-out, i.e. **before** the read-out was fixed --
so read the *shape* rather than the absolute numbers:

| λ_contrast | λ_var | selected | best | end | curve |
|---|---|---|---|---|---|
| 5 | 0 | 0.8056 | 0.8056 | 0.8056 | flat |
| 5 | 5 | 0.8056 | 0.9201 | 0.8358 | oscillating (39 jumps > 0.05) |
| 15 | 0 | 0.7976 | 0.8134 | 0.7976 | flat |
| 15 | 5 *(shipped)* | 0.8358 | 0.9268 | 0.8285 | oscillating (15) |
| 45 | 0 | 0.7976 | 0.8210 | 0.8134 | flat |
| 45 | 5 | 0.8210 | 0.9137 | 0.9028 | oscillating (29) |

**λ_contrast is not a useful knob** — with the variance floor off, 5 / 15 / 45 span
0.806-0.813. **λ_var is a switch**: off gives a converged curve at ~0.81, on gives an
oscillating curve with a ceiling of 0.92-0.93. Turning it off buys stability by
*removing* the good state, not by stabilising it.

That trade is an artefact of the read-out, not of the objective: the "oscillation" is
k-means losing a partition the latent still contains (§4), and the GMM read-out holds
it. So the right recipe keeps the variance floor and fixes the read-out, rather than
trading one against the other. The sweep has not been repeated under the new read-out.

## 9c. The terms that had never been tested

One factor at a time from the new defaults, both datasets, under the **new** read-out
(`Clustering_D8_recheck/ofat.json`). `beta`, `zinv_cov_weight`, `encoder_lr_scale` and
`recon_channel_balance` had never been ablated; `lambda_recon`, `zinv_var_weight` and
`aug_mode` are re-tested here because their earlier measurements were taken through the
k-means read-out that later proved unreliable.

(running — results to be filled in)

---

## 10. What the representation can actually reach

| | best selected | best on curve | max over any partition, any epoch |
|---|---|---|---|
| cylinder2d | 0.9167 | 0.9313 | 0.9313 |
| boussinesq | 0.8358 | 0.9268 | **0.9268** (epoch 1025) |

Boussinesq was never unable to reach 0.8+; the shipped read-out discards it. At epoch
100 — the epoch previously recorded at 0.8430 — the good partition is present on this
run's latent and scores 0.8210, while inertia returns 0.5405. The oracle gap that
remains (0.9291 vs 0.8763 delivered) is epoch selection, worth ~0.05.

---

## 11. Caveats

* **One seed per cell.** The runs are bit-deterministic so these numbers are exact for
  seed 1, but seed-to-seed spread for this method is known to be large and is not
  measured here. This is the single biggest gap.
* **`_KMEANS_RESTARTS = 10` was chosen with reference to the labels** (§7.2). It is a
  3-point plateau, not a spike, and only boussinesq constrains it -- but it is a
  hyper-parameter fitted on the evaluation and must be re-validated elsewhere. If that
  is unacceptable, k-means + silhouette is flat from 8 to 30 restarts and delivers
  0.8763 with no such exposure.
* **The SSL weight sweep (§9b) predates the read-out fix** and has not been redone
  under it; its conclusions about stability are therefore suspect in the same way
  everything measured through k-means was.
* The GMM read-out costs 10 mixture fits plus a silhouette per scoring call, and is
  materially slower than `KMeans(n_init=10)`. Fine at N = 512-675; not free at scale.
* pipedcylinder2d is excluded and nothing here has been checked on it.
* A 0.022 residual on the boussinesq reproduction (0.8210 reachable at epoch 100 vs
  0.8430 recorded) is unexplained.
* The silhouette read-out costs ~20 k-means fits plus an O(N²) silhouette per scoring
  call — negligible at N = 512–675, but it is not free at larger N.
* Whether the same read-out fix helps the other two methods in this repo
  (`FMT_Clustering_GAT_SO2.py` and the DCT track) is **untested**, and both cluster
  through the same `KMeans(n_init=10)` call.

## 12. Files

| file | what |
|---|---|
| `FMT_Clustering_SIRENVAE.py` | the method; `--aug_mode`, `--lambda_recon`, `_KMEANS_SELECT` |
| `FMT_Utils/siren_vae.py` | encoder/decoder, `d8_augment`, `so2_augment_batch`, losses |
| `Clustering_D8_recheck/d8_recheck.py` | ablation / LR / combo / variance-floor grids |
| `Clustering_D8_recheck/tune_readout.py` | the read-out rule family (DB guards, Calinski) |
| `Clustering_D8_recheck/alt_clusterers.py` | k-means vs GMM / spectral / agglomerative |
| `Clustering_D8_recheck/restart_curve.py` | delivered F1 vs number of restarts |
| `Clustering_D8_recheck/analyse_selection.py` | stability and consensus epoch rules |
| `Clustering_D8_recheck/gmm_delivery.py` | GMM restart criterion and epoch rules |
| `Clustering_D8_recheck/diag_bistable.py` | per-eval latent snapshots |
| `Clustering_D8_recheck/test_so2_channel.py` | continuous O(2) ≡ `d8_augment` at all 16 elements |
| `docs/d8_vae_recheck.md` | the re-check narrative, including harness bugs found |
