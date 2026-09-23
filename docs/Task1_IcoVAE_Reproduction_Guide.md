# Task 1 — reproduction guide (IcoVAE pathline clustering, best config)

How to run the best configuration from [`3d_kmeans_ico.md`](3d_kmeans_ico.md) end to end:
**all three SSL tasks (`--aug both` + reconstruction), variance floor off, read out with a
tied-covariance GMM**. Background, ablations and caveats are in that document.

Unsupervised throughout: no split, every timeslice is trained on and scored. The IVD
reference is written into the cache as a metric only.

## Reference result

Macro-F1, GMM-tied, 10 restarts, partition and epoch both picked by max silhouette, seed 7068:

| Re160 | Re640 | Re6400 | tangaroa | deltaWing | **mean** |
|---|---|---|---|---|---|
| 0.8638 | 0.8381 | 0.8326 | 0.8811 | 0.8620 | **0.8555** |

One seed per flow, so expect some seed-to-seed spread.

## 0. Data

Put the five NetCDF files in one directory, `$DATA`:

| dataset id | file |
|---|---|
| `halfcylinderRe160_eth` | `halfcylinderRe160.nc` |
| `halfcylinderRe640_eth` | `halfcylinderRe640.nc` |
| `halfcylinderRe6400_eth` | `halfcylinderRe6400.nc` |
| `tangaroa_eth` | `tangaroa.nc` |
| `deltawing_eth` | `deltaWing_mag0_3reesampled.nc` |

The four half-cylinder / tangaroa files are the public ETH CGL downloads. deltaWing is not
public. Without it, drop `deltawing_eth` from the lists below.

```bash
cd /path/to/FMTCode
DATA=/path/to/flowData3D
FLOWS=halfcylinderRe160_eth,halfcylinderRe640_eth,halfcylinderRe6400_eth,tangaroa_eth,deltawing_eth
```

Use a Python environment with `torch` (CUDA), `numpy`, `scipy`, `scikit-learn` and `netCDF4`.

## 1. Build the 13-line icosahedral cache

```bash
python experiments/Build_Task1_Ico_Cache_1_1.py \
    --data-root "$DATA" --datasets "$FLOWS" \
    --output outputs/exp_Task1_IcoVAE_cache
```

This writes 14 timeslices per flow to `outputs/exp_Task1_IcoVAE_cache/<dataset>/slice_*.npz`.
Slices that already exist are skipped. Add `--limit 1` for a quick smoke test.

## 2. Train the best config, saving latents

One run per flow, about 9 min each on an A100:

```bash
for D in ${FLOWS//,/ }; do
  python experiments/Run_Task1_IcoVAE_1_1.py \
      --cache outputs/exp_Task1_IcoVAE_cache --dataset "$D" \
      --output outputs/exp_Task1_IcoVAE_best --label all3_var0 \
      --seed 7068 --steps 6000 --batch 512 --eval-every 100 \
      --lr 5e-4 --aug both --lambda-recon 1.0 --lambda-contrast 15 \
      --zinv-var-weight 0 \
      --clusterer kmeans \
      --save-latents outputs/exp_Task1_IcoVAE_best_latents
done
```

The flags that define the config:

* `--aug both` — the 120 discrete `I_h` views alternate with continuous SO(3) views.
* `--lambda-recon 1.0` — keeps the SIREN reconstruction term.
* `--zinv-var-weight 0` — turns the VICReg variance floor off.

Everything else is at its default, spelled out above so the command stays self-contained.

`--clusterer kmeans` only controls the F1 curve printed during training, which is cheap to
compute. The reported number comes from the offline GMM-tied read-out in step 3, applied to
the saved `z_inv` latents. You can pass `--clusterer gmm_tied` here to monitor with the real
read-out, but each evaluation will be about 10× slower.

Outputs:

* `outputs/exp_Task1_IcoVAE_best/summary_all3_var0_<dataset>.json` — the online curve summary;
* `outputs/exp_Task1_IcoVAE_best_latents/<dataset>/all3_var0_step*.npy` and `reference.npy`.

## 3. GMM-tied read-out (the reported number)

```bash
python experiments/Analyze_Task1_IcoVAE_Clusterer_1_1.py \
    --latents outputs/exp_Task1_IcoVAE_best_latents --label all3_var0 \
    --datasets "$FLOWS" --clusterers gmm_tied --restarts 10 \
    --output outputs/exp_Task1_IcoVAE_best_gmm.json
```

For each evaluation it keeps (by default) every 5th one, i.e. 21 of the 61 saved steps. It
fits 10 GMM-tied restarts on the L2-normalised `z_inv` and keeps the restart with the highest
silhouette. Then it reports one number per epoch rule:

| column | epoch picked by | label-free? |
|---|---|---|
| `final` | the last evaluation | yes |
| `min_db` | min Davies–Bouldin | yes |
| **`max_silhouette`** | **max silhouette — the headline rule for this config** | **yes** |
| `oracle` | max F1 | no — a ceiling, not a result |

The per-flow `max_silhouette` values should land near the reference table. For this config
all label-free columns agree within about 0.005, because its curve converges.

## Optional: FMT baseline on the same data

```bash
python experiments/Run_Task1_Baselines_Full_1_1.py --datasets "$FLOWS"
```

Compare against the `fmt13` arm (13-line star, same cache). The reference mean is 0.8007.
The script reads the cache from the default path `outputs/exp_Task1_IcoVAE_cache`. The
`fmt7` arm also needs the older `outputs/exp_Task1_ReSweep_0.1` cache and can be ignored here.
