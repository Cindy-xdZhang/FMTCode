"""Screen four changes to the octahedral SIREN-VAE on one dataset and two seeds.

1. batch 1024 instead of 256, with the step count cut 4x so the total number of
   primitive presentations is unchanged.  The workload is launch-bound (49.5 ms
   at batch 256 vs 52.2 at 1024), so this is nearly free wall-clock.
2. the decoder reconstructs the augmented views as well as the original, so
   `z_eq` must carry the orientation of the whole orbit.
3. the VAE trains and selects its checkpoint on development ordinals 0-7 rather
   than 0-5 / 6-7, widening its temporal coverage from frames 31-82 to 31-113 --
   a fully protocol-legal attack on the latent drift found in §10.3.
4. reconstruction is validated on held-out ordinals 8-9, before and after the
   decoder fine-tune, so reconstruction is a measurement rather than a
   training-set fit.

Clustering follows the 1.3 NEW rule (fit k-means on ordinals 8-9, the
development slices closest in time to confirmation); the OLD rule is scored
alongside for reference.  There is no DataLoader in this pipeline -- primitives
live on the GPU as one tensor and are indexed directly, so `num_workers` does not
apply; the thread count is set explicitly instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.OctahedralGroup_3D import (  # noqa: E402
    apply_norm_stats_oh, build_signal_from_raw, group_tensors,
)
from FMT_Utils.SirenVAE_3D import (  # noqa: E402
    READOUTS, encode_readouts, fit_octvae, l2_normalize,
)
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference  # noqa: E402
from FMT_Utils.Task12Evaluation_3D import (  # noqa: E402
    binary_cluster_metrics, calibrate_vortex_cluster,
)

ROOT = Path(__file__).resolve().parents[1]
# Arm references.  FMT/Raw are this experiment's own re-runs on these caches
# (outputs/exp_Task1_OctVAE_3D_1.2/runs.csv); the v1.3 column is read from the
# boundary-transfer run so it never drifts out of sync with a hand-typed copy.
FMT_RAW = {
    "halfcylinderRe160_eth": (0.5225, 0.2188),
    "halfcylinderRe640_eth": (0.5177, 0.3394),
    "halfcylinderRe6400_eth": (0.5340, 0.3597),
    "tangaroa_eth": (0.7261, 0.5752),
}


def _reference(dataset_id):
    fmt, raw = FMT_RAW.get(dataset_id, (None, None))
    entry = {"fmt": fmt, "raw": raw, "v13_gmean": {}}
    path = ROOT / "outputs/exp_Task1_OctVAE_3D_1.3_boundary" / f"{dataset_id}.csv"
    if path.exists():
        for row in csv.DictReader(path.open(encoding="utf-8")):
            if row["rule"] == "NEW_fit8-9" and row["readout"] == "gmean":
                entry["v13_gmean"][int(row["train_seed"])] = round(float(row["f1"]), 4)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threads", type=int, default=8,
                        help="torch/OMP thread count (the num_workers equivalent here)")
    parser.add_argument("--lr", type=float, default=None, help="override training.lr")
    parser.add_argument("--steps", type=int, default=None,
                        help="override training.steps (also rescales the cosine schedule)")
    parser.add_argument("--seeds", default=None, help="comma-separated training seeds")
    parser.add_argument("--dataset-filter", default=None,
                        help="restrict to one dataset id, so flows can run concurrently")
    parser.add_argument("--tag", default=None, help="suffix for the output directory")
    arguments = parser.parse_args()

    torch.set_num_threads(int(arguments.threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(arguments.threads))

    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    settings = dict(config["training"]); settings.update(config["model"])
    if arguments.lr is not None:
        settings["lr"] = float(arguments.lr)
    if arguments.steps is not None:
        settings["steps"] = int(arguments.steps)
    if arguments.seeds:
        config["train_seeds"] = [int(x) for x in arguments.seeds.split(",")]
    if arguments.dataset_filter:
        config["datasets"] = [d for d in config["datasets"]
                              if d["id"] == arguments.dataset_filter]
        if not config["datasets"]:
            raise SystemExit(f"no dataset matches {arguments.dataset_filter}")
    splits, sampled = config["splits"], int(config["sampled_steps"])
    n_init, readouts = int(config["kmeans_n_init"]), tuple(config.get("readouts", READOUTS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrices, permutation = group_tensors(config["group"], device=device)

    rows, histories = [], {}
    for dataset in config["datasets"]:
        development = load_cache_records(ROOT / dataset["development_cache"])
        confirmation = load_cache_records(ROOT / dataset["confirmation_cache"])

        def signal(ordinals):
            return np.concatenate([build_signal_from_raw(development[i]["raw"], sampled)
                                   for i in ordinals])

        confirmation_signal = np.concatenate(
            [build_signal_from_raw(r["raw"], sampled) for r in confirmation])
        confirmation_reference = stack_reference(confirmation)
        reference = {i: np.asarray(development[i]["reference"], dtype=bool)
                     for i in range(len(development))}

        print(f"[{dataset['id']}] batch={settings['batch_size']} steps={settings['steps']} "
              f"recon_augmented={settings.get('recon_augmented')} "
              f"vae_train={splits['vae_train']} lr={settings['lr']:g} steps={settings['steps']} "
              f"lr_warmup={settings.get('lr_warmup_steps')} threads={arguments.threads}", flush=True)

        for seed in config["train_seeds"]:
            started = time.time()
            model, stats, history = fit_octvae(
                signal(splits["vae_train"]), signal(splits["vae_validation"]),
                settings, seed, device, group=config["group"],
                signal_recon_validation=signal(splits["recon_validation"]),
                signal_recon_train_probe=signal(splits["recon_train_probe"]))
            history["seconds"] = time.time() - started
            history["parameters"] = model.parameter_counts()
            histories[f"{dataset['id']}_seed{seed}"] = history

            def encode(values):
                tensor = apply_norm_stats_oh(
                    torch.as_tensor(values, dtype=torch.float32, device=device),
                    stats, settings["clip_sigma"])
                return encode_readouts(model, tensor, matrices, permutation,
                                       batch_size=config["encode_batch_size"],
                                       readouts=readouts)

            per_ordinal = {i: encode(signal([i])) for i in range(len(development))}
            confirmation_latent = encode(confirmation_signal)

            for readout in readouts:
                block = lambda o: l2_normalize(
                    np.concatenate([per_ordinal[i][readout] for i in o]))
                test = l2_normalize(confirmation_latent[readout])
                for rule, fit_ordinals in (("NEW_fit8-9", splits["cluster_calibration"]),
                                           ("OLD_fit0-7", splits["kmeans_train"])):
                    model_km = KMeans(n_clusters=2, random_state=config["kmeans_seeds"][0],
                                      n_init=n_init).fit(block(fit_ordinals))
                    calibration = block(splits["cluster_calibration"])
                    calibration_reference = np.concatenate(
                        [reference[i] for i in splits["cluster_calibration"]])
                    vortex = calibrate_vortex_cluster(calibration_reference,
                                                      model_km.predict(calibration))
                    metrics = binary_cluster_metrics(confirmation_reference,
                                                     model_km.predict(test), vortex)
                    rows.append({"dataset": dataset["id"], "train_seed": seed,
                                 "readout": readout, "rule": rule,
                                 "selected_step": history["selected_step"], **metrics})

            info = _reference(dataset["id"])
            new = {r["readout"]: r["f1"] for r in rows
                   if r["train_seed"] == seed and r["rule"] == "NEW_fit8-9"}
            print(f"  seed {seed}: step {history['selected_step']}, "
                  f"{history['seconds']:.0f}s  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in sorted(new.items())), flush=True)
            print(f"    recon held-out 8-9  pre-finetune "
                  f"{history.get('recon_heldout_pre_finetune', float('nan')):.5f}"
                  f"  post {history.get('recon_heldout_post_finetune', float('nan')):.5f}"
                  f"   |  train pre {history.get('recon_train_pre_finetune', float('nan')):.5f}"
                  f"  post {history.get('recon_train_post_finetune', float('nan')):.5f}"
                  f"   zinv drift {history.get('finetune_zinv_drift')}", flush=True)
            curve = history.get("finetune_curve", {})
            if curve.get("step"):
                pairs = list(zip(curve["step"], curve["recon_in_training"], curve["recon_test"]))
                print("    decoder fine-tune recon curve  step: in-training(6,7) / test(8,9)")
                print("      " + "  ".join(f"{st}:{a:.4f}/{b:.4f}" for st, a, b in pairs), flush=True)
            print(f"    reference: FMT {info.get('fmt')}  raw {info.get('raw')}  "
                  f"v1.3 gmean {info.get('v13_gmean', {}).get(seed)}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output = ROOT / config["output"]["result_dir"]
    if arguments.tag:
        output = output.parent / f"{output.name}_{arguments.tag}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "training_history.json").write_text(json.dumps(histories, indent=2),
                                                  encoding="utf-8")
    print(f"\nwrote {output}", flush=True)


if __name__ == "__main__":
    main()
