"""Does fitting the cluster boundary on the LATEST development slices transfer better?

`exp_Task1_OctVAE_3D.md` §10.3 shows that four of five training seeds learn a
perfectly good vortex-discriminative latent and fail only because the k-means
boundary fitted on development ordinals 0-7 does not transfer to the
confirmation slices: the latent cloud drifts by ~.76 of its own spread, and the
frozen boundary then sweeps 100% of the confirmation primitives into one cluster.

Hypothesis, registered before looking at any confirmation number: a boundary
fitted on the development slices **closest in time** to where it will be applied
transfers further.  For Re160 the source-frame means are

    ordinals 0-7  ~ frame 59      ordinals 8-9 ~ frame 96
    confirmation  ~ frame 124     ->  gap 65 vs gap 28

The hypothesis is validated on development data only, by treating ordinals 8-9
as a pseudo-confirmation set:

    A (current rule) : fit 0-5,  calibrate 6-7, score 8-9
    B (proposed rule): fit 6-7,  calibrate 6-7, score 8-9

Only if B beats A on development is the analogous rule applied to the real split:

    OLD : fit 0-7, calibrate 8-9, score confirmation
    NEW : fit 8-9, calibrate 8-9, score confirmation

NEW fits k-means and calibrates the anonymous cluster identity on the same
ordinals.  That is not leakage: the k-means fit uses no labels at all, the
calibration is a single bit (which of two clusters is the vortex one), and
neither touches a confirmation slice.  Both rules are development-only.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _cluster(fit_features, calibration_features, calibration_reference,
             score_features, score_reference, seed, n_init):
    model = KMeans(n_clusters=2, random_state=int(seed), n_init=int(n_init)).fit(fit_features)
    vortex = calibrate_vortex_cluster(calibration_reference,
                                      model.predict(calibration_features))
    metrics = binary_cluster_metrics(score_reference, model.predict(score_features), vortex)
    sizes = np.bincount(model.predict(score_features), minlength=2)
    metrics["score_min_cluster_fraction"] = float(sizes.min() / sizes.sum())
    return metrics


def _drift(a, b):
    centre = a.mean(axis=0)
    spread = float(np.linalg.norm(a - centre, axis=1).mean())
    return float(np.linalg.norm(centre - b.mean(axis=0)) / max(spread, 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", default="7069,7070")
    parser.add_argument("--output", default="outputs/exp_Task1_OctVAE_3D_1.3_boundary")
    arguments = parser.parse_args()

    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    dataset = [d for d in config["datasets"] if d["id"] == arguments.dataset][0]
    settings = dict(config["training"]); settings.update(config["model"])
    splits, sampled = config["splits"], int(config["sampled_steps"])
    n_init, readouts = int(config["kmeans_n_init"]), tuple(config.get("readouts", READOUTS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    development = load_cache_records(ROOT / dataset["development_cache"])
    confirmation = load_cache_records(ROOT / dataset["confirmation_cache"])
    matrices, permutation = group_tensors(config["group"], device=device)

    def signal(records, ordinals=None):
        chosen = records if ordinals is None else [records[i] for i in ordinals]
        return np.concatenate([build_signal_from_raw(r["raw"], sampled) for r in chosen])

    reference = {i: np.asarray(development[i]["reference"], dtype=bool)
                 for i in range(len(development))}
    confirmation_reference = stack_reference(confirmation)

    rows = []
    for seed in [int(x) for x in arguments.seeds.split(",")]:
        started = time.time()
        print(f"[{arguments.dataset}] seed {seed}", flush=True)
        model, stats, history = fit_octvae(
            signal(development, splits["vae_train"]),
            signal(development, splits["vae_validation"]),
            settings, seed, device, group=config["group"])

        def encode(values):
            tensor = apply_norm_stats_oh(
                torch.as_tensor(values, dtype=torch.float32, device=device),
                stats, settings["clip_sigma"])
            return encode_readouts(model, tensor, matrices, permutation,
                                   batch_size=config["encode_batch_size"],
                                   readouts=readouts)

        per_ordinal = {i: encode(signal(development, [i])) for i in range(len(development))}
        confirmation_latent = encode(signal(confirmation))

        def block(ordinals, readout):
            return l2_normalize(np.concatenate([per_ordinal[i][readout] for i in ordinals]))

        def block_reference(ordinals):
            return np.concatenate([reference[i] for i in ordinals])

        for readout in readouts:
            test = l2_normalize(confirmation_latent[readout])
            rules = {
                # development-only validation, ordinals 8-9 as pseudo-confirmation
                "devcheck_A_fit0-5": ([0, 1, 2, 3, 4, 5], [6, 7], block([8, 9], readout),
                                      block_reference([8, 9])),
                "devcheck_B_fit6-7": ([6, 7], [6, 7], block([8, 9], readout),
                                      block_reference([8, 9])),
                # the real split
                "OLD_fit0-7": (splits["kmeans_train"], splits["cluster_calibration"],
                               test, confirmation_reference),
                "NEW_fit8-9": (splits["cluster_calibration"], splits["cluster_calibration"],
                               test, confirmation_reference),
            }
            for name, (fit_ords, cal_ords, score_x, score_y) in rules.items():
                metrics = _cluster(block(fit_ords, readout), block(cal_ords, readout),
                                   block_reference(cal_ords), score_x, score_y,
                                   config["kmeans_seeds"][0], n_init)
                rows.append({
                    "dataset": arguments.dataset, "train_seed": seed, "readout": readout,
                    "rule": name, "fit_ordinals": str(list(fit_ords)),
                    "selected_step": history["selected_step"],
                    "drift_fit_to_score": _drift(block(fit_ords, readout), score_x),
                    **metrics})
            plain = readouts[0]
            if readout == plain:
                print(f"  {readout}: "
                      + "  ".join(f"{r['rule']}={r['f1']:.4f}" for r in rows[-4:])
                      + f"   ({time.time() - started:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = ROOT / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{arguments.dataset}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / f"{arguments.dataset}_config.json").write_text(
        json.dumps({"config": arguments.config, "seeds": arguments.seeds}, indent=2),
        encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
