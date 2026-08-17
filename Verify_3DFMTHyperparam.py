"""Verify_3DFMTHyperparam_1.1: frozen-reference search for 3D FMT settings."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FLowUtils.ScalarField3d import compute_local_ivd_3D
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_3D_pipeline import load_vector_field_3d


def _frozen_reference(config, seeds):
    field = load_vector_field_3d(Path(config.input.field_path))
    physical_time = float(config.input.physical_time)
    float_time = (physical_time - field.tmin) / field.timeInterval
    t0 = int(np.clip(np.floor(float_time), 0, field.time_steps - 1))
    t1 = int(np.clip(np.ceil(float_time), 0, field.time_steps - 1))
    weight = float_time - t0 if t1 != t0 else 0.0
    frame = (1.0 - weight) * field.field[t0] + weight * field.field[t1]
    dx, dy, dz = (float(v) for v in field.gridInterval)
    volume = compute_local_ivd_3D(
        np.asarray(frame, dtype=np.float32), dx, dy, dz,
        int(config.input.ivd_averaging_size),
    )
    dmin = np.asarray(field.domainMinBoundary); dmax = np.asarray(field.domainMaxBoundary)
    xs = np.linspace(dmin[0], dmax[0], field.Xdim)
    ys = np.linspace(dmin[1], dmax[1], field.Ydim)
    zs = np.linspace(dmin[2], dmax[2], field.Zdim)
    values = RegularGridInterpolator((zs, ys, xs), volume)(seeds[:, [2, 1, 0]])
    return values >= float(config.input.ivd_threshold)


def _score(labels, reference):
    candidates = []
    for cluster in np.unique(labels):
        prediction = labels == cluster
        candidates.append({
            "cluster_as_vortex": int(cluster),
            "f1": float(f1_score(reference, prediction, zero_division=0)),
            "iou": float(jaccard_score(reference, prediction, zero_division=0)),
            "precision": float(precision_score(reference, prediction, zero_division=0)),
            "recall": float(recall_score(reference, prediction, zero_division=0)),
            "predicted_fraction": float(prediction.mean()),
        })
    return max(candidates, key=lambda row: row["f1"])


def _standardized_features(primitives, num_freq, mode, chirality, pool, weight, device):
    features = pathline_dft_features_3d(
        primitives.to(device), num_freq=num_freq, neighbor_weight=1.0,
        neighbor_scale=1.0, neighbor_pool=pool, mode=mode,
        include_chirality=chirality,
    )
    standardized = StandardScaler().fit_transform(features)
    center_width = num_freq * (1 if mode == "magnitude" else 3)
    if chirality:
        center_width += num_freq - 1
    standardized[:, center_width:] *= weight
    return standardized


def run(config):
    started = time.time()
    run_dir = Path(config.input.run_dir)
    with np.load(run_dir / "clustering_result.npz") as data:
        primitives = torch.from_numpy(np.asarray(data["pathlines"], dtype=np.float32))
        seeds = np.asarray(data["seeds"], dtype=np.float64)
    reference = _frozen_reference(config, seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    descriptor_cache = {}
    combinations = list(itertools.product(
        config.search.num_freq, config.search.mode, config.search.include_chirality,
        config.search.neighbor_pool, config.search.neighbor_weight,
    ))
    for index, (freq, mode, chirality, pool, weight) in enumerate(combinations, 1):
        key = (int(freq), str(mode), bool(chirality), str(pool))
        if key not in descriptor_cache:
            descriptor_cache[key] = _standardized_features(
                primitives, *key, weight=1.0, device=device
            )
        standardized = descriptor_cache[key].copy()
        center_width = int(freq) * (1 if str(mode) == "magnitude" else 3)
        if bool(chirality): center_width += int(freq) - 1
        standardized[:, center_width:] *= float(weight)
        labels = KMeans(n_clusters=2, random_state=int(config.seed),
                        n_init=int(config.search.coarse_kmeans_n_init)).fit_predict(standardized)
        result = _score(labels, reference)
        rows.append({"num_freq": int(freq), "mode": str(mode),
                     "include_chirality": bool(chirality), "neighbor_pool": str(pool),
                     "neighbor_weight": float(weight), **result})
        if index % 25 == 0 or index == len(combinations):
            print(f"coarse search {index}/{len(combinations)}; best F1={max(r['f1'] for r in rows):.4f}")

    rows.sort(key=lambda row: row["f1"], reverse=True)
    stability = []
    for rank, row in enumerate(rows[:int(config.search.top_k_for_stability)], 1):
        key = (row["num_freq"], row["mode"], row["include_chirality"], row["neighbor_pool"])
        standardized = descriptor_cache[key].copy()
        center_width = key[0] * (1 if key[1] == "magnitude" else 3)
        if key[2]: center_width += key[0] - 1
        standardized[:, center_width:] *= row["neighbor_weight"]
        scores = []
        for random_state in config.search.stability_seeds:
            labels = KMeans(n_clusters=2, random_state=int(random_state),
                            n_init=int(config.search.stability_kmeans_n_init)).fit_predict(standardized)
            scores.append(_score(labels, reference))
        stability.append({
            **{k: row[k] for k in ("num_freq", "mode", "include_chirality",
                                    "neighbor_pool", "neighbor_weight")},
            "coarse_rank": rank,
            "mean_f1": float(np.mean([s["f1"] for s in scores])),
            "std_f1": float(np.std([s["f1"] for s in scores])),
            "min_f1": float(np.min([s["f1"] for s in scores])),
            "max_f1": float(np.max([s["f1"] for s in scores])),
        })
    stability.sort(key=lambda row: (row["mean_f1"], -row["std_f1"]), reverse=True)

    output_dir = Path(config.output.dir); output_dir.mkdir(parents=True, exist_ok=True)
    def write_csv(path, table):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    write_csv(output_dir / "all_results.csv", rows)
    write_csv(output_dir / "top_stability.csv", stability)
    summary = {
        "experiment": str(config.experiment), "device": str(device),
        "combination_count": len(combinations), "reference_positive_count": int(reference.sum()),
        "reference_positive_fraction": float(reference.mean()),
        "best": stability[0], "elapsed_seconds": time.time() - started,
        "config": config.dict(),
        "warning": "Hyperparameters are selected on one field/time and require held-out validation.",
    }
    (output_dir / "best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top = stability[:10]
    labels_plot = [f"{r['num_freq']}/{r['mode'][0]}/c{int(r['include_chirality'])}/"
                   f"{r['neighbor_pool']}/{r['neighbor_weight']:g}"
                   for r in top]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(top)), [r["mean_f1"] for r in top],
           yerr=[r["std_f1"] for r in top], color="#2878b5", capsize=3)
    ax.set_xticks(np.arange(len(top)), labels_plot, rotation=35, ha="right")
    ax.set(ylabel="F1 (mean over KMeans seeds)",
           xlabel="freq/mode/chirality/pool/neighbor_weight",
           title="Verify_3DFMTHyperparam_1.1: top stable configurations", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(output_dir / "top_hyperparameters.png", dpi=220); plt.close(fig)
    print(json.dumps(summary["best"], indent=2)); print(f"output: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_3DFMTHyperparam_1.1.yaml")
    args = parser.parse_args()
    run(EasyConfig(args.config))
