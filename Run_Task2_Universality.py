"""Train/evaluate raw-VAE and FMT-VAE across held-out timeslices of each 3D flow."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from Train_3DFMT_VAE import _train_vae


def _load_slices(cache_dir):
    records = []
    for path in sorted(Path(cache_dir).glob("slice_*.npz")):
        with np.load(path) as data:
            records.append({
                "path": path, "raw": np.asarray(data["raw_features"], dtype=np.float32),
                "fmt": np.asarray(data["fmt_features"], dtype=np.float32),
                "reference": np.asarray(data["reference"], dtype=bool),
                "metadata": json.loads(str(data["metadata_json"])),
            })
    if len(records) != 10:
        raise RuntimeError(f"expected exactly 10 cached slices in {cache_dir}, found {len(records)}")
    return records


def _metrics(reference, prediction):
    return {
        "f1": float(f1_score(reference, prediction, zero_division=0)),
        "iou": float(jaccard_score(reference, prediction, zero_division=0)),
        "precision": float(precision_score(reference, prediction, zero_division=0)),
        "recall": float(recall_score(reference, prediction, zero_division=0)),
    }


def _fit_cluster(train_features, test_features, reference_test, config, already_scaled=False):
    if already_scaled:
        train_scaled, test_scaled = train_features, test_features
    else:
        scaler = StandardScaler().fit(train_features)
        train_scaled = scaler.transform(train_features)
        test_scaled = scaler.transform(test_features)
    model = KMeans(n_clusters=2, random_state=int(config.task2.kmeans_seed),
                   n_init=int(config.task2.kmeans_n_init)).fit(train_scaled)
    labels = model.predict(test_scaled)
    scores = [(_metrics(reference_test, labels == cluster), cluster) for cluster in (0, 1)]
    score, cluster = max(scores, key=lambda item: item[0]["f1"])
    return labels, int(cluster), score


def _prepare(values, train_lengths, fmt_weight=None):
    train_count = sum(train_lengths)
    scaler = StandardScaler().fit(values[:train_count])
    standardized = scaler.transform(values).astype(np.float32)
    if fmt_weight is not None:
        # Best Task1 encoder: 6 Gram frequencies + 5 chirality values = 23 center slots.
        center_width = 6 * 3 + (6 - 1)
        standardized[:, center_width:] *= float(fmt_weight)
    return standardized[:train_count], standardized[train_count:]


def run_dataset(config, dataset_id):
    records = _load_slices(Path(config.output.cache_dir) / dataset_id)
    train_slice_count = int(config.task2.train_slice_count)
    train_records, test_records = records[:train_slice_count], records[train_slice_count:]
    if not train_records or not test_records:
        raise ValueError("train_slice_count must leave non-empty train and test slices")
    train_lengths = [len(record["reference"]) for record in train_records]
    test_lengths = [len(record["reference"]) for record in test_records]
    reference_train = np.concatenate([record["reference"] for record in train_records])
    reference_test = np.concatenate([record["reference"] for record in test_records])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for representation in ("raw", "fmt"):
        values = np.concatenate([record[representation] for record in records])
        train_x, test_x = _prepare(
            values, train_lengths,
            float(config.task2.fmt_neighbor_weight) if representation == "fmt" else None,
        )
        direct_labels, direct_cluster, direct_score = _fit_cluster(
            train_x, test_x, reference_test, config, already_scaled=True
        )
        rows.append({"dataset": dataset_id, "representation": representation,
                     "variant": f"{representation}_direct", "training_seed": "none",
                     "scope": "all_test", "source_index": "all",
                     "cluster_as_vortex": direct_cluster, **direct_score})
        cursor = 0
        for record, length in zip(test_records, test_lengths):
            prediction = direct_labels[cursor:cursor + length] == direct_cluster
            rows.append({"dataset": dataset_id, "representation": representation,
                         "variant": f"{representation}_direct", "training_seed": "none",
                         "scope": "timeslice", "source_index": record["metadata"]["source_start_index"],
                         "cluster_as_vortex": direct_cluster,
                         **_metrics(record["reference"], prediction)})
            cursor += length

        for seed in config.task2.training_seeds:
            batches_per_epoch = int(math.ceil(len(train_x) / int(config.task2.batch_size)))
            actual_epochs = int(math.ceil(
                int(config.task2.target_optimizer_steps) / batches_per_epoch
            ))
            vae_settings = config.task2.dict()
            vae_settings["epochs"] = actual_epochs
            vae_config = EasyConfig(); vae_config.update({
                "vae": vae_settings, "evaluation": {
                    "kmeans_seed": int(config.task2.kmeans_seed),
                    "kmeans_n_init": int(config.task2.kmeans_n_init),
                }
            })
            train_mu, test_mu, _, losses = _train_vae(
                train_x, test_x, vae_config, int(seed), device
            )
            labels, cluster, aggregate = _fit_cluster(train_mu, test_mu, reference_test, config)
            rows.append({"dataset": dataset_id, "representation": representation,
                         "variant": f"{representation}_vae", "training_seed": int(seed),
                         "scope": "all_test", "source_index": "all",
                         "cluster_as_vortex": cluster, "epochs": actual_epochs,
                         **aggregate, **losses})
            cursor = 0
            for record, length in zip(test_records, test_lengths):
                prediction = labels[cursor:cursor + length] == cluster
                rows.append({"dataset": dataset_id, "representation": representation,
                             "variant": f"{representation}_vae", "training_seed": int(seed),
                             "scope": "timeslice",
                             "source_index": record["metadata"]["source_start_index"],
                             "cluster_as_vortex": cluster, "epochs": actual_epochs,
                             **_metrics(record["reference"], prediction), **losses})
                cursor += length
            print(f"[{dataset_id}] {representation} VAE seed={seed}: "
                  f"held-out-time F1={aggregate['f1']:.4f}")

    output_dir = Path(config.output.result_dir) / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    aggregate_rows = [row for row in rows if row["scope"] == "all_test"]
    for variant in ("raw_direct", "raw_vae", "fmt_direct", "fmt_vae"):
        selected = [row for row in aggregate_rows if row["variant"] == variant]
        values = np.asarray([row["f1"] for row in selected])
        summary[variant] = {"mean_f1": float(values.mean()), "std_f1": float(values.std()),
                            "min_f1": float(values.min()), "max_f1": float(values.max()),
                            "runs": len(values)}
    summary["fmt_vae_minus_raw_vae"] = (
        summary["fmt_vae"]["mean_f1"] - summary["raw_vae"]["mean_f1"]
    )
    payload = {"experiment": str(config.experiment), "dataset": dataset_id,
               "train_source_indices": [r["metadata"]["source_start_index"] for r in train_records],
               "test_source_indices": [r["metadata"]["source_start_index"] for r in test_records],
               "train_samples": int(len(reference_train)), "test_samples": int(len(reference_test)),
               "summary": summary, "config": config.dict()}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = ["Raw", "Raw+VAE", "FMT", "FMT+VAE"]
    variants = ["raw_direct", "raw_vae", "fmt_direct", "fmt_vae"]
    means = [summary[key]["mean_f1"] for key in variants]
    errors = [summary[key]["std_f1"] for key in variants]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=errors, capsize=4,
                  color=["#999999", "#577590", "#f8961e", "#43aa8b"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(ylabel="Held-out-timeslice F1", ylim=(0, 1),
           title=f"Task2 universality: {dataset_id}")
    fig.tight_layout(); fig.savefig(output_dir / "f1_comparison.png", dpi=220); plt.close(fig)
    print(json.dumps(summary, indent=2)); return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_Task2Universality_1.1.yaml")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run_dataset(EasyConfig(args.config), args.dataset)
