"""Fast development-only screen for F22 anchored FMT feature blocks.

This is not the final Task3 network comparison.  It uses a fixed balanced
logistic classifier to remove clearly weak feature recipes before the paired
residual-network search.  Raw-PCA is fit on train ordinals only and always has
the same dimension as the corresponding FMT block.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from FMT_Utils.Task12Data_3D import feature_matrix, load_cache_records


def _best_f1(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    scores = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(scores))
    threshold = 0.5 if index >= len(thresholds) else float(thresholds[index])
    return float(f1_score(labels, probabilities >= threshold)), threshold


def _fit_score(train_x, train_y, validation_x, validation_y):
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=7068,
        solver="lbfgs",
    ).fit(scaler.transform(train_x), train_y)
    probabilities = model.predict_proba(scaler.transform(validation_x))[:, 1]
    f1, threshold = _best_f1(validation_y, probabilities)
    return {
        "f1": f1,
        "average_precision": float(
            average_precision_score(validation_y, probabilities)
        ),
        "threshold": threshold,
    }


def run(config_path: str | Path):
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    group = spec["groups"]["f22raptor"]
    records = load_cache_records(
        Path(group["source_cache_root"]) / "f22raptor",
        expected_count=int(spec["expected_slices"]),
    )
    labels = {}
    for record in records:
        path = Path(group["label_cache_root"]) / "f22raptor" / record["path"].name
        with np.load(path) as data:
            labels[int(record["ordinal"])] = np.asarray(
                data["labels"], dtype=np.int64
            )
    train_ordinals = [int(x) for x in spec["screen_split"]["train_ordinals"]]
    validation_ordinals = [
        int(x) for x in spec["screen_split"]["validation_ordinals"]
    ]
    by_ordinal = {int(record["ordinal"]): record for record in records}
    train_raw = np.concatenate(
        [by_ordinal[i]["raw"] for i in train_ordinals], axis=0
    )
    validation_raw = np.concatenate(
        [by_ordinal[i]["raw"] for i in validation_ordinals], axis=0
    )
    train_y = np.concatenate([labels[i] for i in train_ordinals])
    validation_y = np.concatenate([labels[i] for i in validation_ordinals])
    raw_scaler = StandardScaler().fit(train_raw)
    raw_train_scaled = raw_scaler.transform(train_raw)
    raw_validation_scaled = raw_scaler.transform(validation_raw)
    pca_cache = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for candidate in spec["candidates"]:
        name = str(candidate["fmt_feature"])
        train_fmt = np.concatenate([
            feature_matrix(by_ordinal[i], name, device) for i in train_ordinals
        ])
        validation_fmt = np.concatenate([
            feature_matrix(by_ordinal[i], name, device)
            for i in validation_ordinals
        ])
        dimension = int(train_fmt.shape[1])
        if dimension not in pca_cache:
            pca = PCA(
                n_components=dimension,
                svd_solver="randomized",
                random_state=int(spec["raw_pca_random_state"]),
            ).fit(raw_train_scaled)
            pca_cache[dimension] = (
                pca.transform(raw_train_scaled),
                pca.transform(raw_validation_scaled),
            )
        raw_pca_train, raw_pca_validation = pca_cache[dimension]
        fmt_metrics = _fit_score(
            train_fmt, train_y, validation_fmt, validation_y
        )
        raw_metrics = _fit_score(
            raw_pca_train, train_y, raw_pca_validation, validation_y
        )
        row = {
            "candidate_id": candidate["id"],
            "fmt_feature": name,
            "dimension": dimension,
            "fmt_f1": fmt_metrics["f1"],
            "raw_pca_f1": raw_metrics["f1"],
            "f1_gain": fmt_metrics["f1"] - raw_metrics["f1"],
            "fmt_average_precision": fmt_metrics["average_precision"],
            "raw_pca_average_precision": raw_metrics["average_precision"],
            "average_precision_gain": (
                fmt_metrics["average_precision"]
                - raw_metrics["average_precision"]
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    rows.sort(
        key=lambda row: (
            min(row["f1_gain"], row["average_precision_gain"]),
            row["f1_gain"],
            row["fmt_average_precision"],
        ),
        reverse=True,
    )
    output = Path(spec["output_root"]) / "fast_screen.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
