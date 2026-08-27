"""Deterministic linear pre-screen for robust 3D Task3 FMT blocks.

This development-only diagnostic trains one balanced logistic classifier per
feature block.  Each FMT block is paired with train-only Raw-PCA of exactly
the same width.  The decision threshold is selected on the already exposed
base validation population and then reused unchanged on the exposed spatial
population.  It does not replace the residual-network comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from FMT_Utils.Task12Data_3D import feature_matrix, load_cache_records
from Search_Task3_FMTResidual_3D import (
    _group_for_dataset,
    _load_spec,
    _portable_basename,
    _write_csv,
)
from Verify_Task3_FMTClassifier import _select_f1_threshold


def _load_population(
    source_root, label_root, dataset, expected_count,
    *, allow_missing_label_cache=False,
):
    records = load_cache_records(
        Path(source_root) / dataset, expected_count=int(expected_count)
    )
    for record in records:
        label_path = Path(label_root) / dataset / record["path"].name
        if not label_path.exists() and allow_missing_label_cache:
            print(
                "WARNING stage0 uses source-cache reference because the local "
                f"label mirror is absent: {label_path}",
                flush=True,
            )
            continue
        with np.load(label_path) as source:
            labels = np.asarray(source["labels"], dtype=bool)
            metadata = json.loads(str(source["metadata_json"]))
        if _portable_basename(metadata["source_cache"]) != record["path"].name:
            raise ValueError(f"label/source mismatch for {record['path']}")
        if not np.array_equal(labels, record["reference"]):
            raise RuntimeError(f"label/reference mismatch for {record['path']}")
    return records


def _stack(records, ordinals, feature, device):
    chosen = [records[int(index)] for index in ordinals]
    return (
        np.concatenate([feature_matrix(row, feature, device) for row in chosen]),
        np.concatenate([row["reference"] for row in chosen]),
    )


def _fit(train_x, train_y, base_x, base_y, spatial_x, spatial_y):
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=7068,
        solver="lbfgs",
    ).fit(scaler.transform(train_x), train_y)
    base_probability = classifier.predict_proba(scaler.transform(base_x))[:, 1]
    spatial_probability = classifier.predict_proba(
        scaler.transform(spatial_x)
    )[:, 1]
    threshold = _select_f1_threshold(base_y, base_probability)
    return {
        "base_f1": float(f1_score(base_y, base_probability >= threshold)),
        "base_average_precision": float(
            average_precision_score(base_y, base_probability)
        ),
        "spatial_f1": float(
            f1_score(spatial_y, spatial_probability >= threshold)
        ),
        "spatial_average_precision": float(
            average_precision_score(spatial_y, spatial_probability)
        ),
        "base_selected_threshold": float(threshold),
    }


def _dataset_rows(spec, dataset, device, allow_missing_label_cache=False):
    group_name, group = _group_for_dataset(spec, dataset)
    robust = spec["robust_validation"]
    base = _load_population(
        group["source_cache_root"], group["label_cache_root"], dataset,
        spec["expected_slices"],
        allow_missing_label_cache=allow_missing_label_cache,
    )
    spatial = _load_population(
        group["exposed_spatial_source_cache_root"],
        group["exposed_spatial_label_cache_root"], dataset,
        robust["expected_slices"],
        allow_missing_label_cache=allow_missing_label_cache,
    )
    train_ordinals = spec["screen_split"]["train_ordinals"]
    base_ordinals = spec["screen_split"]["validation_ordinals"]
    spatial_ordinals = robust["ordinals"]
    raw_train, train_y = _stack(base, train_ordinals, "raw", device)
    raw_base, base_y = _stack(base, base_ordinals, "raw", device)
    raw_spatial, spatial_y = _stack(spatial, spatial_ordinals, "raw", device)
    raw_scaler = StandardScaler().fit(raw_train)
    raw_train = raw_scaler.transform(raw_train)
    raw_base = raw_scaler.transform(raw_base)
    raw_spatial = raw_scaler.transform(raw_spatial)
    pca_cache = {}
    rows = []
    for candidate in spec["candidates"]:
        feature = str(candidate["fmt_feature"])
        fmt_train, _ = _stack(base, train_ordinals, feature, device)
        fmt_base, _ = _stack(base, base_ordinals, feature, device)
        fmt_spatial, _ = _stack(spatial, spatial_ordinals, feature, device)
        width = int(fmt_train.shape[1])
        if width not in pca_cache:
            pca = PCA(
                n_components=width,
                svd_solver="randomized",
                random_state=int(spec["raw_pca_random_state"]),
                iterated_power=4,
            ).fit(raw_train)
            pca_cache[width] = (
                pca.transform(raw_train),
                pca.transform(raw_base),
                pca.transform(raw_spatial),
            )
        raw_pca = pca_cache[width]
        fmt_metrics = _fit(
            fmt_train, train_y, fmt_base, base_y, fmt_spatial, spatial_y
        )
        raw_metrics = _fit(
            raw_pca[0], train_y, raw_pca[1], base_y,
            raw_pca[2], spatial_y,
        )
        row = {
            "group": group_name,
            "dataset": dataset,
            "candidate_id": candidate["id"],
            "fmt_feature": feature,
            "feature_width": width,
        }
        for population in ("base", "spatial"):
            for metric in ("f1", "average_precision"):
                row[f"fmt_{population}_{metric}"] = fmt_metrics[
                    f"{population}_{metric}"
                ]
                row[f"raw_pca_{population}_{metric}"] = raw_metrics[
                    f"{population}_{metric}"
                ]
                row[f"gain_{population}_{metric}"] = (
                    fmt_metrics[f"{population}_{metric}"]
                    - raw_metrics[f"{population}_{metric}"]
                )
        row["robust_min_gain"] = min(
            row[f"gain_{population}_{metric}"]
            for population in ("base", "spatial")
            for metric in ("f1", "average_precision")
        )
        row["robust_mean_gain"] = float(np.mean([
            row[f"gain_{population}_{metric}"]
            for population in ("base", "spatial")
            for metric in ("f1", "average_precision")
        ]))
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def run(config_path, datasets=None, allow_missing_label_cache=False):
    spec = _load_spec(config_path)
    selected = list(spec["datasets"] if datasets is None else datasets)
    unknown = sorted(set(selected) - set(spec["datasets"]))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in selected:
        rows.extend(_dataset_rows(
            spec, dataset, device,
            allow_missing_label_cache=allow_missing_label_cache,
        ))
    output = Path(spec["output_root"]) / "stage0"
    _write_csv(output / "per_dataset_candidate.csv", rows)
    family_rows = []
    for group_name in spec["groups"]:
        group_datasets = set(spec["groups"][group_name]["datasets"])
        if not group_datasets.issubset(selected):
            continue
        for candidate in spec["candidates"]:
            chosen = [
                row for row in rows
                if row["dataset"] in group_datasets
                and row["candidate_id"] == candidate["id"]
            ]
            gains = {
                f"family_macro_{population}_{metric}_gain": float(np.mean([
                    row[f"gain_{population}_{metric}"] for row in chosen
                ]))
                for population in ("base", "spatial")
                for metric in ("f1", "average_precision")
            }
            family_rows.append({
                "group": group_name,
                "candidate_id": candidate["id"],
                "fmt_feature": candidate["fmt_feature"],
                "feature_width": chosen[0]["feature_width"],
                "robust_min_family_macro_gain": min(gains.values()),
                "robust_mean_family_macro_gain": float(np.mean(list(gains.values()))),
                **gains,
            })
    family_rows.sort(
        key=lambda row: (
            row["group"], -row["robust_min_family_macro_gain"],
            -row["robust_mean_family_macro_gain"],
        )
    )
    _write_csv(output / "family_leaderboard.csv", family_rows)
    print(f"wrote {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--allow-missing-label-cache",
        action="store_true",
        help=(
            "development-only local fallback to the source-cache reference; "
            "the full residual search never permits this"
        ),
    )
    args = parser.parse_args()
    run(args.config, args.dataset, args.allow_missing_label_cache)


if __name__ == "__main__":
    main()
