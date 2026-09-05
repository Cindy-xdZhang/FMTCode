"""Select one Task1 FMT recipe for every 3D dataset and confirm it.

The selector reads development ordinals only.  It maximizes dataset-macro F1
across all registered datasets, then uses worst-dataset F1 and macro ARI as
tie-breakers.  The selected recipe is frozen before confirmation caches are
opened.  Existing family-specific Task1 results are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.Task12Data_3D import load_cache_records
from experiments.Run_Task1_3D_Main import (
    _final_runs,
    _fit_score,
    _pca_label,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_spec(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "output_dir", "sources", "datasets", "families",
        "splits", "fmt_candidates", "pca_dims", "selection_seed",
        "final_kmeans_seeds", "selection_kmeans_n_init", "kmeans_n_init",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    datasets = list(spec["datasets"])
    if len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be unique")
    if set(spec["families"]) != set(datasets):
        raise ValueError("families must cover every dataset exactly once")
    memberships = [
        dataset for source in spec["sources"].values()
        for dataset in source["datasets"]
    ]
    if sorted(memberships) != sorted(datasets) or len(memberships) != len(datasets):
        raise ValueError("sources must partition datasets exactly once")
    train = {int(value) for value in spec["splits"]["selection_train"]}
    validation = {
        int(value) for value in spec["splits"]["selection_validation"]
    }
    calibration = {
        int(value) for value in spec["splits"]["cluster_calibration"]
    }
    if train & validation or (train | validation) & calibration:
        raise ValueError("selection and calibration ordinals must be disjoint")
    return spec, path


def _source_for_dataset(spec: dict, dataset: str) -> dict:
    matches = [
        source for source in spec["sources"].values()
        if dataset in source["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset!r} matched {len(matches)} sources")
    return matches[0]


def _development(spec: dict, ordinals) -> dict[str, list[dict]]:
    """Open only the explicitly authorized development ordinals."""
    expected = int(spec.get("development_count", 10))
    return {
        dataset: load_cache_records(
            Path(_source_for_dataset(spec, dataset)["development_cache"])
            / dataset,
            expected_count=expected,
            ordinals=[int(value) for value in ordinals],
        )
        for dataset in spec["datasets"]
    }


def _candidate_rows(spec: dict, development: dict, device: str) -> list[dict]:
    rows = []
    train_ids = list(spec["splits"]["selection_train"])
    validation_ids = list(spec["splits"]["selection_validation"])
    candidates = ["raw", *spec["fmt_candidates"]]
    for feature in candidates:
        for pca_dim in spec["pca_dims"]:
            if feature == "raw" and pca_dim is None:
                continue
            for dataset in spec["datasets"]:
                records = development[dataset]
                try:
                    score = _fit_score(
                        [records[index] for index in train_ids],
                        [records[index] for index in validation_ids],
                        feature,
                        pca_dim,
                        int(spec["selection_seed"]),
                        int(spec["selection_kmeans_n_init"]),
                        device,
                    )
                except ValueError as error:
                    if "pca_dim" in str(error):
                        continue
                    raise
                rows.append({
                    "dataset": dataset,
                    "family": spec["families"][dataset],
                    "feature": feature,
                    "pca_dim": _pca_label(pca_dim),
                    **score,
                })
    return rows


def _leaderboard(spec: dict, rows: list[dict], arm: str) -> list[dict]:
    datasets = list(spec["datasets"])
    keys = sorted({
        (row["feature"], row["pca_dim"])
        for row in rows
        if (row["feature"] == "raw") == (arm == "raw")
    })
    result = []
    for feature, pca_dim in keys:
        values = [
            row for row in rows
            if row["feature"] == feature and row["pca_dim"] == pca_dim
        ]
        if {row["dataset"] for row in values} != set(datasets):
            continue
        f1 = np.asarray([float(row["f1"]) for row in values])
        ari = np.asarray([float(row["ari"]) for row in values])
        result.append({
            "arm": arm,
            "feature": feature,
            "pca_dim": pca_dim,
            "dataset_count": len(values),
            "dataset_macro_f1": float(f1.mean()),
            "worst_dataset_f1": float(f1.min()),
            "dataset_macro_ari": float(ari.mean()),
            "dataset_f1_json": json.dumps(
                {row["dataset"]: float(row["f1"]) for row in values},
                sort_keys=True,
            ),
        })
    result.sort(
        key=lambda row: (
            float(row["dataset_macro_f1"]),
            float(row["worst_dataset_f1"]),
            float(row["dataset_macro_ari"]),
            str(row["feature"]),
            str(row["pca_dim"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def select(config_path: str | Path) -> Path:
    spec, config = _load_spec(config_path)
    output = Path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opened_ordinals = sorted(
        set(spec["splits"]["selection_train"])
        | set(spec["splits"]["selection_validation"])
    )
    rows = _candidate_rows(
        spec, _development(spec, opened_ordinals), device
    )
    _write_csv(output / "selection.csv", rows)
    fmt = _leaderboard(spec, rows, "fmt")
    raw = _leaderboard(spec, rows, "raw")
    if not fmt or not raw:
        raise RuntimeError("Task1 uniform selection produced an empty leaderboard")
    _write_csv(output / "global_leaderboard.csv", [*fmt, *raw])
    selected = {
        "experiment": spec["experiment"],
        "status": "frozen_before_confirmation",
        "selection_scope": "one recipe across all datasets",
        "selection_rule": (
            "maximize development dataset-macro F1; tie-break by "
            "worst-dataset F1 and dataset-macro ARI"
        ),
        "opened_ordinals": opened_ordinals,
        "confirmation_opened": False,
        "config_sha256": _sha256(config),
        "fmt": fmt[0],
        "raw": raw[0],
    }
    target = output / "selected_config.json"
    target.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def _choice(row: dict) -> dict:
    return {
        "feature": str(row["feature"]),
        "pca_dim": None if row["pca_dim"] == "none" else int(row["pca_dim"]),
    }


def evaluate(config_path: str | Path) -> Path:
    spec, config = _load_spec(config_path)
    output = Path(spec["output_dir"])
    selection_path = output / "selected_config.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            "run --mode select before opening Task1 confirmation caches"
        )
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    if selected.get("config_sha256") != _sha256(config):
        raise RuntimeError("Task1 uniform config changed after selection freeze")
    if bool(selected.get("confirmation_opened", True)):
        raise RuntimeError("selection manifest was not frozen before confirmation")
    final_development_ordinals = sorted(
        set(spec["splits"]["final_train"])
        | set(spec["splits"]["cluster_calibration"])
    )
    development = _development(spec, final_development_ordinals)
    expected = int(spec.get("confirmation_count", 4))
    confirmation = {
        dataset: load_cache_records(
            Path(_source_for_dataset(spec, dataset)["confirmation_cache"])
            / dataset,
            expected_count=expected,
        )
        for dataset in spec["datasets"]
    }
    global_selected = {
        family: {
            "fmt": _choice(selected["fmt"]),
            "raw": _choice(selected["raw"]),
            "members": [
                dataset for dataset in spec["datasets"]
                if spec["families"][dataset] == family
            ],
        }
        for family in sorted(set(spec["families"].values()))
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = _final_runs(
        spec, development, confirmation, global_selected, device
    )
    _write_csv(output / "final_runs.csv", rows)
    aggregate = [row for row in rows if row["scope"] == "all_confirmation"]
    table = []
    for dataset in spec["datasets"]:
        item = {"dataset": dataset, "family": spec["families"][dataset]}
        for arm in ("raw", "fmt"):
            values = [
                row for row in aggregate
                if row["dataset"] == dataset and row["method"] == arm
            ]
            for metric in ("f1", "iou", "ari", "nmi", "precision", "recall"):
                metric_values = np.asarray(
                    [float(row[metric]) for row in values]
                )
                item[f"{arm}_{metric}_mean"] = float(metric_values.mean())
                item[f"{arm}_{metric}_std"] = float(metric_values.std(ddof=0))
        item["fmt_minus_raw_f1"] = (
            item["fmt_f1_mean"] - item["raw_f1_mean"]
        )
        table.append(item)
    _write_csv(output / "paper_table.csv", table)
    macro = {
        "dataset_count": len(table),
        "kmeans_seed_count": len(spec["final_kmeans_seeds"]),
        "raw_f1": float(np.mean([row["raw_f1_mean"] for row in table])),
        "fmt_f1": float(np.mean([row["fmt_f1_mean"] for row in table])),
        "fmt_minus_raw_f1": float(np.mean([
            row["fmt_minus_raw_f1"] for row in table
        ])),
        "positive_datasets": int(sum(
            row["fmt_minus_raw_f1"] > 0.0 for row in table
        )),
        "worst_dataset_f1_gain": float(min(
            row["fmt_minus_raw_f1"] for row in table
        )),
    }
    summary = {
        "experiment": spec["experiment"],
        "selection": selected,
        "comparison": "one global FMT recipe versus one global Raw recipe",
        "confirmation_opened_after_freeze": True,
        "dataset_macro": macro,
        "datasets": table,
    }
    target = output / "summary.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/mainExp_Task1_3D_4.1_uniform.yaml"
    )
    parser.add_argument("--mode", choices=("select", "evaluate", "all"), required=True)
    args = parser.parse_args()
    if args.mode in ("select", "all"):
        select(args.config)
    if args.mode in ("evaluate", "all"):
        evaluate(args.config)


if __name__ == "__main__":
    main()
