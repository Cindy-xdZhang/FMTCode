"""Independently audit the frozen Task1 global-recipe confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


METRICS = ("f1", "iou", "ari", "nmi", "precision", "recall")

# Independent widths of the frozen Task1 candidate representations.  The
# selector deliberately skips Raw without PCA and any PCA width larger than
# its input feature width (currently kin2/PCA16).  Declaring widths here keeps
# the audit independent of the selector while still detecting a missing cell.
FEATURE_WIDTHS = {
    "fmt_all": 161,
    "fmt_real_all": 42,
    "fmt_real_neighbor": 36,
    "fmt_chirality_all": 35,
    "gram2": 63,
    "gram4": 147,
    "kin2": 12,
    "kin4": 28,
    "gram2+kin2": 75,
    "fmt_real_neighbor+kin2": 48,
    "fmt_all+kin4": 189,
    "raw": 672,
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual: float, expected: float, label: str) -> float:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{label}: recomputed={actual}, published={expected}")
    return actual


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
            raise RuntimeError(f"incomplete selection candidate {feature}/{pca_dim}")
        f1 = np.asarray([float(row["f1"]) for row in values])
        ari = np.asarray([float(row["ari"]) for row in values])
        result.append({
            "feature": feature,
            "pca_dim": pca_dim,
            "dataset_macro_f1": float(f1.mean()),
            "worst_dataset_f1": float(f1.min()),
            "dataset_macro_ari": float(ari.mean()),
        })
    result.sort(key=lambda row: (
        row["dataset_macro_f1"], row["worst_dataset_f1"],
        row["dataset_macro_ari"], row["feature"], row["pca_dim"],
    ), reverse=True)
    return result


def _expected_selection_keys(spec: dict) -> set[tuple[str, str]]:
    declared = [str(value) for value in spec["fmt_candidates"]]
    unknown = sorted(set(declared) - set(FEATURE_WIDTHS))
    if unknown:
        raise RuntimeError(f"Task1 audit lacks feature widths for {unknown}")
    result = set()
    for feature in ["raw", *declared]:
        width = FEATURE_WIDTHS[feature]
        for value in spec["pca_dims"]:
            if feature == "raw" and value is None:
                continue
            if value is not None and int(value) > width:
                continue
            result.add((feature, "none" if value is None else str(int(value))))
    return result


def audit(config_path: Path, output: Path) -> Path:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection_path = output / "selection.csv"
    leaderboard_path = output / "global_leaderboard.csv"
    selected_path = output / "selected_config.json"
    runs_path = output / "final_runs.csv"
    table_path = output / "paper_table.csv"
    summary_path = output / "summary.json"
    required = (
        selection_path, leaderboard_path, selected_path, runs_path,
        table_path, summary_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Task1 evidence: {missing}")

    selection = _read_csv(selection_path)
    expected_candidates = _expected_selection_keys(spec)
    expected_selection = len(spec["datasets"]) * len(expected_candidates)
    if len(selection) != expected_selection:
        raise RuntimeError(
            f"selection rows: {len(selection)} != {expected_selection}"
        )
    observed_candidates = {
        (str(row["feature"]), str(row["pca_dim"])) for row in selection
    }
    if observed_candidates != expected_candidates:
        raise RuntimeError(
            "Task1 selection candidate set changed: "
            f"missing={sorted(expected_candidates - observed_candidates)}, "
            f"extra={sorted(observed_candidates - expected_candidates)}"
        )
    keys = [(row["dataset"], row["feature"], row["pca_dim"]) for row in selection]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate Task1 selection rows")

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("status") != "frozen_before_confirmation":
        raise RuntimeError("Task1 selection was not frozen before confirmation")
    if selected.get("opened_ordinals") != list(range(8)):
        raise RuntimeError(f"unexpected selection ordinals: {selected.get('opened_ordinals')}")
    if selected.get("confirmation_opened") is not False:
        raise RuntimeError("selection manifest says confirmation was opened")
    if selected.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("Task1 config changed after selection")
    winners = {
        arm: _leaderboard(spec, selection, arm)[0] for arm in ("fmt", "raw")
    }
    for arm, winner in winners.items():
        published = selected[arm]
        for key in ("feature", "pca_dim"):
            if str(winner[key]) != str(published[key]):
                raise RuntimeError(f"{arm} winner {key} changed")
        for key in ("dataset_macro_f1", "worst_dataset_f1", "dataset_macro_ari"):
            _close(float(winner[key]), float(published[key]), f"{arm}.{key}")

    runs = _read_csv(runs_path)
    aggregate = [row for row in runs if row["scope"] == "all_confirmation"]
    timeslices = [row for row in runs if row["scope"] == "timeslice"]
    expected_aggregate = len(spec["datasets"]) * 2 * len(spec["final_kmeans_seeds"])
    expected_timeslices = expected_aggregate * int(spec["confirmation_count"])
    if len(aggregate) != expected_aggregate or len(timeslices) != expected_timeslices:
        raise RuntimeError(
            f"confirmation rows aggregate/timeslice={len(aggregate)}/{len(timeslices)}, "
            f"expected={expected_aggregate}/{expected_timeslices}"
        )
    run_keys = [
        (row["dataset"], row["method"], row["kmeans_seed"], row["scope"],
         row.get("source_index", ""))
        for row in runs
    ]
    if len(run_keys) != len(set(run_keys)):
        raise RuntimeError("duplicate Task1 confirmation rows")
    for row in aggregate:
        arm = row["method"]
        if row["feature"] != str(selected[arm]["feature"]):
            raise RuntimeError("confirmation changed the global feature")
        if row["pca_dim"] != str(selected[arm]["pca_dim"]):
            raise RuntimeError("confirmation changed the global PCA dimension")

    published_table = {row["dataset"]: row for row in _read_csv(table_path)}
    recomputed = []
    for dataset in spec["datasets"]:
        item = {"dataset": dataset}
        for arm in ("raw", "fmt"):
            values = [
                row for row in aggregate
                if row["dataset"] == dataset and row["method"] == arm
            ]
            if len(values) != len(spec["final_kmeans_seeds"]):
                raise RuntimeError(f"incomplete {dataset}/{arm} aggregate")
            for metric in METRICS:
                array = np.asarray([float(row[metric]) for row in values])
                item[f"{arm}_{metric}_mean"] = float(array.mean())
                item[f"{arm}_{metric}_std"] = float(array.std(ddof=0))
        item["fmt_minus_raw_f1"] = item["fmt_f1_mean"] - item["raw_f1_mean"]
        recomputed.append(item)
        for key, value in item.items():
            if key != "dataset":
                _close(value, float(published_table[dataset][key]), f"{dataset}.{key}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    macro = summary["dataset_macro"]
    raw_f1 = float(np.mean([row["raw_f1_mean"] for row in recomputed]))
    fmt_f1 = float(np.mean([row["fmt_f1_mean"] for row in recomputed]))
    gain = fmt_f1 - raw_f1
    checks = {
        "raw_f1": _close(raw_f1, float(macro["raw_f1"]), "macro.raw_f1"),
        "fmt_f1": _close(fmt_f1, float(macro["fmt_f1"]), "macro.fmt_f1"),
        "fmt_minus_raw_f1": _close(
            gain, float(macro["fmt_minus_raw_f1"]), "macro.gain"
        ),
        "positive_datasets": sum(row["fmt_minus_raw_f1"] > 0 for row in recomputed),
        "worst_dataset_f1_gain": min(row["fmt_minus_raw_f1"] for row in recomputed),
    }
    if checks["positive_datasets"] != int(macro["positive_datasets"]):
        raise RuntimeError("positive dataset count changed")
    _close(
        checks["worst_dataset_f1_gain"],
        float(macro["worst_dataset_f1_gain"]),
        "macro.worst_dataset_f1_gain",
    )
    payload = {
        "experiment": "Audit_mainExp_Task1_3D_4.1",
        "status": "PASS",
        "selection_rows": len(selection),
        "confirmation_rows": len(runs),
        "selected_fmt": selected["fmt"],
        "selected_raw": selected["raw"],
        "dataset_macro": checks,
        "evidence_sha256": {path.name: _sha256(path) for path in required},
    }
    target = output / "independent_audit.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/mainExp_Task1_3D_4.1_uniform.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/mainExp_Task1_3D_4.1"),
    )
    args = parser.parse_args()
    audit(args.config, args.output)


if __name__ == "__main__":
    main()
