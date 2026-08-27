"""Development-only family search for supervised 3D Task3 FMT residuals.

Every FMT candidate is paired with a train-only Raw-PCA residual of exactly
the same auxiliary width and trainable architecture.  Candidate selection
opens only the registered development train/validation ordinals.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.Task12Data_3D import feature_matrix, load_cache_records
from Verify_Task3_FMTClassifier import (
    _append_csv,
    _normalize_train_only,
    _portable_basename,
    _stack_split,
)
from Verify_Task3_FMTResidual import _train_one


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "groups", "datasets", "candidates",
        "screen_seeds", "screen_split", "training",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    datasets = list(spec["datasets"])
    memberships = [
        dataset for group in spec["groups"].values()
        for dataset in group["datasets"]
    ]
    if len(datasets) != len(set(datasets)) or sorted(datasets) != sorted(memberships):
        raise ValueError("groups must partition unique datasets exactly once")
    candidate_ids = [str(row["id"]) for row in spec["candidates"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate ids must be unique")
    train = {int(value) for value in spec["screen_split"]["train_ordinals"]}
    validation = {
        int(value) for value in spec["screen_split"]["validation_ordinals"]
    }
    outer = {int(value) for value in spec.get("outer_ordinals", [])}
    if train & validation or (train | validation) & outer:
        raise ValueError("Task3 train, validation, and outer ordinals must be disjoint")
    for group in spec["groups"].values():
        for key in ("source_cache_root", "label_cache_root"):
            root = str(group[key]).lower()
            if "confirmation" in root or "test" in root:
                raise ValueError(
                    f"development search cannot read held-out {key}: {root}"
                )
    return spec


def _group_for_dataset(spec: dict, dataset: str) -> tuple[str, dict]:
    matches = [
        (name, group) for name, group in spec["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset!r} matched {len(matches)} groups")
    return matches[0]


def _candidate(spec: dict, index: int) -> dict:
    index = int(index)
    if not 0 <= index < len(spec["candidates"]):
        raise IndexError(
            f"candidate index {index} outside [0,{len(spec['candidates'])})"
        )
    row = dict(spec["candidates"][index])
    row["index"] = index
    return row


def _load_records(spec: dict, dataset: str, candidate: dict, device,
                  ordinals=None) -> list[tuple]:
    _, group = _group_for_dataset(spec, dataset)
    required = sorted(
        {int(value) for value in spec["screen_split"]["train_ordinals"]}
        | {int(value) for value in spec["screen_split"]["validation_ordinals"]}
    ) if ordinals is None else sorted({int(value) for value in ordinals})
    source_dir = Path(group["source_cache_root"]) / dataset
    records = load_cache_records(
        source_dir,
        expected_count=int(spec.get("expected_slices", 10)),
        ordinals=required,
    )
    result = []
    for record in records:
        label_path = Path(group["label_cache_root"]) / dataset / record["path"].name
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        with np.load(label_path) as label_file:
            labels = np.asarray(label_file["labels"], dtype=np.float32)
            metadata = json.loads(str(label_file["metadata_json"]))
        if _portable_basename(metadata["source_cache"]) != record["path"].name:
            raise ValueError(f"label/source mismatch for {record['path']}")
        if not np.array_equal(labels.astype(bool), record["reference"]):
            raise RuntimeError(
                f"Task3 global-IVD labels differ from source reference: {label_path}"
            )
        fmt = feature_matrix(record, str(candidate["fmt_feature"]), device)
        sampled_steps = record["raw"].shape[1] // (7 * 3)
        raw = record["raw"].reshape(-1, 7, sampled_steps, 3)
        if len(raw) != len(fmt) or len(raw) != len(labels):
            raise ValueError(f"feature/label length mismatch in {record['path']}")
        result.append((
            raw, fmt, labels, int(record["ordinal"]), metadata,
        ))
    return result


def _fusion(candidate: dict) -> dict:
    if "fixed_alpha" in candidate:
        return {
            "fixed_alpha": float(candidate["fixed_alpha"]),
            "selection_metric": str(
                candidate.get("selection_metric", "average_precision")
            ),
        }
    return {
        "alpha_min": float(candidate.get("alpha_min", 0.0)),
        "alpha_max": float(candidate.get("alpha_max", 3.0)),
        "alpha_steps": int(candidate.get("alpha_steps", 61)),
        "selection_metric": str(candidate.get("selection_metric", "minimum_gain")),
        "minimum_f1_gain": float(candidate.get("minimum_f1_gain", 0.0)),
    }


def _training(spec: dict, candidate: dict, seed: int) -> dict:
    settings = dict(spec["training"])
    settings.update(candidate.get("training", {}))
    settings["seeds"] = [int(seed)]
    return settings


def _candidate_spec(spec: dict, group: dict, candidate: dict, dataset: str,
                    seed: int, source: str, output_dir: Path,
                    fmt_dim: int) -> dict:
    return {
        "experiment": f"{spec['experiment']}_{candidate['id']}_{source}",
        "source_cache_root": group["source_cache_root"],
        "label_cache_root": group["label_cache_root"],
        "raw_checkpoint_dir": group["raw_checkpoint_dir"],
        "output_dir": str(output_dir),
        "datasets": [dataset],
        "expected_slices": int(spec.get("expected_slices", 10)),
        "sampled_steps": int(spec.get("sampled_steps", 32)),
        "fmt_subset": candidate["fmt_feature"],
        "auxiliary_source": source,
        "raw_pca_components": int(fmt_dim),
        "raw_pca_random_state": int(spec.get("raw_pca_random_state", 7068)),
        "raw_wide_parameter_count": int(spec["raw_wide_parameter_count"]),
        "split": dict(spec["screen_split"]),
        "evaluation": {"test_enabled": False},
        "model": {
            "embedding_dim": int(candidate.get("embedding_dim", 128)),
            "auxiliary_dim": int(candidate.get("auxiliary_dim", 64)),
            "residual_input": str(candidate.get("residual_input", "geometry_fmt")),
        },
        "fusion": _fusion(candidate),
        "training": _training(spec, candidate, seed),
        "search_candidate": candidate,
    }


def _result_path(spec: dict, candidate: dict, dataset: str,
                 seed: int, source: str) -> Path:
    return (
        Path(spec["output_root"]) / "stage1" / "candidates"
        / str(candidate["id"]) / dataset / f"seed{int(seed)}"
        / source / "per_run.csv"
    )


def run_candidate(config_path: str, dataset: str, candidate_index: int) -> Path:
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}")
    candidate = _candidate(spec, candidate_index)
    _, group = _group_for_dataset(spec, dataset)
    device_name = str(spec["training"].get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    records = _load_records(spec, dataset, candidate, device)
    train = _stack_split(records, spec["screen_split"]["train_ordinals"])
    validation = _stack_split(
        records, spec["screen_split"]["validation_ordinals"]
    )
    train, validation, _, stats = _normalize_train_only(train, validation)
    fmt_dim = int(train[1].shape[1])
    if fmt_dim > train[0].reshape(len(train[0]), -1).shape[1]:
        raise ValueError(
            f"matched Raw-PCA cannot provide {fmt_dim} components for "
            f"{candidate['id']}"
        )
    last_path = None
    for seed_value in spec["screen_seeds"]:
        seed = int(seed_value)
        for source in ("fmt", "raw_pca"):
            result_path = _result_path(spec, candidate, dataset, seed, source)
            existing = [
                row for row in _read_csv(result_path)
                if row["dataset"] == dataset and int(row["seed"]) == seed
            ]
            if len(existing) > 1:
                raise RuntimeError(f"duplicate candidate result: {result_path}")
            if existing:
                print(f"cached {candidate['id']} {dataset} seed={seed} {source}")
                last_path = result_path
                continue
            output_dir = result_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, seed, source, output_dir, fmt_dim
            )
            (output_dir / "config_snapshot.yaml").write_text(
                yaml.safe_dump(run_spec, sort_keys=False), encoding="utf-8"
            )
            row = _train_one(
                run_spec, dataset, seed, (train, validation, None), stats,
                device, output_dir,
            )
            row.update({
                "candidate_id": candidate["id"],
                "candidate_index": int(candidate_index),
                "fmt_feature": candidate["fmt_feature"],
                "fmt_dim": fmt_dim,
            })
            _append_csv(result_path, row)
            last_path = result_path
            print(
                f"DONE {candidate['id']} {dataset} seed={seed} {source}: "
                f"F1={row['validation_f1']:.5f} "
                f"AP={row['validation_average_precision']:.5f}",
                flush=True,
            )
    return last_path


def _decode_job(spec: dict, job_index: int) -> tuple[str, int]:
    count = len(spec["datasets"]) * len(spec["candidates"])
    index = int(job_index)
    if not 0 <= index < count:
        raise IndexError(f"Task3 job index {index} outside [0,{count})")
    dataset_index, candidate_index = divmod(index, len(spec["candidates"]))
    return spec["datasets"][dataset_index], candidate_index


def run_job(config_path: str, job_index: int) -> Path:
    spec = _load_spec(config_path)
    dataset, candidate_index = _decode_job(spec, job_index)
    return run_candidate(
        config_path, dataset, candidate_index
    )


def _baseline_from_row(row: dict) -> dict:
    value = row.get("validation_selection_baseline", "")
    if isinstance(value, dict):
        parsed = value
    else:
        parsed = ast.literal_eval(str(value))
    return {
        "f1": float(parsed["f1"]),
        "average_precision": float(parsed["average_precision"]),
    }


def _candidate_summary(spec: dict, group_name: str, candidate: dict) -> dict:
    datasets = spec["groups"][group_name]["datasets"]
    seeds = [int(value) for value in spec["screen_seeds"]]
    metrics = ("f1", "average_precision")
    per_dataset = {}
    seed_f1_gains = {seed: [] for seed in seeds}
    for dataset in datasets:
        per_seed = {}
        for seed in seeds:
            rows = {}
            for source in ("fmt", "raw_pca"):
                path = _result_path(spec, candidate, dataset, seed, source)
                values = _read_csv(path)
                if len(values) != 1:
                    raise RuntimeError(
                        f"incomplete Task3 result {candidate['id']}/{dataset}/"
                        f"seed={seed}/{source}: {len(values)}"
                    )
                rows[source] = values[0]
            strong = _baseline_from_row(rows["fmt"])
            fmt = {metric: float(rows["fmt"][f"validation_{metric}"])
                   for metric in metrics}
            raw_pca = {
                metric: float(rows["raw_pca"][f"validation_{metric}"])
                for metric in metrics
            }
            per_seed[seed] = {"fmt": fmt, "raw_pca": raw_pca, "strong_raw": strong}
            seed_f1_gains[seed].append(fmt["f1"] - raw_pca["f1"])
        per_dataset[dataset] = {}
        for source in ("fmt", "raw_pca", "strong_raw"):
            per_dataset[dataset][source] = {
                metric: float(np.mean([
                    per_seed[seed][source][metric] for seed in seeds
                ])) for metric in metrics
            }
        per_dataset[dataset]["gains"] = {
            f"{metric}_vs_raw_pca": (
                per_dataset[dataset]["fmt"][metric]
                - per_dataset[dataset]["raw_pca"][metric]
            ) for metric in metrics
        }
        per_dataset[dataset]["gains"].update({
            f"{metric}_vs_strong_raw": (
                per_dataset[dataset]["fmt"][metric]
                - per_dataset[dataset]["strong_raw"][metric]
            ) for metric in metrics
        })
    macro = {}
    for source in ("fmt", "raw_pca", "strong_raw"):
        macro[source] = {
            metric: float(np.mean([
                value[source][metric] for value in per_dataset.values()
            ])) for metric in metrics
        }
    gains = {
        f"{metric}_vs_raw_pca": macro["fmt"][metric] - macro["raw_pca"][metric]
        for metric in metrics
    }
    gains.update({
        f"{metric}_vs_strong_raw": macro["fmt"][metric] - macro["strong_raw"][metric]
        for metric in metrics
    })
    tolerance = float(
        spec.get("selection", {}).get("allowed_fmt_below_strong_raw", 0.005)
    )
    guard = min(
        gains["f1_vs_strong_raw"], gains["average_precision_vs_strong_raw"]
    ) >= -tolerance
    seed_gains = {
        str(seed): float(np.mean(values)) for seed, values in seed_f1_gains.items()
    }
    return {
        "group": group_name,
        "candidate_id": candidate["id"],
        "fmt_feature": candidate["fmt_feature"],
        "dataset_count": len(datasets),
        "fmt_f1_macro": macro["fmt"]["f1"],
        "raw_pca_f1_macro": macro["raw_pca"]["f1"],
        "strong_raw_f1_macro": macro["strong_raw"]["f1"],
        "fmt_minus_raw_pca_f1_macro": gains["f1_vs_raw_pca"],
        "fmt_minus_strong_raw_f1_macro": gains["f1_vs_strong_raw"],
        "fmt_ap_macro": macro["fmt"]["average_precision"],
        "raw_pca_ap_macro": macro["raw_pca"]["average_precision"],
        "strong_raw_ap_macro": macro["strong_raw"]["average_precision"],
        "fmt_minus_raw_pca_ap_macro": gains["average_precision_vs_raw_pca"],
        "fmt_minus_strong_raw_ap_macro": gains["average_precision_vs_strong_raw"],
        "worst_seed_f1_gain": min(seed_gains.values()),
        "all_seed_f1_gains_positive": min(seed_gains.values()) > 0.0,
        "strong_raw_guard_passed": guard,
        "seed_gains_json": json.dumps(seed_gains, sort_keys=True),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
    }


def _selection_key(row: dict) -> tuple[float, float, float, float]:
    """Rank by the registered Task3 FMT versus Raw-PCA residual comparison.

    Strong Raw and Raw-wide remain important reported baselines, but they are
    different model routes and do not replace the same-structure Raw-PCA arm
    used to isolate the contribution of FMT.
    """
    return (
        float(row["fmt_minus_raw_pca_f1_macro"]),
        float(row["fmt_minus_raw_pca_ap_macro"]),
        float(row["worst_seed_f1_gain"]),
        float(row["fmt_f1_macro"]),
    )


def select(config_path: str) -> Path:
    spec = _load_spec(config_path)
    top_k = int(spec.get("selection", {}).get("top_k", 3))
    leaderboard = []
    selected = {}
    for group_name in spec["groups"]:
        rows = [
            _candidate_summary(spec, group_name, candidate)
            for candidate in spec["candidates"]
        ]
        ranked = sorted(
            rows,
            key=_selection_key,
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["rank_within_group"] = rank
            leaderboard.append(row)
        selected[group_name] = ranked[:top_k]
    output = Path(spec["output_root"])
    _write_csv(output / "stage1_leaderboard.csv", leaderboard)
    primary = {group: rows[0] for group, rows in selected.items()}
    dataset_rows = []
    for group, row in primary.items():
        details = json.loads(row["datasets_json"])
        for dataset, metrics in details.items():
            dataset_rows.append({"group": group, "dataset": dataset, **metrics})
    f1_gain = float(np.mean([
        row["gains"]["f1_vs_raw_pca"] for row in dataset_rows
    ]))
    ap_gain = float(np.mean([
        row["gains"]["average_precision_vs_raw_pca"] for row in dataset_rows
    ]))
    target_gain = float(
        spec.get("selection", {}).get("target_dataset_macro_f1_gain", 0.15)
    )
    payload = {
        "experiment": spec["experiment"],
        "selection_rule": (
            "family-specific: maximize validation F1 gain over the same-width "
            "Raw-PCA residual; tie-break by AP gain, worst seed, and absolute "
            "FMT F1. Strong Raw is reported but does not select the recipe"
        ),
        "opened_ordinals": sorted(
            set(spec["screen_split"]["train_ordinals"])
            | set(spec["screen_split"]["validation_ordinals"])
        ),
        "outer_ordinals_opened": False,
        "confirmation_opened": False,
        "top_k_by_group": selected,
        "primary_by_group": primary,
        "development_dataset_macro_f1_gain_vs_raw_pca": f1_gain,
        "development_dataset_macro_ap_gain_vs_raw_pca": ap_gain,
        "development_target_gain": target_gain,
        "development_target_reached": f1_gain >= target_gain,
        "dataset_details": dataset_rows,
    }
    target = output / "stage1_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("candidate", "select"), required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--candidate-index", type=int)
    args = parser.parse_args()
    if args.mode == "select":
        select(args.config)
    elif args.job_index is not None:
        run_job(args.config, args.job_index)
    elif args.dataset is not None and args.candidate_index is not None:
        run_candidate(args.config, args.dataset, args.candidate_index)
    else:
        parser.error("candidate mode requires --job-index or dataset/candidate-index")


if __name__ == "__main__":
    main()
