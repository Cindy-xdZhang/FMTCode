"""Development-only VAE sweep for the 3D Task2 representation comparison.

Each physical family uses the same VAE hyperparameters in the Raw and FMT arms.
Confirmation records are deliberately inaccessible to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import binary_cluster_metrics, calibrate_vortex_cluster
from Run_Task2_3D_Main import _prepare_inputs
from Verify_HighReVAE import _train


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_spec(config_path: str | Path) -> dict:
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    required = {"experiment", "output_dir", "groups", "splits", "variants"}
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    variant_ids = [variant["id"] for variant in spec["variants"]]
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("VAE variant ids must be unique")
    if len(spec["task1_development_f1"]) != sum(
        len(group["datasets"]) for group in spec["groups"].values()
    ):
        raise ValueError("Task1 development baselines must cover every dataset exactly once")
    return spec


def _variant(spec: dict, index: int) -> dict:
    if not 0 <= index < len(spec["variants"]):
        raise IndexError(f"variant index {index} outside [0, {len(spec['variants'])})")
    return spec["variants"][index]


def _score_latent(train_mu, validation_mu, reference, spec):
    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=2,
        random_state=int(spec["kmeans_seed"]),
        n_init=int(spec["kmeans_n_init"]),
    ).fit(train_mu)
    labels = model.predict(validation_mu)
    vortex_cluster = calibrate_vortex_cluster(reference, labels)
    return binary_cluster_metrics(reference, labels, vortex_cluster)


def run_variant(config_path: str, group_name: str, variant_index: int, resume: bool) -> Path:
    spec = _load_spec(config_path)
    if group_name not in spec["groups"]:
        raise ValueError(f"unknown group {group_name!r}")
    group = spec["groups"][group_name]
    variant = _variant(spec, variant_index)
    output = Path(spec["output_dir"]) / "shards" / group_name
    result_path = output / f"{variant_index:02d}_{variant['id']}.csv"
    rows = _read_csv(result_path) if resume else []
    completed = {
        (row["dataset"], row["method"], int(row["training_seed"])) for row in rows
    }
    records = {
        dataset: load_cache_records(Path(group["development_cache"]) / dataset, 10)
        for dataset in group["datasets"]
    }
    source = EasyConfig(group["source_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids = spec["splits"]["selection_train"]
    validation_ids = spec["splits"]["selection_validation"]
    print(
        f"device={device} group={group_name} variant={variant_index}:{variant['id']}",
        flush=True,
    )
    for dataset in group["datasets"]:
        train_records = [records[dataset][index] for index in train_ids]
        validation_records = [records[dataset][index] for index in validation_ids]
        reference = stack_reference(validation_records)
        prepared = {}
        for method in ("raw", "fmt"):
            prepared[method] = _prepare_inputs(
                train_records,
                validation_records,
                method,
                group["fmt_feature"],
                device,
            )
        if variant.get("pca_init", False):
            minimum_dim = min(values[0].shape[1] for values in prepared.values())
            if int(variant["latent_dim"]) > minimum_dim:
                print(
                    f"skip incompatible PCA variant: latent={variant['latent_dim']} "
                    f"> minimum input dimension={minimum_dim}",
                    flush=True,
                )
                continue
        for method in ("raw", "fmt"):
            train_x, validation_x = prepared[method]
            for seed in spec["selection_seeds"]:
                key = (dataset, method, int(seed))
                if key in completed:
                    continue
                train_mu, validation_mu, losses = _train(
                    train_x, validation_x, variant, source, int(seed), device
                )
                metrics = _score_latent(
                    train_mu, validation_mu, reference, spec
                )
                row = {
                    "experiment": spec["experiment"],
                    "group": group_name,
                    "dataset": dataset,
                    "method": method,
                    "fmt_feature": group["fmt_feature"],
                    "variant_index": int(variant_index),
                    "variant": variant["id"],
                    "training_seed": int(seed),
                    "task1_development_f1": float(
                        spec["task1_development_f1"][dataset]
                    ),
                    **metrics,
                    **losses,
                }
                rows.append(row)
                _write_csv(result_path, rows)
                completed.add(key)
                print(
                    f"{dataset}/{method}/seed={seed}: F1={metrics['f1']:.4f}",
                    flush=True,
                )
    return result_path


def _candidate_rows(spec: dict, rows: list[dict]) -> list[dict]:
    candidates = []
    for group_name, group in spec["groups"].items():
        for variant_index, variant in enumerate(spec["variants"]):
            subset = [
                row for row in rows
                if row["group"] == group_name and row["variant"] == variant["id"]
            ]
            expected = len(group["datasets"]) * 2 * len(spec["selection_seeds"])
            if len(subset) != expected:
                continue
            method_f1 = {
                method: np.asarray([
                    float(row["f1"]) for row in subset if row["method"] == method
                ])
                for method in ("raw", "fmt")
            }
            task1_values = np.asarray([
                float(spec["task1_development_f1"][dataset])
                for dataset in group["datasets"]
            ])
            raw_mean = float(method_f1["raw"].mean())
            fmt_mean = float(method_f1["fmt"].mean())
            task1_mean = float(task1_values.mean())
            candidates.append({
                "group": group_name,
                "variant_index": variant_index,
                "variant": variant["id"],
                "dataset_count": len(group["datasets"]),
                "raw_f1_mean": raw_mean,
                "fmt_f1_mean": fmt_mean,
                "task1_f1_mean": task1_mean,
                "fmt_minus_raw": fmt_mean - raw_mean,
                "raw_minus_task1": raw_mean - task1_mean,
                "minimum_hierarchy_margin": min(
                    fmt_mean - raw_mean, raw_mean - task1_mean
                ),
            })
    return candidates


def _select_global(spec: dict, candidates: list[dict]) -> dict:
    by_group = {
        group: [row for row in candidates if row["group"] == group]
        for group in spec["groups"]
    }
    missing = [group for group, values in by_group.items() if not values]
    if missing:
        raise RuntimeError(f"no complete VAE candidates for groups: {missing}")

    # A state is dominated when another state has both higher Raw and FMT sums.
    # Pareto pruning makes the exact family-wise combination search tractable.
    states = [(0.0, 0.0, {})]
    for group, values in by_group.items():
        expanded = []
        for raw_sum, fmt_sum, selected in states:
            for value in values:
                weight = int(value["dataset_count"])
                expanded.append((
                    raw_sum + weight * float(value["raw_f1_mean"]),
                    fmt_sum + weight * float(value["fmt_f1_mean"]),
                    {**selected, group: value},
                ))
        expanded.sort(key=lambda state: (-state[0], -state[1]))
        frontier = []
        best_fmt = -np.inf
        for state in expanded:
            if state[1] > best_fmt + 1e-12:
                frontier.append(state)
                best_fmt = state[1]
        states = frontier

    dataset_count = sum(len(group["datasets"]) for group in spec["groups"].values())
    task1_mean = float(np.mean(list(spec["task1_development_f1"].values())))
    ranked = []
    for raw_sum, fmt_sum, selected in states:
        raw_mean = raw_sum / dataset_count
        fmt_mean = fmt_sum / dataset_count
        margin_raw_task1 = raw_mean - task1_mean
        margin_fmt_raw = fmt_mean - raw_mean
        ranked.append({
            "raw_f1_mean": raw_mean,
            "fmt_f1_mean": fmt_mean,
            "task1_f1_mean": task1_mean,
            "raw_minus_task1": margin_raw_task1,
            "fmt_minus_raw": margin_fmt_raw,
            "minimum_hierarchy_margin": min(margin_raw_task1, margin_fmt_raw),
            "hierarchy_satisfied": margin_raw_task1 > 0 and margin_fmt_raw > 0,
            "selected": selected,
        })
    return max(
        ranked,
        key=lambda item: (
            item["hierarchy_satisfied"],
            item["minimum_hierarchy_margin"],
            item["fmt_f1_mean"],
        ),
    )


def summarize(config_path: str) -> Path:
    spec = _load_spec(config_path)
    output = Path(spec["output_dir"])
    rows = []
    for path in sorted((output / "shards").glob("*/*.csv")):
        rows.extend(_read_csv(path))
    candidates = _candidate_rows(spec, rows)
    expected_candidates = len(spec["groups"]) * len(spec["variants"])
    if len(candidates) != expected_candidates:
        raise RuntimeError(
            f"incomplete sweep: {len(candidates)} / {expected_candidates} "
            "group-variant candidates"
        )
    _write_csv(output / "development_candidates.csv", candidates)
    selected = _select_global(spec, candidates)
    (output / "development_selection.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    print(json.dumps(selected, indent=2))
    return output


def describe(config_path: str) -> None:
    spec = _load_spec(config_path)
    print(json.dumps({
        "experiment": spec["experiment"],
        "groups": list(spec["groups"]),
        "dataset_count": sum(len(g["datasets"]) for g in spec["groups"].values()),
        "variant_count": len(spec["variants"]),
        "selection_seed_count": len(spec["selection_seeds"]),
        "training_runs": sum(len(g["datasets"]) for g in spec["groups"].values())
        * len(spec["variants"]) * len(spec["selection_seeds"]) * 2,
        "uses_confirmation": False,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--group")
    parser.add_argument("--variant-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        describe(args.config)
    elif args.summarize:
        summarize(args.config)
    elif args.group is not None and args.variant_index is not None:
        run_variant(args.config, args.group, args.variant_index, args.resume)
    else:
        raise SystemExit("provide --group and --variant-index, --summarize, or --describe")
