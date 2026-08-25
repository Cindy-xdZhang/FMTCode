"""Family-tuned Raw+VAE versus FMT+VAE on fresh 3D flow timeslices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FMT_Utils.RawPathline_3D import (
    normalize_raw_train_eval, raw_pathline_representation,
)
from FMT_Utils.Task12Data_3D import (
    load_cache_records, stack_features, stack_reference,
)
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics, calibrate_vortex_cluster,
)
from Verify_HighReVAE import _train


def _write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cache_dir(spec, split, dataset):
    root = spec["cache_roots"][split]
    override = spec.get("cache_overrides", {}).get(dataset, {}).get(split)
    return Path(override if override is not None else root) / dataset


def _raw_values(records):
    return np.concatenate([
        raw_pathline_representation(record["raw"], "center_relative")
        for record in records
    ])


def _prepare_inputs(train_records, evaluate_records, method, fmt_feature, device):
    if method == "raw":
        train = _raw_values(train_records)
        evaluate = _raw_values(evaluate_records)
        sampled_steps = train_records[0]["raw"].shape[1] // (7 * 3)
        return normalize_raw_train_eval(
            train, evaluate, "center_relative", sampled_steps, "pre_group_rms"
        )
    train = stack_features(train_records, fmt_feature, device)
    evaluate = stack_features(evaluate_records, fmt_feature, device)
    scaler = StandardScaler().fit(train)
    return (scaler.transform(train).astype(np.float32),
            scaler.transform(evaluate).astype(np.float32))


def _latent_score(train_mu, evaluate_mu, reference, kmeans_seed, n_init):
    model = KMeans(n_clusters=2, random_state=int(kmeans_seed),
                   n_init=int(n_init)).fit(train_mu)
    labels = model.predict(evaluate_mu)
    vortex_cluster = calibrate_vortex_cluster(reference, labels)
    return binary_cluster_metrics(reference, labels, vortex_cluster)


def _selection(spec, group, records, source, device, output, resume):
    path = output / "selection.csv"
    rows = _read_csv(path) if resume else []
    completed = {(row["dataset"], row["method"], row["architecture"],
                  int(row["training_seed"])) for row in rows}
    train_ids = spec["splits"]["selection_train"]
    validation_ids = spec["splits"]["selection_validation"]
    fmt_feature = spec["groups"][group]["fmt_feature"]
    for dataset in spec["groups"][group]["datasets"]:
        train_records = [records[dataset][index] for index in train_ids]
        validation_records = [records[dataset][index] for index in validation_ids]
        reference = stack_reference(validation_records)
        for method in ("raw", "fmt"):
            train_x, validation_x = _prepare_inputs(
                train_records, validation_records, method, fmt_feature, device
            )
            for architecture in spec["architectures"]:
                for seed in spec["selection_seeds"]:
                    key = (dataset, method, architecture["id"], int(seed))
                    if key in completed:
                        continue
                    train_mu, validation_mu, losses = _train(
                        train_x, validation_x, architecture, source, int(seed), device
                    )
                    score = _latent_score(
                        train_mu, validation_mu, reference,
                        spec["kmeans_seed"], spec["kmeans_n_init"],
                    )
                    row = {"dataset": dataset, "group": group, "method": method,
                           "fmt_feature": fmt_feature, "architecture": architecture["id"],
                           "training_seed": int(seed), **score, **losses}
                    rows.append(row); _write_csv(path, rows)
                    completed.add(key)
                    print(f"selection {group}/{dataset}/{method}/{architecture['id']}: "
                          f"F1={score['f1']:.4f}", flush=True)
    return rows


def _choose_architectures(spec, group, rows):
    members = spec["groups"][group]["datasets"]
    selected = {}
    for method in ("raw", "fmt"):
        scored = []
        for architecture in spec["architectures"]:
            values = [float(row["f1"]) for row in rows
                      if row["method"] == method
                      and row["architecture"] == architecture["id"]
                      and row["dataset"] in members]
            expected = len(members) * len(spec["selection_seeds"])
            if len(values) != expected:
                raise RuntimeError(
                    f"incomplete selection for {group}/{method}/{architecture['id']}: "
                    f"{len(values)} != {expected}"
                )
            scored.append((float(np.mean(values)), float(np.min(values)), architecture["id"]))
        mean_f1, min_f1, identifier = max(scored, key=lambda item: (item[0], item[1], item[2]))
        selected[method] = {"architecture": identifier,
                            "validation_mean_f1": mean_f1,
                            "validation_min_f1": min_f1}
    return selected


def _architecture(spec, identifier):
    matches = [value for value in spec["architectures"] if value["id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"architecture {identifier!r} matched {len(matches)} entries")
    return matches[0]


def _final(spec, group, development, confirmation, selected, source, device, output, resume):
    path = output / "final_runs.csv"
    rows = _read_csv(path) if resume else []
    completed = {(row["dataset"], row["method"], int(row["training_seed"])) for row in rows}
    train_ids = spec["splits"]["final_train"]
    calibration_ids = spec["splits"]["cluster_calibration"]
    fmt_feature = spec["groups"][group]["fmt_feature"]
    for dataset in spec["groups"][group]["datasets"]:
        train_records = [development[dataset][index] for index in train_ids]
        calibration_records = [development[dataset][index] for index in calibration_ids]
        test_records = confirmation[dataset]
        calibration_reference = stack_reference(calibration_records)
        test_reference = stack_reference(test_records)
        evaluate_records = [*calibration_records, *test_records]
        calibration_count = sum(len(record["reference"]) for record in calibration_records)
        for method in ("raw", "fmt"):
            architecture = _architecture(spec, selected[method]["architecture"])
            train_x, evaluate_x = _prepare_inputs(
                train_records, evaluate_records, method, fmt_feature, device
            )
            for seed in spec["final_training_seeds"]:
                key = (dataset, method, int(seed))
                if key in completed:
                    continue
                train_mu, evaluate_mu, losses = _train(
                    train_x, evaluate_x, architecture, source, int(seed), device
                )
                model = KMeans(n_clusters=2, random_state=int(spec["kmeans_seed"]),
                               n_init=int(spec["kmeans_n_init"])).fit(train_mu)
                calibration_labels = model.predict(evaluate_mu[:calibration_count])
                vortex_cluster = calibrate_vortex_cluster(
                    calibration_reference, calibration_labels
                )
                test_labels = model.predict(evaluate_mu[calibration_count:])
                score = binary_cluster_metrics(test_reference, test_labels, vortex_cluster)
                row = {"dataset": dataset, "group": group,
                       "method": "Raw+VAE" if method == "raw" else "FMT+VAE",
                       "fmt_feature": fmt_feature, "architecture": architecture["id"],
                       "training_seed": int(seed), "cluster_as_vortex": vortex_cluster,
                       **score, **losses}
                rows.append(row); _write_csv(path, rows); completed.add(key)
                print(f"final {group}/{dataset}/{row['method']}/seed={seed}: "
                      f"F1={score['f1']:.4f}", flush=True)
    return rows


def run_group(config_path, group, resume=False):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if group not in spec["groups"]:
        raise ValueError(f"unknown group {group!r}; choose from {sorted(spec['groups'])}")
    output = Path(spec["output_dir"]) / "groups" / group
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    datasets = spec["groups"][group]["datasets"]
    development = {dataset: load_cache_records(
        _cache_dir(spec, "development", dataset), 10
    ) for dataset in datasets}
    confirmation_count = int(spec.get("splits", {}).get("confirmation_count", 4))
    confirmation = {dataset: load_cache_records(
        _cache_dir(spec, "confirmation", dataset), confirmation_count
    ) for dataset in datasets}
    source = EasyConfig(str(spec["source_config"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_architecture = spec["groups"][group].get("fixed_architecture")
    same_vae_from_fmt = bool(
        spec["groups"][group].get("same_vae_from_fmt_validation", False)
    )
    if fixed_architecture is None:
        selection_rows = _selection(
            spec, group, development, source, device, output, resume
        )
        selected = _choose_architectures(spec, group, selection_rows)
        if same_vae_from_fmt:
            fmt_selected = dict(selected["fmt"])
            selected = {
                "raw": {
                    "architecture": fmt_selected["architecture"],
                    "selection_rule": "same_vae_from_fmt_validation",
                },
                "fmt": {
                    **fmt_selected,
                    "selection_rule": "same_vae_from_fmt_validation",
                },
            }
    else:
        # Controlled Task2: the VAE is fixed once per physical family and is
        # identical in the Raw and FMT arms. Only the input representation changes.
        _architecture(spec, fixed_architecture)
        selected = {
            "raw": {"architecture": fixed_architecture,
                    "selection_rule": "same_family_vae"},
            "fmt": {"architecture": fixed_architecture,
                    "selection_rule": "same_family_vae"},
        }
    (output / "selected_architectures.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    _final(spec, group, development, confirmation, selected,
           source, device, output, resume)
    return output


def summarize(config_path):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(spec["output_dir"]); all_rows = []; selected = {}
    for group in spec["groups"]:
        group_dir = output / "groups" / group
        rows = _read_csv(group_dir / "final_runs.csv")
        expected = len(spec["groups"][group]["datasets"]) * 2 * len(spec["final_training_seeds"])
        if len(rows) != expected:
            raise RuntimeError(f"{group}: expected {expected} final rows, found {len(rows)}")
        all_rows.extend(rows)
        selected[group] = json.loads(
            (group_dir / "selected_architectures.json").read_text(encoding="utf-8")
        )
    _write_csv(output / "final_runs.csv", all_rows)
    table = []
    for group, group_spec in spec["groups"].items():
        for dataset in group_spec["datasets"]:
            item = {"dataset": dataset, "group": group,
                    "fmt_feature": group_spec["fmt_feature"]}
            method_values = {}
            for method in ("Raw+VAE", "FMT+VAE"):
                values = [row for row in all_rows
                          if row["dataset"] == dataset and row["method"] == method]
                method_values[method] = values
                prefix = "raw" if method == "Raw+VAE" else "fmt"
                for metric in ("f1", "iou", "ari", "nmi"):
                    array = np.asarray([float(row[metric]) for row in values])
                    item[f"{prefix}_{metric}_mean"] = float(array.mean())
                    item[f"{prefix}_{metric}_std"] = float(array.std())
                item[f"{prefix}_architecture"] = values[0]["architecture"]
            raw_by_seed = {int(row["training_seed"]): float(row["f1"])
                           for row in method_values["Raw+VAE"]}
            fmt_by_seed = {int(row["training_seed"]): float(row["f1"])
                           for row in method_values["FMT+VAE"]}
            gains = np.asarray([fmt_by_seed[seed] - raw_by_seed[seed]
                                for seed in sorted(raw_by_seed)])
            item["paired_f1_gain_mean"] = float(gains.mean())
            item["paired_f1_gain_std"] = float(gains.std())
            item["paired_f1_gain_min"] = float(gains.min())
            table.append(item)
    _write_csv(output / "paper_table.csv", table)
    same_vae = all(
        "fixed_architecture" in group_spec
        or bool(group_spec.get("same_vae_from_fmt_validation", False))
                   for group_spec in spec["groups"].values())
    comparison_note = (
        "Within each physical-family group, Raw+VAE and FMT+VAE use the same frozen "
        "VAE hidden/latent architecture and training hyperparameters; only the input "
        "representation and its required input/output layer width change."
        if same_vae else
        "Raw+VAE and FMT+VAE architectures are selected independently on development "
        "validation slices."
    )
    lines = [
        f"# {spec['experiment']} — fresh-timeslice confirmation", "",
        comparison_note,
        "Cluster identity is frozen using separate calibration slices.", "",
        "| Flow | Config group | Raw+VAE F1 | FMT+VAE F1 | Paired F1 gain | FMT ARI | FMT NMI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['dataset']} | {row['group']} | "
            f"{row['raw_f1_mean']:.4f} ± {row['raw_f1_std']:.4f} | "
            f"{row['fmt_f1_mean']:.4f} ± {row['fmt_f1_std']:.4f} | "
            f"{row['paired_f1_gain_mean']:+.4f} ± {row['paired_f1_gain_std']:.4f} | "
            f"{row['fmt_ari_mean']:.4f} | {row['fmt_nmi_mean']:.4f} |"
        )
    (output / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    aggregate = {
        "dataset_count": len(table),
        "raw_f1_mean": float(np.mean([row["raw_f1_mean"] for row in table])),
        "fmt_f1_mean": float(np.mean([row["fmt_f1_mean"] for row in table])),
    }
    aggregate["fmt_minus_raw"] = (
        aggregate["fmt_f1_mean"] - aggregate["raw_f1_mean"]
    )
    (output / "summary.json").write_text(json.dumps({
        "experiment": spec["experiment"], "same_vae_control": same_vae,
        "aggregate": aggregate,
        "selected_architectures": selected,
        "paper_table": table, "config": spec,
    }, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2)); return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mainExp_Task2_3D_2.1.yaml")
    parser.add_argument("--group")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.config)
    elif args.group:
        run_group(args.config, args.group, args.resume)
    else:
        raise SystemExit("provide --group NAME or --summarize")
