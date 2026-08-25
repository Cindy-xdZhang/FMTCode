"""Select family-level FMT+KMeans settings and evaluate on fresh 3D timeslices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.Task12Data_3D import (
    load_cache_records, stack_features, stack_reference,
)
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics, calibrate_vortex_cluster, fit_kmeans_transform,
)


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _pca_label(value):
    return "none" if value is None else str(int(value))


def _fit_score(train_records, eval_records, feature, pca_dim, seed, n_init, device):
    train = stack_features(train_records, feature, device)
    evaluate = stack_features(eval_records, feature, device)
    reference = stack_reference(eval_records)
    fitted = fit_kmeans_transform(train, pca_dim, seed, n_init)
    labels = fitted.predict(evaluate)
    vortex_cluster = calibrate_vortex_cluster(reference, labels)
    return binary_cluster_metrics(reference, labels, vortex_cluster)


def _select_family_configs(spec, development, device):
    rows = []
    train_ids = list(spec["splits"]["selection_train"])
    validation_ids = list(spec["splits"]["selection_validation"])
    candidates = ["raw", *spec["fmt_candidates"]]
    for feature in candidates:
        for pca_dim in spec["pca_dims"]:
            if feature == "raw" and pca_dim is None:
                # The raw baseline is intentionally dimension-controlled: exact
                # KMeans on all 672 coordinates dominates selection runtime and
                # is not the Task1 method under test.
                continue
            print(f"selection feature={feature}, PCA={_pca_label(pca_dim)}", flush=True)
            for dataset in spec["datasets"]:
                records = development[dataset]
                try:
                    score = _fit_score(
                        [records[i] for i in train_ids],
                        [records[i] for i in validation_ids],
                        feature, pca_dim, int(spec["selection_seed"]),
                        int(spec["selection_kmeans_n_init"]), device,
                    )
                except ValueError as error:
                    if "pca_dim" in str(error):
                        continue
                    raise
                rows.append({"dataset": dataset, "family": spec["families"][dataset],
                             "feature": feature, "pca_dim": _pca_label(pca_dim), **score})

    selected = {}
    for family in sorted(set(spec["families"].values())):
        members = [name for name in spec["datasets"] if spec["families"][name] == family]
        family_rows = [row for row in rows if row["dataset"] in members]
        keys = sorted({(row["feature"], row["pca_dim"]) for row in family_rows})
        fmt_keys = [key for key in keys if key[0] != "raw"]
        raw_keys = [key for key in keys if key[0] == "raw"]

        def choose(options):
            scored = []
            for key in options:
                values = [row for row in family_rows
                          if (row["feature"], row["pca_dim"]) == key]
                if len(values) != len(members):
                    continue
                scored.append((float(np.mean([row["f1"] for row in values])),
                               float(np.mean([row["ari"] for row in values])), key))
            return max(scored, key=lambda item: (item[0], item[1], item[2]))

        fmt_mean, fmt_ari, fmt_key = choose(fmt_keys)
        raw_mean, raw_ari, raw_key = choose(raw_keys)
        selected[family] = {
            "fmt": {"feature": fmt_key[0], "pca_dim": None if fmt_key[1] == "none" else int(fmt_key[1]),
                    "validation_mean_f1": fmt_mean, "validation_mean_ari": fmt_ari},
            "raw": {"feature": raw_key[0], "pca_dim": None if raw_key[1] == "none" else int(raw_key[1]),
                    "validation_mean_f1": raw_mean, "validation_mean_ari": raw_ari},
            "members": members,
        }
    return rows, selected


def _final_runs(spec, development, confirmation, selected, device):
    train_ids = list(spec["splits"]["final_train"])
    calibration_ids = list(spec["splits"]["cluster_calibration"])
    rows = []
    for dataset in spec["datasets"]:
        family = spec["families"][dataset]
        for method in ("raw", "fmt"):
            choice = selected[family][method]
            feature, pca_dim = choice["feature"], choice["pca_dim"]
            train_records = [development[dataset][i] for i in train_ids]
            calibration_records = [development[dataset][i] for i in calibration_ids]
            test_records = confirmation[dataset]
            train = stack_features(train_records, feature, device)
            calibration = stack_features(calibration_records, feature, device)
            calibration_reference = stack_reference(calibration_records)
            test = stack_features(test_records, feature, device)
            test_reference = stack_reference(test_records)
            for seed in spec["final_kmeans_seeds"]:
                fitted = fit_kmeans_transform(
                    train, pca_dim, int(seed), int(spec["kmeans_n_init"])
                )
                calibration_labels = fitted.predict(calibration)
                vortex_cluster = calibrate_vortex_cluster(
                    calibration_reference, calibration_labels
                )
                test_labels = fitted.predict(test)
                aggregate = binary_cluster_metrics(
                    test_reference, test_labels, vortex_cluster
                )
                rows.append({"scope": "all_confirmation", "dataset": dataset,
                             "family": family, "method": method, "feature": feature,
                             "pca_dim": _pca_label(pca_dim), "kmeans_seed": int(seed),
                             "cluster_as_vortex": vortex_cluster, **aggregate})
                cursor = 0
                for record in test_records:
                    count = len(record["reference"])
                    score = binary_cluster_metrics(
                        record["reference"], test_labels[cursor:cursor + count], vortex_cluster
                    )
                    rows.append({"scope": "timeslice", "dataset": dataset,
                                 "family": family, "method": method, "feature": feature,
                                 "pca_dim": _pca_label(pca_dim), "kmeans_seed": int(seed),
                                 "cluster_as_vortex": vortex_cluster,
                                 "source_index": record["metadata"]["source_start_index"],
                                 **score})
                    cursor += count
    return rows


def _summarize(spec, rows, selected, output):
    aggregate = [row for row in rows if row["scope"] == "all_confirmation"]
    table = []
    for dataset in spec["datasets"]:
        item = {"dataset": dataset, "family": spec["families"][dataset]}
        for method in ("raw", "fmt"):
            values = [row for row in aggregate
                      if row["dataset"] == dataset and row["method"] == method]
            for metric in ("f1", "iou", "ari", "nmi", "precision", "recall"):
                array = np.asarray([row[metric] for row in values], dtype=float)
                item[f"{method}_{metric}_mean"] = float(array.mean())
                item[f"{method}_{metric}_std"] = float(array.std())
            item[f"{method}_feature"] = values[0]["feature"]
            item[f"{method}_pca_dim"] = values[0]["pca_dim"]
        item["fmt_minus_raw_f1"] = item["fmt_f1_mean"] - item["raw_f1_mean"]
        table.append(item)
    _write_csv(output / "paper_table.csv", table)
    lines = [
        f"# {spec['experiment']} — fresh-timeslice confirmation",
        "",
        "Cluster identity is calibrated on development slices 9–10 and frozen before",
        f"the {int(spec.get('confirmation_count', 4))} confirmation timeslices are scored.", "",
        "| Flow | Family | FMT config | FMT F1 | FMT IoU | FMT ARI | FMT NMI | Raw F1 | FMT−Raw F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        config = f"{row['fmt_feature']} / PCA {row['fmt_pca_dim']}"
        lines.append(
            f"| {row['dataset']} | {row['family']} | {config} | "
            f"{row['fmt_f1_mean']:.4f} ± {row['fmt_f1_std']:.4f} | "
            f"{row['fmt_iou_mean']:.4f} | {row['fmt_ari_mean']:.4f} | "
            f"{row['fmt_nmi_mean']:.4f} | {row['raw_f1_mean']:.4f} | "
            f"{row['fmt_minus_raw_f1']:+.4f} |"
        )
    (output / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"experiment": spec["experiment"], "selected_family_configs": selected,
               "paper_table": table, "config": spec}
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return table


def run(config_path):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(spec["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    development = {name: load_cache_records(Path(spec["development_cache"]) / name, 10)
                   for name in spec["datasets"]}
    confirmation_count = int(spec.get("confirmation_count", 4))
    confirmation = {name: load_cache_records(
        Path(spec["confirmation_cache"]) / name, confirmation_count
    ) for name in spec["datasets"]}
    selection_rows, selected = _select_family_configs(spec, development, device)
    _write_csv(output / "selection.csv", selection_rows)
    (output / "selected_configs.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    rows = _final_runs(spec, development, confirmation, selected, device)
    _write_csv(output / "final_runs.csv", rows)
    table = _summarize(spec, rows, selected, output)
    print(json.dumps(table, indent=2)); return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mainExp_Task1_3D_2.1.yaml")
    args = parser.parse_args(); run(args.config)
