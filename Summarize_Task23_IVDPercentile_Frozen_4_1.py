"""Build the paper sensitivity tables for the frozen Task2/Task3-4.1 recipes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from Build_Task23_IVDPercentile_Labels import percentile_tag


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _task3_metrics(root: Path, search: dict) -> tuple[dict, list[dict]]:
    result = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    datasets = result["datasets"]
    detailed = []
    family_f1 = []
    family_ap = []
    for family, group in search["groups"].items():
        family_f1.append(float(np.mean([
            datasets[dataset]["gains"]["f1_vs_raw_pca"]
            for dataset in group["datasets"]
        ])))
        family_ap.append(float(np.mean([
            datasets[dataset]["gains"]["average_precision_vs_raw_pca"]
            for dataset in group["datasets"]
        ])))
    for dataset, values in datasets.items():
        family = next(
            name for name, group in search["groups"].items()
            if dataset in group["datasets"]
        )
        detailed.append({
            "dataset": dataset, "family": family,
            "raw_f1": values["raw"]["f1"],
            "raw_wide_f1": values["raw_wide"]["f1"],
            "raw_pca_f1": values["raw_pca_residual"]["f1"],
            "fmt_f1": values["fmt_residual"]["f1"],
            "fmt_minus_raw_pca_f1": values["gains"]["f1_vs_raw_pca"],
            "raw_pca_ap": values["raw_pca_residual"]["average_precision"],
            "fmt_ap": values["fmt_residual"]["average_precision"],
            "fmt_minus_raw_pca_ap": values["gains"]["average_precision_vs_raw_pca"],
        })
    metrics = {
        "task3_dataset_macro_raw_pca_f1": float(np.mean([row["raw_pca_f1"] for row in detailed])),
        "task3_dataset_macro_fmt_f1": float(np.mean([row["fmt_f1"] for row in detailed])),
        "task3_dataset_macro_f1_gain": float(np.mean([row["fmt_minus_raw_pca_f1"] for row in detailed])),
        "task3_family_macro_f1_gain": float(np.mean(family_f1)),
        "task3_positive_datasets_f1": int(sum(row["fmt_minus_raw_pca_f1"] > 0 for row in detailed)),
        "task3_positive_families_f1": int(sum(value > 0 for value in family_f1)),
        "task3_dataset_macro_raw_pca_ap": float(np.mean([row["raw_pca_ap"] for row in detailed])),
        "task3_dataset_macro_fmt_ap": float(np.mean([row["fmt_ap"] for row in detailed])),
        "task3_dataset_macro_ap_gain": float(np.mean([row["fmt_minus_raw_pca_ap"] for row in detailed])),
        "task3_family_macro_ap_gain": float(np.mean(family_ap)),
        "task3_positive_datasets_ap": int(sum(row["fmt_minus_raw_pca_ap"] > 0 for row in detailed)),
        "task3_positive_families_ap": int(sum(value > 0 for value in family_ap)),
    }
    return metrics, detailed


def summarize(config_path: str) -> Path:
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(spec["output_dir"])
    search = yaml.safe_load(
        Path(spec["task3"]["search_config"]).read_text(encoding="utf-8")
    )
    task2_rows = _read_csv(
        Path(spec["task2"]["output_dir"]) / "percentile_summary.csv"
    )
    task2_by_p = {float(row["ivd_percentile"]): row for row in task2_rows}
    requested = [float(value) for value in spec["requested_percentiles"]]
    all_values = requested + [float(spec["audit_percentile"])]
    rows = []
    details = []
    task2_details = _read_csv(
        Path(spec["task2"]["output_dir"]) / "paper_table.csv"
    )
    for percentile in all_values:
        tag = percentile_tag(percentile)
        if percentile == float(spec["audit_percentile"]):
            task3_root = Path(spec["task3"]["published_p95_root"])
        else:
            task3_root = Path(spec["task3"]["output_root"]) / tag / "final_confirmation"
        task3, task3_detail = _task3_metrics(task3_root, search)
        task2 = task2_by_p[percentile]
        row = {
            "ivd_percentile": percentile, "percentile_tag": tag,
            "mean_reference_positive_fraction": float(task2["mean_reference_positive_fraction"]),
            "task2_dataset_macro_raw_f1": float(task2["dataset_macro_raw_f1"]),
            "task2_dataset_macro_fmt_f1": float(task2["dataset_macro_fmt_f1"]),
            "task2_dataset_macro_f1_gain": float(task2["dataset_macro_f1_gain"]),
            "task2_family_macro_f1_gain": float(task2["family_macro_f1_gain"]),
            "task2_positive_datasets": int(task2["positive_dataset_count"]),
            "task2_positive_families": int(task2["positive_family_count"]),
            **task3,
        }
        rows.append(row)
        task2_by_dataset = {
            item["dataset"]: item for item in task2_details
            if float(item["ivd_percentile"]) == percentile
        }
        for item in task3_detail:
            t2 = task2_by_dataset[item["dataset"]]
            details.append({
                "ivd_percentile": percentile, "percentile_tag": tag,
                "dataset": item["dataset"], "family": item["family"],
                "task2_raw_f1": float(t2["raw_f1_mean"]),
                "task2_fmt_f1": float(t2["fmt_f1_mean"]),
                "task2_f1_gain": float(t2["paired_f1_gain_mean"]),
                **{key: value for key, value in item.items()
                   if key not in {"dataset", "family"}},
            })
    _write_csv(output / "task23_ivd_percentile_4p1_table.csv", rows)
    _write_csv(output / "task23_ivd_percentile_4p1_by_dataset.csv", details)

    requested_rows = [row for row in rows if row["ivd_percentile"] in requested]
    maxima = {
        key: max(requested_rows, key=lambda row: row[key])
        for key in (
            "task2_dataset_macro_f1_gain", "task2_family_macro_f1_gain",
            "task3_dataset_macro_f1_gain", "task3_family_macro_f1_gain",
            "task3_dataset_macro_ap_gain", "task3_family_macro_ap_gain",
        )
    }
    lines = [
        "# Task2/Task3 4.1 whole-field IVD percentile sensitivity",
        "",
        "p80--p92.5 是完整预定扫描；p95 是当前论文主表参考。Task2 比较同一 VAE 的 Raw/FMT 输入，Task3 比较同结构同宽度的 Raw-PCA/FMT residual。",
        "",
        "| label | seed-positive | Task2 Raw F1 | Task2 FMT F1 | T2 gain | T2 positive | Task3 Raw-PCA F1 | Task3 FMT F1 | T3 F1 gain | T3 F1 positive | Raw-PCA AP | FMT AP | T3 AP gain | T3 AP positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['percentile_tag']} | {row['mean_reference_positive_fraction']:.3f} | "
            f"{row['task2_dataset_macro_raw_f1']:.4f} | {row['task2_dataset_macro_fmt_f1']:.4f} | "
            f"{row['task2_dataset_macro_f1_gain']:+.4f} | {row['task2_positive_datasets']}/10 | "
            f"{row['task3_dataset_macro_raw_pca_f1']:.4f} | {row['task3_dataset_macro_fmt_f1']:.4f} | "
            f"{row['task3_dataset_macro_f1_gain']:+.4f} | {row['task3_positive_datasets_f1']}/10 | "
            f"{row['task3_dataset_macro_raw_pca_ap']:.4f} | {row['task3_dataset_macro_fmt_ap']:.4f} | "
            f"{row['task3_dataset_macro_ap_gain']:+.4f} | {row['task3_positive_datasets_ap']}/10 |"
        )
    descriptions = {
        "task2_dataset_macro_f1_gain": "Task2 dataset-macro F1",
        "task2_family_macro_f1_gain": "Task2 family-macro F1",
        "task3_dataset_macro_f1_gain": "Task3 dataset-macro F1",
        "task3_family_macro_f1_gain": "Task3 family-macro F1",
        "task3_dataset_macro_ap_gain": "Task3 dataset-macro Average Precision",
        "task3_family_macro_ap_gain": "Task3 family-macro Average Precision",
    }
    lines.extend(["", "请求范围内最大增益：", ""])
    for key, description in descriptions.items():
        best = maxima[key]
        lines.append(f"- {description}: {best['percentile_tag']}，{best[key]:+.4f}。")
    lines.extend([
        "",
        "最大 FMT 增益只说明当前模型对该标签定义最有利；它不能单独证明该百分位是物理上最正确的涡区边界。",
    ])
    markdown = output / "task23_ivd_percentile_4p1_table.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "experiment": spec["experiment"], "rows": rows,
        "maxima_over_requested_percentiles": maxima,
        "task2_training_reused_across_percentiles": True,
        "task3_retrained_for_every_requested_percentile": True,
    }
    (output / "task23_ivd_percentile_4p1_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(markdown.read_text(encoding="utf-8"), flush=True)
    return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Ablation_Task23IVDPercentile_1.2.yaml"
    )
    args = parser.parse_args()
    summarize(args.config)
