"""Combine Task2 and Task3 IVD-percentile sensitivity into paper tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from Build_Task23_IVDPercentile_Labels import percentile_tag


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _task3_summary(directory: Path, percentile: float) -> dict:
    datasets = _read_csv(directory / "paper_table.csv")
    families = _read_csv(directory / "family_summary.csv")
    selected_raw_f1 = []
    selected_raw_ap = []
    for row in datasets:
        baseline = row["selected_raw_variant"]
        selected_raw_f1.append(float(row[f"mean_{baseline}_f1"]))
        selected_raw_ap.append(float(row[f"mean_{baseline}_average_precision"]))
    return {
        "ivd_percentile": float(percentile),
        "percentile_tag": percentile_tag(percentile),
        "task3_dataset_macro_selected_raw_f1": float(np.mean(selected_raw_f1)),
        "task3_dataset_macro_raw_pca_f1": float(np.mean([
            float(row["mean_raw_pca_residual_f1"]) for row in datasets
        ])),
        "task3_dataset_macro_fmt_f1": float(np.mean([
            float(row["mean_raw_fmt_residual_f1"]) for row in datasets
        ])),
        "task3_dataset_macro_fmt_minus_raw_pca_f1": float(np.mean([
            float(row["mean_fmt_minus_raw_pca_f1"]) for row in datasets
        ])),
        "task3_family_macro_fmt_minus_raw_pca_f1": float(np.mean([
            float(row["mean_fmt_minus_raw_pca_f1"]) for row in families
        ])),
        "task3_positive_datasets_f1": int(sum(
            float(row["mean_fmt_minus_raw_pca_f1"]) > 0 for row in datasets
        )),
        "task3_positive_families_f1": int(sum(
            float(row["mean_fmt_minus_raw_pca_f1"]) > 0 for row in families
        )),
        "task3_dataset_macro_selected_raw_ap": float(np.mean(selected_raw_ap)),
        "task3_dataset_macro_raw_pca_ap": float(np.mean([
            float(row["mean_raw_pca_residual_average_precision"]) for row in datasets
        ])),
        "task3_dataset_macro_fmt_ap": float(np.mean([
            float(row["mean_raw_fmt_residual_average_precision"]) for row in datasets
        ])),
        "task3_dataset_macro_fmt_minus_raw_pca_ap": float(np.mean([
            float(row["mean_fmt_minus_raw_pca_average_precision"]) for row in datasets
        ])),
        "task3_family_macro_fmt_minus_raw_pca_ap": float(np.mean([
            float(row["mean_fmt_minus_raw_pca_average_precision"]) for row in families
        ])),
        "task3_positive_datasets_ap": int(sum(
            float(row["mean_fmt_minus_raw_pca_average_precision"]) > 0
            for row in datasets
        )),
        "task3_positive_families_ap": int(sum(
            float(row["mean_fmt_minus_raw_pca_average_precision"]) > 0
            for row in families
        )),
    }


def summarize(config_path: str) -> Path:
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(spec["output_dir"])
    task2_rows = _read_csv(
        Path(spec["task2"]["output_dir"]) / "percentile_summary.csv"
    )
    task2_by_p = {
        float(row["ivd_percentile"]): row for row in task2_rows
    }
    rows = []
    requested = [float(value) for value in spec["requested_percentiles"]]
    for percentile in requested:
        tag = percentile_tag(percentile)
        task3 = _task3_summary(
            Path(spec["task3"]["output_root"]) / tag / "final_confirmation",
            percentile,
        )
        task2 = task2_by_p[percentile]
        rows.append({
            "ivd_percentile": percentile,
            "percentile_tag": tag,
            "mean_reference_positive_fraction": float(
                task2["mean_reference_positive_fraction"]
            ),
            "task2_dataset_macro_raw_f1": float(task2["dataset_macro_raw_f1"]),
            "task2_dataset_macro_fmt_f1": float(task2["dataset_macro_fmt_f1"]),
            "task2_dataset_macro_f1_gain": float(task2["dataset_macro_f1_gain"]),
            "task2_family_macro_f1_gain": float(task2["family_macro_f1_gain"]),
            "task2_positive_datasets": int(task2["positive_dataset_count"]),
            "task2_positive_families": int(task2["positive_family_count"]),
            **{key: value for key, value in task3.items()
               if key not in {"ivd_percentile", "percentile_tag"}},
        })

    audit_p = float(spec["audit_percentile"])
    p95 = _task3_summary(
        Path(spec["task3"]["published_p95_evaluation"]), audit_p
    )
    if audit_p not in task2_by_p:
        raise RuntimeError("Task2 summary has no p95 audit row")
    task2_p95 = task2_by_p[audit_p]
    rows.append({
        "ivd_percentile": audit_p,
        "percentile_tag": percentile_tag(audit_p),
        "mean_reference_positive_fraction": float(
            task2_p95["mean_reference_positive_fraction"]
        ),
        "task2_dataset_macro_raw_f1": float(task2_p95["dataset_macro_raw_f1"]),
        "task2_dataset_macro_fmt_f1": float(task2_p95["dataset_macro_fmt_f1"]),
        "task2_dataset_macro_f1_gain": float(task2_p95["dataset_macro_f1_gain"]),
        "task2_family_macro_f1_gain": float(task2_p95["family_macro_f1_gain"]),
        "task2_positive_datasets": int(task2_p95["positive_dataset_count"]),
        "task2_positive_families": int(task2_p95["positive_family_count"]),
        **{key: value for key, value in p95.items()
           if key not in {"ivd_percentile", "percentile_tag"}},
    })
    _write_csv(output / "task23_ivd_percentile_paper_table.csv", rows)

    requested_rows = [row for row in rows if row["ivd_percentile"] in requested]
    maxima = {
        "task2_dataset_macro_f1_gain": max(
            requested_rows, key=lambda row: row["task2_dataset_macro_f1_gain"]
        ),
        "task2_family_macro_f1_gain": max(
            requested_rows, key=lambda row: row["task2_family_macro_f1_gain"]
        ),
        "task3_dataset_macro_fmt_minus_raw_pca_f1": max(
            requested_rows,
            key=lambda row: row["task3_dataset_macro_fmt_minus_raw_pca_f1"],
        ),
        "task3_family_macro_fmt_minus_raw_pca_f1": max(
            requested_rows,
            key=lambda row: row["task3_family_macro_fmt_minus_raw_pca_f1"],
        ),
        "task3_dataset_macro_fmt_minus_raw_pca_ap": max(
            requested_rows,
            key=lambda row: row["task3_dataset_macro_fmt_minus_raw_pca_ap"],
        ),
        "task3_family_macro_fmt_minus_raw_pca_ap": max(
            requested_rows,
            key=lambda row: row["task3_family_macro_fmt_minus_raw_pca_ap"],
        ),
    }
    lines = [
        "# Task2/Task3 whole-field IVD percentile sensitivity",
        "",
        "p95 为已发表主表参考；p80--p92.5 是本轮预先列出的完整扫描，均报告而不删选。",
        "",
        "| label | seed-positive fraction | Task2 Raw F1 | Task2 FMT F1 | Task2 FMT−Raw F1 | Task3 Raw-PCA F1 | Task3 FMT F1 | Task3 FMT−Raw-PCA F1 | Task3 Raw-PCA AP | Task3 FMT AP | Task3 FMT−Raw-PCA AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['percentile_tag']} | "
            f"{row['mean_reference_positive_fraction']:.3f} | "
            f"{row['task2_dataset_macro_raw_f1']:.4f} | "
            f"{row['task2_dataset_macro_fmt_f1']:.4f} | "
            f"{row['task2_dataset_macro_f1_gain']:+.4f} | "
            f"{row['task3_dataset_macro_raw_pca_f1']:.4f} | "
            f"{row['task3_dataset_macro_fmt_f1']:.4f} | "
            f"{row['task3_dataset_macro_fmt_minus_raw_pca_f1']:+.4f} | "
            f"{row['task3_dataset_macro_raw_pca_ap']:.4f} | "
            f"{row['task3_dataset_macro_fmt_ap']:.4f} | "
            f"{row['task3_dataset_macro_fmt_minus_raw_pca_ap']:+.4f} |"
        )
    lines.extend([
        "",
        "最大增益（只在预先指定的 p80--p92.5 内比较）：",
        "",
    ])
    descriptions = {
        "task2_dataset_macro_f1_gain": "Task2 dataset-macro F1",
        "task2_family_macro_f1_gain": "Task2 family-macro F1",
        "task3_dataset_macro_fmt_minus_raw_pca_f1": "Task3 dataset-macro F1",
        "task3_family_macro_fmt_minus_raw_pca_f1": "Task3 family-macro F1",
        "task3_dataset_macro_fmt_minus_raw_pca_ap": "Task3 dataset-macro Average Precision",
        "task3_family_macro_fmt_minus_raw_pca_ap": "Task3 family-macro Average Precision",
    }
    for key, description in descriptions.items():
        row = maxima[key]
        lines.append(
            f"- {description}: {row['percentile_tag']}，增益 {row[key]:+.4f}。"
        )
    lines.extend([
        "",
        "注意：增益最大只回答“哪个标签定义最有利于当前 FMT 对照”；它不等价于该百分位的涡核物理定义最正确。最终标签选择还应结合可视化语义和正类覆盖率。",
    ])
    (output / "task23_ivd_percentile_paper_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    payload = {
        "experiment": spec["experiment"],
        "requested_percentiles": requested,
        "audit_percentile": audit_p,
        "rows": rows,
        "maxima_over_requested_percentiles": maxima,
        "selection_warning": (
            "largest method gain is descriptive and is not by itself evidence "
            "that the percentile is the physically best vortex definition"
        ),
    }
    (output / "task23_ivd_percentile_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return output / "task23_ivd_percentile_paper_table.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Ablation_Task23IVDPercentile_1.1.yaml"
    )
    args = parser.parse_args()
    summarize(args.config)
