"""Merge the frozen Task1/Task2/Task3 3D confirmation tables.

This script performs no model selection or evaluation.  It only combines the
machine-readable tables produced by the versioned experiments listed below.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "paper_tables_task123_3d"
DOC_PATH = ROOT / "docs" / "paper_tables_task123_3d.md"

TASK1_INPUTS = (
    ROOT / "outputs/mainExp_Task1_3D_2.1_ibex_a100/paper_table.csv",
    ROOT / "outputs/mainExp_Task1_3D_2.2_newflows_ibex_a100/paper_table.csv",
)
TASK2_INPUTS = (
    ROOT / "outputs/mainExp_Task2_3D_2.3_ibex_a100/paper_table.csv",
    ROOT / "outputs/mainExp_Task2_3D_2.4_newflows_ibex_a100/paper_table.csv",
)
TASK3_INPUTS = (
    ROOT / "outputs/mainExp_Task3_3D_3.1_ibex_v100/final_confirmation/paper_table.csv",
)

FLOW_ORDER = (
    "channel", "cylinder3d", "halfcylinderRe640", "halfcylinderRe6400",
    "tangaroa", "deltaWing_resampled", "deltaWing_LBM", "f22raptor",
    "boeing747", "smokeBuoyancy",
)
FLOW_NAMES = {
    "channel": "Channel observer",
    "cylinder3d": "Half-cylinder Re160",
    "halfcylinderRe640": "Half-cylinder Re640",
    "halfcylinderRe6400": "Half-cylinder Re6400",
    "tangaroa": "Tangaroa",
    "deltaWing_resampled": "Delta-wing resampled",
    "deltaWing_LBM": "Delta-wing original LBM",
    "f22raptor": "F-22",
    "boeing747": "Boeing 747",
    "smokeBuoyancy": "Smoke buoyancy",
}
FAMILIES = {
    "channel": "channel",
    "cylinder3d": "half-cylinder",
    "halfcylinderRe640": "half-cylinder",
    "halfcylinderRe6400": "half-cylinder",
    "tangaroa": "Tangaroa",
    "deltaWing_resampled": "delta-wing",
    "deltaWing_LBM": "delta-wing",
    "f22raptor": "F-22",
    "boeing747": "Boeing 747",
    "smokeBuoyancy": "smoke buoyancy",
}


def _read_rows(paths):
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen result {path}. Run/download that experiment first."
            )
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    by_dataset = {row["dataset"]: row for row in rows}
    if len(by_dataset) != len(rows):
        raise ValueError("Duplicate dataset rows found while merging paper tables")
    missing = set(FLOW_ORDER) - set(by_dataset)
    extra = set(by_dataset) - set(FLOW_ORDER)
    if missing or extra:
        raise ValueError(f"Unexpected dataset coverage: missing={missing}, extra={extra}")
    return [by_dataset[name] for name in FLOW_ORDER]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    return sum(values) / len(values)


def _family_macro(rows, gain_key):
    grouped = {}
    for row in rows:
        grouped.setdefault(FAMILIES[row["dataset"]], []).append(float(row[gain_key]))
    values = {family: _mean(gains) for family, gains in grouped.items()}
    return _mean(list(values.values())), sum(value > 0 for value in values.values())


def _pm(mean, std):
    return f"{float(mean):.4f}±{float(std):.4f}"


def _signed(value):
    return f"{float(value):+.4f}"


def build():
    task1 = _read_rows(TASK1_INPUTS)
    task2 = _read_rows(TASK2_INPUTS)
    task3 = _read_rows(TASK3_INPUTS)
    _write_csv(OUTPUT_DIR / "task1_3d.csv", task1)
    _write_csv(OUTPUT_DIR / "task2_3d.csv", task2)
    _write_csv(OUTPUT_DIR / "task3_3d.csv", task3)

    task1_gains = [float(row["fmt_minus_raw_f1"]) for row in task1]
    task2_gains = [float(row["paired_f1_gain_mean"]) for row in task2]
    task2_family_mean, task2_positive_families = _family_macro(
        task2, "paired_f1_gain_mean"
    )
    task3_f1_gains = [float(row["mean_gain_f1"]) for row in task3]
    task3_ap_gains = [float(row["mean_gain_average_precision"]) for row in task3]
    task3_raw_f1_gains = [
        float(row["mean_raw_fmt_residual_f1"]) - float(row["mean_raw_f1"])
        for row in task3
    ]
    task3_raw_ap_gains = [
        float(row["mean_raw_fmt_residual_average_precision"])
        - float(row["mean_raw_average_precision"])
        for row in task3
    ]
    task3_family_f1, task3_positive_families = _family_macro(task3, "mean_gain_f1")
    task3_family_ap, _ = _family_macro(task3, "mean_gain_average_precision")

    lines = [
        "# 3D Task1–Task3 论文性能表",
        "",
        "本页只合并已冻结 confirmation 结果，不重新选择任何 feature、VAE、checkpoint、",
        "cluster 映射或阈值。原始机器表位于 `outputs/paper_tables_task123_3d/`。",
        "",
        "## Task1：training-free FMT + KMeans",
        "",
        "| Flow | FMT feature / PCA | FMT F1 | Raw F1 | FMT−Raw F1 | ARI | NMI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in task1:
        pca = row["fmt_pca_dim"]
        config = f'{row["fmt_feature"]} / PCA-{pca}' if pca != "none" else f'{row["fmt_feature"]} / no PCA'
        lines.append(
            f'| {FLOW_NAMES[row["dataset"]]} | {config} | '
            f'{_pm(row["fmt_f1_mean"], row["fmt_f1_std"])} | '
            f'{float(row["raw_f1_mean"]):.4f} | **{_signed(row["fmt_minus_raw_f1"])}** | '
            f'{float(row["fmt_ari_mean"]):.4f} | {float(row["fmt_nmi_mean"]):.4f} |'
        )
    lines.extend([
        "",
        f'FMT 的条目平均 F1 为 `{_mean([float(row["fmt_f1_mean"]) for row in task1]):.4f}`；'
        f'{sum(gain > 0 for gain in task1_gains)}/10 条目高于 Raw。',
        "",
        "## Task2：Raw+VAE 与 FMT+同一 VAE",
        "",
        "| Flow | 同一 VAE | Raw+VAE F1 | FMT+VAE F1 | 配对 F1 增益 | FMT ARI | FMT NMI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in task2:
        lines.append(
            f'| {FLOW_NAMES[row["dataset"]]} | {row["fmt_architecture"]} | '
            f'{_pm(row["raw_f1_mean"], row["raw_f1_std"])} | '
            f'{_pm(row["fmt_f1_mean"], row["fmt_f1_std"])} | '
            f'**{_signed(row["paired_f1_gain_mean"])}±{float(row["paired_f1_gain_std"]):.4f}** | '
            f'{float(row["fmt_ari_mean"]):.4f} | {float(row["fmt_nmi_mean"]):.4f} |'
        )
    lines.extend([
        "",
        f'{sum(gain > 0 for gain in task2_gains)}/10 条目、{task2_positive_families}/7 family 为正；'
        f'条目平均配对增益 `{_mean(task2_gains):+.4f}`，family-macro `{task2_family_mean:+.4f}`。',
        "",
        "## Task3：监督 IVD 二分类",
        "",
        "下面同时给出原始 Raw baseline 和同结构、同参数量的 Raw-PCA residual 强对照。",
        "Raw-PCA 只用 development-validation Average Precision 选择，并未读取 confirmation。",
        "",
        "### F1 score",
        "",
        "| Flow | Raw | Raw-wide | Raw-PCA residual | Raw+FMT residual | FMT−Raw | FMT−Raw-PCA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in task3:
        raw_gain = float(row["mean_raw_fmt_residual_f1"]) - float(row["mean_raw_f1"])
        lines.append(
            f'| {FLOW_NAMES[row["dataset"]]} | '
            f'{_pm(row["mean_raw_f1"], row["std_raw_f1"])} | '
            f'{_pm(row["mean_raw_wide_f1"], row["std_raw_wide_f1"])} | '
            f'{_pm(row["mean_raw_pca_residual_f1"], row["std_raw_pca_residual_f1"])} | '
            f'{_pm(row["mean_raw_fmt_residual_f1"], row["std_raw_fmt_residual_f1"])} | '
            f'**{_signed(raw_gain)}** | **{_signed(row["mean_gain_f1"])}** |'
        )
    lines.extend([
        "",
        "### Average Precision（平均精确率）",
        "",
        "| Flow | Raw | Raw-wide | Raw-PCA residual | Raw+FMT residual | FMT−Raw | FMT−Raw-PCA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in task3:
        raw_gain = (
            float(row["mean_raw_fmt_residual_average_precision"])
            - float(row["mean_raw_average_precision"])
        )
        lines.append(
            f'| {FLOW_NAMES[row["dataset"]]} | '
            f'{_pm(row["mean_raw_average_precision"], row["std_raw_average_precision"])} | '
            f'{_pm(row["mean_raw_wide_average_precision"], row["std_raw_wide_average_precision"])} | '
            f'{_pm(row["mean_raw_pca_residual_average_precision"], row["std_raw_pca_residual_average_precision"])} | '
            f'{_pm(row["mean_raw_fmt_residual_average_precision"], row["std_raw_fmt_residual_average_precision"])} | '
            f'**{_signed(raw_gain)}** | '
            f'**{_signed(row["mean_gain_average_precision"])}** |'
        )
    lines.extend([
        "",
        f'相对原始 Raw，FMT 在 {sum(gain > 0 for gain in task3_raw_f1_gains)}/10 条目提高 F1、'
        f'{sum(gain > 0 for gain in task3_raw_ap_gains)}/10 条目提高 AP；条目平均增益为 '
        f'`{_mean(task3_raw_f1_gains):+.4f}` / `{_mean(task3_raw_ap_gains):+.4f}`。',
        f'相对 Raw-PCA residual 强对照，FMT 在 {sum(gain > 0 for gain in task3_f1_gains)}/10 条目提高 F1、'
        f'{sum(gain > 0 for gain in task3_ap_gains)}/10 条目提高 AP，'
        f'{task3_positive_families}/7 family 的 F1 均值为正；条目平均 F1/AP 增益为 '
        f'`{_mean(task3_f1_gains):+.4f}` / `{_mean(task3_ap_gains):+.4f}`，family-macro 为 '
        f'`{task3_family_f1:+.4f}` / `{task3_family_ap:+.4f}`。',
        "",
        "## 结果来源",
        "",
        "- Task1：`mainExp_Task1_3D_2.1` + `mainExp_Task1_3D_2.2_newflows`。",
        "- Task2：`mainExp_Task2_3D_2.3` + `mainExp_Task2_3D_2.4_newflows`。",
        "- Task3：`mainExp_Task3_3D_3.1`（Ibex V100；10条目×5训练seed×8 confirmation时间片）。",
    ])
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DOC_PATH)


if __name__ == "__main__":
    build()
