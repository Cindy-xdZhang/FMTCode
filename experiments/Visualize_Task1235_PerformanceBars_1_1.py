"""Create publication-ready bar charts for the frozen 3D Task1/2/3/5 tables.

The script reads the machine-readable confirmation tables directly, verifies
their dataset-macro values against the frozen paper table, writes a tidy source
data CSV, and exports editable SVG/PDF plus PNG/TIFF renderings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


EXPERIMENT_ID = "Other_Task1235_PerformanceBars_1.1"
DATASET_ORDER = (
    "channel",
    "cylinder3d",
    "halfcylinderRe640",
    "halfcylinderRe6400",
    "tangaroa",
    "deltaWing_resampled",
    "deltaWing_LBM",
    "f22raptor",
    "boeing747",
    "smokeBuoyancy",
)
DATASET_LABELS = {
    "channel": "Channel observer",
    "cylinder3d": "Half-cylinder Re160",
    "halfcylinderRe640": "Half-cylinder Re640",
    "halfcylinderRe6400": "Half-cylinder Re6400",
    "tangaroa": "Tangaroa",
    "deltaWing_resampled": "Delta wing (resampled)",
    "deltaWing_LBM": "Delta wing (LBM)",
    "f22raptor": "F-22",
    "boeing747": "Boeing 747",
    "smokeBuoyancy": "Smoke buoyancy",
    "dataset_macro": "Dataset macro",
}

RAW_COLOR = "#A7ABB1"
FMT_COLOR = "#0F4D92"
FIXED_FMT_COLOR = "#78A6D0"
STRONGEST_RAW_COLOR = "#555B63"
GRID_COLOR = "#D9DDE2"
MACRO_COLOR = "#EEF2F6"

SOURCE_FILES = {
    "task1": Path("outputs/mainExp_Task1_3D_4.1_ibex/paper_table.csv"),
    "task2": Path(
        "outputs/mainExp_Task2_3D_6.2_uniform_confirmation/confirmation/"
        "paper_table.csv"
    ),
    "task3": Path(
        "outputs/mainExp_Task3_3D_9.2_uniform_confirmation/confirmation/"
        "paper_table.csv"
    ),
    "task5": Path(
        "outputs/mainExp_Task5_3D_1.1_ibex_v100/outputs/"
        "mainExp_Task5_3D_1.1/final_confirmation/paper_table.csv"
    ),
    "task5_fixed": Path(
        "outputs/mainExp_Task5_3D_1.1_ibex_v100/outputs/"
        "mainExp_Task5_3D_1.1/final_confirmation/"
        "fixed_scale_transfer_per_run.csv"
    ),
}

EXPECTED_MACROS = {
    ("Task1", "f1", "raw"): 0.4510,
    ("Task1", "f1", "fmt"): 0.6014,
    ("Task2", "f1", "raw"): 0.4884,
    ("Task2", "f1", "fmt"): 0.5840,
    ("Task3", "f1", "raw"): 0.6398,
    ("Task3", "f1", "fmt"): 0.8583,
    ("Task3", "average_precision", "raw"): 0.6891,
    ("Task3", "average_precision", "fmt"): 0.9236,
    ("Task5", "f1", "fixed_fmt"): 0.5212,
    ("Task5", "f1", "strongest_raw"): 0.5878,
    ("Task5", "f1", "raw"): 0.5835,
    ("Task5", "f1", "fmt"): 0.6727,
    ("Task5", "average_precision", "raw"): 0.6063,
    ("Task5", "average_precision", "fmt"): 0.7180,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="FMT repository root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"output directory (default: outputs/{EXPERIMENT_ID})",
    )
    parser.add_argument(
        "--qa-tools-dir",
        type=Path,
        default=None,
        help="directory containing audit_panel_alignment.py",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_record(
    records: list[dict[str, Any]],
    *,
    task: str,
    metric: str,
    dataset: str,
    method: str,
    mean: float,
    sd: float | None,
    source_experiment: str,
    source_file: str,
    n: int | None = 5,
    note: str = "",
) -> None:
    records.append(
        {
            "task": task,
            "metric": metric,
            "dataset_id": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "method": method,
            "mean": float(mean),
            "sd": "" if sd is None else float(sd),
            "n": "" if n is None else int(n),
            "source_experiment": source_experiment,
            "source_file": source_file,
            "note": note,
        }
    )


def add_macro_records(records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    provenance: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in records:
        if row["dataset_id"] == "dataset_macro":
            continue
        key = (row["task"], row["metric"], row["method"])
        grouped[key].append(float(row["mean"]))
        provenance[key] = (row["source_experiment"], row["source_file"])

    for key, values in grouped.items():
        if len(values) != len(DATASET_ORDER):
            continue
        task, metric, method = key
        source_experiment, source_file = provenance[key]
        add_record(
            records,
            task=task,
            metric=metric,
            dataset="dataset_macro",
            method=method,
            mean=float(np.mean(values)),
            sd=None,
            n=None,
            source_experiment=source_experiment,
            source_file=source_file,
            note="Unweighted mean across the 10 dataset entries; no seed SD shown.",
        )


def build_records(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {key: repo_root / value for key, value in SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen evidence files:\n" + "\n".join(missing))

    records: list[dict[str, Any]] = []

    for row in read_rows(paths["task1"]):
        dataset = row["dataset"]
        add_record(
            records,
            task="Task1",
            metric="f1",
            dataset=dataset,
            method="raw",
            mean=float(row["raw_f1_mean"]),
            sd=float(row["raw_f1_std"]),
            source_experiment="mainExp_Task1_3D_4.1",
            source_file=str(SOURCE_FILES["task1"]),
        )
        add_record(
            records,
            task="Task1",
            metric="f1",
            dataset=dataset,
            method="fmt",
            mean=float(row["fmt_f1_mean"]),
            sd=float(row["fmt_f1_std"]),
            source_experiment="mainExp_Task1_3D_4.1",
            source_file=str(SOURCE_FILES["task1"]),
        )

    for row in read_rows(paths["task2"]):
        dataset = row["dataset"]
        add_record(
            records,
            task="Task2",
            metric="f1",
            dataset=dataset,
            method="raw",
            mean=float(row["raw_f1"]),
            sd=float(row["raw_f1_std"]),
            source_experiment="mainExp_Task2_3D_6.2_uniform_confirmation",
            source_file=str(SOURCE_FILES["task2"]),
        )
        add_record(
            records,
            task="Task2",
            metric="f1",
            dataset=dataset,
            method="fmt",
            mean=float(row["fmt_f1"]),
            sd=float(row["fmt_f1_std"]),
            source_experiment="mainExp_Task2_3D_6.2_uniform_confirmation",
            source_file=str(SOURCE_FILES["task2"]),
        )

    for row in read_rows(paths["task3"]):
        dataset = row["dataset"]
        for metric, raw_key, raw_sd_key, fmt_key, fmt_sd_key in (
            ("f1", "raw_pca_f1", "raw_pca_f1_std", "fmt_f1", "fmt_f1_std"),
            (
                "average_precision",
                "raw_pca_average_precision",
                "raw_pca_average_precision_std",
                "fmt_average_precision",
                "fmt_average_precision_std",
            ),
        ):
            add_record(
                records,
                task="Task3",
                metric=metric,
                dataset=dataset,
                method="raw",
                mean=float(row[raw_key]),
                sd=float(row[raw_sd_key]),
                source_experiment="mainExp_Task3_3D_9.2_uniform_confirmation",
                source_file=str(SOURCE_FILES["task3"]),
            )
            add_record(
                records,
                task="Task3",
                metric=metric,
                dataset=dataset,
                method="fmt",
                mean=float(row[fmt_key]),
                sd=float(row[fmt_sd_key]),
                source_experiment="mainExp_Task3_3D_9.2_uniform_confirmation",
                source_file=str(SOURCE_FILES["task3"]),
            )

    for row in read_rows(paths["task5"]):
        dataset = row["dataset"]
        selected_raw = row["selected_raw_variant"]
        if selected_raw == "raw_wide":
            strongest_mean_key = "mean_raw_wide_f1"
            strongest_sd_key = "std_raw_wide_f1"
        elif selected_raw == "raw_pca_residual":
            strongest_mean_key = "mean_raw_pca_residual_f1"
            strongest_sd_key = "std_raw_pca_residual_f1"
        else:
            raise ValueError(f"Unsupported strongest Raw variant: {selected_raw}")

        add_record(
            records,
            task="Task5",
            metric="f1",
            dataset=dataset,
            method="strongest_raw",
            mean=float(row[strongest_mean_key]),
            sd=float(row[strongest_sd_key]),
            source_experiment="mainExp_Task5_3D_1.1",
            source_file=str(SOURCE_FILES["task5"]),
            note=f"Development-selected strongest Raw variant: {selected_raw}.",
        )
        for metric, raw_key, raw_sd_key, fmt_key, fmt_sd_key in (
            (
                "f1",
                "mean_raw_pca_residual_f1",
                "std_raw_pca_residual_f1",
                "mean_raw_fmt_residual_f1",
                "std_raw_fmt_residual_f1",
            ),
            (
                "average_precision",
                "mean_raw_pca_residual_average_precision",
                "std_raw_pca_residual_average_precision",
                "mean_raw_fmt_residual_average_precision",
                "std_raw_fmt_residual_average_precision",
            ),
        ):
            add_record(
                records,
                task="Task5",
                metric=metric,
                dataset=dataset,
                method="raw",
                mean=float(row[raw_key]),
                sd=float(row[raw_sd_key]),
                source_experiment="mainExp_Task5_3D_1.1",
                source_file=str(SOURCE_FILES["task5"]),
            )
            add_record(
                records,
                task="Task5",
                metric=metric,
                dataset=dataset,
                method="fmt",
                mean=float(row[fmt_key]),
                sd=float(row[fmt_sd_key]),
                source_experiment="mainExp_Task5_3D_1.1",
                source_file=str(SOURCE_FILES["task5"]),
            )

    fixed_by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in read_rows(paths["task5_fixed"]):
        if row["variant"] == "fixed_fmt":
            fixed_by_dataset[row["dataset"]].append(float(row["f1"]))
    for dataset in DATASET_ORDER:
        values = fixed_by_dataset[dataset]
        if len(values) != 5:
            raise ValueError(f"Expected five fixed-FMT seeds for {dataset}, found {len(values)}")
        add_record(
            records,
            task="Task5",
            metric="f1",
            dataset=dataset,
            method="fixed_fmt",
            mean=statistics.mean(values),
            sd=statistics.stdev(values),
            source_experiment="mainExp_Task5_3D_1.1",
            source_file=str(SOURCE_FILES["task5_fixed"]),
            note="Fixed-scale Task3 FMT model transferred to Task5 confirmation primitives.",
        )

    add_macro_records(records)
    lookup = {
        (row["task"], row["metric"], row["method"], row["dataset_id"]): float(
            row["mean"]
        )
        for row in records
    }
    macro_checks: dict[str, Any] = {}
    for key, expected in EXPECTED_MACROS.items():
        observed = lookup[(*key, "dataset_macro")]
        passed = round(observed, 4) == round(expected, 4)
        macro_checks["/".join(key)] = {
            "expected_paper_value": expected,
            "observed_from_machine_table": observed,
            "pass_at_four_decimals": passed,
        }
        if not passed:
            raise AssertionError(f"Frozen macro mismatch for {key}: {observed} != {expected}")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "source_files": {
            key: {"path": str(SOURCE_FILES[key]), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "macro_checks": macro_checks,
    }
    return records, manifest


def write_tidy_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = (
        "task",
        "metric",
        "dataset_id",
        "dataset_label",
        "method",
        "mean",
        "sd",
        "n",
        "source_experiment",
        "source_file",
        "note",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def record_lookup(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (row["task"], row["metric"], row["method"], row["dataset_id"]): row
        for row in records
    }


def values_for(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    task: str,
    metric: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    datasets = (*DATASET_ORDER, "dataset_macro")
    means = np.asarray(
        [float(lookup[(task, metric, method, dataset)]["mean"]) for dataset in datasets]
    )
    sd = np.asarray(
        [
            np.nan
            if lookup[(task, metric, method, dataset)]["sd"] == ""
            else float(lookup[(task, metric, method, dataset)]["sd"])
            for dataset in datasets
        ]
    )
    return means, sd


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.0,
            "xtick.major.size": 2.5,
            "ytick.major.size": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def style_metric_axis(ax: plt.Axes, *, panel_label: str, title: str) -> None:
    labels = [DATASET_LABELS[dataset] for dataset in (*DATASET_ORDER, "dataset_macro")]
    y = np.arange(len(labels))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)
    ax.axhspan(len(labels) - 1.5, len(labels) - 0.5, color=MACRO_COLOR, zorder=0)
    ax.axhline(len(labels) - 1.5, color="#B9BEC5", linewidth=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#70757A")
    ax.set_title(title, loc="left", pad=5, fontweight="bold")
    ax.text(
        -0.24,
        1.055,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.get_yticklabels()[-1].set_fontweight("bold")


def add_pair_bars(
    ax: plt.Axes,
    raw_mean: np.ndarray,
    raw_sd: np.ndarray,
    fmt_mean: np.ndarray,
    fmt_sd: np.ndarray,
    *,
    panel_label: str,
    title: str,
) -> None:
    y = np.arange(len(raw_mean))
    height = 0.34
    error_kw = {"ecolor": "#2F3337", "elinewidth": 0.6, "capsize": 1.4, "capthick": 0.6}
    ax.barh(
        y - height / 2,
        raw_mean,
        height,
        xerr=np.nan_to_num(raw_sd, nan=0.0),
        color=RAW_COLOR,
        edgecolor="white",
        linewidth=0.35,
        error_kw=error_kw,
        label="Raw baseline",
        zorder=3,
    )
    ax.barh(
        y + height / 2,
        fmt_mean,
        height,
        xerr=np.nan_to_num(fmt_sd, nan=0.0),
        color=FMT_COLOR,
        edgecolor="white",
        linewidth=0.35,
        error_kw=error_kw,
        label="FMT",
        zorder=3,
    )
    style_metric_axis(ax, panel_label=panel_label, title=title)


def require_alignment(
    fig: plt.Figure,
    axes: Iterable[plt.Axes],
    *,
    qa_tools_dir: Path | None,
    json_out: Path,
    overlay_svg: Path,
) -> None:
    if qa_tools_dir is None:
        raise ValueError("--qa-tools-dir is required for multi-panel figure QA")
    qa_tools_dir = qa_tools_dir.resolve()
    if str(qa_tools_dir) not in sys.path:
        sys.path.insert(0, str(qa_tools_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    axes_list = list(axes)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=axes_list,
        panel_ids=[chr(ord("a") + index) for index in range(len(axes_list))],
        json_out=json_out,
        overlay_svg=overlay_svg,
        require_panel_labels=True,
        strict=True,
    )


def export_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def create_main_f1_figure(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    output_dir: Path,
    qa_tools_dir: Path | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.20472, 7.00787))  # 183 mm wide.
    panels = (
        ("Task1", "a", "Task 1 · KMeans clustering"),
        ("Task2", "b", "Task 2 · same VAE + KMeans"),
        ("Task3", "c", "Task 3 · supervised IVD-p95"),
        ("Task5", "d", "Task 5 · variable-scale IVD-p95"),
    )
    for ax, (task, panel_label, title) in zip(axes.flat, panels):
        raw_mean, raw_sd = values_for(lookup, task, "f1", "raw")
        fmt_mean, fmt_sd = values_for(lookup, task, "f1", "fmt")
        add_pair_bars(
            ax,
            raw_mean,
            raw_sd,
            fmt_mean,
            fmt_sd,
            panel_label=panel_label,
            title=title,
        )
        ax.set_xlabel("F1 score")

    legend_handles = (
        Patch(facecolor=RAW_COLOR, edgecolor="none", label="Raw baseline"),
        Patch(facecolor=FMT_COLOR, edgecolor="none", label="FMT"),
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.text(
        0.5,
        0.012,
        "Bars show means; whiskers show sample SD across five seeds. "
        "Dataset macro is the unweighted mean across 10 entries.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#4D5156",
    )
    fig.subplots_adjust(left=0.165, right=0.985, top=0.905, bottom=0.075, wspace=0.47, hspace=0.34)
    require_alignment(
        fig,
        axes.flat,
        qa_tools_dir=qa_tools_dir,
        json_out=output_dir / "qa_main_f1_panel_alignment.json",
        overlay_svg=output_dir / "qa_main_f1_panel_alignment_overlay.svg",
    )
    export_figure(fig, output_dir / "task1235_main_f1_bars")
    plt.close(fig)


def create_ap_figure(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    output_dir: Path,
    qa_tools_dir: Path | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.20472, 3.93701))  # 183 mm wide.
    for ax, task, panel_label, title in (
        (axes[0], "Task3", "a", "Task 3 · supervised IVD-p95"),
        (axes[1], "Task5", "b", "Task 5 · variable-scale IVD-p95"),
    ):
        raw_mean, raw_sd = values_for(lookup, task, "average_precision", "raw")
        fmt_mean, fmt_sd = values_for(lookup, task, "average_precision", "fmt")
        add_pair_bars(
            ax,
            raw_mean,
            raw_sd,
            fmt_mean,
            fmt_sd,
            panel_label=panel_label,
            title=title,
        )
        ax.set_xlabel("Average precision")

    fig.legend(
        handles=(
            Patch(facecolor=RAW_COLOR, edgecolor="none", label="Raw-PCA residual"),
            Patch(facecolor=FMT_COLOR, edgecolor="none", label="Raw+FMT residual"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.58, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.text(
        0.5,
        0.018,
        "Bars show means; whiskers show sample SD across five training seeds.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#4D5156",
    )
    fig.subplots_adjust(left=0.165, right=0.985, top=0.82, bottom=0.14, wspace=0.47)
    require_alignment(
        fig,
        axes,
        qa_tools_dir=qa_tools_dir,
        json_out=output_dir / "qa_task35_ap_panel_alignment.json",
        overlay_svg=output_dir / "qa_task35_ap_panel_alignment_overlay.svg",
    )
    export_figure(fig, output_dir / "task35_average_precision_bars")
    plt.close(fig)


def create_task5_complete_figure(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    output_dir: Path,
    qa_tools_dir: Path | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.20472, 3.93701))  # 183 mm wide.
    ax = axes[0]
    methods = (
        ("fixed_fmt", "Fixed Task3 FMT transfer", FIXED_FMT_COLOR, ""),
        ("strongest_raw", "Strongest Raw", STRONGEST_RAW_COLOR, ""),
        ("raw", "Task5 Raw-PCA", RAW_COLOR, ""),
        ("fmt", "Task5 FMT", FMT_COLOR, ""),
    )
    y = np.arange(len(DATASET_ORDER) + 1)
    height = 0.18
    offsets = (-1.5 * height, -0.5 * height, 0.5 * height, 1.5 * height)
    error_kw = {"ecolor": "#2F3337", "elinewidth": 0.5, "capsize": 1.1, "capthick": 0.5}
    for offset, (method, label, color, hatch) in zip(offsets, methods):
        mean, sd = values_for(lookup, "Task5", "f1", method)
        ax.barh(
            y + offset,
            mean,
            height,
            xerr=np.nan_to_num(sd, nan=0.0),
            color=color,
            edgecolor="white" if not hatch else "#4779A5",
            linewidth=0.35,
            hatch=hatch,
            error_kw=error_kw,
            label=label,
            zorder=3,
        )
    style_metric_axis(ax, panel_label="a", title="Task 5 · complete F1 comparison")
    ax.set_xlabel("F1 score")

    raw_mean, raw_sd = values_for(lookup, "Task5", "average_precision", "raw")
    fmt_mean, fmt_sd = values_for(lookup, "Task5", "average_precision", "fmt")
    add_pair_bars(
        axes[1],
        raw_mean,
        raw_sd,
        fmt_mean,
        fmt_sd,
        panel_label="b",
        title="Task 5 · average precision",
    )
    axes[1].set_xlabel("Average precision")

    legend_handles = tuple(
        Patch(
            facecolor=color,
            edgecolor="white" if not hatch else "#4779A5",
            hatch=hatch,
            label=label,
        )
        for _, label, color, hatch in methods
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.997),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.5,
    )
    fig.text(
        0.5,
        0.016,
        "Means and sample SD across five training seeds; macro rows have no seed whisker.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#4D5156",
    )
    fig.subplots_adjust(left=0.165, right=0.985, top=0.86, bottom=0.14, wspace=0.47)
    require_alignment(
        fig,
        axes,
        qa_tools_dir=qa_tools_dir,
        json_out=output_dir / "qa_task5_complete_panel_alignment.json",
        overlay_svg=output_dir / "qa_task5_complete_panel_alignment_overlay.svg",
    )
    export_figure(fig, output_dir / "task5_complete_bars")
    plt.close(fig)


def write_figure_contract(path: Path) -> None:
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "figure_archetype": "quantitative-grid",
        "core_conclusion": (
            "Across the frozen 10-entry 3D confirmation tables, FMT improves the "
            "matched Raw baseline in dataset-macro F1 for Tasks 1, 2, 3, and 5; "
            "all retained dataset-level counterexamples remain visible."
        ),
        "evidence_logic": [
            "Main 2x2 figure compares matched Raw and FMT F1 values for every dataset entry and task.",
            "The supervised supporting figure repeats the comparison with Average Precision for Tasks 3 and 5.",
            "The Task5 complete figure includes fixed Task3 transfer and strongest-Raw context in addition to the capacity-matched Raw-PCA/FMT pair.",
        ],
        "data_integrity": {
            "selection": "No result or dataset is omitted; the figures use current frozen paper-main confirmation tables only.",
            "uncertainty": "Whiskers are sample SD across five seeds; dataset-macro bars are unweighted means and have no seed whisker.",
            "labels": "Task1/2 evaluate IVD-p95 labels; Task3/5 train and evaluate supervised IVD-p95 vortex-region detection.",
        },
        "outputs": [
            "task1235_main_f1_bars",
            "task35_average_precision_bars",
            "task5_complete_bars",
        ],
    }
    path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "outputs" / EXPERIMENT_ID
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    records, source_manifest = build_records(repo_root)
    write_tidy_csv(records, output_dir / "source_data_task1235.csv")
    (output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_figure_contract(output_dir / "figure_contract.json")

    lookup = record_lookup(records)
    create_main_f1_figure(lookup, output_dir, args.qa_tools_dir)
    create_ap_figure(lookup, output_dir, args.qa_tools_dir)
    create_task5_complete_figure(lookup, output_dir, args.qa_tools_dir)
    print(f"Wrote Task1/2/3/5 figures to {output_dir}")


if __name__ == "__main__":
    main()
