"""Build the Task4-a velocity--curl baseline and comparison figure."""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import yaml

from FMT_Utils.Task4A_StreamlineClustering_3D import (
    ChannelVelocityField3D,
    summarize_topology,
    topology_proxy_rows,
)
from FMT_Utils.Task4A_TraditionalBaseline_3D import (
    paired_topology_comparison,
    proxy_groundtruth_agreement,
    velocity_curl_orientation_classes,
)
from FMT_Utils.VoxelSegmentation_3D import (
    SegmentationField3D,
    VoxelGrid3D,
    render_voxel_segmentation,
)


DEFAULT_CONFIG = Path("config/Verify_Task4A_VelocityCurlBaseline_1.1.yaml")


def _resolve_flow(candidates: list[str], override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    path = next((Path(value) for value in candidates if Path(value).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no channel velocity field found among {candidates}")
    return path.resolve()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _class_color(label: int) -> tuple[float, float, float]:
    hue = (0.6180339887498949 * max(1, int(label))) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.68, 0.95)


def _view_path(paths: list[str], suffix: str) -> Path:
    return next(Path(path) for path in paths if Path(path).stem.endswith(suffix))


def _make_comparison_figure(
    output_dir: Path,
    fmt_views: list[str],
    baseline_views: list[str],
    paired_rows: list[dict],
    paired_summary: dict,
) -> dict:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.955,
                        wspace=0.20, hspace=0.27)
    panels = axes.ravel()
    stream_color = _class_color(1)
    span_color = _class_color(2)

    image_panels = (
        (panels[0], _view_path(fmt_views, "isometric"),
         "Pure-FMT KMeans — isometric"),
        (panels[1], _view_path(baseline_views, "isometric"),
         "Velocity–curl baseline — isometric"),
        (panels[2], _view_path(baseline_views, "plus-y"),
         "Velocity–curl baseline — view along +y"),
    )
    for ax, image_path, title in image_panels:
        ax.imshow(plt.imread(image_path), aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    panels[0].legend(
        handles=[
            Patch(facecolor=stream_color, label="Streamwise / parallel"),
            Patch(facecolor=span_color, label="Spanwise / perpendicular"),
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
    )

    fmt_scores = np.asarray([row["fmt_soft_topology_score"] for row in paired_rows])
    baseline_scores = np.asarray(
        [row["baseline_soft_topology_score"] for row in paired_rows]
    )
    delta = baseline_scores - fmt_scores
    tolerance = 1e-12
    better = delta > tolerance
    worse = delta < -tolerance
    equal = ~(better | worse)
    panels[3].plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.0)
    panels[3].scatter(
        fmt_scores[better], baseline_scores[better], s=28, color="#198754",
        alpha=0.85, label=f"Baseline higher ({int(better.sum())})",
    )
    panels[3].scatter(
        fmt_scores[worse], baseline_scores[worse], s=28, color="#a33a2b",
        marker="x", label=f"FMT higher ({int(worse.sum())})",
    )
    panels[3].scatter(
        fmt_scores[equal], baseline_scores[equal], s=24, color="0.55",
        marker="s", label=f"Equal ({int(equal.sum())})",
    )
    panels[3].set(
        xlabel="Pure-FMT KMeans soft topology score",
        ylabel="Velocity–curl baseline soft topology score",
        xlim=(-0.025, 1.025),
        ylim=(-0.025, 1.025),
        title=(
            "Paired per-instance topology  "
            f"(exact 2+1: {paired_summary['fmt_hard_success_count']} → "
            f"{paired_summary['baseline_hard_success_count']})"
        ),
    )
    panels[3].legend(loc="lower right", frameon=False)

    for label, ax in zip("abcd", panels):
        ax.text(
            -0.07,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    fig.canvas.draw()
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    alignment = require_matplotlib_panel_alignment(
        fig,
        json_out=qa_dir / "panel_alignment.json",
        overlay_svg=qa_dir / "panel_alignment_overlay.svg",
        require_panel_labels=False,
        strict=True,
    )
    paths = {
        "png": output_dir / "task4a_velocity_curl_baseline_comparison.png",
        "pdf": output_dir / "task4a_velocity_curl_baseline_comparison.pdf",
        "svg": output_dir / "task4a_velocity_curl_baseline_comparison.svg",
        "tiff": output_dir / "task4a_velocity_curl_baseline_comparison.tiff",
    }
    fig.savefig(paths["png"], dpi=600)
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    fig.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "panel_alignment_verdict": alignment.get("verdict"),
        "panel_alignment_report": str((qa_dir / "panel_alignment.json").resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--fmt-result")
    parser.add_argument("--flow")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main() -> dict:
    args = _parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_config = config["input"]
    fmt_result_path = Path(args.fmt_result or input_config["fmt_result"])
    if not fmt_result_path.is_file():
        raise FileNotFoundError(fmt_result_path)
    flow_path = _resolve_flow(input_config["flow_path_candidates"], args.flow)
    output_dir = Path(args.output_dir or config["visualization"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    source = np.load(fmt_result_path)
    grid = VoxelGrid3D(
        tuple(int(value) for value in source["resolution_xyz"]),
        source["domain_min_xyz"],
        source["domain_max_xyz"],
    )
    indices_xyz = np.asarray(source["voxel_indices_xyz"], dtype=np.int32)
    seeds_xyz = np.asarray(source["seeds_xyz"], dtype=np.float64)
    vortex_ids = np.asarray(source["vortex_ids"], dtype=np.int32)
    fmt_classes = np.asarray(source["semantic_classes_after_fill"], dtype=np.int8)
    fmt_volume = np.asarray(source["predicted_classes_zyx"], dtype=np.int8)

    instance_volume = np.zeros(grid.shape_zyx, dtype=np.int32)
    ix, iy, iz = indices_xyz.T
    instance_volume[iz, iy, ix] = vortex_ids
    if np.count_nonzero(instance_volume > 0) != len(seeds_xyz):
        raise RuntimeError("cached positive voxel indices are not one-to-one")

    field, flow_metadata = ChannelVelocityField3D.from_vtk(flow_path)
    velocity = field.velocity(seeds_xyz)
    curl = field.vorticity(seeds_xyz)
    criterion = config["criterion"]
    baseline_classes, angle_degrees, valid = velocity_curl_orientation_classes(
        velocity,
        curl,
        streamwise_max_angle_degrees=float(
            criterion["streamwise_if_angle_degrees_lte"]
        ),
        minimum_vector_norm=float(criterion["minimum_vector_norm"]),
    )
    baseline_volume = np.zeros(grid.shape_zyx, dtype=np.int8)
    baseline_volume[iz, iy, ix] = baseline_classes

    topology_config = config["topology_proxy"]
    topology_args = {
        "dominant_min_voxels": int(
            topology_config["dominant_component_min_voxels"]
        ),
        "dominant_min_fraction": float(
            topology_config["dominant_component_min_fraction_of_vortex"]
        ),
    }
    fmt_rows = topology_proxy_rows(instance_volume, fmt_volume, **topology_args)
    baseline_rows = topology_proxy_rows(
        instance_volume, baseline_volume, **topology_args
    )
    fmt_summary = summarize_topology(fmt_rows)
    baseline_summary = summarize_topology(baseline_rows)
    paired_rows, paired_summary = paired_topology_comparison(fmt_rows, baseline_rows)
    metrics_path = output_dir / "per_vortex_fmt_vs_velocity_curl.csv"
    _write_csv(metrics_path, paired_rows)
    proxy_rows, proxy_summary = proxy_groundtruth_agreement(
        fmt_classes,
        baseline_classes,
        vortex_ids,
        angle_degrees,
    )
    proxy_metrics_path = output_dir / "per_vortex_proxy_groundtruth_metrics.csv"
    _write_csv(proxy_metrics_path, proxy_rows)

    result_path = output_dir / "velocity_curl_baseline_result.npz"
    np.savez_compressed(
        result_path,
        resolution_xyz=np.asarray(grid.resolution_xyz),
        domain_min_xyz=grid.domain_min_xyz,
        domain_max_xyz=grid.domain_max_xyz,
        voxel_indices_xyz=indices_xyz,
        vortex_ids=vortex_ids,
        velocity_xyz=velocity,
        curl_xyz=curl,
        absolute_velocity_curl_angle_degrees=angle_degrees,
        valid_orientation=valid,
        baseline_classes=baseline_classes,
        baseline_classes_zyx=baseline_volume,
        fmt_classes=fmt_classes,
        fmt_classes_zyx=fmt_volume,
    )

    visual_config = config["visualization"]
    fmt_render = {}
    baseline_render = {}
    figure = {}
    if not args.no_images or args.interactive:
        common_render = {
            "views": visual_config["views"],
            "surface_mode": str(visual_config["surface_mode"]),
            "shrink": float(visual_config["shrink"]),
            "opacity": float(visual_config["opacity"]),
            "image_size": visual_config["image_size"],
        }
        fmt_segmentation = SegmentationField3D(
            grid, fmt_volume, "pure_fmt_streamwise1_spanwise2"
        )
        baseline_segmentation = SegmentationField3D(
            grid, baseline_volume, "velocity_curl_streamwise1_spanwise2"
        )
        if not args.no_images:
            fmt_render = render_voxel_segmentation(
                fmt_segmentation,
                output_dir=output_dir / "views" / "fmt",
                interactive=False,
                **common_render,
            )
        baseline_render = render_voxel_segmentation(
            baseline_segmentation,
            output_dir=None if args.no_images else output_dir / "views" / "baseline",
            interactive=bool(args.interactive),
            **common_render,
        )
        if not args.no_images:
            figure = _make_comparison_figure(
                output_dir,
                fmt_render["view_paths"],
                baseline_render["view_paths"],
                paired_rows,
                paired_summary,
            )

    valid_classes = valid & (fmt_classes > 0)
    summary = {
        "experiment": config["experiment"],
        "task_scope": config["task"],
        "config": str(config_path.resolve()),
        "source_fmt_result": str(fmt_result_path.resolve()),
        "flow": flow_metadata,
        "criterion": {
            "formula": "theta=acos(abs(dot(v,curl))/(norm(v)*norm(curl)))",
            "streamwise_if_angle_degrees_lte": float(
                criterion["streamwise_if_angle_degrees_lte"]
            ),
            "valid_orientation_count": int(np.count_nonzero(valid)),
            "invalid_orientation_count": int(np.count_nonzero(~valid)),
            "angle_percentiles_degrees": {
                str(percentile): float(np.percentile(angle_degrees[valid], percentile))
                for percentile in (0, 10, 25, 50, 75, 90, 100)
            },
            "baseline_class_counts": {
                "streamwise_parallel": int(np.count_nonzero(baseline_classes == 1)),
                "spanwise_perpendicular": int(np.count_nonzero(baseline_classes == 2)),
                "unassigned": int(np.count_nonzero(baseline_classes == 0)),
            },
        },
        "fmt_topology": fmt_summary,
        "baseline_topology": baseline_summary,
        "paired_topology": paired_summary,
        "proxy_groundtruth_agreement": proxy_summary,
        "partition_similarity_not_accuracy": {
            "semantic_class_agreement": float(
                np.mean(fmt_classes[valid_classes] == baseline_classes[valid_classes])
            ),
            "adjusted_rand_index": float(
                adjusted_rand_score(
                    fmt_classes[valid_classes], baseline_classes[valid_classes]
                )
            ),
            "normalized_mutual_information": float(
                normalized_mutual_info_score(
                    fmt_classes[valid_classes], baseline_classes[valid_classes]
                )
            ),
        },
        "result_path": str(result_path.resolve()),
        "per_vortex_metrics_path": str(metrics_path.resolve()),
        "per_vortex_proxy_groundtruth_metrics_path": str(
            proxy_metrics_path.resolve()
        ),
        "fmt_render": fmt_render,
        "baseline_render": baseline_render,
        "figure": figure,
        "interpretation_boundary": (
            "This is a deterministic physical baseline, not streamwise/spanwise "
            "ground truth. Curl perpendicular to local velocity includes wall-normal "
            "as well as spanwise orientations."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
