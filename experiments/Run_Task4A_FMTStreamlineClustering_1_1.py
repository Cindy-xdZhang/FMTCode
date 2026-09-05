"""Run Other_Task4A_FMTStreamlineClustering_1.1 on channel GT instances."""

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
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

from FMT_Utils.Task4A_StreamlineClustering_3D import (
    ChannelVelocityField3D,
    SPANWISE_CLASS,
    STREAMWISE_CLASS,
    cluster_fmt_primitives,
    fill_invalid_predictions_within_vortex,
    integrate_bidirectional_cross_primitives,
    summarize_topology,
    topology_proxy_rows,
)
from FMT_Utils.VoxelSegmentation_3D import (
    SegmentationField3D,
    load_vtk_segmentation,
    render_voxel_segmentation,
)


DEFAULT_CONFIG = Path("config/Other_Task4A_FMTStreamlineClustering_1.1.yaml")


def _resolve_path(candidates: list[str], override: str | None, label: str) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    path = next((Path(value) for value in candidates if Path(value).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no {label} file found among {candidates}")
    return path.resolve()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vtk_class_color(label: int) -> tuple[float, float, float]:
    hue = (0.6180339887498949 * max(1, int(label))) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.68, 0.95)


def _make_summary_figure(
    output_dir: Path,
    view_paths: list[str],
    clustering: dict,
    topology_rows: list[dict],
    topology_summary: dict,
) -> dict:
    """Compose the spatial evidence and label-free diagnostics in Matplotlib."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    by_name = {Path(path).stem: Path(path) for path in view_paths}
    iso = next(path for name, path in by_name.items() if name.endswith("isometric"))
    plus_y = next(path for name, path in by_name.items() if name.endswith("plus-y"))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.955,
                        wspace=0.20, hspace=0.27)
    panels = axes.ravel()
    stream_color = _vtk_class_color(STREAMWISE_CLASS)
    span_color = _vtk_class_color(SPANWISE_CLASS)

    for ax, image_path, title in (
        (panels[0], iso, "Spatial partition — isometric"),
        (panels[1], plus_y, "Spatial partition — view along +y"),
    ):
        ax.imshow(plt.imread(image_path), aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    panels[0].legend(
        handles=[
            Patch(facecolor=stream_color, label="Streamwise (post-hoc name)"),
            Patch(facecolor=span_color, label="Spanwise (post-hoc name)"),
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
    )

    medians = np.asarray(clustering["cluster_vorticity_axis_alignment_medians_xyz"])
    x = np.arange(3)
    width = 0.34
    stream_cluster = int(clustering["streamwise_raw_cluster"])
    span_cluster = int(clustering["spanwise_raw_cluster"])
    panels[2].bar(
        x - width / 2,
        medians[stream_cluster],
        width,
        color=stream_color,
        label=f"Raw cluster {stream_cluster} → streamwise",
    )
    panels[2].bar(
        x + width / 2,
        medians[span_cluster],
        width,
        color=span_color,
        label=f"Raw cluster {span_cluster} → spanwise",
    )
    panels[2].set(
        xticks=x,
        xticklabels=["|x| streamwise", "|y| spanwise", "|z| wall-normal"],
        ylabel="Median absolute vorticity alignment",
        ylim=(0, 1.02),
        title="Post-hoc vortex-axis naming diagnostic",
    )
    panels[2].grid(axis="y", color="0.9", linewidth=0.7)
    panels[2].legend(loc="upper right", frameon=False)

    ordered = sorted(topology_rows, key=lambda row: row["soft_topology_score"])
    rank = np.arange(1, len(ordered) + 1)
    scores = np.asarray([row["soft_topology_score"] for row in ordered])
    success = np.asarray([row["hard_2plus1_success"] for row in ordered], dtype=bool)
    panels[3].bar(rank, scores, width=0.88, color="0.76", edgecolor="none")
    panels[3].scatter(
        rank[success], scores[success], marker="o", s=20, color="#198754",
        label="Exact 2+1 success", zorder=3,
    )
    panels[3].scatter(
        rank[~success], scores[~success], marker="x", s=18, color="#a33a2b",
        label="Other topology", zorder=3,
    )
    panels[3].axhline(
        topology_summary["mean_soft_topology_score_all"],
        color="0.15", linestyle="--", linewidth=1.0, label="Mean soft score",
    )
    panels[3].set(
        xlabel="Vortex instances, sorted by soft score",
        ylabel="Soft topology score",
        xlim=(0, len(ordered) + 1),
        ylim=(0, 1.02),
        title=(
            "Per-instance 2+1 topology proxy  "
            f"({topology_summary['hard_2plus1_success_count_all']}/"
            f"{topology_summary['vortex_id_count']} exact)"
        ),
    )
    panels[3].grid(axis="y", color="0.9", linewidth=0.7)
    panels[3].legend(loc="upper left", frameon=False)

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
    audit_dir = output_dir / "qa"
    audit_dir.mkdir(parents=True, exist_ok=True)
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    alignment_report = require_matplotlib_panel_alignment(
        fig,
        json_out=audit_dir / "panel_alignment.json",
        overlay_svg=audit_dir / "panel_alignment_overlay.svg",
        require_panel_labels=False,
        strict=True,
    )

    stems = {
        "png": output_dir / "task4a_fmt_streamline_clustering_summary.png",
        "pdf": output_dir / "task4a_fmt_streamline_clustering_summary.pdf",
        "svg": output_dir / "task4a_fmt_streamline_clustering_summary.svg",
        "tiff": output_dir / "task4a_fmt_streamline_clustering_summary.tiff",
    }
    fig.savefig(stems["png"], dpi=600)
    fig.savefig(stems["pdf"])
    fig.savefig(stems["svg"])
    fig.savefig(stems["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {
        "figure_paths": {key: str(path.resolve()) for key, path in stems.items()},
        "panel_alignment_status": alignment_report.get("verdict"),
        "panel_alignment_report": str((audit_dir / "panel_alignment.json").resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gt")
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
    gt_path = _resolve_path(input_config["gt_path_candidates"], args.gt, "GT")
    flow_path = _resolve_path(input_config["flow_path_candidates"], args.flow, "flow")
    output_dir = Path(args.output_dir or config["visualization"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    voxel_config = config["voxelization"]
    segmentation, gt_metadata, selected = load_vtk_segmentation(
        gt_path,
        resolution_xyz=voxel_config.get("resolution_xyz"),
        array_name=input_config["label_array"],
        association=input_config["association"],
        max_voxels=int(voxel_config["preview_max_voxels"]),
        max_axis_resolution=int(voxel_config["max_axis_resolution"]),
    )
    positive = segmentation.labels_zyx > 0
    iz, iy, ix = np.nonzero(positive)
    indices_xyz = np.column_stack((ix, iy, iz)).astype(np.int32, copy=False)
    seeds_xyz = segmentation.grid.domain_min_xyz + (
        indices_xyz + 0.5
    ) * segmentation.grid.voxel_size_xyz
    vortex_ids = segmentation.labels_zyx[iz, iy, ix]

    field, flow_metadata = ChannelVelocityField3D.from_vtk(flow_path)
    streamline_config = config["streamlines"]
    spatial_step = float(
        streamline_config["spatial_step_mean_voxel_scale"]
        * np.mean(segmentation.grid.voxel_size_xyz)
    )
    offset = float(
        streamline_config["offset_min_voxel_scale"]
        * np.min(segmentation.grid.voxel_size_xyz)
    )
    primitives, valid = integrate_bidirectional_cross_primitives(
        field,
        seeds_xyz,
        spatial_step=spatial_step,
        steps_per_direction=int(streamline_config["steps_per_direction"]),
        offset=offset,
        minimum_speed=float(streamline_config["minimum_speed"]),
        chunk_size=int(streamline_config["chunk_size"]),
    )
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("fewer than two valid streamline primitives")
    seed_vorticity = field.vorticity(seeds_xyz)
    if not np.isfinite(seed_vorticity[valid]).all():
        raise RuntimeError("valid streamline seeds have non-finite vorticity")

    encoder = config["encoder"]
    clustering_config = config["clustering"]
    clustering = cluster_fmt_primitives(
        primitives[valid],
        vortex_ids[valid],
        seed_vorticity[valid],
        num_freq=int(encoder["num_freq"]),
        mode=str(encoder["mode"]),
        include_chirality=bool(encoder["include_chirality"]),
        neighbor_pool=str(encoder["neighbor_pool"]),
        neighbor_weight=float(encoder["neighbor_weight_after_standardization"]),
        n_init=int(clustering_config["n_init"]),
        fit_max_seeds_per_vortex=int(clustering_config["fit_max_seeds_per_vortex"]),
        seed=int(config["seed"]),
    )
    semantic_before_fill = np.zeros(len(seeds_xyz), dtype=np.int8)
    semantic_before_fill[valid] = clustering["semantic_classes"]
    semantic_after_fill, fill_metadata = fill_invalid_predictions_within_vortex(
        semantic_before_fill, valid, vortex_ids, indices_xyz
    )
    predicted_zyx = np.zeros(segmentation.grid.shape_zyx, dtype=np.int8)
    predicted_zyx[iz, iy, ix] = semantic_after_fill

    topology_config = config["topology_proxy"]
    rows = topology_proxy_rows(
        segmentation.labels_zyx,
        predicted_zyx,
        dominant_min_voxels=int(topology_config["dominant_component_min_voxels"]),
        dominant_min_fraction=float(
            topology_config["dominant_component_min_fraction_of_vortex"]
        ),
    )
    topology_summary = summarize_topology(rows)
    metrics_path = output_dir / "per_vortex_topology_metrics.csv"
    _write_csv(metrics_path, rows)

    full_raw_clusters = np.full(len(seeds_xyz), -1, dtype=np.int8)
    full_raw_clusters[valid] = clustering["raw_clusters"]
    result_path = output_dir / "task4a_clustering_result.npz"
    np.savez_compressed(
        result_path,
        resolution_xyz=np.asarray(segmentation.grid.resolution_xyz),
        domain_min_xyz=segmentation.grid.domain_min_xyz,
        domain_max_xyz=segmentation.grid.domain_max_xyz,
        voxel_indices_xyz=indices_xyz,
        seeds_xyz=seeds_xyz.astype(np.float32),
        vortex_ids=vortex_ids,
        primitive_valid=valid,
        center_streamlines=primitives[:, 0],
        raw_fmt_features_valid=clustering["raw_features"],
        raw_clusters=full_raw_clusters,
        semantic_classes_before_fill=semantic_before_fill,
        semantic_classes_after_fill=semantic_after_fill,
        predicted_classes_zyx=predicted_zyx,
        centerline_absolute_axis_alignment_valid=clustering[
            "centerline_absolute_axis_alignment"
        ],
        seed_vorticity_xyz=seed_vorticity,
        vorticity_absolute_axis_alignment_valid=clustering[
            "vorticity_absolute_axis_alignment"
        ],
    )

    render_metadata = {}
    figure_metadata = {}
    if not args.no_images or args.interactive:
        predicted_segmentation = SegmentationField3D(
            segmentation.grid,
            predicted_zyx,
            name="task4a_streamwise1_spanwise2",
        )
        visual_config = config["visualization"]
        render_metadata = render_voxel_segmentation(
            predicted_segmentation,
            output_dir=None if args.no_images else output_dir / "views",
            views=visual_config["views"],
            surface_mode=str(visual_config["surface_mode"]),
            shrink=float(visual_config["shrink"]),
            opacity=float(visual_config["opacity"]),
            image_size=visual_config["image_size"],
            interactive=bool(args.interactive),
        )
        if not args.no_images:
            figure_metadata = _make_summary_figure(
                output_dir,
                render_metadata["view_paths"],
                clustering,
                rows,
                topology_summary,
            )

    summary = {
        "experiment": config["experiment"],
        "task_scope": config["task"],
        "config": str(config_path.resolve()),
        "gt_selected_array": selected,
        "gt": gt_metadata,
        "flow": flow_metadata,
        "streamlines": {
            "method": streamline_config["method"],
            "periodic_axes": streamline_config["periodic_axes"],
            "spatial_step": spatial_step,
            "steps_per_direction": int(streamline_config["steps_per_direction"]),
            "offset": offset,
            "primitive_shape": list(primitives.shape),
            "valid_primitive_count": int(np.count_nonzero(valid)),
            "valid_primitive_fraction": float(np.mean(valid)),
        },
        "fmt_kmeans": {
            "feature_dimension": clustering["feature_dimension"],
            "fit_seed_count": int(len(clustering["fit_indices"])),
            "valid_cluster_counts_raw": {
                str(cluster): int(np.count_nonzero(clustering["raw_clusters"] == cluster))
                for cluster in (0, 1)
            },
            "streamwise_raw_cluster": clustering["streamwise_raw_cluster"],
            "spanwise_raw_cluster": clustering["spanwise_raw_cluster"],
            "cluster_vorticity_axis_alignment_medians_xyz": clustering[
                "cluster_vorticity_axis_alignment_medians_xyz"
            ].tolist(),
            "cluster_endpoint_axis_alignment_medians_xyz": clustering[
                "cluster_endpoint_axis_alignment_medians_xyz"
            ].tolist(),
            "semantic_naming_boundary": (
                "The local vorticity-axis alignment only names the completed pure-FMT "
                "KMeans partition; it is not included in the clustering features. "
                "Center-streamline endpoint alignment is retained only as a diagnostic."
            ),
        },
        "invalid_fill": fill_metadata,
        "topology_proxy": topology_summary,
        "result_path": str(result_path.resolve()),
        "per_vortex_metrics_path": str(metrics_path.resolve()),
        "render": render_metadata,
        "figure": figure_metadata,
        "interpretation_boundary": (
            "VortexIds provide instance support, not streamwise/spanwise truth. "
            "The 2+1 topology score is a label-free proxy and visual review remains necessary."
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
