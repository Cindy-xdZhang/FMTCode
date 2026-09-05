"""Publication-style comparison of geometric KMeans and velocity--curl proxy.

Figure contract
---------------
Core conclusion: time-parameterized streamline-cross geometry reproduces the
velocity--curl proxy for most manually annotated hairpin instances.
Results-level question: does geometry-only KMeans reach the frozen 0.90/0.10
proxy-agreement gate, and where does it fail?
Archetype: image plate plus quantitative validation.
Panel a: geometry KMeans spatial segmentation (hero evidence).
Panel b: frozen velocity--curl proxy under the identical camera (control).
Panel c: per-VortexId macro-F1 distribution and explicit failure boundary.
Panel d: time-step/cross-offset robustness, maximized only over the frozen
margin-temperature candidates.
Backend/export: Python; 7.2 x 6.6 inch; PNG/PDF/SVG/TIFF; editable text.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import numpy as np

from FMT_Utils.VoxelSegmentation_3D import (
    SegmentationField3D,
    VoxelGrid3D,
    render_voxel_segmentation,
)


DEFAULT_RESULT = Path(
    "outputs/Verify_Task4A_GeometricCurlKMeans_2.2/channel_GTs/"
    "geometric_curl_kmeans_result.npz"
)
DEFAULT_SUMMARY = Path(
    "outputs/Verify_Task4A_GeometricCurlKMeans_2.2/channel_GTs/summary.json"
)


def _class_color(label: int) -> tuple[float, float, float]:
    hue = (0.6180339887498949 * max(1, int(label))) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.68, 0.95)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> dict:
    args = _parse_args()
    result_path = Path(args.result)
    summary_path = Path(args.summary)
    source = np.load(result_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir or result_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = VoxelGrid3D(
        tuple(int(value) for value in source["resolution_xyz"]),
        source["domain_min_xyz"],
        source["domain_max_xyz"],
    )
    indices = np.asarray(source["voxel_indices_xyz"], dtype=np.int32)
    ix, iy, iz = indices.T
    geometry_volume = np.zeros(grid.shape_zyx, dtype=np.int8)
    proxy_volume = np.zeros(grid.shape_zyx, dtype=np.int8)
    geometry_volume[iz, iy, ix] = source["geometry_kmeans_classes"]
    proxy_volume[iz, iy, ix] = source["proxy_classes"]

    common_render = {
        "views": ["isometric", "+y"],
        "surface_mode": "interfaces",
        "shrink": 0.92,
        "opacity": 1.0,
        "image_size": [1400, 900],
        "interactive": False,
    }
    geometry_render = render_voxel_segmentation(
        SegmentationField3D(grid, geometry_volume, "geometric_kmeans"),
        output_dir=output_dir / "views" / "geometric_kmeans",
        **common_render,
    )
    proxy_render = render_voxel_segmentation(
        SegmentationField3D(grid, proxy_volume, "velocity_curl_proxy"),
        output_dir=output_dir / "views" / "proxy",
        **common_render,
    )
    geometry_isometric = next(
        Path(path) for path in geometry_render["view_paths"]
        if Path(path).stem.endswith("isometric")
    )
    proxy_isometric = next(
        Path(path) for path in proxy_render["view_paths"]
        if Path(path).stem.endswith("isometric")
    )

    metric_rows = np.genfromtxt(
        output_dir / "per_vortex_geometry_kmeans_metrics.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    search_rows = np.genfromtxt(
        output_dir / "parameter_search.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    phase1 = search_rows[search_rows["phase"] == "phase1_geometry"]
    time_scales = np.unique(
        phase1["time_step_mean_voxel_over_median_speed_scale"]
    )
    offset_scales = np.asarray([0.1, 0.25, 0.5, 1.0])
    heatmap = np.full((len(time_scales), len(offset_scales)), np.nan)
    for row_index, time_scale in enumerate(time_scales):
        for column_index, offset_scale in enumerate(offset_scales):
            select = (
                np.isclose(
                    phase1["time_step_mean_voxel_over_median_speed_scale"],
                    time_scale,
                )
                & np.isclose(
                    phase1["cross_offset_min_voxel_scale"], offset_scale
                )
            )
            if np.any(select):
                heatmap[row_index, column_index] = float(np.max(
                    phase1["geometry_named_kmeans_macro_f1"][select]
                ))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.09, top=0.95,
        wspace=0.23, hspace=0.31,
    )
    panels = axes.ravel()
    for ax, image_path, title in (
        (panels[0], geometry_isometric, "Streamline-geometry KMeans"),
        (panels[1], proxy_isometric, "Velocity–curl proxy"),
    ):
        ax.imshow(plt.imread(image_path), aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    panels[0].legend(
        handles=[
            Patch(facecolor=_class_color(1), label="Streamwise / parallel"),
            Patch(facecolor=_class_color(2), label="Spanwise / perpendicular"),
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
    )

    order = np.argsort(metric_rows["macro_f1"])
    sorted_f1 = metric_rows["macro_f1"][order]
    sorted_ids = metric_rows["vortex_id"][order]
    passed = sorted_f1 >= 0.90
    panels[2].axhline(0.90, color="0.35", linestyle="--", linewidth=1.0)
    panels[2].scatter(
        np.flatnonzero(passed), sorted_f1[passed],
        s=18, color="#2f6f9f", alpha=0.85, label="F1 ≥ 0.90",
    )
    panels[2].scatter(
        np.flatnonzero(~passed), sorted_f1[~passed],
        s=25, color="#a33a2b", marker="x", label="F1 < 0.90",
    )
    panels[2].set(
        xlabel="Hairpin rank by proxy macro-F1",
        ylabel="Proxy macro-F1",
        xlim=(-2, len(sorted_f1) + 1),
        ylim=(0.68, 1.01),
        title=(
            "Per-instance agreement  "
            f"({int(np.count_nonzero(passed))}/{len(sorted_f1)} ≥ 0.90)"
        ),
    )
    panels[2].legend(loc="lower right", frameon=False)

    image = panels[3].imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap="Blues",
        vmin=0.89,
        vmax=0.945,
    )
    del image
    for row_index in range(len(time_scales)):
        for column_index in range(len(offset_scales)):
            value = heatmap[row_index, column_index]
            if np.isfinite(value):
                panels[3].text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > 0.923 else "0.12",
                )
            else:
                panels[3].text(
                    column_index,
                    row_index,
                    "invalid",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="0.35",
                )
    panels[3].set_xticks(np.arange(len(offset_scales)))
    panels[3].set_xticklabels([f"{value:g}" for value in offset_scales])
    panels[3].set_yticks(np.arange(len(time_scales)))
    panels[3].set_yticklabels([f"{value:g}" for value in time_scales])
    panels[3].set(
        xlabel="Cross offset / minimum cube size",
        ylabel="Time step / reference step",
        title="Parameter robustness (cell value: best macro-F1)",
    )
    for spine in panels[3].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    for label, ax in zip("abcd", panels):
        offset = ScaledTranslation(-9 / 72, 5 / 72, fig.dpi_scale_trans)
        ax.text(
            0,
            1,
            label,
            transform=ax.transAxes + offset,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    fig.canvas.draw()
    qa_dir = output_dir / "qa_geometric_curl_kmeans"
    qa_dir.mkdir(parents=True, exist_ok=True)
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    alignment = require_matplotlib_panel_alignment(
        fig,
        json_out=qa_dir / "panel_alignment.json",
        overlay_svg=qa_dir / "panel_alignment_overlay.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    paths = {
        "png": output_dir / "task4a_geometric_curl_kmeans_comparison.png",
        "pdf": output_dir / "task4a_geometric_curl_kmeans_comparison.pdf",
        "svg": output_dir / "task4a_geometric_curl_kmeans_comparison.svg",
        "tiff": output_dir / "task4a_geometric_curl_kmeans_comparison.tiff",
    }
    fig.savefig(paths["png"], dpi=600)
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    fig.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    metadata = {
        "figure_contract": {
            "core_conclusion": (
                "Time-parameterized streamline-cross geometry reproduces the "
                "velocity-curl proxy for most hairpin instances."
            ),
            "archetype": "image_plate_plus_quantitative_validation",
            "backend": "python",
            "final_size_inches": [7.2, 6.6],
        },
        "geometry_render": geometry_render,
        "proxy_render": proxy_render,
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "panel_alignment_verdict": alignment.get("verdict"),
        "panel_alignment_report": str((qa_dir / "panel_alignment.json").resolve()),
        "source_result": str(result_path.resolve()),
        "source_summary": str(summary_path.resolve()),
        "reported_macro_f1": summary["winner"]["geometry_named_kmeans_macro_f1"],
    }
    metadata_path = output_dir / "visualization_summary.json"
    metadata["metadata_path"] = str(metadata_path.resolve())
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


if __name__ == "__main__":
    main()
