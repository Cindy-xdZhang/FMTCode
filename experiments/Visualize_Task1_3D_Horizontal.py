"""Render one clean, horizontal three-panel Task1 figure per 3D flow.

Panel order is fixed to match the separate paper-candidate figures:

1. p97 IVD isosurface, simulation geometry when available, and pathlines;
2. both FMT + KMeans clusters;
3. prediction versus the p95 IVD reference, with the p97 surface shown.

The image deliberately contains no title, legend, or colorbar.  The physical
bounding box, x/y/z labels, and coordinate ticks remain visible in every panel.
All three panels use exactly the same physical bounds and orthographic camera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import Visualize_Task1_3D_PaperCandidates as paper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "Task1_3D_horizontal_clean_1.1"
FIGURE_SIZE_INCHES = (21.0, 5.0)
DEFAULT_DPI = 360
OUTER_MARGIN = 0.001
PANEL_GAP = 0.0
VERTICAL_MARGIN = 0.01
PANEL_ZOOM = 1.12


def _new_horizontal_figure():
    """Create three equal, nearly edge-to-edge 3D viewports."""
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, facecolor="white")
    panel_width = (1.0 - 2.0 * OUTER_MARGIN - 2.0 * PANEL_GAP) / 3.0
    panel_height = 1.0 - 2.0 * VERTICAL_MARGIN
    axes = []
    rectangles = []
    for panel_index in range(3):
        left = OUTER_MARGIN + panel_index * (panel_width + PANEL_GAP)
        rectangle = (left, VERTICAL_MARGIN, panel_width, panel_height)
        axes.append(fig.add_axes(rectangle, projection="3d"))
        rectangles.append(rectangle)
    return fig, axes, rectangles


def _prepare_axis(ax, spec, scene):
    paper._set_physical_axes(ax, scene["bounds"], spec["view"])
    bounds = np.asarray(scene["bounds"], dtype=np.float64)
    span = np.maximum(bounds[1] - bounds[0], 1e-12)
    ax.set_box_aspect(span, zoom=PANEL_ZOOM)
    ax.tick_params(axis="both", which="major", labelsize=7, pad=-1)
    ax.set_xlabel("x", labelpad=-2)
    ax.set_ylabel("y", labelpad=-2)
    ax.set_zlabel("z", labelpad=-2)


def _render_horizontal_scene(spec, scene, output_dir, dpi):
    fig, axes, rectangles = _new_horizontal_figure()
    paper._draw_ivd_pathline_layers(axes[0], scene)
    paper._draw_two_cluster_layers(axes[1], scene)
    paper._draw_prediction_layers(axes[2], scene)
    for ax in axes:
        _prepare_axis(ax, spec, scene)

    path = output_dir / f"{spec['dataset']}_task1_horizontal_clean.png"
    fig.savefig(path, dpi=int(dpi), facecolor="white", edgecolor="none")
    plt.close(fig)
    with Image.open(path) as image:
        pixel_size = list(image.size)
    return path, rectangles, pixel_size


def run(
    output_dir=DEFAULT_OUTPUT,
    dense_grid_size=paper.DEFAULT_DENSE_GRID_SIZE,
    evaluation_ivd_percentile=paper.DEFAULT_EVALUATION_IVD_PERCENTILE,
    display_ivd_percentile=paper.DEFAULT_DISPLAY_IVD_PERCENTILE,
    pathline_count=paper.DEFAULT_PATHLINE_COUNT,
    display_integration_steps=paper.DEFAULT_DISPLAY_INTEGRATION_STEPS,
    vortex_pathline_fraction=paper.DEFAULT_VORTEX_PATHLINE_FRACTION,
    dpi=DEFAULT_DPI,
    datasets=None,
):
    if int(dense_grid_size) ** 3 < 2 * paper.ORIGINAL_SEED_COUNT:
        raise ValueError(
            "dense_grid_size must produce at least twice the original 16^3 seeds"
        )
    if int(pathline_count) < 1:
        raise ValueError("pathline_count must be positive")
    if int(display_integration_steps) <= 48:
        raise ValueError("display_integration_steps must be greater than frozen 48")
    if not 0.0 <= float(vortex_pathline_fraction) <= 1.0:
        raise ValueError("vortex_pathline_fraction must be in [0,1]")
    if int(dpi) < 150:
        raise ValueError("dpi must be at least 150")

    selected = set(datasets or [spec["dataset"] for spec in paper.FLOW_SPECS])
    specs = [spec for spec in paper.FLOW_SPECS if spec["dataset"] in selected]
    known = {spec["dataset"] for spec in paper.FLOW_SPECS}
    if len(specs) != len(selected):
        raise ValueError(f"unknown datasets: {sorted(selected - known)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for spec in specs:
        flow_pathline_count = int(
            pathline_count * spec.get("pathline_count_multiplier", 1)
        )
        flow_integration_steps = int(
            display_integration_steps
            * spec.get("integration_steps_multiplier", 1)
        )
        fitted, vortex_cluster, choice = paper._fit_frozen_clusterer(spec, device)
        scene = paper._dense_scene(
            spec,
            fitted,
            vortex_cluster,
            choice,
            int(dense_grid_size),
            float(evaluation_ivd_percentile),
            float(display_ivd_percentile),
            flow_pathline_count,
            flow_integration_steps,
            float(vortex_pathline_fraction),
            device,
        )
        path, rectangles, pixel_size = _render_horizontal_scene(
            spec, scene, output_dir, int(dpi)
        )
        geometry = scene["simulation_geometry"]
        records.append(
            {
                "dataset": spec["dataset"],
                "title": spec["title"],
                "source_index": int(scene["metadata"]["source_start_index"]),
                "source_time": float(scene["metadata"]["source_time"]),
                "image": str(path),
                "pixel_size": pixel_size,
                "panel_rectangles": [list(rectangle) for rectangle in rectangles],
                "panel_order": [
                    "IVD p97 + geometry + pathlines",
                    "FMT + KMeans two clusters",
                    "prediction versus IVD p95; IVD p97 surface shown",
                ],
                "camera": {
                    "projection": "orthographic",
                    "elevation_degrees": float(spec["view"][0]),
                    "azimuth_degrees": float(spec["view"][1]),
                    "physical_bounds": scene["bounds"].tolist(),
                    "box_aspect": (
                        scene["bounds"][1] - scene["bounds"][0]
                    ).tolist(),
                    "panel_zoom": PANEL_ZOOM,
                },
                "display_pathlines": {
                    key: value
                    for key, value in scene["display_pathline_metadata"].items()
                    if key != "load_metadata"
                },
                "simulation_geometry": (
                    None if geometry is None else geometry["metadata"]
                ),
                "metrics_against_ivd_p95": {
                    key: float(value) for key, value in scene["metrics"].items()
                },
            }
        )
        print(
            f"{spec['dataset']}: {pixel_size[0]}x{pixel_size[1]}, "
            f"{flow_pathline_count} paths, {flow_integration_steps} steps, "
            f"F1={scene['metrics']['f1']:.3f}",
            flush=True,
        )

    payload = {
        "name": "Task1_3D_horizontal_clean_1.1",
        "layout": "one 1x3 horizontal image per flow",
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "dpi": int(dpi),
        "panel_gap": PANEL_GAP,
        "annotation_policy": {
            "retained_data_layers": [
                "IVD isosurfaces",
                "pathlines",
                "simulation geometry",
                "cluster points",
                "prediction/error points",
            ],
            "retained_coordinate_frame": [
                "3D bounding box",
                "x/y/z axis labels",
                "coordinate ticks",
            ],
            "removed": [
                "figure titles",
                "panel titles",
                "legends",
                "time colorbar",
            ],
        },
        "shared_camera_within_each_flow": True,
        "device": device,
        "flows": records,
    }
    metadata_path = output_dir / "figure_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dense-grid-size", type=int, default=paper.DEFAULT_DENSE_GRID_SIZE
    )
    parser.add_argument(
        "--evaluation-ivd-percentile",
        type=float,
        default=paper.DEFAULT_EVALUATION_IVD_PERCENTILE,
    )
    parser.add_argument(
        "--display-ivd-percentile",
        type=float,
        default=paper.DEFAULT_DISPLAY_IVD_PERCENTILE,
    )
    parser.add_argument(
        "--pathline-count", type=int, default=paper.DEFAULT_PATHLINE_COUNT
    )
    parser.add_argument(
        "--display-integration-steps",
        type=int,
        default=paper.DEFAULT_DISPLAY_INTEGRATION_STEPS,
    )
    parser.add_argument(
        "--vortex-pathline-fraction",
        type=float,
        default=paper.DEFAULT_VORTEX_PATHLINE_FRACTION,
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Optional subset: cylinder3d halfcylinderRe640 tangaroa "
            "deltaWing_LBM boeing747 smokeBuoyancy"
        ),
    )
    args = parser.parse_args()
    run(
        args.output_dir,
        args.dense_grid_size,
        args.evaluation_ivd_percentile,
        args.display_ivd_percentile,
        args.pathline_count,
        args.display_integration_steps,
        args.vortex_pathline_fraction,
        args.dpi,
        args.datasets,
    )
