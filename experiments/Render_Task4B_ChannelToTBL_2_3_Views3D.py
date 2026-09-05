"""Three-dimensional cube renderings of the channel->TBL 2.3 four-class transfer.

The formal 2.3 visualizer draws deterministic 2D z-buffer projections.  This
helper renders the same audited predictions as 3D shrunken-cube glyphs with the
VTK renderer used for Task4-a, for the proxy target and the four variants of one
optimizer seed.  Two populations are drawn: all valid TBL candidate cubes and
the manual-hairpin-support subset (``target_vortex_ids > 0``).  It reads only
existing prediction NPZ files and the target-cache manifest; it changes no
numbers and performs no selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import FMT_Utils.VoxelSegmentation_3D as voxel_module
from FMT_Utils.VoxelSegmentation_3D import (
    SegmentationField3D,
    VoxelGrid3D,
    render_voxel_segmentation,
)

VARIANTS = ("raw", "raw_wide", "fmt_only", "raw_fmt")
PANEL_TITLES = {
    "proxy": "Proxy target",
    "raw": "Raw",
    "raw_wide": "Raw-wide",
    "fmt_only": "FMT-only",
    "raw_fmt": "Raw+FMT",
}
CLASS_LABELS = (
    "Ordinary streamwise",
    "Ordinary spanwise",
    "Hairpin head",
    "Hairpin limb",
)
# Identical to the frozen Task4-b 1.2 / 2.3 visualization palette.
CLASS_COLORS = ("#0072B2", "#E69F00", "#CC79A7", "#009E73")
DEFAULT_OUTPUT_DIR = Path("outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs")


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _class_lookup_table(max_label: int):
    """Fixed four-class palette: label k in 1..4 maps to CLASS_COLORS[k-1]."""
    import vtk

    table = vtk.vtkLookupTable()
    table.SetNumberOfTableValues(5)
    table.SetTableRange(0.0, 4.0)
    table.Build()
    table.SetTableValue(0, 0.65, 0.65, 0.65, 0.0)
    for label, color in enumerate(CLASS_COLORS, start=1):
        red, green, blue = _hex_to_rgb(color)
        table.SetTableValue(label, red, green, blue, 1.0)
    return table


def _load(output_dir: Path, seed: int) -> dict:
    manifest_path = output_dir / "target_cache" / "target_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grid_meta = manifest["target_grid"]
    grid = VoxelGrid3D(
        tuple(int(v) for v in grid_meta["resolution_xyz"]),
        np.asarray(grid_meta["domain_min_xyz"], dtype=np.float64),
        np.asarray(grid_meta["domain_max_xyz"], dtype=np.float64),
    )
    labels: dict[str, np.ndarray] = {}
    voxels = None
    vortex_ids = None
    for variant in VARIANTS:
        path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
        with np.load(path) as data:
            current_voxels = np.asarray(data["target_voxel_indices_xyz"], dtype=np.int64)
            current_ids = np.asarray(data["target_vortex_ids"], dtype=np.int32)
            targets = np.asarray(data["target_targets"], dtype=np.int64)
            predicted = np.asarray(data["target_predicted_labels"], dtype=np.int64)
        if voxels is None:
            voxels, vortex_ids = current_voxels, current_ids
            labels["proxy"] = targets
        else:
            same = (
                np.array_equal(voxels, current_voxels)
                and np.array_equal(vortex_ids, current_ids)
                and np.array_equal(labels["proxy"], targets)
            )
            if not same:
                raise RuntimeError(f"{variant}: target rows differ between prediction files")
        labels[variant] = predicted
    if not np.array_equal(vortex_ids > 0, labels["proxy"] >= 2):
        raise RuntimeError("manual hairpin support differs from proxy head/limb support")
    resolution = np.asarray(grid.resolution_xyz)[None, :]
    if not np.all((voxels >= 0) & (voxels < resolution)):
        raise RuntimeError("target voxel indices outside the grid")
    return {"grid": grid, "voxels": voxels, "vortex_ids": vortex_ids, "labels": labels}


def _cropped_grid(grid, voxels, mask, margin: int):
    """Sub-grid covering the occupied cubes plus a margin; returns (grid, offset)."""
    sel = voxels[mask]
    resolution = np.asarray(grid.resolution_xyz, dtype=np.int64)
    low = np.maximum(sel.min(axis=0) - margin, 0)
    high = np.minimum(sel.max(axis=0) + margin + 1, resolution)
    size = grid.voxel_size_xyz
    sub = VoxelGrid3D(
        tuple(int(v) for v in (high - low)),
        grid.domain_min_xyz + low * size,
        grid.domain_min_xyz + high * size,
    )
    return sub, low


def _segmentation(grid, voxels, labels, mask, name, crop_margin=None):
    offset = np.zeros(3, dtype=np.int64)
    if crop_margin is not None:
        grid, offset = _cropped_grid(grid, voxels, mask, int(crop_margin))
    field = np.zeros(grid.shape_zyx, dtype=np.int8)
    sel = voxels[mask] - offset[None, :]
    field[sel[:, 2], sel[:, 1], sel[:, 0]] = (labels[mask] + 1).astype(np.int8)
    return SegmentationField3D(grid, field, name=name)


def _montage(view_paths: dict[str, str], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
    for ax, key in zip(axes, ("proxy",) + VARIANTS):
        ax.imshow(plt.imread(view_paths[key]))
        weight = "bold" if key == "raw_fmt" else "normal"
        ax.set_title(PANEL_TITLES[key], fontsize=13, fontweight=weight)
        ax.axis("off")
    handles = [
        Patch(facecolor=color, edgecolor="0.2", label=label)
        for color, label in zip(CLASS_COLORS, CLASS_LABELS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=7068)
    parser.add_argument("--views", nargs="+", default=["isometric", "+y", "+z"])
    parser.add_argument("--image-size", nargs=2, type=int, default=[1600, 1000])
    parser.add_argument("--shrink", type=float, default=0.92)
    parser.add_argument("--figure-dir")
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=2,
        help="Crop the rendered box to the occupied cubes plus this many cubes; "
        "negative disables cropping and draws the full target grid.",
    )
    parser.add_argument(
        "--iso-zoom",
        type=float,
        default=1.0,
        help="Extra camera zoom applied to the isometric view only.",
    )
    parser.add_argument(
        "--x-window",
        nargs=2,
        type=int,
        metavar=("X_LOW", "X_HIGH"),
        help="Restrict both populations to cube x-indices in [X_LOW, X_HIGH) for a close-up.",
    )
    parser.add_argument(
        "--densest-x-window",
        type=int,
        help="Choose the x-window of this many cubes that holds the most manual-hairpin cubes.",
    )
    parser.add_argument("--tag", default="", help="Suffix appended to the figure directory name.")
    args = parser.parse_args()
    crop_margin = None if args.crop_margin < 0 else args.crop_margin
    if args.iso_zoom != 1.0:
        original_set_camera = voxel_module._set_camera

        def _zoomed_set_camera(renderer, grid, view):
            original_set_camera(renderer, grid, view)
            if view == "isometric":
                renderer.GetActiveCamera().Zoom(float(args.iso_zoom))
                renderer.ResetCameraClippingRange()

        voxel_module._set_camera = _zoomed_set_camera

    output_dir = Path(args.output_dir)
    default_figure_dir = (
        output_dir / "figures" / f"task4b_channel_to_tbl_2.3_seed{args.seed}_3d"
    )
    if args.tag:
        default_figure_dir = default_figure_dir.with_name(default_figure_dir.name + f"_{args.tag}")
    figure_dir = Path(args.figure_dir or default_figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    # Fixed four-class palette instead of the renderer's golden-angle hues.
    voxel_module._make_lookup_table = _class_lookup_table

    loaded = _load(output_dir, args.seed)
    grid = loaded["grid"]
    voxels = loaded["voxels"]
    ids = loaded["vortex_ids"]
    labels = loaded["labels"]
    window_mask = np.ones(len(voxels), dtype=bool)
    x_window = None
    if args.densest_x_window:
        width = int(args.densest_x_window)
        counts = np.bincount(voxels[ids > 0, 0], minlength=grid.resolution_xyz[0])
        sums = np.convolve(counts, np.ones(width, dtype=np.int64), mode="valid")
        start = int(np.argmax(sums))
        x_window = (start, start + width)
    if args.x_window:
        x_window = (int(args.x_window[0]), int(args.x_window[1]))
    if x_window is not None:
        window_mask = (voxels[:, 0] >= x_window[0]) & (voxels[:, 0] < x_window[1])
    populations = {
        "all_candidates": window_mask.copy(),
        "true_hairpin_support": window_mask & (ids > 0),
    }
    metadata = {
        "seed": args.seed,
        "grid": {
            "resolution_xyz": list(grid.resolution_xyz),
            "domain_min_xyz": grid.domain_min_xyz.tolist(),
            "domain_max_xyz": grid.domain_max_xyz.tolist(),
        },
        "class_colors": CLASS_COLORS,
        "crop_margin_cubes": crop_margin,
        "iso_zoom": args.iso_zoom,
        "x_window_cube_indices": list(x_window) if x_window is not None else None,
        "x_window_rule": (
            "densest manual-hairpin window" if args.densest_x_window else
            ("explicit" if args.x_window else None)
        ),
        "populations": {},
    }
    for pop_name, mask in populations.items():
        pop_meta = {"cube_count": int(mask.sum()), "renders": {}, "montages": {}}
        view_paths: dict[str, dict[str, str]] = {view: {} for view in args.views}
        for key in ("proxy",) + VARIANTS:
            seg = _segmentation(
                grid, voxels, labels[key], mask, f"tbl_{pop_name}_{key}", crop_margin
            )
            render = render_voxel_segmentation(
                seg,
                output_dir=figure_dir / pop_name / key,
                views=args.views,
                surface_mode="interfaces",
                shrink=args.shrink,
                opacity=1.0,
                image_size=args.image_size,
                interactive=False,
            )
            for view, path in zip(args.views, render["view_paths"]):
                view_paths[view][key] = path
            counts = np.bincount(labels[key][mask], minlength=4).tolist()
            pop_meta["renders"][key] = {
                "class_counts": counts,
                "rendered_cubes": int(render["rendered_cube_count"]),
            }
        for view in args.views:
            safe = view.replace("+", "plus_").replace("-", "minus_")
            path = figure_dir / f"task4b_tbl_3d_{pop_name}_seed{args.seed}_{safe}.png"
            title = (
                f"TBL 3D cube segmentation, {pop_name.replace('_', ' ')}, "
                f"view {view}, seed {args.seed}"
            )
            _montage(view_paths[view], title, path)
            pop_meta["montages"][view] = str(path)
        metadata["populations"][pop_name] = pop_meta
    (figure_dir / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    summary = {
        pop: {key: value["class_counts"] for key, value in meta["renders"].items()}
        for pop, meta in metadata["populations"].items()
    }
    print(json.dumps(summary, indent=1))
    print("figure_dir:", figure_dir)


if __name__ == "__main__":
    main()
