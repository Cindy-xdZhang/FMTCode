"""Load manual VTK vortex-instance labels and render an independent 3D cube grid.

This is the standalone, non-PyflowVis entry point for Other_Task4VoxelGT_1.1.
It can export repeatable camera views, open an interactive trackball-camera
window, or do both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from FMT_Utils.VoxelSegmentation_3D import (
    load_vtk_segmentation,
    render_voxel_segmentation,
)


DEFAULT_CONFIG = Path("config/Other_Task4VoxelGT_1.1.yaml")


def _resolve_input(config: dict, command_line_input: str | None) -> Path:
    if command_line_input:
        path = Path(command_line_input)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates = config.get("input", {}).get("path_candidates", [])
    existing = next((Path(value) for value in candidates if Path(value).is_file()), None)
    if existing is None:
        raise FileNotFoundError(
            "no channel_GTs.vtk input was found; pass --input or update "
            f"input.path_candidates in the config (checked {candidates})"
        )
    return existing.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", help="VTK/VTI/VTU/VTS/VTR/VTP label dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--label-array")
    parser.add_argument("--association", choices=("point", "cell"))
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="explicit independent voxel resolution; overrides the preview budget",
    )
    parser.add_argument(
        "--native-resolution",
        action="store_true",
        help="use the full source-fidelity suggestion without the preview voxel cap",
    )
    parser.add_argument("--max-voxels", type=int)
    parser.add_argument(
        "--surface-mode", choices=("outer", "interfaces", "all")
    )
    parser.add_argument(
        "--label",
        type=int,
        action="append",
        help="render only this positive VortexIds value; repeat for several labels",
    )
    parser.add_argument("--shrink", type=float)
    parser.add_argument("--opacity", type=float)
    parser.add_argument(
        "--view",
        action="append",
        choices=("isometric", "+x", "-x", "+y", "-y", "+z", "-z"),
        help="camera view to save; repeat for several views",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="do not save PNG views (normally used with --interactive)",
    )
    return parser.parse_args()


def main() -> dict:
    args = _parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_path = _resolve_input(config, args.input)
    input_config = config.get("input", {})
    sampling_config = config.get("sampling", {})
    visual_config = config.get("visualization", {})

    if args.resolution is not None and args.native_resolution:
        raise ValueError("--resolution and --native-resolution are mutually exclusive")
    if args.resolution is not None:
        resolution = tuple(args.resolution)
        max_voxels = None
    else:
        resolution = sampling_config.get("resolution_xyz")
        max_voxels = (
            None
            if args.native_resolution
            else (
                args.max_voxels
                if args.max_voxels is not None
                else sampling_config.get("preview_max_voxels", 2_000_000)
            )
        )

    label_array = args.label_array or input_config.get("label_array", "VortexIds")
    association = args.association or input_config.get("association", "cell")
    segmentation, load_metadata, selected = load_vtk_segmentation(
        input_path,
        resolution_xyz=resolution,
        array_name=label_array,
        association=association,
        max_voxels=max_voxels,
        max_axis_resolution=int(sampling_config.get("max_axis_resolution", 1024)),
    )

    output_dir = Path(
        args.output_dir
        or visual_config.get(
            "output_dir", f"outputs/Other_Task4VoxelGT_1.1/{input_path.stem}"
        )
    )
    views = args.view or visual_config.get(
        "views", ["isometric", "+x", "+y", "+z"]
    )
    surface_mode = args.surface_mode or visual_config.get("surface_mode", "outer")
    shrink = args.shrink if args.shrink is not None else visual_config.get("shrink", 0.92)
    opacity = (
        args.opacity if args.opacity is not None else visual_config.get("opacity", 1.0)
    )
    image_size = visual_config.get("image_size", [1200, 900])
    render_metadata = render_voxel_segmentation(
        segmentation,
        output_dir=None if args.no_images else output_dir,
        views=views,
        surface_mode=surface_mode,
        label_filter=args.label,
        shrink=shrink,
        opacity=opacity,
        image_size=image_size,
        interactive=args.interactive,
    )

    summary = {
        "experiment": str(config.get("experiment", "Other_Task4VoxelGT_1.1")),
        "task_scope": "Task4 preparation: manual vortex-instance segmentation viewing",
        "config": str(config_path.resolve()),
        "selected_array": selected,
        **load_metadata,
        **render_metadata,
        "semantic_boundary": (
            "VortexIds are positive vortex-instance identifiers. This file does not "
            "provide streamwise/spanwise/hairpin class names, so it is not yet a "
            "Task4 class-label mapping."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "voxel_segmentation_summary.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()

