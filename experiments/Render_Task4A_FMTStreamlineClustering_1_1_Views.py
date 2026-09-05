"""Render Task4-a two-class cube views from a saved clustering NPZ.

The Ibex reproduction of Other_Task4A_FMTStreamlineClustering_1.1 runs with
``--no-images``.  This helper rebuilds the voxel grid from the NPZ metadata and
renders the same streamwise(1)/spanwise(2) cube views that the runner would
have produced, using the frozen visualization block of the 1.1 config.  It
performs no clustering and changes no numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from FMT_Utils.VoxelSegmentation_3D import (
    SegmentationField3D,
    VoxelGrid3D,
    render_voxel_segmentation,
)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="task4a_clustering_result.npz")
    parser.add_argument(
        "--config", default="config/Other_Task4A_FMTStreamlineClustering_1.1.yaml"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visual = config["visualization"]
    with np.load(args.result) as data:
        grid = VoxelGrid3D(
            tuple(int(v) for v in data["resolution_xyz"]),
            data["domain_min_xyz"],
            data["domain_max_xyz"],
        )
        predicted_zyx = np.asarray(data["predicted_classes_zyx"], dtype=np.int8)
    if predicted_zyx.shape != grid.shape_zyx:
        raise ValueError(f"label shape {predicted_zyx.shape} != grid {grid.shape_zyx}")
    segmentation = SegmentationField3D(grid, predicted_zyx, name="task4a_streamwise1_spanwise2")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = render_voxel_segmentation(
        segmentation,
        output_dir=output_dir,
        views=visual["views"],
        surface_mode=str(visual["surface_mode"]),
        shrink=float(visual["shrink"]),
        opacity=float(visual["opacity"]),
        image_size=visual["image_size"],
        interactive=bool(args.interactive),
    )
    metadata["result"] = str(Path(args.result).resolve())
    metadata["class_counts"] = {
        str(k): int(np.count_nonzero(predicted_zyx == k)) for k in (0, 1, 2)
    }
    (output_dir / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: metadata[k] for k in ("class_counts", "view_paths") if k in metadata}, indent=2))
    return metadata


if __name__ == "__main__":
    main()
