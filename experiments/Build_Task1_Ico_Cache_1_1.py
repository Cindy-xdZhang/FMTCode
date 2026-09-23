"""Build 13-line icosahedral primitive caches for the Task-1 clustering study.

Unsupervised throughout: there is **no train/test split and no development /
confirmation distinction**.  Every timeslice is used for training, and the IVD
reference is written to the cache as a *metric only* -- it never enters a loss,
a checkpoint choice, a read-out rule or a hyper-parameter.

Geometry matches the octahedral pipeline exactly except for the star: same
seeding grid, same RK4 schedule, same offset radius, same validity rule.  Only
`integrate_cross_primitives_3d` (7 lines) is replaced by
`integrate_icosahedral_primitives_3d` (13 lines).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.FMT_3D_pipeline import compute_ivd_reference_3d, generate_seeding_grid_3d
from FMT_Utils.IcosahedralPrimitives_3D import integrate_icosahedral_primitives_3d
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d

DEFAULT_DATA_ROOT = "/home/cheny1a/data/flowData3D"
DATASETS = {
    "halfcylinderRe160_eth": "halfcylinderRe160.nc",
    "halfcylinderRe640_eth": "halfcylinderRe640.nc",
    "halfcylinderRe6400_eth": "halfcylinderRe6400.nc",
    "tangaroa_eth": "tangaroa.nc",
    "deltawing_eth": "deltaWing_mag0_3reesampled.nc",
}
# The union of what the split study called development and confirmation: with no
# split there is no reason to leave a third of the data unused.
ORDINALS = {
    "halfcylinderRe160_eth": [31, 39, 46, 54, 62, 69, 77, 85, 92, 100, 114, 121, 127, 134],
    "halfcylinderRe640_eth": [31, 39, 46, 54, 62, 69, 77, 85, 92, 100, 114, 121, 127, 134],
    "halfcylinderRe6400_eth": [31, 39, 46, 54, 62, 69, 77, 85, 92, 100, 114, 121, 127, 134],
    "tangaroa_eth": [41, 52, 63, 74, 85, 96, 107, 118, 129, 140, 154, 162, 171, 179],
    "deltawing_eth": [34, 42, 50, 58, 67, 75, 83, 91, 99, 107, 116, 124, 132, 140],
}
FRAME_COUNT = 14
MAX_SPATIAL_DIM = 96
GRID_SHAPE = (16, 16, 16)
BOUNDARY_FRACTION = 0.08
DT_SCALE = 0.25
INTEGRATION_STEPS = 48
SAMPLED_STEPS = 32
OFFSET_GRID_SCALE = 0.5
IVD_PERCENTILE = 95.0


def build_slice(path, start_index, ordinal, destination):
    started = time.time()
    field, window = load_netcdf_window_3d(path, start_index, FRAME_COUNT, MAX_SPATIAL_DIM)
    dt = float(field.timeInterval) * DT_SCALE
    if dt <= 0:
        raise ValueError("field must have at least two distinct time samples")
    spacing = np.asarray(field.gridInterval)
    offset = float(np.min(spacing[spacing > 0])) * OFFSET_GRID_SCALE
    seeds, _ = generate_seeding_grid_3d(field, GRID_SHAPE, BOUNDARY_FRACTION, offset)
    seed_time = float(field.tmin)

    primitives, valid, lengths = integrate_icosahedral_primitives_3d(
        field, seeds, seed_time, dt, INTEGRATION_STEPS, SAMPLED_STEPS, offset)
    seeds_valid = seeds[valid]
    volume, sampled, _ = compute_ivd_reference_3d(field, seed_time, seeds_valid)
    threshold = float(np.percentile(volume, IVD_PERCENTILE))
    reference = sampled >= threshold

    geometry = np.ascontiguousarray(primitives[..., :3].astype(np.float32))
    metadata = {
        "dataset": destination.parent.name, "ordinal": ordinal,
        "source_path": str(path), "source_start_index": int(start_index),
        "frame_count": FRAME_COUNT, **{k: window[k] for k in
            ("loaded_shape_TZYXC", "spatial_strides", "source_time", "source_time_step")},
        "line_count": int(geometry.shape[1]), "sampled_steps": SAMPLED_STEPS,
        "primitive_offset": offset, "primitive_offset_mode": "min",
        "star": "icosahedron_12_vertices",
        "total_primitives": int(len(seeds)), "valid_primitives": int(valid.sum()),
        "ivd_threshold": threshold, "ivd_positive_count": int(reference.sum()),
        "ivd_positive_fraction": float(reference.mean()),
        "reference_role": "metric_only_never_used_for_training_or_selection",
        "elapsed_seconds": time.time() - started,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, geometry=geometry, seeds=seeds_valid.astype(np.float32),
                        reference=reference, valid_mask=valid, line_lengths=lengths,
                        metadata_json=json.dumps(metadata))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="halfcylinderRe160_eth,halfcylinderRe640_eth,halfcylinderRe6400_eth")
    parser.add_argument("--output", default="outputs/exp_Task1_IcoVAE_cache")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="directory holding the .nc files named in DATASETS")
    parser.add_argument("--limit", type=int, default=0, help="first N ordinals only (smoke test)")
    arguments = parser.parse_args()

    root = ROOT / arguments.output
    for name in arguments.datasets.split(","):
        ordinals = ORDINALS[name]
        if arguments.limit:
            ordinals = ordinals[:arguments.limit]
        for position, ordinal in enumerate(ordinals):
            target = root / name / f"slice_{position:02d}_index_{ordinal:04d}.npz"
            if target.exists():
                print(f"  {name} ordinal {ordinal}: cached", flush=True)
                continue
            meta = build_slice(Path(arguments.data_root) / DATASETS[name], ordinal, ordinal, target)
            print(f"  {name} ordinal {ordinal:4d}: {meta['valid_primitives']:5d}/"
                  f"{meta['total_primitives']} valid, positives "
                  f"{meta['ivd_positive_fraction']:.4f}, {meta['elapsed_seconds']:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
