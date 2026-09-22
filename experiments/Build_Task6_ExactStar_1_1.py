"""Exact traced icosahedral stars for Task-6 scenes whose field is available.

`deltaWing_mag0_3reesampled.nc` is verified to be the field behind Task 6's
`deltaWing_resampled` scene (median cos 0.998 between stored streamline tangents
and the interpolated field).  Where the field is available the 12 neighbours can
be *traced* from `centre + r d_j` instead of borrowed from nearby seeds, which
removes both the 12-15 degree angular error and the duplicate-neighbour problem,
and -- the practical point -- makes small radii usable at all.

Labels, seeds and the train/test frame split come from the dataset unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.IcosahedralGroup_3D import icosahedron_vertices
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from FMT_Utils.StreamlineStar_3D import icosahedral_star

SOURCE = Path("/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share")
FIELDS = {"deltaWing_resampled": "/home/cheny1a/data/flowData3D/deltaWing_mag0_3reesampled.nc"}


def build_frame(folder, field_path, count, seed, radius_scale, destination, chunk=512,
                steps=65, length=0.5):
    started = time.time()
    meta = json.loads((folder / "frame.json").read_text())
    seeds = np.load(folder / "seeds.npy")
    labels = np.load(folder / "labels.npy").astype(np.int64)
    field, _ = load_netcdf_window_3d(field_path, meta["index"], 1, 640)
    frame = field.field[0]
    lower = np.asarray(field.domainMinBoundary); upper = np.asarray(field.domainMaxBoundary)

    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(seeds), min(count, len(seeds)), replace=False))
    vertices = icosahedron_vertices()
    radius = radius_scale * meta["h"]
    parts = []
    for start in range(0, len(sample), chunk):
        parts.append(icosahedral_star(frame, seeds[sample[start:start + chunk]].astype(np.float64),
                                      vertices, radius, lower, upper, steps=steps, length=length))
    geometry = np.concatenate(parts).astype(np.float32)

    record = {"flow": meta["flow"], "index": meta["index"], "key": meta["key"], "h": meta["h"],
              "steps": int(geometry.shape[2]), "sampled": int(len(sample)),
              "positive_rate": float(labels[sample].mean()),
              "star": "icosahedron_12_vertices_traced", "radius_scale_h": float(radius_scale),
              "arclength": float(length),
              "field": str(field_path), "elapsed_seconds": time.time() - started}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, geometry=geometry.astype(np.float16),
                        labels=labels[sample].astype(np.int8),
                        seeds=seeds[sample].astype(np.float32),
                        metadata_json=json.dumps(record))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="deltaWing_resampled")
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7068)
    parser.add_argument("--steps", type=int, default=65, help="samples along each streamline")
    parser.add_argument("--length", type=float, default=0.5, help="arclength spanned")
    arguments = parser.parse_args()

    output = arguments.output or f"outputs/exp_Task6_Exact_r{arguments.radius_scale:g}"
    split = json.loads((SOURCE / "default_split.json").read_text())
    for role in ("train", "test"):
        for key in split[role]:
            flow, index = key.split(":")
            if flow != arguments.scene:
                continue
            folder = SOURCE / "frames" / flow / f"frame_{int(index):03d}"
            if not (folder / "sample_streamlines.npy").exists():
                continue
            target = ROOT / output / role / f"{flow}_{int(index):03d}.npz"
            if target.exists():
                continue
            record = build_frame(folder, FIELDS[flow], arguments.count, arguments.seed,
                                 arguments.radius_scale, target,
                                 steps=arguments.steps, length=arguments.length)
            print(f"  {role:5s} {key:26s} r={arguments.radius_scale:g}h  "
                  f"{record['sampled']} seeds  pos {record['positive_rate']:.4f}  "
                  f"{record['elapsed_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
