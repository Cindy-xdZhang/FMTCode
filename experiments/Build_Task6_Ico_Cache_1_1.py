"""13-line icosahedral primitives for the Task-6 coreline dataset (supervised).

Task 6 gives, per frame, 400,000 seeds with a 65-step streamline each and a
binary label (is this seed on a vortex coreline).  Unlike Task 1 the neighbour
streamlines cannot be traced on demand -- they are whatever seeds happen to lie
nearby -- so the icosahedral star is built by *assignment* rather than by seeding:

    for each seed, query its K nearest neighbours, then solve the optimal
    assignment of 12 of them to the 12 icosahedron vertex directions by angular
    distance (Hungarian, maximising total cosine).

With K = 64 the mean angular error is 14.8 degrees, comfortably inside the
icosahedral group's own 29.5-degree resolution, so the channel permutation the
group action relies on remains meaningful.  With the dataset's own `order16`
table it is 29.4 degrees -- as coarse as the group itself -- which is why that
table is not used.

The dataset's own `default_split.json` (45 train / 24 test frames) is preserved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.IcosahedralGroup_3D import icosahedron_vertices

SOURCE = Path("/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share")
CANDIDATES = 64


def build_star_at_radius(seeds, sample, vertices, radius, tree=None):
    """Nearest actual seed to each ideal point ``centre + radius * d_j``.

    This is the Task-1 construction adapted to a fixed seed cloud: there the
    neighbour was *traced* from ``centre + r d_j``, here the nearest existing
    seed to that point stands in for it.  Radius and direction are controlled
    together, unlike `build_star` where the radius is whatever the k nearest
    seeds happen to be.
    """
    tree = tree or cKDTree(seeds)
    ideal = seeds[sample][:, None, :] + radius * vertices[None, :, :]   # [N,12,3]
    distance, chosen = tree.query(ideal.reshape(-1, 3), workers=-1)
    chosen = chosen.reshape(len(sample), 12)
    offsets = seeds[chosen] - seeds[sample][:, None]
    unit = offsets / np.maximum(np.linalg.norm(offsets, axis=-1, keepdims=True), 1e-12)
    cosine = (unit * vertices[None]).sum(-1)
    errors = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).astype(np.float32)
    achieved = np.linalg.norm(offsets, axis=-1).astype(np.float32)
    return chosen, errors, achieved


def build_star(seeds, sample, vertices, candidates=CANDIDATES):
    """Assign 12 neighbours per sampled seed to the 12 icosahedral directions."""
    tree = cKDTree(seeds)
    _, neighbours = tree.query(seeds[sample], k=candidates + 1, workers=-1)
    neighbours = neighbours[:, 1:]                                   # drop self
    offsets = seeds[neighbours] - seeds[sample][:, None]
    unit = offsets / np.maximum(np.linalg.norm(offsets, axis=-1, keepdims=True), 1e-12)
    cosine = unit @ vertices.T                                       # [N, K, 12]
    chosen = np.empty((len(sample), 12), dtype=np.int64)
    errors = np.empty((len(sample), 12), dtype=np.float32)
    for row in range(len(sample)):
        candidate, direction = linear_sum_assignment(-cosine[row])
        order = np.argsort(direction)
        chosen[row] = neighbours[row][candidate[order]]
        errors[row] = np.degrees(np.arccos(np.clip(
            cosine[row][candidate[order], direction[order]], -1.0, 1.0)))
    return chosen, errors


def build_frame(folder, count, seed, destination, radius_scale=0.0):
    started = time.time()
    meta = json.loads((folder / "frame.json").read_text())
    seeds = np.load(folder / "seeds.npy")
    labels = np.load(folder / "labels.npy").astype(np.int64)
    streamlines = np.load(folder / "sample_streamlines.npy", mmap_mode="r")

    rng = np.random.default_rng(seed)
    sample = rng.choice(len(seeds), min(count, len(seeds)), replace=False)
    sample.sort()
    vertices = icosahedron_vertices()
    if radius_scale > 0:
        chosen, errors, achieved = build_star_at_radius(
            seeds, sample, vertices, radius_scale * meta["h"])
        duplicate = float(np.mean([len(set(r)) < 12 for r in chosen]))
    else:
        chosen, errors = build_star(seeds, sample, vertices)
        achieved = np.linalg.norm(seeds[chosen] - seeds[sample][:, None], axis=-1).astype(np.float32)
        duplicate = 0.0

    lines = np.concatenate((sample[:, None], chosen), axis=1)        # [N, 13]
    geometry = np.asarray(streamlines[lines.ravel()]).reshape(
        len(sample), 13, streamlines.shape[1], 3).astype(np.float32)

    record = {"flow": meta["flow"], "index": meta["index"], "key": meta["key"],
              "h": meta["h"], "steps": int(streamlines.shape[1]),
              "sampled": int(len(sample)), "of": int(len(seeds)),
              "positive_rate": float(labels[sample].mean()),
              "assignment_error_deg_mean": float(errors.mean()),
              "assignment_error_deg_p90": float(np.percentile(errors, 90)),
              "candidates": CANDIDATES,
              "star": ("icosahedron_12_vertices_at_radius" if radius_scale > 0
                       else "icosahedron_12_vertices_assigned"),
              "radius_scale_h": float(radius_scale),
              "achieved_radius_over_h_mean": float(achieved.mean() / meta["h"]),
              "duplicate_neighbour_fraction": duplicate,
              "elapsed_seconds": time.time() - started}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, geometry=geometry.astype(np.float16),
                        labels=labels[sample].astype(np.int8),
                        seeds=seeds[sample].astype(np.float32),
                        assignment_error=errors.astype(np.float16),
                        metadata_json=json.dumps(record))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/exp_Task6_Ico_cache")
    parser.add_argument("--count", type=int, default=6000, help="seeds sampled per frame")
    parser.add_argument("--seed", type=int, default=7068)
    parser.add_argument("--radius-scale", type=float, default=0.0,
                        help="target neighbour radius in units of h; 0 = k-nearest assignment")
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    split = json.loads((SOURCE / "default_split.json").read_text())
    root = ROOT / arguments.output
    skipped = []
    for role in ("train", "test"):
        keys = split[role][:arguments.limit] if arguments.limit else split[role]
        for key in keys:
            flow, index = key.split(":")
            folder = SOURCE / "frames" / flow / f"frame_{int(index):03d}"
            target = root / role / f"{flow}_{int(index):03d}.npz"
            if target.exists():
                continue
            required = ("seeds.npy", "labels.npy", "sample_streamlines.npy")
            if not folder.exists() or any(not (folder / n).exists() for n in required):
                # the shared snapshot is partial: SquareCylinder and tornado3d ship
                # no `sample_streamlines.npy` at all, halfcylinderRe640 only some
                skipped.append(key)
                print(f"  SKIP (incomplete) {key}", flush=True); continue
            record = build_frame(folder, arguments.count, arguments.seed, target,
                                 arguments.radius_scale)
            print(f"  {role:5s} {key:28s} {record['sampled']:5d} seeds  "
                  f"pos {record['positive_rate']:.4f}  "
                  f"assign {record['assignment_error_deg_mean']:.1f}deg  "
                  f"{record['elapsed_seconds']:.0f}s", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "skipped_frames.json").write_text(json.dumps(
        {"reason": "shared snapshot lacks sample_streamlines.npy", "keys": skipped}, indent=1))
    print(f"  skipped {len(skipped)} incomplete frames", flush=True)


if __name__ == "__main__":
    main()
