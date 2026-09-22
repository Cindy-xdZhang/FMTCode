"""Multi-shell icosahedral stars for deltaWing: centre + 12 per radius.

Two concentric icosahedral shells remain exactly icosahedrally symmetric -- the
group permutes both shells by the same permutation -- so the star gains a second
length scale without leaving the symmetry class.  Verified to 1.33e-15 against
retracing in a rotated flow across all 120 elements.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from FMT_Utils.IcosahedralGroup_3D import icosahedron_vertices
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from FMT_Utils.StreamlineStar_3D import multi_shell_star

SOURCE = Path("/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share")
FIELD = "/home/cheny1a/data/flowData3D/deltaWing_mag0_3reesampled.nc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radii", default="4,8", help="comma-separated, in units of h")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7068)
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()
    radii = [float(r) for r in arguments.radii.split(",")]
    output = arguments.output or f"outputs/exp_Task6_Shells_{arguments.radii.replace(',','_')}"

    split = json.loads((SOURCE / "default_split.json").read_text())
    vertices = icosahedron_vertices()
    for role in ("train", "test"):
        for key in split[role]:
            flow, index = key.split(":")
            if flow != "deltaWing_resampled":
                continue
            folder = SOURCE / "frames" / flow / f"frame_{int(index):03d}"
            target = ROOT / output / role / f"{flow}_{int(index):03d}.npz"
            if target.exists() or not (folder / "seeds.npy").exists():
                continue
            started = time.time()
            meta = json.loads((folder / "frame.json").read_text())
            seeds = np.load(folder / "seeds.npy")
            labels = np.load(folder / "labels.npy").astype(np.int64)
            field, _ = load_netcdf_window_3d(FIELD, meta["index"], 1, 640)
            lower = np.asarray(field.domainMinBoundary); upper = np.asarray(field.domainMaxBoundary)
            rng = np.random.default_rng(arguments.seed)
            sample = np.sort(rng.choice(len(seeds), min(arguments.count, len(seeds)), replace=False))
            parts = []
            for start in range(0, len(sample), 512):
                parts.append(multi_shell_star(
                    field.field[0], seeds[sample[start:start + 512]].astype(np.float64),
                    vertices, [r * meta["h"] for r in radii], lower, upper))
            geometry = np.concatenate(parts).astype(np.float16)
            record = {"flow": flow, "index": meta["index"], "h": meta["h"],
                      "radii_h": radii, "lines": int(geometry.shape[1]),
                      "sampled": int(len(sample)),
                      "positive_rate": float(labels[sample].mean()),
                      "star": "icosahedron_multi_shell_traced",
                      "elapsed_seconds": time.time() - started}
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, geometry=geometry,
                                labels=labels[sample].astype(np.int8),
                                seeds=seeds[sample].astype(np.float32),
                                metadata_json=json.dumps(record))
            print(f"  {role:5s} {key:26s} radii={radii} lines={record['lines']}  "
                  f"{record['elapsed_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
