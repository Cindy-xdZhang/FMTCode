"""How much Task4-c data survives the lattice-exact octahedral requirement.

A primitive is usable only when the seeding centre and all six face-adjacent
lattice neighbours survived both the seeding validity test and the
post-integration curve cleaning.  This script reports the retained fraction by
flow, split, offset scale and label, writes a JSON artifact, and prints the
markdown table used in the documentation.  It trains nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.Task4C_OctVAE_2_1 import octahedral_selection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLOWS = ("channel", "tbl")
SPLITS = ("train", "validation", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/exp_Task4C_LocalRebuild_1.0/physical")
    parser.add_argument("--output", default="outputs/exp_Task4C_OctahedralCoverage_1.1")
    arguments = parser.parse_args()
    cache = ROOT / arguments.cache

    rows, by_scale, by_split = [], defaultdict(lambda: [0, 0]), {}
    histogram = np.zeros(7, dtype=int)
    centre_missing = 0
    for flow in FLOWS:
        for split in SPLITS:
            folder = cache / flow / split
            seeds = np.load(folder / "seeds.npy")
            with np.load(folder / "metadata.npz") as data:
                meta = {key: data[key] for key in data.files}
            select, usable = octahedral_selection(seeds, meta)
            faces = (select[:, 1:] >= 0).sum(axis=1)
            histogram += np.bincount(6 - faces, minlength=7)
            centre_missing += int((select[:, 0] < 0).sum())
            offset = np.round(meta["neighbor_distance"] / meta["local_grid_scale"], 3)
            for value in np.unique(offset):
                pick = offset == value
                by_scale[float(value)][0] += int(usable[pick].sum())
                by_scale[float(value)][1] += int(pick.sum())
            labels = np.asarray(meta["labels"]).astype(int)
            rows.append({
                "flow": flow, "split": split, "bundles": int(len(usable)),
                "usable": int(usable.sum()), "ratio": float(usable.mean()),
                "valid_lines_mean": float(np.asarray(meta["counts"]).mean()),
                "faces_present_mean": float(faces.mean()),
                "positive_rate_all": float(labels.mean()),
                "positive_rate_usable": float(labels[usable].mean()) if usable.any() else 0.0,
            })
            key = by_split.setdefault(split, [0, 0, 0.0, 0.0])
            key[0] += int(usable.sum()); key[1] += int(len(usable))

    total_usable = sum(r["usable"] for r in rows)
    total = sum(r["bundles"] for r in rows)
    output = ROOT / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "cache": str(arguments.cache), "per_split": rows,
        "total_bundles": total, "total_usable": total_usable,
        "total_ratio": total_usable / total,
        "dropped_ratio": 1.0 - total_usable / total,
        "centre_line_missing": centre_missing,
        "missing_face_histogram": histogram.tolist(),
        "by_offset_scale": {str(k): {"usable": v[0], "bundles": v[1], "ratio": v[0] / v[1]}
                            for k, v in sorted(by_scale.items())},
    }
    (output / "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("| flow | split | bundles | complete | ratio | mean valid lines | mean faces | "
          "pos-rate all | pos-rate complete |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['flow']} | {r['split']} | {r['bundles']:,} | {r['usable']:,} | "
              f"{r['ratio']:.3f} | {r['valid_lines_mean']:.2f} | {r['faces_present_mean']:.3f} | "
              f"{r['positive_rate_all']:.4f} | {r['positive_rate_usable']:.4f} |")
    print(f"| **total** | | **{total:,}** | **{total_usable:,}** | "
          f"**{total_usable/total:.4f}** | | | | |")
    print("\n| offset (units of local h) | complete | bundles | ratio |")
    print("|---|---:|---:|---:|")
    for key, value in report["by_offset_scale"].items():
        print(f"| {float(key):.2f} h | {value['usable']:,} | {value['bundles']:,} | "
              f"{value['ratio']:.4f} |")
    print(f"\nmissing-face histogram (0..6): {histogram.tolist()}")
    print(f"bundles whose centre line itself was cleaned away: {centre_missing}")
    print(f"\nwrote {output/'coverage.json'}")


if __name__ == "__main__":
    main()
