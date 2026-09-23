"""How much coreline does the neighbour-shell margin exclude from the positive pool?

`Build_Task6_New_IcoStar_1_1.py` keeps centres at least ``max(radii) * h * 1.05``
clear of the domain boundary so every neighbour seed stays inside the field.
Positive centres are drawn by perturbing coreline points, so any coreline lying
inside that margin contributes **no positives**: the classifier is never shown
corelines near a wall or inflow/outflow plane. This measures that loss as a
fraction of densified coreline arc length, per scene and per candidate margin.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def densify(corelines, step):
    out = []
    for c in corelines:
        if len(c) < 2:
            out.append(np.asarray(c, float)); continue
        seg = np.diff(c, axis=0)
        for i, l in enumerate(np.linalg.norm(seg, axis=1)):
            k = max(int(np.ceil(l / step)), 1)
            out.append(c[i] + np.outer(np.linspace(0.0, 1.0, k, endpoint=False), seg[i]))
        out.append(np.asarray(c[-1:], float))
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share")
    p.add_argument("--radii", default="1,2,4,8")
    p.add_argument("--out", default="outputs/analysis_margin_loss.json")
    args = p.parse_args()

    sys.path.insert(0, args.root)
    from task6_fields import Dataset
    ds = Dataset(args.root)
    radii = [float(x) for x in args.radii.split(",")]

    per_scene = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_frame = []
    for role in ("train", "test"):
        for key in ds.split[role]:
            f = ds.frame(key)
            cores = f.corelines()
            if not cores:
                continue
            dense = densify(cores, f.h / 10.0)
            scene = key.split(":")[0]
            row = {"key": key, "role": role, "points": int(len(dense))}
            for r in radii:
                margin = f.h * r * 1.05
                lo, hi = f.bounds[:, 0] + margin, f.bounds[:, 1] - margin
                inside = np.all((dense >= lo) & (dense <= hi), axis=1)
                per_scene[scene][r][0] += int(inside.sum())
                per_scene[scene][r][1] += int(len(dense))
                row[f"inside_{r:g}h"] = float(inside.mean())
            per_frame.append(row)

    print(f"{'scene':24s} " + " ".join(f"{r:g}h".rjust(9) for r in radii))
    print(f"{'':24s} " + " ".join("kept".rjust(9) for _ in radii))
    worst = {}
    for scene in sorted(per_scene):
        cells = []
        for r in radii:
            ins, tot = per_scene[scene][r]
            frac = ins / tot if tot else float("nan")
            cells.append(f"{frac*100:8.2f}%")
            worst[r] = min(worst.get(r, 1.0), frac)
        print(f"{scene:24s} " + " ".join(cells))
    tot = {r: [sum(per_scene[s][r][0] for s in per_scene), sum(per_scene[s][r][1] for s in per_scene)] for r in radii}
    print(f"{'ALL':24s} " + " ".join(f"{tot[r][0]/tot[r][1]*100:8.2f}%" for r in radii))
    print()
    for r in radii:
        print(f"  {r:g}h margin: {(1-tot[r][0]/tot[r][1])*100:5.2f}% of coreline unreachable overall, "
              f"worst scene keeps {worst[r]*100:.2f}%")

    # frames where the margin removes a large share of the coreline
    print("\nframes losing >5% of their coreline at the largest margin:")
    rmax = max(radii)
    bad = sorted((x for x in per_frame if x[f"inside_{rmax:g}h"] < 0.95),
                 key=lambda x: x[f"inside_{rmax:g}h"])
    for x in bad[:15]:
        print(f"  {x['key']:28s} {x['role']:5s} kept {x[f'inside_{rmax:g}h']*100:6.2f}%")
    if not bad:
        print("  none")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"radii": radii,
         "per_scene": {s: {str(r): per_scene[s][r] for r in radii} for s in per_scene},
         "overall": {str(r): tot[r] for r in radii},
         "per_frame": per_frame}, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
