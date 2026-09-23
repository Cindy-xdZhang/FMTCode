"""Print everything about a Task6 star cache that affects reproducibility.

Run this before training. If any line disagrees with the expected values printed
by the reproduction guide, the cache is not the one the guide's numbers refer to
and training will not reproduce them.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

EXPECTED = {
    "radii": [0.25, 0.5, 1.0],
    "frames": 101,
    "train_frames": 77,
    "test_frames": 24,
    "positive_scale": 1.0,
    "total_length": 0.5,
    "points": 65,
    "lines": 37,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    p.add_argument("--expect-split", default="package",
                   choices=["package", "temporal"],
                   help="which split the cache should encode")
    args = p.parse_args()

    root = Path(args.cache)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {root}")

    per_scene = collections.defaultdict(lambda: [0, 0])
    roles = collections.Counter()
    scene_frames = collections.defaultdict(lambda: collections.defaultdict(list))
    meta0 = None
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            key, role = str(d["key"]), str(d["role"])
            scene = key.split(":")[0]
            roles[role] += 1
            scene_frames[scene][role].append(int(key.split(":")[1]))
            per_scene[scene][0] += int(d["labels"].sum())
            per_scene[scene][1] += len(d["labels"])
            if meta0 is None:
                meta0 = {"radii": [float(x) for x in d["radii"]],
                         "lines": int(d["curves"].shape[1]),
                         "points": int(d["curves"].shape[2]),
                         "total_length": float(d["total_length"]),
                         "positive_scale": float(d["positive_scale"]) if "positive_scale" in d else 1.0}

    print(f"cache: {root}")
    print(f"  frames         {len(files)}   (train {roles['train']} / test {roles['test']})")
    print(f"  radii          {meta0['radii']}   -> boundary margin {max(meta0['radii'])} h")
    print(f"  lines/sample   {meta0['lines']}   points/line {meta0['points']}")
    print(f"  total_length   {meta0['total_length']}")
    print(f"  positive_scale {meta0['positive_scale']}   (positive iff d < scale*h)")
    mj = root / "meta.json"
    if mj.exists():
        m = json.loads(mj.read_text())
        print(f"  split_file     {m.get('split_file', '?')}")
        print(f"  label_rule     {m.get('label_rule', '?')}")

    total = [sum(v[0] for v in per_scene.values()), sum(v[1] for v in per_scene.values())]
    print(f"\n  {'scene':24s} {'stars':>9s} {'positives':>10s} {'rate':>7s}")
    for s in sorted(per_scene):
        pos, n = per_scene[s]
        print(f"  {s:24s} {n:9,d} {pos:10,d} {pos / n:7.4f}")
    print(f"  {'TOTAL':24s} {total[1]:9,d} {total[0]:10,d} {total[0] / total[1]:7.4f}")

    # is the split temporal or interleaved?
    interleaved = []
    for s, r in scene_frames.items():
        if r["train"] and r["test"] and max(r["train"]) > min(r["test"]):
            interleaved.append(s)
    kind = "package default (interleaved in time)" if interleaved else "temporal (early train / late test)"
    print(f"\n  split appears to be: {kind}")
    if interleaved:
        print(f"    scenes with train frames after test frames: {sorted(interleaved)}")

    print("\n  checks against the reproduction guide:")
    ok = True
    for k, want in EXPECTED.items():
        got = {"radii": meta0["radii"], "frames": len(files), "train_frames": roles["train"],
               "test_frames": roles["test"], "positive_scale": meta0["positive_scale"],
               "total_length": meta0["total_length"], "points": meta0["points"],
               "lines": meta0["lines"]}[k]
        good = got == want
        ok &= good
        print(f"    {'ok  ' if good else 'FAIL'} {k:15s} expected {want}  got {got}")
    want_split = "temporal" if args.expect_split == "temporal" else "package"
    got_split = "package" if interleaved else "temporal"
    good = want_split == got_split
    ok &= good
    print(f"    {'ok  ' if good else 'FAIL'} {'split':15s} expected {want_split}  got {got_split}")
    rate_ok = 0.20 <= total[0] / total[1] <= 0.30
    ok &= rate_ok
    print(f"    {'ok  ' if rate_ok else 'FAIL'} {'positive rate':15s} expected 0.20-0.30  got {total[0] / total[1]:.4f}")
    zero = [s for s, v in per_scene.items() if v[0] == 0]
    if zero:
        ok = False
        print(f"    FAIL scenes with NO positives: {sorted(zero)}")
        print("         -> the cache was probably built with --official-centres; the official")
        print("            IVD>0.8*max rule yields no positives in several scenes.")
    print(f"\n  {'CACHE MATCHES THE GUIDE' if ok else 'CACHE DOES NOT MATCH THE GUIDE -- numbers will not reproduce'}")


if __name__ == "__main__":
    main()
