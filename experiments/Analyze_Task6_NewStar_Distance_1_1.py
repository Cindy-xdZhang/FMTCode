"""Resolve Task6 accuracy against distance-to-coreline and against seeding stratum.

A single macro-F1 hides where a coreline classifier actually fails. The cache
stores every centre's exact distance to the continuous coreline (in units of h)
and which stratum it came from, so accuracy can be reported as a function of how
close a query point sits to the structure it is supposed to detect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FMT_Utils.IcoVAE_3D import (  # noqa: E402
    IcosahedralSirenVAE3D, apply_norm_stats_ico, build_signal_ico)
from FMT_Utils.IcosahedralGroup_3D import NEIGHBOUR_COUNT  # noqa: E402

RADII = (1.0, 2.0, 4.0, 8.0)
KINDS = {0: "on-tube (d < h)", 1: "hard shell (h..8h)", 2: "far (IVD region)"}


class Head(torch.nn.Module):
    def __init__(self, z, hidden=128, dropout=0.15, kind="mlp"):
        super().__init__()
        self.net = torch.nn.Linear(z, 2) if kind == "linear" else torch.nn.Sequential(
            torch.nn.Linear(z, hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 2))

    def forward(self, z):
        return self.net(z)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/home/cheny1a/data/task6_icostar_1_1")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="outputs/exp_Task6_NewStar/distance_analysis.json")
    args = p.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    shells = tuple(float(x) for x in a["shells"].split(","))
    idx = [0]
    for s in shells:
        b = 1 + NEIGHBOUR_COUNT * RADII.index(s)
        idx.extend(range(b, b + NEIGHBOUR_COUNT))
    idx = np.asarray(idx)

    curves, labels, dist, kind, scene = [], [], [], [], []
    for path in sorted(Path(args.cache).glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            if str(d["role"]) != "test":
                continue
            curves.append(d["curves"][:, idx]); labels.append(d["labels"])
            dist.append(d["dist"]); kind.append(d["kind"])
            scene.extend([str(d["key"]).split(":")[0]] * len(d["labels"]))
    curves = np.concatenate(curves); y = np.concatenate(labels).astype(np.int64)
    dist = np.concatenate(dist); kind = np.concatenate(kind); scene = np.array(scene)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sig = torch.from_numpy(build_signal_ico(curves)).to(device)
    model = IcosahedralSirenVAE3D(steps=sig.shape[2], z_inv_dim=a["z_inv"], z_eq_dim=a["z_eq"],
                                  encoder_kind=a["encoder"], lines=sig.shape[1]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    head = Head(a["z_inv"], kind=a["head"]).to(device)
    head.load_state_dict(ckpt["head"]); head.eval()
    stats = ckpt["stats"]

    pred = []
    with torch.no_grad():
        for i in range(0, len(sig), 4096):
            x = apply_norm_stats_ico(sig[i:i + 4096], stats, a["clip_sigma"])
            mu, _ = model.encode(x)
            pred.append(head(mu[:, :model.z_inv_dim]).argmax(-1).cpu())
    pred = torch.cat(pred).numpy()

    def acc(m):
        return float((pred[m] == y[m]).mean()) if m.any() else float("nan")

    report = {"weights": args.weights, "test_n": int(len(y)), "overall_accuracy": acc(np.ones(len(y), bool))}
    print(f"test stars {len(y)}, overall accuracy {report['overall_accuracy']:.4f}\n")

    print("by seeding stratum:")
    report["by_stratum"] = {}
    for k, name in KINDS.items():
        m = kind == k
        if not m.any():
            continue
        report["by_stratum"][name] = {"n": int(m.sum()), "accuracy": acc(m), "positive_rate": float(y[m].mean())}
        print(f"  {name:22s} n={m.sum():6d} acc={acc(m):.4f} pos={y[m].mean():.3f}")

    print("\nby distance to coreline (units of h):")
    edges = [0, 1, 2, 3, 4, 6, 8, 12, 20, 50, np.inf]
    report["by_distance"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if not m.any():
            continue
        row = {"lo": lo, "hi": None if np.isinf(hi) else hi, "n": int(m.sum()),
               "accuracy": acc(m), "positive_rate": float(y[m].mean())}
        report["by_distance"].append(row)
        bar = "#" * int(round(acc(m) * 40))
        print(f"  [{lo:5.1f},{hi:5.1f})h n={m.sum():6d} acc={acc(m):.4f} pos={y[m].mean():.3f} {bar}")

    print("\nby scene:")
    report["by_scene"] = {}
    for s in np.unique(scene):
        m = scene == s
        report["by_scene"][s] = {"n": int(m.sum()), "accuracy": acc(m)}
        print(f"  {s:24s} n={m.sum():6d} acc={acc(m):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
