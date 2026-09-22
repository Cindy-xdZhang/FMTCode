"""Offline read-out study on saved latents: clusterer x restarts x epoch rule.

The 2D study (`docs/2d_kmeans_d8.md` 7.1-8) replaced k-means with a
**tied-covariance GMM** -- k-means under a Mahalanobis metric, the pooled
within-cluster covariance estimated jointly with the partition -- and reported
that this alone was worth +0.19, converged the F1 curve, and made Davies-Bouldin
epoch selection actively harmful.  It also found the restart count is *not* free:
an interior optimum at 8-14 with a cliff at 20, from the winner's curse of
maximising a proxy (silhouette, rho ~0.89 against F1) over a larger candidate set.

All three claims are retested here in 3D from latents saved during training, so
no retraining is needed per variant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.ClusterReadout_3D import cluster_readouts


def load(folder, label):
    reference = np.load(folder / "reference.npy").astype(int)
    files = sorted(folder.glob(f"{label}_step*.npy"))
    steps = [int(f.stem.split("step")[1]) for f in files]
    return reference, files, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", default="outputs/exp_Task1_IcoVAE_latents")
    parser.add_argument("--label", default="r5_both")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--clusterers", default="kmeans,gmm_tied,gmm_full,gmm_diag")
    parser.add_argument("--restarts", default="10")
    parser.add_argument("--stride", type=int, default=5, help="use every Nth evaluation")
    parser.add_argument("--output", default="outputs/exp_Task1_IcoVAE_readout.json")
    arguments = parser.parse_args()

    root = ROOT / arguments.latents
    datasets = [d for d in arguments.datasets.split(",") if d] or \
               sorted(p.name for p in root.iterdir() if p.is_dir())
    report = {}
    for clusterer in arguments.clusterers.split(","):
        for restarts in [int(r) for r in arguments.restarts.split(",")]:
            key = f"{clusterer}_r{restarts}"
            per_dataset = {}
            for dataset in datasets:
                folder = root / dataset
                reference, files, steps = load(folder, arguments.label)
                if not files:
                    continue
                curve = []
                for path, step in list(zip(files, steps))[::arguments.stride]:
                    latent = np.load(path).astype(np.float32)
                    scores, diagnostics = cluster_readouts(
                        latent, reference, restarts=restarts, seed=7068,
                        clusterer=clusterer)
                    curve.append({"step": step, **scores,
                                  "best_silhouette": max((d["silhouette"] for d in diagnostics),
                                                         default=0.0),
                                  "min_db": min((d["davies_bouldin"] for d in diagnostics),
                                                default=9e9)})
                if not curve:
                    continue
                values = [c["silhouette"] for c in curve]
                per_dataset[dataset] = {
                    "final": values[-1],
                    "median": float(np.median(values)),
                    "min_db": values[int(np.argmin([c["min_db"] for c in curve]))],
                    "max_silhouette": values[int(np.argmax([c["best_silhouette"] for c in curve]))],
                    "oracle": float(np.max(values)),
                    "converged_last_quarter": float(np.mean(
                        np.abs(np.diff(values[-max(len(values) // 4, 2):])))),
                    "evaluations": len(curve),
                }
                print(f"  {key:16s} {dataset:22s} " + "  ".join(
                    f"{r}={per_dataset[dataset][r]:.4f}"
                    for r in ("final", "median", "min_db", "max_silhouette", "oracle")),
                    flush=True)
            if per_dataset:
                report[key] = per_dataset
    (ROOT / arguments.output).write_text(json.dumps(report, indent=1))

    print(f"\n{'clusterer':18s} " + " ".join(f"{r:>15s}" for r in
          ("final", "median", "min_db", "max_silhouette", "oracle")) + "   drift")
    for key, per in report.items():
        cells = [np.mean([per[d][r] for d in per]) for r in
                 ("final", "median", "min_db", "max_silhouette", "oracle")]
        drift = np.mean([per[d]["converged_last_quarter"] for d in per])
        print(f"{key:18s} " + " ".join(f"{v:15.4f}" for v in cells) + f"   {drift:.4f}")


if __name__ == "__main__":
    main()
