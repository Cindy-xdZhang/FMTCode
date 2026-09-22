"""FMT and Raw baselines on the *full-data* unsupervised setup.

The published Task-1 FMT numbers (Re160 .5225, Re640 .5177, Re6400 .5340) were
measured under the old development / confirmation split: k-means fitted on
development, the cluster-to-class map frozen on ordinals 8-9, confirmation
scored once.  The IcoVAE study has no split -- every timeslice is used and every
timeslice is scored -- so those numbers are not a like-for-like reference.

This recomputes the training-free baselines under the new setup: all fourteen
timeslices pooled, k-means fitted on all of it, scored on all of it, through the
identical read-out machinery (`cluster_readouts`) the IcoVAE runs use.

Two primitives are reported:

* ``fmt7``  -- the published 7-line octahedral star, from the frozen ReSweep
  cache, so the encoder is byte-identical to the published arm;
* ``fmt13`` -- the same FMT encoder on the 13-line icosahedral star the IcoVAE
  sees, so the primitive is matched and only the encoder differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.ClusterReadout_3D import RULES, cluster_readouts
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d

NUM_FREQ = 6
NEIGHBOUR_WEIGHT = 0.5
RESWEEP = "outputs/exp_Task1_ReSweep_0.1"
ICO = "outputs/exp_Task1_IcoVAE_cache"


def load_seven_line(dataset):
    """Pool the frozen ReSweep caches: 10 development + 4 confirmation slices."""
    features, raw, reference = [], [], []
    for folder in ("development_cache", "confirmation_cache"):
        for path in sorted((ROOT / RESWEEP / folder / dataset).glob("*.npz")):
            with np.load(path) as data:
                features.append(data["fmt_features"])
                raw.append(data["raw_features"])
                reference.append(data["reference"])
    if not features:
        raise FileNotFoundError(f"no ReSweep cache for {dataset}")
    return (np.concatenate(features), np.concatenate(raw),
            np.concatenate(reference).astype(int))


def load_thirteen_line(dataset):
    geometry, reference = [], []
    for path in sorted((ROOT / ICO / dataset).glob("*.npz")):
        with np.load(path) as data:
            geometry.append(data["geometry"])
            reference.append(data["reference"])
    if not geometry:
        raise FileNotFoundError(f"no icosahedral cache for {dataset}")
    return np.concatenate(geometry), np.concatenate(reference).astype(int)


def fmt_encode(geometry, device):
    """The published FMT encoder settings, applied to any star size."""
    tensor = torch.from_numpy(np.ascontiguousarray(geometry)).float().to(device)
    features = pathline_dft_features_3d(
        tensor, num_freq=NUM_FREQ, neighbor_weight=1.0, neighbor_scale=1.0,
        neighbor_pool="sort", mode="gram", include_chirality=True)
    return np.asarray(features)


def standardize_like_fmt(features):
    standardized = StandardScaler().fit_transform(features)
    base = NUM_FREQ * 3 + (NUM_FREQ - 1)          # 23 for k=6, gram + chirality
    standardized[:, base:] *= NEIGHBOUR_WEIGHT
    return standardized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="halfcylinderRe160_eth,halfcylinderRe640_eth,halfcylinderRe6400_eth")
    parser.add_argument("--output", default="outputs/exp_Task1_Baselines_Full_1.1")
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7068)
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    destination = ROOT / arguments.output
    destination.mkdir(parents=True, exist_ok=True)

    for dataset in arguments.datasets.split(","):
        report = {"dataset": dataset, "setup": "full_data_no_split", "arms": {}}
        geometry, reference13 = load_thirteen_line(dataset)
        thirteen = fmt_encode(geometry, device)
        raw13 = geometry.reshape(len(geometry), -1)
        arms = {
            "fmt13_icosahedral_star": (standardize_like_fmt(thirteen), reference13),
            "raw13_pathline_coordinates": (StandardScaler().fit_transform(raw13), reference13),
        }
        try:                      # only the three half-cylinders have a 7-line cache
            seven, raw_seven, reference7 = load_seven_line(dataset)
            arms["fmt7_published_star"] = (standardize_like_fmt(seven), reference7)
            arms["raw7_pathline_coordinates"] = (
                StandardScaler().fit_transform(raw_seven), reference7)
        except FileNotFoundError:
            pass
        for name, (matrix, reference) in arms.items():
            scores, _ = cluster_readouts(matrix, reference, restarts=arguments.restarts,
                                         seed=arguments.seed, normalize=False)
            report["arms"][name] = {"dimensions": int(matrix.shape[1]),
                                    "primitives": int(matrix.shape[0]), **scores}
            print(f"  {dataset:24s} {name:26s} {matrix.shape[1]:4d}D  "
                  + "  ".join(f"{r[:4]}={scores[r]:.4f}" for r in RULES), flush=True)
        (destination / f"{dataset}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
