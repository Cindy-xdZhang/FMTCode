"""Which (partition rule x epoch rule) actually delivers the good representation?

Every IcoVAE run stores the whole F1 curve together with the label-free criteria
at each evaluation, so the selection strategy can be studied from the record
instead of by retraining.  Two independent choices:

* **partition rule** -- among the ~20 k-means restarts at one evaluation, which
  partition to keep (inertia / Davies-Bouldin / silhouette / Calinski-Harabasz);
* **epoch rule** -- which evaluation to take the answer from (min DB, max
  silhouette, final, median, oracle).

Both must be label-free to count.  `oracle` rows are the ceiling, not a method.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

PARTITION = ("inertia", "davies_bouldin", "silhouette", "calinski_harabasz")
EPOCH = ("min_db", "max_silhouette", "final", "median", "oracle")


def epoch_index(curve, rule):
    if rule == "min_db":
        return int(np.argmin([c["min_davies_bouldin"] for c in curve]))
    if rule == "max_silhouette":
        return int(np.argmax([c["best_silhouette"] for c in curve]))
    if rule == "final":
        return len(curve) - 1
    return None                                    # median / oracle handled per column


def score(curve, partition_rule, epoch_rule):
    values = [c["rules"][partition_rule] for c in curve]
    if epoch_rule == "median":
        return float(np.median(values))
    if epoch_rule == "oracle":
        return float(np.max(values))
    return float(values[epoch_index(curve, epoch_rule)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="outputs/exp_Task1_IcoVAE_*/summary_*.json")
    parser.add_argument("--labels", default="", help="comma-separated subset of run labels")
    arguments = parser.parse_args()

    runs = defaultdict(dict)
    for path in sorted(glob.glob(arguments.pattern)):
        data = json.load(open(path))
        runs[data["label"]][data["dataset"]] = data["curve"]
    wanted = [l for l in arguments.labels.split(",") if l] or sorted(runs)

    print(f"{'partition rule':18s} " + " ".join(f"{e:>15s}" for e in EPOCH))
    print("-" * (19 + 16 * len(EPOCH)))
    grand = defaultdict(list)
    for label in wanted:
        if label not in runs:
            continue
        print(f"[{label}]  {len(runs[label])} dataset(s)")
        for partition_rule in PARTITION:
            cells = []
            for epoch_rule in EPOCH:
                values = [score(curve, partition_rule, epoch_rule)
                          for curve in runs[label].values()]
                cells.append(np.mean(values))
                grand[(partition_rule, epoch_rule)].append(np.mean(values))
            print(f"  {partition_rule:16s} " + " ".join(f"{v:15.4f}" for v in cells))
    print()
    print("mean over every run above:")
    print(f"{'partition rule':18s} " + " ".join(f"{e:>15s}" for e in EPOCH))
    best = (None, -1)
    for partition_rule in PARTITION:
        cells = []
        for epoch_rule in EPOCH:
            value = float(np.mean(grand[(partition_rule, epoch_rule)]))
            cells.append(value)
            if epoch_rule != "oracle" and value > best[1]:
                best = ((partition_rule, epoch_rule), value)
        print(f"  {partition_rule:16s} " + " ".join(f"{v:15.4f}" for v in cells))
    print(f"\nbest label-free combination: partition={best[0][0]}  epoch={best[0][1]}  -> {best[1]:.4f}")


if __name__ == "__main__":
    main()
