"""Task1 arm: octahedral-equivariant SIREN-VAE on cached 3D pathline primitives.

Runs the new method beside the Raw and FMT k-means arms on exactly the same
cached slices, splits, cluster calibration and confirmation timeslices, so the
only thing that differs between arms is the representation.

Protocol (``docs/research_tasks_and_protocol.md``):

* splits are by timeslice, never by spatial seed;
* the VAE trains on development ordinals only and its checkpoint is chosen by
  Davies-Bouldin on held-out development ordinals -- label-free, so the
  confirmation slices stay frozen;
* k-means is fitted on training features only, the anonymous cluster-to-class
  map is frozen on the calibration ordinals, and confirmation is scored once;
* every arm reports >= 3 seeds and the model reports parameter counts.

This arm is **not** training-free.  Structurally it is
``primitive -> learned encoder -> latent -> k-means``, which is the shape of
Task2's ``Raw+VAE`` arm; it is evaluated here under Task1 because Task1's
acceptance criterion is clustering quality and nothing in this pipeline sees a
label before calibration.  See ``docs/exp_Task1_OctVAE_3D_1.1.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.OctahedralGroup_3D import (  # noqa: E402
    apply_norm_stats_oh, build_signal_from_raw, group_tensors,
)
from FMT_Utils.SirenVAE_3D import (  # noqa: E402
    READOUTS, encode_readouts, fit_octvae, l2_normalize,
)
from FMT_Utils.Task12Data_3D import (  # noqa: E402
    load_cache_records, stack_features, stack_reference,
)
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from FMT_Utils.Task12Evaluation_3D import (  # noqa: E402
    binary_cluster_metrics, calibrate_vortex_cluster, fit_kmeans_transform,
)

ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = ("f1", "iou", "precision", "recall", "balanced_accuracy", "ari",
               "nmi", "positive_fraction", "predicted_positive_fraction")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _score(train_features, calibration_features, calibration_reference,
           test_features, test_reference, pca_dim, seed, n_init):
    """Fit on train, freeze the cluster identity on calibration, score on test."""
    fitted = fit_kmeans_transform(train_features, pca_dim, seed, n_init)
    vortex = calibrate_vortex_cluster(calibration_reference,
                                      fitted.predict(calibration_features))
    return binary_cluster_metrics(test_reference, fitted.predict(test_features), vortex)


def _score_latent(train_features, calibration_features, calibration_reference,
                 test_features, test_reference, seed, n_init, standardize):
    """Score an L2-normalised latent, with or without per-dimension standardising.

    `fit_kmeans_transform` standardises every feature dimension.  That is right
    for Raw and FMT, whose dimensions have wildly different natural scales, and
    it is what the published Task1 pipeline does.  It is **catastrophic** for an
    L2-normalised latent, which already lies on a unit sphere with near-uniform
    per-dimension spread: standardising destroys that geometry and k-means then
    converges on a degenerate split.  Measured on halfcylinderRe640 at 3000
    steps, same latent, same seeds:

        L2 -> StandardScaler -> KMeans(n_init=20)   F1 .1300, 99.99% predicted positive
        L2 ->                   KMeans(n_init=20)   F1 .6750, 10.96% predicted positive
        L2 ->                   KMeans(n_init=100)  F1 .6750   (n_init does not help:
                                                                the degenerate split is a
                                                                genuine inertia optimum)

    The 2D method clusters L2-normalised `z_inv` directly, with no standardising,
    which is what `standardize=False` reproduces.  Both are scored so the failure
    mode stays in the results table instead of being quietly dropped.
    """
    train = np.asarray(train_features, dtype=np.float32)
    calibration = np.asarray(calibration_features, dtype=np.float32)
    test = np.asarray(test_features, dtype=np.float32)
    if standardize:
        scaler = StandardScaler().fit(train)
        train, calibration, test = (scaler.transform(x) for x in (train, calibration, test))
    model = KMeans(n_clusters=2, random_state=int(seed), n_init=int(n_init)).fit(train)
    vortex = calibrate_vortex_cluster(calibration_reference, model.predict(calibration))
    return binary_cluster_metrics(test_reference, model.predict(test), vortex)


def _load(dataset, splits):
    development = load_cache_records(ROOT / dataset["development_cache"])
    confirmation = load_cache_records(ROOT / dataset["confirmation_cache"])
    needed = set(splits["vae_train"]) | set(splits["vae_validation"])
    needed |= set(splits["kmeans_train"]) | set(splits["cluster_calibration"])
    missing = [i for i in sorted(needed) if i >= len(development)]
    if missing:
        raise IndexError(f"{dataset['id']}: development ordinals {missing} do not exist")
    overlap = set(splits["kmeans_train"]) & set(splits["cluster_calibration"])
    if overlap:
        raise ValueError(f"{dataset['id']}: k-means train and calibration share {overlap}")
    return development, confirmation


def _signals(records, ordinals, sampled_steps):
    return np.concatenate([
        build_signal_from_raw(records[i]["raw"], sampled_steps) for i in ordinals
    ])


def run_dataset(dataset, config, device):
    splits = config["splits"]
    training = dict(config["training"])
    training.update(config["model"])
    sampled_steps = int(config["sampled_steps"])
    development, confirmation = _load(dataset, splits)

    calibration_reference = stack_reference(
        [development[i] for i in splits["cluster_calibration"]])
    test_reference = stack_reference(confirmation)

    rows, histories = [], {}

    # ---- reference arms: Raw and FMT k-means on the identical path ----------
    for arm in dataset.get("baselines", config.get("baselines", [])):
        train = stack_features([development[i] for i in splits["kmeans_train"]],
                               arm["feature"], device)
        calibration = stack_features(
            [development[i] for i in splits["cluster_calibration"]], arm["feature"], device)
        test = stack_features(confirmation, arm["feature"], device)
        for seed in config["kmeans_seeds"]:
            metrics = _score(train, calibration, calibration_reference, test,
                             test_reference, arm["pca_dim"], seed,
                             config["kmeans_n_init"])
            rows.append({"dataset": dataset["id"], "family": dataset["family"],
                         "method": arm["name"], "readout": "-",
                         "feature": arm["feature"],
                         "pca_dim": "none" if arm["pca_dim"] is None else int(arm["pca_dim"]),
                         "seed": int(seed), "train_seed": "-", **metrics})
        print(f"  {arm['name']:10s} f1={np.mean([r['f1'] for r in rows if r['method']==arm['name']]):.4f}",
              flush=True)

    # ---- the new arm -------------------------------------------------------
    signal_train = _signals(development, splits["vae_train"], sampled_steps)
    signal_validation = _signals(development, splits["vae_validation"], sampled_steps)
    signal_kmeans = _signals(development, splits["kmeans_train"], sampled_steps)
    signal_calibration = _signals(development, splits["cluster_calibration"], sampled_steps)
    signal_test = np.concatenate([
        build_signal_from_raw(record["raw"], sampled_steps) for record in confirmation])

    matrices, permutation = group_tensors(config["group"], device=device)
    readouts = tuple(config.get("readouts", READOUTS))

    for train_seed in config["train_seeds"]:
        started = time.time()
        print(f"  octvae seed {train_seed} ...", flush=True)
        model, stats, history = fit_octvae(
            signal_train, signal_validation, training, train_seed, device,
            group=config["group"])
        history["seconds"] = time.time() - started
        history["parameters"] = model.parameter_counts()
        histories[f"{dataset['id']}_seed{train_seed}"] = history

        encoded = {}
        for name, values in (("kmeans_train", signal_kmeans),
                             ("calibration", signal_calibration),
                             ("test", signal_test)):
            tensor = apply_norm_stats_oh(
                torch.as_tensor(values, dtype=torch.float32, device=device),
                stats, training["clip_sigma"])
            encoded[name] = encode_readouts(model, tensor, matrices, permutation,
                                            batch_size=config["encode_batch_size"],
                                            readouts=readouts)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for readout in readouts:
            train = l2_normalize(encoded["kmeans_train"][readout])
            calibration = l2_normalize(encoded["calibration"][readout])
            test = l2_normalize(encoded["test"][readout])
            for method, standardize in (("octvae", False), ("octvae_scaled", True)):
                for seed in config["kmeans_seeds"]:
                    metrics = _score_latent(train, calibration, calibration_reference,
                                            test, test_reference, seed,
                                            config["kmeans_n_init"], standardize)
                    rows.append({"dataset": dataset["id"], "family": dataset["family"],
                                 "method": method, "readout": readout,
                                 "feature": f"z_inv_{readout}", "pca_dim": "none",
                                 "seed": int(seed), "train_seed": int(train_seed),
                                 **metrics})
        summary = {r["readout"]: np.mean([x["f1"] for x in rows
                                          if x["method"] == "octvae"
                                          and x["readout"] == r["readout"]
                                          and x["train_seed"] == train_seed])
                   for r in rows if r["method"] == "octvae"}
        print(f"    step {history['selected_step']}, {history['seconds']:.0f}s, "
              + ", ".join(f"{k}={v:.4f}" for k, v in sorted(summary.items())), flush=True)

    return rows, histories


def _aggregate(rows, dataset, method, readout=None):
    values = [r for r in rows if r["dataset"] == dataset and r["method"] == method
              and (readout is None or r["readout"] == readout)]
    if not values:
        return None
    return {key: (float(np.mean([r[key] for r in values])),
                  float(np.std([r[key] for r in values]))) for key in METRIC_KEYS}


def build_comparison(rows, config):
    """Headline table: published FMT, measured FMT/Raw, and every OctVAE readout."""
    readouts = tuple(config.get("readouts", READOUTS))
    table, records = [], []
    for dataset in config["datasets"]:
        name = dataset["id"]
        published = dataset.get("published", {})
        prior = dataset.get("prior_run", {})
        raw = _aggregate(rows, name, "raw")
        fmt = _aggregate(rows, name, "fmt")
        entry = {
            "dataset": name, "family": dataset["family"],
            "label": dataset.get("label", name),
            "published_fmt_f1": published.get("fmt_f1"),
            "published_raw_f1": published.get("raw_f1"),
            "published_gain": published.get("gain"),
            "published_ari": published.get("ari"),
            "prior_fmt_f1": prior.get("fmt_f1"),
            "prior_raw_f1": prior.get("raw_f1"),
            "prior_fmt_ari": prior.get("ari"),
            "measured_fmt_f1": fmt["f1"][0] if fmt else None,
            "measured_fmt_ari": fmt["ari"][0] if fmt else None,
            "measured_raw_f1": raw["f1"][0] if raw else None,
            "measured_raw_ari": raw["ari"][0] if raw else None,
        }
        for readout in readouts:
            aggregated = _aggregate(rows, name, "octvae", readout)
            if aggregated is None:
                continue
            entry[f"octvae_{readout}_f1"] = aggregated["f1"][0]
            entry[f"octvae_{readout}_f1_std"] = aggregated["f1"][1]
            entry[f"octvae_{readout}_ari"] = aggregated["ari"][0]
            entry[f"octvae_{readout}_nmi"] = aggregated["nmi"][0]
            entry[f"octvae_{readout}_iou"] = aggregated["iou"][0]
        best = max(
            ((entry.get(f"octvae_{r}_f1"), r) for r in readouts
             if entry.get(f"octvae_{r}_f1") is not None),
            default=(None, None))
        entry["octvae_best_readout"] = best[1]
        entry["octvae_best_f1"] = best[0]
        if best[0] is not None and entry["measured_fmt_f1"] is not None:
            entry["octvae_minus_fmt_f1"] = best[0] - entry["measured_fmt_f1"]
            entry["octvae_minus_raw_f1"] = best[0] - entry["measured_raw_f1"]
        records.append(entry)
        table.append(entry)

    def macro(key):
        values = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    keys = [k for k in records[0] if k not in {"dataset", "family", "label",
                                               "octvae_best_readout"}]
    summary = {"dataset": "MACRO", "family": "-", "label": "**Dataset macro**",
               "octvae_best_readout": "-"}
    summary.update({key: macro(key) for key in keys})
    table.append(summary)
    return table


def comparison_markdown(table, config):
    """Two tables: the headline arm comparison, then a baseline cross-check."""
    readouts = tuple(config.get("readouts", READOUTS))

    def cell(value, digits=4):
        return "—" if value is None else f"{value:.{digits}f}"

    def signed(value):
        return "—" if value is None else f"{value:+.4f}"

    headline = [
        "### Task1 confirmation F1 — all arms on identical slices, splits and calibration",
        "",
        "| Flow | FMT | Raw | " + " | ".join(f"OctVAE {r}" for r in readouts)
        + " | best OctVAE − FMT | best OctVAE − Raw |",
        "|---" * (5 + len(readouts)) + "|",
    ]
    for row in table:
        macro = row["dataset"] == "MACRO"
        cells = [row["label"], cell(row["measured_fmt_f1"]), cell(row["measured_raw_f1"])]
        for readout in readouts:
            value = cell(row.get(f"octvae_{readout}_f1"))
            if row.get("octvae_best_readout") == readout and not macro:
                value = f"**{value}**"
            cells.append(value)
        cells += [signed(row.get("octvae_minus_fmt_f1")), signed(row.get("octvae_minus_raw_f1"))]
        if macro:
            cells = [c if i == 0 else f"**{c}**" for i, c in enumerate(cells)]
        headline.append("| " + " | ".join(cells) + " |")

    check = [
        "",
        "### Baseline cross-check — published, previously measured, and this run",
        "",
        "Published values are the frozen unified-config Task1 main table in",
        "`paper_tables_tasks_3d.md`.  The prior-run column is",
        "`outputs/exp_Task1_*/paper_table.csv`, measured on these same ETH-original",
        "caches, which are not the same files as several published entries.",
        "",
        "| Flow | FMT published | FMT prior run | FMT this run | Raw published | "
        "Raw prior run | Raw this run |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        macro = row["dataset"] == "MACRO"
        cells = [row["label"], cell(row["published_fmt_f1"]), cell(row.get("prior_fmt_f1")),
                 cell(row["measured_fmt_f1"]), cell(row["published_raw_f1"]),
                 cell(row.get("prior_raw_f1")), cell(row["measured_raw_f1"])]
        if macro:
            cells = [c if i == 0 else f"**{c}**" for i, c in enumerate(cells)]
        check.append("| " + " | ".join(cells) + " |")

    return "\n".join(headline + check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", action="append", default=None,
                        help="restrict to these dataset ids (repeatable)")
    parser.add_argument("--group", default=None, choices=["oh", "o"],
                        help="override the augmentation group")
    parser.add_argument("--tag", default=None, help="suffix for the output directory")
    arguments = parser.parse_args()

    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    if arguments.group:
        config["group"] = arguments.group
    if arguments.dataset:
        wanted = set(arguments.dataset)
        config["datasets"] = [d for d in config["datasets"] if d["id"] in wanted]
        if not config["datasets"]:
            raise SystemExit(f"no configured dataset matches {sorted(wanted)}")

    device = torch.device(config.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    result_dir = ROOT / config["output"]["result_dir"]
    if arguments.tag:
        result_dir = result_dir.parent / f"{result_dir.name}_{arguments.tag}"
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  group={config['group']}  output={result_dir}", flush=True)

    started = time.time()
    rows, histories = [], {}
    for dataset in config["datasets"]:
        print(f"[{dataset['id']}]", flush=True)
        dataset_rows, dataset_histories = run_dataset(dataset, config, device)
        rows.extend(dataset_rows)
        histories.update(dataset_histories)
        _write_csv(result_dir / "runs.csv", rows)

    table = build_comparison(rows, config)
    _write_csv(result_dir / "comparison.csv", table)
    markdown = comparison_markdown(table, config)
    (result_dir / "comparison.md").write_text(markdown + "\n", encoding="utf-8")
    (result_dir / "training_history.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8")
    (result_dir / "summary.json").write_text(json.dumps({
        "experiment": config["experiment"], "group": config["group"],
        "git_commit": _git_commit(), "device": str(device),
        # sklearn KMeans is thread-count dependent: its chunked distance
        # reduction changes floating-point order, which can move the converged
        # partition.  Measured on halfcylinderRe6400 fmt_all+kin4, mean F1 is
        # .535068 at OMP_NUM_THREADS=4 and .534014 at 8 or 16 with every seed,
        # feature and split held fixed.  All arms in one run share this value,
        # so within-run comparisons are unaffected, but cross-run comparisons
        # are only reproducible to ~1e-3 unless the thread count matches.
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "elapsed_seconds": time.time() - started,
        "row_count": len(rows), "config": config,
        "parameters": next(iter(histories.values()))["parameters"] if histories else None,
    }, indent=2), encoding="utf-8")

    print("\n" + markdown, flush=True)
    print(f"\nwrote {result_dir}", flush=True)


if __name__ == "__main__":
    main()
