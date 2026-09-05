"""Confirm the tuned high-Re Task2 comparison on untouched timeslices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from DeepUtils.utils import EasyConfig
from FMT_Utils.RawPathline_3D import (
    normalize_raw_train_eval, raw_pathline_representation,
)
from Run_Task2_Universality import _fit_cluster, _load_slices
from Verify_HighReVAE import _prepare_representation, _train


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def _reference(records):
    return np.concatenate([record["reference"] for record in records[8:10]])


def _raw_direct(records, source, seed):
    values = [raw_pathline_representation(record["raw"], "center_relative")
              for record in records]
    train, evaluate = np.concatenate(values[:8]), np.concatenate(values[8:10])
    sampled_steps = records[0]["raw"].shape[1] // 21
    train, evaluate = normalize_raw_train_eval(
        train, evaluate, "center_relative", sampled_steps, "pre_group_rms"
    )
    with threadpool_limits(limits=4):
        pca = PCA(n_components=2, svd_solver="randomized", random_state=int(seed)).fit(train)
        train, evaluate = pca.transform(train), pca.transform(evaluate)
    return _fit_cluster(train, evaluate, _reference(records), source, already_scaled=True)[2]


def _fmt_direct(records, source):
    train, evaluate, reference = _prepare_representation(
        records, "fmt", 8, [8, 9], source, fmt_feature_subset="real_neighbor"
    )
    return _fit_cluster(train, evaluate, reference, source, already_scaled=True)[2]


def run(config_path="config/Confirm_HighReTask2_1.1.yaml", resume=False):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    sampling = yaml.safe_load(Path(spec["sampling_spec"]).read_text(encoding="utf-8"))
    source = EasyConfig(str(spec["source_config"]))
    cache = Path(sampling["output_dir"]) / "cache" / spec["sampling_variant"]
    output = Path(spec["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    records = {dataset: _load_slices(cache / dataset) for dataset in spec["datasets"]}
    direct_rows = []
    for dataset in spec["datasets"]:
        for method, score in (("Raw+PCA direct", _raw_direct(records[dataset], source, spec["kmeans_seed"])),
                              ("FMT real-neighbor direct", _fmt_direct(records[dataset], source))):
            direct_rows.append({"dataset": dataset, "method": method, **score})
    _write(output / "direct.csv", direct_rows)

    result_path = output / "vae_runs.csv"; rows = []
    if resume and result_path.exists():
        rows = list(csv.DictReader(result_path.open(encoding="utf-8")))
    completed = {(row["dataset"], row["method"], int(row["training_seed"])) for row in rows}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in spec["datasets"]:
        reference = _reference(records[dataset])
        methods = {
            "Raw+VAE": ("center_relative", spec["raw_vae"][dataset], "pre_group_rms", "all"),
            "FMT+VAE": ("fmt", spec["fmt_vae"], "standard", "real_neighbor"),
        }
        for method, (representation, settings, normalization, subset) in methods.items():
            train_x, eval_x, _ = _prepare_representation(
                records[dataset], representation, 8, [8, 9], source,
                normalization, subset,
            )
            for seed in spec["training_seeds"]:
                key = (dataset, method, int(seed))
                if key in completed:
                    continue
                train_mu, eval_mu, losses = _train(
                    train_x, eval_x, settings, source, int(seed), device
                )
                _, _, score = _fit_cluster(train_mu, eval_mu, reference, source)
                rows.append({"dataset": dataset, "method": method,
                             "training_seed": int(seed), **score, **losses})
                _write(result_path, rows)
                print(f"{dataset}/{method}/seed={seed}: F1={score['f1']:.4f}", flush=True)

    numeric_rows = [{**row, "f1": float(row["f1"]),
                     "training_seed": int(row["training_seed"])} for row in rows]
    summary = {"experiment": spec["experiment"], "device": str(device), "direct": direct_rows,
               "vae": {}, "paired_fmt_gain": {}, "config": spec}
    for dataset in spec["datasets"]:
        summary["vae"][dataset] = {}
        by_method = {}
        for method in ("Raw+VAE", "FMT+VAE"):
            values = [row["f1"] for row in numeric_rows
                      if row["dataset"] == dataset and row["method"] == method]
            by_method[method] = values
            summary["vae"][dataset][method] = {
                "mean_f1": float(np.mean(values)), "std_f1": float(np.std(values)),
                "min_f1": float(np.min(values)), "max_f1": float(np.max(values)),
            }
        gains = np.asarray(by_method["FMT+VAE"]) - np.asarray(by_method["Raw+VAE"])
        summary["paired_fmt_gain"][dataset] = {
            "values": gains.tolist(), "mean": float(gains.mean()),
            "min": float(gains.min()),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Confirm_HighReTask2_1.1.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); run(args.config, args.resume)
