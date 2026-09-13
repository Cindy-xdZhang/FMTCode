"""Replay frozen recipes and export exact seedwise predictions for figures.

Run from the existing frozen uniform-experiment checkout on Ibex. This wrapper
does not edit the original training implementation, selection or result files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import load_cache_records, stack_features, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    fit_kmeans_transform, calibrate_vortex_cluster, binary_cluster_metrics,
)
from experiments import Run_UniformFMT_Confirmation_3D as uniform
from experiments import Search_Task2_FMTVAE_3D as search2
from experiments import Search_Task3_FMTResidual_3D as search3
from experiments.Run_Task2_3D_Main import _prepare_inputs
from experiments.Verify_HighReVAE import _train
from experiments.Evaluate_Task3_FrozenConfirmation import _load_residual, _evaluate_residual


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_payload(output, task, dataset, record, predictions, info):
    with np.load(record["path"]) as cache:
        seeds = np.asarray(cache["seeds"], dtype=np.float32)
    reference = np.asarray(record["reference"], dtype=bool)
    assert seeds.shape == (len(reference), 3)
    for value in predictions.values():
        assert np.asarray(value).shape == reference.shape
    info.update({
        "task": task, "dataset": dataset, "metadata": record["metadata"],
        "sample_count": len(reference), "source_cache_sha256": digest(record["path"]),
        "source_cache": str(record["path"]),
        "seedwise_metrics": {
            arm: binary_cluster_metrics(reference, np.asarray(pred), 1)
            for arm, pred in predictions.items()
        },
    })
    target = output / f"{dataset}_{task}_predictions.npz"
    np.savez_compressed(target, seeds=seeds, reference=reference,
                        raw_prediction=predictions["raw"], fmt_prediction=predictions["fmt"],
                        metadata_json=json.dumps(info, sort_keys=True))
    info["prediction_sha256"] = digest(target)
    target.with_suffix(".json").write_text(json.dumps(info, indent=2, sort_keys=True))
    print("EXPORTED", target, info["seedwise_metrics"], flush=True)


def task1(dataset, output, settings, device):
    config = Path("config/mainExp_Task1_3D_4.1_uniform.yaml")
    spec = yaml.safe_load(config.read_text())
    selection_path = Path(spec["output_dir"]) / "selected_config.json"
    selected = json.loads(selection_path.read_text())
    assert selected["config_sha256"] == digest(config)
    assert not selected["confirmation_opened"]
    source = next(v for v in spec["sources"].values() if dataset in v["datasets"])
    development = load_cache_records(Path(source["development_cache"]) / dataset, 10)
    confirmation = load_cache_records(Path(source["confirmation_cache"]) / dataset, 4)
    train = [development[i] for i in spec["splits"]["final_train"]]
    calibration = [development[i] for i in spec["splits"]["cluster_calibration"]]
    ordinal = settings["display_ordinal"]
    predictions, full_metrics = {}, {}
    for arm in ("raw", "fmt"):
        choice = selected[arm]
        dim = None if choice["pca_dim"] == "none" else int(choice["pca_dim"])
        feature = choice["feature"]
        model = fit_kmeans_transform(stack_features(train, feature, device), dim,
                                     settings["seed"], int(spec["kmeans_n_init"]))
        mapping = calibrate_vortex_cluster(stack_reference(calibration),
                  model.predict(stack_features(calibration, feature, device)))
        labels = model.predict(stack_features(confirmation, feature, device))
        full_metrics[arm] = binary_cluster_metrics(stack_reference(confirmation), labels, mapping)
        start = sum(len(r["reference"]) for r in confirmation[:ordinal])
        count = len(confirmation[ordinal]["reference"])
        predictions[arm] = labels[start:start+count] == mapping
    write_payload(output, "task1", dataset, confirmation[ordinal], predictions,
                  {**settings, "selection_sha256": digest(selection_path),
                   "replay_all_confirmation_metrics": full_metrics})


def task2(dataset, output, settings, device):
    config = "config/mainExp_Task2_3D_6.2_uniform_confirmation.yaml"
    manifest, _, spec, source, selected = uniform._frozen_state(config)
    winner = selected["winner"]
    architecture = uniform._task2_architecture(source, winner)
    _, group = search2._group_for_dataset(source, dataset)
    records = load_cache_records(Path(group["development_cache"]) / dataset, 10)
    train = [records[i] for i in spec["splits"]["train"]]
    calibration = [records[i] for i in spec["splits"]["cluster_calibration"]]
    confirmation = [records[i] for i in spec["splits"]["confirmation"]]
    evaluate = calibration + confirmation
    ncal = sum(len(r["reference"]) for r in calibration)
    predictions, full_metrics = {}, {}
    for arm in ("raw", "fmt"):
        train_x, eval_x = _prepare_inputs(train, evaluate, arm, winner["fmt_feature"], device)
        train_mu, eval_mu, losses = _train(train_x, eval_x, architecture,
                                          EasyConfig(group["source_config"]), settings["seed"], device)
        model = KMeans(n_clusters=2, random_state=spec["kmeans_seed"],
                       n_init=spec["kmeans_n_init"]).fit(train_mu)
        mapping = calibrate_vortex_cluster(stack_reference(calibration), model.predict(eval_mu[:ncal]))
        labels = model.predict(eval_mu[ncal:])
        full_metrics[arm] = binary_cluster_metrics(stack_reference(confirmation), labels, mapping)
        i = spec["splits"]["confirmation"].index(settings["display_ordinal"])
        start = sum(len(r["reference"]) for r in confirmation[:i])
        predictions[arm] = labels[start:start+len(confirmation[i]["reference"])] == mapping
    write_payload(output, "task2", dataset, records[settings["display_ordinal"]], predictions,
                  {**settings, "recipe_manifest_sha256": digest(manifest),
                   "replay_all_confirmation_metrics": full_metrics})


def task3(dataset, output, settings, device):
    config = "config/mainExp_Task3_3D_9.2_uniform_confirmation.yaml"
    manifest, _, spec, source, selected = uniform._frozen_state(config)
    candidate = uniform._task3_candidate(source, selected["winner"])
    _, group = search3._group_for_dataset(source, dataset)
    records = search3._load_records(source, dataset, candidate, device, ordinals=list(range(10)))
    original = [search3._stack_split(records, spec["splits"][name])
                for name in ("train", "validation", "confirmation")]
    stats_raw = search3._frozen_raw_normalization(group, dataset, settings["seed"])
    train, validation, confirmation, stats = search3._normalize_train_only(*original, raw_stats=stats_raw)
    fmt_dim = train[1].shape[1]
    predictions, full_metrics, training_rows, checkpoint_hashes = {}, {}, {}, {}
    paths = []
    for source_name, arm in (("raw_pca", "raw"), ("fmt", "fmt")):
        run_dir = output / "temporary_training" / dataset / source_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_spec = search3._candidate_spec(source, group, candidate, dataset,
                                         settings["seed"], source_name, run_dir, fmt_dim)
        run_spec["experiment"] = "Other_Task123_PaperTriptychs_1.1"
        run_spec["split"] = {"train_ordinals": spec["splits"]["train"],
                             "validation_ordinals": spec["splits"]["validation"],
                             "test_ordinals": spec["splits"]["confirmation"]}
        run_spec["evaluation"] = {"test_enabled": True}
        row = search3._train_one(run_spec, dataset, settings["seed"],
                                (train, validation, confirmation), stats, device, run_dir)
        variant = "raw_pca_residual" if arm == "raw" else "raw_fmt_residual"
        path = run_dir / "checkpoints" / f"{dataset}_{variant}_seed{settings['seed']}.pt"
        paths.append(path)
        checkpoint_hashes[arm] = digest(path)
        model, checkpoint = _load_residual(path, fmt_dim, device)
        targets, probabilities, score = _evaluate_residual(model, checkpoint, original[2],
                                                          2048, settings["seed"], device)
        assert np.array_equal(targets, original[2][2])
        full_metrics[arm], training_rows[arm] = score, row
        # Ordinal 8 is the first registered confirmation block, never selected by score.
        assert settings["display_ordinal"] == spec["splits"]["confirmation"][0]
        count = len(next(r for r in records if r[3] == settings["display_ordinal"])[2])
        predictions[arm] = probabilities[:count] >= checkpoint["threshold"]
        del model
        torch.cuda.empty_cache()
    record = load_cache_records(Path(group["source_cache_root"]) / dataset, 10,
                                [settings["display_ordinal"]])[0]
    assert np.array_equal(record["reference"], original[2][2][:len(record["reference"])])
    write_payload(output, "task3", dataset, record, predictions,
                  {**settings, "recipe_manifest_sha256": digest(manifest),
                   "replay_all_confirmation_metrics": full_metrics,
                   "training_rows": training_rows, "temporary_checkpoint_sha256": checkpoint_hashes})
    for path in paths:
        path.unlink()
    (output / f"{dataset}_task3_cleanup.json").write_text(json.dumps({
        "deleted_checkpoint_count": len(paths),
        "remaining_checkpoints": [str(p) for p in (output/"temporary_training"/dataset).rglob("*.pt")],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=("task1", "task2", "task3"), required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    assert args.dataset in spec["datasets"]
    out = Path(spec["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = {"device": str(device), "torch": torch.__version__, "numpy": np.__version__,
            "hostname": os.uname().nodename, "job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "exporter_sha256": digest(__file__), "config_sha256": digest(args.config)}
    (out / f"{args.dataset}_{args.task}_runtime.json").write_text(json.dumps(info, indent=2))
    with threadpool_limits(limits=4):
        globals()[args.task](args.dataset, out, spec[args.task], device)


if __name__ == "__main__":
    main()
