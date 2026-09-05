"""Render genuine dense local Task2 and Task3 macro-close-up triptychs.

The prediction panels use 20,480 newly integrated Sobol pathline primitives
from the frozen local caches. The IVD reference is a continuous surface; Raw
and FMT draw only exact predicted-vortex support in red. Every negative or
background scatter layer is structurally disabled, so a blue background array
cannot occur. Saturated blue/cyan pixels are rejected.
Task2 retrains the frozen same-family VAE recipe on the original development
slices, calibrates cluster identity on the original calibration slices, and
only replaces the evaluated population with the dense local cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import (
    load_cache_records,
    stack_reference,
)
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
)
from Run_Task2_3D_Main import _architecture, _cache_dir, _prepare_inputs
from Verify_HighReVAE import _train
from Visualize_Task3_Multiflow_Diagnostic import FLOW_SPECS, _render_one


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get(
    "FMT_TASK23_DENSE_VIS_CONFIG",
    ROOT / "config/Other_Task23DenseCloseup_2.8.yaml",
))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spec():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _dense_record(dataset):
    cache_dir = ROOT / _spec()["dense_cache_root"] / dataset
    return load_cache_records(cache_dir, expected_count=1)[0]


def _record_seeds(record):
    with np.load(record["path"]) as data:
        seeds = np.asarray(data["seeds"], dtype=np.float32)
    if seeds.shape != (len(record["reference"]), 3):
        raise RuntimeError(f"seed/reference mismatch in {record['path']}")
    return seeds


def _task2_artifact_path(dataset):
    root = ROOT / _spec()["outputs"]["prediction_root"]
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{dataset}_task2_dense_predictions.npz"


def _load_task2_artifact(dataset):
    path = _task2_artifact_path(dataset)
    sidecar_path = path.with_suffix(".json")
    if not path.exists() or not sidecar_path.exists():
        return None
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if _sha256(path) != sidecar["prediction_artifact_sha256"]:
        raise RuntimeError(f"{dataset}: Task2 dense artifact changed after audit")
    task2_config = ROOT / _spec()["task2_config"]
    if _sha256(task2_config) != sidecar["task2_config_sha256"]:
        raise RuntimeError(f"{dataset}: frozen Task2 config changed")
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata_json"]))
        stored_metrics = json.loads(str(data["metrics_json"]))
        sampling_audit = json.loads(str(data["sampling_audit_json"]))
        method_names = json.loads(str(data["method_names_json"]))
        expected_method_names = ["Raw+VAE", "FMT+VAE"]
        if method_names != expected_method_names:
            raise RuntimeError(
                f"{dataset}: unexpected Task2 method order {method_names}"
            )
        if set(stored_metrics) != set(method_names):
            raise RuntimeError(
                f"{dataset}: Task2 metric methods differ from predictions"
            )
        # metrics_json is intentionally serialized with sort_keys=True, which
        # alphabetizes FMT before Raw. Restore the experimental comparison
        # order so audit manifests compute Raw -> FMT gains in the right
        # direction; prediction arrays were already mapped correctly.
        metrics = {name: stored_metrics[name] for name in method_names}
        baseline_prediction = np.asarray(
            data["baseline_prediction"], dtype=bool
        )
        fmt_prediction = np.asarray(data["fmt_prediction"], dtype=bool)
        payload = {
            "task": "task2",
            "dataset": dataset,
            "protocol": sidecar["protocol"],
            "candidate": sidecar["candidate"],
            "training_seed": int(sidecar["training_seed"]),
            "confirmation_ordinal": int(sidecar["source_ordinal"]),
            "metadata": metadata,
            "seeds": np.asarray(data["seeds"], dtype=np.float32),
            "reference": np.asarray(data["reference"], dtype=bool),
            "predictions": {
                method_names[0]: baseline_prediction,
                method_names[1]: fmt_prediction,
            },
            # Task2 KMeans exposes cluster assignments rather than calibrated
            # probabilities.  For a point-free qualitative rendering only,
            # interpolate the exact frozen binary decisions as an occupancy
            # field. Seedwise metrics and saved predictions remain untouched.
            "probabilities": {
                method_names[0]: baseline_prediction.astype(np.float32),
                method_names[1]: fmt_prediction.astype(np.float32),
            },
            "scalar_field_provenance": (
                "render-only inverse-distance interpolation of frozen binary "
                "KMeans vortex decisions; not a model probability"
            ),
            "render_thresholds": {
                method_names[0]: 0.5,
                method_names[1]: 0.5,
            },
            "metrics": metrics,
            "sampling_audit": sampling_audit,
            "prediction_artifact_sha256": sidecar[
                "prediction_artifact_sha256"
            ],
        }
    if "fixed_close_up_bounds" in metadata:
        payload["fixed_close_up_bounds"] = metadata["fixed_close_up_bounds"]
    return payload


def _build_task2_payload(dataset, device, recompute=False):
    if not recompute:
        loaded = _load_task2_artifact(dataset)
        if loaded is not None:
            return loaded

    render_spec = _spec()
    task2_config = ROOT / render_spec["task2_config"]
    task2 = yaml.safe_load(task2_config.read_text(encoding="utf-8"))
    matching_groups = [
        name for name, value in task2["groups"].items()
        if dataset in value["datasets"]
    ]
    if len(matching_groups) != 1:
        raise RuntimeError(
            f"{dataset}: expected one Task2 group, got {matching_groups}"
        )
    group = matching_groups[0]
    group_spec = task2["groups"][group]
    development = load_cache_records(
        _cache_dir(task2, "development", dataset), expected_count=10
    )
    train_records = [
        development[index] for index in task2["splits"]["final_train"]
    ]
    calibration_records = [
        development[index]
        for index in task2["splits"]["cluster_calibration"]
    ]
    dense = _dense_record(dataset)
    evaluate_records = [*calibration_records, dense]
    calibration_count = sum(
        len(record["reference"]) for record in calibration_records
    )
    calibration_reference = stack_reference(calibration_records)
    architecture = _architecture(task2, group_spec["fixed_architecture"])
    source = EasyConfig(str(ROOT / task2["source_config"]))
    training_seed = int(render_spec["task2_training_seed"])
    method_names = ("Raw+VAE", "FMT+VAE")
    predictions = {}
    metrics = {}

    for method, display_name in zip(("raw", "fmt"), method_names):
        train_x, evaluate_x = _prepare_inputs(
            train_records,
            evaluate_records,
            method,
            group_spec["fmt_feature"],
            device,
        )
        train_mu, evaluate_mu, losses = _train(
            train_x,
            evaluate_x,
            architecture,
            source,
            training_seed,
            device,
        )
        kmeans = KMeans(
            n_clusters=2,
            random_state=int(task2["kmeans_seed"]),
            n_init=int(task2["kmeans_n_init"]),
        ).fit(train_mu)
        calibration_labels = kmeans.predict(evaluate_mu[:calibration_count])
        vortex_cluster = calibrate_vortex_cluster(
            calibration_reference, calibration_labels
        )
        dense_labels = kmeans.predict(evaluate_mu[calibration_count:])
        predictions[display_name] = dense_labels == vortex_cluster
        metrics[display_name] = {
            **binary_cluster_metrics(
                dense["reference"], dense_labels, vortex_cluster
            ),
            "vortex_cluster": int(vortex_cluster),
            "architecture": architecture["id"],
            "training_seed": training_seed,
            "train_seconds": float(losses["train_seconds"]),
            "metric_population": "dense local visualization cache",
        }

    seeds = _record_seeds(dense)
    metadata = dict(dense["metadata"])
    sampling_audit = {
        "method": (
            "20,480 newly integrated local Sobol pathline primitives; "
            "no interpolation, copying, or jitter"
        ),
        "local_seed_count": int(len(seeds)),
        "sampling_pattern": metadata.get("local_sampling_pattern", "sobol"),
        "sampling_seed": metadata.get("local_sampling_seed"),
        "same_training_and_calibration_records_as_main_task2": True,
        "same_family_vae_for_raw_and_fmt": True,
        "metrics_are_local_visualization_diagnostics": True,
    }
    payload = {
        "task": "task2",
        "dataset": dataset,
        "protocol": (
            "frozen same-family Task2 VAE and calibration; dense local "
            "evaluation only"
        ),
        "candidate": {
            "group": group,
            "architecture": architecture["id"],
            "fmt_feature": group_spec["fmt_feature"],
        },
        "training_seed": training_seed,
        "confirmation_ordinal": int(dense.get("ordinal", 0)),
        "metadata": metadata,
        "seeds": seeds,
        "reference": np.asarray(dense["reference"], dtype=bool),
        "predictions": predictions,
        "probabilities": {
            method: prediction.astype(np.float32)
            for method, prediction in predictions.items()
        },
        "scalar_field_provenance": (
            "render-only inverse-distance interpolation of frozen binary "
            "KMeans vortex decisions; not a model probability"
        ),
        "render_thresholds": {
            method: 0.5 for method in method_names
        },
        "metrics": metrics,
        "sampling_audit": sampling_audit,
    }
    if "fixed_close_up_bounds" in metadata:
        payload["fixed_close_up_bounds"] = metadata["fixed_close_up_bounds"]

    artifact_path = _task2_artifact_path(dataset)
    np.savez_compressed(
        artifact_path,
        seeds=seeds,
        reference=payload["reference"],
        baseline_prediction=predictions[method_names[0]],
        fmt_prediction=predictions[method_names[1]],
        method_names_json=np.asarray(json.dumps(method_names)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        metrics_json=np.asarray(json.dumps(metrics, sort_keys=True)),
        sampling_audit_json=np.asarray(json.dumps(sampling_audit, sort_keys=True)),
    )
    sidecar = {
        "schema": 1,
        "experiment": render_spec["experiment"],
        "dataset": dataset,
        "task": "task2",
        "protocol": payload["protocol"],
        "candidate": payload["candidate"],
        "training_seed": training_seed,
        "source_ordinal": int(payload["confirmation_ordinal"]),
        "task2_config": str(task2_config),
        "task2_config_sha256": _sha256(task2_config),
        "prediction_artifact": str(artifact_path),
        "prediction_artifact_sha256": _sha256(artifact_path),
        "metrics": metrics,
    }
    artifact_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    payload["prediction_artifact_sha256"] = sidecar[
        "prediction_artifact_sha256"
    ]
    return payload


def _load_task3_payload(dataset):
    render_spec = _spec()
    root = ROOT / render_spec["task3_prediction_root"]
    path = root / f"{dataset}_task3_dense5x_predictions.npz"
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists() and render_spec.get("task3_audit_root"):
        sidecar_path = (
            ROOT / render_spec["task3_audit_root"]
            / f"{dataset}_task3_closeup.json"
        )
    if not path.exists() or not sidecar_path.exists():
        raise FileNotFoundError(f"missing frozen Task3 artifact for {dataset}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    artifact_sha256 = _sha256(path)
    if artifact_sha256 != sidecar["prediction_artifact_sha256"]:
        raise RuntimeError(f"{dataset}: frozen Task3 artifact changed")
    method_names = list(sidecar["metrics"])
    if len(method_names) != 2 or method_names[0] != "Raw-PCA residual":
        raise RuntimeError(f"{dataset}: unexpected Task3 methods {method_names}")
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata_json"]))
        sampling_audit = json.loads(str(data["sampling_audit_json"]))
        payload = {
            "task": "task3",
            "dataset": dataset,
            "protocol": sidecar["protocol"],
            "candidate": sidecar["candidate"],
            "training_seed": int(sidecar["training_seed"]),
            "confirmation_ordinal": int(sidecar.get(
                "confirmation_ordinal", 0
            )),
            "metadata": metadata,
            "seeds": np.asarray(data["seeds"], dtype=np.float32),
            "reference": np.asarray(data["reference"], dtype=bool),
            "predictions": {
                method_names[0]: np.asarray(
                    data["baseline_prediction"], dtype=bool
                ),
                method_names[1]: np.asarray(data["fmt_prediction"], dtype=bool),
            },
            "probabilities": {
                method_names[0]: np.asarray(
                    data["baseline_probability"], dtype=np.float32
                ),
                method_names[1]: np.asarray(
                    data["fmt_probability"], dtype=np.float32
                ),
            },
            "metrics": sidecar["metrics"],
            "sampling_audit": sampling_audit,
            "checkpoint_sha256": sidecar["checkpoint_sha256"],
            "prediction_artifact_sha256": artifact_sha256,
        }
    if "fixed_close_up_bounds" in metadata:
        payload["fixed_close_up_bounds"] = metadata["fixed_close_up_bounds"]
    return payload


def _decorate(payload, render_spec=None):
    spec = _spec() if render_spec is None else render_spec
    task = str(payload["task"])
    close_up = dict(spec["close_up"])
    close_up.update(spec.get(f"{task}_close_up", {}))
    payload["render_experiment"] = spec["experiment"]
    payload["close_up_only"] = True
    payload["close_up"] = close_up
    payload["ivd_surface_cache_root"] = str(
        ROOT / spec["outputs"]["ivd_surface_cache_root"]
    )
    return payload


def _write_manifest(task, figure_dir, datasets):
    audits = []
    # A partial rerender must not erase already audited flows from the shared
    # manifest.  Keep every configured dataset whose current-version sidecar
    # exists, while preserving the configured physical-family order.
    requested = set(datasets)
    for dataset in _spec()["datasets"]:
        path = figure_dir / f"{dataset}_{task}_closeup.json"
        if path.exists():
            audits.append(json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        "experiment": _spec()["experiment"],
        "task": task,
        "role": "dense local visualization diagnostic; not main-table evidence",
        "sampling": dict(_spec()["density"]),
        "crop": {
            **dict(_spec()["close_up"]),
            **dict(_spec().get(f"{task}_close_up", {})),
        },
        "blue_background_point_layer": False,
        "predicted_vortex_point_layer": (
            _spec()["close_up"]["prediction_point_mode"]
            == "predicted_positive"
        ),
        "all_rendered_blue_hue_pixel_counts_zero": all(
            int(item["rendered_blue_hue_pixel_count"]) == 0 for item in audits
        ),
        "requested_in_last_run": sorted(requested),
        "datasets": audits,
    }
    (figure_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run(tasks, datasets, *, recompute_task2=False, dpi=450):
    unknown = sorted(set(datasets) - set(FLOW_SPECS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = _spec()
    for task in tasks:
        figure_dir = ROOT / spec["outputs"][f"{task}_figure_root"]
        figure_dir.mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            payload = (
                _build_task2_payload(dataset, device, recompute_task2)
                if task == "task2" else _load_task3_payload(dataset)
            )
            audit = _render_one(
                dataset, _decorate(payload), figure_dir, int(dpi)
            )
            methods = list(audit["metrics"])
            print(
                f"{task}/{dataset}: local n={audit['sample_count']}, "
                f"visible={audit['close_up_visible_seed_count']}, "
                f"{methods[0]} F1={audit['metrics'][methods[0]]['f1']:.3f}, "
                f"{methods[1]} F1={audit['metrics'][methods[1]]['f1']:.3f}",
                flush=True,
            )
        _write_manifest(task, figure_dir, datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=("task2", "task3"),
        default=("task2", "task3"),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(FLOW_SPECS),
        default=_spec()["datasets"],
    )
    parser.add_argument("--recompute-task2", action="store_true")
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    run(
        args.tasks,
        args.datasets,
        recompute_task2=args.recompute_task2,
        dpi=args.dpi,
    )
