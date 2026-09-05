"""Build and render real 5x-density Task3 local diagnostics.

The dense points are newly integrated pathline primitives, not duplicated,
jittered, or interpolated marks.  The source time, pathline configuration,
frozen network weights, normalization, fusion alpha, and decision threshold
remain unchanged from the audited Task3 visualizations. The current default
reintegrates all 20,480 primitives directly in a physically smaller crop and
draws only exact predicted-positive foreground points. Background and
predicted-negative point arrays are never drawn.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from Build_Channel_Killing_Cache import build as build_channel
from Build_Task2_Universality_Cache import build_dataset
from Build_Task3_LocalCloseup_Cache import build_local_closeup_cache
from DeepUtils.utils import EasyConfig
from Evaluate_Task3_FrozenConfirmation import _evaluate_residual
from Evaluate_Task3_F22_AnchoredConfirmation import _checkpoint
from FMT_Utils.PathlineClassifier_3D import (
    PathlineFMTResidualClassifier3D, residual_model_kwargs,
)
from FMT_Utils.Task12Data_3D import feature_matrix, load_cache_records
from Search_Task3_FMTResidual_3D import _load_spec
from Verify_Task3_FMTResidual import _load_raw_model
from Visualize_Task3_Multiflow_Diagnostic import FLOW_SPECS, _render_one


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get(
    "FMT_TASK3_VIS_CONFIG",
    ROOT / "config/Other_Task3DenseVisualization_4.5.yaml",
))
F22_SEARCH_CONFIG = ROOT / "config/Verify_Task3_F22AnchoredFeatures_1.2_search.yaml"
F22_SELECTION = ROOT / "outputs/Verify_Task3_F22AnchoredFeatures_1.2/anchored_selection.json"
DOWNLOADED_CHECKPOINT_ROOT = (
    ROOT / ".deploy/task3_dense_checkpoints/outputs/"
    "mainExp_Task3_3D_3.2_global_ivd"
)
STANDARD_SEED = 40
F22_SEED = 50


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spec():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _source_group(dataset):
    if dataset == "f22raptor":
        return "f22_anchored"
    if dataset in {"boeing747", "smokeBuoyancy"}:
        return "new2"
    return "old8"


def _dense_source_config(dataset):
    spec = _spec()
    group = _source_group(dataset)
    config = EasyConfig(str(ROOT / spec["source_configs"][group]))
    entries = [item for item in config.datasets if str(item["id"]) == dataset]
    if len(entries) != 1:
        raise RuntimeError(f"{dataset}: expected one source entry, found {len(entries)}")
    config.experiment = spec.get("cache_experiment", spec["experiment"])
    config.datasets = entries
    config.sampling.timeslices = 1
    fixed = EasyConfig()
    fixed.update({dataset: [int(spec["source_indices"][dataset])]})
    config.sampling.fixed_time_indices_by_dataset = fixed
    dense_shape = spec["density"].get("dense_seed_grid_shape")
    if dense_shape is not None:
        config.sampling.seed_grid_shape = list(dense_shape)
    config.output.cache_dir = str(ROOT / spec["outputs"]["cache_root"])
    config.output.result_dir = str(
        ROOT / "outputs/Other_Task3DenseVisualization_1.1/unused_reference"
    )
    return config


def build_dense_cache(dataset, overwrite=False):
    config = _dense_source_config(dataset)
    if _spec().get("local_sampling"):
        return build_local_closeup_cache(
            config, dataset, _spec(), overwrite=bool(overwrite)
        )
    if dataset == "channel":
        return build_channel(config, overwrite=bool(overwrite))
    return build_dataset(config, dataset, overwrite=bool(overwrite))


def _standard_checkpoint_paths(dataset):
    group = "new2" if dataset in {"boeing747", "smokeBuoyancy"} else "old8"
    base = DOWNLOADED_CHECKPOINT_ROOT / f"development_{group}"
    paths = {
        "raw": base / "baselines/checkpoints" / f"{dataset}_raw_seed40.pt",
        "raw_pca": (
            base / "raw_pca_residual/checkpoints"
            / f"{dataset}_raw_pca_residual_seed40.pt"
        ),
        "fmt": (
            base / "fmt_residual/checkpoints"
            / f"{dataset}_raw_fmt_residual_seed40.pt"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing downloaded checkpoints: {missing}")
    return paths


def _f22_checkpoint_paths():
    search = _load_spec(F22_SEARCH_CONFIG)
    selection = json.loads(F22_SELECTION.read_text(encoding="utf-8"))
    candidate = dict(selection["selected_candidate"])
    paths = {
        "raw": (
            ROOT / search["groups"]["f22raptor"]["raw_checkpoint_dir"]
            / f"f22raptor_raw_seed{F22_SEED}.pt"
        ),
        "raw_pca": _checkpoint(search, candidate, F22_SEED, "raw_pca"),
        "fmt": _checkpoint(search, candidate, F22_SEED, "fmt"),
    }
    missing = [str(path) for path in paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing F22 checkpoints: {missing}")
    return paths, candidate


def _load_residual_with_raw(residual_path, raw_path, fmt_dim, device):
    checkpoint = torch.load(
        residual_path, map_location="cpu", weights_only=False
    )
    if checkpoint["variant"] not in {"raw_fmt_residual", "raw_pca_residual"}:
        raise ValueError(f"unexpected residual variant {checkpoint['variant']}")
    auxiliary_dim = int(fmt_dim)
    if checkpoint["variant"] == "raw_pca_residual":
        transform = checkpoint.get("auxiliary_transform")
        if not transform or transform.get("kind") != "raw_pca":
            raise ValueError("Raw-PCA checkpoint misses its PCA transform")
        auxiliary_dim = int(np.asarray(transform["components"]).shape[0])
    raw_model, raw_checkpoint = _load_raw_model(
        raw_path, auxiliary_dim, device
    )
    for key in ("raw_mean", "raw_std"):
        if not np.array_equal(
            np.asarray(checkpoint["normalization"][key]),
            np.asarray(raw_checkpoint["normalization"][key]),
        ):
            raise RuntimeError(
                f"{residual_path}: residual and Raw checkpoints differ on {key}"
            )
    model_spec = checkpoint["config"]["model"]
    model = PathlineFMTResidualClassifier3D(
        raw_model,
        fmt_dim=auxiliary_dim,
        **residual_model_kwargs(model_spec),
    ).to(device)
    state = model.state_dict()
    state.update(checkpoint["residual_state_dict"])
    model.load_state_dict(state)
    return model.eval(), checkpoint


def _dense_record(dataset):
    cache_dir = ROOT / _spec()["outputs"]["cache_root"] / dataset
    return load_cache_records(cache_dir, expected_count=1)[0]


def _dense_split(dataset, device):
    record = _dense_record(dataset)
    steps = record["raw"].shape[1] // (7 * 3)
    raw = record["raw"].reshape(-1, 7, steps, 3)
    feature_name = (
        "aivd2w8" if dataset == "f22raptor"
        else "fmt_all+gram2+kin6"
    )
    fmt = feature_matrix(record, feature_name, device)
    labels = record["reference"].astype(np.float32)
    return record, (raw, fmt, labels), feature_name


def predict_dense(dataset, device):
    spec = _spec()
    record, split, feature_name = _dense_split(dataset, device)
    if dataset == "f22raptor":
        checkpoint_paths, candidate = _f22_checkpoint_paths()
        protocol = "family-specific anchored FMT frozen confirmation"
        training_seed = F22_SEED
        method_names = ("Raw-PCA residual", "Anchored FMT residual")
    else:
        checkpoint_paths = _standard_checkpoint_paths(dataset)
        candidate = {"fmt_feature": feature_name, "id": "all_plus_gram_kinematic"}
        protocol = "global-IVD Task3 frozen main protocol"
        training_seed = STANDARD_SEED
        method_names = ("Raw-PCA residual", "FMT residual")

    predictions = {}
    probabilities_by_method = {}
    metrics = {}
    for key, method in zip(("raw_pca", "fmt"), method_names):
        model, checkpoint = _load_residual_with_raw(
            checkpoint_paths[key], checkpoint_paths["raw"],
            split[1].shape[1], device,
        )
        targets, probabilities, score = _evaluate_residual(
            model, checkpoint, split, 1024, training_seed, device
        )
        if not np.array_equal(targets, split[2].astype(bool)):
            raise RuntimeError(f"{dataset}/{method}: target order changed")
        threshold = float(checkpoint["threshold"])
        predictions[method] = probabilities >= threshold
        probabilities_by_method[method] = probabilities.astype(np.float32)
        metrics[method] = {
            **{name: float(value) for name, value in score.items()},
            "threshold": threshold,
            "alpha": float(checkpoint["alpha"]),
            "training_seed": training_seed,
            "checkpoint": str(checkpoint_paths[key]),
        }

    original_total = int(np.prod(spec["density"]["original_seed_grid_shape"]))
    metadata = dict(record["metadata"])
    actual_valid = int(len(record["reference"]))
    is_local_close_up = bool(spec.get("local_sampling"))
    sampling_audit = {
        "method": (
            "newly integrated local close-up pathline primitives; no "
            "interpolation or duplicated marks"
            if is_local_close_up else
            "newly integrated pathline primitives; no interpolation"
        ),
        "original_total_seed_count": original_total,
        "dense_total_seed_count": int(metadata["total_primitives"]),
        "dense_valid_seed_count": actual_valid,
        "requested_multiplier": float(spec["density"]["requested_multiplier"]),
        "actual_total_seed_multiplier": float(
            metadata["total_primitives"] / original_total
        ),
        "dense_seed_grid_shape": (
            None if spec["density"].get("dense_seed_grid_shape") is None
            else list(spec["density"]["dense_seed_grid_shape"])
        ),
        "sampling_pattern": metadata.get(
            "local_sampling_pattern", "regular_grid"
        ),
        "sampling_seed": metadata.get("local_sampling_seed"),
        "same_source_time_pathline_model_and_threshold": True,
        "visualization_only_local_roi": is_local_close_up,
    }
    payload = {
        "dataset": dataset,
        "protocol": protocol,
        "training_seed": training_seed,
        "candidate": candidate,
        "metadata": metadata,
        "seeds": None,
        "reference": record["reference"].astype(bool),
        "predictions": predictions,
        "probabilities": probabilities_by_method,
        "metrics": metrics,
        "render_experiment": spec["experiment"],
        "close_up_only": bool(spec.get("layout") == "close_up_only"),
        "close_up": dict(spec["close_up"]),
        "sampling_audit": sampling_audit,
        "checkpoint_sha256": {
            name: _sha256(path) for name, path in checkpoint_paths.items()
        },
    }
    if "fixed_close_up_bounds" in metadata:
        payload["fixed_close_up_bounds"] = metadata["fixed_close_up_bounds"]
    with np.load(record["path"]) as cache:
        payload["seeds"] = np.asarray(cache["seeds"], dtype=np.float32)

    prediction_dir = ROOT / spec["outputs"]["prediction_root"]
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{dataset}_task3_dense5x_predictions.npz"
    np.savez_compressed(
        prediction_path,
        seeds=payload["seeds"],
        reference=payload["reference"],
        baseline_prediction=predictions[method_names[0]],
        fmt_prediction=predictions[method_names[1]],
        baseline_probability=probabilities_by_method[method_names[0]],
        fmt_probability=probabilities_by_method[method_names[1]],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        metrics_json=np.asarray(json.dumps(metrics, sort_keys=True)),
        sampling_audit_json=np.asarray(json.dumps(sampling_audit, sort_keys=True)),
    )
    payload["prediction_artifact_sha256"] = _sha256(prediction_path)
    prediction_sidecar = {
        "schema": 1,
        "experiment": spec["experiment"],
        "dataset": dataset,
        "protocol": protocol,
        "training_seed": int(training_seed),
        "candidate": candidate,
        "metrics": metrics,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": payload["prediction_artifact_sha256"],
    }
    prediction_path.with_suffix(".json").write_text(
        json.dumps(prediction_sidecar, indent=2), encoding="utf-8"
    )
    return payload


def load_frozen_dense_prediction(dataset):
    """Reuse an audited dense prediction artifact for render-only revisions.

    This path changes only camera/crop/render settings.  It never reruns a
    classifier and verifies the source artifact against its previous audit
    before exposing the labels and predictions to the renderer.
    """
    spec = _spec()
    prediction_root = spec.get("reuse_prediction_root")
    audit_root = spec.get("reuse_audit_root")
    use_sidecar = bool(spec.get("reuse_prediction_sidecars", False))
    if not prediction_root or (not audit_root and not use_sidecar):
        raise ValueError(
            "render-only reuse requires reuse_prediction_root plus either "
            "reuse_audit_root or reuse_prediction_sidecars=true"
        )
    prediction_path = (
        ROOT / prediction_root / f"{dataset}_task3_dense5x_predictions.npz"
    )
    if use_sidecar:
        audit_path = prediction_path.with_suffix(".json")
    else:
        audit_suffix = (
            "task3_closeup" if spec.get("layout") == "close_up_only"
            else "task3_diagnostic"
        )
        audit_path = ROOT / audit_root / f"{dataset}_{audit_suffix}.json"
    if not prediction_path.exists() or not audit_path.exists():
        raise FileNotFoundError(
            f"missing frozen render source: {prediction_path} or {audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source_sha256 = _sha256(prediction_path)
    if source_sha256 != audit["prediction_artifact_sha256"]:
        raise RuntimeError(
            f"{dataset}: dense prediction artifact changed after its audit"
        )
    method_names = list(audit["metrics"])
    if len(method_names) != 2:
        raise RuntimeError(
            f"{dataset}: expected exactly two audited methods, got {method_names}"
        )
    with np.load(prediction_path) as data:
        metadata = json.loads(str(data["metadata_json"]))
        sampling_audit = json.loads(str(data["sampling_audit_json"]))
        payload = {
            "dataset": dataset,
            "protocol": audit["protocol"],
            "training_seed": int(audit["training_seed"]),
            "candidate": audit["candidate"],
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
            "metrics": audit["metrics"],
            "render_experiment": spec["experiment"],
            "close_up_only": bool(spec.get("layout") == "close_up_only"),
            "close_up": dict(spec["close_up"]),
            "sampling_audit": sampling_audit,
            "checkpoint_sha256": audit["checkpoint_sha256"],
            "prediction_artifact_sha256": source_sha256,
        }
        if audit_root:
            payload["ivd_surface_cache_root"] = str(
                ROOT / audit_root / "ivd_surface_cache"
            )
        if "fixed_close_up_bounds" in metadata:
            payload["fixed_close_up_bounds"] = metadata[
                "fixed_close_up_bounds"
            ]
    lengths = {
        len(payload["seeds"]),
        len(payload["reference"]),
        *(len(values) for values in payload["predictions"].values()),
    }
    if len(lengths) != 1:
        raise RuntimeError(f"{dataset}: frozen render arrays changed length")
    return payload


def _write_manifest(figure_dir):
    audits = []
    for dataset in FLOW_SPECS:
        suffix = (
            "task3_closeup"
            if _spec().get("layout") == "close_up_only"
            else "task3_diagnostic"
        )
        path = figure_dir / f"{dataset}_{suffix}.json"
        if path.exists():
            audits.append(json.loads(path.read_text(encoding="utf-8")))
    rows = []
    for audit in audits:
        raw_name, fmt_name = list(audit["metrics"])
        raw = audit["metrics"][raw_name]
        fmt = audit["metrics"][fmt_name]
        rows.append({
            "dataset": audit["dataset"],
            "sample_count": audit["sample_count"],
            "raw_f1": raw["f1"], "fmt_f1": fmt["f1"],
            "f1_gain": fmt["f1"] - raw["f1"],
            "raw_average_precision": raw["average_precision"],
            "fmt_average_precision": fmt["average_precision"],
            "average_precision_gain": (
                fmt["average_precision"] - raw["average_precision"]
            ),
        })
    csv_path = figure_dir / "metrics_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    density = _spec()["density"]
    local_sampling = _spec().get("local_sampling", {})
    if local_sampling.get("sampling_pattern") == "sobol":
        density_contract = (
            f"new deterministic Sobol sample with "
            f"{int(local_sampling['seed_count'])} locally integrated pathline "
            "primitives; frozen source times, pathline settings, model weights, "
            "normalization, alpha, and thresholds"
        )
    else:
        density_contract = (
            f"new {tuple(density['dense_seed_grid_shape'])} pathline grid, "
            "frozen source times, pathline settings, model weights, "
            "normalization, alpha, and thresholds; local grids are "
            "visualization-only when local_sampling is present"
        )
    manifest = {
        "experiment": _spec()["experiment"],
        "role": "visualization-only dense evaluation; not a replacement main table",
        "density_contract": density_contract,
        "close_up_contract": dict(_spec()["close_up"]),
        "rendered_non_vortex_blue_marks": False,
        "rendered_reference_seed_marks_in_close_up": bool(
            _spec()["close_up"]["show_reference_points"]
        ),
        "rendered_true_positive_marks_in_close_up": bool(
            _spec()["close_up"]["show_true_positives"]
        ),
        "rendered_predicted_positive_marks_in_close_up": bool(
            _spec()["close_up"]["prediction_point_mode"]
            == "predicted_positive"
            and not _spec()["close_up"]["render_prediction_surface"]
        ),
        "rendered_regular_seed_lattice": False,
        "rendered_sparse_paired_error_context_in_close_up": bool(
            _spec()["close_up"]["show_paired_error_context"]
        ),
        "rendered_true_negative_context_color": (
            "neutral gray" if _spec()["close_up"][
                "show_paired_error_context"
            ] else None
        ),
        "close_up_ivd_mesh_clipped_to_roi": True,
        "metric_population": (
            "all valid local close-up seeds; visualization diagnostic only"
            if _spec().get("local_sampling") else
            "all valid dense seeds, including true negatives"
        ),
        "datasets": audits,
        "metrics_summary_csv": str(csv_path),
    }
    (figure_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run(datasets, stage="all", overwrite=False, dpi=300):
    unknown = sorted(set(datasets) - set(FLOW_SPECS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    reuse_frozen = bool(_spec().get("reuse_prediction_root"))
    if stage == "cache" and reuse_frozen:
        raise ValueError(
            "this render-only revision reuses frozen predictions and has no "
            "new cache stage"
        )
    if stage in {"cache", "all"} and not reuse_frozen:
        for dataset in datasets:
            build_dense_cache(dataset, overwrite=overwrite)
    if stage == "cache":
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figure_dir = ROOT / _spec()["outputs"]["figure_root"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        payload = (
            load_frozen_dense_prediction(dataset)
            if reuse_frozen
            else predict_dense(dataset, device)
        )
        if stage in {"render", "all"}:
            audit = _render_one(dataset, payload, figure_dir, int(dpi))
            methods = list(audit["metrics"])
            print(
                f"{dataset}: n={audit['sample_count']}, "
                f"{methods[0]} F1={audit['metrics'][methods[0]]['f1']:.3f}, "
                f"{methods[1]} F1={audit['metrics'][methods[1]]['f1']:.3f}",
                flush=True,
            )
    if stage in {"render", "all"}:
        return _write_manifest(figure_dir)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=list(
            _spec().get("datasets", FLOW_SPECS)
        ),
        choices=list(FLOW_SPECS),
    )
    parser.add_argument(
        "--stage", choices=("cache", "predict", "render", "all"),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args()
    run(
        arguments.datasets, stage=arguments.stage,
        overwrite=arguments.overwrite, dpi=arguments.dpi,
    )
