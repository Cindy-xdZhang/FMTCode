"""Render Task5 variable-scale supervised results as clean 1x3 figures.

Each figure uses one fixed confirmation slice and one fixed training seed:

1. audited whole-field IVD-p95 isosurface plus its exact seed labels;
2. variable-scale Raw-PCA residual prediction;
3. variable-scale Raw+FMT residual prediction.

All three panels share the same orthographic camera, physical bounds, axis
labels, and coordinate ticks.  Titles, legends, and colorbars are intentionally
omitted for later paper composition.  Model thresholds and weights are loaded
unchanged from the frozen development checkpoints; confirmation data are used
only for inference and rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from Evaluate_Task3_FrozenConfirmation import (
    _evaluate_residual,
    _find_checkpoint,
    _load_residual,
)
from Verify_Task3_FMTClassifier import _load_dataset, _stack_split
from Visualize_Task23_3D_Horizontal import (
    DEFAULT_DPI,
    FLOW_SPECS,
    POINT_DENSITY_MULTIPLIER,
    _attach_ivd_reference,
    _render_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/mainExp_Task5_3D_1.1_evaluate.yaml"
OUTPUT = ROOT / "outputs/Task5_3D_horizontal_main_1.1"

DEFAULT_DATASETS = (
    "cylinder3d",
    "halfcylinderRe640",
    "halfcylinderRe6400",
    "channel",
    "f22raptor",
    "tangaroa",
    "smokeBuoyancy",
    "deltaWing_LBM",
)
DEFAULT_ORDINAL = 2
DEFAULT_SEED = 40


def _matching_group(spec, dataset):
    groups = [group for group in spec["groups"] if dataset in group["datasets"]]
    if len(groups) != 1:
        raise RuntimeError(
            f"expected one Task5 group for {dataset}, found {len(groups)}"
        )
    return groups[0]


def _artifact_path(dataset, ordinal, seed):
    return OUTPUT / (
        f"{dataset}_task5_slice{ordinal:02d}_seed{seed}_predictions.npz"
    )


def _save_artifact(path, payload):
    np.savez_compressed(
        path,
        seeds=np.asarray(payload["seeds"], dtype=np.float32),
        reference=np.asarray(payload["reference"], dtype=bool),
        baseline_prediction=np.asarray(payload["predictions"][0], dtype=bool),
        fmt_prediction=np.asarray(payload["predictions"][1], dtype=bool),
        baseline_probability=np.asarray(
            payload["probabilities"][0], dtype=np.float32
        ),
        fmt_probability=np.asarray(payload["probabilities"][1], dtype=np.float32),
        scale_id=np.asarray(payload["scale_id"], dtype=np.int16),
        metadata_json=np.array(json.dumps(payload["metadata"], sort_keys=True)),
        metrics_json=np.array(json.dumps(payload["metrics"], sort_keys=True)),
    )


def _load_artifact(path):
    with np.load(path) as data:
        return {
            "seeds": np.asarray(data["seeds"], dtype=np.float32),
            "reference": np.asarray(data["reference"], dtype=bool),
            "predictions": [
                np.asarray(data["baseline_prediction"], dtype=bool),
                np.asarray(data["fmt_prediction"], dtype=bool),
            ],
            "probabilities": [
                np.asarray(data["baseline_probability"], dtype=np.float32),
                np.asarray(data["fmt_probability"], dtype=np.float32),
            ],
            "scale_id": np.asarray(data["scale_id"], dtype=np.int16),
            "metadata": json.loads(str(data["metadata_json"])),
            "metrics": json.loads(str(data["metrics_json"])),
        }


def _predict_dataset(spec, dataset, ordinal, seed, device, recompute=False):
    artifact = _artifact_path(dataset, ordinal, seed)
    if artifact.exists() and not recompute:
        return _load_artifact(artifact)

    group = _matching_group(spec, dataset)
    source_dir = ROOT / group["source_cache_root"] / dataset
    label_dir = ROOT / group["label_cache_root"] / dataset
    expected_slices = int(group["expected_slices"])
    if not 0 <= int(ordinal) < expected_slices:
        raise ValueError(
            f"confirmation ordinal {ordinal} is outside [0,{expected_slices})"
        )
    records = _load_dataset(
        source_dir,
        label_dir,
        sampled_steps=int(spec["sampled_steps"]),
        fmt_subset=spec["fmt_subset"],
        required_ordinals={int(ordinal)},
        gram_num_freq=int(spec["fmt_gram_num_freq"]),
        expected_slices=expected_slices,
    )
    split = _stack_split(records, [int(ordinal)])
    checkpoint_paths = (
        _find_checkpoint(
            [ROOT / value for value in group["raw_pca_checkpoint_roots"]],
            f"{dataset}_raw_pca_residual_seed{seed}.pt",
        ),
        _find_checkpoint(
            [ROOT / value for value in group["fmt_checkpoint_roots"]],
            f"{dataset}_raw_fmt_residual_seed{seed}.pt",
        ),
    )
    names = ("Raw-PCA residual", "Raw+FMT residual")
    predictions = []
    probabilities = []
    metrics = {}
    reference = None
    for name, checkpoint_path in zip(names, checkpoint_paths):
        model, checkpoint = _load_residual(
            checkpoint_path, split[1].shape[1], device
        )
        targets, probability, score = _evaluate_residual(
            model,
            checkpoint,
            split,
            int(spec["batch_size"]),
            int(seed),
            device,
        )
        if reference is None:
            reference = np.asarray(targets, dtype=bool)
        elif not np.array_equal(reference, np.asarray(targets, dtype=bool)):
            raise RuntimeError(f"Task5 methods disagree on targets for {dataset}")
        threshold = float(checkpoint["threshold"])
        probabilities.append(np.asarray(probability, dtype=np.float32))
        predictions.append(np.asarray(probability >= threshold, dtype=bool))
        metrics[name] = {
            **{key: float(value) for key, value in score.items()},
            "threshold": threshold,
            "alpha": float(checkpoint["alpha"]),
            "training_seed": int(seed),
            "checkpoint": str(checkpoint_path),
        }

    source_path = sorted(source_dir.glob("slice_*.npz"))[int(ordinal)]
    with np.load(source_path) as source:
        seeds = np.asarray(source["seeds"], dtype=np.float32)
        source_reference = np.asarray(source["reference"], dtype=bool)
        scale_id = np.asarray(source["scale_id"], dtype=np.int16)
        metadata = json.loads(str(source["metadata_json"]))
    if not (
        len(seeds) == len(scale_id) == len(reference)
        and np.array_equal(source_reference, reference)
    ):
        raise RuntimeError(f"Task5 cache/label/prediction mismatch for {dataset}")

    payload = {
        "seeds": seeds,
        "reference": reference,
        "predictions": predictions,
        "probabilities": probabilities,
        "scale_id": scale_id,
        "metadata": metadata,
        "metrics": metrics,
    }
    _save_artifact(artifact, payload)
    return payload


def _metadata_record(dataset, ordinal, seed, payload, image_info):
    path, rectangles, pixel_size, counts, sampling = image_info
    baseline = payload["metrics"]["Raw-PCA residual"]
    fmt = payload["metrics"]["Raw+FMT residual"]
    return {
        "dataset": dataset,
        "title": FLOW_SPECS[dataset]["title"],
        "task": "Task5",
        "confirmation_ordinal": int(ordinal),
        "source_index": int(payload["metadata"]["source_start_index"]),
        "source_time": float(payload["metadata"]["source_time"]),
        "training_seed": int(seed),
        "image": str(path),
        "pixel_size": pixel_size,
        "panel_rectangles": [list(value) for value in rectangles],
        "panel_order": [
            "IVD-p95 ground truth",
            "variable-scale Raw-PCA residual",
            "variable-scale Raw+FMT residual",
        ],
        "camera": {
            "projection": "orthographic",
            "elevation_degrees": FLOW_SPECS[dataset]["view"][0],
            "azimuth_degrees": FLOW_SPECS[dataset]["view"][1],
            "physical_bounds": payload["bounds"].tolist(),
        },
        "reference_positive_fraction": float(payload["reference"].mean()),
        "metrics": payload["metrics"],
        "slice_f1_gain": float(fmt["f1"] - baseline["f1"]),
        "slice_average_precision_gain": float(
            fmt["average_precision"] - baseline["average_precision"]
        ),
        "confusion_counts": counts,
        "render_sampling": sampling,
        "ivd_surface_audit": payload["ivd_surface_audit"],
        "scale_table": payload["metadata"]["scale_table"],
    }


def run(
    datasets=DEFAULT_DATASETS,
    ordinal=DEFAULT_ORDINAL,
    seed=DEFAULT_SEED,
    dpi=DEFAULT_DPI,
    point_density_multiplier=POINT_DENSITY_MULTIPLIER,
    recompute=False,
    render_only=False,
):
    unknown = sorted(set(datasets) - set(FLOW_SPECS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if int(seed) not in {int(value) for value in spec["seeds"]}:
        raise ValueError(f"seed {seed} is not one of the frozen Task5 seeds")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for dataset in datasets:
        artifact = _artifact_path(dataset, int(ordinal), int(seed))
        if render_only:
            if not artifact.exists():
                raise FileNotFoundError(artifact)
            payload = _load_artifact(artifact)
        else:
            payload = _predict_dataset(
                spec, dataset, int(ordinal), int(seed), device, recompute
            )
        _attach_ivd_reference(dataset, payload)
        image_info = _render_comparison(
            dataset,
            "task5",
            payload,
            OUTPUT,
            int(dpi),
            float(point_density_multiplier),
        )
        record = _metadata_record(
            dataset, int(ordinal), int(seed), payload, image_info
        )
        records.append(record)
        print(
            f"task5/{dataset}: {record['pixel_size'][0]}x"
            f"{record['pixel_size'][1]}, Raw-PCA F1="
            f"{record['metrics']['Raw-PCA residual']['f1']:.3f}, FMT F1="
            f"{record['metrics']['Raw+FMT residual']['f1']:.3f}",
            flush=True,
        )

    metadata = {
        "name": "Task5_3D_horizontal_main_1.1",
        "method_version": "mainExp_Task5_3D_1.1",
        "layout": "one compact 1x3 comparison image per flow",
        "confirmation_ordinal": int(ordinal),
        "training_seed": int(seed),
        "dpi": int(dpi),
        "point_density_multiplier": 1.0,
        "shared_camera_within_each_flow": True,
        "panel_order": [
            "IVD-p95 ground truth",
            "variable-scale Raw-PCA residual",
            "variable-scale Raw+FMT residual",
        ],
        "selection_policy": (
            "fixed ordinal and seed for every flow; no slice selection by metric"
        ),
        "delta_wing_choice": "deltaWing_LBM (original, not resampled)",
        "device": str(device),
        "flows": records,
    }
    (OUTPUT / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(FLOW_SPECS),
    )
    parser.add_argument("--ordinal", type=int, default=DEFAULT_ORDINAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--point-density-multiplier",
        type=float,
        default=POINT_DENSITY_MULTIPLIER,
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    run(
        datasets=args.datasets,
        ordinal=args.ordinal,
        seed=args.seed,
        dpi=args.dpi,
        point_density_multiplier=args.point_density_multiplier,
        recompute=args.recompute,
        render_only=args.render_only,
    )
