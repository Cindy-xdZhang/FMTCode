"""Render fixed three-panel Task2 and Task3 spatial comparisons.

Every row contains the same held-out slice and orthographic camera:

1. whole-field IVD-p95 ground truth;
2. the strongest no-FMT comparison;
3. the corresponding FMT method.

Task2 compares Raw+VAE with FMT+the same VAE.  Task3 compares the
structure-matched Raw-PCA residual with the Raw+FMT residual.  The script can
generate prediction artifacts on Ibex without rendering and render those
artifacts locally, where the original flow files are available.  Titles,
legends, and colorbars are omitted for later paper composition; physical axes
and coordinate ticks are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
from sklearn.cluster import KMeans
import torch
import yaml

from DeepUtils.utils import EasyConfig
from Evaluate_Task3_FrozenConfirmation import (
    _evaluate_residual,
    _find_checkpoint,
    _load_residual,
)
from FMT_Utils.Task12Data_3D import (
    load_cache_records,
    stack_reference,
)
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
)
from Run_Task2_3D_Main import (
    _architecture,
    _cache_dir,
    _prepare_inputs,
)
from Verify_HighReVAE import _train
from Verify_Task3_FMTClassifier import _load_dataset, _stack_split

try:
    import Visualize_Task1_3D_PaperCandidates as paper
except ModuleNotFoundError:
    # The reduced Ibex experiment bundle intentionally omits rendering-only
    # modules.  Prediction-only mode remains fully functional there.
    paper = None


ROOT = Path(__file__).resolve().parent
TASK2_CONFIG = ROOT / "config/mainExp_Task2_3D_3.3.yaml"
TASK3_CONFIG = ROOT / "config/mainExp_Task3_3D_3.2_global_ivd_evaluate.yaml"
TASK2_OUTPUT = ROOT / "outputs/Task2_3D_horizontal_main_3.3"
TASK3_OUTPUT = ROOT / "outputs/Task3_3D_horizontal_main_3.2"

CONFIRMATION_ORDINAL = 4
TASK2_SEED = 9068
TASK3_SEED = 40
FIGURE_SIZE_INCHES = (21.0, 5.0)
DEFAULT_DPI = 360
OUTER_MARGIN = 0.001
PANEL_GAP = 0.0
VERTICAL_MARGIN = 0.01
PANEL_ZOOM = 1.12
VERTICAL_CROP_PAD_PIXELS = 24

FLOW_SPECS = {
    "cylinder3d": {
        "title": "Half-cylinder Re160",
        "task2_group": "halfcylinder",
        "view": (22, -62),
    },
    "halfcylinderRe640": {
        "title": "Half-cylinder Re640",
        "task2_group": "halfcylinder",
        "view": (22, -62),
    },
    "halfcylinderRe6400": {
        "title": "Half-cylinder Re6400",
        "task2_group": "halfcylinder",
        "view": (22, -62),
    },
    "tangaroa": {
        "title": "Tangaroa",
        "task2_group": "tangaroa",
        "view": (23, -62),
    },
    "deltaWing_resampled": {
        "title": "Delta-wing resampled",
        "task2_group": "deltaWing",
        "view": (22, -58),
    },
    "deltaWing_LBM": {
        "title": "Delta-wing original LBM",
        "task2_group": "deltaWing",
        "view": (22, -58),
    },
    "f22raptor": {
        "title": "F-22",
        "task2_group": "f22raptor",
        "view": (21, -58),
    },
    "channel": {
        "title": "Channel observer",
        "task2_group": "channel",
        "view": (22, -62),
    },
    "boeing747": {
        "title": "Boeing 747",
        "task2_group": "boeing747",
        "view": (21, -58),
    },
    "smokeBuoyancy": {
        "title": "Smoke buoyancy",
        "task2_group": "smokeBuoyancy",
        "view": (22, -58),
    },
}


def _new_comparison_figure():
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, facecolor="white")
    width = (1.0 - 2.0 * OUTER_MARGIN - 2.0 * PANEL_GAP) / 3.0
    height = 1.0 - 2.0 * VERTICAL_MARGIN
    axes = []
    rectangles = []
    for index in range(3):
        left = OUTER_MARGIN + index * (width + PANEL_GAP)
        rectangle = (left, VERTICAL_MARGIN, width, height)
        axes.append(fig.add_axes(rectangle, projection="3d"))
        rectangles.append(rectangle)
    return fig, axes, rectangles


def _physical_bounds(metadata, seeds):
    if paper is None:
        raise RuntimeError(
            "rendering requires Visualize_Task1_3D_PaperCandidates.py"
        )
    try:
        axes = paper._source_coordinate_axes(metadata["source_path"])
        return np.asarray(
            [
                [axes["x"][0], axes["y"][0], axes["z"][0]],
                [axes["x"][-1], axes["y"][-1], axes["z"][-1]],
            ],
            dtype=np.float64,
        )
    except (FileNotFoundError, OSError, KeyError):
        lower = np.asarray(seeds, dtype=np.float64).min(axis=0)
        upper = np.asarray(seeds, dtype=np.float64).max(axis=0)
        pad = np.maximum((upper - lower) * 0.04, 1e-6)
        return np.stack([lower - pad, upper + pad], axis=0)


def _prepare_axis(ax, bounds, view):
    if paper is None:
        raise RuntimeError(
            "rendering requires Visualize_Task1_3D_PaperCandidates.py"
        )
    paper._set_physical_axes(ax, bounds, view)
    span = np.maximum(np.asarray(bounds)[1] - np.asarray(bounds)[0], 1e-12)
    ax.set_box_aspect(span, zoom=PANEL_ZOOM)
    ax.tick_params(axis="both", which="major", labelsize=7, pad=-1)
    ax.set_xlabel("x", labelpad=-2)
    ax.set_ylabel("y", labelpad=-2)
    ax.set_zlabel("z", labelpad=-2)


def _draw_confusion(ax, seeds, reference, prediction):
    if paper is None:
        raise RuntimeError(
            "rendering requires Visualize_Task1_3D_PaperCandidates.py"
        )
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    categories = (
        (~reference & ~prediction, paper.COLORS["non_vortex"], "o", 2.0, 0.035),
        (reference & prediction, paper.COLORS["vortex"], "o", 13.0, 0.92),
        (~reference & prediction, paper.COLORS["false_positive"], "^", 15.0, 0.90),
        (reference & ~prediction, paper.COLORS["false_negative"], "x", 21.0, 0.95),
    )
    for mask, color, marker, size, alpha in categories:
        ax.scatter(
            seeds[mask, 0],
            seeds[mask, 1],
            seeds[mask, 2],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            depthshade=False,
            linewidths=1.0 if marker == "x" else 0,
            rasterized=True,
        )
    return {
        "true_negative": int((~reference & ~prediction).sum()),
        "true_positive": int((reference & prediction).sum()),
        "false_positive": int((~reference & prediction).sum()),
        "false_negative": int((reference & ~prediction).sum()),
    }


def _draw_reference(ax, seeds, reference):
    if paper is None:
        raise RuntimeError(
            "rendering requires Visualize_Task1_3D_PaperCandidates.py"
        )
    reference = np.asarray(reference, dtype=bool)
    for mask, color, size, alpha in (
        (~reference, paper.COLORS["non_vortex"], 2.0, 0.035),
        (reference, paper.COLORS["vortex"], 13.0, 0.92),
    ):
        ax.scatter(
            seeds[mask, 0], seeds[mask, 1], seeds[mask, 2],
            c=color, marker="o", s=size, alpha=alpha,
            depthshade=False, linewidths=0, rasterized=True,
        )
    return {
        "ground_truth_negative": int((~reference).sum()),
        "ground_truth_positive": int(reference.sum()),
    }


def _render_comparison(dataset, task, payload, output_dir, dpi):
    spec = FLOW_SPECS[dataset]
    fig, axes, rectangles = _new_comparison_figure()
    counts = [_draw_reference(axes[0], payload["seeds"], payload["reference"])]
    _prepare_axis(axes[0], payload["bounds"], spec["view"])
    for axis, prediction in zip(axes[1:], payload["predictions"]):
        counts.append(
            _draw_confusion(
                axis, payload["seeds"], payload["reference"], prediction
            )
        )
        _prepare_axis(axis, payload["bounds"], spec["view"])
    path = output_dir / f"{dataset}_{task}_horizontal_clean.png"
    fig.savefig(path, dpi=int(dpi), facecolor="white", edgecolor="none")
    plt.close(fig)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        difference = ImageChops.difference(
            rgb, Image.new("RGB", rgb.size, "white")
        )
        content_box = difference.getbbox()
        if content_box is not None:
            top = max(0, content_box[1] - VERTICAL_CROP_PAD_PIXELS)
            bottom = min(rgb.height, content_box[3] + VERTICAL_CROP_PAD_PIXELS)
            rgb = rgb.crop((0, top, rgb.width, bottom))
            rgb.save(path)
        pixel_size = list(rgb.size)
    return path, rectangles, pixel_size, counts


def _load_prediction_artifact(path):
    with np.load(path) as data:
        return {
            "seeds": np.asarray(data["seeds"], dtype=np.float32),
            "reference": np.asarray(data["reference"], dtype=bool),
            "predictions": [
                np.asarray(data["baseline_prediction"], dtype=bool),
                np.asarray(data["fmt_prediction"], dtype=bool),
            ],
            "metadata": json.loads(str(data["metadata_json"])),
            "metrics": json.loads(str(data["metrics_json"])),
        }


def _save_prediction_artifact(path, payload):
    np.savez_compressed(
        path,
        seeds=np.asarray(payload["seeds"], dtype=np.float32),
        reference=np.asarray(payload["reference"], dtype=bool),
        baseline_prediction=np.asarray(payload["predictions"][0], dtype=bool),
        fmt_prediction=np.asarray(payload["predictions"][1], dtype=bool),
        metadata_json=np.asarray(json.dumps(payload["metadata"], sort_keys=True)),
        metrics_json=np.asarray(json.dumps(payload["metrics"], sort_keys=True)),
    )


def _load_seeds(record):
    with np.load(record["path"]) as data:
        seeds = np.asarray(data["seeds"], dtype=np.float32)
    if len(seeds) != len(record["reference"]):
        raise RuntimeError(f"seed/reference mismatch in {record['path']}")
    return seeds


def _task2_predictions(dataset, output_dir, device, recompute=False):
    artifact_path = output_dir / f"{dataset}_task2_seed{TASK2_SEED}_predictions.npz"
    if artifact_path.exists() and not recompute:
        return _load_prediction_artifact(artifact_path)

    spec = yaml.safe_load(TASK2_CONFIG.read_text(encoding="utf-8"))
    group = FLOW_SPECS[dataset]["task2_group"]
    group_spec = spec["groups"][group]
    if dataset not in group_spec["datasets"]:
        raise ValueError(f"{dataset} is not in Task2 group {group}")
    development = load_cache_records(
        _cache_dir(spec, "development", dataset), 10
    )
    confirmation_count = int(spec["splits"]["confirmation_count"])
    confirmation = load_cache_records(
        _cache_dir(spec, "confirmation", dataset), confirmation_count
    )
    train = [development[index] for index in spec["splits"]["final_train"]]
    calibration = [
        development[index] for index in spec["splits"]["cluster_calibration"]
    ]
    evaluate = [*calibration, *confirmation]
    calibration_count = sum(len(record["reference"]) for record in calibration)
    architecture = _architecture(spec, group_spec["fixed_architecture"])
    source = EasyConfig(str(spec["source_config"]))
    calibration_reference = stack_reference(calibration)
    predictions = []
    metrics = {}
    for method in ("raw", "fmt"):
        train_x, evaluate_x = _prepare_inputs(
            train,
            evaluate,
            method,
            group_spec["fmt_feature"],
            device,
        )
        train_mu, evaluate_mu, losses = _train(
            train_x, evaluate_x, architecture, source, TASK2_SEED, device
        )
        kmeans = KMeans(
            n_clusters=2,
            random_state=int(spec["kmeans_seed"]),
            n_init=int(spec["kmeans_n_init"]),
        ).fit(train_mu)
        calibration_labels = kmeans.predict(evaluate_mu[:calibration_count])
        vortex_cluster = calibrate_vortex_cluster(
            calibration_reference, calibration_labels
        )
        confirmation_labels = kmeans.predict(evaluate_mu[calibration_count:])
        begin = sum(
            len(record["reference"])
            for record in confirmation[:CONFIRMATION_ORDINAL]
        )
        end = begin + len(confirmation[CONFIRMATION_ORDINAL]["reference"])
        selected_prediction = confirmation_labels[begin:end] == vortex_cluster
        selected_reference = confirmation[CONFIRMATION_ORDINAL]["reference"]
        predictions.append(selected_prediction)
        metrics["Raw+VAE" if method == "raw" else "FMT+VAE"] = {
            **binary_cluster_metrics(
                selected_reference, confirmation_labels[begin:end], vortex_cluster
            ),
            "vortex_cluster": int(vortex_cluster),
            "architecture": architecture["id"],
            "training_seed": TASK2_SEED,
            "train_seconds": float(losses["train_seconds"]),
        }
    record = confirmation[CONFIRMATION_ORDINAL]
    seeds = _load_seeds(record)
    metadata = dict(record["metadata"])
    payload = {
        "seeds": seeds,
        "reference": np.asarray(record["reference"], dtype=bool),
        "predictions": predictions,
        "metadata": metadata,
        "metrics": metrics,
    }
    _save_prediction_artifact(artifact_path, payload)
    return payload


def _task3_predictions(dataset, output_dir, device, recompute=False):
    artifact_path = output_dir / f"{dataset}_task3_seed{TASK3_SEED}_predictions.npz"
    if artifact_path.exists() and not recompute:
        return _load_prediction_artifact(artifact_path)

    spec = yaml.safe_load(TASK3_CONFIG.read_text(encoding="utf-8"))
    matching_groups = [
        group for group in spec["groups"] if dataset in group["datasets"]
    ]
    if len(matching_groups) != 1:
        raise RuntimeError(
            f"expected one Task3 group for {dataset}, found {len(matching_groups)}"
        )
    group = matching_groups[0]
    source_dir = ROOT / group["source_cache_root"] / dataset
    label_dir = ROOT / group["label_cache_root"] / dataset
    records = _load_dataset(
        source_dir,
        label_dir,
        sampled_steps=int(spec["sampled_steps"]),
        fmt_subset=spec["fmt_subset"],
        required_ordinals={CONFIRMATION_ORDINAL},
        gram_num_freq=int(spec["fmt_gram_num_freq"]),
        expected_slices=int(group["expected_slices"]),
    )
    split = _stack_split(records, [CONFIRMATION_ORDINAL])
    checkpoint_paths = [
        _find_checkpoint(
            [ROOT / value for value in group["raw_pca_checkpoint_roots"]],
            f"{dataset}_raw_pca_residual_seed{TASK3_SEED}.pt",
        ),
        _find_checkpoint(
            [ROOT / value for value in group["fmt_checkpoint_roots"]],
            f"{dataset}_raw_fmt_residual_seed{TASK3_SEED}.pt",
        ),
    ]
    predictions = []
    metrics = {}
    method_names = ("Raw-PCA residual", "Raw+FMT residual")
    for method, checkpoint_path in zip(method_names, checkpoint_paths):
        model, checkpoint = _load_residual(
            checkpoint_path, split[1].shape[1], device
        )
        targets, probabilities, score = _evaluate_residual(
            model, checkpoint, split, int(spec["batch_size"]), TASK3_SEED, device
        )
        predictions.append(probabilities >= float(checkpoint["threshold"]))
        metrics[method] = {
            **{key: float(value) for key, value in score.items()},
            "threshold": float(checkpoint["threshold"]),
            "alpha": float(checkpoint["alpha"]),
            "training_seed": TASK3_SEED,
            "checkpoint": str(checkpoint_path),
        }

    source_path = sorted(source_dir.glob("slice_*.npz"))[CONFIRMATION_ORDINAL]
    with np.load(source_path) as data:
        seeds = np.asarray(data["seeds"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"]))
    if len(seeds) != len(targets):
        raise RuntimeError(f"Task3 seed/target mismatch for {dataset}")
    payload = {
        "seeds": seeds,
        "reference": np.asarray(targets, dtype=bool),
        "predictions": predictions,
        "metadata": metadata,
        "metrics": metrics,
    }
    _save_prediction_artifact(artifact_path, payload)
    return payload


def _record(dataset, task, payload, image_info):
    path, rectangles, pixel_size, counts = image_info
    return {
        "dataset": dataset,
        "title": FLOW_SPECS[dataset]["title"],
        "task": task,
        "source_index": int(payload["metadata"]["source_start_index"]),
        "source_time": float(payload["metadata"]["source_time"]),
        "image": str(path),
        "pixel_size": pixel_size,
        "panel_rectangles": [list(value) for value in rectangles],
        "panel_order": (
            ["IVD-p95 ground truth", "Raw+VAE", "FMT+VAE"]
            if task == "task2"
            else [
                "IVD-p95 ground truth",
                "Raw-PCA residual",
                "Raw+FMT residual",
            ]
        ),
        "camera": {
            "projection": "orthographic",
            "elevation_degrees": FLOW_SPECS[dataset]["view"][0],
            "azimuth_degrees": FLOW_SPECS[dataset]["view"][1],
            "physical_bounds": payload["bounds"].tolist(),
            "panel_zoom": PANEL_ZOOM,
        },
        "reference_positive_fraction": float(payload["reference"].mean()),
        "metrics": payload["metrics"],
        "confusion_counts": counts,
    }


def run(
    tasks,
    datasets,
    dpi=DEFAULT_DPI,
    recompute=False,
    predictions_only=False,
    render_only=False,
):
    if predictions_only and render_only:
        raise ValueError("predictions_only and render_only are mutually exclusive")
    requested_tasks = list(dict.fromkeys(tasks))
    unknown_tasks = sorted(set(requested_tasks) - {"task2", "task3"})
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {unknown_tasks}")
    unknown_datasets = sorted(set(datasets) - set(FLOW_SPECS))
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = {"task2": TASK2_OUTPUT, "task3": TASK3_OUTPUT}
    records = {task: [] for task in requested_tasks}
    for task in requested_tasks:
        output_dir = outputs[task]
        output_dir.mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            artifact_path = output_dir / (
                f"{dataset}_{task}_seed"
                f"{TASK2_SEED if task == 'task2' else TASK3_SEED}_predictions.npz"
            )
            if render_only:
                if not artifact_path.exists():
                    raise FileNotFoundError(artifact_path)
                payload = _load_prediction_artifact(artifact_path)
            elif task == "task2":
                payload = _task2_predictions(dataset, output_dir, device, recompute)
            else:
                payload = _task3_predictions(
                    dataset, output_dir, device, recompute
                )
            if predictions_only:
                print(f"wrote {artifact_path}", flush=True)
                continue
            payload["bounds"] = _physical_bounds(
                payload["metadata"], payload["seeds"]
            )
            image_info = _render_comparison(
                dataset, task, payload, output_dir, int(dpi)
            )
            record = _record(dataset, task, payload, image_info)
            records[task].append(record)
            _, baseline, fmt = record["panel_order"]
            print(
                f"{task}/{dataset}: {record['pixel_size'][0]}x"
                f"{record['pixel_size'][1]}, {baseline} F1="
                f"{record['metrics'][baseline]['f1']:.3f}, {fmt} F1="
                f"{record['metrics'][fmt]['f1']:.3f}",
                flush=True,
            )
        if predictions_only:
            continue
        metadata = {
            "name": (
                "Task2_3D_horizontal_main_3.3"
                if task == "task2"
                else "Task3_3D_horizontal_main_3.2"
            ),
            "layout": "one compact 1x3 comparison image per flow",
            "figure_size_inches": list(FIGURE_SIZE_INCHES),
            "dpi": int(dpi),
            "panel_gap": PANEL_GAP,
            "annotation_policy": {
                "retained": [
                    "3D bounding box",
                    "x/y/z axis labels",
                    "coordinate ticks",
                    "IVD-p95 ground truth",
                    "TP/FP/FN/TN spatial marks",
                ],
                "removed": [
                    "pathline/IVD scene overview",
                    "figure titles",
                    "panel titles",
                    "legends",
                    "colorbars",
                ],
            },
            "shared_camera_within_each_flow": True,
            "device": str(device),
            "flows": records[task],
        }
        (output_dir / "figure_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", default=["task2", "task3"],
        choices=["task2", "task3"],
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(FLOW_SPECS),
        choices=list(FLOW_SPECS),
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--predictions-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    run(
        args.tasks,
        args.datasets,
        args.dpi,
        args.recompute,
        args.predictions_only,
        args.render_only,
    )
