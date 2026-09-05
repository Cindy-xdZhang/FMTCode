"""Visualize held-out Task3 labels and classifier errors in physical coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score

from FMT_Utils.DFT_FMT_3D import fmt_feature_indices_3d
from FMT_Utils.PathlineClassifier_3D import PathlineBinaryClassifier3D


def _load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = checkpoint["config"]
    model = PathlineBinaryClassifier3D(
        variant=checkpoint["variant"],
        fmt_dim=len(fmt_feature_indices_3d(spec["fmt_subset"])),
        temporal_width=spec["model"]["temporal_width"],
        embedding_dim=spec["model"]["embedding_dim"],
        auxiliary_dim=spec["model"]["auxiliary_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def _probabilities(model, raw, fmt, device, batch_size=2048):
    result = []
    for begin in range(0, len(raw), batch_size):
        raw_batch = torch.from_numpy(raw[begin:begin + batch_size]).to(device)
        fmt_batch = torch.from_numpy(fmt[begin:begin + batch_size]).to(device)
        logits = model(raw_batch, fmt_batch if model.variant == "raw_fmt" else None)
        result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)


def _predict(checkpoint_path, raw, fmt, device):
    model, checkpoint = _load_model(checkpoint_path, device)
    stats = checkpoint["normalization"]
    raw_normalized = ((raw - stats["raw_mean"]) / stats["raw_std"]).astype(np.float32)
    fmt_normalized = ((fmt - stats["fmt_mean"]) / stats["fmt_std"]).astype(np.float32)
    probabilities = _probabilities(model, raw_normalized, fmt_normalized, device)
    return probabilities, float(checkpoint["threshold"])


def _physical_axes(ax, seeds, title):
    mins = seeds.min(axis=0)
    maxs = seeds.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-8)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(extent)
    ax.view_init(elev=18, azim=-64)
    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("y", labelpad=2)
    ax.set_zlabel("z", labelpad=2)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7, pad=0)


def _plot_positive_score(ax, seeds, labels, score):
    positive = labels.astype(bool)
    scale = np.percentile(np.abs(score[positive]), 95) if positive.any() else 1.0
    color = np.clip(score[positive] / max(scale, 1e-8), 0.0, 1.0)
    ax.scatter(
        seeds[positive, 0], seeds[positive, 1], seeds[positive, 2],
        c=color, cmap="inferno", vmin=0.0, vmax=1.0, s=5, alpha=0.75,
        linewidths=0, rasterized=True,
    )


def _plot_confusion(ax, seeds, labels, predicted):
    labels = labels.astype(bool)
    predicted = predicted.astype(bool)
    categories = [
        (labels & predicted, "#2563eb", 5, 0.75),
        (~labels & predicted, "#dc2626", 8, 0.85),
        (labels & ~predicted, "#f59e0b", 8, 0.85),
    ]
    for mask, color, size, alpha in categories:
        ax.scatter(
            seeds[mask, 0], seeds[mask, 1], seeds[mask, 2],
            c=color, s=size, alpha=alpha, linewidths=0, rasterized=True,
        )


def visualize(config_path, ordinal=9):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmt_indices = fmt_feature_indices_3d(spec["fmt_subset"])
    for dataset in spec["datasets"]:
        source_paths = sorted((Path(spec["source_cache_root"]) / dataset).glob("slice_*.npz"))
        source_path = source_paths[int(ordinal)]
        label_path = Path(spec["label_cache_root"]) / dataset / source_path.name
        with np.load(source_path) as source, np.load(label_path) as label_data:
            raw = np.asarray(source["raw_features"], dtype=np.float32).reshape(-1, 7, spec["sampled_steps"], 3)
            fmt = np.asarray(source["fmt_features"], dtype=np.float32)[:, fmt_indices]
            seeds = np.asarray(source["seeds"], dtype=np.float32)
            labels = np.asarray(label_data["labels"], dtype=bool)
            ivd = np.asarray(label_data["ivd_at_seeds"], dtype=np.float32)
            threshold_field = np.asarray(label_data["threshold_at_seeds"], dtype=np.float32)
            metadata = json.loads(str(label_data["metadata_json"]))
        predictions = {}
        metrics = {}
        for variant in ("raw", "raw_fmt"):
            checkpoint = output_dir / "checkpoints" / f"{dataset}_{variant}_seed20.pt"
            probability, threshold = _predict(checkpoint, raw, fmt, device)
            predicted = probability >= threshold
            predictions[variant] = predicted
            metrics[variant] = {
                "f1": f1_score(labels, predicted, zero_division=0),
                "ap": average_precision_score(labels, probability),
            }

        figure = plt.figure(figsize=(16, 4.5), constrained_layout=True)
        titles = [
            "Reference: IVD − local threshold\npositive samples only",
            f"Binary reference\npositive={labels.mean():.1%}",
            f"Raw errors\nF1={metrics['raw']['f1']:.3f}, AP={metrics['raw']['ap']:.3f}",
            f"Raw + FMT errors\nF1={metrics['raw_fmt']['f1']:.3f}, AP={metrics['raw_fmt']['ap']:.3f}",
        ]
        axes = [figure.add_subplot(1, 4, index + 1, projection="3d") for index in range(4)]
        _plot_positive_score(axes[0], seeds, labels, ivd - threshold_field)
        axes[1].scatter(
            seeds[labels, 0], seeds[labels, 1], seeds[labels, 2],
            c="#7c3aed", s=5, alpha=0.72, linewidths=0, rasterized=True,
        )
        _plot_confusion(axes[2], seeds, labels, predictions["raw"])
        _plot_confusion(axes[3], seeds, labels, predictions["raw_fmt"])
        for axis, title in zip(axes, titles):
            _physical_axes(axis, seeds, title)
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563eb", label="true positive", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", label="false positive", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#f59e0b", label="false negative", markersize=7),
        ]
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        figure.suptitle(
            f"{dataset}, held-out slice {ordinal} (source index {metadata['source_start_index']}); "
            "physical axis proportions preserved",
            fontsize=13,
        )
        path = output_dir / f"{dataset}_slice{ordinal:02d}_spatial_classification.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ordinal", type=int, default=9)
    args = parser.parse_args()
    visualize(args.config, args.ordinal)
