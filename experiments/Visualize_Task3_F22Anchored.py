"""Render the frozen F22 anchored-FMT confirmation as a diagnostic triptych.

The three panels use the same evaluated seeds and orthographic camera:

1. whole-field IVD-p95 reference;
2. same-width Raw-PCA residual prediction;
3. anchored FMT residual prediction.

No model, threshold, feature, or time slice is selected in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
import yaml

from Evaluate_Task3_F22_AnchoredConfirmation import (
    _checkpoint,
    _confirmation_records,
)
from Evaluate_Task3_FrozenConfirmation import (
    _evaluate_residual,
    _load_residual,
)
from Search_Task3_FMTResidual_3D import _load_spec
from Verify_Task3_FMTClassifier import _stack_split
from VisualizationIVDReference_3D import reconstruct_ivd_reference_surface
import Visualize_Task1_3D_PaperCandidates as paper


ROOT = Path(__file__).resolve().parents[1]
SEARCH_CONFIG = ROOT / "config/Verify_Task3_F22AnchoredFeatures_1.2_search.yaml"
CONFIRM_CONFIG = ROOT / "config/Confirm_Task3_F22AnchoredFeatures_1.2_evaluate.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/Task3_F22Anchored_visualization_1.2"
DEFAULT_ORDINAL = 4
DEFAULT_SEED = 50
VIEW = (21, -58)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confusion_masks(reference, prediction):
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    masks = {
        "true_negative": ~reference & ~prediction,
        "true_positive": reference & prediction,
        "false_positive": ~reference & prediction,
        "false_negative": reference & ~prediction,
    }
    if not np.all(np.stack(list(masks.values())).sum(axis=0) == 1):
        raise RuntimeError("confusion masks do not partition the seed set")
    return masks


def _visible_seed_mask(seeds, bounds=None):
    seeds = np.asarray(seeds, dtype=np.float64)
    if bounds is None:
        return np.ones(len(seeds), dtype=bool)
    bounds = np.asarray(bounds, dtype=np.float64)
    tolerance = np.maximum(bounds[1] - bounds[0], 1.0) * 1e-10
    return np.all(
        (seeds >= bounds[0] - tolerance)
        & (seeds <= bounds[1] + tolerance),
        axis=1,
    )


def _clipped_mesh_triangles(mesh, bounds=None):
    """Return triangles clipped to a displayed axis-aligned close-up box.

    Matplotlib's 3-D collections are not reliably clipped by ``set_xlim`` /
    ``set_ylim`` / ``set_zlim``.  Without an explicit geometric clip, the
    whole-field IVD surface can leak into a close-up panel and make the camera
    appear much farther away than its numeric bounds.
    """
    if mesh is None:
        return np.empty((0, 3, 3), dtype=np.float64)
    vertices, _, faces = mesh
    triangles = np.asarray(vertices, dtype=np.float64)[
        np.asarray(faces, dtype=np.int64)
    ]
    if bounds is None or not len(triangles):
        return triangles

    bounds = np.asarray(bounds, dtype=np.float64)
    lower, upper = bounds
    overlaps = np.all(
        (triangles.max(axis=1) >= lower)
        & (triangles.min(axis=1) <= upper),
        axis=1,
    )

    def clip_plane(polygon, axis, limit, keep_above):
        """Sutherland-Hodgman clip against one axis-aligned plane."""
        if not len(polygon):
            return polygon
        output = []
        previous = polygon[-1]
        previous_inside = (
            previous[axis] >= limit if keep_above
            else previous[axis] <= limit
        )
        for current in polygon:
            current_inside = (
                current[axis] >= limit if keep_above
                else current[axis] <= limit
            )
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                if abs(float(denominator)) > np.finfo(np.float64).eps:
                    weight = (limit - previous[axis]) / denominator
                    output.append(previous + weight * (current - previous))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
        return np.asarray(output, dtype=np.float64)

    clipped = []
    for triangle in triangles[overlaps]:
        polygon = np.asarray(triangle, dtype=np.float64)
        for axis in range(3):
            polygon = clip_plane(polygon, axis, lower[axis], True)
            polygon = clip_plane(polygon, axis, upper[axis], False)
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            for index in range(1, len(polygon) - 1):
                clipped.append(
                    np.stack([polygon[0], polygon[index], polygon[index + 1]])
                )
    if not clipped:
        return np.empty((0, 3, 3), dtype=np.float64)
    clipped = np.asarray(clipped, dtype=np.float64)
    area2 = np.linalg.norm(
        np.cross(
            clipped[:, 1] - clipped[:, 0],
            clipped[:, 2] - clipped[:, 0],
        ),
        axis=1,
    )
    scale = max(float(np.linalg.norm(upper - lower)), 1.0)
    return clipped[area2 > np.finfo(np.float64).eps * scale * scale]


def _draw_reference(ax, seeds, reference, mesh, bounds=None,
                    show_background=True, show_positive_points=True,
                    marker_scale=1.0):
    marker_scale = float(marker_scale)
    if marker_scale <= 0.0:
        raise ValueError("marker_scale must be positive")
    collection_count = len(ax.collections)
    triangles = _clipped_mesh_triangles(mesh, bounds)
    if len(triangles):
        ax.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolor=paper.COLORS["ivd"],
                edgecolor="none",
                alpha=0.25,
                zorder=1,
            )
        )
    for collection in ax.collections[collection_count:]:
        collection.set_rasterized(True)
    visible = _visible_seed_mask(seeds, bounds)
    styles = []
    if show_background:
        styles.append(
            (~reference, paper.COLORS["non_vortex"], 3.0 * marker_scale, 0.13)
        )
    if show_positive_points:
        styles.append((
            reference, paper.COLORS["vortex"], 15.0 * marker_scale, 0.94
        ))
    for mask, color, size, alpha in styles:
        mask = np.asarray(mask, dtype=bool) & visible
        ax.scatter(
            seeds[mask, 0], seeds[mask, 1], seeds[mask, 2],
            c=color, s=size, alpha=alpha, linewidths=0,
            depthshade=False, rasterized=True,
        )


def _draw_prediction(ax, seeds, reference, prediction, error_focus=False,
                     bounds=None, show_true_negatives=True,
                     show_true_positives=True, context_mask=None,
                     false_positive_color=None, marker_scale=1.0):
    masks = _confusion_masks(reference, prediction)
    visible = _visible_seed_mask(seeds, bounds)
    false_positive_color = (
        paper.COLORS["false_positive"]
        if false_positive_color is None else false_positive_color
    )
    marker_scale = float(marker_scale)
    if marker_scale <= 0.0:
        raise ValueError("marker_scale must be positive")
    if error_focus:
        # A close-up is an error-localization panel.  Correct points form a
        # regular sampling lattice at high density, so they can be omitted
        # without changing any metric; the reference panel still shows the
        # local vortex geometry and all classification errors remain visible.
        styles = []
        if show_true_positives:
            styles.append(
                ("true_positive", paper.COLORS["vortex"], "o", 12.0, 0.82, 0.0)
            )
        if context_mask is not None:
            styles.extend((
                ("true_negative", "#6b7280", "o", 13.0, 0.72, 0.0),
                ("true_positive", paper.COLORS["vortex"], "o", 17.0, 0.88, 0.0),
            ))
        styles.extend((
            ("false_positive", false_positive_color, "^", 48.0, 1.0, 0.0),
            ("false_negative", paper.COLORS["false_negative"], "x", 58.0, 1.0, 1.8),
        ))
    else:
        styles = [
            ("true_positive", paper.COLORS["vortex"], "o", 15.0, 0.94, 0.0),
            ("false_positive", false_positive_color, "^", 20.0, 0.96, 0.0),
            ("false_negative", paper.COLORS["false_negative"], "x", 27.0, 0.98, 1.1),
        ]
        if show_true_negatives:
            styles.insert(
                0,
                ("true_negative", paper.COLORS["non_vortex"], "o", 3.0, 0.13, 0.0),
            )
    for name, color, marker, size, alpha, linewidth in styles:
        mask = masks[name] & visible
        if (
            error_focus and context_mask is not None
            and name in {"true_negative", "true_positive"}
        ):
            mask &= np.asarray(context_mask, dtype=bool)
        ax.scatter(
            seeds[mask, 0], seeds[mask, 1], seeds[mask, 2],
            c=color, marker=marker, s=size * marker_scale, alpha=alpha,
            linewidths=(
                linewidth * min(1.0, float(np.sqrt(marker_scale)))
            ),
            depthshade=False, rasterized=True,
        )
    return {name: int((mask & visible).sum()) for name, mask in masks.items()}


def _prepare_axis(
    ax, bounds, view=VIEW, close_up=False, close_up_zoom=1.30,
    show_tick_labels=True,
):
    paper._set_physical_axes(ax, bounds, view)
    span = np.maximum(np.asarray(bounds)[1] - np.asarray(bounds)[0], 1e-12)
    ax.set_box_aspect(span, zoom=float(close_up_zoom) if close_up else 1.08)
    if close_up:
        bounds_array = np.asarray(bounds, dtype=np.float64)
        ticks = (
            np.linspace(bounds_array[0, 0], bounds_array[1, 0], 2),
            np.linspace(bounds_array[0, 1], bounds_array[1, 1], 2),
            np.linspace(bounds_array[0, 2], bounds_array[1, 2], 2),
        )

        def compact_ticks(values):
            values = np.asarray(values, dtype=np.float64)
            tick_span = max(float(np.ptp(values)), 1e-12)
            precision = int(np.clip(
                np.ceil(-np.log10(tick_span)) + 1,
                1,
                4,
            ))
            labels = []
            for value in values:
                if abs(float(value)) < 0.5 * 10.0 ** (-precision):
                    labels.append("0")
                else:
                    labels.append(
                        f"{float(value):.{precision}f}"
                        .rstrip("0").rstrip(".").replace("-", "−")
                    )
            return labels

        ax.set_xticks(ticks[0], compact_ticks(ticks[0]))
        ax.set_yticks(ticks[1], compact_ticks(ticks[1]))
        ax.set_zticks(ticks[2], compact_ticks(ticks[2]))
    if close_up:
        ax.tick_params(labelsize=6.5, pad=1, length=0)
        # Matplotlib projects 3-D endpoint labels outside the axes rectangle at
        # strong camera zoom.  Report all three endpoint ranges compactly in
        # the reference-panel subtitle and suppress projected tick stubs,
        # which otherwise escape the panel and collide with the legend.
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        for axis_3d in (ax.xaxis, ax.yaxis, ax.zaxis):
            for tick in axis_3d.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
    else:
        ax.tick_params(labelsize=8, pad=4)
    if not show_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    # Matplotlib's automatic 3-D label projection can put an axis name directly
    # on a numeric tick for very different domain aspect ratios.  Keep the full
    # bounding box, tick marks, and numeric coordinate scale, but omit repeated
    # x/y/z glyphs so no dataset-specific label placement is required.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def _vortex_roi(
    bounds,
    seeds,
    reference,
    *,
    positive_fraction=0.25,
    minimum_points=16,
    padding_factor=1.15,
    minimum_span_fraction=0.04,
    target_span_fraction=None,
    exact_target_span=False,
    center_on_selected_neighbourhood=False,
):
    """Find a compact, prediction-independent IVD-positive close-up.

    The old all-positive bounding box was often almost the full domain for
    multi-vortex or elongated flows.  Here the center is the smallest
    normalized-coordinate ball containing 25% of reference-positive seeds.
    Its selected points define the displayed bounding box.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    points = np.asarray(seeds, dtype=np.float64)[np.asarray(reference, dtype=bool)]
    if not len(points):
        raise RuntimeError("cannot define a vortex ROI without positive seeds")
    full_span = np.maximum(bounds[1] - bounds[0], 1e-12)
    normalized = (points - bounds[0]) / full_span
    if not 0.0 < float(positive_fraction) <= 1.0:
        raise ValueError("positive_fraction must be in (0, 1]")
    if int(minimum_points) < 1:
        raise ValueError("minimum_points must be positive")
    if float(padding_factor) < 1.0:
        raise ValueError("padding_factor must be at least 1")
    if not 0.0 < float(minimum_span_fraction) <= 1.0:
        raise ValueError("minimum_span_fraction must be in (0, 1]")
    if target_span_fraction is not None:
        target_span_fraction = np.asarray(
            target_span_fraction, dtype=np.float64
        )
        if target_span_fraction.ndim == 0:
            target_span_fraction = np.repeat(target_span_fraction, 3)
        if target_span_fraction.shape != (3,) or not np.all(
            (target_span_fraction > 0.0) & (target_span_fraction <= 1.0)
        ):
            raise ValueError(
                "target_span_fraction must be a scalar or three values in (0, 1]"
            )
    if not isinstance(exact_target_span, (bool, np.bool_)):
        raise TypeError("exact_target_span must be a boolean")
    if not isinstance(center_on_selected_neighbourhood, (bool, np.bool_)):
        raise TypeError("center_on_selected_neighbourhood must be a boolean")
    target_count = min(
        len(points),
        max(int(minimum_points), int(np.ceil(positive_fraction * len(points)))),
    )

    if target_count == len(points):
        selected = points
        center_normalized = normalized.mean(axis=0)
        best_index = int(np.argmin(np.sum(
            (normalized - center_normalized) ** 2, axis=1
        )))
    else:
        candidate_count = min(len(points), 2048)
        candidate_indices = np.unique(
            np.linspace(0, len(points) - 1, candidate_count, dtype=np.int64)
        )
        kth = target_count - 1
        best_radius = np.inf
        best_index = int(candidate_indices[0])
        for start in range(0, len(candidate_indices), 128):
            block_indices = candidate_indices[start:start + 128]
            delta = (
                normalized[block_indices, None, :] - normalized[None, :, :]
            )
            distance2 = np.einsum("ijk,ijk->ij", delta, delta)
            radii = np.partition(distance2, kth, axis=1)[:, kth]
            local = int(np.argmin(radii))
            if float(radii[local]) < best_radius:
                best_radius = float(radii[local])
                best_index = int(block_indices[local])
        distance2 = np.sum(
            (normalized - normalized[best_index]) ** 2, axis=1
        )
        selected_indices = np.argpartition(distance2, kth)[:target_count]
        selected = points[selected_indices]

    if target_span_fraction is None:
        lower = selected.min(axis=0)
        upper = selected.max(axis=0)
        center = 0.5 * (lower + upper)
        span = np.maximum(
            (upper - lower) * float(padding_factor),
            float(minimum_span_fraction) * full_span,
        )
        lower = np.maximum(bounds[0], center - 0.5 * span)
        upper = np.minimum(bounds[1], center + 0.5 * span)
    else:
        if exact_target_span:
            # The requested scale is authoritative.  Centering on the
            # reference-positive medoid guarantees that the exact crop still
            # contains a physically relevant point, without allowing the
            # display lattice to expand the viewport again.
            focus = (
                0.5 * (selected.min(axis=0) + selected.max(axis=0))
                if center_on_selected_neighbourhood else points[best_index]
            )
            span = np.maximum(
                target_span_fraction * full_span,
                float(minimum_span_fraction) * full_span,
            )
        else:
            # Legacy behaviour: include the complete selected neighbourhood,
            # even when that makes the viewport larger than the target span.
            focus = 0.5 * (selected.min(axis=0) + selected.max(axis=0))
            selected_span = selected.max(axis=0) - selected.min(axis=0)
            span = np.maximum(
                np.maximum(
                    target_span_fraction * full_span,
                    selected_span * float(padding_factor),
                ),
                float(minimum_span_fraction) * full_span,
            )
        lower = np.clip(
            focus - 0.5 * span,
            bounds[0],
            bounds[1] - span,
        )
        upper = lower + span
    return np.stack([lower, upper])


def _load_frozen_inputs(ordinal: int, seed: int, device):
    confirm = yaml.safe_load(CONFIRM_CONFIG.read_text(encoding="utf-8"))
    search_path = ROOT / confirm["search_config"]
    selection_path = ROOT / confirm["selection"]
    schedule_path = ROOT / confirm["confirmation_schedule"]
    search = _load_spec(search_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["search_config_sha256"] != _sha256(search_path):
        raise RuntimeError("search config changed after selection")
    if selection["confirmation_schedule_sha256"] != _sha256(schedule_path):
        raise RuntimeError("confirmation schedule changed after selection")
    if int(seed) not in {int(value) for value in search["screen_seeds"]}:
        raise ValueError(f"seed {seed} is not one of the frozen seeds")

    candidate = dict(selection["selected_candidate"])
    runtime = {
        "source_cache_root": str(ROOT / confirm["source_cache_root"]),
        "label_cache_root": str(ROOT / confirm["label_cache_root"]),
        "expected_slices": int(confirm["expected_slices"]),
    }
    records, source_indices = _confirmation_records(runtime, candidate, device)
    if not 0 <= int(ordinal) < len(records):
        raise IndexError(f"ordinal {ordinal} outside [0,{len(records)})")
    record = records[int(ordinal)]
    if int(record[3]) != int(ordinal):
        raise RuntimeError("confirmation records are not in ordinal order")
    split = _stack_split(records, [int(ordinal)])

    source_paths = sorted(
        (ROOT / confirm["source_cache_root"] / "f22raptor").glob("slice_*.npz")
    )
    source_path = source_paths[int(ordinal)]
    with np.load(source_path) as source:
        seeds = np.asarray(source["seeds"], dtype=np.float32)
        source_metadata = json.loads(str(source["metadata_json"]))
    metadata = {**source_metadata, **dict(record[4])}
    if int(metadata["source_start_index"]) != int(source_indices[int(ordinal)]):
        raise RuntimeError("source index mismatch")
    if len(seeds) != len(split[2]):
        raise RuntimeError("seed and target counts differ")

    predictions = {}
    metrics = {}
    checkpoints = {}
    for source, method in (
        ("raw_pca", "Raw-PCA residual"),
        ("fmt", "Anchored FMT residual"),
    ):
        checkpoint_path = _checkpoint(
            search, candidate, int(seed), source
        )
        model, checkpoint = _load_residual(
            checkpoint_path, split[1].shape[1], device
        )
        targets, probabilities, score = _evaluate_residual(
            model, checkpoint, split,
            int(search["training"]["batch_size"]), int(seed), device,
        )
        if not np.array_equal(targets, split[2].astype(bool)):
            raise RuntimeError(f"{method}: target order changed")
        threshold = float(checkpoint["threshold"])
        predictions[method] = probabilities >= threshold
        metrics[method] = {
            "f1": float(score["f1"]),
            "average_precision": float(score["average_precision"]),
            "threshold": threshold,
        }
        checkpoints[method] = str(checkpoint_path)
    return {
        "candidate": candidate,
        "selection_sha256": _sha256(selection_path),
        "source_path": source_path,
        "source_indices": source_indices,
        "metadata": metadata,
        "seeds": seeds,
        "reference": split[2].astype(bool),
        "predictions": predictions,
        "metrics": metrics,
        "checkpoints": checkpoints,
    }


def render(ordinal=DEFAULT_ORDINAL, seed=DEFAULT_SEED, output_dir=DEFAULT_OUTPUT,
           dpi=300):
    ordinal = int(ordinal)
    seed = int(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = _load_frozen_inputs(ordinal, seed, device)
    reference_surface = reconstruct_ivd_reference_surface(
        ROOT, "f22raptor", payload["metadata"], payload["seeds"],
        payload["reference"], output_dir / "ivd_surface_cache",
    )
    bounds = reference_surface["bounds"]
    mesh = reference_surface["ivd_mesh"]

    figure = plt.figure(figsize=(20.0, 10.0), facecolor="white")
    axes = []
    for row_y in (0.545, 0.095):
        axes.append([
            figure.add_axes(
                (0.005 + index * 0.33, row_y, 0.33, 0.36), projection="3d"
            )
            for index in range(3)
        ])
    roi_bounds = _vortex_roi(bounds, payload["seeds"], payload["reference"])
    counts = {}
    for row_index, (row_axes, row_bounds) in enumerate(
        zip(axes, (bounds, roi_bounds))
    ):
        _draw_reference(
            row_axes[0], payload["seeds"], payload["reference"], mesh,
            bounds=row_bounds if row_index == 1 else None,
            show_background=row_index == 0,
        )
        row_axes[0].set_title(
            (
                "Full domain — IVD-p95 ground truth\n"
                f"positive={payload['reference'].mean():.1%}"
                if row_index == 0 else
                "Densest IVD-positive region close-up"
            ),
            fontsize=12,
        )
        for axis, method in zip(row_axes[1:], payload["predictions"]):
            rendered_counts = _draw_prediction(
                axis, payload["seeds"], payload["reference"],
                payload["predictions"][method], error_focus=row_index == 1,
                bounds=row_bounds if row_index == 1 else None,
            )
            if row_index == 0:
                counts[method] = rendered_counts
            score = payload["metrics"][method]
            axis.set_title(
                (
                    f"{method}\nF1={score['f1']:.3f}, "
                    f"AP={score['average_precision']:.3f}"
                    if row_index == 0 else
                    f"{method} errors — FP={rendered_counts['false_positive']}, "
                    f"FN={rendered_counts['false_negative']}"
                ),
                fontsize=12,
            )
        for axis in row_axes:
            _prepare_axis(axis, row_bounds, close_up=row_index == 1)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="none",
               markerfacecolor=paper.COLORS["non_vortex"], label="true negative",
               markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color="none",
               markerfacecolor=paper.COLORS["vortex"], label="true positive",
               markersize=7),
        Line2D([0], [0], marker="^", linestyle="none", color="none",
               markerfacecolor=paper.COLORS["false_positive"], label="false positive",
               markersize=7),
        Line2D([0], [0], marker="x", linestyle="none",
               color=paper.COLORS["false_negative"], label="false negative",
               markersize=7),
        Line2D([0], [0], color=paper.COLORS["ivd"], linewidth=7,
               alpha=0.45, label="IVD-p95 isosurface"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                  fontsize=9, bbox_to_anchor=(0.5, 0.005))
    figure.suptitle(
        "F-22 Task3 — frozen fresh confirmation "
        f"(source index {payload['metadata']['source_start_index']}, seed {seed})",
        fontsize=14, y=0.995,
    )
    image_path = output_dir / (
        f"f22_task3_anchored_ordinal{ordinal:02d}_"
        f"index{int(payload['metadata']['source_start_index']):04d}_seed{seed}.png"
    )
    figure.savefig(image_path, dpi=int(dpi), facecolor="white", edgecolor="none")
    plt.close(figure)

    audit = {
        "experiment": "Visualize_Task3_F22Anchored_1.2",
        "selection_sha256": payload["selection_sha256"],
        "candidate": payload["candidate"],
        "confirmation_ordinal": ordinal,
        "source_start_index": int(payload["metadata"]["source_start_index"]),
        "training_seed": seed,
        "device": str(device),
        "sample_count": int(len(payload["seeds"])),
        "positive_count": int(payload["reference"].sum()),
        "metrics": payload["metrics"],
        "confusion_counts": counts,
        "checkpoints": payload["checkpoints"],
        "ivd_surface_audit": reference_surface["ivd_surface_audit"],
        "panel_contract": (
            "within each row: same exact seeds, reference, bounds, and camera; "
            "close-up bounds depend only on reference-positive seeds; true "
            "negatives are omitted from close-up panels"
        ),
        "close_up_bounds": roi_bounds.tolist(),
        "close_up_span_fraction": (
            (roi_bounds[1] - roi_bounds[0])
            / np.maximum(np.asarray(bounds)[1] - np.asarray(bounds)[0], 1e-12)
        ).tolist(),
        "close_up_rule": (
            "densest normalized-coordinate region containing 25% of IVD-"
            "positive seeds, expanded by 15% with minimum 4% domain span"
        ),
        "image": str(image_path),
    }
    audit_path = image_path.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(image_path), "audit": str(audit_path),
                      "metrics": payload["metrics"]}, indent=2))
    return image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinal", type=int, default=DEFAULT_ORDINAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args()
    render(arguments.ordinal, arguments.seed, arguments.output_dir, arguments.dpi)
