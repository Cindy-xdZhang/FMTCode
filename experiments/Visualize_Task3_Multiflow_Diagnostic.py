"""Render audited Task3 comparison figures for multiple frozen 3D flows.

Each output is a 2x3 diagnostic plate.  The first row shows the full physical
domain; the second row uses a close-up determined only by IVD-positive seeds
and emphasizes false positives and false negatives.  Every row uses identical
seeds, bounds, and camera across reference, Raw-PCA, and FMT panels.

F-22 uses the frozen family-specific anchored FMT confirmation.  Other flows
use the frozen global-IVD Task3 prediction artifacts.  The distinction is
recorded in every audit JSON and is never hidden in the figure metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
    Path3DCollection,
    Poly3DCollection,
)
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import torch

from FLowUtils.ScalarField3d import marching_cubes_world
from VisualizationIVDReference_3D import reconstruct_ivd_reference_surface
from Visualize_Task3_F22Anchored import (
    _clipped_mesh_triangles,
    _confusion_masks,
    _draw_prediction,
    _draw_reference,
    _load_frozen_inputs,
    _prepare_axis,
    _visible_seed_mask,
    _vortex_roi,
)
import Visualize_Task1_3D_PaperCandidates as paper


NATURE_FIGURE_SKILL_ROOT = Path(os.environ.get(
    "NATURE_FIGURE_SKILL_ROOT",
    Path.home() / ".codex/skills/nature-figure",
))
sys.path.insert(0, str(NATURE_FIGURE_SKILL_ROOT / "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment


matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


ROOT = Path(__file__).resolve().parents[1]
STANDARD_ARTIFACT_ROOT = ROOT / "outputs/Task3_3D_horizontal_main_3.2"
DEFAULT_OUTPUT = ROOT / "outputs/Task3_multiflow_diagnostic_1.2"
STANDARD_ORDINAL = 4
STANDARD_SEED = 40
ANCHORED_F22_SEED = 50
DEFAULT_CLOSE_UP = {
    "focus_basis": "paired_disagreement",
    "positive_fraction": 0.02,
    "minimum_points": 12,
    "padding_factor": 1.03,
    "minimum_span_fraction": 0.015,
    "target_span_fraction": None,
    "physical_cube_longest_axis_fraction": None,
    "exact_target_span": False,
    "center_on_selected_neighbourhood": False,
    "camera_zoom": 1.30,
    "marker_scale": 1.0,
    "false_positive_color": None,
    "show_reference_points": False,
    "show_true_negatives": False,
    "show_true_positives": True,
    "show_paired_error_context": False,
    "show_reference_surface_in_predictions": False,
    "reference_context_color": "#fff4eb",
    "reference_surface_alpha": 0.72,
    "all_positive_surface_alpha": 0.18,
    "shade_surfaces": False,
    "display_crop_scale": 1.0,
    "render_prediction_surface": False,
    "render_reference_surface_from_labels": False,
    "prediction_surface_grid_shape": [48, 48, 48],
    "prediction_surface_neighbors": 8,
    "prediction_surface_power": 2.0,
    "prediction_surface_smoothing_sigma": 0.0,
    "render_scalar_slices": False,
    "require_all_prediction_surfaces_in_crop": True,
    "show_error_markers": True,
    "prediction_point_mode": "confusion",
    "surface_only": False,
    "show_close_up_axes": False,
    "fixed_canvas_export": False,
}

FLOW_SPECS = {
    "cylinder3d": {"title": "Half-cylinder Re160", "view": (22, -62)},
    "halfcylinderRe640": {"title": "Half-cylinder Re640", "view": (22, -62)},
    "halfcylinderRe6400": {"title": "Half-cylinder Re6400", "view": (22, -62)},
    "tangaroa": {"title": "Tangaroa", "view": (23, -62)},
    "deltaWing_resampled": {"title": "Delta-wing resampled", "view": (22, -58)},
    "deltaWing_LBM": {"title": "Delta-wing original LBM", "view": (22, -58)},
    "f22raptor": {"title": "F-22", "view": (21, -58)},
    "channel": {"title": "Channel observer", "view": (22, -62)},
    "boeing747": {"title": "Boeing 747", "view": (21, -58)},
    "smokeBuoyancy": {
        "title": "Smoke buoyancy",
        "view": (22, -58),
        "row_hspace": 0.58,
    },
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_standard_payload(dataset):
    artifact = (
        STANDARD_ARTIFACT_ROOT
        / f"{dataset}_task3_seed{STANDARD_SEED}_predictions.npz"
    )
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    with np.load(artifact) as data:
        seeds = np.asarray(data["seeds"], dtype=np.float32)
        reference = np.asarray(data["reference"], dtype=bool)
        metadata = json.loads(str(data["metadata_json"]))
        metrics = json.loads(str(data["metrics_json"]))
        predictions = {
            "Raw-PCA residual": np.asarray(
                data["baseline_prediction"], dtype=bool
            ),
            "FMT residual": np.asarray(data["fmt_prediction"], dtype=bool),
        }
    if not (
        len(seeds) == len(reference)
        == len(predictions["Raw-PCA residual"])
        == len(predictions["FMT residual"])
    ):
        raise RuntimeError(f"{dataset}: prediction artifact length mismatch")
    return {
        "dataset": dataset,
        "protocol": "global-IVD Task3 frozen main protocol",
        "training_seed": STANDARD_SEED,
        "candidate": "all_plus_gram_kinematic",
        "artifact": str(artifact),
        "artifact_sha256": _sha256(artifact),
        "metadata": metadata,
        "seeds": seeds,
        "reference": reference,
        "predictions": predictions,
        "metrics": {
            "Raw-PCA residual": metrics["Raw-PCA residual"],
            "FMT residual": metrics["Raw+FMT residual"],
        },
    }


def _load_f22_payload(device):
    source = _load_frozen_inputs(STANDARD_ORDINAL, ANCHORED_F22_SEED, device)
    return {
        "dataset": "f22raptor",
        "protocol": "family-specific anchored FMT frozen confirmation",
        "training_seed": ANCHORED_F22_SEED,
        "candidate": source["candidate"],
        "selection_sha256": source["selection_sha256"],
        "metadata": source["metadata"],
        "seeds": source["seeds"],
        "reference": source["reference"],
        "predictions": source["predictions"],
        "metrics": source["metrics"],
        "checkpoints": source["checkpoints"],
    }


def _load_payload(dataset, device):
    return _load_f22_payload(device) if dataset == "f22raptor" else (
        _load_standard_payload(dataset)
    )


def _metric_value(metrics, key):
    if key not in metrics:
        raise KeyError(f"metrics do not contain {key!r}")
    return float(metrics[key])


def _compact_coordinate(value, span):
    precision = int(np.clip(np.ceil(-np.log10(max(float(span), 1e-12))) + 1, 1, 4))
    text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return text.replace("-", "−")


def _count_blue_hue_pixels(rgb):
    """Count saturated cyan/blue/purple pixels in a rendered RGB image."""
    values = np.asarray(rgb, dtype=np.float64)[..., :3]
    hsv = matplotlib.colors.rgb_to_hsv(values)
    blue_like = (
        (hsv[..., 0] >= 0.48)
        & (hsv[..., 0] <= 0.80)
        & (hsv[..., 1] >= 0.22)
        & (hsv[..., 2] >= 0.12)
    )
    return int(np.count_nonzero(blue_like))


def _count_scatter_collections(figure):
    """Count actual 3-D scatter layers, independent of their rendered color."""
    return int(sum(
        isinstance(collection, Path3DCollection)
        for axis in figure.axes
        for collection in axis.collections
    ))


def _spec_close_up(payload):
    """Return the prediction-independent close-up contract for one render."""
    settings = dict(DEFAULT_CLOSE_UP)
    settings.update(payload.get("close_up", {}))
    settings["focus_basis"] = str(settings["focus_basis"])
    if settings["focus_basis"] not in {
        "paired_disagreement", "ivd_positive",
    }:
        raise ValueError(
            "close_up.focus_basis must be paired_disagreement or ivd_positive"
        )
    settings["positive_fraction"] = float(settings["positive_fraction"])
    settings["minimum_points"] = int(settings["minimum_points"])
    settings["padding_factor"] = float(settings["padding_factor"])
    settings["minimum_span_fraction"] = float(
        settings["minimum_span_fraction"]
    )
    if settings["target_span_fraction"] is not None:
        target = np.asarray(settings["target_span_fraction"], dtype=np.float64)
        if target.ndim == 0:
            target = np.repeat(target, 3)
        if target.shape != (3,) or not np.all((target > 0.0) & (target <= 1.0)):
            raise ValueError(
                "close_up.target_span_fraction must be a scalar or three "
                "values in (0, 1]"
            )
        settings["target_span_fraction"] = target.tolist()
    cube_fraction = settings["physical_cube_longest_axis_fraction"]
    if cube_fraction is not None:
        cube_fraction = float(cube_fraction)
        if not 0.0 < cube_fraction <= 1.0:
            raise ValueError(
                "close_up.physical_cube_longest_axis_fraction must be in "
                "(0, 1]"
            )
        if settings["target_span_fraction"] is not None:
            raise ValueError(
                "choose target_span_fraction or "
                "physical_cube_longest_axis_fraction, not both"
            )
        settings["physical_cube_longest_axis_fraction"] = cube_fraction
    settings["camera_zoom"] = float(settings["camera_zoom"])
    settings["marker_scale"] = float(settings["marker_scale"])
    if settings["marker_scale"] <= 0.0:
        raise ValueError("close_up.marker_scale must be positive")
    settings["display_crop_scale"] = float(settings["display_crop_scale"])
    if not 0.0 < settings["display_crop_scale"] <= 1.0:
        raise ValueError("close_up.display_crop_scale must be in (0, 1]")
    settings["prediction_point_mode"] = str(
        settings["prediction_point_mode"]
    )
    if settings["prediction_point_mode"] not in {
        "confusion", "predicted_positive",
    }:
        raise ValueError(
            "close_up.prediction_point_mode must be confusion or "
            "predicted_positive"
        )
    surface_shape = np.asarray(
        settings["prediction_surface_grid_shape"], dtype=np.int64
    )
    if surface_shape.ndim == 0:
        surface_shape = np.repeat(surface_shape, 3)
    if surface_shape.shape != (3,) or np.any(surface_shape < 8):
        raise ValueError(
            "close_up.prediction_surface_grid_shape must be an integer or "
            "three integers >= 8"
        )
    settings["prediction_surface_grid_shape"] = surface_shape.tolist()
    settings["prediction_surface_neighbors"] = int(
        settings["prediction_surface_neighbors"]
    )
    if settings["prediction_surface_neighbors"] < 1:
        raise ValueError(
            "close_up.prediction_surface_neighbors must be positive"
        )
    settings["prediction_surface_power"] = float(
        settings["prediction_surface_power"]
    )
    if settings["prediction_surface_power"] <= 0.0:
        raise ValueError("close_up.prediction_surface_power must be positive")
    settings["prediction_surface_smoothing_sigma"] = float(
        settings["prediction_surface_smoothing_sigma"]
    )
    if settings["prediction_surface_smoothing_sigma"] < 0.0:
        raise ValueError(
            "close_up.prediction_surface_smoothing_sigma must be non-negative"
        )
    settings["reference_surface_alpha"] = float(
        settings["reference_surface_alpha"]
    )
    if not 0.0 < settings["reference_surface_alpha"] <= 1.0:
        raise ValueError(
            "close_up.reference_surface_alpha must be in (0, 1]"
        )
    settings["all_positive_surface_alpha"] = float(
        settings["all_positive_surface_alpha"]
    )
    if not 0.0 < settings["all_positive_surface_alpha"] < 1.0:
        raise ValueError(
            "close_up.all_positive_surface_alpha must be in (0, 1)"
        )
    if (
        settings["false_positive_color"] is not None
        and not matplotlib.colors.is_color_like(
            settings["false_positive_color"]
        )
    ):
        raise ValueError("close_up.false_positive_color is not a valid color")
    if not matplotlib.colors.is_color_like(settings["reference_context_color"]):
        raise ValueError(
            "close_up.reference_context_color is not a valid color"
        )
    for name in (
        "exact_target_span", "center_on_selected_neighbourhood",
        "show_reference_points", "show_true_negatives", "show_true_positives",
        "show_paired_error_context", "show_reference_surface_in_predictions",
        "render_prediction_surface", "render_reference_surface_from_labels",
        "render_scalar_slices",
        "require_all_prediction_surfaces_in_crop", "show_error_markers",
        "surface_only", "shade_surfaces", "show_close_up_axes",
        "fixed_canvas_export",
    ):
        if not isinstance(settings[name], bool):
            raise TypeError(f"close_up.{name} must be a boolean")
    if payload.get("close_up_only", False):
        forbidden_background_layers = {
            "show_true_negatives": settings["show_true_negatives"],
            "show_paired_error_context": settings[
                "show_paired_error_context"
            ],
        }
        enabled = sorted(
            name for name, value in forbidden_background_layers.items()
            if value
        )
        if enabled:
            raise ValueError(
                "close-up-only figures forbid background seed arrays; "
                f"disable {enabled}"
            )
    if settings["surface_only"]:
        incompatible = {
            name: settings[name]
            for name in (
                "show_reference_points", "show_true_negatives",
                "show_true_positives", "show_paired_error_context",
                "show_error_markers",
            )
            if settings[name]
        }
        if incompatible:
            raise ValueError(
                "close_up.surface_only forbids every seed-marker layer; "
                f"disable {sorted(incompatible)}"
            )
        if not (
            settings["render_prediction_surface"]
            or settings["render_scalar_slices"]
        ):
            raise ValueError(
                "close_up.surface_only requires a continuous representation: "
                "render_prediction_surface or render_scalar_slices"
            )
    return settings


def _draw_predicted_positive_points(
    ax, seeds, reference, prediction, *, bounds, marker_scale,
):
    """Draw only the model's predicted vortex support.

    The IVD-p95 panel supplies the reference. Predicted-negative points are
    omitted so a regular background lattice cannot dominate the qualitative
    spatial comparison. Complete confusion counts and metrics remain computed
    from the frozen arrays.
    """
    seeds = np.asarray(seeds)
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    visible = _visible_seed_mask(seeds, bounds)
    selected = prediction & visible
    ax.scatter(
        seeds[selected, 0], seeds[selected, 1], seeds[selected, 2],
        c=paper.COLORS["vortex"], marker="o",
        s=14.0 * float(marker_scale), alpha=0.88,
        linewidths=0.0, depthshade=False, rasterized=True,
    )
    return {
        name: int(np.count_nonzero(mask & visible))
        for name, mask in _confusion_masks(reference, prediction).items()
    }


def _restore_close_up_coordinate_axes(
    ax, bounds, show_tick_labels, range_label_y=0.975, stacked_labels=False,
):
    """Restore a compact neutral coordinate box after close-up preparation.

    Matplotlib's built-in projected 3-D axis lines extend beyond the physical
    data box and can cross figure text even after raster clipping. Replace them
    with the twelve finite physical-box edges and report the three compact
    physical ranges in a reserved figure band.
    All strokes are neutral gray, so the frame cannot be confused with
    non-vortex points.  Put the compact range labels in the reserved header
    band: a strongly magnified 3-D projection can otherwise send decorative
    axis strokes into a footer even when the scientific artists are clipped.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    range_label_y = float(range_label_y)
    if not 0.0 <= range_label_y <= 1.0:
        raise ValueError("range_label_y must be in [0, 1]")

    def compact_labels(values):
        values = np.asarray(values, dtype=np.float64)
        span = max(float(np.ptp(values)), 1e-12)
        precision = int(np.clip(np.ceil(-np.log10(span)) + 1, 1, 4))
        return [
            f"{float(value):.{precision}f}"
            .rstrip("0").rstrip(".")
            for value in values
        ]

    ticks = [
        np.asarray([bounds[0, axis], bounds[1, axis]], dtype=np.float64)
        for axis in range(3)
    ]
    setters = (
        (ax.set_xticks, ax.set_xticklabels),
        (ax.set_yticks, ax.set_yticklabels),
        (ax.set_zticks, ax.set_zticklabels),
    )
    for values, (set_ticks, set_labels) in zip(ticks, setters):
        set_ticks(values)
        set_labels([])

    if show_tick_labels:
        range_labels = [
            f"{name} [{labels[0]}, {labels[1]}]"
            for name, labels in zip(
                "xyz", (compact_labels(values) for values in ticks)
            )
        ]
        panel_box = ax.get_position()
        if stacked_labels:
            for line_index, label in enumerate(range_labels):
                ax.figure.text(
                    panel_box.x0,
                    range_label_y - 0.035 * line_index,
                    label,
                    transform=ax.figure.transFigure,
                    ha="left", va="top", fontsize=5.4, color="#333333",
                )
        else:
            # Keep the third range clear of projected front-left box strokes.
            for fraction, label in zip((0.00, 0.36, 0.75), range_labels):
                ax.figure.text(
                    panel_box.x0 + fraction * panel_box.width,
                    range_label_y,
                    label,
                    transform=ax.figure.transFigure,
                    ha="left", va="top", fontsize=6.2, color="#333333",
                )

    for axis_3d in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis_3d.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis_3d.pane.set_edgecolor("none")
        axis_3d.line.set_visible(False)
        for tick in axis_3d.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
    corners = np.asarray([
        [x, y, z]
        for x in bounds[:, 0]
        for y in bounds[:, 1]
        for z in bounds[:, 2]
    ])
    box_segments = []
    for i, first in enumerate(corners):
        for second in corners[i + 1:]:
            # A physical cuboid edge changes exactly one endpoint coordinate.
            if np.count_nonzero(~np.isclose(first, second)) != 1:
                continue
            box_segments.append(np.stack((first, second)))
    box = Line3DCollection(
        box_segments,
        colors="#777777",
        linewidths=0.60,
        alpha=0.82,
        zorder=20,
        rasterized=True,
    )
    ax.add_collection3d(box)
    ax.tick_params(labelsize=6.2, colors="#333333", pad=2, length=2)
    ax.grid(False)


def _clip_close_up_artists_to_panel(ax):
    """Keep a magnified 3-D projection inside its own subplot rectangle."""
    artists = [*ax.collections, *ax.lines]
    for axis_3d in (ax.xaxis, ax.yaxis, ax.zaxis):
        artists.extend((axis_3d.pane, axis_3d.line))
        for tick in axis_3d.get_major_ticks():
            artists.extend((tick.tick1line, tick.tick2line))
    for artist in artists:
        artist.set_clip_on(True)
        artist.set_clip_box(ax.bbox)


def _enclose_seed_roundoff(bounds, seeds):
    """Expand fixed bounds only enough to contain float32 seed roundoff."""
    bounds = np.asarray(bounds, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.float64)
    if bounds.shape != (2, 3) or seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError("bounds/seeds shape mismatch")
    seed_bounds = np.stack((seeds.min(axis=0), seeds.max(axis=0)))
    magnitude = np.maximum(1.0, np.max(np.abs(bounds), axis=0))
    tolerance = 8.0 * np.finfo(np.float32).eps * magnitude
    excess = np.maximum(bounds[0] - seed_bounds[0], 0.0) + np.maximum(
        seed_bounds[1] - bounds[1], 0.0
    )
    if np.any(excess > tolerance):
        raise ValueError(
            "fixed close-up bounds exclude seeds by more than float32 "
            "roundoff"
        )
    return np.stack((
        np.minimum(bounds[0], seed_bounds[0]),
        np.maximum(bounds[1], seed_bounds[1]),
    ))


def _tighten_roi_around_ivd_boundary(bounds, seeds, reference, scale):
    """Shrink a frozen local ROI around a central opposite-label pair.

    The crop uses only IVD labels.  Model predictions therefore cannot move the
    camera toward a visually convenient success or away from an error.  A
    regular seed grid contains many exactly tied nearest positive/negative
    pairs; choosing the first tie makes the crop depend on flat-array order and
    often moves it to a corner.  Among nearest opposite-label pairs, retain the
    pair whose midpoint is closest to the already frozen local-box center.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.float64)
    reference = np.asarray(reference, dtype=bool)
    scale = float(scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0, 1]")
    span = (bounds[1] - bounds[0]) * scale
    if np.any(reference) and np.any(~reference):
        positive = seeds[reference]
        negative = seeds[~reference]
        distance, negative_index = cKDTree(negative).query(positive, k=1)
        opposite = negative[np.asarray(negative_index, dtype=np.int64)]
        pair_midpoints = 0.5 * (positive + opposite)
        local_center = 0.5 * (bounds[0] + bounds[1])
        full_span = np.maximum(bounds[1] - bounds[0], 1e-12)
        normalized_center_distance = np.linalg.norm(
            (pair_midpoints - local_center) / full_span,
            axis=1,
        )
        # Boundary adjacency is primary; proximity to the frozen ROI center
        # deterministically resolves the many equal-distance lattice ties.
        nearest_distance = float(np.min(distance))
        tied = np.isclose(
            distance, nearest_distance,
            rtol=1e-7,
            atol=max(1e-12, nearest_distance * 1e-7),
        )
        candidate_indices = np.flatnonzero(tied)
        positive_index = int(candidate_indices[np.argmin(
            normalized_center_distance[candidate_indices]
        )])
        focus = pair_midpoints[positive_index]
    else:
        focus = 0.5 * (bounds[0] + bounds[1])
    lower = np.clip(focus - 0.5 * span, bounds[0], bounds[1] - span)
    return np.stack((lower, lower + span))


def _tighten_roi_around_disagreement_boundary(
    bounds, seeds, reference, focus_mask, scale,
):
    """Center a shared close-up on disagreement nearest the IVD boundary.

    The previous disagreement crop used the center of all disagreement seeds.
    For elongated disagreement regions that center can miss the IVD surface.
    Here the selected focus must be both a model-disagreement seed and one end
    of a nearby opposite-IVD-label pair, so the diagnostic retains its ground
    truth reference without choosing a separate crop per method.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.float64)
    reference = np.asarray(reference, dtype=bool)
    focus_mask = np.asarray(focus_mask, dtype=bool)
    scale = float(scale)
    if bounds.shape != (2, 3):
        raise ValueError("bounds must have shape (2,3)")
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0,1]")
    visible = np.all((seeds >= bounds[0]) & (seeds <= bounds[1]), axis=1)
    candidates = np.flatnonzero(visible & focus_mask)
    if not len(candidates):
        return _tighten_roi_around_ivd_boundary(
            bounds, seeds[visible], reference[visible], scale
        )
    span = (bounds[1] - bounds[0]) * scale
    source_span = np.maximum(bounds[1] - bounds[0], 1e-12)
    source_center = 0.5 * (bounds[0] + bounds[1])
    pair_midpoints = np.full((len(candidates), 3), np.nan, dtype=np.float64)
    pair_distance = np.full(len(candidates), np.inf, dtype=np.float64)
    pair_fits = np.zeros(len(candidates), dtype=bool)
    for label in (False, True):
        selected = np.flatnonzero(reference[candidates] == label)
        opposite = seeds[visible & (reference != label)]
        if not len(selected) or not len(opposite):
            continue
        distance, nearest = cKDTree(opposite).query(
            seeds[candidates[selected]], k=1
        )
        paired = opposite[np.asarray(nearest, dtype=np.int64)]
        pair_midpoints[selected] = 0.5 * (
            seeds[candidates[selected]] + paired
        )
        pair_distance[selected] = np.asarray(distance, dtype=np.float64)
        pair_fits[selected] = np.all(
            np.abs(seeds[candidates[selected]] - paired) <= 0.95 * span,
            axis=1,
        )
    eligible = np.flatnonzero(pair_fits & np.isfinite(pair_distance))
    if not len(eligible):
        return _tighten_roi_around_ivd_boundary(
            bounds, seeds[visible], reference[visible], scale
        )
    normalized_pair_distance = pair_distance / float(np.linalg.norm(source_span))
    normalized_center_distance = np.linalg.norm(
        (pair_midpoints - source_center) / source_span, axis=1
    )
    order = np.lexsort((
        normalized_center_distance[eligible],
        normalized_pair_distance[eligible],
    ))
    focus = pair_midpoints[int(eligible[int(order[0])])]
    lower = np.clip(focus - 0.5 * span, bounds[0], bounds[1] - span)
    return np.stack((lower, lower + span))


def _regular_scalar_volume(seeds, values):
    """Reconstruct one regular local scalar volume from evaluated seeds.

    Invalid pathline seeds are rare.  Missing grid cells are filled from the
    nearest valid evaluated seed solely for rendering;
    the classifier metrics continue to use only the original valid seeds.
    """
    seeds = np.asarray(seeds, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if seeds.shape != (len(values), 3):
        raise ValueError("seeds/values shape mismatch")
    axes = [np.unique(seeds[:, axis]) for axis in range(3)]
    if min(len(axis) for axis in axes) < 2:
        return None, None, 0
    nx, ny, nz = (len(axis) for axis in axes)
    volume = np.full((nz, ny, nx), np.nan, dtype=np.float64)
    ix = np.searchsorted(axes[0], seeds[:, 0])
    iy = np.searchsorted(axes[1], seeds[:, 1])
    iz = np.searchsorted(axes[2], seeds[:, 2])
    volume[iz, iy, ix] = values
    missing = ~np.isfinite(volume)
    missing_count = int(missing.sum())
    if missing_count:
        zz, yy, xx = np.meshgrid(
            axes[2], axes[1], axes[0], indexing="ij"
        )
        query = np.stack((xx[missing], yy[missing], zz[missing]), axis=-1)
        nearest = cKDTree(seeds).query(query, k=1)[1]
        volume[missing] = values[np.asarray(nearest, dtype=np.int64)]
    return axes, volume, missing_count


def _regular_probability_mesh(seeds, probabilities, level):
    """Extract an isosurface from a regular evaluated probability volume."""
    axes, volume, missing_count = _regular_scalar_volume(
        seeds, probabilities
    )
    if axes is None:
        return None, missing_count
    spacing = tuple(float(np.mean(np.diff(axis))) for axis in axes)
    mesh = marching_cubes_world(
        volume.astype(np.float32), float(level), spacing,
        (float(axes[0][0]), float(axes[1][0]), float(axes[2][0])),
    )
    return mesh, missing_count


def _scattered_probability_mesh(
    seeds,
    probabilities,
    bounds,
    level,
    *,
    grid_shape=(48, 48, 48),
    neighbors=8,
    power=2.0,
    smoothing_sigma=0.0,
):
    """Interpolate scattered probabilities and extract one local isosurface.

    The interpolation is used only for rendering. Metrics and binary
    predictions continue to use every original evaluated seed. Inverse-
    distance weighting makes the 20,480-point Sobol close-up usable as a
    continuous surface without drawing the samples as a point cloud.
    """
    axes, volume, interpolation_audit = _scattered_scalar_volume(
        seeds,
        probabilities,
        bounds,
        grid_shape=grid_shape,
        neighbors=neighbors,
        power=power,
    )
    x, y, z = axes
    bounds = np.asarray(bounds, dtype=np.float64)
    grid_shape = np.asarray(grid_shape, dtype=np.int64)
    neighbors = int(neighbors)
    power = float(power)
    spacing = (
        float(x[1] - x[0]),
        float(y[1] - y[0]),
        float(z[1] - z[0]),
    )
    smoothing_sigma = float(smoothing_sigma)
    if smoothing_sigma < 0.0:
        raise ValueError("smoothing_sigma must be non-negative")
    mesh_volume = volume.astype(np.float64) - float(level)
    mesh_level = 0.0
    if smoothing_sigma > 0.0:
        # Smooth the signed decision margin rather than probability itself.
        # This preserves the frozen threshold as the zero level while removing
        # isolated sample-sized bubbles caused by high-threshold interpolation.
        epsilon = 1e-6
        clipped = np.clip(volume.astype(np.float64), epsilon, 1.0 - epsilon)
        clipped_level = float(np.clip(level, epsilon, 1.0 - epsilon))
        margin = (
            np.log(clipped) - np.log1p(-clipped)
            - np.log(clipped_level) + np.log1p(-clipped_level)
        )
        mesh_volume = gaussian_filter(
            margin, sigma=smoothing_sigma, mode="nearest"
        )
    margin_min = float(np.min(mesh_volume))
    margin_max = float(np.max(mesh_volume))
    if margin_max <= 0.0:
        decision_region_state = "all_negative"
        mesh = None
    else:
        decision_region_state = (
            "all_positive" if margin_min > 0.0 else "mixed"
        )
        if decision_region_state == "all_positive":
            # A full positive crop has no internal level crossing. Represent
            # the clipped positive volume explicitly as the crop cuboid.
            vertices = np.asarray([
                [x, y, z]
                for x in bounds[:, 0]
                for y in bounds[:, 1]
                for z in bounds[:, 2]
            ], dtype=np.float32)
            faces = np.asarray([
                [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
            ], dtype=np.uint32)
            mesh = (
                vertices,
                np.zeros_like(vertices, dtype=np.float32),
                faces,
            )
        else:
            mesh = marching_cubes_world(
                mesh_volume.astype(np.float32),
                mesh_level,
                spacing,
                tuple(float(value) for value in bounds[0]),
            )
    audit = {
        **interpolation_audit,
        "threshold": float(level),
        "decision_margin_smoothing_sigma_voxels": smoothing_sigma,
        "decision_region_state": decision_region_state,
        "crop_boundary_closed": decision_region_state == "all_positive",
        "surface_present": mesh is not None,
    }
    return mesh, audit


def _scattered_scalar_volume(
    seeds,
    values,
    bounds,
    *,
    grid_shape=(48, 48, 48),
    neighbors=8,
    power=2.0,
):
    """Interpolate scattered scalar samples onto one bounded regular volume.

    This rendering-only reconstruction supports the Sobol close-up samples.
    It avoids the invalid ``unique(x) x unique(y) x unique(z)`` allocation
    that a regular-grid reconstruction would attempt for scattered points.
    No interpolated value is used for metrics or for threshold selection.
    """
    seeds = np.asarray(seeds, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    grid_shape = np.asarray(grid_shape, dtype=np.int64)
    neighbors = int(neighbors)
    power = float(power)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError("seeds must have shape (N, 3)")
    if values.shape != (len(seeds),):
        raise ValueError("values must have shape (N,)")
    if bounds.shape != (2, 3) or np.any(bounds[0] >= bounds[1]):
        raise ValueError("bounds must have shape (2, 3) with positive spans")
    if grid_shape.shape != (3,) or np.any(grid_shape < 8):
        raise ValueError("grid_shape must contain three integers >= 8")
    if neighbors < 1 or power <= 0.0:
        raise ValueError("neighbors and power must be positive")
    if not np.all(np.isfinite(seeds)) or not np.all(np.isfinite(values)):
        raise ValueError("seeds and values must be finite")

    x = np.linspace(bounds[0, 0], bounds[1, 0], int(grid_shape[0]))
    y = np.linspace(bounds[0, 1], bounds[1, 1], int(grid_shape[1]))
    z = np.linspace(bounds[0, 2], bounds[1, 2], int(grid_shape[2]))
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    query = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    k = min(neighbors, len(seeds))
    distance, index = cKDTree(seeds).query(query, k=k)
    if k == 1:
        distance = np.asarray(distance)[:, None]
        index = np.asarray(index)[:, None]
    else:
        distance = np.asarray(distance)
        index = np.asarray(index)
    scale = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1.0)
    epsilon = np.finfo(np.float64).eps * scale * 32.0
    exact = distance <= epsilon
    safe_distance = np.maximum(distance, epsilon)
    weight = safe_distance ** (-power)
    interpolated = np.sum(weight * values[index], axis=1) / np.sum(
        weight, axis=1
    )
    exact_rows = np.any(exact, axis=1)
    if np.any(exact_rows):
        exact_choice = np.argmax(exact[exact_rows], axis=1)
        interpolated[exact_rows] = values[
            index[exact_rows, exact_choice]
        ]
    volume = interpolated.reshape(
        int(grid_shape[2]), int(grid_shape[1]), int(grid_shape[0])
    ).astype(np.float32)
    audit = {
        "source_sample_count": int(len(seeds)),
        "grid_shape_xyz": grid_shape.tolist(),
        "neighbors": int(k),
        "inverse_distance_power": float(power),
        "interpolated_min": float(volume.min()),
        "interpolated_max": float(volume.max()),
    }
    return (x, y, z), volume, audit


def _draw_scalar_slices(
    ax, axes, volume, bounds, *, threshold, reference=False,
):
    """Draw three orthogonal scalar slices without any seed-point marks."""
    bounds = np.asarray(bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    selected = []
    for coordinates, low, high in zip(axes, bounds[0], bounds[1]):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        tolerance = max(float(np.ptp(coordinates)), 1.0) * 1e-10
        indices = np.flatnonzero(
            (coordinates >= low - tolerance)
            & (coordinates <= high + tolerance)
        )
        if len(indices) < 2:
            return 0
        selected.append(indices)
    ix, iy, iz = selected
    cx = int(ix[np.argmin(np.abs(np.asarray(axes[0])[ix] - center[0]))])
    cy = int(iy[np.argmin(np.abs(np.asarray(axes[1])[iy] - center[1]))])
    cz = int(iz[np.argmin(np.abs(np.asarray(axes[2])[iz] - center[2]))])
    x = np.asarray(axes[0])[ix]
    y = np.asarray(axes[1])[iy]
    z = np.asarray(axes[2])[iz]

    if reference:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "ivd_binary", ["#fffdf9", paper.COLORS["ivd"]]
        )
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    else:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "task3_decision_margin",
            ["#bdbdbd", "#fffdf9", "#f4a261", "#d62728"],
            N=256,
        )
        norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

    def facecolors(values):
        values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
        rgba = cmap(norm(values))
        if reference:
            rgba[..., 3] = 1.0
        else:
            epsilon = 1e-6
            clipped_values = np.clip(values, epsilon, 1.0 - epsilon)
            clipped_threshold = float(np.clip(
                threshold, epsilon, 1.0 - epsilon
            ))
            logit_values = np.log(clipped_values) - np.log1p(-clipped_values)
            logit_threshold = (
                np.log(clipped_threshold) - np.log1p(-clipped_threshold)
            )
            # A fixed three-logit-unit range makes Raw and FMT panels
            # comparable while preserving the sign of the frozen decision.
            margin = np.clip(
                (logit_values - logit_threshold) / 3.0, -1.0, 1.0
            )
            rgba = cmap(norm(margin))
            # Transparent 3-D grid cells expose polygon boundaries after
            # projection and can look like a background point lattice.  The
            # slice is a color-encoded decision field, so keep it opaque and
            # encode confidence only through color intensity.
            rgba[..., 3] = 1.0
        return rgba

    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy_values = volume[cz][np.ix_(iy, ix)]
    ax.plot_surface(
        xx, yy, np.full_like(xx, np.asarray(axes[2])[cz]),
        facecolors=facecolors(xy_values), shade=False, antialiased=False,
        linewidth=0, rcount=xy_values.shape[0], ccount=xy_values.shape[1],
        rasterized=True,
    )

    xx, zz = np.meshgrid(x, z, indexing="xy")
    xz_values = volume[np.ix_(iz, [cy], ix)][:, 0, :]
    ax.plot_surface(
        xx, np.full_like(xx, np.asarray(axes[1])[cy]), zz,
        facecolors=facecolors(xz_values), shade=False, antialiased=False,
        linewidth=0, rcount=xz_values.shape[0], ccount=xz_values.shape[1],
        rasterized=True,
    )

    yy, zz = np.meshgrid(y, z, indexing="xy")
    yz_values = volume[np.ix_(iz, iy, [cx])][:, :, 0]
    ax.plot_surface(
        np.full_like(yy, np.asarray(axes[0])[cx]), yy, zz,
        facecolors=facecolors(yz_values), shade=False, antialiased=False,
        linewidth=0, rcount=yz_values.shape[0], ccount=yz_values.shape[1],
        rasterized=True,
    )
    return 3


def _draw_surface(ax, mesh, bounds, color, alpha, zorder, shade=False):
    triangles = _clipped_mesh_triangles(mesh, bounds)
    if not len(triangles):
        return 0
    facecolor = color
    if shade:
        edges_a = triangles[:, 1] - triangles[:, 0]
        edges_b = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(edges_a, edges_b)
        normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(normal_length, 1e-12)
        light = np.asarray([0.35, -0.45, 0.82], dtype=np.float64)
        light /= np.linalg.norm(light)
        diffuse = np.abs(normals @ light)
        brightness = 0.72 + 0.28 * diffuse
        base = np.asarray(matplotlib.colors.to_rgb(color), dtype=np.float64)
        rgb = np.clip(brightness[:, None] * base[None, :], 0.0, 1.0)
        facecolor = np.column_stack((rgb, np.full(len(rgb), float(alpha))))
    collection = Poly3DCollection(
        triangles, facecolor=facecolor, edgecolor="none",
        alpha=None if shade else float(alpha),
        linewidths=0.0, antialiaseds=False,
        zorder=int(zorder), rasterized=True,
    )
    ax.add_collection3d(collection)
    return int(len(triangles))


def _render_one(dataset, payload, output_dir, dpi):
    spec = FLOW_SPECS[dataset]
    task = str(payload.get("task", "task3")).lower()
    expected_first_method = {
        "task2": "Raw+VAE",
        "task3": "Raw-PCA residual",
    }.get(task)
    if expected_first_method is None:
        raise ValueError(f"unsupported visualization task {task!r}")
    surface_cache_dir = Path(payload.get(
        "ivd_surface_cache_root", output_dir / "ivd_surface_cache"
    ))
    surface = reconstruct_ivd_reference_surface(
        ROOT,
        dataset,
        payload["metadata"],
        payload["seeds"],
        payload["reference"],
        surface_cache_dir,
    )
    if int(surface["ivd_surface_audit"]["recomputed_label_mismatch_count"]):
        raise RuntimeError(f"{dataset}: reconstructed IVD labels changed")

    bounds = np.asarray(surface["bounds"], dtype=np.float64)
    close_up = _spec_close_up(payload)
    full_span = np.maximum(bounds[1] - bounds[0], 1e-12)
    effective_target_span_fraction = close_up["target_span_fraction"]
    if close_up["physical_cube_longest_axis_fraction"] is not None:
        cube_side = (
            close_up["physical_cube_longest_axis_fraction"]
            * float(np.max(full_span))
        )
        effective_target_span_fraction = np.minimum(
            cube_side / full_span, 1.0
        ).tolist()
    method_names = list(payload["predictions"])
    if len(method_names) != 2 or method_names[0] != expected_first_method:
        raise RuntimeError(f"{dataset}: unexpected method order {method_names}")
    reference = np.asarray(payload["reference"], dtype=bool)
    explicit_render_thresholds = payload.get("render_thresholds", {})

    def render_threshold(method):
        if method in explicit_render_thresholds:
            return float(explicit_render_thresholds[method])
        return _metric_value(payload["metrics"][method], "threshold")

    probabilities_by_method = payload.get("probabilities")
    if (
        close_up["render_prediction_surface"]
        or close_up["render_scalar_slices"]
    ):
        if probabilities_by_method is None:
            raise RuntimeError(
                f"{dataset}: probability rendering requires frozen probabilities"
            )
        if set(probabilities_by_method) != set(method_names):
            raise RuntimeError(
                f"{dataset}: probability methods differ from predictions"
            )
    prediction_meshes = {}
    prediction_surface_interpolation = {}
    reference_render_mesh = surface["ivd_mesh"]
    reference_surface_interpolation = None
    scalar_axes = None
    scalar_slice_fill_counts = {}
    scalar_slice_interpolation = {}
    probability_volumes = {}
    first_prediction = np.asarray(
        payload["predictions"][method_names[0]], dtype=bool
    )
    second_prediction = np.asarray(
        payload["predictions"][method_names[1]], dtype=bool
    )
    paired_error_context = (
        (first_prediction != reference) | (second_prediction != reference)
    )
    disagreement = first_prediction != second_prediction
    fixed_close_up_bounds = payload.get("fixed_close_up_bounds")
    source_roi_bounds = None
    if fixed_close_up_bounds is not None:
        source_roi_bounds = np.asarray(fixed_close_up_bounds, dtype=np.float64)
        if (
            source_roi_bounds.shape != (2, 3)
            or np.any(source_roi_bounds[0] >= source_roi_bounds[1])
        ):
            raise ValueError("fixed_close_up_bounds must have shape (2,3)")
        if np.isclose(close_up["display_crop_scale"], 1.0):
            source_roi_bounds = _enclose_seed_roundoff(
                source_roi_bounds, payload["seeds"]
            )
        if (
            np.any(source_roi_bounds[0] < bounds[0])
            or np.any(source_roi_bounds[1] > bounds[1])
        ):
            raise ValueError("fixed_close_up_bounds exceed the physical domain")
        if close_up["focus_basis"] == "ivd_positive":
            focus_mask = reference
            close_up_basis = "frozen global-IVD local sampling box"
            roi_bounds = _tighten_roi_around_ivd_boundary(
                source_roi_bounds,
                payload["seeds"],
                reference,
                close_up["display_crop_scale"],
            )
            if close_up["display_crop_scale"] < 1.0:
                close_up_basis += (
                    " tightened around the nearest IVD boundary pair"
                )
        else:
            focus_candidates = (
                (disagreement, "Raw/FMT disagreement"),
                (paired_error_context, "paired error union"),
                (reference, "IVD-positive fallback"),
            )
            focus_mask, focus_name = next(
                (mask, name) for mask, name in focus_candidates if np.any(mask)
            )
            roi_bounds = _tighten_roi_around_disagreement_boundary(
                source_roi_bounds,
                payload["seeds"],
                reference,
                focus_mask,
                close_up["display_crop_scale"],
            )
            close_up_basis = (
                "frozen local sampling box tightened around the nearest "
                "IVD boundary pair incident to " + focus_name
            )
    else:
        if close_up["focus_basis"] == "ivd_positive":
            focus_candidates = ((reference, "densest IVD-positive region"),)
        else:
            focus_candidates = (
                (disagreement, "Raw/FMT disagreement"),
                (paired_error_context, "paired error union"),
                (reference, "IVD-positive fallback"),
            )
        focus_mask, close_up_basis = next(
            (mask, name) for mask, name in focus_candidates if np.any(mask)
        )
        roi_bounds = _vortex_roi(
            bounds,
            payload["seeds"],
            focus_mask,
            positive_fraction=close_up["positive_fraction"],
            minimum_points=close_up["minimum_points"],
            padding_factor=close_up["padding_factor"],
            minimum_span_fraction=close_up["minimum_span_fraction"],
            target_span_fraction=effective_target_span_fraction,
            exact_target_span=close_up["exact_target_span"],
            center_on_selected_neighbourhood=close_up[
                "center_on_selected_neighbourhood"
            ],
        )
    close_up_visible = np.all(
        (np.asarray(payload["seeds"]) >= roi_bounds[0])
        & (np.asarray(payload["seeds"]) <= roi_bounds[1]),
        axis=1,
    )
    if not np.any(focus_mask & close_up_visible):
        # ``_tighten_roi_around_disagreement_boundary`` deliberately falls
        # back to an IVD-only boundary crop when no disagreement seed can be
        # paired with a nearby opposite IVD label.  Preserve that
        # prediction-independent fallback instead of rejecting a valid crop
        # merely because it no longer contains the unavailable disagreement.
        local_reference = reference[close_up_visible]
        if (
            close_up["focus_basis"] == "paired_disagreement"
            and np.any(local_reference)
            and np.any(~local_reference)
        ):
            focus_mask = reference
            close_up_basis += "; IVD-boundary fallback (no eligible paired disagreement)"
        else:
            raise RuntimeError(
                f"{dataset}: close-up crop lost every seed used to choose its "
                "focus"
            )
    if close_up["render_scalar_slices"]:
        # Reconstruct directly on the final physical field of view.  Building
        # a coarse volume on the larger source box and cropping it afterwards
        # undersamples strong close-ups and can leave only a few cells per
        # displayed axis.  This interpolation is rendering-only: labels,
        # predictions, thresholds, and reported metrics remain seedwise.
        for method in method_names:
            (
                method_axes,
                method_volume,
                method_interpolation,
            ) = _scattered_scalar_volume(
                payload["seeds"],
                probabilities_by_method[method],
                roi_bounds,
                grid_shape=close_up["prediction_surface_grid_shape"],
                neighbors=close_up["prediction_surface_neighbors"],
                power=close_up["prediction_surface_power"],
            )
            if scalar_axes is None:
                scalar_axes = method_axes
            elif any(
                not np.array_equal(a, b)
                for a, b in zip(scalar_axes, method_axes)
            ):
                raise RuntimeError(
                    f"{dataset}/{method}: scalar slice grids changed"
                )
            probability_volumes[method] = method_volume
            scalar_slice_fill_counts[method] = 0
            scalar_slice_interpolation[method] = method_interpolation
    if close_up["render_prediction_surface"]:
        for method in method_names:
            (
                prediction_meshes[method],
                prediction_surface_interpolation[method],
            ) = _scattered_probability_mesh(
                payload["seeds"],
                probabilities_by_method[method],
                roi_bounds,
                render_threshold(method),
                grid_shape=close_up["prediction_surface_grid_shape"],
                neighbors=close_up["prediction_surface_neighbors"],
                power=close_up["prediction_surface_power"],
                smoothing_sigma=close_up[
                    "prediction_surface_smoothing_sigma"
                ],
            )
    if close_up["render_reference_surface_from_labels"]:
        (
            reference_render_mesh,
            reference_surface_interpolation,
        ) = _scattered_probability_mesh(
            payload["seeds"],
            reference.astype(np.float64),
            roi_bounds,
            0.5,
            grid_shape=close_up["prediction_surface_grid_shape"],
            neighbors=close_up["prediction_surface_neighbors"],
            power=close_up["prediction_surface_power"],
            smoothing_sigma=close_up[
                "prediction_surface_smoothing_sigma"
            ],
        )
    local_paired_error_context_count = int(np.count_nonzero(
        paired_error_context & close_up_visible
    ))
    close_up_surface_triangles = {
        "IVD-p95": int(len(_clipped_mesh_triangles(
            reference_render_mesh, roi_bounds
        ))),
    }
    if close_up["render_prediction_surface"]:
        close_up_surface_triangles.update({
            method: int(len(_clipped_mesh_triangles(
                prediction_meshes[method], roi_bounds
            )))
            for method in method_names
        })
    if close_up["surface_only"]:
        # Orthogonal scalar slices are a complete continuous representation;
        # they do not require a marching-cubes isosurface to intersect the
        # selected box.  The mesh requirement applies only to mesh rendering.
        required_surfaces = (
            [] if close_up["render_scalar_slices"] else ["IVD-p95"]
        )
        if (
            close_up["render_prediction_surface"]
            and close_up["require_all_prediction_surfaces_in_crop"]
        ):
            required_surfaces.extend(method_names)
        empty_surfaces = sorted(
            name for name in required_surfaces
            if close_up_surface_triangles[name] == 0
        )
        if empty_surfaces:
            raise RuntimeError(
                f"{dataset}: close-up crop removes required surfaces "
                f"{empty_surfaces}; widen the prediction-independent crop"
            )
    close_up_scale_text = (
        "IVD-p95 close-up\n"
        + "x["
        + _compact_coordinate(roi_bounds[0, 0], roi_bounds[1, 0] - roi_bounds[0, 0])
        + ", "
        + _compact_coordinate(roi_bounds[1, 0], roi_bounds[1, 0] - roi_bounds[0, 0])
        + "]  y["
        + _compact_coordinate(roi_bounds[0, 1], roi_bounds[1, 1] - roi_bounds[0, 1])
        + ", "
        + _compact_coordinate(roi_bounds[1, 1], roi_bounds[1, 1] - roi_bounds[0, 1])
        + "]  z["
        + _compact_coordinate(roi_bounds[0, 2], roi_bounds[1, 2] - roi_bounds[0, 2])
        + ", "
        + _compact_coordinate(roi_bounds[1, 2], roi_bounds[1, 2] - roi_bounds[0, 2])
        + "]"
    )

    close_up_only = bool(payload.get("close_up_only", False))
    row_bounds_sequence = (roi_bounds,) if close_up_only else (
        bounds, roi_bounds
    )
    # Final physical width is one double-column plate (183 mm).  Render the
    # PNG at 600 dpi when a 4320-pixel preview is needed; do not double the PDF
    # page width and rely on downstream scaling, which would shrink 6.2 pt text.
    figure = plt.figure(
        figsize=(7.2, 2.55 if close_up_only else 4.1), facecolor="white"
    )
    grid_left = 0.035 if close_up_only else 0.04
    grid_right = 0.965 if close_up_only else 0.92
    grid_bottom = (
        0.245
        if close_up_only and close_up["show_close_up_axes"] else
        0.10 if close_up_only else 0.19
    )
    # Strong 3-D camera zoom projects box edges outside the nominal subplot.
    # Reserve a real header band for the coordinate-range text and panel IDs.
    grid_top = 0.92 if close_up_only else 0.90
    grid = figure.add_gridspec(
        len(row_bounds_sequence), 3,
        left=grid_left, right=grid_right,
        bottom=grid_bottom, top=grid_top,
        wspace=0.06,
        hspace=float(spec.get("row_hspace", 0.48)),
    )
    axes = [
        [figure.add_subplot(grid[row, column], projection="3d")
         for column in range(3)]
        for row in range(len(row_bounds_sequence))
    ]
    panel_ids = list("abc" if close_up_only else "abcdef")
    panel_label_x_pt = 0.0 if close_up_only else -14.0
    panel_label_y_pt = 0.0 if close_up_only else 24.0
    panel_label_xy = (0.02, 1.02) if close_up_only else (0.0, 1.0)
    close_up_title_pad = 104 if dataset == "smokeBuoyancy" else 92
    for axis, panel_id in zip([axis for row in axes for axis in row], panel_ids):
        axis.set_gid(panel_id)
        panel_label_transform = axis.transAxes + ScaledTranslation(
            panel_label_x_pt / 72.0,
            panel_label_y_pt / 72.0,
            figure.dpi_scale_trans,
        )
        axis.text2D(
            *panel_label_xy, panel_id, transform=panel_label_transform,
            fontsize=(8 if close_up_only else 10),
            fontweight="bold", va="bottom", ha="left",
        )

    counts = {
        method: {
            name: int(mask.sum())
            for name, mask in _confusion_masks(
                reference, payload["predictions"][method]
            ).items()
        }
        for method in method_names
    }
    for row_index, (row_axes, row_bounds) in enumerate(
        zip(axes, row_bounds_sequence)
    ):
        is_close_up = close_up_only or row_index == 1
        row_reference_mesh = (
            reference_render_mesh if is_close_up else surface["ivd_mesh"]
        )
        if is_close_up and close_up["surface_only"]:
            _draw_surface(
                row_axes[0], row_reference_mesh, row_bounds,
                paper.COLORS["ivd"],
                alpha=close_up["reference_surface_alpha"], zorder=1,
                shade=close_up["shade_surfaces"],
            )
        else:
            _draw_reference(
                row_axes[0], payload["seeds"], payload["reference"],
                row_reference_mesh,
                bounds=row_bounds if is_close_up else None,
                show_background=False,
                show_positive_points=(
                    not is_close_up or close_up["show_reference_points"]
                ),
                marker_scale=(
                    close_up["marker_scale"] if is_close_up else 1.0
                ),
            )
        # The reference panel already shows the IVD-p95 isosurface.  Adding
        # three interpolated binary-label slices would cover that geometry and
        # reintroduce a regular cell pattern that is not part of the data.
        if not close_up_only:
            row_axes[0].set_title(
                (
                    "IVD-p95 ground truth\n"
                    f"positive={payload['reference'].mean():.1%}"
                    if not is_close_up else ""
                ),
                fontsize=10,
                pad=close_up_title_pad if is_close_up else 10,
            )
        for axis, method in zip(row_axes[1:], method_names):
            if is_close_up and close_up["render_scalar_slices"]:
                if close_up["show_reference_surface_in_predictions"]:
                    _draw_surface(
                        axis, row_reference_mesh, row_bounds,
                        close_up["reference_context_color"],
                        alpha=1.0, zorder=1,
                        # Per-triangle lighting on a transparent context mesh
                        # creates a false triangular background lattice.
                        shade=False,
                    )
                _draw_scalar_slices(
                    axis, scalar_axes, probability_volumes[method], row_bounds,
                    threshold=render_threshold(method),
                    reference=False,
                )
                visible = _visible_seed_mask(payload["seeds"], row_bounds)
                method_counts = {
                    name: int(np.count_nonzero(mask & visible))
                    for name, mask in _confusion_masks(
                        reference, payload["predictions"][method]
                    ).items()
                }
            elif is_close_up and close_up["render_prediction_surface"]:
                if close_up["show_reference_surface_in_predictions"]:
                    _draw_surface(
                        axis, row_reference_mesh, row_bounds,
                        close_up["reference_context_color"],
                        alpha=1.0, zorder=1,
                        shade=False,
                    )
                _draw_surface(
                    axis, prediction_meshes[method], row_bounds,
                    paper.COLORS["vortex"],
                    alpha=(
                        close_up["all_positive_surface_alpha"]
                        if prediction_surface_interpolation[method][
                            "decision_region_state"
                        ] == "all_positive" else 1.00
                    ),
                    zorder=2,
                    shade=close_up["shade_surfaces"],
                )
                visible = _visible_seed_mask(payload["seeds"], row_bounds)
                method_counts = {
                    name: int(np.count_nonzero(mask & visible))
                    for name, mask in _confusion_masks(
                        reference, payload["predictions"][method]
                    ).items()
                }
                if close_up["show_error_markers"]:
                    _draw_prediction(
                        axis,
                        payload["seeds"],
                        reference,
                        payload["predictions"][method],
                        error_focus=True,
                        bounds=row_bounds,
                        show_true_negatives=False,
                        show_true_positives=False,
                        false_positive_color=close_up["false_positive_color"],
                        marker_scale=close_up["marker_scale"],
                    )
            else:
                if (
                    is_close_up
                    and close_up["show_reference_surface_in_predictions"]
                ):
                    _draw_surface(
                        axis, row_reference_mesh, row_bounds,
                        close_up["reference_context_color"],
                        alpha=1.0, zorder=1,
                        # Keep the shared spatial reference continuous; shaded
                        # transparent triangles resemble a point/mesh array.
                        shade=False,
                    )
                if (
                    is_close_up
                    and close_up["prediction_point_mode"]
                    == "predicted_positive"
                ):
                    method_counts = _draw_predicted_positive_points(
                        axis,
                        payload["seeds"],
                        payload["reference"],
                        payload["predictions"][method],
                        bounds=row_bounds,
                        marker_scale=close_up["marker_scale"],
                    )
                else:
                    method_counts = _draw_prediction(
                        axis,
                        payload["seeds"],
                        payload["reference"],
                        payload["predictions"][method],
                        error_focus=is_close_up,
                        bounds=row_bounds if is_close_up else None,
                        show_true_negatives=(
                            not is_close_up and close_up["show_true_negatives"]
                        ),
                        show_true_positives=(
                            not is_close_up or close_up["show_true_positives"]
                        ),
                        context_mask=(
                            paired_error_context
                            if is_close_up
                            and close_up["show_paired_error_context"] else None
                        ),
                        false_positive_color=close_up[
                            "false_positive_color"
                        ],
                        marker_scale=(
                            close_up["marker_scale"] if is_close_up else 1.0
                        ),
                    )
            metrics = payload["metrics"][method]
            if not close_up_only:
                axis.set_title(
                    (
                        f"{method}\nF1={_metric_value(metrics, 'f1'):.3f}, "
                        f"AP={_metric_value(metrics, 'average_precision'):.3f}"
                        if not is_close_up else
                        f"Local errors — FP={method_counts['false_positive']}, "
                        f"FN={method_counts['false_negative']}"
                    ),
                    fontsize=10,
                    pad=close_up_title_pad if is_close_up else 10,
                )
        for column_index, axis in enumerate(row_axes):
            _prepare_axis(
                axis, row_bounds, spec["view"], close_up=is_close_up,
                close_up_zoom=float(spec.get(
                    "close_up_zoom", close_up["camera_zoom"]
                )),
                show_tick_labels=column_index == 0,
            )
            if is_close_up:
                if close_up["show_close_up_axes"]:
                    _restore_close_up_coordinate_axes(
                        axis, row_bounds,
                        show_tick_labels=column_index == 0,
                        # Keep compact coordinate ranges in the footer for a
                        # close-up-only plate. Projected 3-D box edges are
                        # clipped above this reserved band.
                        # Keep the stacked physical ranges below every
                        # rasterized 3-D panel. At 0.145 the first line can
                        # touch the clipped panel fill by about one point.
                        range_label_y=0.120 if close_up_only else 0.975,
                        stacked_labels=close_up_only,
                    )
                    if close_up["fixed_canvas_export"]:
                        _clip_close_up_artists_to_panel(axis)
                else:
                    # A white, grid-free close-up backplane prevents the regular
                    # sampling lattice from being visually suggested by axes.
                    for axis_3d in (axis.xaxis, axis.yaxis, axis.zaxis):
                        axis_3d.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                        # Axes3D projects its decorative spine strokes far
                        # beyond the subplot under strong projection zoom.
                        axis_3d.pane.set_edgecolor("none")
                        axis_3d.line.set_visible(False)
                    axis.grid(False)
                    # Configurations without a coordinate box retain the
                    # compact physical range printed above panel a.
                    axis.set_axis_off()
                if column_index == 0:
                    if not close_up_only:
                        axis.text2D(
                            0.5, -0.12, close_up_scale_text,
                            transform=axis.transAxes, ha="center", va="top",
                            fontsize=7.0, color="#333333",
                        )

    if close_up_only and not close_up["show_close_up_axes"]:
        figure.text(
            0.250, 0.975,
            close_up_scale_text.replace("IVD-p95 close-up\n", ""),
            ha="center", va="top", fontsize=7.0, color="#333333",
        )

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="none",
               markerfacecolor=paper.COLORS["vortex"],
               label="True positive", markersize=6),
        Line2D([0], [0], marker="^", linestyle="none", color="none",
               markerfacecolor=(
                   close_up["false_positive_color"]
                   or paper.COLORS["false_positive"]
               ),
               label="False positive", markersize=6),
        Line2D([0], [0], marker="x", linestyle="none",
               color=paper.COLORS["false_negative"],
               label="False negative", markersize=6),
        Line2D([0], [0], color=paper.COLORS["ivd"], linewidth=6,
               alpha=0.45, label="IVD-p95 isosurface"),
    ]
    if not close_up_only:
        figure.legend(
            handles=handles, loc="lower center", ncol=4, frameon=False,
            fontsize=8, bbox_to_anchor=(0.5, 0.018),
            handletextpad=0.8, columnspacing=3.0,
        )
        figure.suptitle(
            f"{spec['title']} — {task.title()} frozen confirmation "
            f"(source index {payload['metadata']['source_start_index']}, "
            f"seed {payload['training_seed']})",
            fontsize=12, y=0.985,
        )

    prefix = output_dir / (
        f"{dataset}_{task}_closeup" if close_up_only
        else f"{dataset}_{task}_diagnostic"
    )
    row_groups = [panel_ids] if close_up_only else [
        ["a", "b", "c"], ["d", "e", "f"]
    ]
    column_groups = [] if close_up_only else [
        ["a", "d"], ["b", "e"], ["c", "f"]
    ]
    require_matplotlib_panel_alignment(
        figure,
        axes=[axis for row in axes for axis in row],
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        json_out=prefix.with_suffix(".alignment.json"),
        overlay_svg=prefix.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    scatter_collection_count = _count_scatter_collections(figure)
    if close_up["surface_only"] and scatter_collection_count:
        raise RuntimeError(
            f"{dataset}: surface-only figure contains "
            f"{scatter_collection_count} scatter collection(s)"
        )
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")
    svg_path = prefix.with_suffix(".svg")
    tiff_path = prefix.with_suffix(".tiff")
    tight_export = (
        {"bbox_inches": "tight", "pad_inches": 0.04}
        if (
            close_up["show_close_up_axes"]
            and not close_up["fixed_canvas_export"]
        ) else {}
    )
    figure.savefig(
        png_path, dpi=int(dpi), facecolor="white", edgecolor="none",
        **tight_export,
    )
    figure.savefig(
        pdf_path, facecolor="white", edgecolor="none", **tight_export,
    )
    figure.savefig(
        svg_path, facecolor="white", edgecolor="none", **tight_export,
    )
    figure.savefig(
        tiff_path,
        dpi=600,
        facecolor="white",
        edgecolor="none",
        pil_kwargs={"compression": "tiff_lzw"},
        **tight_export,
    )
    rendered_rgb = np.asarray(plt.imread(png_path))[..., :3]
    background_rgb = np.asarray(
        matplotlib.colors.to_rgb(paper.COLORS["non_vortex"]),
        dtype=np.float64,
    )
    background_blue_pixels = int(np.count_nonzero(
        np.linalg.norm(rendered_rgb - background_rgb, axis=-1) < 0.08
    ))
    if background_blue_pixels:
        raise RuntimeError(
            f"{dataset}: rendered {background_blue_pixels} non-vortex blue "
            "pixels although the background point layer is disabled"
        )
    blue_hue_pixels = _count_blue_hue_pixels(rendered_rgb)
    if blue_hue_pixels:
        raise RuntimeError(
            f"{dataset}: rendered {blue_hue_pixels} saturated blue-like "
            "pixels although the close-up palette forbids blue"
        )
    if paper.COLORS["non_vortex"].lower() in svg_path.read_text(
        encoding="utf-8"
    ).lower():
        raise RuntimeError(
            f"{dataset}: non-vortex blue remains in the vector export"
        )
    plt.close(figure)

    audit = {
        "experiment": payload.get(
            "render_experiment", "Visualize_Task3_Multiflow_Diagnostic_1.2"
        ),
        "dataset": dataset,
        "task": task,
        "layout": "close-up-only 1x3" if close_up_only else "full+close-up 2x3",
        "title": spec["title"],
        "protocol": payload["protocol"],
        "candidate": payload["candidate"],
        "training_seed": int(payload["training_seed"]),
        "confirmation_ordinal": int(payload.get(
            "confirmation_ordinal", STANDARD_ORDINAL
        )),
        "source_start_index": int(payload["metadata"]["source_start_index"]),
        "sample_count": int(len(payload["seeds"])),
        "positive_count": int(payload["reference"].sum()),
        "metrics": payload["metrics"],
        "render_thresholds": {
            method: render_threshold(method) for method in method_names
        },
        "scalar_field_provenance": payload.get(
            "scalar_field_provenance",
            "frozen model probability",
        ),
        "confusion_counts": counts,
        "ivd_surface_audit": surface["ivd_surface_audit"],
        "full_bounds": bounds.tolist(),
        "close_up_bounds": roi_bounds.tolist(),
        "source_local_sampling_bounds": (
            None if source_roi_bounds is None else source_roi_bounds.tolist()
        ),
        "close_up_basis": close_up_basis,
        "paired_error_context_count": int(paired_error_context.sum()),
        "local_paired_error_context_count": local_paired_error_context_count,
        "close_up_visible_seed_count": int(close_up_visible.sum()),
        "close_up_visible_positive_count": int(
            np.count_nonzero(close_up_visible & reference)
        ),
        "close_up_span_fraction": (
            (roi_bounds[1] - roi_bounds[0])
            / np.maximum(bounds[1] - bounds[0], 1e-12)
        ).tolist(),
        "close_up_rule": (
            "smallest normalized-coordinate ball centered on the symmetric "
            f"{close_up_basis} mask and containing "
            f"{close_up['positive_fraction']:.1%} of its seeds "
            "(subject to a minimum of "
            f"{close_up['minimum_points']} points); "
            + (
                "fixed normalized span "
                f"{np.asarray(effective_target_span_fraction).tolist()}"
                if effective_target_span_fraction is not None else
                "selected-point bounding box expanded by "
                f"{(close_up['padding_factor'] - 1.0):.0%}"
            )
            + (
                " enforced as an exact crop"
                if close_up["exact_target_span"] else
                " used as a lower bound"
            )
            + (
                "; equal physical side lengths set to "
                f"{close_up['physical_cube_longest_axis_fraction']:.1%} "
                "of the longest domain axis"
                if close_up["physical_cube_longest_axis_fraction"] is not None
                else ""
            )
            + f", minimum {close_up['minimum_span_fraction']:.1%} domain "
            "span per axis and camera zoom "
            f"{float(spec.get('close_up_zoom', close_up['camera_zoom'])):.2f}; "
            f"the frozen local box is displayed at "
            f"{close_up['display_crop_scale']:.0%} of its side length; "
            "the same symmetric crop is used for both paired methods"
        ),
        "render_contract": (
            "all observations retained for metrics; within each row the "
            "three panels use identical evaluated seeds, labels, bounds, and "
            "camera; full-domain and close-up panels omit blue true-negative "
            "marks; the close-up reference shows the clipped IVD surface"
            + (
                " and exact IVD-positive seed marks"
                if close_up["show_reference_points"] else ""
            )
            + "; close-up prediction panels show "
            + (
                "three orthogonal frozen-probability slices"
                if close_up["render_scalar_slices"] else
                "thresholded continuous probability isosurfaces"
                if close_up["render_prediction_surface"] else
                "only each model's predicted vortex-support points"
                if close_up["prediction_point_mode"]
                == "predicted_positive" else
                (
                    "the clipped IVD reference surface plus "
                    if close_up[
                        "show_reference_surface_in_predictions"
                    ] else ""
                )
                + (
                    "exact evaluated true-positive, "
                    if close_up["show_true_positives"] else ""
                )
                + "false-positive, and false-negative seed marks"
            )
            + (
                " with no seed-marker lattice"
                if close_up["surface_only"] else
                "; the true-negative background point array remains omitted"
            )
            + "; all seedwise predictions remain retained in metrics and the "
            "frozen artifact"
        ),
        "close_up_render_layers": {
            "surface_only": bool(close_up["surface_only"]),
            "prediction_point_mode": close_up["prediction_point_mode"],
            "predicted_positive_points": bool(
                close_up["prediction_point_mode"] == "predicted_positive"
            ),
            "scatter_collection_count": scatter_collection_count,
            "true_negative_points": False,
            "true_positive_points": bool(close_up["show_true_positives"]),
            "regular_seed_lattice": False,
            "ivd_reference_surface_in_predictions": bool(
                close_up["show_reference_surface_in_predictions"]
            ),
            "reference_context_color": close_up[
                "reference_context_color"
            ],
            "reference_surface_alpha": float(
                close_up["reference_surface_alpha"]
            ),
            "shade_surfaces": bool(close_up["shade_surfaces"]),
            "prediction_probability_isosurface": bool(
                close_up["render_prediction_surface"]
            ),
            "all_positive_surface_alpha": float(
                close_up["all_positive_surface_alpha"]
            ),
            "orthogonal_scalar_slices": bool(
                close_up["render_scalar_slices"]
            ),
            "scalar_slice_nearest_fill_cells": scalar_slice_fill_counts,
            "scalar_slice_interpolation": scalar_slice_interpolation,
            "prediction_surface_interpolation": (
                prediction_surface_interpolation
            ),
            "reference_surface_interpolation": (
                reference_surface_interpolation
            ),
            "error_markers": bool(close_up["show_error_markers"]),
            "paired_error_context": bool(
                close_up["show_paired_error_context"]
            ),
            "false_positive_color": (
                close_up["false_positive_color"]
                or paper.COLORS["false_positive"]
            ),
            "marker_scale": float(close_up["marker_scale"]),
            "fixed_canvas_export": bool(close_up["fixed_canvas_export"]),
            "clipped_surface_triangle_counts": close_up_surface_triangles,
        },
        "rendered_non_vortex_blue_pixel_count": background_blue_pixels,
        "rendered_blue_hue_pixel_count": blue_hue_pixels,
        "outputs": {
            "png": str(png_path), "pdf": str(pdf_path), "svg": str(svg_path),
            "tiff": str(tiff_path),
        },
    }
    if "sampling_audit" in payload:
        audit["sampling_audit"] = payload["sampling_audit"]
    if "checkpoint_sha256" in payload:
        audit["checkpoint_sha256"] = payload["checkpoint_sha256"]
    if "artifact_sha256" in payload:
        audit["prediction_artifact_sha256"] = payload["artifact_sha256"]
    if "prediction_artifact_sha256" in payload:
        audit["prediction_artifact_sha256"] = payload[
            "prediction_artifact_sha256"
        ]
    if "selection_sha256" in payload:
        audit["selection_sha256"] = payload["selection_sha256"]
    audit_path = prefix.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _write_manifest(output_dir):
    """Merge every completed per-flow audit into one stable index."""
    output_dir = Path(output_dir)
    audits_by_dataset = {}
    for audit_path in sorted(output_dir.glob("*_task3_diagnostic.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        dataset = audit.get("dataset")
        if dataset in FLOW_SPECS:
            audits_by_dataset[dataset] = audit
    audits = [
        audits_by_dataset[dataset]
        for dataset in FLOW_SPECS
        if dataset in audits_by_dataset
    ]

    rows = []
    for audit in audits:
        method_names = list(audit["metrics"])
        if len(method_names) != 2:
            raise RuntimeError(
                f"{audit['dataset']}: expected exactly two methods"
            )
        raw_name, fmt_name = method_names
        raw_metrics = audit["metrics"][raw_name]
        fmt_metrics = audit["metrics"][fmt_name]
        rows.append({
            "dataset": audit["dataset"],
            "title": audit["title"],
            "protocol": audit["protocol"],
            "source_start_index": audit["source_start_index"],
            "training_seed": audit["training_seed"],
            "sample_count": audit["sample_count"],
            "raw_method": raw_name,
            "fmt_method": fmt_name,
            "raw_f1": float(raw_metrics["f1"]),
            "fmt_f1": float(fmt_metrics["f1"]),
            "f1_gain": float(fmt_metrics["f1"] - raw_metrics["f1"]),
            "raw_average_precision": float(
                raw_metrics["average_precision"]
            ),
            "fmt_average_precision": float(
                fmt_metrics["average_precision"]
            ),
            "average_precision_gain": float(
                fmt_metrics["average_precision"]
                - raw_metrics["average_precision"]
            ),
        })

    summary_path = output_dir / "metrics_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    tie_tolerance = 1e-3
    f1_gains = np.asarray([row["f1_gain"] for row in rows])
    ap_gains = np.asarray([row["average_precision_gain"] for row in rows])
    manifest = {
        "experiment": "Visualize_Task3_Multiflow_Diagnostic_1.2",
        "figure_level_claim": (
            "On the fixed visualized confirmation slice, FMT residual "
            "improves average precision for all rendered three-dimensional "
            "flows; thresholded F1 still shows dataset-specific exceptions"
        ),
        "archetype": "image plate plus quantitative diagnostic",
        "backend": "Python/matplotlib",
        "selection_rule": (
            "fixed confirmation ordinal 4; no source slice was selected by "
            "visual appearance"
        ),
        "summary": {
            "dataset_count": len(rows),
            "f1_tie_tolerance": tie_tolerance,
            "f1_improved": int(np.count_nonzero(f1_gains > tie_tolerance)),
            "f1_decreased": int(np.count_nonzero(f1_gains < -tie_tolerance)),
            "f1_tied_within_tolerance": int(
                np.count_nonzero(np.abs(f1_gains) <= tie_tolerance)
            ),
            "average_precision_improved": int(np.count_nonzero(ap_gains > 0)),
        },
        "metrics_summary_csv": str(summary_path),
        "datasets": audits,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run(datasets, output_dir=DEFAULT_OUTPUT, dpi=300):
    unknown = sorted(set(datasets) - set(FLOW_SPECS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audits = []
    for dataset in datasets:
        payload = _load_payload(dataset, device)
        audit = _render_one(dataset, payload, output_dir, int(dpi))
        audits.append(audit)
        methods = list(audit["metrics"])
        print(
            f"{dataset}: {methods[0]} F1={audit['metrics'][methods[0]]['f1']:.3f}; "
            f"{methods[1]} F1={audit['metrics'][methods[1]]['f1']:.3f}",
            flush=True,
        )
    return _write_manifest(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=list(FLOW_SPECS),
        choices=list(FLOW_SPECS),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args()
    run(arguments.datasets, arguments.output_dir, arguments.dpi)
