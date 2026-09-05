"""Traditional velocity--curl orientation baseline for Task4-a."""

from __future__ import annotations

import numpy as np

from FMT_Utils.Task4A_StreamlineClustering_3D import (
    SPANWISE_CLASS,
    STREAMWISE_CLASS,
)


def velocity_curl_orientation_classes(
    velocity_xyz: np.ndarray,
    curl_xyz: np.ndarray,
    *,
    streamwise_max_angle_degrees: float = 45.0,
    minimum_vector_norm: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify curl as closer to parallel or perpendicular to local velocity.

    Curl is an axial vector, so its sign does not change the vortex-axis class.
    The absolute cosine therefore maps the velocity--curl angle to [0, 90] deg.
    Angles at or below the frozen threshold are streamwise; larger angles are
    the perpendicular-to-flow class named spanwise for this two-class baseline.
    """

    velocity = np.asarray(velocity_xyz, dtype=np.float64)
    curl = np.asarray(curl_xyz, dtype=np.float64)
    if velocity.shape != curl.shape or velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("velocity and curl must be same-shape [N,3] arrays")
    threshold = float(streamwise_max_angle_degrees)
    minimum_vector_norm = float(minimum_vector_norm)
    if not 0.0 < threshold < 90.0:
        raise ValueError("streamwise angle threshold must lie strictly in (0, 90)")
    if minimum_vector_norm <= 0:
        raise ValueError("minimum_vector_norm must be positive")

    velocity_norm = np.linalg.norm(velocity, axis=1)
    curl_norm = np.linalg.norm(curl, axis=1)
    valid = (
        np.isfinite(velocity).all(axis=1)
        & np.isfinite(curl).all(axis=1)
        & (velocity_norm >= minimum_vector_norm)
        & (curl_norm >= minimum_vector_norm)
    )
    absolute_cosine = np.full(len(velocity), np.nan, dtype=np.float64)
    absolute_cosine[valid] = np.clip(
        np.abs(np.sum(velocity[valid] * curl[valid], axis=1))
        / (velocity_norm[valid] * curl_norm[valid]),
        0.0,
        1.0,
    )
    angle_degrees = np.full(len(velocity), np.nan, dtype=np.float64)
    angle_degrees[valid] = np.degrees(np.arccos(absolute_cosine[valid]))
    classes = np.zeros(len(velocity), dtype=np.int8)
    classes[valid & (angle_degrees <= threshold)] = STREAMWISE_CLASS
    classes[valid & (angle_degrees > threshold)] = SPANWISE_CLASS
    return classes, angle_degrees.astype(np.float32), valid


def paired_topology_comparison(fmt_rows: list[dict], baseline_rows: list[dict]) -> tuple[list[dict], dict]:
    """Join deterministic per-instance topology metrics without calling them truth."""

    fmt_by_id = {int(row["vortex_id"]): row for row in fmt_rows}
    baseline_by_id = {int(row["vortex_id"]): row for row in baseline_rows}
    if set(fmt_by_id) != set(baseline_by_id):
        raise ValueError("FMT and baseline topology rows must contain identical VortexIds")
    paired = []
    for vortex_id in sorted(fmt_by_id):
        fmt = fmt_by_id[vortex_id]
        baseline = baseline_by_id[vortex_id]
        delta = float(baseline["soft_topology_score"] - fmt["soft_topology_score"])
        paired.append(
            {
                "vortex_id": vortex_id,
                "voxel_count": int(fmt["voxel_count"]),
                "input_single_component": bool(fmt["input_single_component"]),
                "fmt_streamwise_dominant_components": int(
                    fmt["streamwise_dominant_component_count"]
                ),
                "fmt_spanwise_dominant_components": int(
                    fmt["spanwise_dominant_component_count"]
                ),
                "fmt_hard_2plus1_success": bool(fmt["hard_2plus1_success"]),
                "fmt_soft_topology_score": float(fmt["soft_topology_score"]),
                "baseline_streamwise_dominant_components": int(
                    baseline["streamwise_dominant_component_count"]
                ),
                "baseline_spanwise_dominant_components": int(
                    baseline["spanwise_dominant_component_count"]
                ),
                "baseline_hard_2plus1_success": bool(
                    baseline["hard_2plus1_success"]
                ),
                "baseline_soft_topology_score": float(
                    baseline["soft_topology_score"]
                ),
                "baseline_minus_fmt_soft_score": delta,
            }
        )
    deltas = np.asarray([row["baseline_minus_fmt_soft_score"] for row in paired])
    tolerance = 1e-12
    summary = {
        "vortex_id_count": len(paired),
        "baseline_soft_score_better_count": int(np.count_nonzero(deltas > tolerance)),
        "equal_soft_score_count": int(np.count_nonzero(np.abs(deltas) <= tolerance)),
        "baseline_soft_score_worse_count": int(np.count_nonzero(deltas < -tolerance)),
        "mean_paired_soft_score_delta": float(np.mean(deltas)),
        "median_paired_soft_score_delta": float(np.median(deltas)),
        "fmt_hard_success_count": int(
            sum(row["fmt_hard_2plus1_success"] for row in paired)
        ),
        "baseline_hard_success_count": int(
            sum(row["baseline_hard_2plus1_success"] for row in paired)
        ),
    }
    return paired, summary


def proxy_groundtruth_agreement(
    fmt_classes: np.ndarray,
    proxy_classes: np.ndarray,
    vortex_ids: np.ndarray,
    angle_degrees: np.ndarray,
) -> tuple[list[dict], dict]:
    """Measure fixed-semantic FMT classes against the velocity--curl proxy.

    Cubes are nested within manually annotated ``VortexId`` instances.  The
    primary summaries therefore compute each metric inside one instance and
    then give all instances equal weight.  Class 1 is streamwise and class 2
    is spanwise/perpendicular-to-flow.  If a class is absent from both the
    proxy and FMT within one instance, its F1 and intersection-over-union are
    defined as one because the two masks agree exactly.

    The angle-confidence-weighted mean squared error uses binary class values
    (streamwise=1, spanwise=0) and weights each cube by
    ``abs(cos(2 * theta))``.  The weight is zero at the uncertain 45-degree
    boundary and one at exactly parallel or perpendicular orientations.
    """

    fmt = np.asarray(fmt_classes)
    proxy = np.asarray(proxy_classes)
    ids = np.asarray(vortex_ids)
    angles = np.asarray(angle_degrees, dtype=np.float64)
    if not (fmt.ndim == proxy.ndim == ids.ndim == angles.ndim == 1):
        raise ValueError("classes, VortexIds, and angles must be one-dimensional")
    if not (len(fmt) == len(proxy) == len(ids) == len(angles)):
        raise ValueError("classes, VortexIds, and angles must have equal length")

    valid = (
        np.isin(fmt, (STREAMWISE_CLASS, SPANWISE_CLASS))
        & np.isin(proxy, (STREAMWISE_CLASS, SPANWISE_CLASS))
        & (ids > 0)
        & np.isfinite(angles)
    )
    if not np.any(valid):
        raise ValueError("no valid cubes are available for proxy evaluation")

    fmt = fmt[valid]
    proxy = proxy[valid]
    ids = ids[valid]
    angles = angles[valid]

    def class_scores(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> tuple[float, float]:
        true_mask = y_true == label
        pred_mask = y_pred == label
        true_positive = int(np.count_nonzero(true_mask & pred_mask))
        false_positive = int(np.count_nonzero(~true_mask & pred_mask))
        false_negative = int(np.count_nonzero(true_mask & ~pred_mask))
        f1_denominator = 2 * true_positive + false_positive + false_negative
        union = true_positive + false_positive + false_negative
        f1 = 1.0 if f1_denominator == 0 else 2.0 * true_positive / f1_denominator
        iou = 1.0 if union == 0 else true_positive / union
        return float(f1), float(iou)

    def metric_row(y_true: np.ndarray, y_pred: np.ndarray, theta: np.ndarray) -> dict:
        stream_f1, stream_iou = class_scores(y_true, y_pred, STREAMWISE_CLASS)
        span_f1, span_iou = class_scores(y_true, y_pred, SPANWISE_CLASS)
        true_binary = (y_true == STREAMWISE_CLASS).astype(np.float64)
        pred_binary = (y_pred == STREAMWISE_CLASS).astype(np.float64)
        squared_error = np.square(pred_binary - true_binary)
        confidence = np.abs(np.cos(2.0 * np.radians(theta)))
        confidence_sum = float(np.sum(confidence))
        weighted_mse = (
            float(np.sum(confidence * squared_error) / confidence_sum)
            if confidence_sum > 0.0
            else float("nan")
        )
        return {
            "f1_streamwise": stream_f1,
            "f1_spanwise": span_f1,
            "macro_f1": float(0.5 * (stream_f1 + span_f1)),
            "iou_streamwise": stream_iou,
            "iou_spanwise": span_iou,
            "mean_iou": float(0.5 * (stream_iou + span_iou)),
            "hard_label_mse": float(np.mean(squared_error)),
            "angle_confidence_weighted_mse": weighted_mse,
            "mean_angle_confidence": float(np.mean(confidence)),
            "angle_confidence_sum": confidence_sum,
        }

    rows: list[dict] = []
    for vortex_id in sorted(int(value) for value in np.unique(ids)):
        select = ids == vortex_id
        scores = metric_row(proxy[select], fmt[select], angles[select])
        rows.append(
            {
                "vortex_id": vortex_id,
                "cube_count": int(np.count_nonzero(select)),
                "proxy_streamwise_count": int(
                    np.count_nonzero(proxy[select] == STREAMWISE_CLASS)
                ),
                "proxy_spanwise_count": int(
                    np.count_nonzero(proxy[select] == SPANWISE_CLASS)
                ),
                "fmt_streamwise_count": int(
                    np.count_nonzero(fmt[select] == STREAMWISE_CLASS)
                ),
                "fmt_spanwise_count": int(
                    np.count_nonzero(fmt[select] == SPANWISE_CLASS)
                ),
                **scores,
            }
        )

    def equal_weighted_summary(name: str) -> dict:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return {"mean": None, "median": None, "q25": None, "q75": None}
        return {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "q25": float(np.percentile(finite, 25)),
            "q75": float(np.percentile(finite, 75)),
        }

    voxel_scores = metric_row(proxy, fmt, angles)
    proxy_stream = proxy == STREAMWISE_CLASS
    fmt_stream = fmt == STREAMWISE_CLASS
    confusion = {
        "proxy_streamwise_fmt_streamwise": int(np.count_nonzero(proxy_stream & fmt_stream)),
        "proxy_streamwise_fmt_spanwise": int(np.count_nonzero(proxy_stream & ~fmt_stream)),
        "proxy_spanwise_fmt_streamwise": int(np.count_nonzero(~proxy_stream & fmt_stream)),
        "proxy_spanwise_fmt_spanwise": int(np.count_nonzero(~proxy_stream & ~fmt_stream)),
    }
    summary = {
        "proxy_status": "deterministic_proxy_not_manual_type_ground_truth",
        "semantic_mapping": (
            "FMT class names are the frozen vorticity-axis post-hoc mapping from "
            "Other_Task4A_FMTStreamlineClustering_1.1; no label permutation is "
            "chosen against this velocity-curl proxy."
        ),
        "independent_evaluation_unit": "VortexId",
        "vortex_id_count": len(rows),
        "valid_cube_count": int(len(fmt)),
        "primary_vortex_equal_weighted_macro_f1": equal_weighted_summary("macro_f1"),
        "vortex_equal_weighted_class_f1": {
            "streamwise": equal_weighted_summary("f1_streamwise"),
            "spanwise_perpendicular": equal_weighted_summary("f1_spanwise"),
        },
        "vortex_equal_weighted_mean_iou": equal_weighted_summary("mean_iou"),
        "vortex_equal_weighted_hard_label_mse": equal_weighted_summary(
            "hard_label_mse"
        ),
        "secondary_vortex_equal_weighted_angle_confidence_weighted_mse": (
            equal_weighted_summary("angle_confidence_weighted_mse")
        ),
        "voxel_weighted_descriptive_metrics": voxel_scores,
        "confusion_matrix_proxy_rows_fmt_columns": confusion,
        "metric_direction": {
            "f1_and_iou": "higher_is_better_best_1",
            "mse": "lower_is_better_best_0",
        },
    }
    return rows, summary
