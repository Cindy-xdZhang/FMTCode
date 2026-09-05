"""Per-hairpin geometry clustering utilities for Task4-a parameter search."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from FMT_Utils.Task4A_StreamlineClustering_3D import (
    SPANWISE_CLASS,
    STREAMWISE_CLASS,
    topology_proxy_rows,
)


def per_vortex_kmeans(
    raw_features: np.ndarray,
    vortex_ids: np.ndarray,
    *,
    base_feature_width: int,
    neighbor_weight: float,
    scaling: str = "within_vortex",
    init: str = "k-means++",
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-4,
    algorithm: str = "lloyd",
    seed: int = 7068,
) -> tuple[np.ndarray, list[dict]]:
    """Fit independent two-cluster KMeans models inside every ``VortexId``.

    The function consumes geometry/FMT features only.  It returns arbitrary raw
    cluster IDs; semantic class names are attached by a separate mapping rule.
    """

    features = np.asarray(raw_features, dtype=np.float64)
    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    if features.ndim != 2 or len(features) != len(ids):
        raise ValueError("raw_features must be [N,D] and match vortex_ids")
    if not np.isfinite(features).all() or np.any(ids <= 0):
        raise ValueError("features must be finite and vortex IDs positive")
    base_feature_width = int(base_feature_width)
    if not 0 < base_feature_width <= features.shape[1]:
        raise ValueError("base_feature_width is outside the feature dimension")
    if float(neighbor_weight) < 0.0:
        raise ValueError("neighbor_weight must be non-negative")
    if scaling not in {"within_vortex", "global"}:
        raise ValueError("scaling must be 'within_vortex' or 'global'")

    global_scaler = StandardScaler().fit(features) if scaling == "global" else None
    raw_clusters = np.full(len(ids), -1, dtype=np.int8)
    rows: list[dict] = []
    for ordinal, vortex_id in enumerate(np.unique(ids)):
        member = np.flatnonzero(ids == vortex_id)
        if len(member) < 2:
            raise ValueError(f"VortexId {int(vortex_id)} has fewer than two cubes")
        scaler = global_scaler or StandardScaler().fit(features[member])
        transformed = scaler.transform(features[member])
        transformed[:, base_feature_width:] *= float(neighbor_weight)
        model = KMeans(
            n_clusters=2,
            init=str(init),
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
            algorithm=str(algorithm),
            random_state=int(seed) + ordinal,
        ).fit(transformed)
        labels = model.labels_.astype(np.int8, copy=False)
        if len(np.unique(labels)) != 2:
            raise RuntimeError(f"KMeans collapsed for VortexId {int(vortex_id)}")
        raw_clusters[member] = labels
        rows.append(
            {
                "vortex_id": int(vortex_id),
                "cube_count": int(len(member)),
                "raw_cluster_0_count": int(np.count_nonzero(labels == 0)),
                "raw_cluster_1_count": int(np.count_nonzero(labels == 1)),
                "inertia": float(model.inertia_),
                "iterations": int(model.n_iter_),
            }
        )
    return raw_clusters, rows


def _binary_macro_f1(proxy: np.ndarray, semantic: np.ndarray) -> float:
    scores = []
    for label in (STREAMWISE_CLASS, SPANWISE_CLASS):
        truth = proxy == label
        predicted = semantic == label
        true_positive = int(np.count_nonzero(truth & predicted))
        false_positive = int(np.count_nonzero(~truth & predicted))
        false_negative = int(np.count_nonzero(truth & ~predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(1.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def map_clusters_by_proxy_permutation(
    raw_clusters: np.ndarray,
    proxy_classes: np.ndarray,
    vortex_ids: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Best-match arbitrary cluster IDs to the proxy independently per instance.

    This is a standard external clustering evaluation and an explicit oracle:
    it must not be presented as a deployable semantic naming rule.
    """

    raw = np.asarray(raw_clusters, dtype=np.int8).reshape(-1)
    proxy = np.asarray(proxy_classes, dtype=np.int8).reshape(-1)
    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    if not (len(raw) == len(proxy) == len(ids)):
        raise ValueError("raw clusters, proxy classes, and vortex IDs must match")
    if np.any(~np.isin(raw, (0, 1))) or np.any(
        ~np.isin(proxy, (STREAMWISE_CLASS, SPANWISE_CLASS))
    ):
        raise ValueError("raw clusters and proxy classes must be binary and assigned")

    semantic = np.zeros(len(raw), dtype=np.int8)
    rows: list[dict] = []
    for vortex_id in np.unique(ids):
        member = ids == vortex_id
        candidate_0 = np.where(
            raw[member] == 0, STREAMWISE_CLASS, SPANWISE_CLASS
        ).astype(np.int8)
        candidate_1 = np.where(
            raw[member] == 1, STREAMWISE_CLASS, SPANWISE_CLASS
        ).astype(np.int8)
        score_0 = _binary_macro_f1(proxy[member], candidate_0)
        score_1 = _binary_macro_f1(proxy[member], candidate_1)
        streamwise_raw_cluster = 0 if score_0 >= score_1 else 1
        semantic[member] = np.where(
            raw[member] == streamwise_raw_cluster,
            STREAMWISE_CLASS,
            SPANWISE_CLASS,
        )
        rows.append(
            {
                "vortex_id": int(vortex_id),
                "streamwise_raw_cluster": int(streamwise_raw_cluster),
                "macro_f1_raw0_streamwise": score_0,
                "macro_f1_raw1_streamwise": score_1,
                "selected_macro_f1": max(score_0, score_1),
            }
        )
    return semantic, rows


def map_clusters_by_vorticity_axis(
    raw_clusters: np.ndarray,
    vortex_ids: np.ndarray,
    vorticity_xyz: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Name each instance's cluster with higher median |omega_x|/|omega| streamwise."""

    raw = np.asarray(raw_clusters, dtype=np.int8).reshape(-1)
    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    vorticity = np.asarray(vorticity_xyz, dtype=np.float64)
    if vorticity.shape != (len(raw), 3) or len(ids) != len(raw):
        raise ValueError("vorticity must be [N,3] and match clusters and vortex IDs")
    norm = np.linalg.norm(vorticity, axis=1)
    if not np.isfinite(vorticity).all() or np.any(norm <= 0.0):
        raise ValueError("vorticity vectors must be finite and nonzero")
    x_alignment = np.abs(vorticity[:, 0]) / norm

    semantic = np.zeros(len(raw), dtype=np.int8)
    rows: list[dict] = []
    for vortex_id in np.unique(ids):
        member = ids == vortex_id
        medians = [
            float(np.median(x_alignment[member & (raw == cluster)]))
            for cluster in (0, 1)
        ]
        streamwise_raw_cluster = int(np.argmax(medians))
        semantic[member] = np.where(
            raw[member] == streamwise_raw_cluster,
            STREAMWISE_CLASS,
            SPANWISE_CLASS,
        )
        rows.append(
            {
                "vortex_id": int(vortex_id),
                "streamwise_raw_cluster": streamwise_raw_cluster,
                "raw_cluster_0_median_abs_vorticity_x_alignment": medians[0],
                "raw_cluster_1_median_abs_vorticity_x_alignment": medians[1],
            }
        )
    return semantic, rows


def map_clusters_by_geometry_score(
    raw_clusters: np.ndarray,
    vortex_ids: np.ndarray,
    streamwise_geometry_score: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Name the per-instance cluster with higher median geometry score streamwise."""

    raw = np.asarray(raw_clusters, dtype=np.int8).reshape(-1)
    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    score = np.asarray(streamwise_geometry_score, dtype=np.float64).reshape(-1)
    if not (len(raw) == len(ids) == len(score)):
        raise ValueError("clusters, vortex IDs, and geometry score must match")
    if np.any(~np.isin(raw, (0, 1))) or np.any(ids <= 0) or not np.isfinite(score).all():
        raise ValueError("clusters/IDs/geometry scores must be assigned and finite")

    semantic = np.zeros(len(raw), dtype=np.int8)
    rows: list[dict] = []
    for vortex_id in np.unique(ids):
        member = ids == vortex_id
        medians = [
            float(np.median(score[member & (raw == cluster)]))
            for cluster in (0, 1)
        ]
        streamwise_raw_cluster = int(np.argmax(medians))
        semantic[member] = np.where(
            raw[member] == streamwise_raw_cluster,
            STREAMWISE_CLASS,
            SPANWISE_CLASS,
        )
        rows.append(
            {
                "vortex_id": int(vortex_id),
                "streamwise_raw_cluster": streamwise_raw_cluster,
                "raw_cluster_0_median_streamwise_geometry_score": medians[0],
                "raw_cluster_1_median_streamwise_geometry_score": medians[1],
            }
        )
    return semantic, rows


def map_clusters_by_hairpin_topology(
    raw_clusters: np.ndarray,
    vortex_ids: np.ndarray,
    indices_xyz: np.ndarray,
    *,
    dominant_min_voxels: int = 3,
    dominant_min_fraction: float = 0.05,
) -> tuple[np.ndarray, list[dict]]:
    """Attach semantic names using only the expected two-legs/one-head geometry.

    Both label permutations are scored by the frozen 2+1 soft topology proxy.
    If the scores tie, the cluster with the lower median z-index is named
    streamwise, reflecting legs closer to the lower wall in this half-channel.
    Neither velocity, curl, nor proxy class labels enter this mapping.
    """

    raw = np.asarray(raw_clusters, dtype=np.int8).reshape(-1)
    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    indices = np.asarray(indices_xyz, dtype=np.int32)
    if indices.shape != (len(raw), 3) or len(ids) != len(raw):
        raise ValueError("indices must be [N,3] and match clusters and vortex IDs")
    if np.any(~np.isin(raw, (0, 1))) or np.any(ids <= 0):
        raise ValueError("raw clusters must be binary and vortex IDs positive")

    semantic = np.zeros(len(raw), dtype=np.int8)
    rows: list[dict] = []
    for vortex_id in np.unique(ids):
        member_index = np.flatnonzero(ids == vortex_id)
        member_xyz = indices[member_index]
        lower = member_xyz.min(axis=0)
        local_xyz = member_xyz - lower
        size_xyz = local_xyz.max(axis=0) + 1
        shape_zyx = tuple(int(value) for value in size_xyz[::-1])
        iz, iy, ix = local_xyz[:, 2], local_xyz[:, 1], local_xyz[:, 0]
        instance_volume = np.zeros(shape_zyx, dtype=np.int32)
        instance_volume[iz, iy, ix] = int(vortex_id)

        candidate_rows = []
        for streamwise_raw_cluster in (0, 1):
            local_classes = np.where(
                raw[member_index] == streamwise_raw_cluster,
                STREAMWISE_CLASS,
                SPANWISE_CLASS,
            ).astype(np.int8)
            class_volume = np.zeros(shape_zyx, dtype=np.int8)
            class_volume[iz, iy, ix] = local_classes
            candidate_rows.append(
                topology_proxy_rows(
                    instance_volume,
                    class_volume,
                    dominant_min_voxels=int(dominant_min_voxels),
                    dominant_min_fraction=float(dominant_min_fraction),
                )[0]
            )

        scores = [float(row["soft_topology_score"]) for row in candidate_rows]
        if abs(scores[0] - scores[1]) <= 1e-12:
            median_z = [
                float(np.median(member_xyz[raw[member_index] == cluster, 2]))
                for cluster in (0, 1)
            ]
            streamwise_raw_cluster = int(np.argmin(median_z))
            tie_break = "lower_median_z_index"
        else:
            streamwise_raw_cluster = int(np.argmax(scores))
            median_z = [
                float(np.median(member_xyz[raw[member_index] == cluster, 2]))
                for cluster in (0, 1)
            ]
            tie_break = "not_needed"
        semantic[member_index] = np.where(
            raw[member_index] == streamwise_raw_cluster,
            STREAMWISE_CLASS,
            SPANWISE_CLASS,
        )
        rows.append(
            {
                "vortex_id": int(vortex_id),
                "streamwise_raw_cluster": streamwise_raw_cluster,
                "soft_score_raw0_streamwise": scores[0],
                "soft_score_raw1_streamwise": scores[1],
                "raw_cluster_0_median_z_index": median_z[0],
                "raw_cluster_1_median_z_index": median_z[1],
                "tie_break": tie_break,
            }
        )
    return semantic, rows


def fmt_base_feature_width(num_freq: int, mode: str, include_chirality: bool) -> int:
    """Return the center-line block width used by ``pathline_dft_features_3d``."""

    num_freq = int(num_freq)
    return num_freq * (1 if str(mode) == "magnitude" else 3) + (
        num_freq - 1 if bool(include_chirality) else 0
    )


def velocity_curl_from_time_parameterized_cross(
    primitives: np.ndarray,
    *,
    parameter_step: float,
    cross_offset: float,
    minimum_vector_norm: float = 1e-9,
) -> dict:
    """Reconstruct seed velocity and curl from seven-line cross geometry.

    Central temporal differences estimate the velocity on the center and six
    offset trajectories.  Central spatial differences across opposite offset
    pairs estimate the velocity gradient and hence curl.  The input must come
    from ``dx/dt=v(x)`` integration rather than unit-velocity sampling.
    """

    xyz = np.asarray(primitives, dtype=np.float64)
    if xyz.ndim != 4 or xyz.shape[1] != 7 or xyz.shape[2] < 3:
        raise ValueError("primitives must have shape [N,7,L>=3,3]")
    if xyz.shape[2] % 2 != 1 or xyz.shape[3] != 3:
        raise ValueError("primitive sample count must be odd and coordinates 3D")
    parameter_step = float(parameter_step)
    cross_offset = float(cross_offset)
    if parameter_step <= 0.0 or cross_offset <= 0.0:
        raise ValueError("parameter_step and cross_offset must be positive")

    center = xyz.shape[2] // 2
    line_velocity = (
        xyz[:, :, center + 1] - xyz[:, :, center - 1]
    ) / (2.0 * parameter_step)
    dv_dx = (line_velocity[:, 2] - line_velocity[:, 1]) / (2.0 * cross_offset)
    dv_dy = (line_velocity[:, 4] - line_velocity[:, 3]) / (2.0 * cross_offset)
    dv_dz = (line_velocity[:, 6] - line_velocity[:, 5]) / (2.0 * cross_offset)
    curl = np.column_stack(
        (
            dv_dy[:, 2] - dv_dz[:, 1],
            dv_dz[:, 0] - dv_dx[:, 2],
            dv_dx[:, 1] - dv_dy[:, 0],
        )
    )
    velocity = line_velocity[:, 0]
    velocity_norm = np.linalg.norm(velocity, axis=1)
    curl_norm = np.linalg.norm(curl, axis=1)
    valid = (
        np.isfinite(velocity).all(axis=1)
        & np.isfinite(curl).all(axis=1)
        & (velocity_norm >= float(minimum_vector_norm))
        & (curl_norm >= float(minimum_vector_norm))
    )
    absolute_cosine = np.full(len(xyz), np.nan, dtype=np.float64)
    absolute_cosine[valid] = np.clip(
        np.abs(np.sum(velocity[valid] * curl[valid], axis=1))
        / (velocity_norm[valid] * curl_norm[valid]),
        0.0,
        1.0,
    )
    angle_degrees = np.full(len(xyz), np.nan, dtype=np.float64)
    angle_degrees[valid] = np.degrees(np.arccos(absolute_cosine[valid]))
    return {
        "velocity_xyz": velocity.astype(np.float32),
        "curl_xyz": curl.astype(np.float32),
        "absolute_cosine": absolute_cosine.astype(np.float32),
        "angle_degrees": angle_degrees.astype(np.float32),
        "valid": valid,
    }
