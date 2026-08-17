"""Post-hoc search for the local-IVD parameters most consistent with clusters."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator

from FLowUtils.ScalarField3d import compute_local_ivd_3D, marching_cubes_world


def _interpolated_frame(vector_field, physical_time):
    float_time = (physical_time - vector_field.tmin) / vector_field.timeInterval
    t0 = int(np.clip(np.floor(float_time), 0, vector_field.time_steps - 1))
    t1 = int(np.clip(np.ceil(float_time), 0, vector_field.time_steps - 1))
    weight = float_time - t0 if t1 != t0 else 0.0
    return np.asarray(
        (1.0 - weight) * vector_field.field[t0] + weight * vector_field.field[t1],
        dtype=np.float32,
    )


def _best_binary_threshold(values, labels, cluster):
    """Exact best F1 over all masks values>=threshold for one cluster ID."""
    order = np.argsort(values, kind="stable")[::-1]
    sorted_values = values[order]
    truth = (labels[order] == cluster).astype(np.int64)
    tp = np.cumsum(truth)
    predicted = np.arange(1, len(values) + 1)
    positives = int(truth.sum())
    f1 = 2.0 * tp / np.maximum(predicted + positives, 1)
    # Only evaluate after the final sample sharing a threshold value.
    is_boundary = np.r_[sorted_values[:-1] != sorted_values[1:], True]
    candidates = np.flatnonzero(is_boundary)
    best_index = int(candidates[np.argmax(f1[candidates])])
    tp_best = int(tp[best_index]); predicted_best = best_index + 1
    fp = predicted_best - tp_best; fn = positives - tp_best
    return {
        "cluster_as_vortex": int(cluster),
        "threshold": float(sorted_values[best_index]),
        "f1": float(2 * tp_best / max(2 * tp_best + fp + fn, 1)),
        "precision": float(tp_best / max(tp_best + fp, 1)),
        "recall": float(tp_best / max(tp_best + fn, 1)),
        "iou": float(tp_best / max(tp_best + fp + fn, 1)),
        "positive_seed_count": int(predicted_best),
        "positive_seed_fraction": float(predicted_best / len(values)),
    }


def search_local_ivd_parameters_3d(
    vector_field, physical_time, seeds_xyz, labels, averaging_sizes, output_dir
):
    """Search local averaging size and isovalue; save table, arrays and best overlay."""
    output_dir = Path(output_dir)
    frame = _interpolated_frame(vector_field, physical_time)
    dx, dy, dz = (float(v) for v in vector_field.gridInterval)
    dmin = np.asarray(vector_field.domainMinBoundary, dtype=np.float64)
    dmax = np.asarray(vector_field.domainMaxBoundary, dtype=np.float64)
    xs = np.linspace(dmin[0], dmax[0], vector_field.Xdim)
    ys = np.linspace(dmin[1], dmax[1], vector_field.Ydim)
    zs = np.linspace(dmin[2], dmax[2], vector_field.Zdim)
    sample_zyx = np.asarray(seeds_xyz)[:, [2, 1, 0]]

    rows = []
    volumes = {}
    for raw_size in averaging_sizes:
        size = None if str(raw_size).lower() == "global" else int(raw_size)
        volume = compute_local_ivd_3D(frame, dx, dy, dz, size)
        key = "global" if size is None else str(size)
        volumes[key] = volume
        values = RegularGridInterpolator((zs, ys, xs), volume, bounds_error=True)(sample_zyx)
        for cluster in np.unique(labels):
            result = _best_binary_threshold(values, labels, int(cluster))
            result["averaging_size"] = key
            rows.append(result)

    unconstrained_best = max(rows, key=lambda row: row["f1"])
    cluster_counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    candidate_rows = [
        row for row in rows
        if cluster_counts[row["cluster_as_vortex"]] <= len(labels) / 2
        and row["positive_seed_fraction"] <= 0.5
    ]
    if not candidate_rows:
        raise RuntimeError("no non-majority local-IVD vortex candidate remained")
    best = max(candidate_rows, key=lambda row: row["f1"])
    summary = {
        "selection_rule": (
            "vortex_candidate_best requires both the selected cluster and the IVD-positive "
            "seed set to occupy at most 50%; unconstrained_best is retained for audit"
        ),
        "vortex_candidate_best": best,
        "unconstrained_best": unconstrained_best,
    }
    fieldnames = ["averaging_size", "cluster_as_vortex", "threshold", "f1", "iou",
                  "precision", "recall", "positive_seed_count", "positive_seed_fraction"]
    with (output_dir / "ivd_parameter_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "ivd_parameter_search_best.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    best_volume = volumes[best["averaging_size"]]
    np.savez_compressed(
        output_dir / "ivd_parameter_search_best.npz",
        ivd_volume=best_volume,
        threshold=np.float64(best["threshold"]),
        averaging_size=np.asarray(best["averaging_size"]),
        cluster_as_vortex=np.int64(best["cluster_as_vortex"]),
    )

    fig = plt.figure(figsize=(12, 6)); ax = fig.add_subplot(111, projection="3d")
    mesh = marching_cubes_world(best_volume, best["threshold"], (dx, dy, dz), dmin)
    if mesh is not None:
        vertices, _, faces = mesh
        ax.add_collection3d(Poly3DCollection(vertices[faces], alpha=0.28,
                                             facecolor="#f8961e", edgecolor="none"))
    mask = labels == best["cluster_as_vortex"]
    ax.scatter(seeds_xyz[mask, 0], seeds_xyz[mask, 1], seeds_xyz[mask, 2], s=8,
               color="#00b4d8", alpha=0.9, label=f"cluster {best['cluster_as_vortex']}")
    ax.set(xlabel="x", ylabel="y", zlabel="z",
           title=(f"Best post-hoc local IVD match: a={best['averaging_size']}, "
                  f"level={best['threshold']:.5g}, F1={best['f1']:.3f}"))
    ax.set_xlim(dmin[0], dmax[0]); ax.set_ylim(dmin[1], dmax[1]); ax.set_zlim(dmin[2], dmax[2])
    ax.set_box_aspect(np.maximum(dmax - dmin, 1e-12)); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "ivd_parameter_search_best.png", dpi=220)
    plt.close(fig)
    return summary, rows
