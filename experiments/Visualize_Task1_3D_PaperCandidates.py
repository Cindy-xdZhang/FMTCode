"""Render dense three-figure Task1 3D paper candidates.

For every scene this script writes exactly three 3D views:

1. a high-IVD isosurface with dense, time-coloured centre pathlines;
2. both FMT+KMeans clusters, with vortex red and non-vortex blue;
3. the prediction compared with the same high-IVD reference.

The KMeans model and anonymous-cluster calibration are reconstructed from the
frozen Task1 A100 development results.  The displayed confirmation slice is
ordinal 2 for every flow and is not selected using confirmation performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, to_rgb
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import netCDF4 as nc
import numpy as np
import torch

from FLowUtils.ScalarField3d import marching_cubes_world
from FLowUtils.flowlineIntegral import compute_pathlines_3D_batch
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_3D_pipeline import (
    compute_ivd_reference_3d,
    generate_seeding_grid_3d,
    integrate_cross_primitives_3d,
)
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from FMT_Utils.Task12Data_3D import (
    feature_matrix,
    load_cache_records,
    stack_features,
    stack_reference,
)
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
    fit_kmeans_transform,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "Task1_3D_paper_candidates_1.9"
KMEANS_SEED = 7068
KMEANS_N_INIT = 20
CONFIRMATION_ORDINAL = 2
ORIGINAL_SEED_COUNT = 16 ** 3
DEFAULT_DENSE_GRID_SIZE = 21
DEFAULT_EVALUATION_IVD_PERCENTILE = 95.0
DEFAULT_DISPLAY_IVD_PERCENTILE = 97.0
DEFAULT_PATHLINE_COUNT = 240
DEFAULT_DISPLAY_INTEGRATION_STEPS = 96
DEFAULT_VORTEX_PATHLINE_FRACTION = 0.70
DISPLAY_BLUE_NOISE_SEED = 7068
BOEING_STL_PATH = Path(
    r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets"
    r"\FluidX3DSTLAssets\stl\Boeing747\files\techtris_airplane.stl"
)

COLORS = {
    "ivd": "#f4a261",
    "pathline": "#263238",
    "vortex": "#d62728",
    "non_vortex": "#2468b4",
    "false_positive": "#9c27b0",
    "false_negative": "#f4a261",
    "geometry": "#59636f",
}

FLOW_SPECS = (
    {
        "dataset": "cylinder3d",
        "title": "Half-cylinder Re160",
        "development": ROOT / "outputs/Verify_Task2Universality_1.1/cache",
        "confirmation": ROOT / "outputs/mainExp_Task3Universality_2.2/confirmation_cache",
        "selected": ROOT / "outputs/mainExp_Task1_3D_2.1_ibex_a100/selected_configs.json",
        "family": "halfcylinder",
        "view": (22, -62),
    },
    {
        "dataset": "halfcylinderRe640",
        "title": "Half-cylinder Re640",
        "development": ROOT / "outputs/Verify_Task2Universality_1.1/cache",
        "confirmation": ROOT / "outputs/mainExp_Task3Universality_2.2/confirmation_cache",
        "selected": ROOT / "outputs/mainExp_Task1_3D_2.1_ibex_a100/selected_configs.json",
        "family": "halfcylinder",
        "view": (22, -62),
    },
    {
        "dataset": "tangaroa",
        "title": "Tangaroa",
        "development": ROOT / "outputs/Verify_Task2Universality_1.1/cache",
        "confirmation": ROOT / "outputs/mainExp_Task3Universality_2.2/confirmation_cache",
        "selected": ROOT / "outputs/mainExp_Task1_3D_2.1_ibex_a100/selected_configs.json",
        "family": "tangaroa",
        "view": (23, -62),
    },
    {
        "dataset": "deltaWing_LBM",
        "title": "Delta-wing original LBM",
        "development": ROOT / "outputs/Verify_Task2Universality_1.1/cache",
        "confirmation": ROOT / "outputs/mainExp_Task3Universality_2.2/confirmation_cache",
        "selected": ROOT / "outputs/mainExp_Task1_3D_2.1_ibex_a100/selected_configs.json",
        "family": "deltaWing",
        "view": (22, -58),
        "geometry": "delta_wing_triangle",
        "pathline_count_multiplier": 2,
        "integration_steps_multiplier": 2,
    },
    {
        "dataset": "boeing747",
        "title": "Boeing 747",
        "development": ROOT / "outputs/mainExp_Task123NewFlows_1.1/development_cache",
        "confirmation": ROOT / "outputs/mainExp_Task123NewFlows_1.1/confirmation_cache",
        "selected": ROOT / "outputs/mainExp_Task1_3D_2.2_newflows_ibex_a100/selected_configs.json",
        "family": "boeing747",
        "view": (21, -58),
        "geometry": "boeing747_stl",
        "pathline_count_multiplier": 2,
        "integration_steps_multiplier": 2,
    },
    {
        "dataset": "smokeBuoyancy",
        "title": "Smoke buoyancy",
        "development": ROOT / "outputs/mainExp_Task123NewFlows_1.1/development_cache",
        "confirmation": ROOT / "outputs/mainExp_Task123NewFlows_1.1/confirmation_cache",
        "selected": ROOT / "outputs/mainExp_Task1_3D_2.2_newflows_ibex_a100/selected_configs.json",
        "family": "smokeBuoyancy",
        "view": (22, -58),
    },
)

FIGURE_SIZE = (9.2, 7.2)
CAMERA_AXES_RECT = (0.06, 0.06, 0.76, 0.74)
COLORBAR_AXES_RECT = (0.86, 0.18, 0.025, 0.54)


def _new_camera_figure():
    """Create the identical canvas and 3D viewport used by all three figures."""
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_axes(CAMERA_AXES_RECT, projection="3d")
    return fig, ax


def _set_physical_axes(ax, bounds, view):
    bounds = np.asarray(bounds, dtype=np.float64)
    lower = bounds[0]
    upper = bounds[1]
    span = np.maximum(upper - lower, 1e-12)
    padding = 0.018 * span
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.set_zlim(lower[2] - padding[2], upper[2] + padding[2])
    ax.set_box_aspect(span)
    ax.set_proj_type("ortho")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=8, pad=0)
    ax.grid(False)


def _add_ivd_surface(ax, mesh, alpha):
    if mesh is None:
        return
    vertices, _, faces = mesh
    ax.add_collection3d(
        Poly3DCollection(
            vertices[faces],
            facecolor=COLORS["ivd"],
            edgecolor="none",
            alpha=float(alpha),
            zorder=1,
        )
    )


def _source_coordinate_axes(path):
    """Read the full-resolution physical coordinate arrays from a NetCDF file."""
    with nc.Dataset(path) as dataset:
        axes = {}
        for axis in "xyz":
            if axis not in dataset.variables:
                raise ValueError(f"{path}: missing coordinate variable {axis!r}")
            values = np.ma.asarray(dataset.variables[axis][:])
            if np.ma.isMaskedArray(values):
                if values.count() != values.size:
                    raise ValueError(f"{path}: masked values in {axis!r} coordinates")
                values = values.filled(np.nan)
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
                raise ValueError(f"{path}: invalid {axis!r} coordinate array")
            axes[axis] = values
    return axes


def _lattice_to_physical(points, axes):
    """Map FluidX3D lattice coordinates into the exported NetCDF coordinates."""
    points = np.asarray(points, dtype=np.float64)
    physical = np.empty_like(points)
    for component, axis in enumerate("xyz"):
        values = axes[axis]
        physical[..., component] = (
            values[0]
            + points[..., component] / float(len(values) - 1)
            * (values[-1] - values[0])
        )
    return physical


def _read_binary_stl_triangles(path):
    """Read the binary STL format accepted by FluidX3D's read_stl()."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path}: truncated STL")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    expected_size = 84 + 50 * triangle_count
    if triangle_count < 1 or len(raw) != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} bytes for {triangle_count} "
            f"triangles, found {len(raw)}"
        )
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
         ("attribute", "<u2")]
    )
    records = np.frombuffer(raw, dtype=dtype, count=triangle_count, offset=84)
    return np.asarray(records["vertices"], dtype=np.float64)


def _vertex_cluster_triangles(triangles, bins=100):
    """Reduce a dense STL deterministically while preserving its full silhouette."""
    triangles = np.asarray(triangles, dtype=np.float64)
    flat = triangles.reshape(-1, 3)
    lower = flat.min(axis=0)
    span = np.maximum(flat.max(axis=0) - lower, 1e-12)
    keys = np.rint((flat - lower) / span * int(bins)).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    vertex_count = int(inverse.max()) + 1
    clustered_vertices = np.zeros((vertex_count, 3), dtype=np.float64)
    np.add.at(clustered_vertices, inverse, flat)
    counts = np.bincount(inverse, minlength=vertex_count)
    clustered_vertices /= counts[:, None]
    faces = inverse.reshape(-1, 3)
    nondegenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 0] != faces[:, 2])
        & (faces[:, 1] != faces[:, 2])
    )
    faces = faces[nondegenerate]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(first)]
    return clustered_vertices[faces]


def _cpp_integer_ratio(numerator, denominator):
    """Match C++ signed-integer division, which truncates toward zero."""
    return int(float(numerator) / float(denominator))


def _load_simulation_geometry(spec, source_path):
    kind = spec.get("geometry")
    if kind is None:
        return None
    axes = _source_coordinate_axes(source_path)

    if kind == "delta_wing_triangle":
        length = 110
        if len(axes["x"]) != length or len(axes["z"]) != length:
            raise ValueError(
                "Delta-wing geometry expects the original 110 x 628 x 110 "
                "FluidX3D simulation"
            )
        center_xz = 0.5 * length - 0.5
        lattice_vertices = np.array(
            [
                [center_xz, _cpp_integer_ratio(5 * length, 64),
                 center_xz + _cpp_integer_ratio(20 * length, 64)],
                [center_xz + _cpp_integer_ratio(-20 * length, 64),
                 _cpp_integer_ratio(90 * length, 64),
                 center_xz + _cpp_integer_ratio(-10 * length, 64)],
                [center_xz + _cpp_integer_ratio(20 * length, 64),
                 _cpp_integer_ratio(90 * length, 64),
                 center_xz + _cpp_integer_ratio(-10 * length, 64)],
            ],
            dtype=np.float64,
        )
        physical_vertices = _lattice_to_physical(lattice_vertices, axes)
        return {
            "triangles": physical_vertices[None, ...],
            "label": "simulation geometry (delta wing)",
            "alpha": 0.90,
            "edge_width": 0.65,
            "metadata": {
                "kind": kind,
                "source_definition": "FluidX3D p0/p1/p2 triangle",
                "simulation_box_Nxyz": [110, 628, 110],
                "lattice_vertices": lattice_vertices.tolist(),
                "physical_vertices": physical_vertices.tolist(),
                "cpp_integer_division": "truncation toward zero",
            },
        }

    if kind == "boeing747_stl":
        nx, ny, nz = (len(axes[axis]) for axis in "xyz")
        raw_triangles = _read_binary_stl_triangles(BOEING_STL_PATH)
        angle = np.deg2rad(-15.0)
        rotation = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, np.cos(angle), -np.sin(angle)],
             [0.0, np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
        rotated = raw_triangles @ rotation.T
        flat = rotated.reshape(-1, 3)
        rotated_min = flat.min(axis=0)
        rotated_max = flat.max(axis=0)
        scale = float(nx) / float((rotated_max - rotated_min).max())
        center = np.array(
            [0.5 * nx - 0.5, 0.55 * nx, 0.5 * nz - 0.5],
            dtype=np.float64,
        )
        lattice_triangles = center + scale * (
            rotated - 0.5 * (rotated_min + rotated_max)
        )
        physical_triangles = _lattice_to_physical(lattice_triangles, axes)
        display_triangles = _vertex_cluster_triangles(physical_triangles, bins=100)
        physical_bounds = np.array(
            [physical_triangles.reshape(-1, 3).min(axis=0),
             physical_triangles.reshape(-1, 3).max(axis=0)]
        )
        return {
            "triangles": display_triangles,
            "label": "simulation geometry (Boeing 747)",
            "alpha": 0.94,
            "edge_width": 0.0,
            "metadata": {
                "kind": kind,
                "stl_path": str(BOEING_STL_PATH),
                "source_triangle_count": int(len(raw_triangles)),
                "display_triangle_count": int(len(display_triangles)),
                "simulation_box_Nxyz": [int(nx), int(ny), int(nz)],
                "center_lattice": center.tolist(),
                "rotation_axis": [1.0, 0.0, 0.0],
                "rotation_degrees": -15.0,
                "longest_side_target_lattice": float(nx),
                "scale": scale,
                "physical_bounds": physical_bounds.tolist(),
                "transform_source": "FluidX3D read_stl_raw/voxelize_stl",
            },
        }
    raise ValueError(f"unknown simulation geometry kind: {kind}")


def _add_simulation_geometry(ax, geometry):
    if geometry is None:
        return
    triangles = np.asarray(geometry["triangles"], dtype=np.float64)
    first_edge = triangles[:, 1] - triangles[:, 0]
    second_edge = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(first_edge, second_edge)
    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(normal_length, 1e-12)
    light = np.array([0.35, -0.45, 0.82], dtype=np.float64)
    light /= np.linalg.norm(light)
    illumination = 0.35 + 0.65 * np.abs(normals @ light)
    base_color = np.asarray(to_rgb(COLORS["geometry"]), dtype=np.float64)
    facecolors = np.empty((len(triangles), 4), dtype=np.float64)
    facecolors[:, :3] = np.clip(
        base_color[None, :] * (0.72 + 0.48 * illumination[:, None]),
        0.0,
        1.0,
    )
    facecolors[:, 3] = float(geometry["alpha"])
    collection = Poly3DCollection(
        triangles,
        facecolor=facecolors,
        edgecolor=COLORS["geometry"] if geometry["edge_width"] else "none",
        linewidth=float(geometry["edge_width"]),
        zorder=2,
    )
    collection.set_rasterized(True)
    ax.add_collection3d(collection)


def _load_choice(spec):
    selected = json.loads(Path(spec["selected"]).read_text(encoding="utf-8"))
    return selected[spec["family"]]["fmt"]


def _fit_frozen_clusterer(spec, device):
    dataset = spec["dataset"]
    development = load_cache_records(Path(spec["development"]) / dataset, 10)
    choice = _load_choice(spec)
    feature = choice["feature"]
    fitted = fit_kmeans_transform(
        stack_features(development[:8], feature, device),
        choice["pca_dim"],
        KMEANS_SEED,
        KMEANS_N_INIT,
    )
    calibration_labels = fitted.predict(
        stack_features(development[8:10], feature, device)
    )
    vortex_cluster = calibrate_vortex_cluster(
        stack_reference(development[8:10]), calibration_labels
    )
    return fitted, int(vortex_cluster), choice


def _raw_local_features(primitives):
    xyz = np.asarray(primitives[..., :3], dtype=np.float32)
    return (xyz - xyz[:, :1, :1, :]).reshape(len(xyz), -1)


def _selected_features(primitives, choice, device):
    tensor = torch.from_numpy(np.asarray(primitives)).to(device)
    raw = _raw_local_features(primitives)
    fmt = pathline_dft_features_3d(
        tensor,
        num_freq=6,
        # These values must match experiments/Build_Task2_Universality_Cache.py.  The
        # frozen KMeans scaler was fitted to that exact feature convention.
        neighbor_weight=1.0,
        neighbor_scale=1.0,
        neighbor_pool="sort",
        mode="gram",
        include_chirality=True,
    ).astype(np.float32)
    record = {"raw": raw, "fmt": fmt, "features": {}}
    return feature_matrix(record, choice["feature"], device)


def _blue_noise_like_indices(seeds, count, random_seed=DISPLAY_BLUE_NOISE_SEED):
    """Select deterministic maximin seeds, a discrete blue-noise approximation."""
    points = np.asarray(seeds, dtype=np.float64)
    count = min(int(count), len(seeds))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(random_seed))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(len(points)))
    minimum_distance_squared = np.sum(
        (points - points[selected[0]]) ** 2, axis=1
    )
    minimum_distance_squared[selected[0]] = -np.inf
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum_distance_squared))
        distance_squared = np.sum(
            (points - points[selected[index]]) ** 2, axis=1
        )
        minimum_distance_squared = np.minimum(
            minimum_distance_squared, distance_squared
        )
        minimum_distance_squared[selected[:index + 1]] = -np.inf
    return selected


def _stratified_blue_noise_indices(seeds, vortex_mask, count, vortex_fraction):
    """Blue-noise sample separately inside and outside the IVD core proxy."""
    count = min(int(count), len(seeds))
    vortex_candidates = np.flatnonzero(np.asarray(vortex_mask, dtype=bool))
    background_candidates = np.flatnonzero(~np.asarray(vortex_mask, dtype=bool))
    requested_vortex = int(round(float(vortex_fraction) * count))
    vortex_count = min(requested_vortex, len(vortex_candidates))
    background_count = min(count - vortex_count, len(background_candidates))
    remaining = count - vortex_count - background_count
    if remaining:
        extra_vortex = min(remaining, len(vortex_candidates) - vortex_count)
        vortex_count += extra_vortex
        remaining -= extra_vortex
    if remaining:
        background_count += min(
            remaining, len(background_candidates) - background_count
        )

    vortex_local = _blue_noise_like_indices(
        np.asarray(seeds)[vortex_candidates],
        vortex_count,
        DISPLAY_BLUE_NOISE_SEED,
    )
    background_local = _blue_noise_like_indices(
        np.asarray(seeds)[background_candidates],
        background_count,
        DISPLAY_BLUE_NOISE_SEED + 1,
    )
    selected = np.concatenate(
        (vortex_candidates[vortex_local], background_candidates[background_local])
    )
    return selected, int(vortex_count), int(background_count)


def _integrate_display_pathlines(metadata, seeds, seed_ivd, vortex_threshold,
                                 pathline_count, display_integration_steps,
                                 vortex_fraction):
    """Integrate IVD-stratified display paths without changing FMT primitives."""
    count = min(int(pathline_count), len(seeds))
    selected_indices, vortex_count, background_count = (
        _stratified_blue_noise_indices(
            seeds,
            np.asarray(seed_ivd) >= float(vortex_threshold),
            count,
            vortex_fraction,
        )
    )
    selected_seeds = np.asarray(seeds[selected_indices], dtype=np.float64)

    # dt is one quarter of a source-frame interval.  Load enough future frames
    # for the longer display integration while leaving the frozen 48-step FMT
    # primitive untouched.
    dt_scale = 0.25
    frame_count = int(np.ceil(dt_scale * int(display_integration_steps))) + 2
    display_field, display_load_metadata = load_netcdf_window_3d(
        metadata["source_path"],
        int(metadata["source_start_index"]),
        frame_count,
        96,
    )
    dt = float(display_field.timeInterval) * dt_scale
    target_time = dt * int(display_integration_steps)
    seeds_xyzt = np.column_stack(
        (selected_seeds, np.zeros(len(selected_seeds), dtype=np.float64))
    )
    result = compute_pathlines_3D_batch(
        display_field,
        seeds_xyzt,
        min_time=0.0,
        max_time=target_time,
        step_size=dt,
        max_iteration=int(display_integration_steps),
        method="RK4",
    )
    if result is None:
        raise RuntimeError("3D RK4 display pathline integration is unavailable")
    positions, raw_lengths = result
    lower = np.asarray(display_field.domainMinBoundary, dtype=np.float64)
    upper = np.asarray(display_field.domainMaxBoundary, dtype=np.float64)
    clean_paths = []
    clean_lengths = []
    for position, raw_length in zip(positions, raw_lengths):
        path = np.asarray(position[:int(raw_length)], dtype=np.float32)
        spatial_ok = ((path[:, :3] >= lower) & (path[:, :3] <= upper)).all(axis=1)
        time_ok = path[:, 3] <= target_time + 1e-6
        invalid = np.flatnonzero(~(spatial_ok & time_ok))
        if len(invalid):
            path = path[:int(invalid[0])]
        clean_paths.append(path)
        clean_lengths.append(len(path))
    clean_lengths = np.asarray(clean_lengths, dtype=np.int64)

    pairwise_distance = np.linalg.norm(
        selected_seeds[:, None, :] - selected_seeds[None, :, :], axis=-1
    )
    pairwise_distance[np.eye(len(selected_seeds), dtype=bool)] = np.inf
    return clean_paths, clean_lengths, {
        "requested_pathline_count": int(pathline_count),
        "actual_pathline_count": int(len(clean_paths)),
        "integration_steps": int(display_integration_steps),
        "loaded_frame_count": int(frame_count),
        "minimum_points": int(clean_lengths.min()),
        "median_points": float(np.median(clean_lengths)),
        "maximum_points": int(clean_lengths.max()),
        "relative_time_min": 0.0,
        "relative_time_max": float(target_time),
        "time_coloring": "continuous relative physical integration time",
        "seed_pool_count": int(len(seeds)),
        "seed_sampling": (
            "IVD-stratified deterministic greedy maximin "
            "(blue-noise approximation within each region)"
        ),
        "seed_sampling_random_seed": int(DISPLAY_BLUE_NOISE_SEED),
        "seed_sampling_uses_ivd": True,
        "seed_sampling_uses_cluster_labels": False,
        "vortex_region_definition": "seed IVD >= displayed IVD isosurface level",
        "vortex_region_threshold": float(vortex_threshold),
        "requested_vortex_fraction": float(vortex_fraction),
        "actual_vortex_seed_count": int(vortex_count),
        "actual_background_seed_count": int(background_count),
        "actual_vortex_fraction": float(vortex_count / len(selected_seeds)),
        "minimum_seed_distance_physical": float(pairwise_distance.min()),
        "load_metadata": display_load_metadata,
    }


def _dense_scene(spec, fitted, vortex_cluster, choice, dense_grid_size,
                 evaluation_ivd_percentile, display_ivd_percentile,
                 pathline_count, display_integration_steps,
                 vortex_pathline_fraction, device):
    confirmation = load_cache_records(
        Path(spec["confirmation"]) / spec["dataset"], 4
    )
    cached_record = confirmation[CONFIRMATION_ORDINAL]
    metadata = cached_record["metadata"]
    field, load_metadata = load_netcdf_window_3d(
        metadata["source_path"],
        int(metadata["source_start_index"]),
        int(metadata["frame_count"]),
        96,
    )
    offset = float(metadata["primitive_offset"])
    display_seed_pool, _ = generate_seeding_grid_3d(
        field,
        (dense_grid_size, dense_grid_size, dense_grid_size),
        boundary_fraction=0.08,
        offset=offset,
    )
    primitives, valid_mask, line_lengths = integrate_cross_primitives_3d(
        field,
        display_seed_pool,
        seed_time=0.0,
        dt=float(field.timeInterval) * 0.25,
        integration_steps=48,
        sampled_steps=32,
        offset=offset,
        method="RK4",
        chunk_size=2048,
    )
    seeds = display_seed_pool[valid_mask]
    original_valid = int(metadata["valid_primitives"])
    if len(seeds) < 2 * original_valid:
        raise RuntimeError(
            f"{spec['dataset']}: dense valid primitive count {len(seeds)} is less "
            f"than 2x the original {original_valid}"
        )

    labels = fitted.predict(_selected_features(primitives, choice, device))
    prediction = labels == vortex_cluster
    ivd_volume, ivd_at_display_seeds, axes = compute_ivd_reference_3d(
        field, 0.0, display_seed_pool
    )
    ivd_at_seeds = np.asarray(ivd_at_display_seeds)[valid_mask]
    finite_ivd = ivd_volume[np.isfinite(ivd_volume)]
    evaluation_threshold = float(
        np.percentile(finite_ivd, float(evaluation_ivd_percentile))
    )
    display_threshold = float(
        np.percentile(finite_ivd, float(display_ivd_percentile))
    )
    reference = np.asarray(ivd_at_seeds >= evaluation_threshold, dtype=bool)
    spacing = tuple(float(np.median(np.diff(values))) for values in axes)
    bounds = np.array(
        [[axes[0][0], axes[1][0], axes[2][0]],
         [axes[0][-1], axes[1][-1], axes[2][-1]]],
        dtype=np.float64,
    )
    mesh = marching_cubes_world(ivd_volume, display_threshold, spacing, bounds[0])
    metrics = binary_cluster_metrics(reference, labels, vortex_cluster)
    display_pathlines, display_pathline_lengths, display_pathline_metadata = (
        _integrate_display_pathlines(
            metadata,
            display_seed_pool,
            ivd_at_display_seeds,
            display_threshold,
            pathline_count,
            display_integration_steps,
            vortex_pathline_fraction,
        )
    )
    simulation_geometry = _load_simulation_geometry(
        spec, metadata["source_path"]
    )
    return {
        "cached_record": cached_record,
        "metadata": metadata,
        "load_metadata": load_metadata,
        "choice": choice,
        "seeds": seeds,
        "primitives": primitives,
        "line_lengths": line_lengths,
        "labels": labels,
        "prediction": prediction,
        "reference": reference,
        "ivd_at_seeds": ivd_at_seeds,
        "ivd_volume_shape": list(ivd_volume.shape),
        "evaluation_ivd_threshold": evaluation_threshold,
        "evaluation_ivd_percentile": float(evaluation_ivd_percentile),
        "display_ivd_threshold": display_threshold,
        "display_ivd_percentile": float(display_ivd_percentile),
        "ivd_mesh": mesh,
        "display_pathlines": display_pathlines,
        "display_pathline_lengths": display_pathline_lengths,
        "display_pathline_metadata": display_pathline_metadata,
        "simulation_geometry": simulation_geometry,
        "bounds": bounds,
        "metrics": metrics,
        "vortex_cluster": vortex_cluster,
        "requested_seed_count": int(dense_grid_size ** 3),
        "valid_seed_count": int(len(seeds)),
        "original_valid_seed_count": original_valid,
    }


def _base_title(spec, scene):
    metadata = scene["metadata"]
    return (
        f"{spec['title']} | source index {metadata['source_start_index']}, "
        f"t={metadata['source_time']:.4g}"
    )


def _save_figure(fig, path):
    # Fixed canvas bounds are required for frame-by-frame comparison.  In
    # particular, bbox_inches="tight" would crop figures differently depending
    # on whether a colorbar is present, changing the apparent camera framing.
    fig.savefig(path, dpi=260)
    plt.close(fig)


def _set_figure_title(fig, ax, headline, subtitle):
    # One figure-level text object avoids Matplotlib's 3D tight-layout bug,
    # which can otherwise crop the first of two independently positioned titles.
    fig.suptitle(f"{headline}\n{subtitle}", y=0.96, fontsize=12)
    ax.set_title("")


def _draw_ivd_pathline_layers(ax, scene):
    """Draw only the reusable data layers for the first figure."""
    # These are explanatory overlays, so preserve a fixed semantic layer order:
    # translucent IVD, solid simulation geometry, then time-coloured paths.
    ax.computed_zorder = False
    geometry = scene["simulation_geometry"]
    _add_ivd_surface(ax, scene["ivd_mesh"], alpha=0.18 if geometry else 0.27)
    _add_simulation_geometry(ax, geometry)
    display_pathlines = scene["display_pathlines"]
    time_min = float(scene["display_pathline_metadata"]["relative_time_min"])
    time_max = float(scene["display_pathline_metadata"]["relative_time_max"])
    time_norm = Normalize(vmin=time_min, vmax=time_max)
    time_cmap = plt.get_cmap("viridis")
    density_scale = min(
        1.0,
        np.sqrt(DEFAULT_PATHLINE_COUNT / max(len(display_pathlines), 1)),
    )
    for pathline in display_pathlines:
        if len(pathline) < 2:
            continue
        xyz = np.asarray(pathline[:, :3], dtype=np.float64)
        segments = np.stack((xyz[:-1], xyz[1:]), axis=1)
        segment_times = 0.5 * (pathline[:-1, 3] + pathline[1:, 3])
        collection = Line3DCollection(
            segments,
            cmap=time_cmap,
            norm=time_norm,
            linewidths=0.58 * density_scale,
            alpha=0.66,
            zorder=3,
        )
        collection.set_array(np.asarray(segment_times, dtype=np.float64))
        ax.add_collection3d(collection)
        ax.scatter(
            pathline[0, 0], pathline[0, 1], pathline[0, 2],
            color=[time_cmap(time_norm(float(pathline[0, 3])))],
            s=4.5 * density_scale ** 3,
            alpha=0.42 if len(display_pathlines) > DEFAULT_PATHLINE_COUNT else 0.68,
            depthshade=False,
            zorder=4,
        )
    time_mappable = ScalarMappable(norm=time_norm, cmap=time_cmap)
    time_mappable.set_array([])
    return time_mappable, time_cmap


def _render_ivd_pathlines(spec, scene, output_dir):
    fig, ax = _new_camera_figure()
    time_mappable, time_cmap = _draw_ivd_pathline_layers(ax, scene)
    display_pathlines = scene["display_pathlines"]
    colorbar_ax = fig.add_axes(COLORBAR_AXES_RECT)
    colorbar = fig.colorbar(time_mappable, cax=colorbar_ax)
    colorbar.set_label("Relative integration time Δt", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    _set_physical_axes(ax, scene["bounds"], spec["view"])
    _set_figure_title(
        fig,
        ax,
        _base_title(spec, scene),
        f"IVD p{scene['display_ivd_percentile']:g} "
        f"(level={scene['display_ivd_threshold']:.4g}) + "
        f"{len(display_pathlines)} time-coloured pathlines "
        f"({100 * scene['display_pathline_metadata']['actual_vortex_fraction']:.0f}% "
        f"core; {scene['display_pathline_metadata']['integration_steps']} steps)",
    )
    handles = [
        Line2D([0], [0], color=COLORS["ivd"], linewidth=7, alpha=0.55,
               label=f"IVD p{scene['display_ivd_percentile']:g} isosurface"),
        Line2D([0], [0], color=time_cmap(0.55), linewidth=1.5,
               label=(
                   "stratified blue-noise pathlines "
                   f"({scene['display_pathline_metadata']['actual_vortex_seed_count']} "
                   "core + "
                   f"{scene['display_pathline_metadata']['actual_background_seed_count']} "
                   "background; colour = time)"
               )),
    ]
    if scene["simulation_geometry"] is not None:
        handles.insert(
            0,
            Line2D(
                [0], [0], color=COLORS["geometry"], linewidth=7,
                alpha=0.72,
                label=scene["simulation_geometry"]["label"],
            ),
        )
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    path = output_dir / f"{spec['dataset']}_01_ivd_pathlines.png"
    _save_figure(fig, path)
    return path


def _draw_two_cluster_layers(ax, scene):
    """Draw only the reusable data layers for the second figure."""
    labels = scene["labels"]
    vortex_cluster = scene["vortex_cluster"]
    non_vortex = labels != vortex_cluster
    vortex = labels == vortex_cluster
    ax.scatter(
        scene["seeds"][non_vortex, 0],
        scene["seeds"][non_vortex, 1],
        scene["seeds"][non_vortex, 2],
        c=COLORS["non_vortex"], s=4.0, alpha=0.24, depthshade=False,
        label=f"non-vortex cluster (n={int(non_vortex.sum())})",
    )
    ax.scatter(
        scene["seeds"][vortex, 0],
        scene["seeds"][vortex, 1],
        scene["seeds"][vortex, 2],
        c=COLORS["vortex"], s=10.0, alpha=0.92, depthshade=False,
        label=f"vortex cluster (n={int(vortex.sum())})",
    )
    return non_vortex, vortex


def _render_two_clusters(spec, scene, output_dir):
    fig, ax = _new_camera_figure()
    _draw_two_cluster_layers(ax, scene)
    _set_physical_axes(ax, scene["bounds"], spec["view"])
    # Keep the cluster-view title on one line: the long physical domains make
    # Matplotlib's 3D tight bounding box unreliable for a two-line suptitle.
    fig.suptitle(
        f"{spec['title']} | FMT + KMeans clusters | "
        f"{len(scene['seeds'])} primitives",
        y=0.95,
        fontsize=12,
    )
    ax.set_title("")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    path = output_dir / f"{spec['dataset']}_02_fmt_kmeans_two_clusters.png"
    _save_figure(fig, path)
    return path


def _draw_prediction_layers(ax, scene):
    """Draw only the reusable data layers for the third figure."""
    _add_ivd_surface(ax, scene["ivd_mesh"], alpha=0.15)
    reference = scene["reference"]
    prediction = scene["prediction"]
    true_negative = ~reference & ~prediction
    true_positive = reference & prediction
    false_positive = ~reference & prediction
    false_negative = reference & ~prediction
    ax.scatter(
        scene["seeds"][true_negative, 0],
        scene["seeds"][true_negative, 1],
        scene["seeds"][true_negative, 2],
        c=COLORS["non_vortex"], s=2.0, alpha=0.035, depthshade=False,
        label=f"true negative (n={int(true_negative.sum())})",
    )
    ax.scatter(
        scene["seeds"][true_positive, 0],
        scene["seeds"][true_positive, 1],
        scene["seeds"][true_positive, 2],
        c=COLORS["vortex"], marker="o", s=13, alpha=0.92, depthshade=False,
        label=f"true positive (n={int(true_positive.sum())})",
    )
    ax.scatter(
        scene["seeds"][false_positive, 0],
        scene["seeds"][false_positive, 1],
        scene["seeds"][false_positive, 2],
        c=COLORS["false_positive"], marker="^", s=15, alpha=0.90,
        depthshade=False, label=f"false positive (n={int(false_positive.sum())})",
    )
    ax.scatter(
        scene["seeds"][false_negative, 0],
        scene["seeds"][false_negative, 1],
        scene["seeds"][false_negative, 2],
        c=COLORS["false_negative"], marker="x", s=21, alpha=0.95,
        depthshade=False, label=f"false negative (n={int(false_negative.sum())})",
    )
    return true_negative, true_positive, false_positive, false_negative


def _render_prediction_vs_ivd(spec, scene, output_dir):
    fig, ax = _new_camera_figure()
    _draw_prediction_layers(ax, scene)
    _set_physical_axes(ax, scene["bounds"], spec["view"])
    metrics = scene["metrics"]
    _set_figure_title(
        fig,
        ax,
        _base_title(spec, scene),
        f"Prediction versus IVD p{scene['evaluation_ivd_percentile']:g} labels; "
        f"p{scene['display_ivd_percentile']:g} isosurface shown: "
        f"F1={metrics['f1']:.3f}, "
        f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(
        0,
        Line2D(
            [0], [0], color=COLORS["ivd"], linewidth=7, alpha=0.42,
            label=f"IVD p{scene['display_ivd_percentile']:g} isosurface",
        ),
    )
    labels.insert(0, f"IVD p{scene['display_ivd_percentile']:g} isosurface")
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8)
    path = output_dir / f"{spec['dataset']}_03_prediction_vs_ivd.png"
    _save_figure(fig, path)
    return path


def _render_scene(spec, scene, output_dir):
    paths = {
        "ivd_pathlines": _render_ivd_pathlines(spec, scene, output_dir),
        "two_clusters": _render_two_clusters(spec, scene, output_dir),
        "prediction_vs_ivd": _render_prediction_vs_ivd(spec, scene, output_dir),
    }
    reference = scene["reference"]
    prediction = scene["prediction"]
    return {
        "dataset": spec["dataset"],
        "title": spec["title"],
        "confirmation_ordinal": CONFIRMATION_ORDINAL,
        "source_index": int(scene["metadata"]["source_start_index"]),
        "source_time": float(scene["metadata"]["source_time"]),
        "fmt_feature": scene["choice"]["feature"],
        "pca_dim": scene["choice"]["pca_dim"],
        "kmeans_seed": KMEANS_SEED,
        "cluster_as_vortex": int(scene["vortex_cluster"]),
        "original_valid_primitive_count": scene["original_valid_seed_count"],
        "requested_dense_primitive_count": scene["requested_seed_count"],
        "valid_dense_primitive_count": scene["valid_seed_count"],
        "valid_count_ratio": (
            scene["valid_seed_count"] / scene["original_valid_seed_count"]
        ),
        "evaluation_ivd_percentile": scene["evaluation_ivd_percentile"],
        "evaluation_ivd_threshold": scene["evaluation_ivd_threshold"],
        "display_ivd_percentile": scene["display_ivd_percentile"],
        "display_ivd_threshold": scene["display_ivd_threshold"],
        "ivd_volume_shape_zyx": scene["ivd_volume_shape"],
        "display_pathlines": {
            key: value for key, value in scene["display_pathline_metadata"].items()
            if key != "load_metadata"
        },
        "simulation_geometry": (
            None if scene["simulation_geometry"] is None
            else scene["simulation_geometry"]["metadata"]
        ),
        "reference_positive_count": int(reference.sum()),
        "prediction_positive_count": int(prediction.sum()),
        "true_positive_count": int((reference & prediction).sum()),
        "false_positive_count": int((~reference & prediction).sum()),
        "false_negative_count": int((reference & ~prediction).sum()),
        "metrics_against_evaluation_ivd": {
            name: float(value) for name, value in scene["metrics"].items()
        },
        "figures": {name: str(path) for name, path in paths.items()},
        "camera": {
            "elevation_degrees": float(spec["view"][0]),
            "azimuth_degrees": float(spec["view"][1]),
            "projection": "orthographic",
            "physical_bounds": np.asarray(scene["bounds"]).tolist(),
            "box_aspect": (
                np.asarray(scene["bounds"])[1]
                - np.asarray(scene["bounds"])[0]
            ).tolist(),
            "figure_size_inches": list(FIGURE_SIZE),
            "shared_axes_rectangle": list(CAMERA_AXES_RECT),
        },
    }


def run(output_dir=DEFAULT_OUTPUT, dense_grid_size=DEFAULT_DENSE_GRID_SIZE,
        evaluation_ivd_percentile=DEFAULT_EVALUATION_IVD_PERCENTILE,
        display_ivd_percentile=DEFAULT_DISPLAY_IVD_PERCENTILE,
        pathline_count=DEFAULT_PATHLINE_COUNT,
        display_integration_steps=DEFAULT_DISPLAY_INTEGRATION_STEPS,
        vortex_pathline_fraction=DEFAULT_VORTEX_PATHLINE_FRACTION,
        datasets=None):
    if int(dense_grid_size) ** 3 < 2 * ORIGINAL_SEED_COUNT:
        raise ValueError(
            "dense_grid_size must produce at least twice the original 16^3 seeds"
        )
    if not 0.0 < float(evaluation_ivd_percentile) < 100.0:
        raise ValueError("evaluation_ivd_percentile must be in (0,100)")
    if not 0.0 < float(display_ivd_percentile) < 100.0:
        raise ValueError("display_ivd_percentile must be in (0,100)")
    if int(pathline_count) < 1:
        raise ValueError("pathline_count must be positive")
    if int(display_integration_steps) <= 48:
        raise ValueError("display_integration_steps must be greater than frozen 48")
    if not 0.0 <= float(vortex_pathline_fraction) <= 1.0:
        raise ValueError("vortex_pathline_fraction must be in [0,1]")
    selected_datasets = set(datasets or [spec["dataset"] for spec in FLOW_SPECS])
    specs = [spec for spec in FLOW_SPECS if spec["dataset"] in selected_datasets]
    if len(specs) != len(selected_datasets):
        known = {spec["dataset"] for spec in FLOW_SPECS}
        raise ValueError(f"unknown datasets: {sorted(selected_datasets - known)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for spec in specs:
        flow_pathline_count = int(
            pathline_count * spec.get("pathline_count_multiplier", 1)
        )
        flow_integration_steps = int(
            display_integration_steps
            * spec.get("integration_steps_multiplier", 1)
        )
        fitted, vortex_cluster, choice = _fit_frozen_clusterer(spec, device)
        scene = _dense_scene(
            spec, fitted, vortex_cluster, choice, int(dense_grid_size),
            float(evaluation_ivd_percentile), float(display_ivd_percentile),
            flow_pathline_count,
            flow_integration_steps, float(vortex_pathline_fraction), device,
        )
        records.append(_render_scene(spec, scene, output_dir))
        print(
            f"{spec['dataset']}: {scene['valid_seed_count']} valid primitives, "
            f"IVD display p{display_ivd_percentile:g} "
            f"level={scene['display_ivd_threshold']:.6g}, "
            f"evaluation p{evaluation_ivd_percentile:g}, "
            f"F1={scene['metrics']['f1']:.3f}",
            flush=True,
        )

    payload = {
        "name": "Task1_3D_paper_candidates_1.9",
        "selection_rule": (
            "Four predeclared diverse flows; fixed confirmation ordinal 2 for "
            "every flow; no confirmation-score-based timeslice selection."
        ),
        "layout": "three independent 3D figures per flow; no 2D projections",
        "dense_grid_shape": [int(dense_grid_size)] * 3,
        "requested_dense_primitive_count": int(dense_grid_size) ** 3,
        "original_requested_primitive_count": ORIGINAL_SEED_COUNT,
        "evaluation_ivd_percentile": float(evaluation_ivd_percentile),
        "display_ivd_percentile": float(display_ivd_percentile),
        "pathline_count": int(pathline_count),
        "pathline_count_role": (
            "base count; deltaWing_LBM and boeing747 use a 2x per-flow multiplier"
        ),
        "vortex_pathline_fraction": float(vortex_pathline_fraction),
        "display_integration_steps": int(display_integration_steps),
        "display_integration_steps_role": (
            "base steps; deltaWing_LBM and boeing747 use a 2x per-flow multiplier"
        ),
        "frozen_fmt_integration_steps": 48,
        "device": device,
        "flows": records,
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dense-grid-size", type=int, default=DEFAULT_DENSE_GRID_SIZE)
    parser.add_argument(
        "--evaluation-ivd-percentile",
        type=float,
        default=DEFAULT_EVALUATION_IVD_PERCENTILE,
    )
    parser.add_argument(
        "--display-ivd-percentile",
        type=float,
        default=DEFAULT_DISPLAY_IVD_PERCENTILE,
    )
    parser.add_argument("--pathline-count", type=int, default=DEFAULT_PATHLINE_COUNT)
    parser.add_argument(
        "--display-integration-steps",
        type=int,
        default=DEFAULT_DISPLAY_INTEGRATION_STEPS,
    )
    parser.add_argument(
        "--vortex-pathline-fraction",
        type=float,
        default=DEFAULT_VORTEX_PATHLINE_FRACTION,
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Optional subset of tangaroa deltaWing_LBM boeing747 smokeBuoyancy",
    )
    args = parser.parse_args()
    run(
        args.output_dir,
        args.dense_grid_size,
        args.evaluation_ivd_percentile,
        args.display_ivd_percentile,
        args.pathline_count,
        args.display_integration_steps,
        args.vortex_pathline_fraction,
        args.datasets,
    )
