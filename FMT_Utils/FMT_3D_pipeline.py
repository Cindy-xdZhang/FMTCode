"""Data generation and visualization helpers for ``mainExp_3DFMT_1.1``."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch

from FLowUtils.VectorField3d import UnsteadyVectorField3D
from FLowUtils.flowlineIntegral import compute_pathlines_3D_batch


def _axis_name(names, aliases):
    for name in names:
        if name.lower() in aliases:
            return name
    raise ValueError(f"missing axis {sorted(aliases)}; available names: {list(names)}")


def load_vector_field_3d(path: str | Path) -> UnsteadyVectorField3D:
    """Load a canonical ``[T,Z,Y,X,3]`` 3D field from NetCDF or NPZ.

    NPZ must contain ``field`` and may contain ``domain_min``, ``domain_max``,
    ``tmin`` and ``tmax``.  NetCDF supports common scalar component names and
    reorders each variable from its declared dimensions rather than assuming an
    on-disk axis order.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        field = np.asarray(data["field"], dtype=np.float32)
        if field.ndim == 4:
            field = field[None]
        if field.ndim != 5 or field.shape[-1] != 3:
            raise ValueError(f"NPZ field must be [T,Z,Y,X,3], got {field.shape}")
        t, z, y, x, _ = field.shape
        dmin = np.asarray(data.get("domain_min", [0.0, 0.0, 0.0]), dtype=np.float32)
        dmax = np.asarray(data.get("domain_max", [x - 1, y - 1, z - 1]), dtype=np.float32)
        tmin = float(data.get("tmin", 0.0))
        tmax = float(data.get("tmax", max(t - 1, 0)))
    elif path.suffix.lower() in (".nc", ".netcdf"):
        with nc.Dataset(path, "r") as ds:
            dims = list(ds.dimensions)
            variables = list(ds.variables)
            xname = _axis_name(dims, {"x", "xdim"})
            yname = _axis_name(dims, {"y", "ydim"})
            zname = _axis_name(dims, {"z", "zdim"})
            time_names = [n for n in dims if n.lower() in {"time", "t", "tdim"}]
            tname = time_names[0] if time_names else None

            def coordinate(axis_name, aliases, size):
                candidates = [axis_name] + [n for n in variables if n.lower() in aliases]
                for candidate in candidates:
                    if candidate in ds.variables and ds.variables[candidate].ndim == 1:
                        return np.asarray(ds.variables[candidate][:])
                return np.arange(size, dtype=np.float32)

            component_sets = (
                ("u", "v", "w"),
                ("velocity_x", "velocity_y", "velocity_z"),
                ("a", "b", "c"),
                ("Component1", "Component2", "Component3"),
            )
            components = next(
                (names for names in component_sets if all(name in ds.variables for name in names)),
                None,
            )
            if components is None:
                raise ValueError(f"no supported velocity components in {variables}")

            canonical = ([tname] if tname else []) + [zname, yname, xname]
            arrays = []
            for component in components:
                var = ds.variables[component]
                source_dims = list(var.dimensions)
                missing = [name for name in canonical if name not in source_dims]
                if missing:
                    raise ValueError(
                        f"component {component} dimensions {source_dims} miss {missing}"
                    )
                arr = np.asarray(var[:], dtype=np.float32)
                order = [source_dims.index(name) for name in canonical]
                arr = np.transpose(arr, order)
                if not tname:
                    arr = arr[None]
                arrays.append(arr)
            field = np.stack(arrays, axis=-1)
            t, z, y, x, _ = field.shape
            xv = coordinate(xname, {"x", "xdim"}, x)
            yv = coordinate(yname, {"y", "ydim"}, y)
            zv = coordinate(zname, {"z", "zdim"}, z)
            dmin = np.array([xv.min(), yv.min(), zv.min()], dtype=np.float32)
            dmax = np.array([xv.max(), yv.max(), zv.max()], dtype=np.float32)
            time_variable = next(
                (n for n in variables if n.lower() in {"time", "t", "tdim"}), None
            )
            if tname and time_variable:
                tv = np.asarray(ds.variables[time_variable][:], dtype=np.float64)
                tmin, tmax = float(tv.min()), float(tv.max())
            else:
                tmin, tmax = 0.0, float(max(t - 1, 0))
    else:
        raise ValueError(f"unsupported field extension: {path.suffix}")

    vector_field = UnsteadyVectorField3D(x, y, z, t, dmin, dmax, tmin, tmax)
    vector_field.field = np.ascontiguousarray(field, dtype=np.float32)
    return vector_field


def generate_seeding_grid_3d(vector_field, grid_shape, boundary_fraction, offset):
    """Return flattened seeds and axis arrays, safely inset for the 7-line cross."""
    grid_shape = tuple(int(v) for v in grid_shape)
    if len(grid_shape) != 3 or min(grid_shape) < 2:
        raise ValueError("grid_shape must contain three integers >= 2")
    dmin = np.asarray(vector_field.domainMinBoundary, dtype=np.float64)
    dmax = np.asarray(vector_field.domainMaxBoundary, dtype=np.float64)
    span = dmax - dmin
    margin = np.maximum(float(boundary_fraction) * span, float(offset) * 1.01)
    if np.any(2 * margin >= span):
        raise ValueError("boundary margin leaves no seeding volume")
    # grid_shape is [nx,ny,nz], while meshgrid output is [nz,ny,nx].
    xs = np.linspace(dmin[0] + margin[0], dmax[0] - margin[0], grid_shape[0])
    ys = np.linspace(dmin[1] + margin[1], dmax[1] - margin[1], grid_shape[1])
    zs = np.linspace(dmin[2] + margin[2], dmax[2] - margin[2], grid_shape[2])
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1), (xs, ys, zs)


def integrate_cross_primitives_3d(
    vector_field,
    seeds_xyz,
    seed_time,
    dt,
    integration_steps,
    sampled_steps,
    offset,
    method="RK4",
    chunk_size=2048,
):
    """Integrate center/x±/y±/z± and return only fully valid primitives."""
    if integration_steps < 1 or not 2 <= sampled_steps <= integration_steps + 1:
        raise ValueError("require integration_steps>=1 and 2<=sampled_steps<=steps+1")
    target_time = float(seed_time) + float(dt) * int(integration_steps)
    if not vector_field.tmin <= seed_time <= vector_field.tmax:
        raise ValueError("seed_time is outside the field time range")
    if target_time > vector_field.tmax + 1e-12:
        raise ValueError(
            f"integration target {target_time:g} exceeds field tmax={vector_field.tmax:g}"
        )

    offsets = np.array(
        [[0, 0, 0], [offset, 0, 0], [-offset, 0, 0],
         [0, offset, 0], [0, -offset, 0], [0, 0, offset], [0, 0, -offset]],
        dtype=np.float64,
    )
    expanded = (np.asarray(seeds_xyz)[:, None, :] + offsets[None]).reshape(-1, 3)
    expanded = np.column_stack((expanded, np.full(len(expanded), seed_time)))
    desired_length = integration_steps + 1
    sample_index = np.linspace(0, integration_steps, sampled_steps).round().astype(np.int64)
    chunks, length_chunks = [], []
    for start in range(0, len(expanded), int(chunk_size)):
        result = compute_pathlines_3D_batch(
            vector_field,
            expanded[start:start + int(chunk_size)],
            min_time=float(seed_time),
            max_time=target_time,
            step_size=float(dt),
            max_iteration=int(integration_steps),
            method=method,
        )
        if result is None:
            raise RuntimeError(f"unsupported 3D integration method: {method}")
        positions, lengths = result
        chunks.append(positions[:, :desired_length])
        length_chunks.append(lengths)
    positions = np.concatenate(chunks).reshape(-1, 7, desired_length, 4)
    lengths = np.concatenate(length_chunks).reshape(-1, 7)
    dmin = np.asarray(vector_field.domainMinBoundary).reshape(1, 1, 1, 3)
    dmax = np.asarray(vector_field.domainMaxBoundary).reshape(1, 1, 1, 3)
    xyz = positions[..., :3]
    spatially_valid = ((xyz >= dmin) & (xyz <= dmax)).all(axis=(1, 2, 3))
    valid = (lengths == desired_length).all(axis=1) & spatially_valid
    return positions[valid][:, :, sample_index], valid, lengths


def visualize_3d_clustering(seeds, labels, primitives, output_dir, max_lines=32):
    """Write 3D scatter, orthogonal projections and representative center lines."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    colors = np.where(labels == 0, "#277da1", "#f94144")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], c=colors, s=6, alpha=0.75)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="3D FMT KMeans clusters (IDs are arbitrary)")
    fig.tight_layout(); fig.savefig(output_dir / "clusters_3d.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (a, b, names) in zip(axes, ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz"))):
        ax.scatter(seeds[:, a], seeds[:, b], c=colors, s=5, alpha=0.7)
        ax.set(xlabel=names[0], ylabel=names[1], title=f"{names} projection")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(); fig.savefig(output_dir / "clusters_projections.png", dpi=220); plt.close(fig)

    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(0)
    for cluster in np.unique(labels):
        candidates = np.flatnonzero(labels == cluster)
        chosen = rng.choice(candidates, min(max_lines // 2, len(candidates)), replace=False)
        for index in chosen:
            line = primitives[index, 0, :, :3]
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=("#277da1" if cluster == 0 else "#f94144"), alpha=0.65)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Representative center pathlines")
    fig.tight_layout(); fig.savefig(output_dir / "cluster_pathlines.png", dpi=220); plt.close(fig)


def write_run_metadata(path, metadata):
    Path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
