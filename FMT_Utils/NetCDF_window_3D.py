"""Out-of-core NetCDF windows for large 3D unsteady vector fields."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np

from FLowUtils.VectorField3d import UnsteadyVectorField3D


_AXIS_ALIASES = {
    "x": {"x", "xdim"}, "y": {"y", "ydim"},
    "z": {"z", "zdim"}, "t": {"t", "time", "tdim"},
}


def _axis_dimension(dataset, axis):
    aliases = _AXIS_ALIASES[axis]
    for name in dataset.dimensions:
        if name.lower() in aliases:
            return name
    raise ValueError(f"NetCDF misses a {axis!r} dimension; found {list(dataset.dimensions)}")


def _coordinate(dataset, dimension, indices):
    candidates = [dimension] + [name for name in dataset.variables
                                if name.lower() in _AXIS_ALIASES[dimension[0].lower()]]
    for name in candidates:
        if name in dataset.variables and dataset.variables[name].dimensions == (dimension,):
            values = np.ma.asarray(dataset.variables[name][indices])
            if values.count() == values.size and np.isfinite(values).all():
                return np.asarray(values, dtype=np.float64)
    return np.asarray(indices, dtype=np.float64)


def inspect_netcdf_3d(path):
    path = Path(path)
    with nc.Dataset(path) as dataset:
        dims = {axis: _axis_dimension(dataset, axis) for axis in "xyzt"}
        shape = {axis: len(dataset.dimensions[dims[axis]]) for axis in "xyzt"}
        time = _coordinate(dataset, dims["t"], np.arange(shape["t"]))
        time_source = "coordinate" if not np.array_equal(time, np.arange(shape["t"])) else "index"
    return {"path": str(path.resolve()), "dimensions": dims, "shape": shape,
            "time_min": float(time[0]), "time_max": float(time[-1]),
            "time_source": time_source}


def interior_time_indices(time_count, count=10, begin_fraction=0.20, end_fraction=0.90,
                          required_future_frames=0):
    """Evenly choose unique indices in [20%,90%), leaving integration look-ahead."""
    first = int(np.ceil(float(begin_fraction) * time_count))
    exclusive_end = int(np.floor(float(end_fraction) * time_count))
    last = min(exclusive_end - 1, time_count - 1 - int(required_future_frames))
    if first > last or last - first + 1 < int(count):
        raise ValueError(
            f"not enough time frames: count={time_count}, usable=[{first},{last}], "
            f"requested={count}, future={required_future_frames}"
        )
    indices = np.rint(np.linspace(first, last, int(count))).astype(np.int64)
    if len(np.unique(indices)) != int(count):
        raise RuntimeError("time-index selection produced duplicates")
    return indices


def resolve_time_indices(time_count, count, begin_fraction, end_fraction,
                         required_future_frames=0, fixed_indices=None):
    """Select time indices or validate a caller-supplied frozen schedule."""
    if fixed_indices is None:
        return interior_time_indices(
            time_count, count, begin_fraction, end_fraction, required_future_frames
        )
    raw = np.asarray(fixed_indices)
    if raw.ndim != 1 or len(raw) != int(count):
        raise ValueError("fixed_time_indices count does not match sampling.timeslices")
    if not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
        raise ValueError("fixed_time_indices must contain finite integers")
    indices = raw.astype(np.int64)
    if len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError("fixed_time_indices must be unique and strictly increasing")
    if indices[0] < 0 or indices[-1] + int(required_future_frames) >= int(time_count):
        raise ValueError("fixed_time_indices do not leave enough future frames")
    return indices


def load_netcdf_window_3d(path, start_index, frame_count, max_spatial_dim=96):
    """Read a temporal window and strided spatial grid into UnsteadyVectorField3D."""
    path = Path(path)
    with nc.Dataset(path) as dataset:
        dims = {axis: _axis_dimension(dataset, axis) for axis in "xyzt"}
        sizes = {axis: len(dataset.dimensions[dims[axis]]) for axis in "xyzt"}
        start = int(start_index); stop = start + int(frame_count)
        if start < 0 or stop > sizes["t"]:
            raise ValueError(f"requested time window [{start},{stop}) outside [0,{sizes['t']})")
        strides = {axis: max(1, int(np.ceil(sizes[axis] / int(max_spatial_dim))))
                   for axis in "xyz"}
        index = {
            "t": slice(start, stop),
            **{axis: slice(0, sizes[axis], strides[axis]) for axis in "xyz"},
        }
        canonical = [dims[axis] for axis in "tzyx"]
        components = []
        for names in (("u", "v", "w"), ("velocity_x", "velocity_y", "velocity_z"),
                      ("Component1", "Component2", "Component3")):
            if all(name in dataset.variables for name in names):
                components = list(names); break
        if not components:
            raise ValueError("could not find 3D velocity components")
        arrays = []
        for component in components:
            variable = dataset.variables[component]
            source_dims = list(variable.dimensions)
            if any(name not in source_dims for name in canonical):
                raise ValueError(f"{component} dimensions {source_dims} do not contain {canonical}")
            source_slices = tuple(index["tzyx"[canonical.index(name)]]
                                  if name in canonical else 0 for name in source_dims)
            raw = np.ma.asarray(variable[source_slices])
            raw = raw.filled(0.0) if np.ma.isMaskedArray(raw) else raw
            order = [source_dims.index(name) for name in canonical]
            arrays.append(np.transpose(np.asarray(raw, dtype=np.float32), order))
        field_data = np.ascontiguousarray(np.stack(arrays, axis=-1))
        coords = {}
        for axis in "xyz":
            selected = np.arange(sizes[axis])[index[axis]]
            coords[axis] = _coordinate(dataset, dims[axis], selected)
        all_time = _coordinate(dataset, dims["t"], np.arange(sizes["t"]))
        selected_time = all_time[start:stop]
        if len(selected_time) > 1:
            differences = np.diff(selected_time)
            if not np.allclose(differences, differences[0], rtol=1e-4, atol=1e-8):
                raise ValueError("non-uniform time coordinates are unsupported")
            time_step = float(differences[0])
        else:
            time_step = 1.0
    z_count, y_count, x_count = field_data.shape[1:4]
    dmin = np.array([coords["x"][0], coords["y"][0], coords["z"][0]], dtype=np.float32)
    dmax = np.array([coords["x"][-1], coords["y"][-1], coords["z"][-1]], dtype=np.float32)
    local_tmax = float((len(selected_time) - 1) * time_step)
    vector_field = UnsteadyVectorField3D(
        x_count, y_count, z_count, len(selected_time), dmin, dmax, 0.0, local_tmax
    )
    vector_field.field = field_data
    metadata = {
        "source_path": str(path.resolve()), "source_start_index": start,
        "source_time": float(selected_time[0]), "source_time_step": time_step,
        "frame_count": len(selected_time), "spatial_strides": strides,
        "loaded_shape_TZYXC": list(field_data.shape),
    }
    return vector_field, metadata
