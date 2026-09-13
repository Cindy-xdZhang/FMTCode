"""Channel hairpin membership from local curve geometry; version 2.1.

GT supplies center-point labels only. Neither model receives flow values,
absolute positions, instance identifiers, or any GT-clipped geometry.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d


def read_dataset(path):
    import vtk
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    data = reader.GetOutput()
    if reader.GetErrorCode() or data is None or data.GetNumberOfPoints() == 0:
        raise ValueError(f"Cannot read dataset: {path}")
    return data


def load_channel(path, vector_name="vorticity"):
    from scipy.interpolate import RegularGridInterpolator
    from vtk.util.numpy_support import vtk_to_numpy
    grid = read_dataset(path)
    if not grid.IsA("vtkStructuredGrid"):
        raise ValueError("Expected the original structured Channel dataset")
    dimensions = [0, 0, 0]
    grid.GetDimensions(dimensions)
    nx, ny, nz = dimensions
    xyz = vtk_to_numpy(grid.GetPoints().GetData()).reshape(nz, ny, nx, 3)
    axes = [xyz[0, 0, :, 0].copy(), xyz[0, :, 0, 1].copy(), xyz[:, 0, 0, 2].copy()]
    for axis in axes:
        if not np.all(np.diff(axis) > 0):
            raise ValueError("Native axes must increase; no periodic padding is added")
    for component, reference in enumerate((axes[0][None, None, :], axes[1][None, :, None], axes[2][:, None, None])):
        if not np.allclose(xyz[..., component], reference, atol=1e-7, rtol=0):
            raise ValueError("Structured grid is not separable in the supplied x/y/z axes")
    vectors = vtk_to_numpy(grid.GetPointData().GetArray(vector_name)).reshape(nz, ny, nx, 3).copy()
    scalar = vtk_to_numpy(grid.GetPointData().GetArray("lambda2")).reshape(nz, ny, nx)
    # Trilinear interpolation at each native cell center, including nonuniform z.
    cell_lambda2 = sum(scalar[k:k+nz-1, j:j+ny-1, i:i+nx-1].astype(np.float64)
                       for k in (0, 1) for j in (0, 1) for i in (0, 1)) / 8
    interpolator = RegularGridInterpolator(tuple(axes[::-1]), vectors, bounds_error=False, fill_value=np.nan)
    return axes, cell_lambda2, interpolator


def gt_instance_bounds(gt):
    """Bounds of every annotated cell instance, INCLUDING VortexIds == 0."""
    from vtk.util.numpy_support import vtk_to_numpy
    values = vtk_to_numpy(gt.GetCellData().GetArray("VortexIds")).astype(np.int64)
    unique, inverse = np.unique(values, return_inverse=True)
    if np.any(unique < 0):
        raise ValueError("GT contains a negative ID; inspect its semantics before using it")
    cells = gt.GetCells()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    owners = np.repeat(inverse, np.diff(offsets))
    points = vtk_to_numpy(gt.GetPoints().GetData())
    low = np.full((len(unique), 3), np.inf)
    high = np.full_like(low, -np.inf)
    for axis in range(3):
        coordinate = points[connectivity, axis]
        np.minimum.at(low[:, axis], owners, coordinate)
        np.maximum.at(high[:, axis], owners, coordinate)
    return unique, low, high, values


def sample_gt(gt, points, locator=None):
    """Exact GT cell membership; -1 denotes outside, zero is a valid instance."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    if locator is None:
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(gt)
        locator.BuildLocator()
    values = vtk_to_numpy(gt.GetCellData().GetArray("VortexIds"))
    cells = np.fromiter((locator.FindCell(point) for point in points), dtype=np.int64, count=len(points))
    identifiers = np.full(len(points), -1, np.int64)
    inside = cells >= 0
    identifiers[inside] = values[cells[inside]].astype(np.int64)
    return identifiers, locator


def trace_cross(interpolator, seeds, step, neighbor_radius, steps=16):
    """Seven equal-arclength RK4 curves along the selected unit vector field."""
    seeds = np.asarray(seeds, np.float64)
    offsets = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                        [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float64) * neighbor_radius
    initial = seeds[:, None, :] + offsets
    query_low = np.full((len(seeds), 3), np.inf)
    query_high = np.full_like(query_low, -np.inf)

    def direction(points):
        finite = np.isfinite(points).all(-1)
        vector = np.full_like(points, np.nan)
        vector[finite] = interpolator(points[finite, ::-1])
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        unit = np.full_like(vector, np.nan)
        np.divide(vector, norm, out=unit, where=np.isfinite(norm) & (norm > 1e-12))
        query_low[:] = np.minimum(query_low, np.where(finite[..., None], points, np.inf).min(1))
        query_high[:] = np.maximum(query_high, np.where(finite[..., None], points, -np.inf).max(1))
        return unit

    halves = []
    for sign in (-1, 1):
        points = initial.copy()
        path = [points.copy()]
        for _ in range(steps):
            k1 = direction(points)
            k2 = direction(points + .5 * sign * step * k1)
            k3 = direction(points + .5 * sign * step * k2)
            k4 = direction(points + sign * step * k3)
            points = points + sign * step * (k1 + 2*k2 + 2*k3 + k4) / 6
            path.append(points.copy())
        halves.append(path)
    physical = np.stack(halves[0][:0:-1] + halves[1], axis=2)
    valid = np.isfinite(physical).all(axis=(1, 2, 3))
    support = neighbor_radius + steps * step
    normalized = (physical - seeds[:, None, None, :]) / support
    if valid.any() and np.abs(normalized[valid]).max() > 1 + 1e-6:
        raise ValueError("A curve escaped its analytic integration support")
    return normalized.astype(np.float32), valid, query_low, query_high


def fmt_features(geometry, options):
    with torch.no_grad():
        result = pathline_dft_features_3d(torch.as_tensor(geometry, dtype=torch.float32),
            num_freq=options["num_freq"], neighbor_weight=options["neighbor_weight"],
            neighbor_scale=options["neighbor_scale"], neighbor_pool="sort", mode="gram",
            include_chirality=True, return_numpy=False)
    if result.shape[1] != 161 or not torch.isfinite(result).all():
        raise ValueError("Invalid frozen FMT features")
    return result


@torch.no_grad()
def geometry_voxels(geometry, resolution=24, subdivisions=4):
    """Trilinear curve splatting: occupancy and mean signed tangent (4 channels)."""
    if geometry.ndim != 4 or geometry.shape[1:] != (7, 33, 3):
        raise ValueError("Conv3D expects the same seven 33-point curves as FMT")
    if not torch.isfinite(geometry).all() or geometry.abs().max() > 1.00001:
        raise ValueError("Invalid or out-of-support curve coordinates")
    start, delta = geometry[:, :, :-1], torch.diff(geometry, dim=2)
    fractions = torch.linspace(0, 1, subdivisions + 1, device=geometry.device, dtype=geometry.dtype)
    points = (start[..., None, :] + delta[..., None, :] * fractions[:, None]).reshape(len(geometry), -1, 3)
    tangent = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tangent = tangent[..., None, :].expand(-1, -1, -1, len(fractions), -1).reshape_as(points)
    coordinate = ((points + 1) * ((resolution-1)/2)).clamp(0, resolution-1)
    lower = coordinate.floor().long()
    fraction = coordinate - lower
    storage = geometry.new_zeros((len(geometry), 4, resolution**3))
    features = torch.cat((torch.ones_like(tangent[..., :1]), tangent), -1).transpose(1, 2)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = torch.tensor([dx, dy, dz], device=geometry.device)
                index = (lower + offset).clamp_max(resolution-1)
                weight = torch.where(offset.bool(), fraction, 1-fraction).prod(-1)
                flat = index[..., 2]*resolution**2 + index[..., 1]*resolution + index[..., 0]
                storage.scatter_add_(2, flat[:, None].expand(-1, 4, -1), features * weight[:, None])
    count = storage[:, :1].clone()
    storage[:, 1:] /= count.clamp_min(1e-12)
    storage[:, :1] = count.clamp_max(1)
    return storage.reshape(len(geometry), 4, resolution, resolution, resolution)


class FMTClassifier(nn.Module):
    def __init__(self, dropout=.1):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(161, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


class Conv3DClassifier(nn.Module):
    def __init__(self, dropout=.1):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(), nn.AdaptiveAvgPool3d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 256), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, voxels):
        return self.classifier(self.convolution(voxels)).squeeze(-1)
