"""Paper-preprocessed vortex bundles for binary Task4-c, version 3.1.

Geometry is constructed without GT. GT is used only for bundle labels and
instance isolation. Frozen 1.1/2.1 scientific implementations are not modified.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.Task4B_CrossFlow_3D import finite_difference_curl_zyx
from FMT_Utils.Task4C_Bundles_1_1 import resample_line
from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset, Conv3DClassifier


def load_flow(path, threshold):
    """Read native fields; use stored oyf and derive missing TBL curl from u."""
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    import vtk
    grid = read_dataset(path)
    dimensions = [0, 0, 0]
    if not grid.IsA("vtkStructuredGrid"):
        raise ValueError("Expected a native structured flow snapshot")
    grid.GetDimensions(dimensions)
    nx, ny, nz = dimensions
    xyz = vtk_to_numpy(grid.GetPoints().GetData()).reshape(nz, ny, nx, 3)
    axes = [xyz[0, 0, :, 0].copy(), xyz[0, :, 0, 1].copy(), xyz[:, 0, 0, 2].copy()]
    for d, ref in enumerate((axes[0][None, None, :], axes[1][None, :, None], axes[2][:, None, None])):
        if np.any(np.diff(axes[d]) <= 0) or not np.allclose(xyz[..., d], ref, atol=1e-7, rtol=0):
            raise ValueError("Grid must have separable, increasing native x/y/z axes")
    pd = grid.GetPointData()
    scalar = vtk_to_numpy(pd.GetArray("lambda2")).reshape(nz, ny, nx)
    oyf = vtk_to_numpy(pd.GetArray("oyf")).reshape(nz, ny, nx).copy()
    if not np.isfinite(scalar).all() or not np.isfinite(oyf).all():
        raise ValueError("Nonfinite lambda2 or stored spanwise vorticity fluctuation")
    derived = pd.GetArray("vorticity") is None
    if derived:
        velocity = vtk_to_numpy(pd.GetArray("velocity")).reshape(nz, ny, nx, 3)
        omega = finite_difference_curl_zyx(velocity, tuple(axes[::-1]))
    else:
        omega = vtk_to_numpy(pd.GetArray("vorticity")).reshape(nz, ny, nx, 3)
    # All eight vertices must satisfy lambda2, so the complete interpolation
    # cell is a candidate. Positive oyf is required at each actual seed too.
    vortex_cells = np.ones((nz-1, ny-1, nx-1), dtype=bool)
    cell_oyf = np.zeros(vortex_cells.shape, dtype=np.float64)
    for k in (0, 1):
        for j in (0, 1):
            for i in (0, 1):
                part = np.s_[k:k+nz-1, j:j+ny-1, i:i+nx-1]
                vortex_cells &= scalar[part] < threshold
                cell_oyf += oyf[part] / 8
    head_cells = vortex_cells & (cell_oyf > 0)
    point_head = (scalar < threshold) & (oyf > 0)
    iz = np.nonzero(point_head)[0]
    metadata = {
        "dimensions_xyz": dimensions, "bounds_xyz": [[float(a[0]), float(a[-1])] for a in axes],
        "lambda2_threshold": threshold, "all_grid_points": int(scalar.size),
        "lambda2_points": int((scalar < threshold).sum()), "head_points": int(point_head.sum()),
        "lambda2_point_fraction": float((scalar < threshold).mean()),
        "head_point_fraction": float(point_head.mean()), "vortex_cells_all_eight_vertices": int(vortex_cells.sum()),
        "head_cells": int(head_cells.sum()),
        "head_z_quantiles_0_50_90_99_100": np.quantile(axes[2][iz], [0, .5, .9, .99, 1]).tolist(),
        "vorticity_source": "second_order_native_grid_curl_of_velocity" if derived else "stored_vorticity",
        "oyf_source": "stored_oyf_without_recomputing_a_crop_mean", "periodic_wrapping": False,
        "extra_wall_height_cut": None,
    }
    # Tracer input contains only geometry and omega, never labels or instance IDs.
    trace_grid = vtk.vtkStructuredGrid()
    trace_grid.SetDimensions(dimensions)
    trace_grid.SetPoints(grid.GetPoints())
    vector = numpy_to_vtk(np.ascontiguousarray(omega.reshape(-1, 3)), deep=True)
    vector.SetName("vorticity")
    trace_grid.GetPointData().AddArray(vector)
    return axes, vortex_cells, head_cells, oyf, trace_grid, metadata


def cell_centers(indices, shape, axes):
    z, y, x = np.unravel_index(indices, shape)
    return np.column_stack([(axes[0][x]+axes[0][x+1])/2,
                            (axes[1][y]+axes[1][y+1])/2,
                            (axes[2][z]+axes[2][z+1])/2])


def native_partition(axes, cuts, split, stencil_halo):
    """Leave two native x nodes at each cut; include curl stencil in audit."""
    xs = axes[0]
    left = int(np.searchsorted(xs, cuts[split], side="left")) + 2
    right = int(np.searchsorted(xs, cuts[split+1], side="right")) - 3
    if right <= left + 2:
        raise ValueError("A spatial partition is too narrow")
    source = [max(0, left-stencil_halo), min(len(xs)-1, right+stencil_halo)]
    return left, right, source


def extract_vortex_grid(grid, vortex_cells, left, right):
    import vtk
    mask = vortex_cells.copy()
    mask[..., :left] = False
    mask[..., right:] = False
    ids = np.flatnonzero(mask)
    selected = vtk.vtkIdList()
    selected.SetNumberOfIds(len(ids))
    for i, value in enumerate(ids):
        selected.SetId(i, int(value))
    extract = vtk.vtkExtractCells()
    extract.SetInputData(grid)
    extract.SetCellList(selected)
    extract.Update()
    result = vtk.vtkUnstructuredGrid()
    result.ShallowCopy(extract.GetOutput())
    if not result.GetNumberOfCells():
        raise ValueError("No candidate cells in spatial partition")
    return result


def sample_head_seeds(cell_ids, shape, axes, oyf, count, rng):
    """Stratified jitter in head cells, rejecting nonpositive interpolated oyf."""
    accepted = []
    total = 0
    for _ in range(20):
        need = count-total
        chosen = rng.choice(cell_ids, max(need*2, 32), replace=True)
        z, y, x = np.unravel_index(chosen, shape)
        fractions = rng.uniform(.001, .999, (len(chosen), 3))
        values = np.zeros(len(chosen))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    weight = np.prod(np.where(np.array([dx, dy, dz]), fractions, 1-fractions), axis=1)
                    values += weight * oyf[z+dz, y+dy, x+dx]
        points = np.column_stack([axes[0][x] + fractions[:, 0]*(axes[0][x+1]-axes[0][x]),
                                  axes[1][y] + fractions[:, 1]*(axes[1][y+1]-axes[1][y]),
                                  axes[2][z] + fractions[:, 2]*(axes[2][z+1]-axes[2][z])])
        good = points[values > 0][:need]
        accepted.append(good)
        total += len(good)
        if total == count:
            result = np.concatenate(accepted)
            if len(np.unique(result, axis=0)) != count:
                raise ValueError("Duplicate physical head seeds")
            return result
    raise ValueError("Could not sample enough positive-oyf head seeds")


def trace_clean_lines(grid, seeds, options, wall_span):
    """VTK Fehlberg RK45, bidirectional omega lines, paper Sec. 3.1.1 cleanup."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    source = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(seeds, np.float64), deep=True))
    source.SetPoints(points)
    halves = []
    reasons = {}
    initial = options["initial_step_wall_span"] * wall_span
    for forward in (False, True):
        tracer = vtk.vtkStreamTracer()
        tracer.SetInputData(grid)
        tracer.SetSourceData(source)
        tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vorticity")
        tracer.SetIntegratorTypeToRungeKutta45()
        tracer.SetInterpolatorTypeToCellLocator()
        tracer.SetIntegrationStepUnit(vtk.vtkStreamTracer.LENGTH_UNIT)
        tracer.SetMaximumPropagation(options["maximum_length_wall_span"] * wall_span)
        tracer.SetInitialIntegrationStep(initial)
        tracer.SetMinimumIntegrationStep(initial * .05)
        tracer.SetMaximumIntegrationStep(initial * 2)
        tracer.SetMaximumError(options["maximum_error_wall_span"] * wall_span)
        tracer.SetMaximumNumberOfSteps(10000)
        tracer.SetTerminalSpeed(1e-12)
        tracer.SetComputeVorticity(False)
        if forward:
            tracer.SetIntegrationDirectionToForward()
        else:
            tracer.SetIntegrationDirectionToBackward()
        tracer.Update()
        result = tracer.GetOutput()
        traces = {}
        if result.GetNumberOfCells():
            seed_ids = vtk_to_numpy(result.GetCellData().GetArray("SeedIds"))
            coordinates = vtk_to_numpy(result.GetPoints().GetData())
            termination = vtk_to_numpy(result.GetCellData().GetArray("ReasonForTermination"))
            for k in range(result.GetNumberOfCells()):
                cell = result.GetCell(k)
                ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                traces[int(seed_ids[k])] = coordinates[ids].copy()
                reason = str(int(termination[k]))
                reasons[reason] = reasons.get(reason, 0)+1
        halves.append(traces)
    cleaned = np.zeros((len(seeds), options["points_per_line"], 3), np.float32)
    valid = np.zeros(len(seeds), bool)
    low = np.full((len(seeds), 3), np.inf)
    high = np.full((len(seeds), 3), -np.inf)
    stats = {"seeds": len(seeds), "short_or_missing": 0, "axis_degenerate": 0, "nonfinite": 0,
             "valid": 0, "termination_reasons": reasons}
    # A valid combined curve may have a missing half; do not discard it merely
    # for being one-sided when its combined P and spatial variance are valid.
    for i in range(len(seeds)):
        back = halves[0].get(i, np.asarray(seeds[i:i+1]))
        front = halves[1].get(i, np.asarray(seeds[i:i+1]))
        curve = np.concatenate((back[:0:-1], front))
        if not np.isfinite(curve).all():
            stats["nonfinite"] += 1
            continue
        curve = curve[np.r_[True, np.linalg.norm(np.diff(curve, axis=0), axis=1) > 0]]
        if len(curve) <= 2:
            stats["short_or_missing"] += 1
            continue
        if np.any(np.ptp(curve, axis=0) == 0):
            stats["axis_degenerate"] += 1
            continue
        sampled = resample_line(curve, options["points_per_line"])
        if sampled is None:
            raise ValueError("Unexpected disagreement in line cleaning")
        cleaned[i], valid[i] = sampled, True
        low[i], high[i] = curve.min(0), curve.max(0)
    stats["valid"] = int(valid.sum())
    return cleaned, valid, low, high, stats


def normalize_view(lines, seeds, max_lines=256):
    """Equal-arc geometry; Eq. 8 before zero padding. Mask excludes padding."""
    lines = np.asarray(lines, np.float64)
    if len(lines) < 10 or len(lines) > max_lines or lines.shape[1:] != (32, 3):
        raise ValueError("Each accepted bundle needs 10..256 valid 32-point lines")
    if not np.isfinite(lines).all() or np.any(np.ptp(lines, axis=1) == 0):
        raise ValueError("Invalid or axis-degenerate resampled geometry")
    centroid = lines.mean(axis=(0, 1))
    radius = float(np.linalg.norm(lines-centroid, axis=-1).max())
    if not radius > 0:
        raise ValueError("Zero bundle normalization radius")
    geometry = np.zeros((max_lines, 32, 3), np.float32)
    normalized_seeds = np.zeros((max_lines, 3), np.float32)
    geometry[:len(lines)] = (lines-centroid)/radius
    normalized_seeds[:len(lines)] = (np.asarray(seeds)-centroid)/radius
    return geometry, normalized_seeds, centroid, radius


@torch.no_grad()
def bundle_fmt(geometry, seeds, counts):
    """161D frozen FMT per line's six nearest seed neighbors; mean/max pool."""
    b, n, p, _ = geometry.shape
    mask = torch.arange(n, device=geometry.device)[None] < counts[:, None]
    if (counts < 10).any() or not torch.isfinite(geometry).all():
        raise ValueError("FMT requires cleaned, masked bundles")
    distance = torch.cdist(seeds, seeds)
    distance.masked_fill_(~mask[:, None, :], float("inf"))
    distance.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    nearest = torch.argsort(distance, dim=-1, stable=True)[..., :6]
    central = torch.arange(n, device=geometry.device)[None, :, None].expand(b, -1, -1)
    ids = torch.cat((central, nearest), dim=-1)
    batch, line = mask.nonzero(as_tuple=True)
    features = geometry.new_zeros((b, n, 161))
    for offset in range(0, len(batch), 1024):
        bi, li = batch[offset:offset+1024], line[offset:offset+1024]
        primitive = geometry[bi[:, None], ids[bi, li]]
        features[bi, li] = pathline_dft_features_3d(primitive, num_freq=6,
            neighbor_weight=.5, neighbor_scale=100., neighbor_pool="sort", mode="gram",
            include_chirality=True, return_numpy=False)
    mean = features.sum(1)/counts[:, None]
    maximum = features.masked_fill(~mask[..., None], -float("inf")).max(1).values
    result = torch.cat((mean, maximum), dim=1)
    if result.shape != (b, 322) or not torch.isfinite(result).all():
        raise ValueError("Invalid bundle FMT output")
    return result


@torch.no_grad()
def bundle_voxels(geometry, counts, resolution=24, subdivisions=4):
    """The frozen 2.1 spatial splat generalized to masked 10..256-line bundles."""
    if geometry.ndim != 4 or geometry.shape[2:] != (32, 3):
        raise ValueError("Expected 32-point bundle geometry")
    if not torch.isfinite(geometry).all() or geometry.abs().max() > 1.00001:
        raise ValueError("Invalid normalized geometry")
    start, delta = geometry[:, :, :-1], geometry.diff(dim=2)
    fractions = torch.linspace(0, 1, subdivisions+1, device=geometry.device)
    points = (start[..., None, :] + delta[..., None, :]*fractions[:, None]).reshape(len(geometry), -1, 3)
    tangent = delta/delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tangent = tangent[..., None, :].expand(-1, -1, -1, len(fractions), -1).reshape_as(points)
    valid = (torch.arange(geometry.shape[1], device=geometry.device)[None] < counts[:, None])
    valid = valid[:, :, None, None].expand(-1, -1, 31, len(fractions)).reshape(len(geometry), -1)
    coordinate = ((points+1)*((resolution-1)/2)).clamp(0, resolution-1)
    lower = coordinate.floor().long()
    fraction = coordinate-lower
    storage = geometry.new_zeros((len(geometry), 4, resolution**3))
    features = torch.cat((torch.ones_like(tangent[..., :1]), tangent), -1).transpose(1, 2)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = torch.tensor([dx, dy, dz], device=geometry.device)
                index = (lower+offset).clamp_max(resolution-1)
                weight = torch.where(offset.bool(), fraction, 1-fraction).prod(-1)*valid
                flat = index[..., 2]*resolution**2+index[..., 1]*resolution+index[..., 0]
                storage.scatter_add_(2, flat[:, None].expand(-1, 4, -1), features*weight[:, None])
    count = storage[:, :1].clone()
    storage[:, 1:] /= count.clamp_min(1e-12)
    storage[:, :1] = count.clamp_max(1)
    return storage.reshape(len(geometry), 4, resolution, resolution, resolution)


class FMTBundleClassifier(nn.Module):
    def __init__(self, dropout=.1):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(322, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)
