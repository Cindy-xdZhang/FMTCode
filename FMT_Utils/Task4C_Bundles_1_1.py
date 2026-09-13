"""Task4-c vortex-line preparation and an explicit FMT bundle adaptation.

No taxonomy labels are inferred from VortexIds or geometric proxy rules.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d


def resample_line(points, count=32):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) <= 2 or not np.isfinite(points).all():
        return None
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, distance > 0]
    points = points[keep]
    if len(points) <= 2 or np.any(np.ptp(points, axis=0) == 0):
        return None  # Manuscript Sec. 3.1.1 rejects axis-degenerate traces.
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    query = np.linspace(0., arc[-1], count)
    return np.column_stack([np.interp(query, arc, points[:, i]) for i in range(3)])


def normalize_bundle(lines, seeds, *, max_lines=256, points=32, seed=94013):
    """Eqs. 8/9; normalize valid geometry before line subsampling or padding."""
    valid, valid_seeds = [], []
    if len(lines) != len(seeds):
        raise ValueError("Each trace needs its original head seed")
    for line, start in zip(lines, seeds):
        sampled = resample_line(line, points)
        if sampled is not None:
            valid.append(sampled)
            valid_seeds.append(start)
    if len(valid) < 10:
        raise ValueError("A paper bundle requires at least 10 nondegenerate lines")
    xyz = np.asarray(valid)
    centroid = xyz.mean(axis=(0, 1))
    radius = np.linalg.norm(xyz - centroid, axis=-1).max()
    if not radius > 0:
        raise ValueError("Zero bundle radius")
    xyz = (xyz - centroid) / radius
    tangent = np.diff(xyz, axis=1)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12)
    tangent = np.concatenate((tangent, tangent[:, -1:]), axis=1)
    features = np.concatenate((xyz, tangent), axis=-1)
    ids = np.arange(len(features))
    if len(ids) > max_lines:
        ids = np.sort(np.random.default_rng(seed).choice(ids, max_lines, replace=False))
    x = np.zeros((max_lines, points, 6), np.float32)
    mask = np.arange(max_lines) < len(ids)
    x[mask] = features[ids]
    head_seeds = np.zeros((max_lines, 3), np.float32)
    head_seeds[mask] = (np.asarray(valid_seeds)[ids] - centroid) / radius
    return x, mask, head_seeds, {"centroid": centroid.tolist(), "radius": float(radius),
                                "valid_line_count": len(valid), "selected_indices": ids.tolist()}


def fmt_bundle_features(x, mask, head_seeds):
    """Frozen 161D FMT on seven-line neighborhoods within the same vortex bundle.

    For each line, its head seed selects six nearest other lines. These are
    equal-arc-length vortex curves, NOT the material pathline cross in Task1-5.
    Neighborhood selection uses geometry only and never the taxonomy labels.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
    head_seeds = torch.as_tensor(head_seeds, dtype=x.dtype, device=x.device)
    if x.ndim != 4 or x.shape[-1] != 6 or mask.shape != x.shape[:2]:
        raise ValueError("Expected [batch, lines, points, 6] and [batch, lines] mask")
    output = x.new_zeros((*x.shape[:2], 161))
    with torch.no_grad():
        for b in range(len(x)):
            ids = mask[b].nonzero().flatten()
            if len(ids) < 10:
                raise ValueError("Need at least 10 lines for a paper-compatible bundle")
            seeds = head_seeds[b, ids]
            distance = torch.cdist(seeds, seeds)
            distance.fill_diagonal_(float("inf"))
            nearest = torch.argsort(distance, dim=1, stable=True)[:, :6]
            neighborhoods = torch.cat((torch.arange(len(ids), device=x.device)[:, None], nearest), 1)
            primitive = x[b, ids, :, :3][neighborhoods]
            output[b, ids] = pathline_dft_features_3d(
                primitive, num_freq=6, neighbor_weight=0.5, neighbor_scale=100.,
                neighbor_pool="sort", mode="gram", include_chirality=True, return_numpy=False)
    return output


class AuxiliaryEncoder(nn.Module):
    """Identical learned 161-to-128 residual for FMT and train-only Raw-PCA."""
    def __init__(self, paper_encoder):
        super().__init__()
        self.base = paper_encoder
        self.auxiliary = nn.Linear(161, 128)
        nn.init.zeros_(self.auxiliary.weight)
        nn.init.zeros_(self.auxiliary.bias)

    def forward(self, x, features, mask):
        lines = self.base.encode_lines(x)
        addition = self.auxiliary(features) * mask[..., None]
        return self.base.encode_bundle(lines + addition)


def read_vtk(path):
    import vtk
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(Path(path)))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    grid = reader.GetOutput()
    if reader.GetErrorCode() or grid is None or not grid.GetNumberOfPoints():
        raise ValueError(f"Cannot read VTK dataset: {path}")
    return grid


def trace_vortex_lines(grid, seeds, *, maximum_length, initial_step, maximum_error):
    """Bidirectional RK45 through the VORTICITY vector field, without wrapping."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    vectors = grid.GetPointData().GetArray("vorticity")
    if vectors is None or vectors.GetNumberOfComponents() != 3:
        raise ValueError("Vortex-line integration requires three-component vorticity")
    seed_data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(seeds, dtype=np.float64), deep=True))
    seed_data.SetPoints(vtk_points)
    halves = []
    for forward in (False, True):
        tracer = vtk.vtkStreamTracer()
        tracer.SetInputData(grid)
        tracer.SetSourceData(seed_data)
        tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vorticity")
        tracer.SetIntegratorTypeToRungeKutta45()
        tracer.SetIntegrationStepUnit(vtk.vtkStreamTracer.LENGTH_UNIT)
        tracer.SetMaximumPropagation(float(maximum_length))
        tracer.SetInitialIntegrationStep(float(initial_step))
        tracer.SetMinimumIntegrationStep(float(initial_step) * 0.05)
        tracer.SetMaximumIntegrationStep(float(initial_step) * 2.)
        tracer.SetMaximumError(float(maximum_error))
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
            seed_ids = result.GetCellData().GetArray("SeedIds")
            if seed_ids is None:
                raise RuntimeError("Tracer did not preserve seed identity")
            coordinates = vtk_to_numpy(result.GetPoints().GetData())
            for k in range(result.GetNumberOfCells()):
                cell = result.GetCell(k)
                ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                traces[int(seed_ids.GetTuple1(k))] = coordinates[ids].copy()
        halves.append(traces)
    lines = []
    starts = []
    for k in sorted(halves[0].keys() & halves[1].keys()):
        line = np.concatenate((halves[0][k][:0:-1], halves[1][k]), axis=0)
        lines.append(line)
        starts.append(seeds[k])
    return lines, starts
