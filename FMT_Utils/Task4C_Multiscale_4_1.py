"""Multiscale head-seeded vortex bundles and regularized classifiers, 4.1."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from FMT_Utils.Task4C_PaperBundles_3_1 import (
    cell_centers, normalize_view, trace_clean_lines, bundle_fmt, bundle_voxels,
    FMTBundleClassifier, Conv3DClassifier,
)


def extract_native_partition(grid, axes, left, right):
    """Full omega volume in a source-disjoint slab; lambda2 gates seeds only."""
    import vtk
    extract = vtk.vtkExtractGrid()
    extract.SetInputData(grid)
    extract.SetVOI(left, right, 0, len(axes[1])-1, 0, len(axes[2])-1)
    extract.Update()
    result = vtk.vtkStructuredGrid()
    result.ShallowCopy(extract.GetOutput())
    return result


def interpolate_scalar(points, axes, values):
    """Native trilinear interpolation with explicit inside-domain validity."""
    indices = [np.searchsorted(a, points[:, j], side="right")-1 for j, a in enumerate(axes)]
    valid = np.ones(len(points), bool)
    for j in range(3):
        valid &= (indices[j] >= 0) & (indices[j] < len(axes[j])-1)
        indices[j] = np.clip(indices[j], 0, len(axes[j])-2)
    fraction = np.column_stack([(points[:, j]-axes[j][indices[j]]) /
                               (axes[j][indices[j]+1]-axes[j][indices[j]]) for j in range(3)])
    result = np.zeros(len(points))
    x, y, z = indices
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                weight = np.prod(np.where(np.array([dx, dy, dz]), fraction, 1-fraction), axis=1)
                result += weight*values[z+dz, y+dy, x+dx]
    return result, valid, np.ravel_multi_index((z, y, x), np.array(values.shape)-1)


def center_and_neighbors(row, component_grid, axes, oyf, center_number, scale, base_seed):
    """Same center at different scales; 27 cubic offsets change actual distance.

    Coordinates are sampled independently of GT. A neighbor is a seed only if
    it remains in the same candidate head and has positive interpolated oyf.
    """
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, row["head_component"], center_number]))
    cells = row["cell_ids"]
    # All scales at a given center share its cell and within-cell displacement.
    cell = int(rng.choice(cells))
    iz, iy, ix = np.unravel_index(cell, component_grid.shape)
    spacing = np.array([axes[0][ix+1]-axes[0][ix], axes[1][iy+1]-axes[1][iy], axes[2][iz+1]-axes[2][iz]])
    center = np.array([axes[0][ix], axes[1][iy], axes[2][iz]]) + rng.uniform(.25, .75, 3)*spacing
    offsets = np.array([[x, y, z] for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)], float)
    center_index = int(np.flatnonzero((offsets == 0).all(1))[0])
    offsets[[0, center_index]] = offsets[[center_index, 0]]
    distance = float(np.prod(spacing)**(1/3))*scale["neighbor_grid_scale"]
    seeds = center + offsets*distance
    fluct, inside, cell_id = interpolate_scalar(seeds, axes, oyf)
    valid = inside & (fluct > 0) & (component_grid.ravel()[cell_id] == row["head_component"])
    if not valid[0]:
        return None
    return {"center": center, "seeds": seeds[valid], "source_cell": cell,
            "offset_grid_ids": np.flatnonzero(valid), "neighbor_distance": distance,
            "seed_rms_distance": float(np.sqrt(np.mean(np.sum((seeds[valid]-center)**2, axis=1))))}


def trace_primitive_batch(mesh, samples, scale, spec, wall_span, x_bounds):
    """Trace all accepted seeds in one call; clean lines, then reject short bundles."""
    lengths = [len(s["seeds"]) for s in samples]
    offsets = np.r_[0, np.cumsum(lengths)]
    options = dict(spec["bundles"])
    options.update(initial_step_wall_span=scale["initial_step_wall_span"],
                   maximum_length_wall_span=scale["maximum_length_wall_span"],
                   maximum_error_wall_span=scale["maximum_error_wall_span"])
    curves, valid, low, high, stats = trace_clean_lines(mesh, np.concatenate([s["seeds"] for s in samples]), options, wall_span)
    records, rejected = [], []
    guard = 2*scale["initial_step_wall_span"]*wall_span
    for j, sample in enumerate(samples):
        sl = slice(offsets[j], offsets[j+1])
        good = valid[sl]
        if good.sum() < spec["bundles"]["minimum_valid_lines"]:
            rejected.append({"reason": "fewer_than_10_clean_lines", "head_component": sample["head_component"], "label": sample["label"]})
            continue
        lo, hi = low[sl][good].min(0), high[sl][good].max(0)
        if lo[0] <= x_bounds[0]+guard or hi[0] >= x_bounds[1]-guard:
            rejected.append({"reason": "artificial_partition_boundary", "head_component": sample["head_component"], "label": sample["label"]})
            continue
        lines = curves[sl][good]
        if np.any(np.ptp(lines, axis=1) == 0):
            rejected.append({"reason": "resampled_axis_degenerate", "head_component": sample["head_component"], "label": sample["label"]})
            continue
        geometry, seeds, centroid, radius = normalize_view(lines, sample["seeds"][good], max_lines=27)
        record = {k: v for k, v in sample.items() if k != "seeds"}
        record.update(geometry=geometry, normalized_seeds=seeds, line_count=int(good.sum()),
                      centroid=centroid, radius=radius, bounds=np.stack((lo, hi)),
                      measured_mean_arc_length=float(np.linalg.norm(np.diff(lines, axis=1), axis=-1).sum(1).mean()))
        records.append(record)
    return records, rejected, stats


class RegularizedFMT(nn.Module):
    def __init__(self, dropout, normalized=True):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(322, 256), nn.LayerNorm(256) if normalized else nn.Identity(),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128),
            nn.LayerNorm(128) if normalized else nn.Identity(), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


class RegularizedConv3D(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.GELU(), nn.Dropout3d(dropout/2), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(), nn.Dropout3d(dropout/2), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(), nn.Dropout3d(dropout/2), nn.AdaptiveAvgPool3d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        return self.classifier(self.convolution(x)).squeeze(-1)


def make_model(method, candidate):
    if candidate["architecture"] == "frozen_3.1":
        return FMTBundleClassifier(candidate["dropout"]) if method == "fmt_mlp" else Conv3DClassifier(candidate["dropout"])
    return RegularizedFMT(candidate["dropout"]) if method == "fmt_mlp" else RegularizedConv3D(candidate["dropout"])


def transform_fmt(values, log_transform):
    values = np.asarray(values, np.float32)
    return np.sign(values)*np.log1p(np.abs(values)) if log_transform else values
