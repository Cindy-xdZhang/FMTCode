"""Nested FTLE grids, complete five-line integration, and blocked time splits."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import netCDF4
import numpy as np
import torch
from scipy.ndimage import binary_erosion, map_coordinates
from torch.nn import functional as F


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metadata(root, name):
    if name == 'doublegyre2d':
        return {'name': name, 'bounds': [0., 2., 0., 1.], 'tmin': 0., 'tmax': 10.,
                'source_dt': 0., 'source': 'PyFlowVis double_gyre_2D analytic formula',
                'parameters': {'eps': .25, 'A': .1, 'omega': .2 * math.pi}}
    path = Path(root) / (name + '.nc')
    with netCDF4.Dataset(path) as ds:
        axes = {}
        for axis in ('x', 'y', 't'):
            key = next(k for k in (axis, axis + 'dim', 'time' if axis == 't' else axis)
                       if k in ds.variables)
            values = np.asarray(ds[key][:], dtype=np.float64)
            if len(values) < 2 or np.any(np.diff(values) <= 0):
                raise ValueError(f'Invalid {key} coordinates')
            if not np.allclose(np.diff(values), np.diff(values).mean(), rtol=2e-4, atol=2e-6):
                raise ValueError('This adapter requires regular source axes')
            axes[axis] = (key, values)
        result = {'name': name, 'source': str(path), 'source_sha256': file_sha256(path),
                  'axis_names': {a: b[0] for a, b in axes.items()},
                  'bounds': [float(axes['x'][1][0]), float(axes['x'][1][-1]),
                             float(axes['y'][1][0]), float(axes['y'][1][-1])],
                  'tmin': float(axes['t'][1][0]), 'tmax': float(axes['t'][1][-1]),
                  'source_dt': float(np.max(np.diff(axes['t'][1])))}
        if 'radius' in ds.variables:
            result['obstacle'] = [float(ds['obstacle_pos_x'][0]) if 'obstacle_pos_x' in ds.variables else 0.,
                                  float(ds['obstacle_pos_y'][0]) if 'obstacle_pos_y' in ds.variables else 0.,
                                  float(ds['radius'][0])]
    return result


def time_splits(meta, counts, tau):
    span = meta['tmax'] - meta['tmin']
    lower = meta['tmin'] + .1 * span
    if 'cylinder' in meta['name']:
        lower = max(meta['tmin'] + .5 * span, 7.)
    usable = meta['tmax'] - lower
    b1, b2 = lower + .6 * usable, lower + .8 * usable
    guard = max(2.1 * meta['source_dt'], 1e-6)
    intervals = {'train': (lower, b1 - guard), 'validation': (b1 + guard, b2 - guard),
                 'test': (b2 + guard, meta['tmax'])}
    splits = {}
    for split, (left, right) in intervals.items():
        if right - left <= tau:
            raise ValueError(f'{meta["name"]} {split}: no complete integration window')
        splits[split] = np.linspace(left, right - tau, counts[split]).tolist()
    return splits


class Velocity:
    def __init__(self, meta, t0, tau, device):
        self.meta = meta
        self.device = device
        self.bounds = torch.tensor(meta['bounds'], device=device, dtype=torch.float64)
        if meta['name'] == 'doublegyre2d':
            return
        with netCDF4.Dataset(meta['source']) as ds:
            names = meta['axis_names']
            times = np.asarray(ds[names['t']][:], dtype=np.float64)
            first = max(0, np.searchsorted(times, t0, side='right') - 1)
            last = min(len(times), np.searchsorted(times, t0 + tau, side='left') + 1)
            self.tmin, self.tmax = float(times[first]), float(times[last - 1])
            if self.tmin > t0 + 1e-8 or self.tmax < t0 + tau - 1e-8:
                raise ValueError('Source does not bracket the whole integration interval')
            arrays = []
            for component in ('u', 'v'):
                var = ds[component]
                if tuple(var.dimensions) != tuple(names[a] for a in ('t', 'y', 'x')):
                    raise ValueError('Unexpected velocity dimension order')
                data = var[first:last]
                if np.ma.isMaskedArray(data) and np.ma.getmaskarray(data).any():
                    raise ValueError('Masked source velocities require an explicit source mask')
                arrays.append(np.asarray(data, dtype=np.float64))
            data = np.stack(arrays, axis=0)
            if not np.isfinite(data).all():
                raise ValueError('Nonfinite source velocity')
            self.field = torch.from_numpy(data[None]).to(device)

    def valid(self, xy):
        xmin, xmax, ymin, ymax = self.bounds
        ok = torch.isfinite(xy).all(-1) & (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax)
        ok &= (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
        if 'obstacle' in self.meta:
            x, y, radius = self.meta['obstacle']
            ok &= (xy[:, 0] - x).square() + (xy[:, 1] - y).square() > radius ** 2
        return ok

    def __call__(self, xy, t):
        if self.meta['name'] == 'doublegyre2d':
            x, y = xy.unbind(-1)
            a = .25 * math.sin(.2 * math.pi * t)
            f = a * x.square() + (1 - 2 * a) * x
            u = -.1 * math.pi * torch.sin(math.pi * f) * torch.cos(math.pi * y)
            v = .1 * math.pi * torch.cos(math.pi * f) * torch.sin(math.pi * y) * (2 * a * x + 1 - 2 * a)
            return torch.stack((u, v), dim=-1)
        xmin, xmax, ymin, ymax = self.bounds
        grid = torch.stack((2 * (xy[:, 0] - xmin) / (xmax - xmin) - 1,
                            2 * (xy[:, 1] - ymin) / (ymax - ymin) - 1,
                            torch.full_like(xy[:, 0], 2 * (t - self.tmin) / (self.tmax - self.tmin) - 1)), -1)
        # grid_sample with align_corners=True maps actual source nodes to themselves.
        return F.grid_sample(self.field, grid.view(1, 1, 1, -1, 3), mode='bilinear',
                             padding_mode='border', align_corners=True).view(2, -1).T


@torch.no_grad()
def integrate(velocity, seeds, t0, tau, samples=32, max_dt=.005, offset=.005):
    """RK4 with both endpoints and all intermediate stages checked; no early tails."""
    seeds = torch.as_tensor(seeds, dtype=torch.float64, device=velocity.device)
    offsets = seeds.new_tensor([[0, 0], [offset, 0], [-offset, 0], [0, offset], [0, -offset]])
    xy = (seeds[:, None] + offsets).reshape(-1, 2)
    valid = velocity.valid(xy)
    substeps = math.ceil(tau / (samples - 1) / max_dt)
    dt = tau / ((samples - 1) * substeps)
    history = [xy.clone()]
    for step in range((samples - 1) * substeps):
        t = t0 + step * dt
        a = velocity(xy, t)
        p2 = xy + .5 * dt * a
        b = velocity(p2, t + .5 * dt)
        p3 = xy + .5 * dt * b
        c = velocity(p3, t + .5 * dt)
        p4 = xy + dt * c
        d = velocity(p4, t + dt)
        xy = xy + (dt / 6) * (a + 2*b + 2*c + d)
        valid &= velocity.valid(p2) & velocity.valid(p3) & velocity.valid(p4) & velocity.valid(xy)
        if (step + 1) % substeps == 0:
            history.append(xy.clone())
    paths = torch.stack(history, dim=1).view(len(seeds), 5, samples, 2)
    return paths, valid.view(-1, 5).all(1), {'actual_dt': dt, 'steps': (samples - 1) * substeps,
                                                't0': t0, 't1': t0 + tau, 'tau': tau}


def ftle(paths, tau):
    """log(sigma_max(dPhi/dx))/abs(tau), without clipping contracting flows to zero."""
    if paths.ndim != 4 or paths.shape[1] != 5 or paths.shape[-1] != 2 or tau == 0:
        raise ValueError('FTLE requires [N,5,L,2] and a nonzero physical duration')
    first, last = paths[:, :, 0], paths[:, :, -1]
    a0 = torch.stack((first[:, 1] - first[:, 2], first[:, 3] - first[:, 4]), -1)
    a1 = torch.stack((last[:, 1] - last[:, 2], last[:, 3] - last[:, 4]), -1)
    jacobian = torch.linalg.solve(a0.transpose(-1, -2), a1.transpose(-1, -2)).transpose(-1, -2)
    largest = torch.linalg.svdvals(jacobian)[:, 0]
    if torch.any(largest <= 0) or not torch.isfinite(largest).all():
        raise ValueError('Degenerate FTLE differential')
    return largest.log() / abs(tau)


def interpolation(low, scale, order=1):
    """Node-coincident interpolation, output ((H-1)s+1,(W-1)s+1)."""
    yy, xx = np.meshgrid(np.arange((low.shape[0]-1)*scale+1) / scale,
                         np.arange((low.shape[1]-1)*scale+1) / scale, indexing='ij')
    return map_coordinates(low, [yy, xx], order=order, mode='nearest', prefilter=order > 1)


def evaluation_mask(valid_high, valid_low, scale):
    # Exclude two low-grid cells around absent trajectories, then require all
    # bilinear supporting sites and all seven-by-seven SSIM neighborhoods valid.
    safe_low = binary_erosion(valid_low, iterations=2, border_value=0)
    supported = interpolation(safe_low.astype(np.float64), scale, order=1) >= 1 - 1e-10
    return valid_high & supported
