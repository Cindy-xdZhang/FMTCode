"""Task6 2.1: seven material trajectories -> frozen token -> VAE -> geometry."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d


def fmt_tokens(geometry):
    x = np.asarray(geometry, dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (7, 32, 3) or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,7,32,3] local geometry")
    with torch.no_grad():
        out = pathline_dft_features_3d(x, num_freq=6, neighbor_weight=.5,
            neighbor_scale=100., neighbor_pool="sort", mode="gram",
            include_chirality=True, return_numpy=True)
    out = np.asarray(out, np.float32)
    if out.shape != (len(x), 161) or not np.isfinite(out).all():
        raise ValueError("Invalid frozen FMT output")
    return out


def resample_time(paths, samples=32):
    """Linear resampling of fixed-clock integration output, retaining both ends."""
    loc = np.linspace(0, paths.shape[-2] - 1, samples)
    lo = np.floor(loc).astype(int)
    hi = np.minimum(lo + 1, paths.shape[-2] - 1)
    w = (loc - lo)[:, None]
    return paths[..., lo, :] * (1 - w) + paths[..., hi, :] * w


def time_plan(times, available, original_range, cylinder=False, horizon_frames=12):
    """Chronological source blocks, with at least two complete seed times per split.

    Boundaries use source availability only. Entire interpolation-frame sets are
    disjoint; windows within a split may overlap. No geometry/labels are inspected.
    """
    t = np.asarray(times, float)
    dt = np.median(np.diff(t))
    if len(t) < 2 or dt <= 0 or not np.allclose(np.diff(t), dt, rtol=2e-4, atol=1e-7):
        raise ValueError("Need uniform increasing source time")
    tmin, tmax = original_range
    lower = max(7., tmin + .5 * (tmax - tmin)) if cylinder else tmin + .1 * (tmax - tmin)
    upper = tmin + .8 * (tmax - tmin)
    available = set(map(int, available))
    h = int(np.ceil(horizon_frames))
    starts = [i for i in range(len(t) - h) if lower - 1e-6 <= t[i] <= upper + 1e-6
              and set(range(i, i + h + 1)) <= available]
    if not starts:
        raise ValueError("No complete source windows inside seed-time policy")
    first, end = starts[0], starts[-1] + h
    choices = []
    for validation_start in starts:
        train = [i for i in starts if i + h < validation_start]
        if len(train) < 2:
            continue
        for test_start in starts:
            validation = [i for i in starts if i >= validation_start and i + h < test_start]
            test = [i for i in starts if i >= test_start]
            if len(validation) < 2 or len(test) < 2:
                continue
            cost = abs(validation_start - (first + .60 * (end - first + 1)))
            cost += abs(test_start - (first + .80 * (end - first + 1)))
            choices.append((cost, validation_start, test_start, train, validation, test))
    if not choices:
        raise ValueError("Insufficient disjoint complete time windows for three splits")
    _, v, q, train, validation, test = min(choices, key=lambda row: row[:3])
    return dict(original_time_range=list(map(float, original_range)),
                seed_time_bounds=[float(lower), float(upper)], max_horizon_frames=h,
                time_step=float(dt), source_times=t.tolist(),
                boundaries={"train": [first, v - 1], "validation": [v, q - 1], "test": [q, end]},
                starts={"train": train, "validation": validation, "test": test})


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.SiLU(), nn.Linear(width, width),
                                    nn.SiLU(), nn.Linear(width, width))

    def forward(self, x):
        return x + .1 * self.layers(x)


class PrimitiveVAE(nn.Module):
    def __init__(self, input_dim, width=512, latent_dim=64, blocks=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, width),
            *[ResidualBlock(width) for _ in range(blocks)], nn.SiLU())
        self.mean = nn.Linear(width, latent_dim)
        self.log_variance = nn.Linear(width, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, width),
            *[ResidualBlock(width) for _ in range(blocks)], nn.SiLU(), nn.Linear(width, 672))

    def forward(self, x, sample=True):
        hidden = self.encoder(x)
        mean = self.mean(hidden)
        logvar = self.log_variance(hidden).clamp(-20., 10.)
        z = mean + torch.randn_like(mean) * torch.exp(.5 * logvar) if sample else mean
        return self.decoder(z).reshape(-1, 7, 32, 3), mean, logvar


def vae_loss(prediction, target, mean, logvar, beta):
    if beta <= 0:
        raise ValueError("Task6 2.1 requires a positive VAE prior weight")
    reconstruction = (prediction - target).square().sum(-1).mean()
    kl = .5 * (mean.square() + logvar.exp() - 1. - logvar).mean()
    return reconstruction + beta * kl, reconstruction, kl


def geometry_metrics(prediction, truth):
    """Corresponding Euclidean positions, already expressed in initial-radius units."""
    p, y = np.asarray(prediction, float), np.asarray(truth, float)
    if p.shape != y.shape or p.shape[1:] != (7, 32, 3) or not np.isfinite(p).all():
        raise ValueError("Invalid geometry prediction")
    e2 = np.sum((p - y) ** 2, -1)
    i, j = np.triu_indices(7, 1)
    pair = np.linalg.norm(p[:, i] - p[:, j], axis=-1) - np.linalg.norm(y[:, i] - y[:, j], axis=-1)
    return dict(position_rmse_r=float(np.sqrt(e2[:, :, 1:].mean())),
        all_time_rmse_r=float(np.sqrt(e2.mean())), initial_rmse_r=float(np.sqrt(e2[:, :, 0].mean())),
        center_rmse_r=float(np.sqrt(e2[:, 0, 1:].mean())),
        neighbor_rmse_r=float(np.sqrt(e2[:, 1:, 1:].mean())),
        endpoint_rmse_r=float(np.sqrt(e2[:, :, -1].mean())),
        pair_distance_rmse_r=float(np.sqrt((pair[:, :, 1:] ** 2).mean())),
        time_rmse_r=np.sqrt(e2.mean((0, 1))).tolist(), samples=len(y))
