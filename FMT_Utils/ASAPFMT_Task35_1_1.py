"""Frozen c156 representation/classifier applied to seven-line primitives."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from FMT_Utils.ASAPFrame_1_1 import asap_frame
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import FourierClassifier, fourier_tokens, normalize, resample

ARMS = ('raw_c156', 'fmt_c156', 'asap_fmt')


def classifier(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    model = FourierClassifier('p35', 'wide_residual', 'h0')
    if arm == 'raw_c156':
        # All later layers are identical. Three additional inputs cost 768 weights.
        model.line[0][0] = nn.Linear(144, 256)
    return model


@torch.no_grad()
def encode(geometry, times, arm, device='cpu'):
    if arm not in ARMS:
        raise ValueError(arm)
    x = asap_frame(geometry, times) if arm == 'asap_fmt' else np.asarray(geometry)
    x = torch.as_tensor(x, device=device, dtype=torch.float64)
    # Camera fitting always precedes per-line arc resampling, which loses
    # synchronized time. Resampling never feeds back into the camera fit.
    x = resample(x, 48, 'uniform')
    counts = torch.full((len(x),), 7, device=device, dtype=torch.long)
    if arm == 'raw_c156':
        normalized, _ = normalize(x, counts)
        values = normalized.flatten(-2)
        return torch.cat([values, torch.ones((*values.shape[:2], 1), device=device)], -1)
    # Each of the seven anchors has exactly six other lines. Their ordering
    # cannot affect p35's featurewise mean and numeric maximum.
    neighbors = torch.tensor([[j for j in range(7) if j != i] for i in range(7)], device=device)
    neighbors = neighbors[None].expand(len(x), -1, -1)
    return fourier_tokens(x, counts, neighbors, 'p35')


def normalizer(values):
    valid = np.asarray(values[..., :-1], dtype=np.float64).reshape(-1, values.shape[-1]-1)
    mean, std = valid.mean(0), valid.std(0)
    std[std < 1e-8] = 1.
    return dict(mean=mean.tolist(), std=std.tolist(), fit_tokens=len(valid), train_only=True)


def standardize(values, stats):
    result = np.asarray(values, dtype=np.float32).copy()
    result[..., :-1] = (result[..., :-1].astype(np.float64)-np.array(stats['mean']))/np.array(stats['std'])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite standardized features.')
    return result
