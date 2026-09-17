"""One fixed center, P35 Fourier descriptors, and one vector per primitive.

No learned feature extraction, center cycling, Raw branch, or convolution.
The signed coordinate/tangent block is retained from P35; this descriptor is
not claimed to be invariant under arbitrary time-dependent rigid observers.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from FMT_Utils.DFT_FMT_3D import dft_rotation_invariants_3d
from FMT_Utils.FMT_P35_NormFrequency_3_1 import direction_spectrum
from FMT_Utils.FMTNoConvolution_1_1 import assert_no_fmt_convolution


def fixed_center_ids(seeds, counts, original_center):
    """Select once on CPU/float64, using stored order to break exact ties."""
    seeds = np.asarray(seeds, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    center = np.asarray(original_center, dtype=np.float64)
    valid = np.arange(seeds.shape[1])[None] < counts[:, None]
    distance = np.linalg.norm(seeds - center[:, None], axis=-1)
    distance[~valid] = np.inf
    selected = distance.argmin(1)
    if not np.isfinite(distance[np.arange(len(seeds)), selected]).all():
        raise ValueError('No valid fixed center')
    return selected, distance[np.arange(len(seeds)), selected]


@torch.no_grad()
def encode(geometry, counts=None, center_ids=None, frequencies=6):
    """All valid non-center curves contribute once to mean/max pooling.

    Input [B,L,T,3]; output [B,24*K-3]. The number of lines and points may
    vary across tasks; no center is ever changed during feature extraction.
    All geometry and Fourier arithmetic is float64, output is float32.
    """
    x = torch.as_tensor(geometry).double()
    if x.ndim != 4 or x.shape[-1] != 3 or x.shape[2] < 2 * frequencies:
        raise ValueError('Expected [B,L,T,3] with enough points for K bins')
    batch, lines, points, _ = x.shape
    counts = torch.full((batch,), lines, device=x.device, dtype=torch.long) if counts is None else torch.as_tensor(counts, device=x.device, dtype=torch.long)
    center_ids = torch.zeros(batch, device=x.device, dtype=torch.long) if center_ids is None else torch.as_tensor(center_ids, device=x.device, dtype=torch.long)
    if counts.shape != (batch,) or center_ids.shape != (batch,) or torch.any(counts < 2) or torch.any(counts > lines) or torch.any(center_ids < 0) or torch.any(center_ids >= counts):
        raise ValueError('Invalid counts or fixed center indices')
    mask = torch.arange(lines, device=x.device)[None] < counts[:, None]
    # Mask with where so arbitrary padding, including NaN, cannot affect a sample.
    x = torch.where(mask[:, :, None, None], x, 0.)
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite valid geometry')
    centroid = x.sum((1, 2), keepdim=True) / (counts[:, None, None, None] * points)
    x = torch.where(mask[:, :, None, None], x - centroid, 0.)
    radius = x.norm(dim=-1).amax((1, 2))
    if torch.any(radius <= 1e-12):
        raise ValueError('Degenerate primitive')
    x = x / radius[:, None, None, None]
    center = x[torch.arange(batch, device=x.device), center_ids]
    center_features = dft_rotation_invariants_3d(center.diff(dim=1), frequencies)
    relative_delta = (x - center[:, None]).diff(dim=2)
    neighbors = dft_rotation_invariants_3d(relative_delta.reshape(-1, points - 1, 3), frequencies).reshape(batch, lines, -1)
    neighbor_mask = mask & (torch.arange(lines, device=x.device)[None] != center_ids[:, None])
    mean = torch.where(neighbor_mask[:, :, None], neighbors, 0.).sum(1) / (counts - 1)[:, None]
    maximum = neighbors.masked_fill(~neighbor_mask[:, :, None], -torch.inf).amax(1)
    result = torch.cat((center_features, direction_spectrum(center, frequencies), mean, maximum), dim=-1).float()
    if result.shape != (batch, 24 * frequencies - 3) or not torch.isfinite(result).all():
        raise ValueError('Invalid single-center features')
    return result


class SingleCenterMLP(nn.Module):
    """A plain multilayer perceptron consuming exactly one feature vector."""

    def __init__(self, dimensions=141, dropout=.15):
        super().__init__()
        self.dimensions = dimensions
        layers = []
        widths = (dimensions, 256, 128, 64)
        for incoming, outgoing in zip(widths[:-1], widths[1:]):
            layers.extend((nn.Linear(incoming, outgoing), nn.LayerNorm(outgoing), nn.GELU(), nn.Dropout(dropout)))
        layers.append(nn.Linear(widths[-1], 2))
        self.network = nn.Sequential(*layers)
        assert_no_fmt_convolution(self)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.dimensions:
            raise ValueError('One feature vector per primitive is required')
        return self.network(features)
