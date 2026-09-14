"""FMT v8.1: center 23 + coordinate/tangent 72 + signed neighbor top4/mean 5."""
from __future__ import annotations

import torch

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d

VERSION = '8.1'
FEATURE_DIM = 100


def pool_neighbor_features(neighbors):
    """Pool all 138 entries before feature standardization, retaining signs.

    The first four outputs are selected by descending absolute value; exact
    magnitude ties retain the original column order. The last output is the
    arithmetic mean of all signed entries, including structural zero columns.
    """
    if neighbors.shape[-1] != 138:
        raise ValueError('Expected exactly 138 neighbor coefficients')
    order = torch.argsort(neighbors.abs(), dim=-1, descending=True, stable=True)[..., :4]
    return torch.cat((neighbors.gather(-1, order), neighbors.mean(-1, keepdim=True)), -1)


def coordinate_tangent_spectrum(center):
    """Frozen 4.14 signed spectrum on caller-supplied, normalized geometry."""
    if center.shape[-2:] != (32, 3):
        raise ValueError('Expected 32 three-dimensional center samples')
    delta = center.diff(dim=-2)
    tangent = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tangent = torch.cat((tangent, tangent[..., -1:, :]), -2)
    seq = torch.cat((center, tangent), -1)
    spectrum = torch.fft.rfft(seq, dim=-2, norm='ortho')[..., :6, :]
    return torch.view_as_real(spectrum).flatten(-3)


def compose_features(original, direction):
    """Keep V7's first 95 columns, then append four signed extrema and mean."""
    if original.shape[-1] != 161 or direction.shape != (*original.shape[:-1], 72):
        raise ValueError('Expected corresponding original 161D and direction 72D blocks')
    result = torch.cat((original[..., :23], direction, pool_neighbor_features(original[..., 23:])), -1)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite V8 coefficient')
    return result


@torch.no_grad()
def from_task4c_cache(tokens):
    """Convert frozen [B,N,234] tokens into [B,N,101], preserving the mask."""
    if tokens.ndim != 3 or tokens.shape[-1] != 234:
        raise ValueError('Expected frozen Task4-c 4.14 tokens including mask')
    return torch.cat((compose_features(tokens[..., :161], tokens[..., 161:233]), tokens[..., -1:]), -1)


@torch.no_grad()
def fmt_v8(pathlines, *, neighbor_scale=100., neighbor_weight=.5,
           original_features=None, direction_geometry=None, return_numpy=True):
    """Encode [B,7,32,3] primitives into 100 fixed, non-trainable coefficients.

    Geometry normalization belongs to the versioned task adapter. By default,
    the supplied center coordinates feed the direction spectrum unchanged.
    An exact original feature cache and separately normalized direction
    geometry may be supplied explicitly; neither uses labels or batch means.
    """
    xyz = torch.as_tensor(pathlines)
    if xyz.ndim != 4 or xyz.shape[1:] != (7, 32, 3):
        raise ValueError('Expected seven lines, 32 samples, three coordinates')
    if not xyz.is_floating_point():
        xyz = xyz.float()
    original = pathline_dft_features_3d(xyz, num_freq=6, neighbor_scale=neighbor_scale,
        neighbor_weight=neighbor_weight, return_numpy=False) if original_features is None else torch.as_tensor(original_features, device=xyz.device)
    center = xyz[:, 0] if direction_geometry is None else torch.as_tensor(direction_geometry, device=xyz.device)
    result = compose_features(original, coordinate_tangent_spectrum(center))
    return result.cpu().numpy() if return_numpy else result
