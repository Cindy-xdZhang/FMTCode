"""Scalar-distance-only objective_fmt_nTDO_v2, method version 2.1.

All 21 same-time Euclidean point-pair distances enter the temporal Fourier
transform. No relative-vector Fourier branch or temporal differencing is used.
"""
from __future__ import annotations

from numbers import Integral

import torch

VERSION = '2.1'


def _coordinates(pathlines):
    x = torch.as_tensor(pathlines)
    if x.ndim != 4 or x.shape[1] != 7 or x.shape[-1] not in (3, 4):
        raise ValueError('Expected [N,7,L,3 or 4], with center at line 0.')
    if x.shape[0] < 1 or x.shape[2] < 2:
        raise ValueError('Require at least one primitive and two time samples.')
    if x.is_complex() or x.dtype == torch.bool:
        raise ValueError('Coordinates must be real numbers.')
    if x.dtype not in (torch.float32, torch.float64):
        x = x.to(torch.float64)
    xyz = x[..., :3]
    if not torch.isfinite(xyz).all():
        raise ValueError('Coordinates must be finite; samples are not filtered.')
    return xyz


def build_fourier_inputs(pathlines):
    """Return distances [N,21,L] and lexicographic pair_indices [2,21].

    Six center-neighbor and fifteen neighbor-neighbor distances are included.
    Each subtraction uses points at the same time; material point identities
    and time samples are held fixed. Only the scalar distances enter rFFT.
    """
    xyz = _coordinates(pathlines)
    pairs = torch.triu_indices(7, 7, offset=1, device=xyz.device)
    distances = torch.linalg.vector_norm(xyz[:, pairs[1]] - xyz[:, pairs[0]], dim=-1)
    return {'distances': distances, 'pair_indices': pairs}


def objective_fmt_nTDO_v2(pathlines, num_freq=6, *, return_blocks=False, return_numpy=True):
    """Encode distances as 21*(2F-1) real features; F=6 gives 231.

    Keep real coefficients of bins 0..F-1, then imaginary coefficients of
    bins 1..F-1, flattened in pair-major order within each block. Distances
    are unsquared; DC is retained. No normalization, initial-time subtraction,
    temporal difference, kinematic or IVD feature is added. Frequencies are in
    cycles per uniformly sampled window. Tensor output preserves autograd,
    dtype and device; default output is NumPy.
    """
    distances = build_fourier_inputs(pathlines)['distances']
    if (isinstance(num_freq, bool) or not isinstance(num_freq, Integral)
            or not 2 <= num_freq <= distances.shape[2] // 2 + 1):
        raise ValueError('num_freq must be an integer in [2, L//2+1].')
    spectrum = torch.fft.rfft(distances, dim=2)[:, :, :num_freq]
    features = torch.cat((spectrum.real.flatten(1), spectrum.imag[:, :, 1:].flatten(1)), dim=1)
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite Fourier features.')
    blocks = {'features': features, 'distance_fourier': features}
    if return_numpy:
        blocks = {key: value.detach().cpu().numpy() for key, value in blocks.items()}
    return blocks if return_blocks else blocks['features']


__all__ = ['VERSION', 'build_fourier_inputs', 'objective_fmt_nTDO_v2']
