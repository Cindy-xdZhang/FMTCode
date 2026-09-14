"""Same-time center-vertex angles followed by a scalar Fourier transform.

The six material neighbors define 15 unordered pairs. Their identities and
sample times must remain fixed under a change of observer. This module never
differences coordinates or vectors across time.
"""
from __future__ import annotations

from numbers import Integral

import torch

from FMT_Utils.objective_fmt_nTDO_v2 import _coordinates


def center_angle_inputs(pathlines):
    """Return angles [N,15,L] in radians and material pair indices [2,15].

    atan2(||u_i cross u_j||, u_i dot u_j) is the angle in [0, pi].
    Unit vectors avoid scale-dependent division cutoffs. Coincident center
    and neighbor points have undefined angles and raise; no rows are dropped.
    Geometry is computed in float64 even when stored positions are float32.
    """
    xyz = _coordinates(pathlines).to(torch.float64)
    relative = xyz[:, 1:] - xyz[:, :1]
    radii = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    if torch.any(radii == 0):
        raise ValueError('Undefined center angle: a neighbor coincides with the center.')
    unit = relative / radii
    pairs = torch.triu_indices(6, 6, offset=1, device=xyz.device)
    left, right = unit[:, pairs[0]], unit[:, pairs[1]]
    sine = torch.linalg.vector_norm(torch.cross(left, right, dim=-1), dim=-1)
    cosine = (left * right).sum(dim=-1)
    angles = torch.atan2(sine, cosine)
    if not torch.isfinite(angles).all():
        raise FloatingPointError('Nonfinite center angles.')
    return {'angles': angles, 'pair_indices': pairs + 1}


def center_angle_fourier(pathlines, num_freq=6, *, return_numpy=True):
    """Keep real bins 0..F-1 and imaginary bins 1..F-1: 15*(2F-1).

    No time difference, normalization, phase removal or sorting is applied.
    Coefficients describe the stored sample-index window, not physical hertz.
    """
    angles = center_angle_inputs(pathlines)['angles']
    if (isinstance(num_freq, bool) or not isinstance(num_freq, Integral)
            or not 2 <= num_freq <= angles.shape[2] // 2 + 1):
        raise ValueError('num_freq must be an integer in [2, L//2+1].')
    spectrum = torch.fft.rfft(angles, dim=2)[:, :, :num_freq]
    value = torch.cat((spectrum.real.flatten(1), spectrum.imag[:, :, 1:].flatten(1)), dim=1)
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite angle Fourier features.')
    return value.detach().cpu().numpy() if return_numpy else value
