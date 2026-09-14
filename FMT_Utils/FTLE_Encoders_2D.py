"""Five-line adapters for Other_FTLEUpsampling2D_1.1; frozen 3D code is untouched.

Coordinates are [N,5,L,2] or [N,5,L,3=(x,y,t)]. Time is never a spatial axis.
FMT uses the existing two-dimensional complex Fourier descriptor (both signs).
"""
from numbers import Integral

import torch

from FMT_Utils.DCT_FMT_encoder import DCT_FMT

VERSION = '1.1'
LINE_ORDER = ('center', 'x+', 'x-', 'y+', 'y-')


def coordinates(pathlines):
    x = torch.as_tensor(pathlines)
    if x.ndim != 4 or x.shape[1] != 5 or x.shape[-1] not in (2, 3):
        raise ValueError('Require [N,5,L,2 or 3], center,x+,x-,y+,y-; never seven lines.')
    if x.shape[0] < 1 or x.shape[2] < 3 or x.is_complex() or x.dtype == torch.bool:
        raise ValueError('Require nonempty real trajectories with at least three times.')
    xy = x[..., :2].to(torch.float64)
    if not torch.isfinite(xy).all():
        raise ValueError('Nonfinite geometry; filter using the common data mask first.')
    return xy


def scalar_inputs(pathlines):
    xy = coordinates(pathlines)
    pairs = torch.triu_indices(5, 5, offset=1, device=xy.device)
    distances = torch.linalg.vector_norm(xy[:, pairs[1]] - xy[:, pairs[0]], dim=-1)
    relative = xy[:, 1:] - xy[:, :1]
    radii = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    if torch.any(radii == 0):
        raise ValueError('A neighbor coincides with the center; angle is undefined.')
    unit = relative / radii
    angles_idx = torch.triu_indices(4, 4, offset=1, device=xy.device)
    left, right = unit[:, angles_idx[0]], unit[:, angles_idx[1]]
    sine = (left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]).abs()
    angles = torch.atan2(sine, (left * right).sum(-1))
    return {'distances': distances, 'angles': angles, 'distance_pairs': pairs,
            'angle_pairs': angles_idx + 1}


def scalar_fourier(series, bins=6):
    if isinstance(bins, bool) or not isinstance(bins, Integral) or not 2 <= bins <= series.shape[-1] // 2 + 1:
        raise ValueError('bins must be an integer in [2,L//2+1].')
    z = torch.fft.rfft(series, dim=-1)[..., :bins]
    return torch.cat((z.real.flatten(1), z.imag[..., 1:].flatten(1)), dim=1)


def encode(pathlines, name, bins=6):
    """FMT=65, v5=131, objective 2.2=176, Raw=320 at L=32 and bins=6.

    The existing DCT_FMT max+mean pooling for a one-primitive window is retained
    exactly, including its factor of two. No spatial pooling between seed sites.
    """
    xy = coordinates(pathlines)
    if name == 'raw':
        center = xy[:, :1] - xy[:, :1, :1]
        relative = xy[:, 1:] - xy[:, :1]
        return torch.cat((center, relative), dim=1).flatten(1).float()
    if name not in ('fmt', 'fmt_v5', 'fmt_objective_ntod_v2'):
        raise ValueError(name)
    if name in ('fmt', 'fmt_v5'):
        if not 1 <= bins <= (xy.shape[2] - 2) // 2:
            raise ValueError('Not enough samples for distinct positive/negative Fourier bins.')
        original = DCT_FMT(5, xy.shape[2], dct_k=bins, dct_weight=0.5,
                           neighbor_diff_scale=100.0)(xy[:, None])
        if name == 'fmt':
            return original
    values = scalar_inputs(xy)
    angles = scalar_fourier(values['angles'], bins).float()
    if name == 'fmt_v5':
        return torch.cat((original, angles), dim=1)
    return torch.cat((scalar_fourier(values['distances'], bins).float(), angles), dim=1)
