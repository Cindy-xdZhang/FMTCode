"""FMT v5.1: frozen original FMT plus 15 same-time center-angle spectra.

The original branch retains center motion and neighbor-vector temporal
differences. Adding objective scalar angles does not make the full output
invariant under arbitrary time-dependent rigid observers.
"""
import torch

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_Angles_3D import center_angle_fourier

VERSION = '5.1'


def fmt_v5(pathlines, num_freq=6, *, neighbor_scale=100.0, neighbor_weight=0.5,
           return_blocks=False, return_numpy=True):
    """Return 7*(4F-1) + 15*(2F-1) features; F=6 gives 326.

    The original 161 coordinates are unchanged, including neighbor_scale=100,
    neighbor_weight=0.5, sorting and chirality. The added 165 coordinates are
    computed from angles before Fourier, rather than angles between spectra.
    """
    angles = center_angle_fourier(pathlines, num_freq, return_numpy=False)
    original = pathline_dft_features_3d(pathlines, num_freq=num_freq,
                                      neighbor_scale=neighbor_scale,
                                      neighbor_weight=neighbor_weight, return_numpy=False)
    value = torch.cat((original, angles.to(original.dtype)), dim=1)
    blocks = {'features': value, 'original_fourier': original,
              'angle_fourier': angles.to(original.dtype)}
    if return_numpy:
        blocks = {key: item.detach().cpu().numpy() for key, item in blocks.items()}
    return blocks if return_blocks else blocks['features']
