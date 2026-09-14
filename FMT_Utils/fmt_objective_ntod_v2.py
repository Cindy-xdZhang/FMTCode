"""Distance-and-angle extension 2.2 under the user-requested method name.

The existing objective_fmt_nTDO_v2.py (distance-only 2.1) stays frozen.
Only scalar distances and center angles enter Fourier here. No temporal
coordinate differences, vector Fourier branch or kinematic block is added.
"""
import torch

from FMT_Utils.FMT_Angles_3D import center_angle_fourier
from FMT_Utils.objective_fmt_nTDO_v2 import objective_fmt_nTDO_v2

VERSION = '2.2'


def fmt_objective_ntod_v2(pathlines, num_freq=6, *, return_blocks=False, return_numpy=True):
    """Return (21+15)*(2F-1) coordinates; F=6 gives 396.

    Objective scalar input/output assumes corresponding material identities
    and time samples. It does not imply invariance to reseeding the primitive.
    """
    distances = objective_fmt_nTDO_v2(pathlines, num_freq, return_numpy=False)
    angles = center_angle_fourier(pathlines, num_freq, return_numpy=False).to(distances.dtype)
    value = torch.cat((distances, angles), dim=1)
    blocks = {'features': value, 'distance_fourier': distances, 'angle_fourier': angles}
    if return_numpy:
        blocks = {key: item.detach().cpu().numpy() for key, item in blocks.items()}
    return blocks if return_blocks else blocks['features']
