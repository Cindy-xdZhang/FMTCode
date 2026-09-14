"""Five matched feature combinations for existing seven-line pathline caches."""
import numpy as np
import torch

from FMT_Utils.FMT_Angles_3D import center_angle_fourier
from FMT_Utils.fmt_v8 import coordinate_tangent_spectrum, compose_features

WIDTHS = {'fmt_161':161,'fmt_direction_233':233,'fmt_v5_326':326,'fmt_v5_direction_398':398,'fmt_v8_100':100}


@torch.no_grad()
def shared_features(raw, original):
    """Copy the frozen 161D branch; build one common direction/angle extension.

    Only the new direction branch uses a per-primitive centroid and maximum
    radius over its 7x32 points. The original cached branch is never rescaled.
    """
    raw=np.asarray(raw,dtype=np.float32);original=np.asarray(original,dtype=np.float32)
    if raw.shape[1:]!=(7,32,3) or original.shape!=(len(raw),161):
        raise ValueError('Wrong raw/cache shape')
    output=np.empty((len(raw),398),np.float32);output[:,:161]=original
    for start in range(0,len(raw),1024):
        sl=slice(start,start+1024);x=torch.from_numpy(raw[sl]).double()
        centered=x-x.mean((1,2),keepdim=True)
        radius=centered.norm(dim=-1).amax((1,2),keepdim=True)[...,None]
        if torch.any(radius<=0) or not torch.isfinite(radius).all():
            raise ValueError('Degenerate primitive radius')
        center=(centered/radius).float()[:,0]
        output[sl,161:233]=coordinate_tangent_spectrum(center).numpy()
        output[sl,233:398]=center_angle_fourier(raw[sl],num_freq=6).astype(np.float32)
    if not np.isfinite(output).all():raise ValueError('Nonfinite geometry feature')
    return output


def select_features(shared, name):
    if name=='fmt_161':return shared[:,:161]
    if name=='fmt_direction_233':return shared[:,:233]
    if name=='fmt_v5_326':return np.concatenate((shared[:,:161],shared[:,233:398]),-1)
    if name=='fmt_v5_direction_398':return shared
    if name=='fmt_v8_100':
        result=compose_features(torch.from_numpy(shared[:,:161]),torch.from_numpy(shared[:,161:233])).numpy()
        return result
    raise ValueError(name)
