"""P35 3.1: common sample geometry normalization and configurable Fourier bins."""
from __future__ import annotations

import numpy as np
import torch

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d


FREQUENCIES = (2, 4, 6, 10, 16)
RECIPES = (
    dict(id='n0', geometry='max_radius', features='zscore'),
    dict(id='n1', geometry='max_radius', features='signed_log_zscore_clip8'),
    dict(id='n2', geometry='rms_radius', features='zscore'),
    dict(id='n3', geometry='rms_radius', features='signed_log_zscore_clip8'),
)


def candidates():
    return [dict(recipe, id=f"{recipe['id']}_k{k:02d}", recipe=recipe['id'],
                 frequencies=k, feature_dimensions=24*k-3)
            for recipe in RECIPES for k in FREQUENCIES]


def normalize_geometry(geometry, counts, kind):
    """One centroid/scale over every valid point in the complete input sample."""
    x=torch.as_tensor(geometry).double()
    counts=torch.as_tensor(counts,device=x.device,dtype=torch.long)
    if x.ndim!=4 or x.shape[2:]!=(32,3) or torch.any(counts<7) or torch.any(counts>x.shape[1]):
        raise ValueError('Expected valid 32-point input lines')
    mask=torch.arange(x.shape[1],device=x.device)[None]<counts[:,None]
    weights=mask[:,:,None,None]
    center=(x*weights).sum((1,2),keepdim=True)/(counts[:,None,None,None]*32)
    x=(x-center)*weights
    radius2=x.square().sum(-1)
    if kind=='max_radius':scale=radius2.amax((1,2)).sqrt()
    elif kind=='rms_radius':scale=(radius2.sum((1,2))/(counts*32)).sqrt()
    else:raise ValueError(kind)
    if not torch.isfinite(x).all() or not torch.all(scale>1e-12):
        raise ValueError('Nonfinite or degenerate sample geometry')
    return (x/scale[:,None,None,None]).float(),mask


def direction_spectrum(center, frequencies):
    delta=center.diff(dim=-2)
    tangent=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    tangent=torch.cat((tangent,tangent[...,-1:,:]),dim=-2)
    sequence=torch.cat((center,tangent),dim=-1)
    return torch.view_as_real(torch.fft.rfft(sequence,dim=-2,norm='ortho')[...,:frequencies,:]).flatten(-3)


def primitive_features(primitive, frequencies, neighbor_scale=1., neighbor_weight=1.):
    """Center, signed direction, neighbor descriptor means and numeric maxima."""
    d=4*frequencies-1
    base=pathline_dft_features_3d(primitive,num_freq=frequencies,neighbor_scale=neighbor_scale,
        neighbor_weight=neighbor_weight,neighbor_pool='sort',mode='gram',include_chirality=True,return_numpy=False)
    neighbors=base[:,d:].reshape(-1,6,d)
    result=torch.cat((base[:,:d],direction_spectrum(primitive[:,0],frequencies),
                      neighbors.mean(1),neighbors.amax(1)),-1)
    if result.shape[-1]!=24*frequencies-3 or not torch.isfinite(result).all():
        raise ValueError('Invalid P35 features')
    return result


@torch.no_grad()
def encode(geometry, definition, seeds=None, counts=None):
    g=torch.as_tensor(geometry)
    if counts is None:counts=torch.full((len(g),),7,device=g.device,dtype=torch.long)
    x,mask=normalize_geometry(g,counts,definition['geometry'])
    if seeds is None:
        if x.shape[1]!=7:raise ValueError('Task3/5 requires seven lines')
        return primitive_features(x,definition['frequencies'])
    # Neighbor selection uses frozen seeds; a common affine rescaling preserves distance order.
    s=torch.as_tensor(seeds,device=g.device)
    distances=torch.cdist(s,s)
    distances.masked_fill_(~mask[:,None,:],torch.inf)
    distances.diagonal(dim1=1,dim2=2).fill_(torch.inf)
    nearest=torch.argsort(distances,dim=-1,stable=True)[...,:6]
    central=torch.arange(x.shape[1],device=g.device)[None,:,None].expand(len(g),-1,-1)
    ids=torch.cat((central,nearest),-1)
    bi,li=mask.nonzero(as_tuple=True)
    result=x.new_zeros((len(g),x.shape[1],definition['feature_dimensions']))
    for start in range(0,len(bi),1024):
        bs,ls=bi[start:start+1024],li[start:start+1024]
        primitive=x[bs[:,None],ids[bs,ls]]
        result[bs,ls]=primitive_features(primitive,definition['frequencies'])
    return result


def pretransform(x, kind):
    x=np.asarray(x,dtype=np.float32)
    if kind=='signed_log_zscore_clip8':return np.sign(x)*np.log1p(np.abs(x))
    if kind=='zscore':return x
    raise ValueError(kind)


def fit_normalizer(training, mask, kind):
    """Fit only training tokens; zero/near-constant dimensions remain bounded."""
    x=pretransform(training,kind)
    valid=x.reshape(-1,x.shape[-1]) if mask is None else x[np.asarray(mask)[...,0]>.5]
    mean=valid.mean(0,dtype=np.float64)
    std=valid.std(0,dtype=np.float64)
    std[std<1e-8]=1.
    return dict(kind=kind,mean=mean,std=std,fit_tokens=len(valid),constant_rule='std_below_1e-8_replaced_by_1')


def apply_normalizer(x, state, mask=None):
    values=pretransform(x,state['kind'])
    values=((values-np.asarray(state['mean']))/np.asarray(state['std'])).astype(np.float32)
    if state['kind']=='signed_log_zscore_clip8':values=np.clip(values,-8,8)
    if mask is not None:values*=np.asarray(mask)
    if not np.isfinite(values).all():raise ValueError('Nonfinite normalized features')
    return values
