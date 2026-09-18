"""Task4-c descriptors with exactly one original physical seed as center."""
from __future__ import annotations

import numpy as np
import torch
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_P35_NormFrequency_3_1 import direction_spectrum
from FMT_Utils.FMTNoConvolution_1_1 import assert_no_fmt_convolution
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import (
    FourierClassifier, POOLS, pool_neighbors, resample, normalize)


def original_center_indices(seeds, metadata):
    """Match the actual seed; return -1 when absent, NEVER a substitute."""
    seeds=np.asarray(seeds,dtype=np.float64)
    center=(metadata['center']-metadata['centroid'])/metadata['radius'][:,None]
    distance=np.linalg.norm(seeds-center[:,None],axis=-1)
    distance[np.arange(seeds.shape[1])[None]>=metadata['counts'][:,None]]=np.inf
    ids=distance.argmin(1)
    ratio=distance[np.arange(len(ids)),ids]*metadata['radius']/metadata['neighbor_distance']
    # float32 cache error is <6e-6 of the neighbor spacing; other seeds are >=1.
    ids[ratio>=1e-4]=-1
    return ids,ratio


@torch.no_grad()
def select_neighbors(seeds,counts,center_ids,strategy):
    """Select six neighbors for only the declared original center.

    Preserve the frozen float32 pairwise-distance arithmetic and tie rule.
    Pairwise distances do not encode other centers or produce other tokens.
    """
    if strategy not in ('nearest6','fps6'):raise ValueError(strategy)
    b,n,_=seeds.shape;batch=torch.arange(b,device=seeds.device)
    if torch.any(center_ids<0) or torch.any(center_ids>=counts):
        raise ValueError('Missing original center; nearest-line substitution is forbidden')
    if torch.any(counts<7):raise ValueError('Six real neighbors are required')
    distances=torch.cdist(seeds,seeds)
    excluded=torch.arange(n,device=seeds.device)[None]>=counts[:,None]
    excluded[batch,center_ids]=True
    minimum=distances[batch,center_ids].clone()
    if strategy=='nearest6':
        return minimum.masked_fill(excluded,torch.inf).argsort(dim=-1,stable=True)[:,:6]
    selected=[]
    for _ in range(6):
        pick=minimum.masked_fill(excluded,-torch.inf).max(-1).indices
        selected.append(pick);excluded[batch,pick]=True
        minimum=torch.minimum(minimum,distances[batch,pick])
    return torch.stack(selected,-1)


@torch.no_grad()
def encode(geometry,seeds,counts,center_ids,method):
    """Exactly one [141 features + validity] token per original sample."""
    if method not in ('p35_h0','c156'):raise ValueError(method)
    if geometry.shape[2:]!=(32,3):raise ValueError('Frozen 32-point source required')
    strategy='nearest6' if method=='p35_h0' else 'fps6'
    neighbors=select_neighbors(seeds,counts,center_ids,strategy)
    if method=='c156':
        geometry,_=normalize(resample(geometry,48,'uniform'),counts)
        scale,weight=1.,1.
    else:
        # Original p35 uses the saved whole-bundle normalization, without refitting.
        scale,weight=100.,.5
    indices=torch.cat((center_ids[:,None],neighbors),-1)
    batch=torch.arange(len(geometry),device=geometry.device)
    primitive=geometry[batch[:,None],indices]
    base=pathline_dft_features_3d(primitive,num_freq=6,neighbor_scale=scale,
        neighbor_weight=weight,neighbor_pool='sort',mode='gram',include_chirality=True,return_numpy=False)
    features=torch.cat((base[:,:23],direction_spectrum(primitive[:,0],6),
                        pool_neighbors(base[:,23:],POOLS['p35'])),-1)
    if features.shape!=(len(geometry),141) or not torch.isfinite(features).all():
        raise ValueError('Invalid single-original-center features')
    tokens=torch.cat((features,torch.ones_like(features[:,:1])),-1)[:,None]
    return tokens,neighbors


def make_model(method):
    if method not in ('p35_h0','c156'):raise ValueError(method)
    model=FourierClassifier('p35','original' if method=='p35_h0' else 'wide_residual','h0')
    assert_no_fmt_convolution(model)
    expected=76738 if method=='p35_h0' else 992386
    assert sum(p.numel() for p in model.parameters())==expected
    def require_one_token(module,args):
        if args[0].ndim!=3 or args[0].shape[1:]!=(1,142):
            raise ValueError('Exactly one original-center token per sample is required')
    model.register_forward_pre_hook(require_one_token)
    return model
