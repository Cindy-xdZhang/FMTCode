"""Original-center FMT with 16 FPS neighbors; unchanged 141-feature heads."""
import torch
from FMT_Utils import Task4C_OriginalCenter_1_1 as original


@torch.no_grad()
def fps_neighbors(seeds,counts,centers,k=16):
    if k not in (6,16):raise ValueError('Only frozen controls and requested FPS16 are allowed')
    if torch.any(centers<0) or torch.any(centers>=counts):raise ValueError('Original center is required')
    if torch.any(counts<k+1):raise ValueError('Not enough distinct valid neighbors')
    b,n,_=seeds.shape;batch=torch.arange(b,device=seeds.device)
    distance=torch.cdist(seeds,seeds)
    excluded=torch.arange(n,device=seeds.device)[None]>=counts[:,None]
    excluded[batch,centers]=True
    minimum=distance[batch,centers].clone();selected=[]
    for _ in range(k):
        pick=minimum.masked_fill(excluded,-torch.inf).max(-1).indices
        selected.append(pick);excluded[batch,pick]=True
        minimum=torch.minimum(minimum,distance[batch,pick])
    return torch.stack(selected,-1)


@torch.no_grad()
def encode(geometry,seeds,counts,centers,candidate):
    name=candidate['base_method'];k=candidate['neighbors']
    if k==6:return original.encode(geometry,seeds,counts,centers,name)
    if k!=16:raise ValueError(k)
    if geometry.shape[2:]!=(32,3):raise ValueError('Frozen 32-point physical cache required')
    neighbors=fps_neighbors(seeds,counts,centers,16)
    if name=='c156':
        geometry,_=original.normalize(original.resample(geometry,48,'uniform'),counts)
        scale,weight=1.,1.
    elif name=='p35_h0':scale,weight=100.,.5
    else:raise ValueError(name)
    ids=torch.cat((centers[:,None],neighbors),-1);batch=torch.arange(len(geometry),device=geometry.device)
    primitive=geometry[batch[:,None],ids]
    base=original.pathline_dft_features_3d(primitive,num_freq=6,neighbor_scale=scale,
        neighbor_weight=weight,neighbor_pool='sort',mode='gram',include_chirality=True,return_numpy=False)
    # Same per-feature mean / numerical maximum, now over 16 descriptors.
    grouped=base[:,23:].reshape(len(geometry),16,23)
    feature=torch.cat((base[:,:23],original.direction_spectrum(primitive[:,0],6),
                       grouped.mean(1),grouped.amax(1)),-1)
    assert feature.shape==(len(geometry),141) and torch.isfinite(feature).all()
    return torch.cat((feature,torch.ones_like(feature[:,:1])),-1)[:,None],neighbors


make_model=original.make_model
