"""Parameter-matched PointNet++ single-scale grouping, with frozen geometry indices.

Topology: charlesq34/pointnet2 at 42926632a3c33461aebfbee2d829098b30a23aaa.
See third_party/pointnet2/LICENSE and the experiment protocol for adaptations.
"""
from __future__ import annotations
import torch
from torch import nn
from FMT_Utils.Task4C_PointNet_1_1 import PointLayer, dense

PARAMETERS = 76723
CACHE_SHAPES = dict(order=(864,), sample1=(512,), group1=(512,32), sample2=(128,), group2=(128,64))


def index_points(x, indices):
    batch = torch.arange(len(x),device=x.device).reshape(-1,*([1]*(indices.ndim-1)))
    return x[batch,indices]


@torch.no_grad()
def canonical_order(x, counts):
    valid = torch.arange(x.shape[1],device=x.device)[None] < counts[:,None]
    xyz = x.double().masked_fill(~valid[...,None],torch.inf)
    order = torch.arange(x.shape[1],device=x.device)[None].expand(len(x),-1)
    for axis in (2,1,0):
        perm = torch.argsort(xyz[...,axis],dim=1,stable=True)
        xyz = index_points(xyz,perm); order = order.gather(1,perm)
    return order


@torch.no_grad()
def farthest_points(x, counts, number):
    """Use each valid point at most once; mask excess center slots if N < number."""
    x=x.double(); b,n,_=x.shape
    valid = torch.arange(n,device=x.device)[None] < counts[:,None]
    distances = torch.full((b,n),torch.inf,device=x.device,dtype=x.dtype).masked_fill(~valid,-1)
    selected = torch.zeros(b,device=x.device,dtype=torch.long)
    result = torch.zeros((b,number),device=x.device,dtype=torch.long)
    batch = torch.arange(b,device=x.device)
    for i in range(number):
        active = i < counts
        result[:,i] = torch.where(active,selected,0)
        delta = (x-x[batch,selected,None]).square().sum(-1)
        distances = torch.minimum(distances,delta)
        distances[batch,selected] = -1
        selected = distances.max(1).indices
    return result


@torch.no_grad()
def ball_query(x, centers, counts, center_counts, radius, neighbors):
    """Official first-in-input-order ball rule; repeat the first hit when undersized."""
    x=x.double(); centers=centers.double(); b,n,_=x.shape; output=[]
    ids=torch.arange(n,device=x.device)[None,None]
    for first in range(0,centers.shape[1],64):
        c=centers[:,first:first+64]
        inside=(c[:,:,None]-x[:,None]).square().sum(-1) < radius*radius
        inside &= ids < counts[:,None,None]
        selected=torch.where(inside,ids,n).sort(-1).values[...,:neighbors]
        valid_center=torch.arange(first,first+len(c[0]),device=x.device)[None] < center_counts[:,None]
        if torch.any(valid_center & (selected[...,0] == n)): raise ValueError('Empty valid ball')
        selected=torch.where(selected==n,selected[...,:1],selected)
        selected=selected.masked_fill(~valid_center[...,None],0)
        output.append(selected)
    return torch.cat(output,1)


@torch.no_grad()
def build_hierarchy(geometry, counts):
    """Only input geometry determines FPS and ball groups; no labels or learned state."""
    flat=geometry.reshape(len(geometry),-1,3); n=counts*geometry.shape[2]
    order=canonical_order(flat,n)
    x=index_points(flat,order).double()
    valid=torch.arange(x.shape[1],device=x.device)[None] < n[:,None]
    x=x.masked_fill(~valid[...,None],0)
    sample1=farthest_points(x,n,512); c1=n.clamp_max(512); x1=index_points(x,sample1)
    group1=ball_query(x,x1,n,c1,.2,32)
    sample2=farthest_points(x1,c1,128); c2=c1.clamp_max(128); x2=index_points(x1,sample2)
    group2=ball_query(x1,x2,c1,c2,.4,64)
    return dict(order=order,sample1=sample1,group1=group1,sample2=sample2,group2=group2)


class SetAbstraction(nn.Module):
    def __init__(self, cin, widths):
        super().__init__(); dims=(cin,*widths)
        self.layers=nn.ModuleList(PointLayer(a,b) for a,b in zip(dims[:-1],dims[1:]))

    def forward(self, xyz, features, samples, groups, center_counts):
        centers=index_points(xyz,samples)
        local=index_points(xyz,groups)-centers[:,:,None]
        x=local if features is None else torch.cat((local,index_points(features,groups)),-1)
        valid=torch.arange(samples.shape[1],device=xyz.device)[None] < center_counts[:,None]
        grouped_valid=valid[:,:,None].expand(-1,-1,groups.shape[-1])
        for layer in self.layers: x=layer(x,grouped_valid)
        return centers,x.amax(2)


class PointNetPlusPlus(nn.Module):
    """512/128/global abstraction; all input points, no line identity or T-Nets."""
    def __init__(self, dropout=.15):
        super().__init__()
        self.first=SetAbstraction(3,(16,16,32))
        self.second=SetAbstraction(35,(32,32,64))
        self.global_layers=nn.ModuleList((PointLayer(67,64),PointLayer(64,128),PointLayer(128,256)))
        self.classifier=nn.Sequential(dense(256,80),nn.Dropout(dropout),dense(80,45),nn.Dropout(dropout),nn.Linear(45,2))

    def forward(self, batch):
        geometry,counts,hierarchy=batch
        x=index_points(geometry.reshape(len(geometry),-1,3),hierarchy['order'])
        n=counts*geometry.shape[2]
        valid=torch.arange(x.shape[1],device=x.device)[None] < n[:,None]
        x=x.masked_fill(~valid[...,None],0)
        c1=n.clamp_max(512); c2=c1.clamp_max(128)
        x,f=self.first(x,None,hierarchy['sample1'],hierarchy['group1'],c1)
        x,f=self.second(x,f,hierarchy['sample2'],hierarchy['group2'],c2)
        # Official group-all uses the origin; it does not subtract the sampled-point mean.
        f=torch.cat((x,f),-1)
        valid=torch.arange(f.shape[1],device=x.device)[None] < c2[:,None]
        for layer in self.global_layers: f=layer(f,valid)
        return self.classifier(f.masked_fill(~valid[...,None],-torch.inf).amax(1))


def make_model(dropout=.15):
    model=PointNetPlusPlus(dropout)
    assert sum(p.numel() for p in model.parameters())==PARAMETERS
    return model
