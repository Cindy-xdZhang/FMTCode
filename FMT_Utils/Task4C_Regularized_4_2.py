"""Wall-relative Fourier features, geometry augmentation and cosine heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_fmt, bundle_voxels
from FMT_Utils.Task4C_Multiscale_4_1 import RegularizedConv3D


def augment_bundle(geometry, seeds, kind, angles=None):
    """Mirror y with curve reversal, or rotate slightly about streamwise x.

    Vorticity is axial: reflection requires reversing the directed curve.
    This preserves positive head-direction y for a mirrored hairpin.
    """
    if kind == "mirror":
        sign = geometry.new_tensor([1., -1., 1.])
        return geometry.flip(2)*sign, seeds*sign
    if kind == "tilt":
        c, s = angles.cos(), angles.sin()
        rotation = torch.zeros((len(geometry),3,3), device=geometry.device, dtype=geometry.dtype)
        rotation[:,0,0] = 1
        rotation[:,1,1] = rotation[:,2,2] = c
        rotation[:,2,1] = s
        rotation[:,1,2] = -s
        return torch.einsum("bij,bnpj->bnpi",rotation,geometry), torch.einsum("bij,bnj->bni",rotation,seeds)
    raise ValueError("Unknown physical augmentation")


@torch.no_grad()
def extended_fmt(geometry, seeds, counts):
    """Frozen 322D local FMT + 288D wall-relative position/tangent spectrum."""
    mask = torch.arange(geometry.shape[1], device=geometry.device)[None] < counts[:,None]
    delta = geometry.diff(dim=2)
    tangent = delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    tangent = torch.cat((tangent,tangent[:,:,-1:]),dim=2)
    sequence = torch.cat((geometry,tangent),dim=-1)
    coefficients = torch.view_as_real(torch.fft.rfft(sequence,dim=2,norm="ortho")[:,:,:6]).flatten(2)
    mean = (coefficients*mask[...,None]).sum(1)/counts[:,None]
    std = (((coefficients-mean[:,None]).square()*mask[...,None]).sum(1)/counts[:,None]).sqrt()
    maximum = coefficients.masked_fill(~mask[...,None],-torch.inf).amax(1)
    minimum = coefficients.masked_fill(~mask[...,None],torch.inf).amin(1)
    result = torch.cat((bundle_fmt(geometry,seeds,counts),mean,std,maximum,minimum),dim=1)
    if result.shape[1]!=610 or not torch.isfinite(result).all():
        raise ValueError("Invalid extended FMT")
    return result


class Classifier(nn.Module):
    def __init__(self, method, candidate):
        super().__init__()
        dropout = candidate["dropout"]
        self.cosine = candidate["classifier"] == "cosine_margin"
        if method == "fmt_mlp":
            self.encoder = nn.Sequential(nn.Linear(610,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(dropout),
                nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64))
        else:
            original = RegularizedConv3D(dropout)
            self.encoder = nn.Sequential(original.convolution,*list(original.classifier.children())[:-1],nn.Linear(256,64))
        self.head = nn.Linear(64,2,bias=not self.cosine)

    def forward(self, x):
        latent = self.encoder(x)
        if self.cosine:
            return 30*F.linear(F.normalize(latent,dim=-1),F.normalize(self.head.weight,dim=-1))
        return self.head(latent)


def make_model(method,candidate):
    return Classifier(method,candidate)


def classifier_loss(logits, labels, candidate, positive_weight):
    labels = labels.long()
    if candidate["classifier"] == "cosine_margin":
        margins = logits.new_tensor([.1,.3])
        logits = logits-30*F.one_hot(labels,num_classes=2)*margins
    weights = torch.stack((positive_weight.new_ones(()),positive_weight))
    return F.cross_entropy(logits,labels,weight=weights,label_smoothing=candidate["label_smoothing"])


def classifier_probability(logits):
    return logits.softmax(dim=-1)[:,1]
