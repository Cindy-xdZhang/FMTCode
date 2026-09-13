"""Frozen per-line FMT with fixed-statistic or learned set aggregation."""
from __future__ import annotations

import torch
from torch import nn

from FMT_Utils.Task4C_PaperBundles_3_1 import pathline_dft_features_3d, bundle_voxels
from FMT_Utils.Task4C_Regularized_4_2 import Classifier as PreviousClassifier
from FMT_Utils.Task4C_Regularized_4_2 import classifier_loss, classifier_probability


@torch.no_grad()
def line_fmt(geometry, seeds, counts):
    """Keep the 161 local coefficients and 72 signed coefficients per line."""
    b, n, _, _ = geometry.shape
    mask = torch.arange(n, device=geometry.device)[None] < counts[:, None]
    if (counts < 10).any() or not torch.isfinite(geometry).all():
        raise ValueError("Expected cleaned bundles")
    distance = torch.cdist(seeds, seeds)
    distance.masked_fill_(~mask[:, None, :], torch.inf)
    distance.diagonal(dim1=1, dim2=2).fill_(torch.inf)
    nearest = torch.argsort(distance, dim=-1, stable=True)[..., :6]
    central = torch.arange(n, device=geometry.device)[None, :, None].expand(b, -1, -1)
    ids = torch.cat((central, nearest), -1)
    bi, li = mask.nonzero(as_tuple=True)
    local = geometry.new_zeros((b, n, 161))
    for start in range(0, len(bi), 1024):
        bs, ls = bi[start:start+1024], li[start:start+1024]
        primitive = geometry[bs[:, None], ids[bs, ls]]
        local[bs, ls] = pathline_dft_features_3d(primitive, num_freq=6,
            neighbor_weight=.5, neighbor_scale=100., neighbor_pool="sort", mode="gram",
            include_chirality=True, return_numpy=False)
    delta = geometry.diff(dim=2)
    tangent = delta/delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tangent = torch.cat((tangent, tangent[:, :, -1:]), dim=2)
    sequence = torch.cat((geometry, tangent), -1)
    signed = torch.view_as_real(torch.fft.rfft(sequence, dim=2, norm="ortho")[:, :, :6]).flatten(2)
    result = torch.cat((local, signed*mask[..., None], mask[..., None].to(geometry.dtype)), -1)
    if result.shape != (b, n, 234) or not torch.isfinite(result).all():
        raise ValueError("Invalid per-line FMT")
    return result


def fixed_statistics(tokens):
    """Recover the frozen 4.2 610D vector without losing additional information."""
    mask = tokens[..., -1] > .5
    x, count = tokens[..., :-1], mask.sum(1, keepdim=True)
    mean = (x*mask[..., None]).sum(1)/count
    maximum = x.masked_fill(~mask[..., None], -torch.inf).amax(1)
    signed = x[..., 161:]
    std = (((signed-mean[:, None, 161:]).square()*mask[..., None]).sum(1)/count).sqrt()
    minimum = signed.masked_fill(~mask[..., None], torch.inf).amin(1)
    return torch.cat((mean[:, :161], maximum[:, :161], mean[:, 161:], std, maximum[:, 161:], minimum), -1)


class LearnedLinePooling(nn.Module):
    """Shared nonlinear map before masked mean/max; independent of line order."""
    def __init__(self, dropout):
        super().__init__()
        self.line = nn.Sequential(nn.Linear(233,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(128,128), nn.LayerNorm(128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(256,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(128,64), nn.GELU(), nn.Linear(64,2))

    def forward(self, tokens):
        mask = tokens[..., -1] > .5
        features = self.line(tokens[..., :-1])
        mean = (features*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
        maximum = features.masked_fill(~mask[..., None], -torch.inf).amax(1)
        return self.head(torch.cat((mean, maximum), -1))


class WiderConv3D(nn.Module):
    """Same 24-cube input and pooling topology; twice the convolution channels."""
    def __init__(self, dropout):
        super().__init__()
        layers = []
        for i, (cin, cout) in enumerate(((4,16), (16,32), (32,64))):
            layers.extend((nn.Conv3d(cin,cout,3,padding=1), nn.GroupNorm(4,cout), nn.GELU(),
                           nn.Dropout3d(dropout/2), nn.MaxPool3d(2) if i<2 else nn.AdaptiveAvgPool3d(2)))
        self.network = nn.Sequential(*layers, nn.Flatten(), nn.Linear(512,256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256,64), nn.Linear(64,2))

    def forward(self, x):
        return self.network(x)


def make_model(method, candidate):
    if candidate["architecture"] == "reference_4.2":
        return PreviousClassifier(method, candidate)
    return LearnedLinePooling(candidate["dropout"]) if method == "fmt_mlp" else WiderConv3D(candidate["dropout"])
