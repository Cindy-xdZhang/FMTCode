"""Three geometry-only classifiers for frozen Task4-c bundles.

Point-NN equations adapted from ZrrSkywalker/Point-NN, commit
a85bfc365258a2c65a5f6ac289537b4d7a7cec0f (Renrui Zhang, MIT).
See third_party/point_nn/LICENSE and the experiment protocol for adaptations.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

PARAMETERS = dict(baseline0=76782, baseline1=76704, baseline2=76786)


def tangent_curvature(geometry, counts):
    """Mean/max of oriented unit tangent xyz and unsigned circumcircle curvature."""
    x = geometry.double()
    chord = torch.cat((x[:, :, 1:2]-x[:, :, :1], x[:, :, 2:]-x[:, :, :-2],
                       x[:, :, -1:]-x[:, :, -2:-1]), dim=2)
    tangent = chord / torch.linalg.vector_norm(chord, dim=-1, keepdim=True).clamp_min(1e-12)
    left, right = x[:, :, 1:-1]-x[:, :, :-2], x[:, :, 2:]-x[:, :, 1:-1]
    denominator = (torch.linalg.vector_norm(left, dim=-1)*torch.linalg.vector_norm(right, dim=-1)
                   *torch.linalg.vector_norm(left+right, dim=-1))
    curvature = 2*torch.linalg.vector_norm(torch.cross(left, right, dim=-1), dim=-1)/denominator.clamp_min(1e-24)
    curvature = torch.cat((curvature[:, :, :1], curvature, curvature[:, :, -1:]), dim=2)
    features = torch.cat((tangent, curvature.unsqueeze(-1)), dim=-1)
    valid = torch.arange(x.shape[1], device=x.device)[None, :, None, None] < counts[:, None, None, None]
    average = torch.where(valid, features, 0).sum((1, 2))/(counts[:, None]*x.shape[2])
    maximum = features.masked_fill(~valid, -torch.inf).flatten(1, 2).max(1).values
    return torch.cat((average, maximum), dim=-1).float()


def index_points(points, indices):
    batches = torch.arange(len(points), device=points.device).reshape(-1, *([1]*(indices.ndim-1)))
    return points[batches, indices]


def canonical_points(x):
    """Lexicographic xyz order removes dependence on input point/line ordering."""
    for axis in (2, 1, 0):
        x = index_points(x, torch.argsort(x[..., axis], dim=1, stable=True))
    return x


def farthest_points(x, number):
    """Deterministic FPS, starting at the lexicographically first point."""
    distance = torch.full(x.shape[:2], torch.inf, device=x.device)
    indices = torch.empty((len(x), number), device=x.device, dtype=torch.long)
    selected = torch.zeros(len(x), device=x.device, dtype=torch.long)
    batch = torch.arange(len(x), device=x.device)
    for i in range(number):
        indices[:, i] = selected
        delta = ((x-x[batch, selected, None])**2).sum(-1)
        distance = torch.minimum(distance, delta)
        distance[batch, selected] = -1
        selected = distance.max(1).indices
    return indices


def position_embedding(x, width):
    frequencies = width//6
    denominator = 100.**(torch.arange(frequencies, device=x.device, dtype=x.dtype)/frequencies)
    phase = 1000.*x.unsqueeze(-1)/denominator
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-3)


class PointNNEncoder(nn.Module):
    """Fixed four-stage whole-cloud Point-NN; all valid points, zero learned parameters."""
    @torch.no_grad()
    def forward(self, points):
        x = canonical_points(points)
        features = position_embedding(x, 72)
        for _ in range(4):
            anchors = farthest_points(x, x.shape[1]//2)
            centers = index_points(x, anchors)
            center_features = index_points(features, anchors)
            distances = ((centers[:, :, None]-x[:, None])**2).sum(-1)
            # Stable sorting makes distance ties independent of accelerator top-k tie rules.
            neighbors = torch.argsort(distances, dim=-1, stable=True)[..., :min(90, x.shape[1])]
            dp = index_points(x, neighbors)-centers[:, :, None]
            df = index_points(features, neighbors)-center_features[:, :, None]
            # Per-cloud rather than upstream cross-batch normalization.
            dp = dp/(dp.flatten(1).std(1, unbiased=True)[:, None, None, None]+1e-5)
            df = df/(df.flatten(1).std(1, unbiased=True)[:, None, None, None]+1e-5)
            expanded = torch.cat((df, center_features[:, :, None].expand_as(df)), dim=-1)
            weight = position_embedding(dp, features.shape[-1]*2)
            weighted = (expanded+weight)*weight
            pooled = weighted.max(2).values+weighted.mean(2)
            # Equivalent to the original untrained BatchNorm in evaluation mode + GELU.
            features = F.gelu(pooled/math.sqrt(1.+1e-5))
            x = centers
        return features.max(1).values+features.mean(1)


def mlp(widths, dropout):
    layers = []
    for cin, cout in zip(widths[:-2], widths[1:-1]):
        layers.extend((nn.Linear(cin, cout), nn.LayerNorm(cout), nn.GELU(), nn.Dropout(dropout)))
    layers.append(nn.Linear(widths[-2], widths[-1]))
    return nn.Sequential(*layers)


class BundleBiLSTM(nn.Module):
    """Shared per-line spatial BiLSTM followed by symmetric pooling over valid lines."""
    def __init__(self, dropout):
        super().__init__()
        self.encoder = nn.LSTM(3, 81, batch_first=True, bidirectional=True)
        self.classifier = mlp((324, 64, 2), dropout)

    def forward(self, batch):
        geometry, counts = batch
        b, lines, points, _ = geometry.shape
        _, (hidden, _) = self.encoder(geometry.reshape(b*lines, points, 3))
        features = hidden.transpose(0, 1).reshape(b, lines, 162)
        valid = torch.arange(lines, device=geometry.device)[None, :, None] < counts[:, None, None]
        average = torch.where(valid, features, 0).sum(1)/counts[:, None]
        maximum = features.masked_fill(~valid, -torch.inf).max(1).values
        return self.classifier(torch.cat((average, maximum), -1))


def make_model(method, dropout=.15):
    if method == 'baseline0':
        model = mlp((8, 256, 228, 64, 2), dropout)
    elif method == 'baseline1':
        model = mlp((1152, 62, 76, 2), dropout)
    elif method == 'baseline2':
        model = BundleBiLSTM(dropout)
    else:
        raise ValueError(method)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[method]
    return model
