"""Classic PointNet topology with both T-Nets, reduced for a matched parameter budget.

Architecture and orthogonality loss follow charlesq34/pointnet at
2618f72bc1a0fd21b074096e748016960d44ef55. See third_party/pointnet/LICENSE.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

PARAMETERS = 76749


class PointLayer(nn.Module):
    """Shared pointwise affine map, valid-point BatchNorm, and ReLU."""
    def __init__(self, cin, cout):
        super().__init__()
        self.linear = nn.Linear(cin, cout)
        self.norm = nn.BatchNorm1d(cout, eps=1e-3, momentum=.1)

    def forward(self, x, valid):
        # Neither padded coordinates nor padded affine biases enter BatchNorm statistics.
        value = F.relu(self.norm(self.linear(x[valid])))
        result = value.new_zeros((*valid.shape, value.shape[-1]))
        result[valid] = value
        return result


def max_points(x, valid):
    return x.masked_fill(~valid[..., None], -torch.inf).max(1).values


def dense(cin, cout):
    return nn.Sequential(nn.Linear(cin, cout), nn.BatchNorm1d(cout, eps=1e-3, momentum=.1), nn.ReLU())


class TransformNet(nn.Module):
    """PointNet's shared MLP / max / MLP predictor of an identity-initialized KxK map."""
    def __init__(self, dimensions, width=11, global_width=168, hidden=(88, 44)):
        super().__init__()
        self.dimensions = dimensions
        self.points = nn.ModuleList((PointLayer(dimensions, width), PointLayer(width, 2*width),
                                     PointLayer(2*width, global_width)))
        self.hidden = nn.Sequential(dense(global_width, hidden[0]), dense(hidden[0], hidden[1]))
        self.output = nn.Linear(hidden[1], dimensions*dimensions)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad(): self.output.bias.copy_(torch.eye(dimensions).flatten())

    def forward(self, x, valid):
        for layer in self.points: x = layer(x, valid)
        return self.output(self.hidden(max_points(x, valid))).reshape(-1, self.dimensions, self.dimensions)


class PointNet(nn.Module):
    """Whole-bundle xyz classifier; no line identity, voxelization, or handcrafted features."""
    def __init__(self, dropout=.15, width=11, global_width=168, transform_hidden=(88, 44), head=(88, 39)):
        super().__init__()
        self.input_transform = TransformNet(3, width, global_width, transform_hidden)
        self.first = nn.ModuleList((PointLayer(3, width), PointLayer(width, width)))
        self.feature_transform = TransformNet(width, width, global_width, transform_hidden)
        self.last = nn.ModuleList((PointLayer(width, width), PointLayer(width, 2*width),
                                  PointLayer(2*width, global_width)))
        self.classifier = nn.Sequential(dense(global_width, head[0]), nn.Dropout(dropout),
                                        dense(head[0], head[1]), nn.Dropout(dropout), nn.Linear(head[1], 2))
        self.last_feature_transform = None

    def forward(self, batch):
        geometry, counts = batch
        batch_size, lines, points, _ = geometry.shape
        valid = torch.arange(lines*points, device=geometry.device)[None] < (counts*points)[:, None]
        x = geometry.reshape(batch_size, lines*points, 3).masked_fill(~valid[..., None], 0)
        transform = self.input_transform(x, valid)
        x = torch.bmm(x, transform)
        for layer in self.first: x = layer(x, valid)
        self.last_feature_transform = self.feature_transform(x, valid)
        x = torch.bmm(x, self.last_feature_transform)
        for layer in self.last: x = layer(x, valid)
        return self.classifier(max_points(x, valid))

    def orthogonality_penalty(self):
        """Exact official tf.nn.l2_loss reduction: half sum over batch and matrix entries."""
        transform = self.last_feature_transform
        if transform is None: raise RuntimeError('A forward pass is required before the regularizer')
        eye = torch.eye(transform.shape[-1], device=transform.device, dtype=transform.dtype)
        return .5*(torch.bmm(transform, transform.transpose(1, 2))-eye).square().sum()


def make_model(dropout=.15):
    model = PointNet(dropout)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    return model
