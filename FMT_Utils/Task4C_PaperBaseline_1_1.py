"""Frozen architecture implementation of the supplied hairpin manuscript, Sec. 3.1.

This reproduces the stated hierarchy and equations, not the unpublished data or
unreported training settings. Keep this baseline unchanged in later FMT trials.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


CLASS_NAMES = ("hairpin", "quasi_hairpin", "fragment", "non_hairpin")


class HierarchicalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.line_lstm = nn.LSTM(6, 64, batch_first=True, bidirectional=True)
        self.bundle_lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(256, 64)

    def encode_lines(self, x):
        if x.ndim != 4 or x.shape[-1] != 6:
            raise ValueError("Expected [batch, lines, points, 6] geometry/tangents")
        b, n, p, _ = x.shape
        _, (hidden, _) = self.line_lstm(x.reshape(b * n, p, 6))
        return torch.cat((hidden[-2], hidden[-1]), -1).reshape(b, n, 128)

    def encode_bundle(self, line_embeddings):
        _, (hidden, _) = self.bundle_lstm(line_embeddings)
        return self.projection(torch.cat((hidden[-2], hidden[-1]), -1))

    def forward(self, x):
        # The manuscript specifies zero-padded line slots, not packed sequences.
        return self.encode_bundle(self.encode_lines(x))


class HierarchicalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(64, 256)
        self.bundle_lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.line_lstm = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.output = nn.Linear(128, 6)

    def forward(self, z, line_count=256, point_count=32):
        b = z.shape[0]
        repeated = self.projection(z)[:, None].expand(-1, line_count, -1)
        lines, _ = self.bundle_lstm(repeated)
        repeated_points = lines.reshape(b * line_count, 1, 256).expand(-1, point_count, -1)
        points, _ = self.line_lstm(repeated_points)
        return self.output(points).tanh().reshape(b, line_count, point_count, 6)


class PaperAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HierarchicalEncoder()
        self.decoder = HierarchicalDecoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z, x.shape[1], x.shape[2]), z


def reconstruction_loss(prediction, target, line_mask, epsilon=1e-8):
    """Eq. 10: mean squared error and six-component cosine distance, 1:1."""
    if prediction.shape != target.shape or line_mask.shape != target.shape[:2]:
        raise ValueError("Prediction/target/mask shapes differ")
    mask = line_mask.bool()[..., None] & (target.square().sum(-1) > 0)
    if not mask.any():
        raise ValueError("No nonzero, valid reconstruction targets")
    p, t = prediction[mask], target[mask]
    mse = (p - t).square().mean(-1)
    cosine = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + epsilon)
    return (0.5 * mse + 0.5 * (1 - cosine)).mean()


class AsymmetricMarginClassifier(nn.Module):
    """Eq. 11 additive COSINE margin, not an acos/angular ArcFace transform."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 64))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.tensor([0.3, 0.3, 0.3, 0.1]))
        self.scale = 30.0

    def forward(self, z, targets=None):
        cosine = F.linear(F.normalize(z, dim=-1), F.normalize(self.weight, dim=-1))
        if targets is not None:
            if targets.ndim != 1 or len(targets) != len(z) or torch.any((targets < 0) | (targets > 3)):
                raise ValueError("Targets must contain one of four class IDs per bundle")
            cosine = cosine - F.one_hot(targets.long(), 4) * self.margins
        return self.scale * cosine


def inverse_frequency_weights(labels):
    counts = torch.bincount(labels.long(), minlength=4)
    if len(counts) != 4 or (counts == 0).any():
        raise ValueError("All four human taxonomy classes must occur in training")
    weights = counts.float().reciprocal()
    return weights / weights.mean()


def structural_confidence(probabilities):
    """Eq. 20 ranking score; it is not a calibrated probability of a hairpin."""
    return probabilities @ probabilities.new_tensor([1.0, 0.7, 0.4, 0.0])
