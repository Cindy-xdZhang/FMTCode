"""Supervised geometric classifiers for 3D pathline-cross primitives."""

from __future__ import annotations

import torch
from torch import nn


class TemporalLineEncoder3D(nn.Module):
    """Encode each pathline with shared temporal convolutions.

    Input is ``[B,K,L,3]``.  Sharing the encoder across K lines and pooling
    only after temporal convolution preserves time order without treating the
    primitive as an unordered point cloud.
    """

    def __init__(self, width=64):
        super().__init__()
        width = int(width)
        self.network = nn.Sequential(
            nn.Conv1d(3, width // 2, kernel_size=5, padding=2),
            nn.GroupNorm(4, width // 2),
            nn.GELU(),
            nn.Conv1d(width // 2, width, kernel_size=5, padding=2),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )

    def forward(self, pathlines):
        if pathlines.ndim != 4 or pathlines.shape[-1] != 3:
            raise ValueError(f"pathlines must be [B,K,L,3], got {tuple(pathlines.shape)}")
        batch, lines, length, _ = pathlines.shape
        encoded = self.network(
            pathlines.reshape(batch * lines, length, 3).transpose(1, 2)
        )
        pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1)
        return pooled.reshape(batch, lines, -1)


class PathlineGeometryEncoder3D(nn.Module):
    """Aggregate one center line and an unordered set of neighbour lines."""

    def __init__(self, temporal_width=64, embedding_dim=128):
        super().__init__()
        self.line_encoder = TemporalLineEncoder3D(temporal_width)
        line_dim = 2 * int(temporal_width)
        self.projection = nn.Sequential(
            nn.Linear(3 * line_dim, int(embedding_dim)),
            nn.LayerNorm(int(embedding_dim)),
            nn.GELU(),
        )

    def forward(self, pathlines):
        line_features = self.line_encoder(pathlines)
        if line_features.shape[1] < 2:
            raise ValueError("a primitive requires one center and at least one neighbour")
        center = line_features[:, 0]
        neighbours = line_features[:, 1:]
        aggregate = torch.cat(
            (center, neighbours.mean(dim=1), neighbours.amax(dim=1)), dim=-1
        )
        return self.projection(aggregate)


class PathlineBinaryClassifier3D(nn.Module):
    """Classify pathline primitives with raw geometry, optionally plus FMT.

    Variants:
    - ``raw``: geometric encoder only.
    - ``raw_wide``: geometric encoder plus a larger raw-only capacity branch.
    - ``raw_fmt``: the same geometric encoder plus fixed cached FMT features.
    """

    VALID_VARIANTS = {"raw", "raw_wide", "raw_fmt"}

    def __init__(self, variant="raw", fmt_dim=161, temporal_width=64,
                 embedding_dim=128, auxiliary_dim=64):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = str(variant)
        embedding_dim = int(embedding_dim)
        auxiliary_dim = int(auxiliary_dim)
        self.geometry = PathlineGeometryEncoder3D(temporal_width, embedding_dim)
        if variant == "raw_wide":
            self.auxiliary = nn.Sequential(
                nn.Linear(embedding_dim, 2 * embedding_dim),
                nn.LayerNorm(2 * embedding_dim),
                nn.GELU(),
                nn.Linear(2 * embedding_dim, auxiliary_dim),
                nn.LayerNorm(auxiliary_dim),
                nn.GELU(),
            )
        elif variant == "raw_fmt":
            self.auxiliary = nn.Sequential(
                nn.Linear(int(fmt_dim), auxiliary_dim),
                nn.LayerNorm(auxiliary_dim),
                nn.GELU(),
            )
        else:
            self.auxiliary = None
        head_input = embedding_dim + (0 if self.auxiliary is None else auxiliary_dim)
        self.head = nn.Sequential(
            nn.Linear(head_input, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, pathlines, fmt_features=None):
        geometry = self.geometry(pathlines)
        if self.variant == "raw_fmt":
            if fmt_features is None:
                raise ValueError("raw_fmt requires fmt_features")
            fused = torch.cat((geometry, self.auxiliary(fmt_features)), dim=-1)
        elif self.variant == "raw_wide":
            fused = torch.cat((geometry, self.auxiliary(geometry)), dim=-1)
        else:
            fused = geometry
        return self.head(fused).squeeze(-1)


class PathlineFMTResidualClassifier3D(nn.Module):
    """Add a trainable FMT correction without changing a frozen Raw model.

    The frozen Raw logit is always available as the ``alpha=0`` case.  Only
    the FMT auxiliary and residual head are trained.  This separates genuine
    incremental FMT information from changes to the raw geometry backbone.
    """

    def __init__(self, raw_model, fmt_dim=161, embedding_dim=128,
                 auxiliary_dim=64, residual_input="geometry_fmt"):
        super().__init__()
        if not isinstance(raw_model, PathlineBinaryClassifier3D):
            raise TypeError("raw_model must be PathlineBinaryClassifier3D")
        if raw_model.variant != "raw":
            raise ValueError("raw_model must use the raw variant")
        self.raw_model = raw_model
        for parameter in self.raw_model.parameters():
            parameter.requires_grad_(False)
        embedding_dim = int(embedding_dim)
        auxiliary_dim = int(auxiliary_dim)
        if residual_input not in {"geometry_fmt", "fmt_only", "dual"}:
            raise ValueError(
                "residual_input must be 'geometry_fmt', 'fmt_only', or 'dual'"
            )
        self.residual_input = str(residual_input)
        self.fmt_encoder = nn.Sequential(
            nn.Linear(int(fmt_dim), auxiliary_dim),
            nn.LayerNorm(auxiliary_dim),
            nn.GELU(),
        )
        residual_width = (
            embedding_dim + auxiliary_dim
            if self.residual_input in {"geometry_fmt", "dual"} else auxiliary_dim
        )
        self.residual_head = nn.Sequential(
            nn.Linear(residual_width, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )
        self.fmt_only_head = None
        if self.residual_input == "dual":
            self.fmt_only_head = nn.Sequential(
                nn.Linear(auxiliary_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, 1),
            )

    def forward_components(self, pathlines, fmt_features):
        if fmt_features is None:
            raise ValueError("FMT residual classifier requires fmt_features")
        with torch.no_grad():
            geometry = self.raw_model.geometry(pathlines)
            raw_logit = self.raw_model.head(geometry).squeeze(-1)
        auxiliary = self.fmt_encoder(fmt_features)
        residual_input = (
            torch.cat((geometry.detach(), auxiliary), dim=-1)
            if self.residual_input in {"geometry_fmt", "dual"} else auxiliary
        )
        residual_logit = self.residual_head(residual_input).squeeze(-1)
        if self.fmt_only_head is not None:
            residual_logit = residual_logit + self.fmt_only_head(auxiliary).squeeze(-1)
        return raw_logit, residual_logit

    def forward(self, pathlines, fmt_features, alpha=1.0):
        raw_logit, residual_logit = self.forward_components(pathlines, fmt_features)
        return raw_logit + float(alpha) * residual_logit


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
