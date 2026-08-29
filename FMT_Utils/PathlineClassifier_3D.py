"""Supervised geometric classifiers for 3D pathline-cross primitives."""

from __future__ import annotations

import torch
from torch import nn


class _ResidualMLPBlock(nn.Module):
    def __init__(self, width, dropout=0.0):
        super().__init__()
        width = int(width)
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, width),
            nn.Dropout(float(dropout)),
        )

    def forward(self, values):
        return values + self.network(values)


class _ResidualMLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=2, dropout=0.0):
        super().__init__()
        if int(depth) < 1:
            raise ValueError("residual MLP depth must be positive")
        self.input_projection = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            _ResidualMLPBlock(hidden_dim, dropout) for _ in range(int(depth))
        ])
        self.output = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)), nn.Linear(int(hidden_dim), 1)
        )

    def forward(self, values):
        values = self.input_projection(values)
        for block in self.blocks:
            values = block(values)
        return self.output(values)


class _GatedFusionHead(nn.Module):
    def __init__(self, geometry_dim, auxiliary_dim, hidden_dim, dropout=0.0):
        super().__init__()
        geometry_dim = int(geometry_dim)
        auxiliary_dim = int(auxiliary_dim)
        hidden_dim = int(hidden_dim)
        self.geometry_projection = nn.Linear(geometry_dim, hidden_dim)
        self.auxiliary_projection = nn.Linear(auxiliary_dim, hidden_dim)
        self.gate = nn.Linear(geometry_dim + auxiliary_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, geometry, auxiliary):
        geometry_hidden = self.geometry_projection(geometry)
        auxiliary_hidden = self.auxiliary_projection(auxiliary)
        gate = torch.sigmoid(self.gate(torch.cat((geometry, auxiliary), dim=-1)))
        fused = gate * auxiliary_hidden + (1.0 - gate) * geometry_hidden
        return self.output(fused)


class _LowRankBilinearFusionHead(nn.Module):
    def __init__(self, geometry_dim, auxiliary_dim, hidden_dim, rank,
                 dropout=0.0):
        super().__init__()
        rank = int(rank)
        if rank < 1:
            raise ValueError("bilinear rank must be positive")
        self.geometry_projection = nn.Linear(int(geometry_dim), rank)
        self.auxiliary_projection = nn.Linear(int(auxiliary_dim), rank)
        self.output = nn.Sequential(
            nn.Linear(3 * rank, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, geometry, auxiliary):
        geometry_low_rank = self.geometry_projection(geometry)
        auxiliary_low_rank = self.auxiliary_projection(auxiliary)
        interaction = geometry_low_rank * auxiliary_low_rank
        return self.output(torch.cat(
            (geometry_low_rank, auxiliary_low_rank, interaction), dim=-1
        ))


class _AttentionFusionHead(nn.Module):
    def __init__(self, geometry_dim, auxiliary_dim, hidden_dim, heads=4,
                 dropout=0.0):
        super().__init__()
        hidden_dim = int(hidden_dim)
        heads = int(heads)
        if heads < 1 or hidden_dim % heads:
            raise ValueError("attention hidden dimension must divide by heads")
        self.geometry_projection = nn.Linear(int(geometry_dim), hidden_dim)
        self.auxiliary_projection = nn.Linear(int(auxiliary_dim), hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=float(dropout), batch_first=True
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, geometry, auxiliary):
        tokens = torch.stack((
            self.geometry_projection(geometry),
            self.auxiliary_projection(auxiliary),
        ), dim=1)
        attended, _ = self.attention(
            tokens, tokens, tokens, need_weights=False
        )
        tokens = self.normalization(tokens + attended)
        return self.output(tokens.mean(dim=1))


def _dense_head(input_dim, hidden_dim, depth, dropout):
    if int(depth) < 1:
        raise ValueError("dense MLP depth must be positive")
    layers = []
    current = int(input_dim)
    for _ in range(int(depth)):
        layers.extend((
            nn.Linear(current, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        ))
        current = int(hidden_dim)
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


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

    VALID_HEAD_ARCHITECTURES = {
        "linear", "mlp", "deep_mlp", "residual_mlp", "gated_fusion",
        "bilinear_fusion", "attention_fusion",
    }

    def __init__(self, raw_model, fmt_dim=161, embedding_dim=128,
                 auxiliary_dim=64, residual_input="geometry_fmt",
                 head_architecture="mlp", head_hidden_dim=None,
                 head_depth=2, bilinear_rank=32, attention_heads=4,
                 head_dropout=0.0):
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
        self.head_architecture = str(head_architecture)
        if self.head_architecture not in self.VALID_HEAD_ARCHITECTURES:
            raise ValueError(
                f"unknown residual head architecture {self.head_architecture!r}"
            )
        fusion_architectures = {
            "gated_fusion", "bilinear_fusion", "attention_fusion"
        }
        if (self.head_architecture in fusion_architectures
                and self.residual_input != "geometry_fmt"):
            raise ValueError(
                f"{self.head_architecture} requires residual_input='geometry_fmt'"
            )
        hidden_dim = (
            embedding_dim if head_hidden_dim is None else int(head_hidden_dim)
        )
        if hidden_dim < 1:
            raise ValueError("head_hidden_dim must be positive")
        self.fmt_encoder = nn.Sequential(
            nn.Linear(int(fmt_dim), auxiliary_dim),
            nn.LayerNorm(auxiliary_dim),
            nn.GELU(),
        )
        residual_width = (
            embedding_dim + auxiliary_dim
            if self.residual_input in {"geometry_fmt", "dual"} else auxiliary_dim
        )
        self.fusion_head = None
        if self.head_architecture == "linear":
            self.residual_head = nn.Linear(residual_width, 1)
        elif self.head_architecture == "mlp":
            # Keep the historical default byte-for-byte compatible with old
            # checkpoints: one embedding-width hidden layer and no Dropout.
            self.residual_head = nn.Sequential(
                nn.Linear(residual_width, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, 1),
            )
        elif self.head_architecture == "deep_mlp":
            self.residual_head = _dense_head(
                residual_width, hidden_dim, head_depth, head_dropout
            )
        elif self.head_architecture == "residual_mlp":
            self.residual_head = _ResidualMLPHead(
                residual_width, hidden_dim, head_depth, head_dropout
            )
        elif self.head_architecture == "gated_fusion":
            self.residual_head = None
            self.fusion_head = _GatedFusionHead(
                embedding_dim, auxiliary_dim, hidden_dim, head_dropout
            )
        elif self.head_architecture == "bilinear_fusion":
            self.residual_head = None
            self.fusion_head = _LowRankBilinearFusionHead(
                embedding_dim, auxiliary_dim, hidden_dim, bilinear_rank,
                head_dropout,
            )
        elif self.head_architecture == "attention_fusion":
            self.residual_head = None
            self.fusion_head = _AttentionFusionHead(
                embedding_dim, auxiliary_dim, hidden_dim, attention_heads,
                head_dropout,
            )
        self.fmt_only_head = None
        if self.residual_input == "dual":
            self.fmt_only_head = nn.Sequential(
                nn.Linear(auxiliary_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, 1),
            )

    def forward_components(self, pathlines, fmt_features,
                           return_auxiliary=False):
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
        residual_logit = (
            self.fusion_head(geometry.detach(), auxiliary).squeeze(-1)
            if self.fusion_head is not None
            else self.residual_head(residual_input).squeeze(-1)
        )
        if self.fmt_only_head is not None:
            residual_logit = residual_logit + self.fmt_only_head(auxiliary).squeeze(-1)
        if return_auxiliary:
            return raw_logit, residual_logit, auxiliary
        return raw_logit, residual_logit

    def forward(self, pathlines, fmt_features, alpha=1.0):
        raw_logit, residual_logit = self.forward_components(pathlines, fmt_features)
        return raw_logit + float(alpha) * residual_logit


def residual_model_kwargs(model_spec):
    """Normalize checkpoint/config fields for residual model construction."""
    return {
        "embedding_dim": int(model_spec.get("embedding_dim", 128)),
        "auxiliary_dim": int(model_spec.get("auxiliary_dim", 64)),
        "residual_input": str(
            model_spec.get("residual_input", "geometry_fmt")
        ),
        "head_architecture": str(
            model_spec.get("head_architecture", "mlp")
        ),
        "head_hidden_dim": int(model_spec.get(
            "head_hidden_dim", model_spec.get("embedding_dim", 128)
        )),
        "head_depth": int(model_spec.get("head_depth", 2)),
        "bilinear_rank": int(model_spec.get("bilinear_rank", 32)),
        "attention_heads": int(model_spec.get("attention_heads", 4)),
        "head_dropout": float(model_spec.get("head_dropout", 0.0)),
    }


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
