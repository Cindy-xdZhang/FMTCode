"""Four-class neural baselines for channel-flow Task4-b."""

from __future__ import annotations

import torch
from torch import nn

from FMT_Utils.PathlineClassifier_3D import PathlineGeometryEncoder3D


class PathlineMulticlassClassifier3D(nn.Module):
    """Classify a seven-line primitive into four vortex types.

    ``raw`` and ``raw_wide`` are no-FMT controls. ``fmt_only`` is the literal
    FMT-to-neural-network diagnostic requested for Task4-b. ``raw_fmt`` mirrors
    Task3's effective design by combining local Raw geometry with fixed FMT.
    """

    VALID_VARIANTS = {"raw", "raw_wide", "fmt_only", "raw_fmt"}

    def __init__(
        self,
        *,
        variant: str,
        fmt_dim: int,
        num_classes: int = 4,
        temporal_width: int = 64,
        embedding_dim: int = 128,
        auxiliary_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        variant = str(variant)
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown multiclass variant {variant!r}")
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least two")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.variant = variant
        self.num_classes = int(num_classes)
        embedding_dim = int(embedding_dim)
        auxiliary_dim = int(auxiliary_dim)

        self.geometry = None
        if variant in {"raw", "raw_wide", "raw_fmt"}:
            self.geometry = PathlineGeometryEncoder3D(
                temporal_width=int(temporal_width),
                embedding_dim=embedding_dim,
            )

        self.auxiliary = None
        if variant in {"fmt_only", "raw_fmt"}:
            self.auxiliary = nn.Sequential(
                nn.Linear(int(fmt_dim), auxiliary_dim),
                nn.LayerNorm(auxiliary_dim),
                nn.GELU(),
            )
        elif variant == "raw_wide":
            # Deliberately larger than the FMT projection so capacity alone is
            # not a favorable explanation for Raw+FMT.
            self.auxiliary = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, auxiliary_dim),
                nn.LayerNorm(auxiliary_dim),
                nn.GELU(),
            )

        if variant == "raw":
            fused_dim = embedding_dim
        elif variant == "fmt_only":
            fused_dim = auxiliary_dim
        else:
            fused_dim = embedding_dim + auxiliary_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(embedding_dim, self.num_classes),
        )

    def forward(
        self, pathlines: torch.Tensor, fmt_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.variant == "fmt_only":
            if fmt_features is None:
                raise ValueError("fmt_only requires fmt_features")
            fused = self.auxiliary(fmt_features)
        else:
            if self.geometry is None:
                raise RuntimeError("Raw variant has no geometry encoder")
            geometry = self.geometry(pathlines)
            if self.variant == "raw":
                fused = geometry
            elif self.variant == "raw_wide":
                fused = torch.cat((geometry, self.auxiliary(geometry)), dim=-1)
            else:
                if fmt_features is None:
                    raise ValueError("raw_fmt requires fmt_features")
                fused = torch.cat(
                    (geometry, self.auxiliary(fmt_features)), dim=-1
                )
        logits = self.head(fused)
        if logits.shape[-1] != self.num_classes:
            raise RuntimeError("multiclass head returned the wrong logit width")
        return logits


def trainable_parameter_count(module: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )
