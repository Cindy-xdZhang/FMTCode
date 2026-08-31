"""Supervised geometric classifiers for 3D pathline-cross primitives."""

from __future__ import annotations

import math

import torch
from torch import nn


class _RMSNorm(nn.Module):
    """Root-mean-square normalization without mean subtraction.

    Unlike LayerNorm, this remains informative for a one-dimensional
    auxiliary bottleneck because it does not subtract the sole coordinate
    from itself.
    """

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(width)))

    def forward(self, values):
        scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return values * scale * self.weight


def _balanced_block_widths(total_width, block_count):
    """Split ``total_width`` across blocks without dropping a block."""
    total_width = int(total_width)
    block_count = int(block_count)
    if block_count < 1 or total_width < block_count:
        raise ValueError(
            "blockwise auxiliary projection requires output width >= block count"
        )
    quotient, remainder = divmod(total_width, block_count)
    return tuple(
        quotient + int(index < remainder) for index in range(block_count)
    )


class _BlockwiseAuxiliaryProjection(nn.Module):
    """Project fixed semantic feature blocks before allowing cross-block mixing.

    The same block boundaries and branch network are used by FMT and its
    train-only Raw-PCA control.  Only the meaning of the fixed input values
    differs between the paired arms.
    """

    def __init__(self, input_dim, output_dim, block_dims, architecture,
                 hidden_dim=64):
        super().__init__()
        self.block_dims = tuple(int(value) for value in block_dims)
        if not self.block_dims or any(value < 1 for value in self.block_dims):
            raise ValueError("auxiliary_block_dims must contain positive widths")
        if sum(self.block_dims) != int(input_dim):
            raise ValueError(
                f"auxiliary block widths sum to {sum(self.block_dims)}, "
                f"expected input width {int(input_dim)}"
            )
        output_dims = _balanced_block_widths(output_dim, len(self.block_dims))
        hidden_dim = int(hidden_dim)
        if hidden_dim < 1:
            raise ValueError("blockwise auxiliary hidden width must be positive")
        branches = []
        for block_dim, branch_output_dim in zip(self.block_dims, output_dims):
            if architecture == "blockwise_linear_gelu":
                branch = nn.Sequential(
                    nn.Linear(block_dim, branch_output_dim), nn.GELU(),
                )
            elif architecture == "blockwise_layernorm_gelu":
                branch = nn.Sequential(
                    nn.Linear(block_dim, branch_output_dim),
                    nn.LayerNorm(branch_output_dim),
                    nn.GELU(),
                )
            elif architecture == "blockwise_rmsnorm_gelu":
                branch = nn.Sequential(
                    nn.Linear(block_dim, branch_output_dim),
                    _RMSNorm(branch_output_dim),
                    nn.GELU(),
                )
            elif architecture == "blockwise_mlp_gelu":
                branch = nn.Sequential(
                    nn.Linear(block_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, branch_output_dim),
                    nn.GELU(),
                )
            else:
                raise ValueError(
                    f"unknown blockwise auxiliary projection {architecture!r}"
                )
            branches.append(branch)
        self.branches = nn.ModuleList(branches)

    def forward(self, values):
        if values.shape[-1] != sum(self.block_dims):
            raise ValueError(
                f"expected auxiliary width {sum(self.block_dims)}, "
                f"got {values.shape[-1]}"
            )
        blocks = torch.split(values, self.block_dims, dim=-1)
        return torch.cat(
            tuple(branch(block) for branch, block in zip(self.branches, blocks)),
            dim=-1,
        )


def _auxiliary_projection(input_dim, output_dim, architecture,
                          hidden_dim=64, block_dims=None):
    """Build the paired FMT/Raw-PCA auxiliary projection.

    The historical default is kept byte-for-byte compatible.  Alternative
    projections are shared by both experimental arms; only their fixed input
    representation differs.
    """
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    hidden_dim = int(hidden_dim)
    architecture = str(architecture)
    if output_dim < 1 or hidden_dim < 1:
        raise ValueError("auxiliary projection dimensions must be positive")
    if architecture.startswith("blockwise_"):
        if block_dims is None:
            raise ValueError(
                f"{architecture} requires model.auxiliary_block_dims"
            )
        return _BlockwiseAuxiliaryProjection(
            input_dim, output_dim, block_dims, architecture, hidden_dim
        )
    if architecture == "linear_layernorm_gelu":
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
    if architecture == "linear":
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    if architecture == "linear_gelu":
        return nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.GELU()
        )
    if architecture == "linear_silu":
        return nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.SiLU()
        )
    if architecture == "linear_rmsnorm_gelu":
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            _RMSNorm(output_dim),
            nn.GELU(),
        )
    if architecture == "mlp_layernorm_gelu":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
    if architecture == "mlp_layernorm_silu":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
        )
    if architecture == "mlp_rmsnorm_gelu":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
    raise ValueError(f"unknown auxiliary projection {architecture!r}")


def _initialize_auxiliary_normalization(module, initial_scale=None):
    """Optionally initialize only normalization scales in an auxiliary encoder.

    ``None`` is an exact historical control.  An explicit value changes every
    trainable LayerNorm/RMSNorm affine weight inside ``module`` without
    consuming random numbers or touching linear layers and downstream heads.
    """
    if initial_scale is None:
        return 0
    initial_scale = float(initial_scale)
    if not math.isfinite(initial_scale) or initial_scale < 0.0:
        raise ValueError(
            "auxiliary_normalization_initial_scale must be finite and "
            "non-negative"
        )
    initialized = 0
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.LayerNorm):
                if child.weight is not None:
                    child.weight.fill_(initial_scale)
                    initialized += 1
            elif isinstance(child, _RMSNorm):
                child.weight.fill_(initial_scale)
                initialized += 1
    if initialized == 0:
        raise ValueError(
            "auxiliary_normalization_initial_scale requires an auxiliary "
            "projection with trainable LayerNorm or RMSNorm"
        )
    return initialized


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


def _head_normalization(architecture, width):
    """Build one registered residual-head normalization layer."""
    architecture = str(architecture).lower()
    if architecture == "layernorm":
        return nn.LayerNorm(int(width))
    if architecture == "rmsnorm":
        return _RMSNorm(int(width))
    if architecture == "none":
        return nn.Identity()
    raise ValueError(
        "head_normalization must be 'layernorm', 'rmsnorm', or 'none'"
    )


def _head_activation(architecture):
    """Build one registered residual-head activation layer."""
    architecture = str(architecture).lower()
    if architecture == "gelu":
        return nn.GELU()
    if architecture == "silu":
        return nn.SiLU()
    if architecture == "relu":
        return nn.ReLU()
    raise ValueError(
        "head_activation must be 'gelu', 'silu', or 'relu'"
    )


def _dense_head(input_dim, hidden_dim, depth, dropout,
                normalization="layernorm", activation="gelu"):
    if int(depth) < 1:
        raise ValueError("dense MLP depth must be positive")
    layers = []
    current = int(input_dim)
    for _ in range(int(depth)):
        layers.extend((
            nn.Linear(current, int(hidden_dim)),
            _head_normalization(normalization, hidden_dim),
            _head_activation(activation),
            nn.Dropout(float(dropout)),
        ))
        current = int(hidden_dim)
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


def _last_linear(module):
    """Return the terminal registered Linear layer of one output module."""
    if module is None:
        return None
    linears = [child for child in module.modules()
               if isinstance(child, nn.Linear)]
    if not linears:
        raise ValueError("residual output module contains no Linear layer")
    return linears[-1]


def _initialize_residual_outputs(modules, initialization="default", scale=1.0):
    """Initialize only inference-time residual output layers.

    Small or zero terminal weights preserve the frozen Raw logit at the start
    of training while leaving every upstream representation layer unchanged.
    The same rule can therefore be paired exactly between FMT and Raw-PCA.
    """
    initialization = str(initialization).lower()
    scale = float(scale)
    if initialization not in {"default", "zero", "normal", "xavier_uniform"}:
        raise ValueError(
            "residual_output_initialization must be 'default', 'zero', "
            "'normal', or 'xavier_uniform'"
        )
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "residual_output_initialization_scale must be finite and non-negative"
        )
    if initialization in {"normal", "xavier_uniform"} and scale <= 0.0:
        raise ValueError(
            f"{initialization} residual initialization requires a positive scale"
        )
    if initialization == "default":
        return

    terminal_layers = []
    seen = set()
    for module in modules:
        layer = _last_linear(module)
        if layer is not None and id(layer) not in seen:
            seen.add(id(layer))
            terminal_layers.append(layer)
    if not terminal_layers:
        raise ValueError("no residual output layers were available to initialize")

    for layer in terminal_layers:
        if initialization == "zero":
            nn.init.zeros_(layer.weight)
        elif initialization == "normal":
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
        else:
            nn.init.xavier_uniform_(layer.weight, gain=scale)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


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
                 head_dropout=0.0, head_normalization="layernorm",
                 head_activation="gelu",
                 auxiliary_projection="linear_layernorm_gelu",
                 auxiliary_hidden_dim=64, auxiliary_block_dims=None,
                 auxiliary_normalization_initial_scale=None,
                 auxiliary_dropout=0.0,
                 auxiliary_classifier_architecture="none",
                 auxiliary_classifier_hidden_dim=64,
                 residual_output_initialization="default",
                 residual_output_initialization_scale=1.0):
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
        self.head_normalization = str(head_normalization).lower()
        self.head_activation = str(head_activation).lower()
        if self.head_architecture != "deep_mlp" and (
            self.head_normalization != "layernorm"
            or self.head_activation != "gelu"
        ):
            raise ValueError(
                "head_normalization/head_activation overrides require "
                "head_architecture='deep_mlp'"
            )
        self.auxiliary_projection = str(auxiliary_projection)
        self.auxiliary_hidden_dim = int(auxiliary_hidden_dim)
        self.auxiliary_block_dims = (
            None if auxiliary_block_dims is None
            else tuple(int(value) for value in auxiliary_block_dims)
        )
        self.fmt_encoder = _auxiliary_projection(
            int(fmt_dim), auxiliary_dim, self.auxiliary_projection,
            self.auxiliary_hidden_dim, self.auxiliary_block_dims,
        )
        self.auxiliary_normalization_initial_scale = (
            None if auxiliary_normalization_initial_scale is None
            else float(auxiliary_normalization_initial_scale)
        )
        self.auxiliary_normalization_layer_count = (
            _initialize_auxiliary_normalization(
                self.fmt_encoder,
                self.auxiliary_normalization_initial_scale,
            )
        )
        auxiliary_dropout = float(auxiliary_dropout)
        if not 0.0 <= auxiliary_dropout < 1.0:
            raise ValueError("auxiliary_dropout must be in [0, 1)")
        # This regularizer acts only on the projected auxiliary representation,
        # before it is fused with the frozen Raw geometry.  Dropout has no
        # parameters or checkpoint state, so p=0 remains byte-compatible with
        # historical residual checkpoints.
        self.auxiliary_dropout = nn.Dropout(auxiliary_dropout)
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
                residual_width, hidden_dim, head_depth, head_dropout,
                self.head_normalization, self.head_activation,
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
        # Build the training-only classifier after every inference module so
        # enabling deep supervision cannot change their random initialization.
        self.auxiliary_classifier_architecture = str(
            auxiliary_classifier_architecture
        ).lower()
        auxiliary_classifier_hidden_dim = int(auxiliary_classifier_hidden_dim)
        if auxiliary_classifier_hidden_dim < 1:
            raise ValueError("auxiliary_classifier_hidden_dim must be positive")
        if self.auxiliary_classifier_architecture == "none":
            self.auxiliary_classifier = None
        elif self.auxiliary_classifier_architecture == "linear":
            self.auxiliary_classifier = nn.Linear(auxiliary_dim, 1)
        elif self.auxiliary_classifier_architecture == "mlp":
            self.auxiliary_classifier = nn.Sequential(
                nn.Linear(auxiliary_dim, auxiliary_classifier_hidden_dim),
                nn.GELU(),
                nn.Linear(auxiliary_classifier_hidden_dim, 1),
            )
        else:
            raise ValueError(
                "auxiliary_classifier_architecture must be 'none', "
                "'linear', or 'mlp'"
            )
        self.residual_output_initialization = str(
            residual_output_initialization
        ).lower()
        self.residual_output_initialization_scale = float(
            residual_output_initialization_scale
        )
        _initialize_residual_outputs(
            (self.residual_head, self.fusion_head, self.fmt_only_head),
            self.residual_output_initialization,
            self.residual_output_initialization_scale,
        )

    def forward_components(self, pathlines, fmt_features,
                           return_auxiliary=False):
        if fmt_features is None:
            raise ValueError("FMT residual classifier requires fmt_features")
        with torch.no_grad():
            geometry = self.raw_model.geometry(pathlines)
            raw_logit = self.raw_model.head(geometry).squeeze(-1)
        auxiliary = self.auxiliary_dropout(self.fmt_encoder(fmt_features))
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

    def auxiliary_classification_logits(self, auxiliary):
        """Classify the projected auxiliary representation during training.

        The auxiliary head is deliberately absent from the inference fusion.
        It only provides direct supervision to the shared projection, and the
        exact same head is trained for the FMT and train-only Raw-PCA arms.
        """
        if self.auxiliary_classifier is None:
            raise RuntimeError(
                "auxiliary classification logits require a configured "
                "auxiliary classifier"
            )
        return self.auxiliary_classifier(auxiliary).squeeze(-1)

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
        "head_normalization": str(model_spec.get(
            "head_normalization", "layernorm"
        )),
        "head_activation": str(model_spec.get(
            "head_activation", "gelu"
        )),
        "auxiliary_projection": str(model_spec.get(
            "auxiliary_projection", "linear_layernorm_gelu"
        )),
        "auxiliary_hidden_dim": int(model_spec.get(
            "auxiliary_hidden_dim", 64
        )),
        "auxiliary_block_dims": (
            None if model_spec.get("auxiliary_block_dims") is None
            else [int(value) for value in model_spec["auxiliary_block_dims"]]
        ),
        "auxiliary_normalization_initial_scale": (
            None
            if model_spec.get("auxiliary_normalization_initial_scale") is None
            else float(model_spec["auxiliary_normalization_initial_scale"])
        ),
        "auxiliary_dropout": float(model_spec.get("auxiliary_dropout", 0.0)),
        "auxiliary_classifier_architecture": str(model_spec.get(
            "auxiliary_classifier_architecture", "none"
        )),
        "auxiliary_classifier_hidden_dim": int(model_spec.get(
            "auxiliary_classifier_hidden_dim", 64
        )),
        "residual_output_initialization": str(model_spec.get(
            "residual_output_initialization", "default"
        )),
        "residual_output_initialization_scale": float(model_spec.get(
            "residual_output_initialization_scale", 1.0
        )),
    }


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
