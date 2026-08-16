import torch
import torch.nn as nn


def group_same_timestep(xyz: torch.Tensor, x: torch.Tensor, LSteps: int):
    """
    Same-timestep cross-line grouping for pathline-cross primitives.

    Our data is not an unordered point cloud, so generic KNN makes no sense here:
    for every point (line i, timestep t) the neighbor set is the K points
    {(line j, timestep t) : j = 0..K-1} (self included) -- the other lines of the
    same primitive at the SAME timestep. Same-time differences cancel any
    time-dependent translation of the reference frame, which is why this grouping
    is the geometrically meaningful one (see docs/first_principles_analysis.md P0).

    Flattening convention: N = K*L, flat index = line_id * L + t
    (i.e. a reshape of [K, L, ...]).

    Inputs:
        xyz: [B, N, 3]
        x:   [B, N, C]
    Returns:
        lc_xyz  [B, N, 3], lc_x [B, N, C]
        knn_xyz [B, N, K, 3], knn_x [B, N, K, C]  (neighbors ordered by line id)

    Implemented with pure reshape/expand (no index gather). Replaces the removed
    `GeoLinePicker` class with numerically identical output.
    """
    B, N, _ = xyz.shape
    L = int(LSteps)
    assert N % L == 0, f"N must be divisible by L={L}, got N={N}"
    K = N // L

    def gather_neighbors(t: torch.Tensor) -> torch.Tensor:
        C = t.shape[-1]
        # [B, K, L, C] -> [B, L, K, C]: for each timestep, the K points of all lines
        per_t = t.view(B, K, L, C).permute(0, 2, 1, 3)
        # every center point (i, t) shares the same K-line neighbor set of its timestep
        out = per_t.unsqueeze(1).expand(B, K, L, K, C)  # [B, K_center, L, K_neighbor, C]
        return out.reshape(B, N, K, C)

    return xyz, x, gather_neighbors(xyz), gather_neighbors(x)


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.GELU())

    def forward(self, knn_x_w):
        # knn_x_w: (B, out_dim, N, K)
        # Feature Aggregation (Pooling along the KNN dimension as merge nerighboring information)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x


# PosE for Raw-point Embedding
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        # xyz: (B,  3, N)
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim, device=xyz.device, dtype=torch.float32)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        return position_embed


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim, device=knn_xyz.device, dtype=torch.float32)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w

# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        diff_x = knn_x - mean_x
        std_x = torch.std(diff_x, dim=2, keepdim=True, unbiased=False)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        diff_xyz = knn_xyz - mean_xyz
        std_xyz = torch.std(diff_xyz, dim=2, keepdim=True, unbiased=False)

        knn_x = diff_x / (std_x + 1e-5)
        knn_xyz = diff_xyz / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x_w


class TemporalDFT(nn.Module):
    """
    沿时间维 L 做离散傅里叶变换 (DFT) 的可学习时序模块：
      x --rfft--> X(f) --(learnable complex filter)--> Y(f) --irfft--> y
    设计目标：把“轨迹是有序序列”的归纳偏置放到 encoder 的 temporal head 里，
    避免把 N=K*L 当成无序点云直接做全局池化。
    注意：本模块含可学习参数（复数滤波权重 + BatchNorm），启用它的 FMT 不是无参数编码器。
    """
    def __init__(self, channels: int, L: int, dropout: float = 0.0, residual: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.L = int(L)
        self.residual = bool(residual)
        # rfft 的频点数
        self.F = self.L // 2 + 1

        # 逐通道、逐频点的复数权重：Y = X * W
        self.weight_real = nn.Parameter(torch.ones(self.channels, self.F))
        self.weight_imag = nn.Parameter(torch.zeros(self.channels, self.F))

        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        # BN1d 支持 [B, C, L]
        self.norm = nn.BatchNorm1d(self.channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        return: [B, C, L]
        """
        assert x.dim() == 3, f"TemporalDFT expects [B,C,L], got {tuple(x.shape)}"
        B, C, L = x.shape
        assert C == self.channels, f"channels mismatch: expect {self.channels}, got {C}"
        assert L == self.L, f"L mismatch: expect {self.L}, got {L}"

        X = torch.fft.rfft(x, dim=-1, norm='ortho')  # [B, C, F], complex
        W = torch.complex(self.weight_real, self.weight_imag).unsqueeze(0)  # [1, C, F]
        Y = X * W
        y = torch.fft.irfft(Y, n=self.L, dim=-1, norm='ortho')  # [B, C, L], real
        y = self.dropout(self.act(self.norm(y)))
        return (x + y) if self.residual else y


class FMT(nn.Module):
    """
    Pathline-cross primitive encoder (per-primitive token).

    Input convention (see `group_same_timestep`):
        xyz: [B, N=K*L, 3]  flattened line-major (flat = line_id * L + t), 3 = (x, y, t)
        x:   [B, 3, N]      same coordinates, channel-first (fed to PosE_Initial)
    Output:
        [B, out_dim], out_dim = embed_dim * 2**num_stages

    Pipeline per stage: same-timestep cross-line grouping -> LGA (normalize + PosE_Geo
    weighting) -> max+mean pooling (+ BatchNorm+GELU).
    temporal_head="dft" appends the learnable TemporalDFT head; temporal_head=None
    falls back to global max+mean pooling (the Task1 clustering configuration).

    NOTE: Pooling contains BatchNorm (learnable affine + running stats), so this
    encoder is NOT strictly parameter-free; behavior differs between train/eval
    mode (see docs/first_principles_analysis.md P1).
    """
    def __init__(self, PathlineLtimesteps:int, num_stages, embed_dim, alpha, beta, temporal_head: str = "dft"):
        super().__init__()
        self.PathlineLtimesteps = PathlineLtimesteps
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta
        self.temporal_head = str(temporal_head).lower()

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)

        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling

        out_dim = self.embed_dim

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))

        # temporal head：在最后把 N=K*L 还原为时序并做频域学习
        self.out_dim = out_dim
        if self.temporal_head == "dft":
            self.temporal_dft = TemporalDFT(channels=self.out_dim, L=self.PathlineLtimesteps, dropout=0.0, residual=True)
        else:
            self.temporal_dft = None


    def forward(self, xyz, x):
        # xyz: (B, N=K*Ltimesteps, 3)
        # x:   (B, 3, N)
        # Raw-point Embedding
        x = self.raw_point_embed(x)
        B, embed_dim, N = x.shape
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # Same-timestep cross-line grouping (replaces the removed GeoLinePicker)
            xyz, lc_x, knn_xyz, knn_x = group_same_timestep(xyz, x.permute(0, 2, 1), self.PathlineLtimesteps)
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        # x: [B, out_dim, N]
        x = x.view(B, -1, N)

        # Temporal head (DFT along L): N must be K*L
        if self.temporal_dft is not None:
            L = int(self.PathlineLtimesteps)
            assert N % L == 0, f"N must be divisible by L={L}, got N={N}"
            K = N // L
            # 还原为 [B, C, K, L]
            x_ckl = x.reshape(B, self.out_dim, K, L)
            # 逐线做 DFT，再跨线聚合得到时序 token: [B, C, L]
            x_line = x_ckl.permute(0, 2, 1, 3).reshape(B * K, self.out_dim, L)  # [B*K, C, L]
            x_line = self.temporal_dft(x_line)  # [B*K, C, L]
            x_ckl = x_line.reshape(B, K, self.out_dim, L).permute(0, 2, 1, 3)  # [B, C, K, L]
            x_cl = x_ckl.max(dim=2)[0] + x_ckl.mean(dim=2)  # [B, C, L]
            # time pooling -> token
            every_cross_feature = x_cl.max(dim=-1)[0] + x_cl.mean(dim=-1)  # [B, C]
            return every_cross_feature

        # Fallback: old global pooling (treat as unordered points)
        every_cross_feature = x.max(-1)[0] + x.mean(-1)
        return every_cross_feature
