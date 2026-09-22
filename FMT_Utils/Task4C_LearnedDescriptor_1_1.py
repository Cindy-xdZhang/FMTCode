"""Learned per-line descriptors: a transformer where FMT uses a DFT/DCT.

FMT's per-line block is a *fixed* map: difference the track, transform it, and
read off SO(3) invariants of the spectrum (23-D for DFT, 15-D for DCT at k=6).
This module replaces that map with the OctVAE-style CLS transformer, keeping
everything else about the graph identical, so the comparison isolates the
descriptor.

One property is deliberately given up.  The DFT/DCT blocks are rotation
invariant *by construction*; a transformer on raw 3-vectors is not.  What is
kept is translation invariance -- the encoder is fed the same temporal
difference sequence FMT uses, and neighbours are expressed relative to their
root first -- and the 72-D direction spectrum was never invariant either, so the
model already depends on bundle orientation.  Whether the invariance was load
bearing is exactly what this experiment measures.

Cost.  FMT precomputes its blocks once; a learned descriptor must be recomputed
every step, and the neighbour block genuinely differs per (root, neighbour)
pair, so a bundle needs 27 + 27*6 encoder calls rather than 27.  Batches are
therefore smaller here than the 4096 used elsewhere.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from FMT_Utils.FMT_P35_NormFrequency_3_1 import direction_spectrum
from FMT_Utils.OctahedralGroup_3D import octahedral_group
from FMT_Utils.Task4C_HierGraph_1_1 import DIRECTION, FREQUENCIES, GATv2, block


class LineTransformer(nn.Module):
    """``[N, T, 3]`` track -> ``[N, out_dim]`` descriptor, via a CLS token."""

    def __init__(self, out_dim, seq_len, d_model=64, nhead=4, layers=2,
                 dim_ff=128, dropout=0.0, grad_checkpoint=False, chunk=None):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        if chunk:
            self.CHUNK = int(chunk)
        self.embed = nn.Linear(3, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_dim)

    # A bundle contributes 27 root tracks and 27*6 neighbour tracks, so the
    # sequence batch overruns the 65535 limit of the fused attention kernel well
    # before the bundle batch is large.  Chunk rather than cap the batch size.
    CHUNK = 32768

    def _encode(self, seq):
        h = self.embed(seq)
        h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
        h = self.norm(self.transformer(h + self.pos_embed[:, :h.shape[1]]))
        return self.out(h[:, 0])

    def forward(self, seq):
        # Activations dominate memory here: a batch of 4096 bundles is ~110k
        # sequences through the whole stack.  Recomputing them in the backward
        # pass keeps peak memory at roughly one chunk instead of the whole batch,
        # for about a third more compute.
        if self.grad_checkpoint and self.training and seq.requires_grad is False:
            seq = seq.detach().requires_grad_(False)
        if seq.shape[0] <= self.CHUNK:
            if self.grad_checkpoint and self.training:
                return checkpoint(self._encode, seq, use_reentrant=False)
            return self._encode(seq)
        parts = []
        for part in seq.split(self.CHUNK, dim=0):
            parts.append(checkpoint(self._encode, part, use_reentrant=False)
                         if (self.grad_checkpoint and self.training) else self._encode(part))
        return torch.cat(parts, 0)


class MultiRootLearnedGraph(nn.Module):
    """`MultiRootFMTGraph` with the DFT/DCT block swapped for `LineTransformer`."""

    def __init__(self, seq_len, representation_dim=256, descriptor_dim=32, heads=4,
                 dropout=0.15, use_direction=True, head_hidden=128, pool="meanmax",
                 use_orbit=False, root_dropout=0.0, d_model=64, enc_layers=2,
                 enc_heads=4, dim_ff=128, mode="pair", rot_aug="none",
                 grad_checkpoint=False, enc_chunk=None):
        super().__init__()
        # The DFT/DCT blocks are rotation invariant by construction; this encoder
        # is not.  Rotating the whole bundle during training is the direct
        # substitute -- the label does not depend on orientation, so the encoder
        # can only lose by keying on it.  'oh' uses the 48 octahedral elements
        # (the lattice's own symmetry), 'so3' samples continuously.
        self.rot_aug = rot_aug
        # `centre_dir` arrives standardised by the training-split statistics.  When
        # augmentation rotates the track the cached block no longer matches, so it
        # is rebuilt -- and must then be standardised with the *same* statistics or
        # it enters the network on a completely different scale.
        self.register_buffer("dir_mean", torch.zeros(DIRECTION), persistent=False)
        self.register_buffer("dir_std", torch.ones(DIRECTION), persistent=False)
        if rot_aug == "oh":
            matrices, _ = octahedral_group("oh")
            self.register_buffer("group", torch.as_tensor(matrices, dtype=torch.float32),
                                 persistent=False)
        self.use_direction, self.pool, self.use_orbit = use_direction, pool, use_orbit
        self.root_dropout = root_dropout
        # 'pair' mirrors FMT exactly: the neighbour block encodes the track of j
        # *relative to* its root i, so a bundle costs 27 + 27*6 encoder calls.
        # 'line' encodes every track once (27 calls) and lets the GATv2 -- which
        # already sees root and neighbour together -- model the relation, which is
        # 7x cheaper and makes long schedules affordable.
        self.mode = mode
        steps = seq_len - 1                              # one temporal difference
        self.centre_encoder = LineTransformer(descriptor_dim, steps, d_model,
                                              enc_heads, enc_layers, dim_ff, dropout,
                                              grad_checkpoint, enc_chunk)
        self.neighbour_encoder = LineTransformer(descriptor_dim, steps, d_model,
                                                 enc_heads, enc_layers, dim_ff, dropout,
                                                 grad_checkpoint, enc_chunk)
        if use_orbit:
            self.orbit_embed = nn.Embedding(5, representation_dim)
            nn.init.normal_(self.orbit_embed.weight, std=0.02)
        centre_in = descriptor_dim + (DIRECTION if use_direction else 0)
        self.centre_proj = block(centre_in, representation_dim, dropout)
        self.neighbour_proj = block(descriptor_dim, representation_dim, dropout)
        self.gat = GATv2(representation_dim, representation_dim, representation_dim,
                         heads, dropout)
        self.token = nn.Sequential(block(2 * representation_dim, representation_dim, dropout),
                                   block(representation_dim, representation_dim, dropout))
        width = representation_dim * (2 if pool == "meanmax" else 1)
        self.fuse = block(width, representation_dim, dropout)
        self.head = nn.Sequential(block(representation_dim, head_hidden, dropout),
                                  nn.Linear(head_hidden, 2))

    def set_direction_stats(self, mean, std):
        self.dir_mean.copy_(torch.as_tensor(mean, dtype=self.dir_mean.dtype))
        self.dir_std.copy_(torch.as_tensor(std, dtype=self.dir_std.dtype))

    def _rotations(self, size, device):
        if self.rot_aug == "oh":
            return self.group[torch.randint(len(self.group), (size,), device=device)]
        noise = torch.randn(size, 3, 3, device=device)
        q, r = torch.linalg.qr(noise)
        q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
        flip = torch.where(torch.linalg.det(q) < 0, -1.0, 1.0)      # keep det = +1
        return torch.cat((q[:, :, :2], q[:, :, 2:] * flip[:, None, None]), dim=-1)

    def forward(self, batch):
        track, index, mask = batch["geometry"], batch["nbr_index"].long(), batch["mask"]
        size, lines = track.shape[0], track.shape[1]
        if self.training and self.root_dropout > 0:
            kept = mask & (torch.rand_like(mask, dtype=torch.float) >= self.root_dropout)
            mask = torch.where(kept.any(1, keepdim=True), kept, mask)

        track = track.float()
        direction = batch["centre_dir"] if self.use_direction else None
        if self.training and self.rot_aug != "none":
            rotation = self._rotations(size, track.device)
            track = torch.einsum("bltc,bdc->bltd", track, rotation)
            if self.use_direction:
                # the cached spectrum belongs to the unrotated track, so rebuild it
                direction = direction_spectrum(
                    track.reshape(size * lines, track.shape[2], 3), FREQUENCIES
                ).reshape(size, lines, DIRECTION)
                direction = (direction - self.dir_mean) / self.dir_std
        centre_signal = track[:, :, 1:] - track[:, :, :-1]
        rows = torch.arange(size, device=index.device)[:, None, None]
        if self.mode == "pair":
            relative = track[rows, index] - track[:, :, None]      # [B,L,6,T,3]
            neighbour_signal = relative[:, :, :, 1:] - relative[:, :, :, :-1]

        if self.mode == "line":
            per_line = self.centre_encoder(centre_signal.reshape(size * lines, -1, 3))
            c = per_line
        else:
            c = self.centre_encoder(centre_signal.reshape(size * lines, -1, 3))
        if self.use_direction:
            c = torch.cat((c, direction.reshape(size * lines, DIRECTION).float()), -1)
        c = self.centre_proj(c)
        if self.use_orbit:
            c = c + self.orbit_embed(batch["orbit"].reshape(size * lines).long())
        if self.mode == "line":
            grid = per_line.reshape(size, lines, -1)
            n = grid[rows, index].reshape(size * lines, index.shape[2], -1)
        else:
            n = self.neighbour_encoder(
                neighbour_signal.reshape(size * lines * index.shape[2], -1, 3)
            ).reshape(size * lines, index.shape[2], -1)
        pooled, _ = self.gat(c, self.neighbour_proj(n))
        tokens = self.token(torch.cat((c, pooled), -1)).reshape(size, lines, -1)

        weight = mask[..., None].to(tokens.dtype)
        mean = (tokens * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        if self.pool == "meanmax":
            maximum = tokens.masked_fill(~mask[..., None], float("-inf")).amax(1)
            merged = torch.cat((mean, torch.nan_to_num(maximum, neginf=0.0)), -1)
        else:
            merged = mean
        return self.head(self.fuse(merged))

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        encoder = sum(p.numel() for p in self.centre_encoder.parameters()) + \
                  sum(p.numel() for p in self.neighbour_encoder.parameters())
        return {"total": int(total), "backbone": int(total - head), "head": int(head),
                "descriptor_encoder": int(encoder)}
