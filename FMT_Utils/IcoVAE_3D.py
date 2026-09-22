"""Icosahedral SIREN-VAE for 13-line 3D pathline primitives.

The 3D counterpart of the 2D ``D8`` SIREN-VAE documented in
``docs/2d_kmeans_d8.md``, with the neighbour star raised from 6 octahedral faces
to 12 icosahedral vertices and the group from ``O_h`` (48) to ``I_h`` (120).

Nothing here sees a label.  The IVD reference travels with the cache purely as a
metric, and training, checkpoint choice and read-out choice are all label-free.

Latent: ``z = [z_inv | z_eq]``.  A positive-pairs-only contrastive term pulls
``z_inv`` together across group views; ``z_eq`` is unconstrained and absorbs the
orientation so the decoder can still reconstruct a rotated star.  Because the
contrastive term has no negatives its optimum is ``z_inv = const``, so a VICReg
variance hinge and covariance penalty put a floor under it.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from FMT_Utils.IcosahedralGroup_3D import (
    LINE_COUNT, NEIGHBOUR_COUNT, VECTOR_DIM, apply_group, apply_group_shells,
    group_tensors, random_group, random_so3)
from FMT_Utils.SirenVAE_3D import (
    PureTransformerEncoder3D, ResTransformerEncoder3D, SIRENDecoder3D)


# ---------------------------------------------------------------------------
# Signal and the equivariant normaliser
# ---------------------------------------------------------------------------

def build_signal_ico(geometry):
    """``[N,13,T,3]`` traced positions -> ``[N,13,T-1,3]`` group-covariant signal.

        channel 0     : d/dt x_0(t)                  centre velocity
        channels 1-12 : d/dt ( x_i(t) - x_0(t) )     neighbour rate rel. centre

    Both are translation invariant and rotate as vectors, which is what the
    group action requires.
    """
    paths = np.asarray(geometry, dtype=np.float32)
    if paths.ndim != 4 or paths.shape[-1] != VECTOR_DIM or (paths.shape[1] - 1) % NEIGHBOUR_COUNT:
        raise ValueError(f"expected [N,1+12k,T,3], got {tuple(paths.shape)}")
    centre = paths[:, :1]
    relative = paths[:, 1:] - centre
    signal = np.concatenate((np.diff(centre, axis=2), np.diff(relative, axis=2)), axis=1)
    if not np.isfinite(signal).all():
        raise ValueError("non-finite pathline signal")
    return np.ascontiguousarray(signal, dtype=np.float32)


def apply_norm_stats_ico(signal, stats, clip_sigma=5.0):
    """Divide by scalar RMS (no mean subtraction) and clip by vector magnitude.

    Both operations commute with a rotation, and the scale is shared across all
    twelve neighbours so it also commutes with the channel permutation.
    """
    centre_rms, neighbour_rms = stats
    scaled = torch.cat((signal[:, :1] / centre_rms, signal[:, 1:] / neighbour_rms), dim=1)
    if clip_sigma and clip_sigma > 0:
        limit = float(clip_sigma) * np.sqrt(VECTOR_DIM)
        norm = scaled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scaled = scaled * torch.clamp(limit / norm, max=1.0)
    return scaled


def normalize_signal_ico(signal, clip_sigma=5.0):
    if signal.dim() != 4 or signal.shape[-1] != VECTOR_DIM or (signal.shape[1] - 1) % NEIGHBOUR_COUNT:
        raise ValueError(f"signal must be [N,1+12k,T,3], got {tuple(signal.shape)}")
    centre_rms = signal[:, :1].pow(2).mean().sqrt().clamp_min(1e-8)
    neighbour_rms = signal[:, 1:].pow(2).mean().sqrt().clamp_min(1e-8)
    stats = (centre_rms.detach(), neighbour_rms.detach())
    return apply_norm_stats_ico(signal, stats, clip_sigma), stats


def channel_balance_weights_ico(signal_n):
    """One weight for the centre, one shared across the twelve neighbours.

    Sharing the neighbour weight is what keeps the reconstruction loss invariant
    under the channel permutation; a per-channel weight would not be.
    """
    centre = signal_n[:, :1].var(dim=2, unbiased=False).mean().clamp_min(1e-8)
    neighbour = signal_n[:, 1:].var(dim=2, unbiased=False).mean().clamp_min(1e-8)
    weights = torch.ones(signal_n.shape[1], device=signal_n.device, dtype=signal_n.dtype)
    weights[0] = 1.0 / centre
    weights[1:] = 1.0 / neighbour
    return (weights / weights.mean()).view(1, signal_n.shape[1], 1, 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IcosahedralSirenVAE3D(nn.Module):
    """VAE over ``[B, 13, T, 3]`` primitives with a split ``[z_inv | z_eq]`` latent."""

    def __init__(self, steps, z_inv_dim=16, z_eq_dim=12, conv_channels=(64, 128),
                 d_model=128, nhead=4, trans_layers=3, dim_ff=256,
                 dec_hidden=128, dec_layers=4, w0=30.0, w0_first=30.0,
                 encoder_kind="conv", lines=LINE_COUNT):
        super().__init__()
        self.lines = int(lines)
        self.steps = int(steps)
        self.z_inv_dim, self.z_eq_dim = int(z_inv_dim), int(z_eq_dim)
        self.z_dim = self.z_inv_dim + self.z_eq_dim
        self.channels = self.lines * VECTOR_DIM
        self.encoder_kind = encoder_kind
        if encoder_kind == "pure":
            self.encoder = PureTransformerEncoder3D(
                self.channels, self.steps, self.z_dim, d_model=d_model, nhead=nhead,
                num_layers=trans_layers, dim_ff=dim_ff)
        elif encoder_kind == "conv":
            self.encoder = ResTransformerEncoder3D(
                self.channels, self.steps, self.z_dim, conv_channels=conv_channels,
                d_model=d_model, nhead=nhead, num_layers=trans_layers, dim_ff=dim_ff)
        else:
            raise ValueError(f"encoder_kind must be 'conv' or 'pure', got {encoder_kind!r}")
        self.decoder = SIRENDecoder3D(self.z_dim, self.channels, hidden_dim=dec_hidden,
                                      num_layers=dec_layers, w0_first=w0_first, w0=w0)
        self.register_buffer("t_grid", torch.linspace(0.0, 1.0, self.steps).view(1, self.steps, 1),
                             persistent=False)

    def _flatten(self, signal):
        batch, lines, steps, comps = signal.shape
        return signal.permute(0, 1, 3, 2).reshape(batch, lines * comps, steps)

    def encode(self, signal):
        return self.encoder(self._flatten(signal))

    def decode(self, latent):
        decoded = self.decoder(self.t_grid.expand(latent.shape[0], -1, -1), latent)
        return decoded.view(latent.shape[0], self.steps, self.lines,
                            VECTOR_DIM).permute(0, 2, 1, 3)

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.encoder.parameters())
        return {"total": int(total), "encoder": int(encoder), "decoder": int(total - encoder)}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def multi_view_contrastive_loss(views):
    """Mean ``1 - cosine`` over every pair of views of the same primitive."""
    total, count = views[0].new_zeros(()), 0
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            total = total + (1.0 - F.cosine_similarity(views[i], views[j], dim=-1)).mean()
            count += 1
    return total / max(count, 1)


def zinv_variance_covariance(views, target_std=0.5, eps=1e-4):
    variance_total, covariance_total = views[0].new_zeros(()), views[0].new_zeros(())
    for z in views:
        std = torch.sqrt(z.var(dim=0) + eps)
        variance_total = variance_total + F.relu(target_std - std).mean()
        centred = z - z.mean(dim=0)
        covariance = (centred.T @ centred) / max(len(z) - 1, 1)
        off = covariance - torch.diag_embed(torch.diagonal(covariance))
        covariance_total = covariance_total + off.pow(2).sum() / z.shape[1]
    return variance_total / len(views), covariance_total / len(views)


def vae_ico_loss(recon, target, mu, logvar, contrastive, beta=1e-4,
                 lambda_recon=1.0, lambda_contrast=15.0, channel_weights=None):
    if recon is None:
        recon_loss = mu.new_zeros(())
    elif channel_weights is not None:
        recon_loss = (channel_weights * (recon - target).pow(2)).mean()
    else:
        recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = lambda_recon * recon_loss + beta * kl + lambda_contrast * contrastive
    return total, recon_loss.detach(), kl.detach(), contrastive.detach()


def make_views(signal_raw, stats, clip_sigma, matrices, permutation, mode="ih",
               generator=None, count=2):
    """``count`` augmented views of the raw signal, re-normalised with ``stats``.

    ``mode`` is ``ih`` / ``i`` for the discrete group, ``so3`` for a continuous
    rotation, or ``both`` to draw one of each.
    """
    views = []
    for index in range(count):
        choice = mode
        if mode == "both":
            choice = "ih" if index % 2 == 0 else "so3"
        shells = (signal_raw.shape[1] - 1) // NEIGHBOUR_COUNT
        if choice == "so3" and shells == 1:
            augmented, _ = random_so3(signal_raw, generator=generator)
        elif shells == 1:
            augmented, _ = random_group(signal_raw, matrices, permutation,
                                        generator=generator, exclude_identity=True)
        else:
            element = torch.randint(1, len(matrices), (signal_raw.shape[0],),
                                    device=signal_raw.device, generator=generator)
            augmented = apply_group_shells(signal_raw, element, matrices, permutation, shells)
        views.append(apply_norm_stats_ico(augmented, stats, clip_sigma))
    return views
