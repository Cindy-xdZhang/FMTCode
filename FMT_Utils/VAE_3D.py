"""Small multilayer-perceptron variational autoencoder for Task2 3D features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeatureVAE3D(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), latent_dim=16):
        super().__init__()
        dims = [int(input_dim), *(int(v) for v in hidden_dims)]
        encoder = []
        for left, right in zip(dims[:-1], dims[1:]):
            encoder.extend((nn.Linear(left, right), nn.GELU()))
        self.encoder = nn.Sequential(*encoder)
        self.mu = nn.Linear(dims[-1], int(latent_dim))
        self.logvar = nn.Linear(dims[-1], int(latent_dim))
        decoder_dims = [int(latent_dim), *reversed(dims[1:]), int(input_dim)]
        decoder = []
        for index, (left, right) in enumerate(zip(decoder_dims[:-1], decoder_dims[1:])):
            decoder.append(nn.Linear(left, right))
            if index < len(decoder_dims) - 2:
                decoder.append(nn.GELU())
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden).clamp(-20.0, 10.0)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta, reconstruction_weight=None):
    error2 = (reconstruction - target) ** 2
    if reconstruction_weight is None:
        reconstruction_loss = torch.mean(error2)
    else:
        weight = torch.as_tensor(
            reconstruction_weight, device=error2.device, dtype=error2.dtype
        )
        if weight.ndim != 1 or weight.shape[0] != error2.shape[-1]:
            raise ValueError(
                "reconstruction_weight must be a 1-D tensor matching the feature dimension"
            )
        reconstruction_loss = torch.mean(error2 * weight)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    return reconstruction_loss + float(beta) * kl, reconstruction_loss, kl


def relational_distance_loss(source, latent, pair_count=2048):
    """Preserve normalized sample-pair distances without labels.

    Random non-self pairs keep the cost linear in ``pair_count`` rather than
    quadratic in batch size. Distances are normalized by their detached batch
    mean, so the loss constrains geometry but not an arbitrary global scale.
    """
    if source.ndim != 2 or latent.ndim != 2 or len(source) != len(latent):
        raise ValueError("source and latent must be 2-D tensors with equal batch size")
    batch = len(source)
    if batch < 2:
        return latent.sum() * 0.0
    count = int(pair_count)
    if count < 1:
        raise ValueError("pair_count must be positive")
    left = torch.randint(batch, (count,), device=source.device)
    shift = torch.randint(1, batch, (count,), device=source.device)
    right = (left + shift) % batch
    source_distance = (source[left] - source[right]).square().mean(dim=1)
    latent_distance = (latent[left] - latent[right]).square().mean(dim=1)
    source_normalized = source_distance / source_distance.mean().detach().clamp_min(1e-8)
    latent_normalized = latent_distance / latent_distance.mean().detach().clamp_min(1e-8)
    return F.smooth_l1_loss(latent_normalized, source_normalized)
