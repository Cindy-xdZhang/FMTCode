"""Small multilayer-perceptron variational autoencoder for Task2 3D features."""

from __future__ import annotations

import torch
from torch import nn


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


def vae_loss(reconstruction, target, mu, logvar, beta):
    reconstruction_loss = torch.mean((reconstruction - target) ** 2)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    return reconstruction_loss + float(beta) * kl, reconstruction_loss, kl
