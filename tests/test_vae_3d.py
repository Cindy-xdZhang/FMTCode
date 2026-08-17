import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.VAE_3D import FeatureVAE3D, vae_loss


def test_vae_shapes_and_finite_loss():
    model = FeatureVAE3D(20, hidden_dims=(12, 8), latent_dim=3)
    x = torch.randn(7, 20)
    reconstruction, mu, logvar = model(x)
    assert reconstruction.shape == x.shape
    assert mu.shape == logvar.shape == (7, 3)
    loss, rec, kl = vae_loss(reconstruction, x, mu, logvar, beta=1e-3)
    assert all(torch.isfinite(value) for value in (loss, rec, kl))
    loss.backward()


if __name__ == "__main__":
    test_vae_shapes_and_finite_loss()
    print("3D VAE TEST PASSED")
