import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.VAE_3D import FeatureVAE3D, relational_distance_loss, vae_loss
from DeepUtils.utils import EasyConfig
from Verify_HighReVAE import _train


def test_vae_shapes_and_finite_loss():
    model = FeatureVAE3D(20, hidden_dims=(12, 8), latent_dim=3)
    x = torch.randn(7, 20)
    reconstruction, mu, logvar = model(x)
    assert reconstruction.shape == x.shape
    assert mu.shape == logvar.shape == (7, 3)
    loss, rec, kl = vae_loss(reconstruction, x, mu, logvar, beta=1e-3)
    assert all(torch.isfinite(value) for value in (loss, rec, kl))
    loss.backward()


def test_weighted_vae_loss_respects_feature_weights():
    target = torch.zeros(2, 3)
    reconstruction = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    mu = torch.zeros(2, 2)
    logvar = torch.zeros(2, 2)
    weights = torch.tensor([1.0, 0.0, 0.0])
    _, rec, _ = vae_loss(
        reconstruction, target, mu, logvar, beta=0.0,
        reconstruction_weight=weights,
    )
    assert torch.isclose(rec, torch.tensor(1.0 / 3.0))


def test_weighted_vae_loss_rejects_wrong_dimension():
    x = torch.zeros(2, 3)
    mu = torch.zeros(2, 2)
    logvar = torch.zeros(2, 2)
    try:
        vae_loss(x, x, mu, logvar, beta=0.0, reconstruction_weight=torch.ones(2))
    except ValueError as error:
        assert "feature dimension" in str(error)
    else:
        raise AssertionError("wrong reconstruction-weight dimension was accepted")


def test_relational_distance_loss_is_small_for_scaled_isometry():
    torch.manual_seed(7068)
    source = torch.randn(32, 5)
    latent = 3.5 * source
    loss = relational_distance_loss(source, latent, pair_count=4096)
    assert loss < 1e-6
    assert torch.isfinite(relational_distance_loss(source, torch.randn(32, 3), 256))


def test_pca_initialized_training_honors_exact_step_budget():
    rng = np.random.default_rng(7068)
    train = rng.normal(size=(32, 6)).astype(np.float32)
    evaluate = rng.normal(size=(8, 6)).astype(np.float32)
    source = EasyConfig()
    source.update({"task2": {"learning_rate": 1e-3, "weight_decay": 0.0,
                              "batch_size": 8, "target_optimizer_steps": 99}})
    settings = {"hidden_dims": [], "latent_dim": 2, "beta": 1e-5,
                "pca_init": True, "pca_anchor_weight": 0.1,
                "logvar_init": -6.0, "optimizer_steps": 3,
                "relational_weight": 0.0, "pair_count": 16}
    train_mu, eval_mu, metrics = _train(
        train, evaluate, settings, source, 7068, torch.device("cpu")
    )
    assert train_mu.shape == (32, 2) and eval_mu.shape == (8, 2)
    assert metrics["completed_optimizer_steps"] == 3
    assert metrics["pca_init"] is True
    assert np.isfinite(metrics["eval_pca_anchor"])


if __name__ == "__main__":
    test_vae_shapes_and_finite_loss()
    test_weighted_vae_loss_respects_feature_weights()
    test_weighted_vae_loss_rejects_wrong_dimension()
    test_relational_distance_loss_is_small_for_scaled_isometry()
    test_pca_initialized_training_honors_exact_step_budget()
    print("3D VAE TEST PASSED")
