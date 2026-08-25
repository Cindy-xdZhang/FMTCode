from pathlib import Path

from Sweep_Task2_VAE_3D import _load_spec, _variant


CONFIG = Path("config/Verify_Task2_VAEGrid_3D_3.1.yaml")


def test_grid_has_complete_development_only_scope():
    spec = _load_spec(CONFIG)
    datasets = [dataset for group in spec["groups"].values() for dataset in group["datasets"]]
    assert len(datasets) == 10
    assert len(set(datasets)) == 10
    assert len(spec["variants"]) == 36
    assert "confirmation_cache" not in spec
    assert all("confirmation_cache" not in group for group in spec["groups"].values())
    assert set(spec["task1_development_f1"]) == set(datasets)


def test_raw_and_fmt_share_one_variant_object():
    spec = _load_spec(CONFIG)
    for index in range(len(spec["variants"])):
        variant = _variant(spec, index)
        assert "hidden_dims" in variant
        assert "latent_dim" in variant
        assert "beta" in variant
        assert "learning_rate" in variant
        assert variant["optimizer_steps"] > 0
