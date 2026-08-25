from pathlib import Path

from Sweep_Task2_VAE_3D import (
    _load_spec,
    _select_global_worst_seed,
    _variant,
)


CONFIG = Path("config/Verify_Task2_VAEGrid_3D_3.1.yaml")
ROBUST_CONFIG = Path("config/Verify_Task2_VAEGrid_3D_3.2.yaml")


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


def test_robust_grid_inherits_variants_without_confirmation_access():
    spec = _load_spec(ROBUST_CONFIG)
    assert len(spec["variants"]) == 36
    assert spec["selection_seeds"] == list(range(7068, 7075))
    assert spec["selection_rule"] == "worst_seed_hierarchy"
    assert "confirmation_cache" not in spec
    assert all("confirmation_cache" not in group for group in spec["groups"].values())


def test_robust_selection_uses_worst_seed_not_only_mean():
    spec = {
        "groups": {
            "g1": {"datasets": ["a"]},
            "g2": {"datasets": ["b"]},
        },
        "variants": [{"id": "unstable"}, {"id": "stable"}],
        "selection_seeds": [1, 2],
        "task1_development_f1": {"a": 0.5, "b": 0.5},
    }
    rows = []
    values = {
        ("g1", "unstable", 1): (0.9, 1.0),
        ("g1", "unstable", 2): (0.2, 0.9),
        ("g1", "stable", 1): (0.65, 0.72),
        ("g1", "stable", 2): (0.65, 0.72),
        ("g2", "unstable", 1): (0.65, 0.72),
        ("g2", "unstable", 2): (0.65, 0.72),
        ("g2", "stable", 1): (0.65, 0.72),
        ("g2", "stable", 2): (0.65, 0.72),
    }
    for (group, variant, seed), (raw, fmt) in values.items():
        dataset = spec["groups"][group]["datasets"][0]
        for method, f1 in (("raw", raw), ("fmt", fmt)):
            rows.append({
                "group": group, "variant": variant, "training_seed": seed,
                "dataset": dataset, "method": method, "f1": f1,
            })
    candidates = [
        {"group": group, "variant": variant}
        for group in spec["groups"] for variant in ("unstable", "stable")
    ]
    selected = _select_global_worst_seed(spec, candidates, rows)
    assert selected["selected"]["g1"]["variant"] == "stable"
    assert selected["hierarchy_satisfied"] is True
