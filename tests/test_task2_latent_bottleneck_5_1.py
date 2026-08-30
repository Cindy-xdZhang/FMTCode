"""Contracts for the Task2 latent-dimension-only search 5.1."""

from pathlib import Path
import hashlib
import yaml

from Search_Task2_LatentBottleneck_5_1 import (
    _candidate,
    _decode_job,
    _load_spec,
    _selection_key,
)


CONFIG = "config/Verify_Task2_LatentBottleneck_5.1.yaml"


def test_grid_contains_every_frozen_control_and_ultranarrow_latent():
    spec = _load_spec(CONFIG)
    assert spec["latent_dims"][0] == 1
    assert len(spec["latent_dims"]) == 12
    for group in spec["groups"].values():
        assert int(group["base_vae"]["control_latent_dim"]) in spec["latent_dims"]


def test_array_mapping_covers_exact_dataset_latent_product():
    spec = _load_spec(CONFIG)
    count = len(spec["_datasets"]) * len(spec["latent_dims"])
    observed = {_decode_job(spec, index) for index in range(count)}
    expected = {
        (dataset, latent_index)
        for dataset in spec["_datasets"]
        for latent_index in range(len(spec["latent_dims"]))
    }
    assert count == 120
    assert observed == expected


def test_candidate_changes_only_latent_dimension():
    spec = _load_spec(CONFIG)
    for group in spec["groups"].values():
        candidates = [_candidate(spec, group, index) for index in range(12)]
        invariant = [{key: value for key, value in row.items()
                      if key not in {"id", "latent_dim"}} for row in candidates]
        assert all(row == invariant[0] for row in invariant)
        assert [row["latent_dim"] for row in candidates] == spec["latent_dims"]


def test_source_selection_is_exactly_frozen_when_present():
    spec = _load_spec(CONFIG)
    path = Path(spec["source"]["stage2_selection"])
    if path.exists():
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == spec["source"]["stage2_selection_sha256"]


def test_frozen_group_recipes_equal_source_stage2_architectures():
    spec = _load_spec(CONFIG)
    source = yaml.safe_load(
        Path(spec["source"]["family_search_config"]).read_text(encoding="utf-8")
    )
    lookup = {row["id"]: dict(row) for row in source["stage2_architectures"]}
    for group in spec["groups"].values():
        expected = dict(lookup[group["source_architecture_id"]])
        expected.pop("id")
        observed = dict(group["base_vae"])
        observed["latent_dim"] = observed.pop("control_latent_dim")
        assert observed == expected


def test_selection_key_prioritizes_gain_then_robustness_then_fmt_f1():
    base = {
        "fmt_minus_raw_f1_macro": 0.20,
        "worst_seed_f1_gain": 0.10,
        "worst_dataset_f1_gain": 0.05,
        "fmt_f1_macro": 0.70,
    }
    assert _selection_key({**base, "fmt_minus_raw_f1_macro": 0.21}) > _selection_key(base)
    assert _selection_key({**base, "worst_seed_f1_gain": 0.11}) > _selection_key(base)
    assert _selection_key({**base, "fmt_f1_macro": 0.71}) > _selection_key(base)


def test_development_roots_and_forbidden_ordinals_are_separate():
    spec = _load_spec(CONFIG)
    opened = set(spec["splits"]["selection_train"]) | set(
        spec["splits"]["selection_validation"]
    )
    forbidden = set(spec["splits"]["forbidden"])
    assert opened == set(range(8))
    assert forbidden == {8, 9}
    assert opened.isdisjoint(forbidden)
    assert spec["confirmation_opened"] is False
