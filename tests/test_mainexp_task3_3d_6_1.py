import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import Build_Task3_AnchoredFeature_Confirmation_6_1 as spatial
import Confirm_Task3_AnchoredFeature_6_1 as confirm
import Prepare_Task3_AnchoredFeature_SourceManifest_6_1 as prepare


CONFIG = Path("config/mainExp_Task3_3D_6.1.yaml")


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        result += (index % base) * factor
        index //= base
        factor /= base
    return result


def test_phase_is_deterministic_and_distinct_from_exposed_populations():
    digest = hashlib.sha256(spatial.PHASE_KEY.encode("utf-8")).hexdigest()
    index = 1 + int(digest[:8], 16) % 1024
    phase = [_radical_inverse(index, base) - 0.5 for base in (2, 3, 5)]
    assert digest == spatial.PHASE_KEY_SHA256
    assert index == spatial.HALTON_INDEX == 417
    assert np.allclose(phase, spatial.SEED_GRID_PHASE, rtol=0.0, atol=1e-15)
    exposed = (
        [0.31, -0.23, 0.17],
        [-0.37, 0.29, -0.11],
        [0.318359375, 0.4561042524005485, -0.3352],
    )
    assert all(phase != previous for previous in exposed)


def test_final_config_freezes_task3_comparison_and_ivd_p95():
    spec = confirm._load_spec(CONFIG)
    assert spec["status"] == "fresh_spatial_confirmation"
    assert spec["paired_seeds"] == [40, 41]
    assert spec["expected_ivd_percentile"] == 95.0
    assert spec["target_dataset_macro_f1_gain"] == 0.15
    assert spec["confirmation_count"] == 4
    assert len(spec["datasets"]) == 10
    assert set(spec["confirmation_roots"]) == {"old8", "new2"}
    assert spec["source_model"]["expected_experiment"] == (
        "Verify_Task3_AnchoredFeatureDecomposition_22.1"
    )


def test_cache_builder_requires_recipe_before_data_access(tmp_path, monkeypatch):
    missing = tmp_path / "missing_recipe.json"
    monkeypatch.setenv(spatial.RECIPE_MANIFEST_ENV, str(missing))
    try:
        spatial._require_recipe_frozen()
    except FileNotFoundError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("cache builder opened without a frozen recipe")


def test_source_manifest_derivation_reuses_only_frozen_temporal_sources(
    tmp_path, monkeypatch
):
    entries = {}
    for dataset in sorted(spatial._expected_datasets()):
        source = tmp_path / f"{dataset}.nc"
        source.write_bytes(dataset.encode("utf-8"))
        indices = next(
            settings["indices"][dataset]
            for settings in spatial.SETTINGS.values()
            if dataset in settings["indices"]
        )
        entries[dataset] = {
            "kind": "test_temporal_source",
            "path": str(source),
            "original_fixed_indices": indices,
            "effective_fixed_indices": indices,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "all_windows_verified_exact": True,
        }
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps({
            "datasets": entries,
            "scientific_protocol_unchanged": True,
            "equivalence": "test exact source",
        }, sort_keys=True),
        encoding="utf-8",
    )
    derived = tmp_path / "derived.json"
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["source_staging"] = {
        "parent_manifest": str(parent),
        "parent_manifest_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "derived_manifest": str(derived),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = prepare.derive(config_path)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["temporal_sources_are_phase_independent"] is True
    assert payload["seed_grid_phase"] == spatial.SEED_GRID_PHASE
    assert set(payload["datasets"]) == spatial._expected_datasets()
    assert "local_packs" not in payload
    monkeypatch.setenv(spatial.SOURCE_STAGING_ENV, str(result))
    assert spatial.source_staging_identity()["parent_manifest_sha256"] == (
        hashlib.sha256(parent.read_bytes()).hexdigest()
    )
