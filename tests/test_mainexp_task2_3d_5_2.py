import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

import Build_Task2_LatentConfirmation_5_2 as spatial
import Confirm_Task2_LatentBottleneck_5_2 as confirm


CONFIG = Path("config/mainExp_Task2_3D_5.2.yaml")
SEARCH = Path("config/Verify_Task2_LatentBottleneck_5.1.yaml")
LOCAL_SELECTION = Path(
    "output/Verify_Task2_LatentBottleneck_5.1_ibex/selection.json"
)


def _available_selection() -> Path:
    root = os.environ.get("TASK2_LATENT51_ROOT")
    if root:
        remote = (
            Path(root) / "outputs/Verify_Task2_LatentBottleneck_5.1/selection.json"
        )
        if remote.exists():
            return remote
    return LOCAL_SELECTION


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        result += (index % base) * factor
        index //= base
        factor /= base
    return result


def test_phase_is_deterministic_and_distinct_from_prior_populations():
    digest = hashlib.sha256(spatial.PHASE_KEY.encode("utf-8")).hexdigest()
    index = 1 + int(digest[:8], 16) % 1024
    phase = [_radical_inverse(index, base) - 0.5 for base in (2, 3, 5)]
    assert digest == spatial.PHASE_KEY_SHA256
    assert index == spatial.HALTON_INDEX == 544
    assert np.allclose(phase, spatial.SEED_GRID_PHASE, rtol=0.0, atol=1e-15)
    exposed = (
        [0.31, -0.23, 0.17],
        [-0.37, 0.29, -0.11],
        [0.318359375, 0.4561042524005485, -0.3352],
        [0.021484375, -0.34224965706447186, 0.0328],
    )
    assert all(not np.allclose(phase, value, rtol=0.0, atol=1e-15)
               for value in exposed)


def test_config_freezes_task2_same_vae_confirmation_contract():
    spec = confirm._load_spec(CONFIG)
    assert spec["status"] == "fresh_spatial_confirmation"
    assert spec["recipes"] == ["selected", "control"]
    assert spec["final_training_seeds"] == [9090, 9091, 9092, 9093, 9094]
    assert spec["splits"]["final_train"] == list(range(8))
    assert spec["splits"]["cluster_calibration"] == [8, 9]
    assert spec["confirmation_count"] == 4
    assert spec["target_dataset_macro_f1_gain"] == 0.15
    assert len(spec["datasets"]) == 10
    assert set(spec["confirmation_roots"]) == {"old8", "new2"}


def test_selected_and_control_change_only_latent_dimension_when_selection_exists():
    selection_path = _available_selection()
    if not selection_path.exists():
        return
    search = yaml.safe_load(SEARCH.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = {
        "boeing747": 6,
        "channel": 8,
        "deltaWing": 12,
        "f22raptor": 1,
        "halfcylinder": 64,
        "smokeBuoyancy": 24,
        "tangaroa": 1,
    }
    for group_name, selected_dim in expected.items():
        selected = confirm._recipe_settings(
            search, selection, group_name, "selected"
        )
        control = confirm._recipe_settings(
            search, selection, group_name, "control"
        )
        assert selected["latent_dim"] == selected_dim
        invariant_keys = set(selected) - {"id", "latent_dim"}
        assert {key: selected[key] for key in invariant_keys} == {
            key: control[key] for key in invariant_keys
        }
        assert control["latent_dim"] == int(
            search["groups"][group_name]["base_vae"]["control_latent_dim"]
        )


def test_cache_builder_refuses_data_access_before_recipe(tmp_path, monkeypatch):
    missing = tmp_path / "missing_recipe.json"
    monkeypatch.setenv(spatial.RECIPE_MANIFEST_ENV, str(missing))
    try:
        spatial._require_recipe_frozen()
    except FileNotFoundError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("Task2 5.2 cache opened before recipe freeze")


def test_array_mapping_covers_each_dataset_once():
    jobs = spatial.jobs()
    assert len(jobs) == 10
    assert len(set(jobs)) == 10
    assert {dataset for _, dataset in jobs} == spatial._expected_datasets()


def test_source_hashes_match_completed_development_artifacts_when_present():
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    normalized = SEARCH.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized).hexdigest() == (
        spec["source_search"]["search_config_sha256"]
    )
    selection_path = _available_selection()
    if selection_path.exists():
        assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == (
            spec["source_search"]["selection_sha256"]
        )


def run_dependency_free_contracts():
    test_phase_is_deterministic_and_distinct_from_prior_populations()
    test_config_freezes_task2_same_vae_confirmation_contract()
    test_selected_and_control_change_only_latent_dimension_when_selection_exists()
    test_array_mapping_covers_each_dataset_once()
    test_source_hashes_match_completed_development_artifacts_when_present()
    previous = os.environ.get(spatial.RECIPE_MANIFEST_ENV)
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "missing_recipe.json"
        os.environ[spatial.RECIPE_MANIFEST_ENV] = str(missing)
        try:
            spatial._require_recipe_frozen()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Task2 5.2 cache opened before recipe freeze")
    if previous is None:
        os.environ.pop(spatial.RECIPE_MANIFEST_ENV, None)
    else:
        os.environ[spatial.RECIPE_MANIFEST_ENV] = previous
    print("Task2 5.2 dependency-free contracts: 6 passed")


if __name__ == "__main__":
    run_dependency_free_contracts()
