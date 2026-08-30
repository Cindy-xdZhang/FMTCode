import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import Build_Task3_AnchoredFeature_Confirmation_6_1 as prior_spatial
import Build_Task3_FinalTuned_Confirmation_7_1 as spatial
import Confirm_Task3_FinalTuned_7_1 as confirm


CONFIG = Path("config/mainExp_Task3_3D_7.1.yaml")


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        result += (index % base) * factor
        index //= base
        factor /= base
    return result


def test_phase_is_pre_registered_and_distinct_from_all_exposed_populations():
    digest = hashlib.sha256(spatial.PHASE_KEY.encode("utf-8")).hexdigest()
    index = 1 + int(digest[:8], 16) % 1024
    phase = [_radical_inverse(index, base) - 0.5 for base in (2, 3, 5)]
    assert digest == spatial.PHASE_KEY_SHA256
    assert index == spatial.HALTON_INDEX == 187
    assert np.allclose(phase, spatial.SEED_GRID_PHASE, rtol=0.0, atol=1e-15)
    exposed = (
        [0.31, -0.23, 0.17],
        [-0.37, 0.29, -0.11],
        [0.318359375, 0.4561042524005485, -0.3352],
        [0.021484375, -0.34224965706447186, 0.0328],
    )
    assert all(not np.allclose(phase, old, rtol=0.0, atol=1e-15)
               for old in exposed)


def test_final_config_freezes_task3_comparison_and_ivd_p95():
    spec = confirm._load_spec(CONFIG)
    assert spec["status"] == "fresh_spatial_confirmation"
    assert spec["paired_seeds"] == [40, 41]
    assert spec["source_model"]["source_paired_seeds"] == [40, 41, 42]
    assert spec["source_model"]["expected_experiment"] == (
        "Verify_Task3_FinalPortfolio_49.1"
    )
    assert spec["expected_ivd_percentile"] == 95.0
    assert spec["target_dataset_macro_f1_gain"] == 0.15
    assert spec["aspirational_dataset_macro_f1_gain"] == 0.20
    assert spec["confirmation_count"] == 4
    assert len(spec["datasets"]) == 10


def test_validated_builder_scope_does_not_mutate_completed_6_1():
    original = {
        "EXPERIMENT": prior_spatial.EXPERIMENT,
        "PHASE_KEY": prior_spatial.PHASE_KEY,
        "SEED_GRID_PHASE": prior_spatial.SEED_GRID_PHASE,
        "SETTINGS": prior_spatial.SETTINGS,
    }
    with spatial._configured_base():
        assert prior_spatial.EXPERIMENT == spatial.EXPERIMENT
        assert prior_spatial.PHASE_KEY == spatial.PHASE_KEY
        assert prior_spatial.SEED_GRID_PHASE == spatial.SEED_GRID_PHASE
        assert prior_spatial.SETTINGS == spatial.SETTINGS
    for name, value in original.items():
        assert getattr(prior_spatial, name) == value


def _build_fake_49_1_source(tmp_path: Path) -> tuple[Path, Path]:
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_root = tmp_path / "source"
    experiment = "Verify_Task3_FinalPortfolio_49.1"
    source_config = source_root / "config/source.yaml"
    source_config.parent.mkdir(parents=True)
    overlay = {
        "experiment": experiment,
        "confirmation_opened": False,
    }
    source_config.write_text(
        yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
    )
    canonical_hash = confirm._canonical_text_sha256(source_config)

    families = {
        "channel": ["channel"],
        "halfcylinder": [
            "cylinder3d", "halfcylinderRe640", "halfcylinderRe6400",
        ],
        "tangaroa": ["tangaroa"],
        "deltaWing": ["deltaWing_resampled", "deltaWing_LBM"],
        "f22raptor": ["f22raptor"],
        "boeing747": ["boeing747"],
        "smokeBuoyancy": ["smokeBuoyancy"],
    }
    recipe = {"id": "u00", "fmt_feature": "aivd2w8_dft"}
    selection_path = source_root / "outputs/selection.json"
    selection_path.parent.mkdir(parents=True)
    models = []
    for family, datasets in families.items():
        for dataset in datasets:
            for seed in base["paired_seeds"]:
                for arm, variant in (
                    ("fmt", "raw_fmt_residual"),
                    ("raw_pca", "raw_pca_residual"),
                ):
                    result_path = (
                        source_root / "model_artifacts" / dataset
                        / f"seed{seed}" / arm / "per_run.csv"
                    )
                    checkpoint = (
                        result_path.parent / "checkpoints"
                        / f"{dataset}_{variant}_seed{seed}.pt"
                    )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(
                        f"{dataset},{seed},{arm}\n", encoding="utf-8"
                    )
                    checkpoint.write_bytes(f"{dataset}-{seed}-{arm}".encode())
                    models.append({
                        "dataset": dataset,
                        "physical_family": family,
                        "seed": seed,
                        "source": arm,
                        "variant": variant,
                        "candidate_id": "u00",
                        "fmt_feature": "aivd2w8_dft",
                        "fmt_dim": 3,
                        "parameter_count": 100,
                        "trainable_residual_parameter_count": 10,
                        "result": str(result_path),
                        "result_sha256": hashlib.sha256(
                            result_path.read_bytes()
                        ).hexdigest(),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                    })
    selection = {
        "experiment": experiment,
        "config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "confirmation_opened": False,
        "source_paired_seeds": [40, 41, 42],
        "frozen_confirmation_seeds": [40, 41],
        "development_dataset_macro_f1_gain_vs_raw_pca": 0.2,
        "family_datasets": families,
        "primary_by_group": {
            family: {
                "optimization_id": "u00",
                "optimization_recipe_json": json.dumps(recipe, sort_keys=True),
            }
            for family in families
        },
        "models": models,
    }
    selection_path.write_text(
        json.dumps(selection, sort_keys=True), encoding="utf-8"
    )

    base["source_model"] = {
        "repo_root": str(source_root),
        "environment": "TASK71_TEST_SOURCE_ROOT",
        "expected_experiment": experiment,
        "expected_config_canonical_sha256": canonical_hash,
        "paths": {
            "config": str(source_config.relative_to(source_root)),
            "selection": str(selection_path.relative_to(source_root)),
        },
        "source_paired_seeds": [40, 41, 42],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
    )
    return config_path, source_root


def test_source_selection_and_all_40_models_are_frozen_by_content(
    tmp_path, monkeypatch
):
    config_path, source_root = _build_fake_49_1_source(tmp_path)
    monkeypatch.setenv("TASK71_TEST_SOURCE_ROOT", str(source_root))
    spec = confirm._load_spec(config_path)
    root, paths, source, _, selection = confirm._source_state(spec)
    models = confirm._collect_models(spec, root, source, selection)
    assert len(models) == 40
    assert len({row["checkpoint_sha256"] for row in models}) == 40
    assert set(row["source"] for row in models) == {"fmt", "raw_pca"}
    assert spec["source_model"]["sha256"]["selection"] == hashlib.sha256(
        paths["selection"].read_bytes()
    ).hexdigest()


def test_confirmation_code_cannot_train_or_select_on_fifth_population():
    text = Path("Confirm_Task3_FinalTuned_7_1.py").read_text(encoding="utf-8")
    assert "_train_one" not in text
    assert "_select_f1_threshold" not in text
    assert "_load_residual" not in text
    assert "return _base.run_dataset" in text
    scripts = sorted(Path("ibex_bash").glob("mainexp_task3_3d_7.1_*.sh"))
    assert len(scripts) == 8
    for script in scripts:
        body = script.read_text(encoding="utf-8")
        assert "config/mainExp_Task3_3D_7.1.yaml" in body
        assert "Confirm_Task3_FinalTuned_7_1.py" in body or (
            script.name.endswith("static_preflight.sh")
            and "Prepare_Task3_FinalTuned_SourceManifest_7_1.py" in body
        )
