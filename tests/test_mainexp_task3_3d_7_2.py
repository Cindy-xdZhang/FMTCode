import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import Build_Task3_FinalTuned_Confirmation_7_1 as prior_spatial
import Build_Task3_AdaptiveTuned_Confirmation_7_2 as spatial
import Confirm_Task3_AdaptiveTuned_7_2 as confirm
import Prepare_Task3_FinalTuned_SourceManifest_7_1 as prior_prepare
import Prepare_Task3_AdaptiveTuned_SourceManifest_7_2 as prepare


CONFIG = Path("config/mainExp_Task3_3D_7.2.yaml")


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        result += (index % base) * factor
        index //= base
        factor /= base
    return result


class MainExpTask3_7_2Tests(unittest.TestCase):
    def test_phase_is_preregistered_and_new(self):
        digest = hashlib.sha256(spatial.PHASE_KEY.encode("utf-8")).hexdigest()
        index = 1 + int(digest[:8], 16) % 1024
        phase = [
            _radical_inverse(index, base) - 0.5 for base in (2, 3, 5)
        ]
        self.assertEqual(digest, spatial.PHASE_KEY_SHA256)
        self.assertEqual(index, spatial.HALTON_INDEX)
        self.assertEqual(index, 678)
        self.assertTrue(np.allclose(
            phase, spatial.SEED_GRID_PHASE, rtol=0.0, atol=1e-15
        ))
        exposed = (
            [0.31, -0.23, 0.17],
            [-0.37, 0.29, -0.11],
            [0.318359375, 0.4561042524005485, -0.3352],
            [0.021484375, -0.34224965706447186, 0.0328],
            prior_spatial.SEED_GRID_PHASE,
        )
        self.assertTrue(all(
            not np.allclose(phase, old, rtol=0.0, atol=1e-15)
            for old in exposed
        ))

    def test_config_freezes_adaptive_portfolio_and_ivd_p95(self):
        spec = confirm._load_spec(CONFIG)
        self.assertEqual(spec["status"], "fresh_spatial_confirmation")
        self.assertEqual(spec["paired_seeds"], [40, 41])
        self.assertEqual(
            spec["source_model"]["source_paired_seeds"], [40, 41, 42]
        )
        self.assertEqual(
            spec["source_model"]["expected_experiment"],
            confirm.SOURCE_EXPERIMENT,
        )
        self.assertEqual(spec["expected_ivd_percentile"], 95.0)
        self.assertEqual(spec["target_dataset_macro_f1_gain"], 0.15)
        self.assertEqual(spec["aspirational_dataset_macro_f1_gain"], 0.20)
        self.assertEqual(spec["confirmation_count"], 4)
        self.assertEqual(len(spec["datasets"]), 10)
        source_config = Path(
            "config/Verify_Task3_AdaptivePortfolio_52.1.yaml"
        )
        canonical = source_config.read_text(encoding="utf-8").replace(
            "\r\n", "\n"
        ).replace("\r", "\n")
        self.assertEqual(
            spec["source_model"]["expected_config_canonical_sha256"],
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def test_builder_scope_restores_completed_7_1(self):
        original = {
            "EXPERIMENT": prior_spatial.EXPERIMENT,
            "PHASE_KEY": prior_spatial.PHASE_KEY,
            "SEED_GRID_PHASE": prior_spatial.SEED_GRID_PHASE,
            "SETTINGS": prior_spatial.SETTINGS,
        }
        with spatial._configured_base():
            self.assertEqual(prior_spatial.EXPERIMENT, spatial.EXPERIMENT)
            self.assertEqual(prior_spatial.PHASE_KEY, spatial.PHASE_KEY)
            self.assertEqual(
                prior_spatial.SEED_GRID_PHASE, spatial.SEED_GRID_PHASE
            )
            self.assertEqual(prior_spatial.SETTINGS, spatial.SETTINGS)
        for name, value in original.items():
            self.assertEqual(getattr(prior_spatial, name), value)

        original_prepare_spatial = prior_prepare.spatial
        with prepare._configured_base():
            self.assertIs(prior_prepare.spatial, spatial)
        self.assertIs(prior_prepare.spatial, original_prepare_spatial)

        original_confirm_spatial = confirm._base.spatial
        with confirm._configured_base():
            self.assertIs(confirm._base.spatial, spatial)
        self.assertIs(confirm._base.spatial, original_confirm_spatial)

    def _build_fake_source(self, root: Path) -> tuple[Path, Path]:
        base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        source_root = root / "source"
        source_config = source_root / "config/source.yaml"
        source_config.parent.mkdir(parents=True)
        source_config.write_text(
            yaml.safe_dump({
                "experiment": confirm.SOURCE_EXPERIMENT,
                "confirmation_opened": False,
            }, sort_keys=False),
            encoding="utf-8",
        )
        canonical_hash = confirm._base._canonical_text_sha256(source_config)
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
        models = []
        for family, datasets in families.items():
            for dataset in datasets:
                for seed in base["paired_seeds"]:
                    for arm, variant in (
                        ("fmt", "raw_fmt_residual"),
                        ("raw_pca", "raw_pca_residual"),
                    ):
                        artifact_root = (
                            source_root / "outputs/frozen_artifacts" / dataset
                            / f"seed{seed}" / arm
                        )
                        result = artifact_root / "per_run.csv"
                        checkpoint = artifact_root / "model.pt"
                        artifact_root.mkdir(parents=True, exist_ok=True)
                        result.write_text(
                            f"{dataset},{seed},{arm}\n", encoding="utf-8"
                        )
                        checkpoint.write_bytes(
                            f"{dataset}-{seed}-{arm}".encode()
                        )
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
                            "result": str(result),
                            "result_sha256": hashlib.sha256(
                                result.read_bytes()
                            ).hexdigest(),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                        })
        selection_path = source_root / "outputs/selection.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection = {
            "experiment": confirm.SOURCE_EXPERIMENT,
            "config_sha256": hashlib.sha256(
                source_config.read_bytes()
            ).hexdigest(),
            "confirmation_opened": False,
            "source_paired_seeds": [40, 41, 42],
            "frozen_confirmation_seeds": [40, 41],
            "development_dataset_macro_f1_gain_vs_raw_pca": 0.2,
            "family_datasets": families,
            "primary_by_group": {
                family: {
                    "optimization_id": "u00",
                    "optimization_recipe_json": json.dumps(
                        recipe, sort_keys=True
                    ),
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
            "environment": "TASK72_TEST_SOURCE_ROOT",
            "expected_experiment": confirm.SOURCE_EXPERIMENT,
            "expected_config_canonical_sha256": canonical_hash,
            "paths": {
                "config": str(source_config.relative_to(source_root)),
                "selection": str(selection_path.relative_to(source_root)),
            },
            "source_paired_seeds": [40, 41, 42],
        }
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
        )
        return config_path, source_root

    def test_all_40_copied_models_are_frozen_by_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path, source_root = self._build_fake_source(
                Path(temporary)
            )
            with patch.dict(os.environ, {
                "TASK72_TEST_SOURCE_ROOT": str(source_root)
            }):
                spec = confirm._load_spec(config_path)
                root, paths, source, _, selection = confirm._source_state(spec)
                models = confirm._collect_models(
                    spec, root, source, selection
                )
            self.assertEqual(len(models), 40)
            self.assertEqual(
                len({row["checkpoint_sha256"] for row in models}), 40
            )
            self.assertEqual(
                {row["source"] for row in models}, {"fmt", "raw_pca"}
            )
            self.assertEqual(
                spec["source_model"]["sha256"]["selection"],
                hashlib.sha256(paths["selection"].read_bytes()).hexdigest(),
            )
            self.assertTrue(all(
                "frozen_artifacts" in Path(row["checkpoint"]).parts
                for row in models
            ))

    def test_static_preflight_is_sealed_before_any_confirmation_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, source_root = self._build_fake_source(root)
            spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            spec["output_root"] = str(root / "confirmation")
            spec["recipe_manifest"] = str(
                root / "confirmation/frozen_recipe_manifest.json"
            )
            staging_path = root / "source_staging_manifest.json"
            spec["source_staging"]["derived_manifest"] = str(staging_path)

            temporary_settings = {}
            for group, original in spatial.SETTINGS.items():
                label_config = root / f"labels_{group}.yaml"
                label_output = root / f"labels_{group}"
                label_config.write_text(
                    yaml.safe_dump({"output_dir": str(label_output)}),
                    encoding="utf-8",
                )
                temporary_settings[group] = {
                    **original,
                    "cache_dir": str(root / f"cache_{group}"),
                    "label_config": str(label_config),
                }
                spec["confirmation_roots"][group]["source_root"] = str(
                    root / f"cache_{group}"
                )
                spec["confirmation_roots"][group]["label_root"] = str(
                    label_output / "labels"
                )
            config_path.write_text(
                yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
            )
            staging_path.write_text(
                json.dumps({
                    "experiment": f"{spatial.EXPERIMENT}_source_staging",
                    "parent_manifest_sha256": "1" * 64,
                    "scientific_protocol_unchanged": True,
                    "temporal_sources_are_phase_independent": True,
                    "seed_grid_phase": list(spatial.SEED_GRID_PHASE),
                    "datasets": {
                        dataset: {} for dataset in spatial._expected_datasets()
                    },
                }),
                encoding="utf-8",
            )
            environment = {
                "TASK72_TEST_SOURCE_ROOT": str(source_root),
                spatial.SOURCE_STAGING_ENV: str(staging_path),
            }
            with (
                patch.dict(os.environ, environment),
                patch.object(spatial, "SETTINGS", temporary_settings),
            ):
                target = confirm.static_preflight(config_path)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment"], spatial.EXPERIMENT)
            self.assertEqual(payload["frozen_model_count"], 40)
            self.assertEqual(payload["expected_evaluations"], 40)
            self.assertFalse(payload["confirmation_opened"])
            self.assertTrue(all(
                count == 0
                for group in payload["confirmation_artifact_counts"].values()
                for count in group.values()
            ))

    def test_confirmation_has_no_training_or_selection_path(self):
        text = Path("Confirm_Task3_AdaptiveTuned_7_2.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_train_one", text)
        self.assertNotIn("_select_f1_threshold", text)
        self.assertNotIn("_load_residual", text)
        self.assertIn("return _base.run_dataset", text)
        self.assertIn(confirm.SOURCE_EXPERIMENT, text)
        scripts = sorted(
            Path("ibex_bash").glob("mainexp_task3_3d_7.2_*.sh")
        )
        self.assertEqual(len(scripts), 8)
        for script in scripts:
            body = script.read_text(encoding="utf-8")
            self.assertIn("config/mainExp_Task3_3D_7.2.yaml", body)
            self.assertTrue(
                "Confirm_Task3_AdaptiveTuned_7_2.py" in body
                or (
                    script.name.endswith("static_preflight.sh")
                    and "Prepare_Task3_AdaptiveTuned_SourceManifest_7_2.py"
                    in body
                )
            )


if __name__ == "__main__":
    unittest.main()
