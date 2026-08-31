import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import yaml

import Audit_Task3_AdaptivePortfolio as audit_module
import Select_Task3_AdaptivePortfolio_52_1 as portfolio
from tests.test_task3_final_portfolio_49_1 import _fake_source


CONFIG = Path("config/Verify_Task3_AdaptivePortfolio_52.1.yaml")


class AdaptivePortfolioTests(unittest.TestCase):
    def test_static_contract_is_training_free_and_uses_five_sources(self):
        spec = portfolio._load_spec(CONFIG)
        report = portfolio.static_preflight(CONFIG)
        self.assertEqual(report["source_count"], 5)
        self.assertEqual(report["training_runs"], 0)
        self.assertFalse(report["confirmation_opened"])
        self.assertEqual(set(spec["sources"]), portfolio.EXPECTED_SOURCES)
        self.assertEqual(spec["frozen_confirmation_seeds"], [40, 41])

    def test_sources_are_exact_declared_search_versions(self):
        spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        observed = {
            name: row["expected_experiment"]
            for name, row in spec["sources"].items()
        }
        self.assertEqual(observed, {
            "safe_factor": "Verify_Task3_SafeFactorCombination_44.1",
            "head_alpha_clip": "Verify_Task3_HeadAlphaClipCombination_45.1",
            "full_stack": "Verify_Task3_HeadFullStackCombination_48.1",
            "focal_gamma_low": "Verify_Task3_FocalGammaLow_50.1",
            "dropout_high": "Verify_Task3_ResidualDropoutHigh_51.1",
        })
        observed_hashes = {
            name: row["expected_config_canonical_sha256"]
            for name, row in spec["sources"].items()
        }
        self.assertEqual(observed_hashes, {
            "safe_factor": (
                "ab8032e5536cac5fe0f23456561ee1279c07977ea5694f4f03c1572a018a1713"
            ),
            "head_alpha_clip": (
                "8b27378355505edbc1a2cadbe3388645d9b4f660ca7ce12d8c5bd3c8cc1cbbb2"
            ),
            "full_stack": (
                "a7508abdae343af1397166e28e22e82942231f0685fa0aed1b25bf2dc82f00c7"
            ),
            "focal_gamma_low": (
                "a0b706e867dd923ec7ee5b0f56f8c84e1d88708b8565646371626ae634eb5a58"
            ),
            "dropout_high": (
                "4f67ee6fbbe9baf779b990eab6515ec18e472f6a3026aa4dad0a5a0aad0713ae"
            ),
        })

    def test_base_loader_is_restored_after_scoped_use(self):
        original = portfolio._base._load_spec
        with portfolio._configured_base():
            self.assertIs(portfolio._base._load_spec, portfolio._load_spec)
        self.assertIs(portfolio._base._load_spec, original)

    def test_source_identity_preflight_checks_deployed_config_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            spec["output_root"] = str(tmp_path / "output")
            source_paths = []
            for name, section in spec["sources"].items():
                root = tmp_path / name
                config = root / section["paths"]["config"]
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text(
                    yaml.safe_dump({
                        "experiment": section["expected_experiment"],
                    }),
                    encoding="utf-8",
                )
                section["repo_root"] = str(root)
                section["expected_config_canonical_sha256"] = (
                    portfolio._canonical_text_sha256(config)
                )
                source_paths.append(config)
            config_path = tmp_path / "portfolio.yaml"
            config_path.write_text(
                yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
            )
            with redirect_stdout(io.StringIO()):
                target = portfolio.source_identity_preflight(config_path)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_count"], 5)
            self.assertFalse(payload["performance_artifacts_read"])

            source_paths[0].write_text(
                source_paths[0].read_text(encoding="utf-8") + "changed: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "canonical SHA-256"):
                portfolio.source_identity_preflight(config_path)

    def test_freeze_copies_and_repoints_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            output = tmp_path / "output"
            source = tmp_path / "source"
            models = []
            for dataset_index in range(10):
                dataset = f"dataset{dataset_index}"
                for seed in (40, 41):
                    for arm in ("fmt", "raw_pca"):
                        root = source / dataset / f"seed{seed}" / arm
                        result = root / "per_run.csv"
                        checkpoint = root / "checkpoints" / "model.pt"
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        result.write_text(
                            f"{dataset},{seed},{arm}\n", encoding="utf-8"
                        )
                        checkpoint.write_bytes(
                            f"{dataset}-{seed}-{arm}".encode()
                        )
                        models.append({
                            "dataset": dataset,
                            "seed": seed,
                            "source": arm,
                            "result": str(result),
                            "result_sha256": hashlib.sha256(
                                result.read_bytes()
                            ).hexdigest(),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                        })
            selection = output / "portfolio_selection.json"
            selection.parent.mkdir(parents=True)
            selection.write_text(
                json.dumps({
                    "experiment": portfolio.EXPERIMENT,
                    "models": models,
                }),
                encoding="utf-8",
            )
            portfolio._freeze_local_copies(selection)
            frozen = json.loads(selection.read_text(encoding="utf-8"))
            self.assertEqual(frozen["frozen_model_count"], 40)
            self.assertEqual(frozen["frozen_artifact_file_count"], 80)
            self.assertEqual(len(frozen["models"]), 40)
            for model in frozen["models"]:
                self.assertTrue(Path(model["result"]).is_file())
                self.assertTrue(Path(model["checkpoint"]).is_file())
                self.assertEqual(
                    hashlib.sha256(
                        Path(model["result"]).read_bytes()
                    ).hexdigest(),
                    model["result_sha256"],
                )
                self.assertEqual(
                    hashlib.sha256(
                        Path(model["checkpoint"]).read_bytes()
                    ).hexdigest(),
                    model["checkpoint_sha256"],
                )

    def test_selector_contains_no_training_or_confirmation_access(self):
        text = Path("Select_Task3_AdaptivePortfolio_52_1.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_train_one", text)
        self.assertIn("confirmation data", text)
        self.assertNotIn("Confirm_Task3", text)

    def test_independent_audit_reconstructs_winners_and_frozen_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            families = {
                "channel": ["channel"],
                "halfcylinder": [
                    "cylinder3d", "halfcylinderRe640",
                    "halfcylinderRe6400",
                ],
                "tangaroa": ["tangaroa"],
                "deltaWing": ["deltaWing_resampled", "deltaWing_LBM"],
                "f22raptor": ["f22raptor"],
                "boeing747": ["boeing747"],
                "smokeBuoyancy": ["smokeBuoyancy"],
            }
            for rank, (name, source) in enumerate(
                spec["sources"].items(), 1
            ):
                root = tmp_path / name
                source["repo_root"] = str(root)
                _fake_source(
                    root, source, spec["datasets"], families, rank
                )
            artifact_dir = tmp_path / "portfolio_output"
            spec["output_root"] = str(artifact_dir)
            config_path = tmp_path / "portfolio.yaml"
            config_path.write_text(
                yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
            )

            with redirect_stdout(io.StringIO()):
                selection_path = portfolio.select(config_path)
            result = audit_module.audit(
                config_path,
                selection_path.parent,
                selection_path.parent / "independent_audit.json",
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["counts"]["sources"], 5)
            self.assertEqual(result["counts"]["frozen_models"], 40)
            self.assertEqual(result["counts"]["frozen_artifact_files"], 80)
            self.assertLessEqual(
                result["maximum_absolute_difference_vs_portfolio"], 1e-12
            )
            self.assertEqual(
                set(result["selected_source_by_family"].values()),
                {"dropout_high"},
            )

            payload = json.loads(selection_path.read_text(encoding="utf-8"))
            checkpoint = Path(payload["models"][0]["checkpoint"])
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
            with self.assertRaisesRegex(RuntimeError, "checkpoint hash differs"):
                audit_module.audit(
                    config_path,
                    selection_path.parent,
                    selection_path.parent / "tampered_audit.json",
                )


if __name__ == "__main__":
    unittest.main()
