import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import yaml

import Audit_Task3_AdaptivePortfolio as audit_module
import Select_Task3_ExtendedPortfolio_54_1 as portfolio
from tests.test_task3_final_portfolio_49_1 import _fake_source


CONFIG = Path("config/Verify_Task3_ExtendedPortfolio_54.1.yaml")


class ExtendedPortfolioTests(unittest.TestCase):
    def test_static_contract_is_training_free_and_uses_six_sources(self):
        spec = portfolio._load_spec(CONFIG)
        report = portfolio.static_preflight(CONFIG)
        self.assertEqual(report["source_count"], 6)
        self.assertEqual(report["training_runs"], 0)
        self.assertFalse(report["confirmation_opened"])
        self.assertEqual(set(spec["sources"]), portfolio.EXPECTED_SOURCES)
        self.assertEqual(spec["frozen_confirmation_seeds"], [40, 41])

    def test_sources_are_exact_preregistered_searches(self):
        spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        observed = {
            name: row["expected_experiment"]
            for name, row in spec["sources"].items()
        }
        self.assertEqual(observed, {
            "safe_factor": "Verify_Task3_SafeFactorCombination_44.1",
            "head_alpha_clip": (
                "Verify_Task3_HeadAlphaClipCombination_45.1"
            ),
            "full_stack": "Verify_Task3_HeadFullStackCombination_48.1",
            "focal_gamma_low": "Verify_Task3_FocalGammaLow_50.1",
            "dropout_high": "Verify_Task3_ResidualDropoutHigh_51.1",
            "auxiliary_dropout": "Verify_Task3_AuxiliaryDropout_53.1",
        })
        expected_hashes = {
            "safe_factor": (
                "cb278708c88823b07443c8cbbdc3deaa25555c7cdab6316297dd253494abaa40"
            ),
            "head_alpha_clip": (
                "5dc8535491f8b7edc0cc5458f1ed5c81c502739b8c16e8c77a04f1c02e136b6f"
            ),
            "full_stack": (
                "90c47bc6531e8a88d4b144b4d05d1e8070a8dad722dadfcb089e2d991fe32dd4"
            ),
            "focal_gamma_low": (
                "e37d7be09fc3719ece7c34b790d4e3aeb28527ae869321c5f0dc9f1b0f047888"
            ),
            "dropout_high": (
                "c57331663bf32011077706d0444c2aca1a6d76ca331c2a7b0c37cf28750153f3"
            ),
            "auxiliary_dropout": (
                "58936cd1460cb5f8359c58af6408f73b9fd354b94f9e13a5a8c2a2050aa17e7b"
            ),
        }
        self.assertEqual({
            name: row["expected_config_canonical_sha256"]
            for name, row in spec["sources"].items()
        }, expected_hashes)

    def test_base_loader_is_restored_after_scoped_use(self):
        original = portfolio._base._load_spec
        with portfolio._configured_base():
            self.assertIs(portfolio._base._load_spec, portfolio._load_spec)
        self.assertIs(portfolio._base._load_spec, original)

    def test_source_identity_preflight_reads_no_performance_artifact(self):
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
            self.assertEqual(payload["source_count"], 6)
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
            for model in frozen["models"]:
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
        text = Path("Select_Task3_ExtendedPortfolio_54_1.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_train_one", text)
        self.assertIn("confirmation data", text)
        self.assertNotIn("Confirm_Task3", text)

    def test_independent_audit_handles_six_sources(self):
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
                _fake_source(root, source, spec["datasets"], families, rank)
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
            self.assertEqual(result["counts"]["sources"], 6)
            self.assertEqual(result["counts"]["frozen_models"], 40)
            self.assertEqual(result["counts"]["frozen_artifact_files"], 80)
            self.assertLessEqual(
                result["maximum_absolute_difference_vs_portfolio"], 1e-12
            )
            self.assertEqual(
                set(result["selected_source_by_family"].values()),
                {"auxiliary_dropout"},
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
