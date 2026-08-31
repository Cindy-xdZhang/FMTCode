"""Contracts for the preregistered Task3 portfolio 56.1."""

import csv
from contextlib import redirect_stdout
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

import Audit_Task3_AuxiliaryLearningRatePortfolio_56_1 as audit_module
from Audit_Task3_AuxiliaryLearningRatePortfolio_56_1 import audit
from Select_Task3_AuxiliaryLearningRatePortfolio_56_1 import (
    _canonical_sha256,
    _load_spec,
    source_identity_preflight,
    static_preflight,
    select,
)


CONFIG = Path("config/Verify_Task3_AuxiliaryLearningRatePortfolio_56.1.yaml")
SOURCE_55_CONFIG = Path("config/Verify_Task3_AuxiliaryLearningRate_55.1.yaml")


class AuxiliaryLearningRatePortfolioTests(unittest.TestCase):
    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path):
        return __import__("hashlib").sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _family_map():
        return {
            "channel": ["channel"],
            "halfcylinder": [
                "cylinder3d", "halfcylinderRe640", "halfcylinderRe6400"
            ],
            "tangaroa": ["tangaroa"],
            "deltaWing": ["deltaWing_resampled", "deltaWing_LBM"],
            "f22raptor": ["f22raptor"],
            "boeing747": ["boeing747"],
            "smokeBuoyancy": ["smokeBuoyancy"],
        }

    @classmethod
    def _selection_rows(cls, gain, candidate):
        rows = {}
        for family, datasets in cls._family_map().items():
            details = {
                dataset: {
                    "fmt": {"f1": 0.80 + gain, "average_precision": 0.90},
                    "raw_pca": {"f1": 0.80, "average_precision": 0.70},
                    "f1_gain": gain,
                    "average_precision_gain": 0.20,
                }
                for dataset in datasets
            }
            rows[family] = {
                "optimization_id": candidate,
                "datasets_json": json.dumps(details, sort_keys=True),
                "dataset_macro_f1_gain_vs_raw_pca": gain,
                "dataset_macro_fmt_f1": 0.80 + gain,
                "dataset_macro_average_precision_gain_vs_raw_pca": 0.20,
                "dataset_macro_fmt_average_precision": 0.90,
                "positive_dataset_count": len(datasets),
                "worst_dataset_f1_gain": gain,
                "worst_seed_f1_gain": gain,
                "eligible": True,
                "absolute_fmt_guard_passed": True,
            }
        return rows

    def test_registered_sources_metrics_and_seeds(self):
        spec = _load_spec(CONFIG)
        self.assertEqual(
            set(spec["sources"]),
            {"current_portfolio", "auxiliary_learning_rate"},
        )
        self.assertEqual(spec["source_paired_seeds"], [40, 41, 42])
        self.assertEqual(spec["frozen_confirmation_seeds"], [40, 41])
        self.assertEqual(len(spec["datasets"]), 10)
        self.assertEqual(
            spec["selection_metrics"][0],
            "dataset_macro_f1_gain_vs_raw_pca",
        )
        self.assertEqual(
            spec["selection"]["target_dataset_macro_f1_gain"], 0.20552
        )
        self.assertEqual(spec["selection"]["target_absolute_fmt_f1"], 0.893)

    def test_static_preflight_opens_no_performance_artifact(self):
        with redirect_stdout(io.StringIO()):
            report = static_preflight(CONFIG)
        self.assertEqual(report["training_runs"], 0)
        self.assertFalse(report["confirmation_opened"])
        self.assertFalse(report["performance_artifacts_read"])

    def test_registered_55_source_hash_matches_frozen_config(self):
        overlay = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        expected = overlay["sources"]["auxiliary_learning_rate"][
            "expected_config_canonical_sha256"
        ]
        self.assertEqual(_canonical_sha256(SOURCE_55_CONFIG), expected)

    def test_source_identity_preflight_needs_configs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            overlay["output_root"] = str(root / "output")
            experiments = {
                "current_portfolio": "Verify_Task3_ExtendedPortfolio_54.1",
                "auxiliary_learning_rate": (
                    "Verify_Task3_AuxiliaryLearningRate_55.1"
                ),
            }
            for name, experiment in experiments.items():
                source_root = root / name
                source_root.mkdir()
                config = source_root / "source.yaml"
                config.write_text(
                    yaml.safe_dump({"experiment": experiment}, sort_keys=False),
                    encoding="utf-8",
                )
                overlay["sources"][name]["repo_root"] = str(source_root)
                overlay["sources"][name]["paths"]["config"] = "source.yaml"
                overlay["sources"][name][
                    "expected_config_canonical_sha256"
                ] = _canonical_sha256(config)
            temporary_config = root / "portfolio.yaml"
            temporary_config.write_text(
                yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
            )
            with redirect_stdout(io.StringIO()):
                target = source_identity_preflight(temporary_config)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertFalse(payload["performance_artifacts_read"])
            self.assertEqual(set(payload["sources"]), set(experiments))

    def test_auditor_does_not_import_selector(self):
        source = inspect.getsource(audit_module)
        self.assertNotIn(
            "Select_Task3_AuxiliaryLearningRatePortfolio_56_1", source
        )
        self.assertIn("independent_of_portfolio_selector_implementation", source)

    def test_evidence_and_cleanup_contract_counts(self):
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_learning_rate_55.1_evidence.sh"
        ).read_text(encoding="utf-8")
        cleanup = Path(
            "ibex_bash/verify_task3_auxiliary_learning_rate_55.1_cleanup.sh"
        ).read_text(encoding="utf-8")
        portfolio_evidence = Path(
            "ibex_bash/verify_task3_auxiliary_learning_rate_portfolio_56.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('per_run_count" -ne 540', evidence)
        self.assertIn('checkpoint_count_before" -ne 540', cleanup)
        self.assertIn('result_count" -ne 40', portfolio_evidence)
        self.assertIn('total_count" -ne 80', portfolio_evidence)
        self.assertLess(
            cleanup.index("portfolio_audit"), cleanup.index("-delete")
        )

    def test_synthetic_selector_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_root = root / "current"
            auxiliary_root = root / "auxiliary"
            portfolio_root = root / "portfolio"
            for value in (current_root, auxiliary_root, portfolio_root):
                value.mkdir()

            current_config = current_root / "source.yaml"
            current_config.write_text(
                yaml.safe_dump({
                    "experiment": "Verify_Task3_ExtendedPortfolio_54.1",
                    "output_root": "outputs/current",
                }, sort_keys=False), encoding="utf-8"
            )
            current_output = current_root / "outputs/current"
            current_models = []
            for datasets in self._family_map().values():
                for dataset in datasets:
                    for seed in (40, 41):
                        for arm in ("fmt", "raw_pca"):
                            location = current_output / "frozen" / dataset / f"seed{seed}" / arm
                            location.mkdir(parents=True, exist_ok=True)
                            result = location / "per_run.csv"
                            checkpoint = location / "best.pt"
                            result.write_text(f"current,{dataset},{seed},{arm}\n", encoding="utf-8")
                            checkpoint.write_bytes(f"current-{dataset}-{seed}-{arm}".encode())
                            current_models.append({
                                "dataset": dataset, "seed": seed, "source": arm,
                                "fmt_dim": 8, "parameter_count": 100,
                                "trainable_residual_parameter_count": 40,
                                "result": str(result.resolve()),
                                "result_sha256": self._sha256(result),
                                "checkpoint": str(checkpoint.resolve()),
                                "checkpoint_sha256": self._sha256(checkpoint),
                            })
            current_selection = current_output / "portfolio_selection.json"
            self._write_json(current_selection, {
                "experiment": "Verify_Task3_ExtendedPortfolio_54.1",
                "confirmation_opened": False,
                "primary_by_group": self._selection_rows(0.10, "current"),
                "models": current_models,
            })
            current_audit = current_output / "independent_audit.json"
            self._write_json(current_audit, {
                "status": "passed", "all_frozen_hashes_verified": True,
                "input_sha256": {
                    "config": self._sha256(current_config),
                    "portfolio_selection": self._sha256(current_selection),
                },
            })

            auxiliary_config = auxiliary_root / "source.yaml"
            auxiliary_config.write_text(
                yaml.safe_dump({
                    "experiment": "Verify_Task3_AuxiliaryLearningRate_55.1",
                    "output_root": "outputs/auxiliary",
                }, sort_keys=False), encoding="utf-8"
            )
            auxiliary_output = auxiliary_root / "outputs/auxiliary"
            preflight = auxiliary_output / "preflight_manifest.json"
            self._write_json(preflight, {
                "experiment": "Verify_Task3_AuxiliaryLearningRate_55.1",
                "confirmation_opened": False,
            })
            auxiliary_rows = self._selection_rows(0.15, "auxiliary")
            auxiliary_selection = auxiliary_output / "optimization_selection.json"
            self._write_json(auxiliary_selection, {
                "experiment": "Verify_Task3_AuxiliaryLearningRate_55.1",
                "confirmation_opened": False,
                "primary_by_group": auxiliary_rows,
            })
            for family, datasets in self._family_map().items():
                candidate = auxiliary_rows[family]["optimization_id"]
                for dataset in datasets:
                    for seed in (40, 41):
                        for arm in ("fmt", "raw_pca"):
                            location = (
                                auxiliary_output / "candidates" / candidate
                                / dataset / f"seed{seed}" / arm
                            )
                            checkpoints = location / "checkpoints"
                            checkpoints.mkdir(parents=True, exist_ok=True)
                            checkpoint = checkpoints / "best.pt"
                            checkpoint.write_bytes(
                                f"auxiliary-{dataset}-{seed}-{arm}".encode()
                            )
                            result = location / "per_run.csv"
                            with result.open("w", newline="", encoding="utf-8") as stream:
                                writer = csv.DictWriter(stream, fieldnames=[
                                    "dataset", "seed", "variant", "auxiliary_source",
                                    "optimization_id", "fmt_feature", "fmt_dim",
                                    "parameter_count",
                                    "trainable_residual_parameter_count", "checkpoint",
                                ])
                                writer.writeheader()
                                writer.writerow({
                                    "dataset": dataset, "seed": seed,
                                    "variant": (
                                        "raw_fmt_residual" if arm == "fmt"
                                        else "raw_pca_residual"
                                    ),
                                    "auxiliary_source": arm,
                                    "optimization_id": candidate,
                                    "fmt_feature": "synthetic", "fmt_dim": 8,
                                    "parameter_count": 100,
                                    "trainable_residual_parameter_count": 40,
                                    "checkpoint": str(checkpoint.resolve()),
                                })
            archive = auxiliary_output / "per_run_csv.tar.gz"
            archive.write_bytes(b"synthetic-archive")
            auxiliary_audit = auxiliary_output / "independent_audit.json"
            self._write_json(auxiliary_audit, {
                "status": "passed", "all_source_hashes_consistent": True,
                "input_sha256": {
                    "optimization_selection": self._sha256(auxiliary_selection),
                    "preflight_manifest": self._sha256(preflight),
                    "per_run_csv_archive": self._sha256(archive),
                },
            })
            evidence = auxiliary_output / "evidence_archive.json"
            self._write_json(evidence, {
                "status": "passed", "archived_per_run_csv": 540,
                "stable_archive_sha256": self._sha256(archive),
            })

            config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            config["output_root"] = str(portfolio_root / "outputs/portfolio")
            current = config["sources"]["current_portfolio"]
            current["repo_root"] = str(current_root)
            current["expected_config_canonical_sha256"] = _canonical_sha256(
                current_config
            )
            current["paths"] = {
                "config": "source.yaml",
                "selection": "outputs/current/portfolio_selection.json",
                "audit": "outputs/current/independent_audit.json",
                "frozen_root": "outputs/current/frozen",
            }
            auxiliary = config["sources"]["auxiliary_learning_rate"]
            auxiliary["repo_root"] = str(auxiliary_root)
            auxiliary["expected_config_canonical_sha256"] = _canonical_sha256(
                auxiliary_config
            )
            auxiliary["paths"] = {
                "config": "source.yaml",
                "preflight": "outputs/auxiliary/preflight_manifest.json",
                "selection": "outputs/auxiliary/optimization_selection.json",
                "audit": "outputs/auxiliary/independent_audit.json",
                "evidence": "outputs/auxiliary/evidence_archive.json",
                "archive": "outputs/auxiliary/per_run_csv.tar.gz",
                "candidate_root": "outputs/auxiliary/candidates",
            }
            config_path = portfolio_root / "portfolio.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )

            with redirect_stdout(io.StringIO()):
                source_identity_preflight(config_path)
                selection_path = select(config_path)
            portfolio = json.loads(selection_path.read_text(encoding="utf-8"))
            self.assertEqual(len(portfolio["models"]), 40)
            self.assertEqual(
                {row["portfolio_source"] for row in portfolio["primary_by_group"].values()},
                {"auxiliary_learning_rate"},
            )
            audit_path = selection_path.parent / "independent_audit.json"
            result = audit(config_path, selection_path.parent, audit_path)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["all_frozen_hashes_verified"])
            self.assertEqual(result["counts"]["frozen_artifact_files"], 80)


if __name__ == "__main__":
    unittest.main()
