from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml

import Audit_Task3_ParameterSearch as audit_module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Task3ParameterSearchAuditTests(unittest.TestCase):
    def _fixture(self, root: Path, *, tamper_dataset: bool = False):
        artifact = root / "artifact"
        artifact.mkdir()
        config = {
            "experiment": "Synthetic_Task3_Search",
            "groups": {"family": {"datasets": ["flow"]}},
            "paired_seeds": [7],
            "optimization_candidates": [{"id": "c0"}, {"id": "c1"}],
            "selection": {
                "primary_metric": "dataset_macro_f1_gain_vs_raw_pca",
                "tie_breakers": ["dataset_macro_fmt_f1"],
                "absolute_fmt_guard": {
                    "control_optimization_id": "c0",
                    "f1_tolerance": 0.0,
                    "average_precision_tolerance": 0.0,
                },
            },
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        config_hash = "a" * 64
        upstream_hash = "b" * 64
        manifest = {
            "experiment": config["experiment"],
            "expected_training_runs": 4,
            "dataset_count": 1,
            "optimization_candidate_count": 2,
            "optimization_config_sha256": config_hash,
            "upstream_selection_sha256": upstream_hash,
        }
        manifest_path = artifact / "preflight_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_hash = _sha256(manifest_path)

        values = {
            ("c0", "raw_pca"): (0.40, 0.50),
            ("c0", "fmt"): (0.60, 0.70),
            ("c1", "raw_pca"): (0.20, 0.30),
            ("c1", "fmt"): (0.65, 0.75),
        }
        fieldnames = [
            "dataset",
            "seed",
            "optimization_id",
            "parameter_count",
            "trainable_residual_parameter_count",
            "validation_f1",
            "validation_average_precision",
            "optimization_config_sha256",
            "preflight_manifest_sha256",
            "upstream_selection_sha256",
        ]
        with tarfile.open(artifact / "per_run_csv.tar.gz", "w:gz") as archive:
            for candidate in ("c0", "c1"):
                for arm in ("fmt", "raw_pca"):
                    f1, average_precision = values[(candidate, arm)]
                    row = {
                        "dataset": (
                            "wrong-flow"
                            if tamper_dataset and candidate == "c1" and arm == "fmt"
                            else "flow"
                        ),
                        "seed": 7,
                        "optimization_id": candidate,
                        "parameter_count": 100,
                        "trainable_residual_parameter_count": 10,
                        "validation_f1": f1,
                        "validation_average_precision": average_precision,
                        "optimization_config_sha256": config_hash,
                        "preflight_manifest_sha256": manifest_hash,
                        "upstream_selection_sha256": upstream_hash,
                    }
                    stream = io.StringIO(newline="")
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)
                    payload = stream.getvalue().encode("utf-8")
                    member = tarfile.TarInfo(
                        f"candidates/{candidate}/flow/seed7/{arm}/per_run.csv"
                    )
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))

        leaderboard_fieldnames = [
            "physical_family",
            "optimization_id",
            "eligible",
            "dataset_macro_fmt_f1",
            "dataset_macro_raw_pca_f1",
            "dataset_macro_fmt_average_precision",
            "dataset_macro_raw_pca_average_precision",
            "dataset_macro_f1_gain_vs_raw_pca",
            "dataset_macro_average_precision_gain_vs_raw_pca",
            "positive_dataset_count",
            "worst_dataset_f1_gain",
            "worst_seed_f1_gain",
            "minimum_total_parameter_count",
            "maximum_total_parameter_count",
        ]
        leaderboard = [
            {
                "physical_family": "family",
                "optimization_id": "c0",
                "eligible": True,
                "dataset_macro_fmt_f1": 0.60,
                "dataset_macro_raw_pca_f1": 0.40,
                "dataset_macro_fmt_average_precision": 0.70,
                "dataset_macro_raw_pca_average_precision": 0.50,
                "dataset_macro_f1_gain_vs_raw_pca": 0.20,
                "dataset_macro_average_precision_gain_vs_raw_pca": 0.20,
                "positive_dataset_count": 1,
                "worst_dataset_f1_gain": 0.20,
                "worst_seed_f1_gain": 0.20,
                "minimum_total_parameter_count": 100,
                "maximum_total_parameter_count": 100,
            },
            {
                "physical_family": "family",
                "optimization_id": "c1",
                "eligible": True,
                "dataset_macro_fmt_f1": 0.65,
                "dataset_macro_raw_pca_f1": 0.20,
                "dataset_macro_fmt_average_precision": 0.75,
                "dataset_macro_raw_pca_average_precision": 0.30,
                "dataset_macro_f1_gain_vs_raw_pca": 0.45,
                "dataset_macro_average_precision_gain_vs_raw_pca": 0.45,
                "positive_dataset_count": 1,
                "worst_dataset_f1_gain": 0.45,
                "worst_seed_f1_gain": 0.45,
                "minimum_total_parameter_count": 100,
                "maximum_total_parameter_count": 100,
            },
        ]
        with (artifact / "optimization_leaderboard.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=leaderboard_fieldnames)
            writer.writeheader()
            writer.writerows(leaderboard)

        selector = {
            "experiment": config["experiment"],
            "optimization_config_sha256": config_hash,
            "preflight_manifest_sha256": manifest_hash,
            "upstream_selection_sha256": upstream_hash,
            "primary_by_group": {
                "family": {"optimization_id": "c1"},
            },
            "development_dataset_macro_fmt_f1": 0.65,
            "development_dataset_macro_raw_pca_f1": 0.20,
            "development_dataset_macro_f1_gain_vs_raw_pca": 0.45,
            "development_dataset_macro_fmt_average_precision": 0.75,
            "development_dataset_macro_raw_pca_average_precision": 0.30,
            "development_dataset_macro_ap_gain_vs_raw_pca": 0.45,
        }
        (artifact / "optimization_selection.json").write_text(
            json.dumps(selector), encoding="utf-8"
        )
        return config_path, artifact

    def test_reconstructs_complete_selection_without_selector_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, artifact = self._fixture(root)
            result = audit_module.audit(
                config, artifact, artifact / "independent_audit.json"
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["selected_optimization_id_by_family"], {"family": "c1"}
        )
        self.assertEqual(result["counts"]["per_run_csv"], 4)
        self.assertAlmostEqual(result["dataset_macro"]["f1_gain"], 0.45)
        self.assertAlmostEqual(result["control_dataset_macro"]["f1_gain"], 0.20)
        self.assertLessEqual(result["maximum_absolute_difference_vs_selector"], 1e-12)

    def test_rejects_row_metadata_that_differs_from_archive_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, artifact = self._fixture(root, tamper_dataset=True)
            with self.assertRaisesRegex(RuntimeError, "dataset differs from path"):
                audit_module.audit(
                    config, artifact, artifact / "independent_audit.json"
                )


if __name__ == "__main__":
    unittest.main()
