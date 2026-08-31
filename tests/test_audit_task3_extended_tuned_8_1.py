import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import Audit_Task3_AdaptiveTuned_7_2 as base_audit
import Audit_Task3_ExtendedTuned_8_1 as audit_module
import tests.test_audit_task3_adaptive_tuned_7_2 as fixture_module


CONFIG = Path("config/mainExp_Task3_3D_8.1.yaml")


class ExtendedTunedAuditTests(unittest.TestCase):
    def _fixture(self, root: Path):
        previous_config = fixture_module.BASE_CONFIG
        previous_source = base_audit.EXPECTED_SOURCE_EXPERIMENT
        try:
            fixture_module.BASE_CONFIG = CONFIG
            base_audit.EXPECTED_SOURCE_EXPERIMENT = (
                audit_module.EXPECTED_SOURCE_EXPERIMENT
            )
            return fixture_module._fixture(root)
        finally:
            fixture_module.BASE_CONFIG = previous_config
            base_audit.EXPECTED_SOURCE_EXPERIMENT = previous_source

    def test_independent_audit_recomputes_complete_8_1_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path, artifact_dir = self._fixture(Path(temporary))
            with redirect_stdout(io.StringIO()):
                report = audit_module.audit(
                    config_path,
                    artifact_dir,
                    artifact_dir / "independent_audit.json",
                )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["experiment"], "mainExp_Task3_3D_8.1")
            self.assertEqual(report["counts"], {
                "rows": 40,
                "datasets": 10,
                "families": 7,
                "paired_seeds": 2,
                "frozen_models": 40,
            })
            self.assertAlmostEqual(
                report["dataset_macro"]["f1_gain"], 0.20, places=12
            )

    def test_audit_rejects_summary_identity_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path, artifact_dir = self._fixture(Path(temporary))
            summary_path = artifact_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["source_portfolio_experiment"] = (
                "Verify_Task3_AdaptivePortfolio_52.1"
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "summary source_portfolio_experiment"
            ):
                audit_module.audit(config_path, artifact_dir)

    def test_audit_scope_restores_base_identity(self):
        original_experiment = base_audit.EXPECTED_EXPERIMENT
        original_source = base_audit.EXPECTED_SOURCE_EXPERIMENT
        with audit_module._configured_base():
            self.assertEqual(
                base_audit.EXPECTED_EXPERIMENT,
                audit_module.EXPECTED_EXPERIMENT,
            )
            self.assertEqual(
                base_audit.EXPECTED_SOURCE_EXPERIMENT,
                audit_module.EXPECTED_SOURCE_EXPERIMENT,
            )
        self.assertEqual(base_audit.EXPECTED_EXPERIMENT, original_experiment)
        self.assertEqual(base_audit.EXPECTED_SOURCE_EXPERIMENT, original_source)


if __name__ == "__main__":
    unittest.main()
