"""Contracts for residual-head weight-decay portfolio 104.1."""

from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import unittest

import yaml

import Audit_Task3_ResidualHeadWeightDecayRefreshPortfolio_104_1 as audit_module
import Audit_Task3_AuxiliaryLearningRatePortfolio_56_1 as base_audit
import Select_Task3_AuxiliaryLearningRatePortfolio_56_1 as base_selector
from Select_Task3_ResidualHeadWeightDecayRefreshPortfolio_104_1 import (
    _canonical_sha256,
    _configured_contract,
    _load_spec,
    static_preflight,
)


CONFIG = Path(
    "config/Verify_Task3_ResidualHeadWeightDecayRefreshPortfolio_104.1.yaml"
)
SOURCE_102_CONFIG = Path(
    "config/Verify_Task3_ResidualHeadLearningRateRefreshPortfolio_102.1.yaml"
)
SOURCE_103_CONFIG = Path(
    "config/Verify_Task3_ResidualHeadWeightDecayRefresh_103.1.yaml"
)


class ResidualHeadWeightDecayRefreshPortfolioTests(unittest.TestCase):
    def test_registered_contract(self):
        spec = _load_spec(CONFIG)
        self.assertEqual(
            set(spec["sources"]),
            {"current_portfolio", "residual_head_weight_decay_refresh"},
        )
        self.assertEqual(spec["source_paired_seeds"], [40, 41, 42])
        self.assertEqual(spec["frozen_confirmation_seeds"], [40, 41])
        self.assertEqual(len(spec["datasets"]), 10)
        self.assertEqual(
            spec["selection"]["target_dataset_macro_f1_gain"], 0.242
        )
        self.assertEqual(spec["selection"]["target_absolute_fmt_f1"], 0.904)
        self.assertFalse(spec["confirmation_opened"])

    def test_source_config_hashes_are_frozen(self):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            _canonical_sha256(SOURCE_102_CONFIG),
            config["sources"]["current_portfolio"][
                "expected_config_canonical_sha256"
            ],
        )
        self.assertEqual(
            _canonical_sha256(SOURCE_103_CONFIG),
            config["sources"]["residual_head_weight_decay_refresh"][
                "expected_config_canonical_sha256"
            ],
        )

    def test_static_preflight_reads_no_performance_artifact(self):
        before = (
            base_selector.EXPERIMENT,
            base_selector.AUXILIARY_SOURCE_NAME,
            base_selector.AUXILIARY_ARCHIVE_COUNT,
        )
        with redirect_stdout(io.StringIO()):
            report = static_preflight(CONFIG)
        self.assertFalse(report["performance_artifacts_read"])
        self.assertFalse(report["confirmation_opened"])
        self.assertEqual(report["training_runs"], 0)
        self.assertEqual(report["dataset_count"], 10)
        self.assertEqual(report["source_count"], 2)
        self.assertEqual(
            before,
            (
                base_selector.EXPERIMENT,
                base_selector.AUXILIARY_SOURCE_NAME,
                base_selector.AUXILIARY_ARCHIVE_COUNT,
            ),
        )

    def test_wrapper_reconfigures_and_restores_shared_selector(self):
        original = (
            base_selector.EXPERIMENT,
            base_selector.CURRENT_SOURCE_LABEL,
            base_selector.AUXILIARY_SOURCE_LABEL,
            base_selector.PORTFOLIO_LABEL,
        )
        with _configured_contract():
            self.assertEqual(
                (
                    base_selector.EXPERIMENT,
                    base_selector.CURRENT_SOURCE_LABEL,
                    base_selector.AUXILIARY_SOURCE_LABEL,
                    base_selector.PORTFOLIO_LABEL,
                ),
                (
                    "Verify_Task3_ResidualHeadWeightDecayRefreshPortfolio_104.1",
                    "102.1", "103.1", "104.1",
                ),
            )
        self.assertEqual(
            original,
            (
                base_selector.EXPERIMENT,
                base_selector.CURRENT_SOURCE_LABEL,
                base_selector.AUXILIARY_SOURCE_LABEL,
                base_selector.PORTFOLIO_LABEL,
            ),
        )

    def test_auditor_and_cleanup_contract(self):
        source = inspect.getsource(audit_module)
        self.assertNotIn(
            "Select_Task3_ResidualHeadWeightDecayRefreshPortfolio_104_1",
            source,
        )
        self.assertNotIn(
            "Select_Task3_AuxiliaryLearningRatePortfolio_56_1", source
        )
        self.assertEqual(base_audit.AUXILIARY_ARCHIVE_COUNT, 540)
        evidence = Path(
            "ibex_bash/verify_task3_residual_head_weight_decay_refresh_portfolio_104.1_evidence.sh"
        ).read_text(encoding="utf-8")
        cleanup = Path(
            "ibex_bash/verify_task3_residual_head_weight_decay_refresh_103.1_cleanup.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('result_count" -ne 40', evidence)
        self.assertIn('total_count" -ne 80', evidence)
        self.assertIn('checkpoint_count_before" -ne 660', cleanup)
        self.assertLess(cleanup.index("portfolio_audit"), cleanup.index("-delete"))


if __name__ == "__main__":
    unittest.main()
