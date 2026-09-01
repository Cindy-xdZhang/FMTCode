"""Contracts for residual-head normalization/activation refresh 99.1."""

import json
from pathlib import Path
import tempfile
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ResidualHeadNormActivationRefresh_99.1.yaml"


class ResidualHeadNormActivationRefreshTests(unittest.TestCase):
    def test_registered_factorial_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 10)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 99), ("smokeBuoyancy", 9))
        self.assertEqual(
            spec["allowed_source_overrides"],
            ["model.head_normalization", "model.head_activation"],
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.238)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.902)
        self.assertFalse(selection["confirmation_opened"])

    def test_exact_source_control_and_complete_three_by_three_grid(self):
        spec = _load_optimization_spec(CONFIG)
        control = spec["optimization_candidates"][0]
        self.assertEqual(control, {
            "id": "n00_control_source",
            "sources": ["portfolio"],
        })
        observed = {
            (row["model"]["head_normalization"], row["model"]["head_activation"])
            for row in spec["optimization_candidates"][1:]
        }
        expected = {
            (normalization, activation)
            for normalization in ("layernorm", "rmsnorm", "none")
            for activation in ("gelu", "silu", "relu")
        }
        self.assertEqual(observed, expected)
        for row in spec["optimization_candidates"][1:]:
            self.assertEqual(row["sources"], ["portfolio"])
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(
                set(row["model"]), {"head_normalization", "head_activation"}
            )

    def test_source_recipe_is_preserved_except_two_head_factors(self):
        spec = _load_optimization_spec(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "portfolio_selection.json"
            primary = {}
            for group in spec["groups"]:
                primary[group] = {
                    "optimization_id": f"source_{group}",
                    "optimization_recipe_json": json.dumps({
                        "id": f"source_{group}",
                        "fmt_feature": "fmt_all",
                        "model": {
                            "head_hidden_dim": 80,
                            "head_depth": 3,
                            "head_normalization": "rmsnorm",
                            "head_activation": "relu",
                            "auxiliary_projection_activation_residual_gain": 0.5,
                        },
                        "training": {"max_epochs": 160},
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_ResidualHeadCapacityRefreshPortfolio_98.1"
                ),
                "confirmation_opened": False,
                "primary_by_group": primary,
            }), encoding="utf-8")
            spec["combination_sources"]["portfolio"]["selection"] = str(
                source_path
            )
            resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(hashes), {"portfolio"})
        for rows in resolved.values():
            self.assertEqual(rows[0]["model"]["head_normalization"], "rmsnorm")
            self.assertEqual(rows[0]["model"]["head_activation"], "relu")
            self.assertEqual(rows[1]["model"]["head_normalization"], "layernorm")
            self.assertEqual(rows[1]["model"]["head_activation"], "gelu")
            self.assertEqual(rows[1]["model"]["head_hidden_dim"], 80)
            self.assertEqual(rows[1]["model"]["head_depth"], 3)
            self.assertEqual(
                rows[1]["model"][
                    "auxiliary_projection_activation_residual_gain"
                ],
                0.5,
            )
            self.assertEqual(rows[1]["training"]["max_epochs"], 160)

    def test_zero_tolerance_guard_and_evidence_contract(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "n00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_residual_head_norm_activation_refresh_99.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_residual_head_norm_activation_refresh_99.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-99%24", gpu)
        self.assertIn("expected 600 per-run CSV files", evidence)
        self.assertIn("expected 600 temporary checkpoints", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
