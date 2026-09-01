"""Contracts for paired residual-head capacity refresh 97.1."""

import json
from pathlib import Path
import tempfile
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ResidualHeadCapacityRefresh_97.1.yaml"


class ResidualHeadCapacityRefreshTests(unittest.TestCase):
    def test_registered_factorial_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 13)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 129), ("smokeBuoyancy", 12))
        self.assertEqual(
            spec["allowed_source_overrides"],
            ["model.head_hidden_dim", "model.head_depth"],
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.236)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.901)
        self.assertFalse(selection["confirmation_opened"])

    def test_exact_source_control_and_complete_four_by_three_grid(self):
        spec = _load_optimization_spec(CONFIG)
        control = spec["optimization_candidates"][0]
        self.assertEqual(control, {
            "id": "h00_control_source",
            "sources": ["portfolio"],
        })
        observed = {
            (row["model"]["head_hidden_dim"], row["model"]["head_depth"])
            for row in spec["optimization_candidates"][1:]
        }
        expected = {
            (width, depth)
            for width in (32, 48, 64, 80)
            for depth in (1, 2, 3)
        }
        self.assertEqual(observed, expected)
        for row in spec["optimization_candidates"][1:]:
            self.assertEqual(row["sources"], ["portfolio"])
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(
                set(row["model"]), {"head_hidden_dim", "head_depth"}
            )

    def test_source_recipe_is_preserved_except_capacity(self):
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
                            "head_hidden_dim": 72,
                            "head_depth": 4,
                            "auxiliary_projection": "blockwise_mlp_gelu",
                            "auxiliary_projection_activation_residual_gain": 0.5,
                        },
                        "training": {"max_epochs": 160},
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_AuxiliaryActivationResidualGainPortfolio_96.1"
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
            self.assertEqual(rows[0]["model"]["head_hidden_dim"], 72)
            self.assertEqual(rows[0]["model"]["head_depth"], 4)
            self.assertEqual(rows[1]["model"]["head_hidden_dim"], 32)
            self.assertEqual(rows[1]["model"]["head_depth"], 1)
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
                "control_optimization_id": "h00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_residual_head_capacity_refresh_97.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_residual_head_capacity_refresh_97.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-129%24", gpu)
        self.assertIn("expected 780 per-run CSV files", evidence)
        self.assertIn("expected 780 temporary checkpoints", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
