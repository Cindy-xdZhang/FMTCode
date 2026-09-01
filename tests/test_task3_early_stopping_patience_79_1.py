"""Contracts for Task3 early-stopping patience search 79.1."""

import json
from pathlib import Path
import tempfile
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_candidate_overrides,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_EarlyStoppingPatience_79.1.yaml"
EXPECTED = [None, 3, 5, 10, 15, 20, 30, 50, 80]


class EarlyStoppingPatienceTests(unittest.TestCase):
    def test_grid_job_count_source_and_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        source = spec["combination_sources"]["portfolio"]
        self.assertEqual(
            source["expected_experiment"],
            "Verify_Task3_AuxiliaryGradientClipPortfolio_78.1",
        )
        self.assertEqual(
            spec["allowed_source_overrides"], ["training.patience"]
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.219)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_patience(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["portfolio"])
            if index == 0:
                self.assertEqual(set(row), {"id", "sources"})
                observed.append(None)
            else:
                self.assertEqual(set(row), {"id", "sources", "training"})
                self.assertEqual(set(row["training"]), {"patience"})
                observed.append(row["training"]["patience"])
        self.assertEqual(observed, EXPECTED)

    def test_source_recipe_is_preserved_except_patience(self):
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
                        "model": {"auxiliary_post_normalization": "rms"},
                        "training": {
                            "patience": 20,
                            "max_epochs": 160,
                            "auxiliary_gradient_clip_norm": 0.3,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": "Verify_Task3_AuxiliaryGradientClipPortfolio_78.1",
                "confirmation_opened": False,
                "primary_by_group": primary,
            }), encoding="utf-8")
            spec["combination_sources"]["portfolio"]["selection"] = str(
                source_path
            )
            resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(hashes), {"portfolio"})
        for rows in resolved.values():
            self.assertEqual(rows[0]["training"]["patience"], 20)
            self.assertEqual(rows[0]["training"]["max_epochs"], 160)
            self.assertEqual(rows[1]["training"]["patience"], 3)
            self.assertEqual(rows[1]["training"]["max_epochs"], 160)
            self.assertEqual(
                rows[1]["training"]["auxiliary_gradient_clip_norm"], 0.3
            )

    def test_override_remains_strict_by_default(self):
        merged = {"id": "source", "training": {"patience": 20}}
        candidate = {"id": "candidate", "training": {"patience": 3}}
        with self.assertRaisesRegex(ValueError, "frozen training.patience"):
            _merge_candidate_overrides(merged, candidate)
        result = _merge_candidate_overrides(
            merged, candidate, ["training.patience"]
        )
        self.assertEqual(result["training"]["patience"], 3)

    def test_patience_grid_is_positive_integer_and_monotone(self):
        values = EXPECTED[1:]
        self.assertTrue(all(isinstance(value, int) for value in values))
        self.assertTrue(all(value > 0 for value in values))
        self.assertEqual(values, sorted(set(values)))

    def test_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "p00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_slurm_scripts_match_grid_and_retain_models(self):
        gpu = Path(
            "ibex_bash/verify_task3_early_stopping_patience_79.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_early_stopping_patience_79.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-89%24", gpu)
        self.assertIn("expected 540 per-run CSV files", evidence)
        self.assertIn("expected 540 temporary checkpoints", evidence)
        self.assertIn("Audit_Task3_ParameterSearch.py", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
