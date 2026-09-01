"""Contracts for Task3 early-stopping minimum-delta search 81.1."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_candidate_overrides,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_EarlyStoppingMinDelta_81.1.yaml"
EXPECTED = [None, 0.0, 1e-6, 3e-6, 1e-5, 3e-5, 3e-4, 1e-3, 3e-3, 1e-2]


class EarlyStoppingMinDeltaTests(unittest.TestCase):
    def test_grid_job_count_source_and_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 10)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 99), ("smokeBuoyancy", 9))
        source = spec["combination_sources"]["portfolio"]
        self.assertEqual(
            source["expected_experiment"],
            "Verify_Task3_EarlyStoppingPatiencePortfolio_80.1",
        )
        self.assertEqual(
            spec["allowed_source_overrides"], ["training.min_delta"]
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.220)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_min_delta(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["portfolio"])
            if index == 0:
                self.assertEqual(set(row), {"id", "sources"})
                observed.append(None)
            else:
                self.assertEqual(set(row), {"id", "sources", "training"})
                self.assertEqual(set(row["training"]), {"min_delta"})
                observed.append(row["training"]["min_delta"])
        self.assertEqual(observed, EXPECTED)

    def test_source_recipe_is_preserved_except_min_delta(self):
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
                            "min_delta": 1e-4,
                            "patience": 30,
                            "max_epochs": 160,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": "Verify_Task3_EarlyStoppingPatiencePortfolio_80.1",
                "confirmation_opened": False,
                "primary_by_group": primary,
            }), encoding="utf-8")
            spec["combination_sources"]["portfolio"]["selection"] = str(
                source_path
            )
            resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(hashes), {"portfolio"})
        for rows in resolved.values():
            self.assertEqual(rows[0]["training"]["min_delta"], 1e-4)
            self.assertEqual(rows[0]["training"]["patience"], 30)
            self.assertEqual(rows[1]["training"]["min_delta"], 0.0)
            self.assertEqual(rows[1]["training"]["patience"], 30)
            self.assertEqual(rows[1]["training"]["max_epochs"], 160)

    def test_override_remains_strict_by_default(self):
        merged = {"id": "source", "training": {"min_delta": 1e-4}}
        candidate = {"id": "candidate", "training": {"min_delta": 0.0}}
        with self.assertRaisesRegex(ValueError, "frozen training.min_delta"):
            _merge_candidate_overrides(merged, candidate)
        result = _merge_candidate_overrides(
            merged, candidate, ["training.min_delta"]
        )
        self.assertEqual(result["training"]["min_delta"], 0.0)

    def test_grid_is_finite_nonnegative_unique_and_ordered(self):
        values = EXPECTED[1:]
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(all(value >= 0.0 for value in values))
        self.assertEqual(values, sorted(set(values)))
        self.assertNotIn(1e-4, values)

    def test_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "d00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_slurm_scripts_match_grid_and_retain_models(self):
        gpu = Path(
            "ibex_bash/verify_task3_early_stopping_min_delta_81.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_early_stopping_min_delta_81.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-99%24", gpu)
        self.assertIn("expected 600 per-run CSV files", evidence)
        self.assertIn("expected 600 temporary checkpoints", evidence)
        self.assertIn("Audit_Task3_ParameterSearch.py", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
