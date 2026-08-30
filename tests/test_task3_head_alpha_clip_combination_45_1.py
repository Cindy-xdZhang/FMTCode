"""Contracts for Task3 head x alpha x clipping combination 45.1."""

import json
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_combination_recipe,
)


CONFIG = "config/Verify_Task3_HeadAlphaClipCombination_45.1.yaml"


class HeadAlphaClipCombinationTests(unittest.TestCase):
    def test_complete_factorial_and_mapping_are_frozen(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = spec["optimization_candidates"]
        self.assertEqual(len(candidates), 8)
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        with self.assertRaises(IndexError):
            _decode_job(spec, 80)
        observed = {
            frozenset(row["sources"]) - {"feature"}
            for row in candidates
        }
        expected = {
            frozenset(name for index, name in enumerate(
                ("head", "alpha", "clipping")) if mask & (1 << index))
            for mask in range(8)
        }
        self.assertEqual(observed, expected)

    def test_sources_targets_and_control_are_frozen(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(set(spec["combination_sources"]), {
            "feature", "head", "alpha", "clipping",
        })
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        selection = spec["optimization_selection"]
        self.assertEqual(
            selection["absolute_fmt_guard"]["control_optimization_id"],
            "k00_feature_control",
        )
        self.assertEqual(selection["absolute_fmt_guard"]["f1_tolerance"], 0.0)
        self.assertEqual(
            selection["absolute_fmt_guard"]["average_precision_tolerance"],
            0.0,
        )
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_three_factors_merge_without_hidden_overwrite(self):
        source_rows = {
            "feature": {
                "optimization_id": "d06_aivd1w3_early",
                "optimization_recipe_json": json.dumps({
                    "id": "d06_aivd1w3_early",
                    "fmt_feature": "aivd1w3_early",
                }),
            },
            "head": {
                "optimization_id": "h09_w80_d1",
                "optimization_recipe_json": json.dumps({
                    "id": "h09_w80_d1",
                    "fmt_feature": "aivd1w3_early",
                    "model": {"head_hidden_dim": 80, "head_depth": 1},
                }),
            },
            "alpha": {
                "optimization_id": "t03_a05",
                "optimization_recipe_json": json.dumps({
                    "id": "t03_a05",
                    "fmt_feature": "aivd1w3_early",
                    "training": {"training_alpha": 0.5},
                }),
            },
            "clipping": {
                "optimization_id": "g04_clip100",
                "optimization_recipe_json": json.dumps({
                    "id": "g04_clip100",
                    "fmt_feature": "aivd1w3_early",
                    "training": {"gradient_clip_norm": 1.0},
                }),
            },
        }
        merged = _merge_combination_recipe(
            "k07_head_alpha_clipping",
            ["feature", "head", "alpha", "clipping"],
            source_rows,
        )
        self.assertEqual(merged["fmt_feature"], "aivd1w3_early")
        self.assertEqual(merged["model"], {
            "head_hidden_dim": 80,
            "head_depth": 1,
        })
        self.assertEqual(merged["training"], {
            "training_alpha": 0.5,
            "gradient_clip_norm": 1.0,
        })


if __name__ == "__main__":
    unittest.main()
