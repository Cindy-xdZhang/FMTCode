"""Contracts for the preregistered Task3 head/full-stack combination 48.1."""

import json
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_combination_recipe,
)


CONFIG = "config/Verify_Task3_HeadFullStackCombination_48.1.yaml"


class HeadFullStackCombinationTests(unittest.TestCase):
    def test_sources_and_candidate_grid_are_frozen(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 16)
        self.assertEqual(_decode_job(spec, 159), ("smokeBuoyancy", 15))
        with self.assertRaises(IndexError):
            _decode_job(spec, 160)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(set(spec["combination_sources"]), {
            "feature", "head", "alpha", "clipping", "betas", "batch",
            "positive_weight", "dropout", "focal", "ema", "smoothing",
            "epsilon", "cosine", "horizon",
        })

    def test_control_targets_and_confirmation_are_frozen(self):
        spec = _load_optimization_spec(CONFIG)
        selection = spec["optimization_selection"]
        self.assertEqual(selection["absolute_fmt_guard"], {
            "control_optimization_id": "u00_feature_control",
            "f1_tolerance": 0.0,
            "average_precision_tolerance": 0.0,
        })
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.20)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_every_stack_beyond_horizon_only_contains_head(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = {
            row["id"]: set(row["sources"])
            for row in spec["optimization_candidates"]
        }
        self.assertEqual(candidates["u00_feature_control"], {"feature"})
        self.assertEqual(candidates["u02_horizon"], {"feature", "horizon"})
        for candidate_id, sources in candidates.items():
            if candidate_id not in {"u00_feature_control", "u02_horizon"}:
                self.assertIn("head", sources)
        self.assertEqual(
            candidates["u10_head_all"],
            set(spec["combination_sources"]),
        )

    def test_head_dropout_ema_horizon_recipes_merge_without_conflict(self):
        source_rows = {
            "feature": {
                "optimization_id": "feature",
                "optimization_recipe_json": json.dumps({
                    "id": "feature", "fmt_feature": "aivd1w3_early",
                }),
            },
            "head": {
                "optimization_id": "head",
                "optimization_recipe_json": json.dumps({
                    "id": "head",
                    "model": {"head_hidden_dim": 128, "head_depth": 3},
                }),
            },
            "dropout": {
                "optimization_id": "dropout",
                "optimization_recipe_json": json.dumps({
                    "id": "dropout", "model": {"head_dropout": 0.4},
                }),
            },
            "ema": {
                "optimization_id": "ema",
                "optimization_recipe_json": json.dumps({
                    "id": "ema", "training": {"ema_decay": 0.99},
                }),
            },
            "horizon": {
                "optimization_id": "horizon",
                "optimization_recipe_json": json.dumps({
                    "id": "horizon", "training": {"max_epochs": 150},
                }),
            },
        }
        merged = _merge_combination_recipe(
            "combined", ["feature", "head", "dropout", "ema", "horizon"],
            source_rows,
        )
        self.assertEqual(merged["fmt_feature"], "aivd1w3_early")
        self.assertEqual(merged["model"], {
            "head_hidden_dim": 128,
            "head_depth": 3,
            "head_dropout": 0.4,
        })
        self.assertEqual(merged["training"], {
            "ema_decay": 0.99,
            "max_epochs": 150,
        })


if __name__ == "__main__":
    unittest.main()
