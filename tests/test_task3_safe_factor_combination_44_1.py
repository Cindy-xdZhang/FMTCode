"""Contracts for the preregistered Task3 safe-factor combination 44.1."""

import json
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_combination_recipe,
)


CONFIG = "config/Verify_Task3_SafeFactorCombination_44.1.yaml"


class SafeFactorCombinationTests(unittest.TestCase):
    def test_sources_and_candidate_grid_are_frozen(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 28)
        self.assertEqual(_decode_job(spec, 279), ("smokeBuoyancy", 27))
        with self.assertRaises(IndexError):
            _decode_job(spec, 280)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(set(spec["combination_sources"]), {
            "feature", "alpha", "clipping", "betas", "batch",
            "positive_weight", "dropout", "focal", "ema", "smoothing",
            "epsilon", "cosine",
        })
        self.assertNotIn("warmup", spec["combination_sources"])

    def test_control_and_joint_target_are_unchanged(self):
        spec = _load_optimization_spec(CONFIG)
        selection = spec["optimization_selection"]
        self.assertEqual(selection["absolute_fmt_guard"], {
            "control_optimization_id": "k00_feature_control",
            "f1_tolerance": 0.0,
            "average_precision_tolerance": 0.0,
        })
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_alpha_interactions_cover_every_other_factor(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = {
            row["id"]: set(row["sources"])
            for row in spec["optimization_candidates"]
        }
        other = {
            "clipping", "betas", "batch", "positive_weight", "dropout",
            "focal", "ema", "smoothing", "epsilon", "cosine",
        }
        observed = {
            next(iter(sources - {"feature", "alpha"}))
            for candidate_id, sources in candidates.items()
            if candidate_id.startswith("k1") or candidate_id in {
                "k20_alpha_epsilon", "k21_alpha_cosine"
            }
            if sources.issuperset({"feature", "alpha"})
            and len(sources) == 3
        }
        self.assertEqual(observed, other)

    def test_compatible_frozen_recipes_merge_without_overwrite(self):
        source_rows = {
            "feature": {
                "optimization_id": "d06_aivd1w3_early",
                "optimization_recipe_json": json.dumps({
                    "id": "d06_aivd1w3_early",
                    "fmt_feature": "aivd1w3_early",
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
            "focal": {
                "optimization_id": "f07_gamma200",
                "optimization_recipe_json": json.dumps({
                    "id": "f07_gamma200",
                    "fmt_feature": "aivd1w3_early",
                    "training": {"loss": "focal", "focal_gamma": 2.0},
                }),
            },
            "smoothing": {
                "optimization_id": "s03_eps0005",
                "optimization_recipe_json": json.dumps({
                    "id": "s03_eps0005",
                    "fmt_feature": "aivd1w3_early",
                    "training": {"label_smoothing": 0.005},
                }),
            },
        }
        merged = _merge_combination_recipe(
            "k23_loss_regularization_stack",
            ["feature", "alpha", "focal", "smoothing"],
            source_rows,
        )
        self.assertEqual(merged["fmt_feature"], "aivd1w3_early")
        self.assertEqual(merged["training"], {
            "training_alpha": 0.5,
            "loss": "focal",
            "focal_gamma": 2.0,
            "label_smoothing": 0.005,
        })


if __name__ == "__main__":
    unittest.main()
