import json
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_combination_recipe,
)


CONFIG = "config/Verify_Task3_OverlapBalancedCombination_16.1.yaml"


class OverlapBalancedCombinationContractTests(unittest.TestCase):
    def test_config_declares_complete_two_by_two_factorial(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            set(spec["combination_sources"]),
            {"core", "balanced", "overlap"},
        )
        self.assertEqual(
            [row["sources"] for row in spec["optimization_candidates"]],
            [
                ["core"],
                ["core", "balanced"],
                ["core", "overlap"],
                ["core", "balanced", "overlap"],
            ],
        )
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(_decode_job(spec, 39), ("smokeBuoyancy", 3))
        with self.assertRaises(IndexError):
            _decode_job(spec, 40)

    def test_three_frozen_sources_merge_without_hidden_overwrite(self):
        source_rows = {
            "core": {
                "optimization_id": "m15_all",
                "optimization_recipe_json": json.dumps({
                    "id": "m15_all",
                    "sources": ["loss", "hardness", "correction", "ranking"],
                    "source_optimization_ids": {
                        "loss": "o09",
                        "hardness": "h02",
                        "correction": "r00",
                        "ranking": "q04",
                    },
                    "training": {
                        "focal_gamma": 2.0,
                        "raw_hardness_scale": 1.0,
                    },
                }),
            },
            "balanced": {
                "optimization_id": "b07_q020_batch256",
                "optimization_recipe_json": json.dumps({
                    "id": "b07_q020_batch256",
                    "training": {
                        "minibatch_positive_fraction": 0.2,
                        "batch_size": 256,
                    },
                }),
            },
            "overlap": {
                "optimization_id": "o07_tversky3070_w030",
                "optimization_recipe_json": json.dumps({
                    "id": "o07_tversky3070_w030",
                    "training": {
                        "overlap_loss_weight": 0.3,
                        "overlap_false_positive_weight": 0.3,
                        "overlap_false_negative_weight": 0.7,
                    },
                }),
            },
        }
        merged = _merge_combination_recipe(
            "x03_core_balanced_overlap",
            ["core", "balanced", "overlap"],
            source_rows,
        )
        self.assertEqual(merged["source_optimization_ids"], {
            "core": "m15_all",
            "balanced": "b07_q020_batch256",
            "overlap": "o07_tversky3070_w030",
        })
        self.assertEqual(merged["training"], {
            "focal_gamma": 2.0,
            "raw_hardness_scale": 1.0,
            "minibatch_positive_fraction": 0.2,
            "batch_size": 256,
            "overlap_loss_weight": 0.3,
            "overlap_false_positive_weight": 0.3,
            "overlap_false_negative_weight": 0.7,
        })

    def test_conflicting_source_value_is_rejected(self):
        source_rows = {
            "core": {
                "optimization_id": "m00",
                "optimization_recipe_json": json.dumps({
                    "id": "m00", "training": {"batch_size": 128}
                }),
            },
            "balanced": {
                "optimization_id": "b00",
                "optimization_recipe_json": json.dumps({
                    "id": "b00", "training": {"batch_size": 256}
                }),
            },
        }
        with self.assertRaisesRegex(ValueError, "conflicting training.batch_size"):
            _merge_combination_recipe(
                "x01_core_balanced", ["core", "balanced"], source_rows
            )


if __name__ == "__main__":
    unittest.main()
