import json
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_combination_recipe,
)


CONFIG = "config/Verify_Task3_BalancedCombination_14.1.yaml"


class BalancedCombinationContractTests(unittest.TestCase):
    def test_config_and_array_bounds(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(set(spec["combination_sources"]), {"core", "balanced"})
        self.assertEqual(len(spec["optimization_candidates"]), 2)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(_decode_job(spec, 19), ("smokeBuoyancy", 1))

    def test_nested_selection_recipe_is_merged_without_hidden_overwrite(self):
        source_rows = {
            "core": {
                "optimization_id": "m15_all",
                "optimization_recipe_json": json.dumps({
                    "id": "m15_all",
                    "sources": [
                        "loss", "hardness", "correction", "ranking"
                    ],
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
        }
        merged = _merge_combination_recipe(
            "z01_core_balanced", ["core", "balanced"], source_rows
        )
        self.assertEqual(merged["source_optimization_ids"], {
            "core": "m15_all",
            "balanced": "b07_q020_batch256",
        })
        self.assertEqual(merged["training"], {
            "focal_gamma": 2.0,
            "raw_hardness_scale": 1.0,
            "minibatch_positive_fraction": 0.2,
            "batch_size": 256,
        })

    def test_nested_merge_rejects_training_conflict(self):
        source_rows = {
            "core": {
                "optimization_id": "m01",
                "optimization_recipe_json": json.dumps({
                    "id": "m01",
                    "sources": ["loss"],
                    "source_optimization_ids": {"loss": "o01"},
                    "training": {"batch_size": 128},
                }),
            },
            "balanced": {
                "optimization_id": "b09",
                "optimization_recipe_json": json.dumps({
                    "id": "b09",
                    "training": {
                        "batch_size": 256,
                        "minibatch_positive_fraction": 0.5,
                    },
                }),
            },
        }
        with self.assertRaisesRegex(ValueError, "conflicting training.batch_size"):
            _merge_combination_recipe(
                "z01_core_balanced", ["core", "balanced"], source_rows
            )


if __name__ == "__main__":
    unittest.main()
