import json
import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_combination_recipe,
)


CONFIG = "config/Verify_Task3_RepresentationCombination_19.1.yaml"


class RepresentationCombinationContractTests(unittest.TestCase):
    def test_config_declares_complete_three_factor_factorial(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            set(spec["combination_sources"]),
            {"core", "bottleneck", "contrastive"},
        )
        self.assertEqual(
            [row["sources"] for row in spec["optimization_candidates"]],
            [
                [],
                ["core"],
                ["bottleneck"],
                ["contrastive"],
                ["core", "bottleneck"],
                ["core", "contrastive"],
                ["bottleneck", "contrastive"],
                ["core", "bottleneck", "contrastive"],
            ],
        )
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        with self.assertRaises(IndexError):
            _decode_job(spec, 80)

    def test_three_sources_merge_training_and_model_without_overwrite(self):
        source_rows = {
            "core": {
                "optimization_id": "m11_loss_hardness_correction",
                "optimization_recipe_json": json.dumps({
                    "id": "m11_loss_hardness_correction",
                    "training": {
                        "loss": "focal",
                        "focal_gamma": 3.0,
                        "raw_error_boost": 4.0,
                    },
                }),
            },
            "bottleneck": {
                "optimization_id": "d00_aux4",
                "optimization_recipe_json": json.dumps({
                    "id": "d00_aux4",
                    "model": {"auxiliary_dim": 4},
                }),
            },
            "contrastive": {
                "optimization_id": "s03_w0010_t010",
                "optimization_recipe_json": json.dumps({
                    "id": "s03_w0010_t010",
                    "training": {
                        "supervised_contrastive_loss_weight": 0.01,
                    },
                }),
            },
        }
        merged = _merge_combination_recipe(
            "r07_all",
            ["core", "bottleneck", "contrastive"],
            source_rows,
        )
        self.assertEqual(merged["source_optimization_ids"], {
            "core": "m11_loss_hardness_correction",
            "bottleneck": "d00_aux4",
            "contrastive": "s03_w0010_t010",
        })
        self.assertEqual(merged["model"], {"auxiliary_dim": 4})
        self.assertEqual(merged["training"], {
            "loss": "focal",
            "focal_gamma": 3.0,
            "raw_error_boost": 4.0,
            "supervised_contrastive_loss_weight": 0.01,
        })

    def test_conflicting_source_value_is_rejected(self):
        source_rows = {
            "core": {
                "optimization_id": "m00",
                "optimization_recipe_json": json.dumps({
                    "id": "m00", "model": {"auxiliary_dim": 64}
                }),
            },
            "bottleneck": {
                "optimization_id": "d00",
                "optimization_recipe_json": json.dumps({
                    "id": "d00", "model": {"auxiliary_dim": 4}
                }),
            },
        }
        with self.assertRaisesRegex(ValueError, "conflicting model.auxiliary_dim"):
            _merge_combination_recipe(
                "r04_core_bottleneck", ["core", "bottleneck"], source_rows
            )


if __name__ == "__main__":
    unittest.main()
