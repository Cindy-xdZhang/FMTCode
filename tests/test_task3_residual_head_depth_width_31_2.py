"""Contracts for capacity-safe Task3 head depth x width search 31.2."""

import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ResidualHeadDepthWidth_31.2.yaml"


class ResidualHeadDepthWidthRevisionTests(unittest.TestCase):
    def test_capacity_safe_factorial_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 12)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 119), ("smokeBuoyancy", 11))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_only_width96_level_was_removed_from_31_1(self):
        failed = _load_optimization_spec(
            "config/Verify_Task3_ResidualHeadDepthWidth_31.1.yaml"
        )
        revised = _load_optimization_spec(CONFIG)
        failed_cells = {
            (row["model"]["head_hidden_dim"], row["model"]["head_depth"])
            for row in failed["optimization_candidates"]
        }
        revised_cells = {
            (row["model"]["head_hidden_dim"], row["model"]["head_depth"])
            for row in revised["optimization_candidates"]
        }
        self.assertEqual(
            failed_cells - revised_cells,
            {(96, 1), (96, 2), (96, 3)},
        )
        self.assertEqual(revised_cells, {
            (width, depth)
            for width in (32, 48, 64, 80)
            for depth in (1, 2, 3)
        })

    def test_control_and_absolute_guard_are_unchanged(self):
        failed = _load_optimization_spec(
            "config/Verify_Task3_ResidualHeadDepthWidth_31.1.yaml"
        )
        revised = _load_optimization_spec(CONFIG)
        self.assertEqual(
            failed["optimization_candidates"][0],
            revised["optimization_candidates"][0],
        )
        self.assertEqual(
            failed["optimization_selection"],
            revised["optimization_selection"],
        )

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 12)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(
                    set(recipe["model"]),
                    {"head_hidden_dim", "head_depth"},
                )


if __name__ == "__main__":
    unittest.main()
