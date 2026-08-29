import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.Task12Data_3D import _anchored_recipe, feature_matrix
from Search_Task3_FMTResidual_3D import (
    _apply_absolute_fmt_guard,
    _decode_job,
    _load_spec,
    _selection_key,
)


CONFIG = Path("config/Verify_Task3_AnchoredFeatureDecomposition_22.1.yaml")
CONTROL = Path("config/Verify_Task3_SpatialRobust_5.2.yaml")


class Task3AnchoredFeatureDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = _load_spec(CONFIG)
        cls.control = _load_spec(CONTROL)

    def test_registered_scale_and_job_mapping(self):
        self.assertEqual(len(self.spec["datasets"]), 10)
        self.assertEqual(len(self.spec["candidates"]), 16)
        self.assertEqual(self.spec["screen_seeds"], [40, 41])
        self.assertEqual(10 * 16, 160)
        self.assertEqual(160 * 2 * 2, 640)
        self.assertEqual(_decode_job(self.spec, 159), ("smokeBuoyancy", 15))
        with self.assertRaises(IndexError):
            _decode_job(self.spec, 160)

    def test_population_training_and_split_are_exact_5_2_controls(self):
        for key in (
            "datasets", "groups", "exposed_training", "robust_validation",
            "screen_split", "training", "raw_wide_parameter_count",
            "raw_pca_random_state",
        ):
            self.assertEqual(self.spec[key], self.control[key], key)
        self.assertEqual(self.spec.get("outer_ordinals"), [])
        self.assertEqual(self.spec["screen_split"]["test_ordinals"], [])

    def test_exact_family_controls_and_absolute_guard_are_registered(self):
        controls = {
            row["fmt_feature"] for row in self.spec["candidates"][:5]
        }
        self.assertEqual(controls, {
            "aivd1w3", "aivd1w3d2", "aivd2w4", "aivd2w8d2",
            "fmt_all+aivd2w8",
        })
        guard = self.spec["selection"]["absolute_fmt_guard"]
        self.assertEqual(set(guard["by_group"]), set(self.spec["groups"]))
        self.assertAlmostEqual(float(guard["tolerance"]), 0.002)
        self.assertEqual(
            guard["source_selection_sha256"],
            "cc24b79ace6420b61cb0a1edfa17ea9cabe704b397c565dd00bfd98bf4d68422",
        )
        self.assertTrue(Path(guard["source_selection"]).is_file())
        candidate_features = {
            row["fmt_feature"] for row in self.spec["candidates"]
        }
        for control in guard["by_group"].values():
            self.assertIn(control["feature"], candidate_features)
        self.assertAlmostEqual(
            float(self.spec["selection"]["target_dataset_macro_f1_gain"]),
            0.15,
        )
        self.assertAlmostEqual(
            float(self.spec["selection"]["target_absolute_fmt_f1"]),
            0.89,
        )

    def test_anchor_recipes_have_declared_meaning_and_width(self):
        expected = {
            "aivd1w3_first": (False, ("first",), 1),
            "aivd1w3_early": (False, ("first", "early_mean"), 2),
            "aivd1w3_dft": (True, (), 1),
            "aivd1w3_core": (True, ("first", "early_mean"), 3),
            "aivd1w3_stats": (
                True, ("first", "early_mean", "mean", "std"), 5,
            ),
            "aivd2w8_dft": (True, (), 3),
            "aivd2w8_core": (True, ("first", "early_mean"), 5),
            "aivd2w8_stats": (
                True, ("first", "early_mean", "mean", "std"), 7,
            ),
        }
        generator = torch.Generator().manual_seed(7068)
        velocity = torch.randn(4, 7, 31, 3, generator=generator)
        primitives = torch.cat((torch.zeros(4, 7, 1, 3), velocity), dim=2)
        primitives = primitives.cumsum(dim=2).numpy().astype(np.float32)
        record = {
            "raw": primitives.reshape(4, -1),
            "fmt": np.zeros((4, 161), dtype=np.float32),
            "features": {},
        }
        for name, (include_dft, anchors, width) in expected.items():
            recipe = _anchored_recipe(name)
            self.assertEqual(recipe["include_dft"], include_dft, name)
            self.assertEqual(recipe["anchor_names"], anchors, name)
            values = feature_matrix(record, name, "cpu")
            self.assertEqual(values.shape, (4, width), name)
            self.assertTrue(np.isfinite(values).all(), name)

    def test_absolute_guard_precedes_gain_but_legacy_ranking_is_unchanged(self):
        failing_gain_only = {
            "fmt_minus_raw_pca_f1_macro": 0.40,
            "fmt_minus_raw_pca_ap_macro": 0.40,
            "worst_seed_f1_gain": 0.30,
            "fmt_f1_macro": 0.80,
            "fmt_ap_macro": 0.80,
        }
        preserving = {
            "fmt_minus_raw_pca_f1_macro": 0.15,
            "fmt_minus_raw_pca_ap_macro": 0.16,
            "worst_seed_f1_gain": 0.10,
            "fmt_f1_macro": 0.82,
            "fmt_ap_macro": 0.89,
        }
        guarded_failing = _apply_absolute_fmt_guard(
            self.spec, "channel", failing_gain_only
        )
        guarded_preserving = _apply_absolute_fmt_guard(
            self.spec, "channel", preserving
        )
        self.assertFalse(guarded_failing["absolute_fmt_guard_passed"])
        self.assertTrue(guarded_preserving["absolute_fmt_guard_passed"])
        self.assertGreater(
            _selection_key(guarded_preserving),
            _selection_key(guarded_failing),
        )
        self.assertGreater(
            _selection_key(failing_gain_only),
            _selection_key(preserving),
        )
        no_guard = copy.deepcopy(self.spec)
        no_guard["selection"].pop("absolute_fmt_guard")
        self.assertEqual(
            _apply_absolute_fmt_guard(no_guard, "channel", preserving),
            preserving,
        )


if __name__ == "__main__":
    unittest.main()
