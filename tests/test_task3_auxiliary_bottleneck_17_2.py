import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _optimization_candidate,
)


CONFIG = "config/Verify_Task3_AuxiliaryBottleneck_17.2.yaml"


class Task3AuxiliaryBottleneckRevisionTests(unittest.TestCase):
    def test_revision_removes_only_ineligible_aux128(self):
        spec = _load_optimization_spec(CONFIG)
        widths = [
            row["model"]["auxiliary_dim"]
            for row in spec["optimization_candidates"]
        ]
        self.assertEqual(widths, [4, 8, 16, 24, 32, 48, 64, 96])
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        with self.assertRaises(IndexError):
            _decode_job(spec, 80)

    def test_width_override_preserves_upstream_training_recipe(self):
        spec = _load_optimization_spec(CONFIG)
        base = {
            "id": "upstream",
            "fmt_feature": "kin2",
            "auxiliary_dim": 64,
            "residual_input": "geometry_fmt",
            "training": {"learning_rate": 0.001},
        }
        manifest = {"base_candidate_by_group": {
            name: dict(base) for name in spec["groups"]
        }}
        narrow = _optimization_candidate(spec, manifest, "channel", 0)
        control = _optimization_candidate(spec, manifest, "channel", 6)
        self.assertEqual(narrow["auxiliary_dim"], 4)
        self.assertEqual(control["auxiliary_dim"], 64)
        self.assertEqual(narrow["training"], base["training"])
        self.assertEqual(narrow["residual_input"], "geometry_fmt")


if __name__ == "__main__":
    unittest.main()
