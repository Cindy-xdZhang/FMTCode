import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _optimization_candidate,
)


CONFIG = "config/Verify_Task3_UltraNarrowBottleneck_20.1.yaml"


class Task3UltraNarrowBottleneckTests(unittest.TestCase):
    def test_grid_refines_below_17_2_and_retains_controls(self):
        spec = _load_optimization_spec(CONFIG)
        widths = [
            int(row["model"]["auxiliary_dim"])
            for row in spec["optimization_candidates"]
        ]
        self.assertEqual(widths, [1, 2, 3, 4, 6, 8, 12, 96])
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(
            spec["model_override"],
            {
                "head_architecture": "deep_mlp",
                "head_hidden_dim": 64,
                "head_depth": 2,
                "head_dropout": 0.0,
            },
        )
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        with self.assertRaises(IndexError):
            _decode_job(spec, 80)

    def test_width_override_preserves_frozen_upstream_training(self):
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
        candidate = _optimization_candidate(spec, manifest, "channel", 0)
        self.assertEqual(candidate["auxiliary_dim"], 1)
        self.assertEqual(candidate["training"], base["training"])
        self.assertEqual(candidate["residual_input"], "geometry_fmt")


if __name__ == "__main__":
    unittest.main()
