import unittest

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _optimization_candidate,
)


CONFIG = "config/Verify_Task3_AuxiliaryBottleneck_17.1.yaml"


class Task3AuxiliaryBottleneckTests(unittest.TestCase):
    def test_grid_and_array_bounds_are_frozen(self):
        spec = _load_optimization_spec(CONFIG)
        widths = [
            row["model"]["auxiliary_dim"]
            for row in spec["optimization_candidates"]
        ]
        self.assertEqual(widths, [4, 8, 16, 24, 32, 48, 64, 96, 128])
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        with self.assertRaises(IndexError):
            _decode_job(spec, 90)

    def test_candidate_model_override_changes_only_declared_width(self):
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
        first = _optimization_candidate(spec, manifest, "channel", 0)
        control = _optimization_candidate(spec, manifest, "channel", 6)
        self.assertEqual(first["auxiliary_dim"], 4)
        self.assertEqual(control["auxiliary_dim"], 64)
        self.assertEqual(first["training"], base["training"])
        self.assertEqual(first["residual_input"], base["residual_input"])

    def test_parameter_count_increases_monotonically_with_bottleneck_width(self):
        counts = []
        for width in (4, 8, 16, 32, 64, 128):
            raw = PathlineBinaryClassifier3D(variant="raw")
            model = PathlineFMTResidualClassifier3D(
                raw,
                fmt_dim=64,
                **residual_model_kwargs({
                    "auxiliary_dim": width,
                    "head_architecture": "deep_mlp",
                    "head_hidden_dim": 64,
                    "head_depth": 2,
                    "head_dropout": 0.0,
                }),
            )
            counts.append(sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ))
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(counts), len(set(counts)))


if __name__ == "__main__":
    unittest.main()
