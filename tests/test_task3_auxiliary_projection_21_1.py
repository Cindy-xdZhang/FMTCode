import json
import unittest

import torch

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_candidate_overrides,
    _merge_combination_recipe,
)


CONFIG = "config/Verify_Task3_AuxiliaryProjection_21.1.yaml"


class Task3AuxiliaryProjectionTests(unittest.TestCase):
    def test_config_freezes_ultranarrow_source_and_eight_projections(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(set(spec["combination_sources"]), {"ultranarrow"})
        projections = [
            row["model"]["auxiliary_projection"]
            for row in spec["optimization_candidates"]
        ]
        self.assertEqual(projections, [
            "linear_layernorm_gelu",
            "linear",
            "linear_gelu",
            "linear_silu",
            "linear_rmsnorm_gelu",
            "mlp_layernorm_gelu",
            "mlp_layernorm_silu",
            "mlp_rmsnorm_gelu",
        ])
        self.assertTrue(all(
            row["sources"] == ["ultranarrow"]
            for row in spec["optimization_candidates"]
        ))
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))

    def test_local_projection_merges_without_overwriting_frozen_width(self):
        source_rows = {"ultranarrow": {
            "optimization_id": "u01_aux2",
            "optimization_recipe_json": json.dumps({
                "id": "u01_aux2", "model": {"auxiliary_dim": 2}
            }),
        }}
        merged = _merge_combination_recipe(
            "p01_linear", ["ultranarrow"], source_rows
        )
        merged = _merge_candidate_overrides(merged, {
            "id": "p01_linear",
            "sources": ["ultranarrow"],
            "model": {
                "auxiliary_projection": "linear",
                "auxiliary_hidden_dim": 64,
            },
        })
        self.assertEqual(merged["model"], {
            "auxiliary_dim": 2,
            "auxiliary_projection": "linear",
            "auxiliary_hidden_dim": 64,
        })
        with self.assertRaisesRegex(ValueError, "frozen model.auxiliary_dim"):
            _merge_candidate_overrides(merged, {
                "id": "bad", "sources": ["ultranarrow"],
                "model": {"auxiliary_dim": 4},
            })

    def test_width_one_control_collapses_but_alternatives_remain_informative(self):
        torch.manual_seed(7068)
        features = torch.randn(9, 13)
        control = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D("raw", fmt_dim=13),
            fmt_dim=13, auxiliary_dim=1,
            auxiliary_projection="linear_layernorm_gelu",
        )
        self.assertTrue(torch.equal(
            control.fmt_encoder(features), torch.zeros(9, 1)
        ))
        for projection in (
            "linear", "linear_gelu", "linear_silu",
            "linear_rmsnorm_gelu", "mlp_layernorm_gelu",
            "mlp_layernorm_silu", "mlp_rmsnorm_gelu",
        ):
            model = PathlineFMTResidualClassifier3D(
                PathlineBinaryClassifier3D("raw", fmt_dim=13),
                fmt_dim=13, auxiliary_dim=1,
                auxiliary_projection=projection,
            )
            encoded = model.fmt_encoder(features)
            self.assertEqual(tuple(encoded.shape), (9, 1))
            self.assertTrue(torch.isfinite(encoded).all())
            self.assertGreater(float(encoded.std()), 0.0)

    def test_default_checkpoint_keys_remain_historical(self):
        implicit = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D("raw", fmt_dim=17), fmt_dim=17
        )
        explicit = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D("raw", fmt_dim=17), fmt_dim=17,
            **residual_model_kwargs({
                "auxiliary_projection": "linear_layernorm_gelu",
                "auxiliary_hidden_dim": 64,
            }),
        )
        self.assertEqual(
            list(implicit.fmt_encoder.state_dict()),
            list(explicit.fmt_encoder.state_dict()),
        )


if __name__ == "__main__":
    unittest.main()
