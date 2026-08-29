"""Contracts for Task3 semantic-block auxiliary projections."""

import unittest

import torch

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
    trainable_parameter_count,
)
from FMT_Utils.Task12Data_3D import feature_block_dims
from Search_Task3_LossOptimization_7_1 import _optimization_candidate
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
)


CONFIG = "config/Verify_Task3_SemanticBlockProjection_25.1.yaml"


class SemanticBlockProjectionTests(unittest.TestCase):
    def test_preregistered_grid_is_complete_and_confirmation_closed(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 8)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        self.assertEqual(
            spec["optimization_selection"]["target_absolute_fmt_f1"], 0.89
        )
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])

    def test_feature_blocks_cover_existing_and_decomposed_candidates(self):
        expected = {
            "aivd1w3": (1, 1, 1, 1, 1, 1, 1, 1),
            "aivd1w3_core": (1, 1, 1),
            "aivd2w8": (3, 1, 1, 1, 1, 1, 1, 1),
            "aivd2w8_dft": (3,),
            "fmt_all": (23, 23, 23, 23, 23, 23, 23),
            "fmt_all+aivd2w8_core": (
                23, 23, 23, 23, 23, 23, 23, 3, 1, 1,
            ),
        }
        for name, dims in expected.items():
            self.assertEqual(feature_block_dims(name), dims, name)

    def test_all_blockwise_projections_preserve_requested_width(self):
        pathlines = torch.randn(5, 7, 32, 3)
        features = torch.randn(5, 10)
        for architecture in (
            "blockwise_linear_gelu",
            "blockwise_layernorm_gelu",
            "blockwise_rmsnorm_gelu",
            "blockwise_mlp_gelu",
        ):
            torch.manual_seed(9)
            raw = PathlineBinaryClassifier3D(variant="raw")
            model = PathlineFMTResidualClassifier3D(
                raw,
                fmt_dim=10,
                auxiliary_dim=16,
                head_architecture="deep_mlp",
                head_hidden_dim=32,
                auxiliary_projection=architecture,
                auxiliary_hidden_dim=12,
                auxiliary_block_dims=(3, 2, 1, 4),
            )
            raw_logit, residual_logit, auxiliary = model.forward_components(
                pathlines, features, return_auxiliary=True
            )
            self.assertEqual(tuple(raw_logit.shape), (5,))
            self.assertEqual(tuple(residual_logit.shape), (5,))
            self.assertEqual(tuple(auxiliary.shape), (5, 16))
            self.assertTrue(torch.isfinite(auxiliary).all())

    def test_paired_arms_have_identical_trainable_parameter_count(self):
        kwargs = residual_model_kwargs({
            "embedding_dim": 128,
            "auxiliary_dim": 64,
            "head_architecture": "deep_mlp",
            "head_hidden_dim": 64,
            "head_depth": 2,
            "auxiliary_projection": "blockwise_mlp_gelu",
            "auxiliary_hidden_dim": 16,
            "auxiliary_block_dims": [3, 1, 1, 1, 1, 1, 1, 1],
        })
        torch.manual_seed(1)
        fmt_model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"), fmt_dim=10, **kwargs
        )
        torch.manual_seed(2)
        raw_pca_model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"), fmt_dim=10, **kwargs
        )
        self.assertEqual(
            trainable_parameter_count(fmt_model),
            trainable_parameter_count(raw_pca_model),
        )

    def test_invalid_block_contracts_fail_loudly(self):
        raw = PathlineBinaryClassifier3D(variant="raw")
        with self.assertRaisesRegex(ValueError, "requires model.auxiliary_block_dims"):
            PathlineFMTResidualClassifier3D(
                raw, fmt_dim=5,
                auxiliary_projection="blockwise_linear_gelu",
            )
        raw = PathlineBinaryClassifier3D(variant="raw")
        with self.assertRaisesRegex(ValueError, "sum to"):
            PathlineFMTResidualClassifier3D(
                raw, fmt_dim=5,
                auxiliary_projection="blockwise_linear_gelu",
                auxiliary_block_dims=(2, 2),
            )
        raw = PathlineBinaryClassifier3D(variant="raw")
        with self.assertRaisesRegex(ValueError, "output width >= block count"):
            PathlineFMTResidualClassifier3D(
                raw, fmt_dim=5, auxiliary_dim=1,
                auxiliary_projection="blockwise_linear_gelu",
                auxiliary_block_dims=(2, 3),
            )

    def test_optimization_recipe_freezes_inferred_block_contract(self):
        spec = {
            "groups": {"family": {"datasets": ["flow"]}},
            "optimization_candidates": [{"id": "blocks"}],
        }
        manifest = {
            "base_candidate_by_group": {"family": {
                "fmt_feature": "aivd2w8_core",
                "auxiliary_projection": "blockwise_mlp_gelu",
                "training": {},
            }},
        }
        candidate = _optimization_candidate(spec, manifest, "flow", 0)
        self.assertEqual(candidate["auxiliary_block_dims"], [3, 1, 1])
        self.assertEqual(
            candidate["optimization_recipe"]["model"][
                "auxiliary_block_dims"
            ],
            [3, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
