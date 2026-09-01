"""Contracts for residual-head gradient clipping refresh 111.1."""

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import torch

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import (
    _clip_residual_head_gradients,
    _residual_head_gradient_clip_norm,
    _train_one,
)


CONFIG = "config/Verify_Task3_ResidualHeadGradientClipRefresh_111.1.yaml"
EXPECTED = [None, 0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0, 30.0]


class ResidualHeadGradientClipRefreshTests(unittest.TestCase):
    def test_registered_grid_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        self.assertEqual(
            spec["allowed_source_overrides"],
            ["training.residual_head_gradient_clip_norm"],
        )
        source = spec["combination_sources"]["portfolio"]
        self.assertEqual(
            source["expected_experiment"],
            "Verify_Task3_ResidualHeadEpsilonRefreshPortfolio_110.1",
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.250)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.908)
        self.assertFalse(selection["confirmation_opened"])

    def test_exact_control_and_registered_grid(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["portfolio"])
            if index == 0:
                self.assertEqual(row, {
                    "id": "h00_control_source",
                    "sources": ["portfolio"],
                })
                observed.append(None)
            else:
                self.assertEqual(set(row), {"id", "sources", "training"})
                self.assertEqual(
                    set(row["training"]),
                    {"residual_head_gradient_clip_norm"},
                )
                observed.append(float(
                    row["training"]["residual_head_gradient_clip_norm"]
                ))
        self.assertEqual(observed, EXPECTED)

    def test_source_recipe_preserves_global_and_auxiliary_caps(self):
        spec = _load_optimization_spec(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "portfolio_selection.json"
            primary = {}
            for group in spec["groups"]:
                primary[group] = {
                    "optimization_id": f"source_{group}",
                    "optimization_recipe_json": json.dumps({
                        "id": f"source_{group}",
                        "fmt_feature": "fmt_all",
                        "model": {"head_hidden_dim": 80},
                        "training": {
                            "gradient_clip_norm": 2.0,
                            "auxiliary_gradient_clip_norm": 0.3,
                            "residual_head_optimizer_epsilon": 1e-6,
                            "max_epochs": 160,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_ResidualHeadEpsilonRefreshPortfolio_110.1"
                ),
                "confirmation_opened": False,
                "primary_by_group": primary,
            }), encoding="utf-8")
            spec["combination_sources"]["portfolio"]["selection"] = str(
                source_path
            )
            resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(hashes), {"portfolio"})
        for rows in resolved.values():
            self.assertNotIn(
                "residual_head_gradient_clip_norm", rows[0]["training"]
            )
            self.assertEqual(
                rows[1]["training"]["residual_head_gradient_clip_norm"],
                0.01,
            )
            self.assertEqual(rows[1]["training"]["gradient_clip_norm"], 2.0)
            self.assertEqual(
                rows[1]["training"]["auxiliary_gradient_clip_norm"], 0.3
            )
            self.assertEqual(
                rows[1]["training"]["residual_head_optimizer_epsilon"],
                1e-6,
            )
            self.assertEqual(rows[1]["model"]["head_hidden_dim"], 80)

    @staticmethod
    def _model():
        return PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=8,
            auxiliary_dim=4,
            head_architecture="deep_mlp",
            head_hidden_dim=16,
        )

    def test_validation_and_disabled_noop(self):
        self.assertIsNone(_residual_head_gradient_clip_norm({}))
        self.assertIsNone(_residual_head_gradient_clip_norm({
            "residual_head_gradient_clip_norm": None
        }))
        self.assertEqual(_residual_head_gradient_clip_norm({
            "residual_head_gradient_clip_norm": 0.3
        }), 0.3)
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(
                ValueError, "residual_head_gradient_clip_norm"
            ):
                _residual_head_gradient_clip_norm({
                    "residual_head_gradient_clip_norm": value
                })
        self.assertIsNone(_clip_residual_head_gradients(self._model(), None))

    def test_clipping_touches_only_downstream_gradients(self):
        model = self._model()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 3.0)
        auxiliary_before = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("fmt_encoder.")
        }
        original_norm = _clip_residual_head_gradients(model, 0.25)
        self.assertGreater(original_norm, 0.25)
        downstream_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.norm()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("fmt_encoder.")
        ]))
        self.assertLessEqual(float(downstream_norm), 0.25001)
        for name, expected in auxiliary_before.items():
            parameter = dict(model.named_parameters())[name]
            self.assertTrue(torch.equal(parameter.grad, expected), name)

    def test_evidence_fields_guard_and_scripts(self):
        source = inspect.getsource(_train_one)
        self.assertIn("residual_head_gradient_clip_norm", source)
        self.assertIn("maximum_preclip_residual_head_gradient_norm", source)
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "h00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_residual_head_gradient_clip_refresh_111.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_residual_head_gradient_clip_refresh_111.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-89%24", gpu)
        self.assertIn("expected 540 per-run CSV files", evidence)
        self.assertIn("expected 540 temporary checkpoints", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
