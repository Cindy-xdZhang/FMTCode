"""Contracts for residual-head learning-rate refresh 101.1."""

import json
from pathlib import Path
import tempfile
import unittest

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
    _build_optimizer,
    _optimizer_parameter_spec,
    _residual_head_learning_rate_multiplier,
)


CONFIG = "config/Verify_Task3_ResidualHeadLearningRateRefresh_101.1.yaml"
EXPECTED = [1.0, 0.05, 0.10, 0.25, 0.50, 2.0, 4.0, 8.0, 16.0]


def _model():
    return PathlineFMTResidualClassifier3D(
        PathlineBinaryClassifier3D(variant="raw"),
        fmt_dim=8,
        head_architecture="deep_mlp",
        head_hidden_dim=16,
    )


class ResidualHeadLearningRateRefreshTests(unittest.TestCase):
    def test_registered_grid_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        self.assertEqual(
            spec["allowed_source_overrides"],
            ["training.residual_head_learning_rate_multiplier"],
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.240)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.903)
        self.assertFalse(selection["confirmation_opened"])

    def test_exact_source_control_and_log_spaced_grid(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["portfolio"])
            if index == 0:
                self.assertEqual(row, {
                    "id": "r00_control_source",
                    "sources": ["portfolio"],
                })
                observed.append(1.0)
            else:
                self.assertEqual(set(row), {"id", "sources", "training"})
                self.assertEqual(
                    set(row["training"]),
                    {"residual_head_learning_rate_multiplier"},
                )
                observed.append(float(
                    row["training"]["residual_head_learning_rate_multiplier"]
                ))
        self.assertEqual(observed, EXPECTED)

    def test_source_recipe_is_preserved_except_registered_multiplier(self):
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
                        "model": {
                            "head_hidden_dim": 80,
                            "head_depth": 3,
                            "head_normalization": "rmsnorm",
                            "head_activation": "relu",
                        },
                        "training": {
                            "learning_rate": 0.0005,
                            "auxiliary_learning_rate_multiplier": 4.0,
                            "max_epochs": 160,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_ResidualHeadNormActivationRefreshPortfolio_100.1"
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
                "residual_head_learning_rate_multiplier", rows[0]["training"]
            )
            self.assertEqual(
                rows[1]["training"]["residual_head_learning_rate_multiplier"],
                0.05,
            )
            self.assertEqual(
                rows[1]["training"]["auxiliary_learning_rate_multiplier"], 4.0
            )
            self.assertEqual(rows[1]["training"]["learning_rate"], 0.0005)
            self.assertEqual(rows[1]["training"]["max_epochs"], 160)
            self.assertEqual(rows[1]["model"]["head_hidden_dim"], 80)
            self.assertEqual(rows[1]["model"]["head_normalization"], "rmsnorm")

    def test_default_is_exact_flat_optimizer_control(self):
        model = _model()
        training = {"learning_rate": 0.001, "weight_decay": 0.0001}
        parameters, auxiliary_multiplier, auxiliary_group = (
            _optimizer_parameter_spec(model, training)
        )
        optimizer, _, _ = _build_optimizer(training, parameters)
        self.assertEqual(_residual_head_learning_rate_multiplier(training), 1.0)
        self.assertEqual(auxiliary_multiplier, 1.0)
        self.assertEqual(auxiliary_group, 0)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)

    def test_head_multiplier_changes_only_downstream_group_rate(self):
        model = _model()
        training = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "residual_head_learning_rate_multiplier": 0.25,
        }
        parameters, auxiliary_multiplier, auxiliary_group = (
            _optimizer_parameter_spec(model, training)
        )
        optimizer, _, _ = _build_optimizer(training, parameters)
        self.assertEqual(auxiliary_multiplier, 1.0)
        self.assertEqual(auxiliary_group, 1)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.00025)
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.001)
        auxiliary_ids = {
            id(parameter) for parameter in model.fmt_encoder.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            {id(parameter) for parameter in optimizer.param_groups[1]["params"]},
            auxiliary_ids,
        )
        downstream_ids = {
            id(parameter) for parameter in optimizer.param_groups[0]["params"]
        }
        self.assertFalse(downstream_ids.intersection(auxiliary_ids))
        self.assertEqual(
            len(downstream_ids) + len(auxiliary_ids),
            sum(1 for parameter in model.parameters() if parameter.requires_grad),
        )

    def test_head_and_auxiliary_multipliers_coexist_without_overlap(self):
        model = _model()
        training = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "residual_head_learning_rate_multiplier": 0.25,
            "auxiliary_learning_rate_multiplier": 4.0,
        }
        parameters, auxiliary_multiplier, auxiliary_group = (
            _optimizer_parameter_spec(model, training)
        )
        optimizer, _, _ = _build_optimizer(training, parameters)
        self.assertEqual(auxiliary_multiplier, 4.0)
        self.assertEqual(auxiliary_group, 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.00025)
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.004)
        self.assertFalse(
            {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
            .intersection(
                id(parameter)
                for parameter in optimizer.param_groups[1]["params"]
            )
        )

    def test_multiplier_validation_and_evidence_contract(self):
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(
                ValueError, "residual_head_learning_rate_multiplier"
            ):
                _residual_head_learning_rate_multiplier({
                    "residual_head_learning_rate_multiplier": invalid
                })
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "r00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_residual_head_learning_rate_refresh_101.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_residual_head_learning_rate_refresh_101.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-89%24", gpu)
        self.assertIn("expected 540 per-run CSV files", evidence)
        self.assertIn("expected 540 temporary checkpoints", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
