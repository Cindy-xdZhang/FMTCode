"""Contracts for residual-head Adam epsilon refresh 109.1."""

import inspect
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
    _residual_head_optimizer_epsilon,
    _train_one,
)


CONFIG = "config/Verify_Task3_ResidualHeadEpsilonRefresh_109.1.yaml"
EXPECTED = [None, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _model():
    return PathlineFMTResidualClassifier3D(
        PathlineBinaryClassifier3D(variant="raw"),
        fmt_dim=8,
        head_architecture="deep_mlp",
        head_hidden_dim=16,
    )


class ResidualHeadEpsilonRefreshTests(unittest.TestCase):
    def test_registered_grid_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 11)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 109), ("smokeBuoyancy", 10))
        self.assertEqual(
            spec["allowed_source_overrides"],
            ["training.residual_head_optimizer_epsilon"],
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.248)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.907)
        self.assertFalse(selection["confirmation_opened"])

    def test_exact_source_control_and_registered_grid(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["portfolio"])
            if index == 0:
                self.assertEqual(row, {
                    "id": "e00_control_source",
                    "sources": ["portfolio"],
                })
                observed.append(None)
            else:
                self.assertEqual(set(row), {"id", "sources", "training"})
                self.assertEqual(
                    set(row["training"]),
                    {"residual_head_optimizer_epsilon"},
                )
                observed.append(float(
                    row["training"]["residual_head_optimizer_epsilon"]
                ))
        self.assertEqual(observed, EXPECTED)

    def test_source_recipe_is_preserved_except_registered_epsilon(self):
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
                        "model": {"head_hidden_dim": 80, "head_depth": 3},
                        "training": {
                            "learning_rate": 0.0005,
                            "weight_decay": 0.0002,
                            "optimizer_epsilon": 1e-4,
                            "residual_head_optimizer_beta1": 0.5,
                            "residual_head_optimizer_beta2": 0.8,
                            "auxiliary_optimizer_epsilon": 1e-6,
                            "max_epochs": 160,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_ResidualHeadBeta1RefreshPortfolio_108.1"
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
                "residual_head_optimizer_epsilon", rows[0]["training"]
            )
            self.assertEqual(
                rows[1]["training"]["residual_head_optimizer_epsilon"], 1e-12
            )
            self.assertEqual(rows[1]["training"]["optimizer_epsilon"], 1e-4)
            self.assertEqual(
                rows[1]["training"]["auxiliary_optimizer_epsilon"], 1e-6
            )
            self.assertEqual(
                rows[1]["training"]["residual_head_optimizer_beta1"], 0.5
            )
            self.assertEqual(
                rows[1]["training"]["residual_head_optimizer_beta2"], 0.8
            )
            self.assertEqual(rows[1]["training"]["max_epochs"], 160)
            self.assertEqual(rows[1]["model"]["head_hidden_dim"], 80)

    def test_default_is_exact_flat_optimizer_control(self):
        model = _model()
        training = {"learning_rate": 0.001, "weight_decay": 0.0001}
        parameters, auxiliary_multiplier, auxiliary_group = (
            _optimizer_parameter_spec(model, training)
        )
        optimizer, _, _ = _build_optimizer(training, parameters)
        self.assertEqual(
            _residual_head_optimizer_epsilon(training), (1e-8, False)
        )
        self.assertEqual(auxiliary_multiplier, 1.0)
        self.assertEqual(auxiliary_group, 0)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["eps"], 1e-8)

    def test_head_epsilon_changes_only_downstream_group(self):
        model = _model()
        training = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer_epsilon": 1e-4,
            "residual_head_optimizer_epsilon": 1e-12,
            "auxiliary_optimizer_epsilon": 1e-6,
        }
        parameters, _, auxiliary_group = _optimizer_parameter_spec(
            model, training
        )
        optimizer, _, _ = _build_optimizer(training, parameters)
        self.assertEqual(auxiliary_group, 1)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["eps"], 1e-12)
        self.assertEqual(optimizer.param_groups[1]["eps"], 1e-6)
        auxiliary_ids = {
            id(parameter) for parameter in model.fmt_encoder.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            {id(parameter) for parameter in optimizer.param_groups[1]["params"]},
            auxiliary_ids,
        )
        self.assertFalse(
            {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
            .intersection(auxiliary_ids)
        )

    def test_validation_evidence_and_record_contract(self):
        self.assertEqual(
            _residual_head_optimizer_epsilon({
                "optimizer_epsilon": 1e-4,
            }),
            (1e-4, False),
        )
        self.assertEqual(
            _residual_head_optimizer_epsilon({
                "optimizer_epsilon": 1e-4,
                "residual_head_optimizer_epsilon": 1e-12,
            }),
            (1e-12, True),
        )
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(
                ValueError, "residual_head_optimizer_epsilon"
            ):
                _residual_head_optimizer_epsilon({
                    "residual_head_optimizer_epsilon": invalid
                })
        source = inspect.getsource(_train_one)
        self.assertIn("training_residual_head_optimizer_epsilon", source)
        self.assertIn("current_residual_head_epsilon", source)
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "e00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_residual_head_epsilon_refresh_109.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_residual_head_epsilon_refresh_109.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-109%24", gpu)
        self.assertIn("expected 660 per-run CSV files", evidence)
        self.assertIn("expected 660 temporary checkpoints", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
