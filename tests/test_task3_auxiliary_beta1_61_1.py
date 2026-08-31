"""Contracts for Task3 auxiliary-projection Adam beta1 search 61.1."""

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
    _auxiliary_optimizer_betas,
    _build_optimizer,
    _optimizer_parameter_spec,
)


CONFIG = "config/Verify_Task3_AuxiliaryBeta1_61.1.yaml"
EXPECTED = [None, 0.0, 0.25, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]


class AuxiliaryBeta1Tests(unittest.TestCase):
    def test_grid_job_count_source_and_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 11)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 109), ("smokeBuoyancy", 10))
        source = spec["combination_sources"]["portfolio"]
        self.assertEqual(
            source["expected_experiment"],
            "Verify_Task3_AuxiliaryBeta2Portfolio_60.1",
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.209)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_auxiliary_beta1(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["portfolio"])
            if index == 0:
                self.assertEqual(set(row), {"id", "sources"})
                observed.append(None)
            else:
                self.assertEqual(set(row), {"id", "sources", "training"})
                self.assertEqual(
                    set(row["training"]), {"auxiliary_optimizer_beta1"}
                )
                observed.append(float(
                    row["training"]["auxiliary_optimizer_beta1"]
                ))
        self.assertEqual(observed, EXPECTED)

    def test_portfolio_recipe_preserves_beta2_before_local_override(self):
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
                        "training": {
                            "optimizer_beta1": 0.8,
                            "optimizer_beta2": 0.98,
                            "auxiliary_optimizer_beta2": 0.95,
                            "auxiliary_learning_rate_multiplier": 2.0,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": "Verify_Task3_AuxiliaryBeta2Portfolio_60.1",
                "confirmation_opened": False,
                "primary_by_group": primary,
            }), encoding="utf-8")
            spec["combination_sources"]["portfolio"]["selection"] = str(
                source_path
            )
            resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(hashes), {"portfolio"})
        for rows in resolved.values():
            control = rows[0]["training"]
            self.assertEqual(control["auxiliary_optimizer_beta2"], 0.95)
            self.assertNotIn("auxiliary_optimizer_beta1", control)
            candidate = rows[1]["training"]
            self.assertEqual(candidate["auxiliary_optimizer_beta2"], 0.95)
            self.assertEqual(candidate["auxiliary_optimizer_beta1"], 0.0)

    @staticmethod
    def _model():
        return PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=8,
            head_architecture="deep_mlp",
            head_hidden_dim=16,
        )

    def test_noncontrol_changes_only_auxiliary_group_beta1(self):
        model = self._model()
        training = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer_beta1": 0.8,
            "optimizer_beta2": 0.98,
            "auxiliary_optimizer_beta1": 0.5,
            "auxiliary_optimizer_beta2": 0.95,
        }
        parameters, multiplier, auxiliary_group = _optimizer_parameter_spec(
            model, training
        )
        optimizer, _, _ = _build_optimizer(training, parameters)
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(auxiliary_group, 1)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.8, 0.98))
        self.assertEqual(optimizer.param_groups[1]["betas"], (0.5, 0.95))

    def test_beta1_inherits_auxiliary_beta2_and_validates(self):
        betas, overridden = _auxiliary_optimizer_betas({
            "optimizer_beta1": 0.8,
            "optimizer_beta2": 0.98,
            "auxiliary_optimizer_beta2": 0.95,
            "auxiliary_optimizer_beta1": 0.0,
        })
        self.assertEqual(betas, (0.0, 0.95))
        self.assertTrue(overridden)
        for invalid in (-0.1, 1.0, float("inf"), float("nan")):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(
                    ValueError, "auxiliary_optimizer_beta1"
                ):
                    _auxiliary_optimizer_betas({
                        "auxiliary_optimizer_beta1": invalid
                    })

    def test_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "c00_control_auxbeta1",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_slurm_scripts_match_scale_and_retain_models(self):
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_beta1_61.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_beta1_61.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-109%24", gpu)
        self.assertIn("expected 660 per-run CSV files", evidence)
        self.assertIn("expected 660 temporary checkpoints", evidence)
        self.assertIn("Audit_Task3_ParameterSearch.py", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
