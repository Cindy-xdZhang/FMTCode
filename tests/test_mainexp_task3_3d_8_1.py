import hashlib
from pathlib import Path
import unittest

import numpy as np
import yaml

import Build_Task3_AdaptiveTuned_Confirmation_7_2 as prior_spatial
import Build_Task3_ExtendedTuned_Confirmation_8_1 as spatial
import Confirm_Task3_AdaptiveTuned_7_2 as prior_confirm
import Confirm_Task3_ExtendedTuned_8_1 as confirm


CONFIG = Path("config/mainExp_Task3_3D_8.1.yaml")


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        result += (index % base) * factor
        index //= base
        factor /= base
    return result


class MainExpTask3_8_1Tests(unittest.TestCase):
    def test_phase_is_preregistered_and_new(self):
        digest = hashlib.sha256(spatial.PHASE_KEY.encode("utf-8")).hexdigest()
        index = 1 + int(digest[:8], 16) % 1024
        phase = [
            _radical_inverse(index, base) - 0.5 for base in (2, 3, 5)
        ]
        self.assertEqual(digest, spatial.PHASE_KEY_SHA256)
        self.assertEqual(index, spatial.HALTON_INDEX)
        self.assertEqual(index, 798)
        self.assertTrue(np.allclose(
            phase, spatial.SEED_GRID_PHASE, rtol=0.0, atol=1e-15
        ))
        exposed = (
            [0.31, -0.23, 0.17],
            [-0.37, 0.29, -0.11],
            [0.318359375, 0.4561042524005485, -0.3352],
            [0.021484375, -0.34224965706447186, 0.0328],
            [-0.1044921875, -0.3655692729766804, 0.11632],
            prior_spatial.SEED_GRID_PHASE,
        )
        self.assertTrue(all(
            not np.allclose(phase, old, rtol=0.0, atol=1e-15)
            for old in exposed
        ))

    def test_config_freezes_54_1_portfolio_and_ivd_p95(self):
        spec = confirm._load_spec(CONFIG)
        self.assertEqual(spec["status"], "fresh_spatial_confirmation")
        self.assertEqual(spec["paired_seeds"], [40, 41])
        self.assertEqual(
            spec["source_model"]["source_paired_seeds"], [40, 41, 42]
        )
        self.assertEqual(
            spec["source_model"]["expected_experiment"],
            confirm.SOURCE_EXPERIMENT,
        )
        self.assertEqual(spec["expected_ivd_percentile"], 95.0)
        self.assertEqual(spec["target_dataset_macro_f1_gain"], 0.15)
        self.assertEqual(spec["aspirational_dataset_macro_f1_gain"], 0.20)
        self.assertEqual(spec["confirmation_count"], 4)
        self.assertEqual(len(spec["datasets"]), 10)
        source_config = Path(
            "config/Verify_Task3_ExtendedPortfolio_54.1.yaml"
        )
        canonical = source_config.read_text(encoding="utf-8").replace(
            "\r\n", "\n"
        ).replace("\r", "\n")
        self.assertEqual(
            spec["source_model"]["expected_config_canonical_sha256"],
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def test_label_configs_are_whole_field_ivd_p95(self):
        for suffix, expected_count in (("old8", 8), ("new2", 2)):
            path = Path(f"config/mainExp_Task3_3D_8.1_labels_{suffix}.yaml")
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(spec["label"], {
                "definition": "standard_global_ivd",
                "percentile": 95.0,
            })
            self.assertEqual(spec["expected_slices"], 4)
            self.assertEqual(len(spec["datasets"]), expected_count)
            self.assertIn("mainExp_Task3_3D_8.1", spec["source_cache_root"])

    def test_builder_scope_restores_7_2(self):
        original = {
            "EXPERIMENT": prior_spatial.EXPERIMENT,
            "PHASE_KEY": prior_spatial.PHASE_KEY,
            "SEED_GRID_PHASE": prior_spatial.SEED_GRID_PHASE,
            "SETTINGS": prior_spatial.SETTINGS,
        }
        with spatial._configured_base():
            self.assertEqual(prior_spatial.EXPERIMENT, spatial.EXPERIMENT)
            self.assertEqual(prior_spatial.PHASE_KEY, spatial.PHASE_KEY)
            self.assertEqual(
                prior_spatial.SEED_GRID_PHASE, spatial.SEED_GRID_PHASE
            )
            self.assertEqual(prior_spatial.SETTINGS, spatial.SETTINGS)
        for name, value in original.items():
            self.assertEqual(getattr(prior_spatial, name), value)

    def test_confirmation_scope_restores_7_2(self):
        original_spatial = prior_confirm.spatial
        original_source = prior_confirm.SOURCE_EXPERIMENT
        with confirm._configured_base():
            self.assertIs(prior_confirm.spatial, spatial)
            self.assertEqual(
                prior_confirm.SOURCE_EXPERIMENT, confirm.SOURCE_EXPERIMENT
            )
        self.assertIs(prior_confirm.spatial, original_spatial)
        self.assertEqual(prior_confirm.SOURCE_EXPERIMENT, original_source)

    def test_confirmation_contains_no_training_entrypoint(self):
        text = Path("Confirm_Task3_ExtendedTuned_8_1.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_train_one", text)
        self.assertNotIn("optimizer.step", text)
        self.assertNotIn("backward()", text)


if __name__ == "__main__":
    unittest.main()
