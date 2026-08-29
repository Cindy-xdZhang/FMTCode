import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from Confirm_Task3_CombinedOptimization_12_1 import (
    _aggregate,
    _load_spec,
    _sha256,
    _validate_shard_rows,
)


class Task3CombinedConfirmationTests(unittest.TestCase):
    def test_paired_summary_uses_fmt_minus_raw_pca(self):
        rows = []
        for dataset, family, raw, fmt in (
            ("a", "family_a", 0.50, 0.70),
            ("b", "family_a", 0.60, 0.70),
            ("c", "family_b", 0.55, 0.75),
        ):
            for seed in (40, 41):
                rows.extend((
                    {"dataset": dataset, "physical_family": family,
                     "seed": seed, "source": "raw_pca", "f1": raw,
                     "average_precision": raw + 0.05},
                    {"dataset": dataset, "physical_family": family,
                     "seed": seed, "source": "fmt", "f1": fmt,
                     "average_precision": fmt + 0.05},
                ))
        result = _aggregate(rows, ["a", "b", "c"])
        self.assertAlmostEqual(
            result["dataset_macro_f1_gain_vs_raw_pca"], 1.0 / 6.0
        )
        self.assertAlmostEqual(
            result["family_macro_f1_gain_vs_raw_pca"], 0.175
        )
        self.assertEqual(result["positive_dataset_f1_gain_count"], 3)
        self.assertEqual(result["positive_family_f1_gain_count"], 2)

    def test_paired_summary_rejects_a_missing_arm(self):
        rows = [{
            "dataset": "a", "physical_family": "family_a",
            "seed": 40, "source": "fmt", "f1": 0.7,
            "average_precision": 0.75,
        }]
        with self.assertRaisesRegex(RuntimeError, "missing.*raw_pca"):
            _aggregate(rows, ["a"])

    def test_static_config_must_keep_confirmation_closed(self):
        source = Path(
            "config/Confirm_Task3_CombinedOptimization_12.1.yaml"
        )
        spec = yaml.safe_load(source.read_text(encoding="utf-8"))
        self.assertFalse(spec["confirmation_opened_by_static_preflight"])
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(len(spec["datasets"]), 10)
        self.assertEqual(spec["target_dataset_macro_f1_gain"], 0.15)
        self.assertEqual(spec["aspirational_dataset_macro_f1_gain"], 0.16)
        self.assertEqual(
            spec["confirmation_seed_grid_phase"],
            [0.318359375, 0.4561042524005485, -0.3352],
        )

        with tempfile.TemporaryDirectory() as directory:
            changed = dict(spec)
            changed["confirmation_opened_by_static_preflight"] = True
            path = Path(directory) / "bad.yaml"
            path.write_text(
                yaml.safe_dump(changed, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "must not open"):
                _load_spec(path)

    def test_static_config_rejects_phase_or_partition_changes(self):
        source = Path(
            "config/Confirm_Task3_CombinedOptimization_12.1.yaml"
        )
        spec = yaml.safe_load(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            changed_phase = copy.deepcopy(spec)
            changed_phase["confirmation_seed_grid_phase"][0] += 0.01
            phase_path = directory / "changed_phase.yaml"
            phase_path.write_text(
                yaml.safe_dump(changed_phase, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "phase changed"):
                _load_spec(phase_path)

            changed_partition = copy.deepcopy(spec)
            changed_partition["confirmation_roots"]["new2"]["datasets"][0] = (
                "smokeBuoyancy"
            )
            partition_path = directory / "changed_partition.yaml"
            partition_path.write_text(
                yaml.safe_dump(changed_partition, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dataset order changed"):
                _load_spec(partition_path)

    def test_12_2_is_only_an_operational_source_staging_repair(self):
        previous = yaml.safe_load(Path(
            "config/Confirm_Task3_CombinedOptimization_12.1.yaml"
        ).read_text(encoding="utf-8"))
        repaired = yaml.safe_load(Path(
            "config/Confirm_Task3_CombinedOptimization_12.2.yaml"
        ).read_text(encoding="utf-8"))
        allowed = {"experiment", "output_root", "recipe_manifest"}
        for key in set(previous) | set(repaired):
            if key not in allowed:
                self.assertEqual(
                    repaired[key], previous[key],
                    f"12.2 changed scientific setting {key}",
                )
        self.assertNotEqual(repaired["experiment"], previous["experiment"])
        _load_spec("config/Confirm_Task3_CombinedOptimization_12.2.yaml")

    def test_shard_validation_rejects_duplicates_and_requires_both_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"sealed-checkpoint")
            spec = {"experiment": "confirmation", "paired_seeds": [40]}
            common = {
                "experiment": "confirmation",
                "recipe_manifest_sha256": "manifest-hash",
                "optimization_selection_sha256": "selection-hash",
                "dataset": "flow",
                "physical_family": "family",
                "seed": 40,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "f1": 0.75,
                "average_precision": 0.80,
            }
            fmt = {
                **common, "source": "fmt", "method": "fmt_residual",
            }
            raw = {
                **common, "source": "raw_pca", "method": "raw_pca_residual",
            }
            _validate_shard_rows(
                [fmt, raw], spec, "flow", "family", "selection-hash",
                "manifest-hash", require_complete=True,
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                _validate_shard_rows(
                    [fmt, raw, fmt], spec, "flow", "family",
                    "selection-hash", "manifest-hash", require_complete=True,
                )
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                _validate_shard_rows(
                    [fmt], spec, "flow", "family", "selection-hash",
                    "manifest-hash", require_complete=True,
                )


if __name__ == "__main__":
    unittest.main()
