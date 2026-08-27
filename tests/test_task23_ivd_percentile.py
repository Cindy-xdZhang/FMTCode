import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from Build_Task23_IVDPercentile_Labels import (
    all_percentiles,
    percentile_labels,
    percentile_tag,
)
from Prepare_Task3_IVDPercentile_Configs import prepare
from Prepare_Task3_IVDPercentile_Frozen_4_1 import prepare as prepare_4p1


ROOT = Path(__file__).resolve().parents[1]


class IVDPercentileContractTests(unittest.TestCase):
    def test_tags_and_complete_predeclared_sweep(self):
        spec = yaml.safe_load(
            (ROOT / "config" / "Ablation_Task23IVDPercentile_1.1.yaml")
            .read_text(encoding="utf-8")
        )
        values = all_percentiles(spec)
        self.assertEqual(values, [80.0, 85.0, 87.5, 90.0, 92.5, 95.0])
        self.assertEqual([percentile_tag(value) for value in values], [
            "p80", "p85", "p87p5", "p90", "p92p5", "p95",
        ])

    def test_higher_percentile_is_nested_inside_lower_percentile(self):
        volume = np.arange(100, dtype=np.float64).reshape(4, 5, 5)
        seeds = np.asarray([0, 79, 80, 84, 85, 89, 90, 94, 95, 99], dtype=float)
        values = [80.0, 85.0, 87.5, 90.0, 92.5, 95.0]
        thresholds, masks = percentile_labels(volume, seeds, values)
        self.assertTrue(all(
            thresholds[right] >= thresholds[left]
            for left, right in zip(values[:-1], values[1:])
        ))
        self.assertTrue(all(
            not np.any(masks[right] & ~masks[left])
            for left, right in zip(values[:-1], values[1:])
        ))

    def test_generated_task3_configs_are_linux_portable_and_frozen(self):
        source = yaml.safe_load(
            (ROOT / "config" / "Ablation_Task23IVDPercentile_1.1.yaml")
            .read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            source["output_dir"] = (temp / "experiment").as_posix()
            source["task3"]["output_root"] = (temp / "task3").as_posix()
            source["task3"]["generated_config_dir"] = (temp / "configs").as_posix()
            for key, value in source["task3"]["templates"].items():
                source["task3"]["templates"][key] = (ROOT / value).as_posix()
            config = temp / "master.yaml"
            config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
            prepare(str(config))
            generated = yaml.safe_load(
                (temp / "configs" / "task3_p87p5_fmt_old8.yaml")
                .read_text(encoding="utf-8")
            )
            original = yaml.safe_load(
                (ROOT / "config" / "mainExp_Task3_3D_3.2_global_ivd_fmt_old8.yaml")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(generated["model"], original["model"])
            self.assertEqual(generated["training"], original["training"])
            self.assertEqual(generated["fusion"], original["fusion"])
            for key in ("label_cache_root", "raw_checkpoint_dir", "output_dir"):
                self.assertNotIn("\\", generated[key])

    def test_4p1_sweep_keeps_frozen_recipe_and_retrains_percentile_models(self):
        source = yaml.safe_load(
            (ROOT / "config" / "Ablation_Task23IVDPercentile_1.2.yaml")
            .read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            source["output_dir"] = (temp / "experiment").as_posix()
            source["task3"]["output_root"] = (temp / "task3").as_posix()
            source["task3"]["generated_config_dir"] = (temp / "configs").as_posix()
            for key in ("main_config", "search_config"):
                source["task3"][key] = (ROOT / source["task3"][key]).as_posix()
            for key, value in source["task3"]["baseline_templates"].items():
                source["task3"]["baseline_templates"][key] = (
                    ROOT / value
                ).as_posix()
            config = temp / "master.yaml"
            config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
            prepare_4p1(str(config))

            search = yaml.safe_load(
                (temp / "configs" / "task3_p87p5_search.yaml")
                .read_text(encoding="utf-8")
            )
            original_search = yaml.safe_load(
                (ROOT / "config" / "Verify_Task3_FMTResidualFamilySearch_4.1.yaml")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(search["candidates"], original_search["candidates"])
            self.assertEqual(search["stage2_networks"], original_search["stage2_networks"])
            self.assertEqual(search["training"], original_search["training"])
            self.assertEqual(search["expected_ivd_percentile"], 87.5)
            self.assertFalse(search["require_source_reference_match"])

            final = yaml.safe_load(
                (temp / "configs" / "task3_p87p5_final.yaml")
                .read_text(encoding="utf-8")
            )
            original_main = yaml.safe_load(
                (ROOT / "config" / "mainExp_Task3_3D_4.1.yaml")
                .read_text(encoding="utf-8")
            )
            for key in (
                "stage2_selection", "frozen_recipe_manifest",
                "final_training_seeds",
            ):
                self.assertEqual(final[key], original_main[key])
            self.assertEqual(final["expected_ivd_percentile"], 87.5)
            self.assertFalse(final["require_confirmation_reference_match"])

            baseline = yaml.safe_load(
                (temp / "configs" / "task3_p87p5_baselines_old8.yaml")
                .read_text(encoding="utf-8")
            )
            original_baseline = yaml.safe_load(
                (ROOT / "config" / "mainExp_Task3_3D_3.2_global_ivd_baselines_old8.yaml")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(baseline["model"], original_baseline["model"])
            self.assertEqual(baseline["training"], original_baseline["training"])
            self.assertEqual(baseline["expected_ivd_percentile"], 87.5)


if __name__ == "__main__":
    unittest.main()
