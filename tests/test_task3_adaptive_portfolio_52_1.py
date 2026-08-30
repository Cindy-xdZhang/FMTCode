import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

import Select_Task3_AdaptivePortfolio_52_1 as portfolio


CONFIG = Path("config/Verify_Task3_AdaptivePortfolio_52.1.yaml")


class AdaptivePortfolioTests(unittest.TestCase):
    def test_static_contract_is_training_free_and_uses_five_sources(self):
        spec = portfolio._load_spec(CONFIG)
        report = portfolio.static_preflight(CONFIG)
        self.assertEqual(report["source_count"], 5)
        self.assertEqual(report["training_runs"], 0)
        self.assertFalse(report["confirmation_opened"])
        self.assertEqual(set(spec["sources"]), portfolio.EXPECTED_SOURCES)
        self.assertEqual(spec["frozen_confirmation_seeds"], [40, 41])

    def test_sources_are_exact_declared_search_versions(self):
        spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        observed = {
            name: row["expected_experiment"]
            for name, row in spec["sources"].items()
        }
        self.assertEqual(observed, {
            "safe_factor": "Verify_Task3_SafeFactorCombination_44.1",
            "head_alpha_clip": "Verify_Task3_HeadAlphaClipCombination_45.1",
            "full_stack": "Verify_Task3_HeadFullStackCombination_48.1",
            "focal_gamma_low": "Verify_Task3_FocalGammaLow_50.1",
            "dropout_high": "Verify_Task3_ResidualDropoutHigh_51.1",
        })
        self.assertTrue(all(
            len(row["expected_config_canonical_sha256"]) == 64
            for row in spec["sources"].values()
        ))

    def test_base_loader_is_restored_after_scoped_use(self):
        original = portfolio._base._load_spec
        with portfolio._configured_base():
            self.assertIs(portfolio._base._load_spec, portfolio._load_spec)
        self.assertIs(portfolio._base._load_spec, original)

    def test_freeze_copies_and_repoints_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            output = tmp_path / "output"
            source = tmp_path / "source"
            models = []
            for dataset_index in range(10):
                dataset = f"dataset{dataset_index}"
                for seed in (40, 41):
                    for arm in ("fmt", "raw_pca"):
                        root = source / dataset / f"seed{seed}" / arm
                        result = root / "per_run.csv"
                        checkpoint = root / "checkpoints" / "model.pt"
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        result.write_text(
                            f"{dataset},{seed},{arm}\n", encoding="utf-8"
                        )
                        checkpoint.write_bytes(
                            f"{dataset}-{seed}-{arm}".encode()
                        )
                        models.append({
                            "dataset": dataset,
                            "seed": seed,
                            "source": arm,
                            "result": str(result),
                            "result_sha256": hashlib.sha256(
                                result.read_bytes()
                            ).hexdigest(),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                        })
            selection = output / "portfolio_selection.json"
            selection.parent.mkdir(parents=True)
            selection.write_text(
                json.dumps({
                    "experiment": portfolio.EXPERIMENT,
                    "models": models,
                }),
                encoding="utf-8",
            )
            portfolio._freeze_local_copies(selection)
            frozen = json.loads(selection.read_text(encoding="utf-8"))
            self.assertEqual(frozen["frozen_model_count"], 40)
            self.assertEqual(frozen["frozen_artifact_file_count"], 80)
            self.assertEqual(len(frozen["models"]), 40)
            for model in frozen["models"]:
                self.assertTrue(Path(model["result"]).is_file())
                self.assertTrue(Path(model["checkpoint"]).is_file())
                self.assertEqual(
                    hashlib.sha256(
                        Path(model["result"]).read_bytes()
                    ).hexdigest(),
                    model["result_sha256"],
                )
                self.assertEqual(
                    hashlib.sha256(
                        Path(model["checkpoint"]).read_bytes()
                    ).hexdigest(),
                    model["checkpoint_sha256"],
                )

    def test_selector_contains_no_training_or_confirmation_access(self):
        text = Path("Select_Task3_AdaptivePortfolio_52_1.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_train_one", text)
        self.assertIn("confirmation data", text)
        self.assertNotIn("Confirm_Task3", text)


if __name__ == "__main__":
    unittest.main()
