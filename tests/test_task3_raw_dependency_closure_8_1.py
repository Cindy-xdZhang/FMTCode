import hashlib
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import zipfile

import yaml

import Freeze_Task3_RawDependencyClosure_8_1 as closure


def _write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    member = f"{path.stem}/data.pkl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, pickle.dumps(payload, protocol=4))


class Task3RawDependencyClosureTests(unittest.TestCase):
    def test_literal_path_extraction_does_not_unpickle(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            _write_checkpoint(checkpoint, {
                "raw_checkpoint": "outputs/main/raw_seed40.pt",
                "variant": "raw_fmt_residual",
            })
            self.assertEqual(
                closure._extract_checkpoint_string(
                    checkpoint, "raw_checkpoint"
                ),
                "outputs/main/raw_seed40.pt",
            )

    def test_unsafe_recorded_paths_are_rejected(self):
        for value in ("../raw.pt", "/absolute/raw.pt", "C:/raw.pt"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    closure._safe_relative_path(value)

    def test_freeze_and_verify_cover_paired_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_repo = root / "source"
            frozen_repo = root / "portfolio"
            output = root / "confirmation"
            models = []
            for dataset_index in range(10):
                dataset = f"dataset{dataset_index}"
                for seed in (40, 41):
                    recorded = Path(
                        "outputs/base/checkpoints"
                    ) / f"{dataset}_raw_seed{seed}.pt"
                    raw = source_repo / recorded
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_bytes(f"raw-{dataset}-{seed}".encode())
                    for arm in ("fmt", "raw_pca"):
                        source_checkpoint = (
                            source_repo / "outputs/search" / dataset
                            / f"seed{seed}" / arm / "checkpoints"
                            / f"{dataset}_{arm}_seed{seed}.pt"
                        )
                        frozen_checkpoint = (
                            frozen_repo / dataset / f"seed{seed}" / arm
                            / source_checkpoint.name
                        )
                        payload = {
                            "raw_checkpoint": recorded.as_posix(),
                            "variant": (
                                "raw_fmt_residual" if arm == "fmt"
                                else "raw_pca_residual"
                            ),
                        }
                        _write_checkpoint(source_checkpoint, payload)
                        _write_checkpoint(frozen_checkpoint, payload)
                        models.append({
                            "dataset": dataset,
                            "seed": seed,
                            "source": arm,
                            "checkpoint": str(frozen_checkpoint),
                            "checkpoint_sha256": hashlib.sha256(
                                frozen_checkpoint.read_bytes()
                            ).hexdigest(),
                            "source_checkpoint": str(source_checkpoint),
                        })
            recipe = output / "frozen_recipe_manifest.json"
            recipe.parent.mkdir(parents=True)
            recipe.write_text(json.dumps({
                "experiment": closure.EXPECTED_EXPERIMENT,
                "source_model_selection_sha256": "a" * 64,
                "models": models,
            }), encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "experiment": closure.EXPECTED_EXPERIMENT,
                "output_root": str(output),
                "recipe_manifest": str(recipe),
            }), encoding="utf-8")

            manifest = closure.freeze(config)
            closure.verify(config)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["frozen_model_count"], 40)
            self.assertEqual(payload["raw_dependency_count"], 20)
            self.assertFalse(payload["scientific_configuration_changed"])
            self.assertFalse(payload["confirmation_metrics_read"])
            self.assertEqual(sum(
                len(entry["referenced_models"])
                for entry in payload["entries"]
            ), 40)


if __name__ == "__main__":
    unittest.main()
