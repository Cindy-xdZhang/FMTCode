import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import yaml

import Audit_Task3_AdaptiveTuned_7_2 as audit_module


BASE_CONFIG = Path("config/mainExp_Task3_3D_7.2.yaml")
FAMILIES = {
    "channel": "channel",
    "cylinder3d": "halfcylinder",
    "halfcylinderRe640": "halfcylinder",
    "halfcylinderRe6400": "halfcylinder",
    "tangaroa": "tangaroa",
    "deltaWing_resampled": "deltaWing",
    "deltaWing_LBM": "deltaWing",
    "f22raptor": "f22raptor",
    "boeing747": "boeing747",
    "smokeBuoyancy": "smokeBuoyancy",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _fixture(root: Path) -> tuple[Path, Path]:
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True)
    spec = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    spec["output_root"] = str(artifact_dir)
    spec["recipe_manifest"] = str(
        artifact_dir / "frozen_recipe_manifest.json"
    )
    config_path = root / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    config_hash = _sha256(config_path)

    models = []
    row_inputs = []
    for dataset_index, dataset in enumerate(spec["datasets"]):
        family = FAMILIES[dataset]
        for seed in spec["paired_seeds"]:
            seed_offset = (int(seed) - 40) * 0.002
            raw_f1 = 0.50 + dataset_index * 0.01 + seed_offset
            raw_ap = 0.55 + dataset_index * 0.01 + seed_offset
            for source in ("fmt", "raw_pca"):
                root_dir = (
                    root / "frozen_artifacts" / dataset
                    / f"seed{seed}" / source
                )
                root_dir.mkdir(parents=True)
                result_path = root_dir / "per_run.csv"
                checkpoint_path = root_dir / "model.pt"
                result_path.write_text(
                    f"{dataset},{seed},{source}\n", encoding="utf-8"
                )
                checkpoint_path.write_bytes(
                    f"{dataset}-{seed}-{source}".encode("utf-8")
                )
                model = {
                    "dataset": dataset,
                    "physical_family": family,
                    "seed": int(seed),
                    "source": source,
                    "variant": (
                        "raw_fmt_residual" if source == "fmt"
                        else "raw_pca_residual"
                    ),
                    "candidate_id": "u00",
                    "fmt_feature": "aivd2w8_dft",
                    "fmt_dim": 24,
                    "parameter_count": 1000,
                    "trainable_residual_parameter_count": 900,
                    "result": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                }
                models.append(model)
                row_inputs.append((
                    model,
                    raw_f1 + (0.20 if source == "fmt" else 0.0),
                    raw_ap + (0.18 if source == "fmt" else 0.0),
                ))

    source_selection_hash = "a" * 64
    manifest = {
        "schema": 1,
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": config_hash,
        "source_model_selection_sha256": source_selection_hash,
        "confirmation_seed_grid_phase": spec["confirmation_seed_grid_phase"],
        "phase_key": spec["phase_key"],
        "phase_key_sha256": spec["phase_key_sha256"],
        "halton_index": spec["halton_index"],
        "new_spatial_primitive_population": True,
        "confirmation_data_opened": False,
        "models": models,
    }
    manifest_path = artifact_dir / "frozen_recipe_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest_hash = _sha256(manifest_path)
    evaluation = {
        "schema": 1,
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": config_hash,
        "recipe_manifest_sha256": manifest_hash,
        "source_model_selection_sha256": source_selection_hash,
        "confirmation_seed_grid_phase": spec["confirmation_seed_grid_phase"],
        "confirmation_was_generated_after_recipe_freeze": True,
        "expected_evaluations": 40,
        "models": models,
    }
    evaluation_path = artifact_dir / "evaluation_preflight.json"
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8"
    )
    evaluation_hash = _sha256(evaluation_path)

    rows = []
    for model, f1, average_precision in row_inputs:
        rows.append({
            "experiment": spec["experiment"],
            "status": spec["status"],
            "config_sha256": config_hash,
            "recipe_manifest_sha256": manifest_hash,
            "evaluation_preflight_sha256": evaluation_hash,
            "dataset": model["dataset"],
            "physical_family": model["physical_family"],
            "candidate_id": model["candidate_id"],
            "fmt_feature": model["fmt_feature"],
            "seed": model["seed"],
            "source": model["source"],
            "method": (
                "fmt_residual" if model["source"] == "fmt"
                else "raw_pca_residual"
            ),
            "sample_count": 200,
            "positive_fraction": 0.05,
            "frozen_threshold": 0.50,
            "frozen_alpha": 0.10,
            "checkpoint": model["checkpoint"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "f1": f1,
            "average_precision": average_precision,
        })
    per_run_path = artifact_dir / "per_run.csv"
    _write_csv(per_run_path, rows)

    datasets = {}
    for dataset in spec["datasets"]:
        selected = [row for row in rows if row["dataset"] == dataset]
        raw_rows = [row for row in selected if row["source"] == "raw_pca"]
        fmt_rows = [row for row in selected if row["source"] == "fmt"]
        raw = {
            metric: _mean(float(row[metric]) for row in raw_rows)
            for metric in ("f1", "average_precision")
        }
        fmt = {
            metric: _mean(float(row[metric]) for row in fmt_rows)
            for metric in ("f1", "average_precision")
        }
        datasets[dataset] = {
            "physical_family": FAMILIES[dataset],
            "raw_pca_residual": raw,
            "fmt_residual": fmt,
            "f1_gain": fmt["f1"] - raw["f1"],
            "average_precision_gain": (
                fmt["average_precision"] - raw["average_precision"]
            ),
        }
    families = {}
    for family in sorted(set(FAMILIES.values())):
        family_datasets = sorted(
            dataset for dataset in spec["datasets"]
            if FAMILIES[dataset] == family
        )
        families[family] = {
            "datasets": family_datasets,
            "f1_gain": _mean(datasets[name]["f1_gain"] for name in family_datasets),
            "average_precision_gain": _mean(
                datasets[name]["average_precision_gain"]
                for name in family_datasets
            ),
        }
    raw_f1 = _mean(row["raw_pca_residual"]["f1"] for row in datasets.values())
    fmt_f1 = _mean(row["fmt_residual"]["f1"] for row in datasets.values())
    raw_ap = _mean(
        row["raw_pca_residual"]["average_precision"]
        for row in datasets.values()
    )
    fmt_ap = _mean(
        row["fmt_residual"]["average_precision"]
        for row in datasets.values()
    )
    f1_gain = fmt_f1 - raw_f1
    ap_gain = fmt_ap - raw_ap
    summary = {
        "experiment": spec["experiment"],
        "status": spec["status"],
        "comparison": "synthetic paired confirmation",
        "source_search_experiment": audit_module.EXPECTED_SOURCE_EXPERIMENT,
        "source_portfolio_experiment": audit_module.EXPECTED_SOURCE_EXPERIMENT,
        "source_portfolio_artifacts_copied_before_source_cleanup": True,
        "fresh_confirmation": True,
        "confirmation_data_was_not_used_for_selection": True,
        "recipe_manifest_sha256": manifest_hash,
        "evaluation_preflight_sha256": evaluation_hash,
        "source_model_selection_sha256": source_selection_hash,
        "confirmation_seed_grid_phase": spec["confirmation_seed_grid_phase"],
        "phase_key_sha256": spec["phase_key_sha256"],
        "halton_index": spec["halton_index"],
        "paired_seeds": spec["paired_seeds"],
        "dataset_macro_raw_pca_f1": raw_f1,
        "dataset_macro_fmt_f1": fmt_f1,
        "dataset_macro_raw_pca_ap": raw_ap,
        "dataset_macro_fmt_ap": fmt_ap,
        "dataset_macro_f1_gain_vs_raw_pca": f1_gain,
        "dataset_macro_ap_gain_vs_raw_pca": ap_gain,
        "family_macro_f1_gain_vs_raw_pca": _mean(
            row["f1_gain"] for row in families.values()
        ),
        "family_macro_ap_gain_vs_raw_pca": _mean(
            row["average_precision_gain"] for row in families.values()
        ),
        "positive_dataset_f1_gain_count": sum(
            row["f1_gain"] > 0 for row in datasets.values()
        ),
        "positive_family_f1_gain_count": sum(
            row["f1_gain"] > 0 for row in families.values()
        ),
        "minimum_dataset_f1_gain": min(
            row["f1_gain"] for row in datasets.values()
        ),
        "datasets": datasets,
        "families": families,
        "source_development_f1_gain": 0.21,
        "target_dataset_macro_f1_gain": spec["target_dataset_macro_f1_gain"],
        "target_reached": f1_gain >= spec["target_dataset_macro_f1_gain"],
        "aspirational_dataset_macro_f1_gain": spec[
            "aspirational_dataset_macro_f1_gain"
        ],
        "aspirational_target_reached": (
            f1_gain >= spec["aspirational_dataset_macro_f1_gain"]
        ),
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return config_path, artifact_dir


class AdaptiveTunedAuditTests(unittest.TestCase):
    def test_independent_audit_passes_and_rejects_metric_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path, artifact_dir = _fixture(Path(temporary))
            output = artifact_dir / "independent_audit.json"
            with redirect_stdout(io.StringIO()):
                report = audit_module.audit(
                    config_path, artifact_dir, output
                )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["rows"], 40)
            self.assertEqual(report["counts"]["frozen_models"], 40)
            self.assertAlmostEqual(
                report["dataset_macro"]["f1_gain"], 0.20, places=12
            )
            self.assertTrue(report["primary_target_reached"])
            self.assertTrue(output.is_file())

            with (artifact_dir / "per_run.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["f1"] = str(float(rows[0]["f1"]) - 0.05)
            _write_csv(artifact_dir / "per_run.csv", rows)
            with self.assertRaisesRegex(RuntimeError, "summary .* differs"):
                with redirect_stdout(io.StringIO()):
                    audit_module.audit(config_path, artifact_dir)

    def test_independent_audit_rejects_provenance_and_checkpoint_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path, artifact_dir = _fixture(Path(temporary))
            summary_path = artifact_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["source_portfolio_experiment"] = "wrong-source"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source_portfolio_experiment"):
                audit_module.audit(config_path, artifact_dir)

        with tempfile.TemporaryDirectory() as temporary:
            config_path, artifact_dir = _fixture(Path(temporary))
            manifest = json.loads(
                (artifact_dir / "frozen_recipe_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            checkpoint = Path(manifest["models"][0]["checkpoint"])
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
            with self.assertRaisesRegex(RuntimeError, "checkpoint hash"):
                audit_module.audit(config_path, artifact_dir)


if __name__ == "__main__":
    unittest.main()
