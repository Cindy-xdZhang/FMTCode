import hashlib
import json
from pathlib import Path

import yaml

import Select_Task3_FinalPortfolio_49_1 as portfolio
from Search_Task3_FMTResidual_3D import _write_csv


CONFIG = Path("config/Verify_Task3_FinalPortfolio_49.1.yaml")


def test_portfolio_is_training_free_and_confirmation_closed():
    spec = portfolio._load_spec(CONFIG)
    report = portfolio.static_preflight(CONFIG)
    assert report["source_count"] == 3
    assert report["dataset_count"] == 10
    assert report["training_runs"] == 0
    assert report["confirmation_opened"] is False
    assert spec["frozen_confirmation_seeds"] == [40, 41]
    code = Path("Select_Task3_FinalPortfolio_49_1.py").read_text(
        encoding="utf-8"
    )
    assert "_train_one" not in code
    assert "torch" not in code


def test_selection_metrics_are_fixed_before_source_results():
    spec = portfolio._load_spec(CONFIG)
    assert spec["selection_metrics"] == [
        "dataset_macro_f1_gain_vs_raw_pca",
        "dataset_macro_fmt_f1",
        "dataset_macro_average_precision_gain_vs_raw_pca",
        "dataset_macro_fmt_average_precision",
        "positive_dataset_count",
        "worst_dataset_f1_gain",
        "worst_seed_f1_gain",
    ]
    assert spec["selection"]["require_source_absolute_fmt_guard"] is True
    assert spec["selection"]["target_dataset_macro_f1_gain"] == 0.20


def test_portfolio_sources_are_exact_completed_search_versions():
    spec = portfolio._load_spec(CONFIG)
    observed = {
        name: source["expected_experiment"]
        for name, source in spec["sources"].items()
    }
    assert observed == {
        "safe_factor": "Verify_Task3_SafeFactorCombination_44.1",
        "head_alpha_clip": "Verify_Task3_HeadAlphaClipCombination_45.1",
        "full_stack": "Verify_Task3_HeadFullStackCombination_48.1",
    }
    assert all(
        len(source["expected_config_canonical_sha256"]) == 64
        for source in spec["sources"].values()
    )


def _fake_source(root: Path, source_spec: dict, datasets: list[str],
                 families: dict[str, list[str]], rank: int) -> None:
    config_path = root / source_spec["paths"]["config"]
    preflight_path = root / source_spec["paths"]["preflight"]
    selection_path = root / source_spec["paths"]["selection"]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(f"outputs/source_{rank}")
    candidate_id = f"candidate_{rank}"
    recipe = {"id": candidate_id, "fmt_feature": "aivd2w8_dft"}
    overlay = {
        "experiment": source_spec["expected_experiment"],
        "output_root": str(output_root),
        "optimization_candidates": [{"id": candidate_id}],
    }
    config_path.write_text(
        yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
    )
    source_spec["expected_config_canonical_sha256"] = (
        portfolio._canonical_text_sha256(config_path)
    )
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    preflight = {
        "experiment": source_spec["expected_experiment"],
        "optimization_config_sha256": config_hash,
        "confirmation_opened": False,
    }
    preflight_path.write_text(
        json.dumps(preflight, sort_keys=True), encoding="utf-8"
    )
    preflight_hash = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    gain = 0.17 + 0.01 * rank
    primary = {}
    for family, family_datasets in families.items():
        details = {
            dataset: {
                "fmt": {"f1": 0.88 + 0.001 * rank,
                        "average_precision": 0.95},
                "raw_pca": {"f1": 0.88 + 0.001 * rank - gain,
                            "average_precision": 0.73},
                "f1_gain": gain,
                "average_precision_gain": 0.22,
            }
            for dataset in family_datasets
        }
        primary[family] = {
            "optimization_id": candidate_id,
            "optimization_recipe_json": json.dumps(recipe, sort_keys=True),
            "eligible": True,
            "absolute_fmt_guard_passed": True,
            "dataset_macro_f1_gain_vs_raw_pca": gain,
            "dataset_macro_fmt_f1": 0.88 + 0.001 * rank,
            "dataset_macro_average_precision_gain_vs_raw_pca": 0.22,
            "dataset_macro_fmt_average_precision": 0.95,
            "positive_dataset_count": len(family_datasets),
            "worst_dataset_f1_gain": gain,
            "worst_seed_f1_gain": gain,
            "datasets_json": json.dumps(details, sort_keys=True),
        }
    selection = {
        "experiment": source_spec["expected_experiment"],
        "optimization_config_sha256": config_hash,
        "preflight_manifest_sha256": preflight_hash,
        "confirmation_opened": False,
        "paired_seeds": [40, 41, 42],
        "absolute_fmt_guard": {"control_optimization_id": "control"},
        "primary_by_group": primary,
    }
    selection_path.write_text(
        json.dumps(selection, sort_keys=True), encoding="utf-8"
    )

    for dataset in datasets:
        for seed in (40, 41):
            for arm, variant in (
                ("fmt", "raw_fmt_residual"),
                ("raw_pca", "raw_pca_residual"),
            ):
                result_path = (
                    root / output_root / "candidates" / candidate_id / dataset
                    / f"seed{seed}" / arm / "per_run.csv"
                )
                checkpoint = (
                    result_path.parent / "checkpoints"
                    / f"{dataset}_{variant}_seed{seed}.pt"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"{rank}-{dataset}-{seed}-{arm}".encode())
                _write_csv(result_path, [{
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "auxiliary_source": arm,
                    "optimization_id": candidate_id,
                    "optimization_recipe_json": json.dumps(
                        recipe, sort_keys=True
                    ),
                    "fmt_feature": "aivd2w8_dft",
                    "fmt_dim": 3,
                    "parameter_count": 100,
                    "trainable_residual_parameter_count": 10,
                    "checkpoint": str(checkpoint.relative_to(root)),
                    "optimization_config_sha256": config_hash,
                    "preflight_manifest_sha256": preflight_hash,
                }])


def test_portfolio_selects_best_guarded_source_and_freezes_40_models(tmp_path):
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    families = {
        "channel": ["channel"],
        "halfcylinder": [
            "cylinder3d", "halfcylinderRe640", "halfcylinderRe6400",
        ],
        "tangaroa": ["tangaroa"],
        "deltaWing": ["deltaWing_resampled", "deltaWing_LBM"],
        "f22raptor": ["f22raptor"],
        "boeing747": ["boeing747"],
        "smokeBuoyancy": ["smokeBuoyancy"],
    }
    for rank, (name, source) in enumerate(spec["sources"].items(), 1):
        root = tmp_path / name
        source["repo_root"] = str(root)
        _fake_source(root, source, spec["datasets"], families, rank)
    spec["output_root"] = str(tmp_path / "portfolio_output")
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    result_path = portfolio.select(config_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["models"]) == 40
    assert set(
        row["portfolio_source"] for row in result["primary_by_group"].values()
    ) == {"full_stack"}
    assert result["development_dataset_macro_f1_gain_vs_raw_pca"] == 0.2
    assert result["confirmation_opened"] is False
