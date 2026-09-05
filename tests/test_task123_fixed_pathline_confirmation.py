import hashlib
import json
from pathlib import Path

import yaml

from experiments.Audit_Task123_FixedPathline_Confirmation import (
    audit,
    cleanup,
)
from experiments.Run_Task123_FixedPathline_Confirmation import summarize


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_confirmation_summary_audit_and_cleanup(tmp_path: Path) -> None:
    source = Path("config/Verify_Task123_FixedPathline_1.1.yaml")
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    spec["output_root"] = str(tmp_path / "output")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    config_sha = _sha(config)
    root = Path(spec["output_root"])
    selection_path = root / "selection" / "selection.json"
    _write(selection_path, {
        "experiment": spec["experiment"],
        "config_sha256": config_sha,
        "status": "development_winner_frozen_before_confirmation",
        "confirmation_opened": False,
        "winner": {"variant": "dt_small"},
    })
    development_audit_path = root / "selection" / "independent_audit.json"
    _write(development_audit_path, {
        "experiment": spec["experiment"],
        "status": "PASS",
        "config_sha256": config_sha,
        "selection_sha256": _sha(selection_path),
        "winner": {"variant": "dt_small"},
    })
    gate = {
        "development_selection_sha256": _sha(selection_path),
        "development_audit_sha256": _sha(development_audit_path),
    }
    role_variants = {"baseline": "baseline", "selected_winner": "dt_small"}
    train_ordinals = sorted(spec["splits"]["train"])
    confirmation_ordinals = sorted(spec["splits"]["confirmation"])

    def identity(task: str, role: str, dataset: str, seed):
        return {
            "experiment": spec["experiment"],
            "phase": "outer_confirmation",
            "config_sha256": config_sha,
            **gate,
            "task": task,
            "confirmation_role": role,
            "variant": role_variants[role],
            "selected_winner": "dt_small",
            "dataset": dataset,
            "family": spec["families"][dataset],
            "seed": seed,
            "train_ordinals": train_ordinals,
            "confirmation_ordinals": confirmation_ordinals,
            "development_validation_opened": False,
            "confirmation_opened": True,
            "selection_performed_on_confirmation": False,
        }

    for role, variant in role_variants.items():
        advantage = 0.1 if role == "baseline" else 0.2
        for dataset in spec["datasets"]:
            rows = []
            for seed in spec["task1"]["confirmation_kmeans_seeds"]:
                rows.extend([
                    {"arm": "raw", "kmeans_seed": seed, "f1": 0.2},
                    {"arm": "fmt", "kmeans_seed": seed, "f1": 0.2 + advantage},
                ])
            _write(
                root / "confirmation" / "task1" / role / f"{dataset}.json",
                {**identity("Task1", role, dataset, None), "rows": rows},
            )
            for seed in spec["task2"]["confirmation_training_seeds"]:
                _write(
                    root / "confirmation" / "task2" / role / dataset
                    / f"seed{seed}.json",
                    {
                        **identity("Task2", role, dataset, seed),
                        "rows": [
                            {"arm": "raw", "f1": 0.3},
                            {"arm": "fmt", "f1": 0.3 + advantage},
                        ],
                    },
                )
            for seed in spec["task3"]["confirmation_training_seeds"]:
                residual = {
                    "parameter_count": 115266,
                    "trainable_residual_parameter_count": 25345,
                }
                _write(
                    root / "confirmation" / "task3" / role / dataset
                    / f"seed{seed}" / "result.json",
                    {
                        **identity("Task3", role, dataset, seed),
                        "rows": [
                            {"arm": "raw_backbone", "validation_f1": 0.2,
                             "validation_average_precision": 0.3},
                            {"arm": "raw_wide_guard", "parameter_count": 148225,
                             "validation_f1": 0.2,
                             "validation_average_precision": 0.3},
                            {"arm": "raw_pca", **residual,
                             "validation_f1": 0.4,
                             "validation_average_precision": 0.5},
                            {"arm": "fmt", **residual,
                             "validation_f1": 0.4 + advantage,
                             "validation_average_precision": 0.5 + advantage},
                        ],
                    },
                )

    summary_path = summarize(config)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["result_file_counts"] == {
        "Task1": 20, "Task2": 100, "Task3": 100,
    }
    assert abs(summary["winner_minus_baseline_equal_task_mean_f1_gain"] - 0.1) < 1e-12

    audit_path = audit(config)
    evidence = json.loads(audit_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    assert evidence["audited_result_count"] == 220

    checkpoint = (
        root / "confirmation" / "task3" / "baseline" / "channel"
        / "seed60" / "backbone" / "checkpoints" / "temporary.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"temporary")
    cleanup_path = cleanup(config)
    cleaned = json.loads(cleanup_path.read_text(encoding="utf-8"))
    assert cleaned["deleted_checkpoint_count"] == 1
    assert not checkpoint.exists()
