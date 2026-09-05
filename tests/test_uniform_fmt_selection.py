import json
import hashlib
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import yaml

from experiments.Audit_UniformFMT_3D import (
    _task2_key as independent_task2_key,
    _task3_key as independent_task3_key,
)
from experiments.Audit_Task1_Uniform_4_1 import _expected_selection_keys
from experiments.Run_Task1_3D_Uniform import _development, _leaderboard
from experiments.Search_Task2_FMTVAE_3D import _global_candidate as task2_global
from experiments.Search_Task3_FMTResidual_3D import (
    _global_candidate as task3_global,
    _validate_screen_seed_contract,
)
from experiments.Run_UniformFMT_Confirmation_3D import (
    _load_confirmation_spec,
    freeze_recipe,
)


def test_task1_global_leaderboard_is_dataset_macro_not_family_macro():
    spec = {"datasets": ["a", "b", "c"]}
    rows = []
    for feature, scores in (("fmt_a", [0.9, 0.61, 0.61]), ("fmt_b", [0.7, 0.7, 0.7])):
        for dataset, score in zip(spec["datasets"], scores):
            rows.append({
                "dataset": dataset,
                "feature": feature,
                "pca_dim": "2",
                "f1": score,
                "ari": score / 2.0,
            })
    board = _leaderboard(spec, rows, "fmt")
    assert board[0]["feature"] == "fmt_a"
    assert math.isclose(board[0]["dataset_macro_f1"], 0.7066666667, abs_tol=1e-9)


def test_task1_development_opens_only_authorized_ordinals():
    spec = {
        "development_count": 10,
        "datasets": ["flow"],
        "sources": {
            "source": {
                "datasets": ["flow"],
                "development_cache": "cache",
            }
        },
    }
    sentinel = [{"ordinal": 0}]
    with patch(
        "experiments.Run_Task1_3D_Uniform.load_cache_records",
        return_value=sentinel,
    ) as mocked:
        assert _development(spec, [0, 2, 4]) == {"flow": sentinel}
    mocked.assert_called_once_with(
        Path("cache") / "flow", expected_count=10, ordinals=[0, 2, 4]
    )


def test_task1_audit_counts_only_executable_frozen_candidates():
    import yaml

    spec = yaml.safe_load(
        Path("config/mainExp_Task1_3D_4.1_uniform.yaml").read_text(
            encoding="utf-8"
        )
    )
    keys = _expected_selection_keys(spec)
    assert len(keys) == 46
    assert ("raw", "none") not in keys
    assert ("kin2", "16") not in keys
    assert ("kin2", "8") in keys
    assert ("fmt_all+kin4", "8") in keys


def test_task2_global_candidate_weights_datasets_equally():
    spec = {
        "groups": {"single": {}, "triple": {}},
        "datasets": ["a", "b", "c", "d"],
        "screen_seeds": [1, 2],
    }
    rows = {
        "single": {
            "dataset_count": 1,
            "datasets_json": json.dumps({
                "a": {"raw_f1": 0.1, "fmt_f1": 0.9,
                      "fmt_minus_raw_f1": 0.8}
            }),
            "seed_gains_json": json.dumps({"1": 0.8, "2": 0.8}),
        },
        "triple": {
            "dataset_count": 3,
            "datasets_json": json.dumps({
                name: {"raw_f1": 0.4, "fmt_f1": 0.5,
                       "fmt_minus_raw_f1": 0.1}
                for name in ("b", "c", "d")
            }),
            "seed_gains_json": json.dumps({"1": 0.1, "2": 0.1}),
        },
    }
    with patch(
        "experiments.Search_Task2_FMTVAE_3D._paired_candidate",
        side_effect=lambda _spec, group, _feature, _arch: rows[group],
    ):
        result = task2_global(
            spec, {"id": "f", "name": "fmt"}, {"id": "vae"}
        )
    assert math.isclose(result["fmt_minus_raw_f1_macro"], 0.275, abs_tol=1e-12)
    assert math.isclose(result["worst_seed_f1_gain"], 0.275, abs_tol=1e-12)
    assert result["positive_dataset_count"] == 4


def test_task3_global_candidate_weights_datasets_equally():
    spec = {
        "groups": {"single": {}, "triple": {}},
        "datasets": ["a", "b", "c", "d"],
        "screen_seeds": [1, 2],
    }

    def dataset(fmt_f1, raw_f1, fmt_ap, raw_ap):
        return {
            "fmt": {"f1": fmt_f1, "average_precision": fmt_ap},
            "raw_pca": {"f1": raw_f1, "average_precision": raw_ap},
            "strong_raw": {"f1": raw_f1 - 0.05,
                           "average_precision": raw_ap - 0.05},
            "gains": {
                "f1_vs_raw_pca": fmt_f1 - raw_f1,
                "average_precision_vs_raw_pca": fmt_ap - raw_ap,
            },
        }

    rows = {
        "single": {
            "dataset_count": 1,
            "datasets_json": json.dumps({"a": dataset(0.9, 0.1, 0.8, 0.2)}),
            "seed_gains_json": json.dumps({"1": 0.8, "2": 0.8}),
        },
        "triple": {
            "dataset_count": 3,
            "datasets_json": json.dumps({
                name: dataset(0.6, 0.5, 0.7, 0.6)
                for name in ("b", "c", "d")
            }),
            "seed_gains_json": json.dumps({"1": 0.1, "2": 0.1}),
        },
    }
    with patch(
        "experiments.Search_Task3_FMTResidual_3D._candidate_summary",
        side_effect=lambda _spec, group, _candidate: rows[group],
    ):
        result = task3_global(spec, {"id": "c", "fmt_feature": "fmt"})
    assert math.isclose(result["fmt_f1_macro"], 0.675, abs_tol=1e-12)
    assert math.isclose(
        result["fmt_minus_raw_pca_f1_macro"], 0.275, abs_tol=1e-12
    )
    assert math.isclose(result["worst_seed_f1_gain"], 0.275, abs_tol=1e-12)
    assert result["positive_f1_dataset_count"] == 4


def test_task3_screen_seeds_require_declared_frozen_raw_backbones():
    _validate_screen_seed_contract({
        "screen_seeds": [40, 41, 42],
        "available_raw_checkpoint_seeds": [40, 41, 42, 43, 44],
    })

    try:
        _validate_screen_seed_contract({
            "screen_seeds": [93, 94, 95],
            "available_raw_checkpoint_seeds": [40, 41, 42, 43, 44],
        })
    except ValueError as error:
        assert "93" in str(error) and "frozen Raw checkpoints" in str(error)
    else:
        raise AssertionError("missing Raw checkpoint seeds must be rejected")


def test_independent_task2_audit_uses_registered_paired_gain_order():
    larger_gain = {
        "fmt_minus_raw_f1_macro": 0.2,
        "worst_seed_f1_gain": 0.1,
        "fmt_f1_macro": 0.6,
    }
    larger_absolute_f1 = {
        "fmt_minus_raw_f1_macro": 0.1,
        "worst_seed_f1_gain": 0.2,
        "fmt_f1_macro": 0.9,
    }
    assert independent_task2_key(larger_gain) > independent_task2_key(
        larger_absolute_f1
    )


def test_independent_task3_audit_uses_registered_absolute_fmt_order():
    larger_absolute_f1 = {
        "fmt_f1_macro": 0.9,
        "fmt_minus_raw_pca_f1_macro": 0.1,
        "fmt_ap_macro": 0.8,
        "worst_dataset_f1_gain": 0.01,
        "worst_seed_f1_gain": 0.02,
    }
    larger_gain = {
        "fmt_f1_macro": 0.8,
        "fmt_minus_raw_pca_f1_macro": 0.2,
        "fmt_ap_macro": 0.9,
        "worst_dataset_f1_gain": 0.1,
        "worst_seed_f1_gain": 0.2,
    }
    assert independent_task3_key(larger_absolute_f1) > independent_task3_key(
        larger_gain
    )


def test_uniform_confirmation_configs_partition_outer_ordinals():
    for path, task, calibration_name in (
        (
            "config/mainExp_Task2_3D_6.2_uniform_confirmation.yaml",
            "Task2",
            "cluster_calibration",
        ),
        (
            "config/mainExp_Task3_3D_9.2_uniform_confirmation.yaml",
            "Task3",
            "validation",
        ),
    ):
        _, spec = _load_confirmation_spec(path)
        assert spec["task"] == task
        assert spec["splits"]["train"] == [0, 1, 2, 3, 4, 5]
        assert spec["splits"][calibration_name] == [6, 7]
        assert spec["splits"]["confirmation"] == [8, 9]
        assert len(spec["training_seeds"]) >= 3


def test_uniform_selection_configs_have_no_family_recipe_overrides():
    cases = (
        (
            "config/Verify_Task2_UniformFMT_6.1.yaml",
            {"datasets", "development_cache", "source_config"},
        ),
        (
            "config/Verify_Task3_UniformFMT_9.1.yaml",
            {
                "datasets", "source_cache_root", "label_cache_root",
                "raw_checkpoint_dir",
            },
        ),
    )
    for path, allowed_group_keys in cases:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        grouped = []
        for group_name, group in spec["groups"].items():
            unexpected = set(group).difference(allowed_group_keys)
            assert not unexpected, (
                f"{path} group {group_name} contains recipe overrides: "
                f"{sorted(unexpected)}"
            )
            grouped.extend(group["datasets"])
        assert len(grouped) == len(set(grouped))
        assert set(grouped) == set(spec["datasets"])


def test_confirmation_recipe_requires_unchanged_independently_audited_selection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_path = root / "selection.yaml"
        selection_path = root / "global_selection.json"
        audit_path = root / "audit.json"
        output = root / "confirmation"
        source = {
            "experiment": "Verify_Task2_Dummy",
            "task": "Task2",
            "datasets": ["flow"],
        }
        source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
        selection = {
            "experiment": source["experiment"],
            "outer_ordinals_opened": False,
            "confirmation_opened": False,
            "winner": {
                "architecture": "linear",
                "feature_id": "f",
                "fmt_feature": "fmt",
            },
        }
        selection_path.write_text(
            json.dumps(selection, sort_keys=True), encoding="utf-8"
        )

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        audit = {
            "status": "PASS",
            "audit_is_independent_of_formal_selector": True,
            "outer_or_confirmation_data_opened": False,
            "config_sha256": digest(source_path),
            "evidence_sha256": {
                selection_path.as_posix(): digest(selection_path)
            },
        }
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        confirmation = {
            "experiment": "mainExp_Task2_Dummy",
            "task": "Task2",
            "selection_config": str(source_path),
            "selection_path": str(selection_path),
            "selection_audit_path": str(audit_path),
            "output_root": str(output),
            "splits": {
                "train": [0, 1, 2, 3, 4, 5],
                "cluster_calibration": [6, 7],
                "confirmation": [8, 9],
            },
            "training_seeds": [100, 101, 102],
            "expected_slices": 10,
            "checkpoint_policy": "none",
        }
        confirmation_path = root / "confirmation.yaml"
        confirmation_path.write_text(
            yaml.safe_dump(confirmation, sort_keys=False), encoding="utf-8"
        )
        manifest_path = freeze_recipe(confirmation_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["winner"] == selection["winner"]
        assert manifest["selection_audit_status"] == "PASS"

        selection["winner"]["fmt_feature"] = "changed_after_freeze"
        selection_path.write_text(
            json.dumps(selection, sort_keys=True), encoding="utf-8"
        )
        try:
            freeze_recipe(confirmation_path)
        except RuntimeError as error:
            assert "selection JSON" in str(error)
        else:
            raise AssertionError("selection changes after audit must be rejected")
