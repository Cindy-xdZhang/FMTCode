import json
from pathlib import Path

import yaml

from Build_Task2_VAE_Confirmation import build_confirmation_config


SWEEP = Path("config/Verify_Task2_VAEGrid_3D_3.1.yaml")


def _selection(satisfied):
    sweep = yaml.safe_load(SWEEP.read_text(encoding="utf-8"))
    return {
        "hierarchy_satisfied": satisfied,
        "task1_f1_mean": 0.60,
        "raw_f1_mean": 0.62,
        "fmt_f1_mean": 0.65,
        "minimum_hierarchy_margin": 0.02,
        "selected": {
            group: {"variant": sweep["variants"][0]["id"]}
            for group in sweep["groups"]
        },
    }


def test_confirmation_is_blocked_without_development_hierarchy(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(_selection(False)), encoding="utf-8")
    try:
        build_confirmation_config(SWEEP, selection, tmp_path / "final.yaml")
    except RuntimeError as error:
        assert "confirmation is forbidden" in str(error)
    else:
        raise AssertionError("confirmation was not blocked")


def test_confirmation_uses_new_seeds_and_same_family_vae(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(_selection(True)), encoding="utf-8")
    output = build_confirmation_config(SWEEP, selection, tmp_path / "final.yaml")
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["final_training_seeds"] == [8068, 8069, 8070, 8071, 8072]
    assert config["selection_provenance"]["confirmation_labels_used_for_selection"] is False
    assert len(config["groups"]) == 7
    assert all("fixed_architecture" in group for group in config["groups"].values())
