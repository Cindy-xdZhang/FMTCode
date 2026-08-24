import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Verify_Task3_FMTClassifier import _completion_message, _validate_config_snapshot


def test_only_seed_expansion_is_allowed():
    original = {
        "output_dir": "x", "model": {"width": 8},
        "training": {"seeds": [20], "max_epochs": 25},
    }
    with tempfile.TemporaryDirectory() as directory:
        snapshot = Path(directory) / "config.yaml"
        snapshot.write_text(yaml.safe_dump(original), encoding="utf-8")
        expanded = {
            **original,
            "training": {"seeds": [20, 21], "max_epochs": 25},
        }
        _validate_config_snapshot(expanded, snapshot)
        changed = {
            **original,
            "training": {"seeds": [20], "max_epochs": 60},
        }
        try:
            _validate_config_snapshot(changed, snapshot)
        except RuntimeError:
            pass
        else:
            raise AssertionError("changed training protocol was silently accepted")


def test_completion_message_supports_disabled_test_split():
    message = _completion_message(
        ("toy", "raw", 30),
        {"validation_f1": 0.625, "validation_average_precision": 0.75},
    )
    assert "validation F1=0.62500" in message
    assert "AP=0.75000" in message
    assert "test disabled" in message


if __name__ == "__main__":
    test_only_seed_expansion_is_allowed()
    test_completion_message_supports_disabled_test_split()
    print("TASK3 CLASSIFIER CONFIG GUARD TEST PASSED")
