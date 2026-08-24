import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Verify_Task3_FMTClassifier import _validate_config_snapshot


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


if __name__ == "__main__":
    test_only_seed_expansion_is_allowed()
    print("TASK3 CLASSIFIER CONFIG GUARD TEST PASSED")
