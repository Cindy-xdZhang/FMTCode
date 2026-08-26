import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Build_Task3_GlobalIVD_Labels import build


def test_global_ivd_labels_are_copied_exactly():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source" / "toy"
        source.mkdir(parents=True)
        reference = np.asarray([False, True, False, True])
        metadata = {
            "source_start_index": 17,
            "ivd_threshold": 2.5,
        }
        np.savez_compressed(
            source / "slice_000.npz",
            reference=reference,
            seeds=np.zeros((4, 3), dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        spec = {
            "experiment": "test",
            "source_cache_root": str(root / "source"),
            "output_dir": str(root / "labels"),
            "datasets": ["toy"],
            "expected_slices": 1,
            "label": {"definition": "standard_global_ivd", "percentile": 95.0},
        }
        config = root / "config.yaml"
        config.write_text(yaml.safe_dump(spec), encoding="utf-8")
        output = build(str(config))
        with np.load(output / "toy" / "slice_000.npz") as result:
            assert np.array_equal(result["labels"], reference)
            assert np.all(result["threshold_at_seeds"] == np.float32(2.5))
            saved = json.loads(str(result["metadata_json"]))
        assert saved["label_mode"] == "standard_global_ivd_percentile"
        assert saved["copied_exactly_from_source_reference"] is True


if __name__ == "__main__":
    test_global_ivd_labels_are_copied_exactly()
    print("TASK3 GLOBAL IVD LABEL TEST PASSED")
