import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_task3_label_cache_contract(path=None):
    if path is None:
        return
    with np.load(path) as data:
        labels = data["labels"]
        ivd = data["ivd_at_seeds"]
        threshold = data["threshold_at_seeds"]
        metadata = json.loads(str(data["metadata_json"]))
    assert labels.dtype == np.bool_ and labels.ndim == 1
    assert labels.shape == ivd.shape == threshold.shape
    assert metadata["sample_count"] == len(labels)
    assert metadata["positive_count"] == int(labels.sum())
    assert np.array_equal(labels, ivd > threshold)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    test_task3_label_cache_contract(args.path)
    print("TASK3 LABEL CACHE TEST PASSED")
