import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Run_Task2_Universality import _prepare


def test_fmt_weight_is_applied_after_training_scaler():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(8, 30)).astype(np.float32)
    train, test = _prepare(values, train_lengths=[6], fmt_weight=0.5)
    np.testing.assert_allclose(train[:, :23].std(axis=0), 1.0, atol=1e-5)
    np.testing.assert_allclose(train[:, 23:].std(axis=0), 0.5, atol=1e-5)
    assert test.shape == (2, 30)


if __name__ == "__main__":
    test_fmt_weight_is_applied_after_training_scaler()
    print("TASK2 UNIVERSALITY TEST PASSED")
