import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Verify_HighReSampling3D import _load_common_records
from FMT_Utils.DFT_FMT_3D import fmt_feature_indices_3d


def test_common_record_loader_symbol_is_available():
    assert callable(_load_common_records)


def test_fmt_semantic_feature_indices_have_expected_widths():
    assert fmt_feature_indices_3d("real_neighbor").shape == (36,)
    assert fmt_feature_indices_3d("real_all").shape == (42,)
    assert fmt_feature_indices_3d("all").shape == (161,)


if __name__ == "__main__":
    test_common_record_loader_symbol_is_available()
    test_fmt_semantic_feature_indices_have_expected_widths()
    print("HIGH-RE SAMPLING TEST PASSED")
