import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Diagnose_VAE_HighRe import _block_indices


blocks = _block_indices(161)
assert len(blocks["center"]) == 23
assert len(blocks["neighbors"]) == 138
assert len(blocks["real_norm"]) == 42
assert len(blocks["imag_norm"]) == 42
assert len(blocks["cosine"]) == 42
assert len(blocks["chirality"]) == 35
assert np.array_equal(np.sort(np.r_[blocks["center"], blocks["neighbors"]]), np.arange(161))
assert np.array_equal(np.sort(np.r_[blocks["real_norm"], blocks["imag_norm"],
                                      blocks["cosine"], blocks["chirality"]]), np.arange(161))

print("VAE HIGH-RE DIAGNOSIS TEST PASSED")
