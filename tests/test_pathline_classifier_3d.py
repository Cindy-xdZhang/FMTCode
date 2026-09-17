import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D, PathlineFMTResidualClassifier3D,
    trainable_parameter_count,
)
from FMT_Utils.FMTNoConvolution_1_1 import ForbiddenFMTConvolutionError


def test_shapes_capacity_and_neighbour_permutation():
    torch.manual_seed(7)
    pathlines = torch.randn(4, 7, 32, 3)
    fmt = torch.randn(4, 161)
    models = {
        name: PathlineBinaryClassifier3D(name, fmt_dim=161).eval()
        for name in ("raw", "raw_wide")
    }
    for name, model in models.items():
        output = model(pathlines, fmt if name == "raw_fmt" else None)
        assert output.shape == (4,)
        permuted = pathlines[:, [0, 4, 2, 6, 1, 5, 3]]
        output_permuted = model(permuted, fmt if name == "raw_fmt" else None)
        assert torch.allclose(output, output_permuted, atol=1e-6)
    assert trainable_parameter_count(models["raw_wide"]) > trainable_parameter_count(models["raw"])

    with pytest.raises(ForbiddenFMTConvolutionError):
        PathlineBinaryClassifier3D('raw_fmt')
    with pytest.raises(ForbiddenFMTConvolutionError):
        PathlineFMTResidualClassifier3D(models["raw"], fmt_dim=161)


if __name__ == "__main__":
    test_shapes_capacity_and_neighbour_permutation()
    print("PATHLINE CLASSIFIER 3D TEST PASSED")
