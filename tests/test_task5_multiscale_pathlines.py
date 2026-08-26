import numpy as np

from FLowUtils.VectorField3d import UnsteadyVectorField3D
from FMT_Utils.MultiscalePathline_3D import (
    balanced_scale_assignment,
    integrate_multiscale_primitives_3d,
    parse_scale_table,
)


def _zero_field():
    field = UnsteadyVectorField3D(
        9, 9, 9, 17,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        0.0, 2.0,
    )
    field.field = np.zeros((17, 9, 9, 9, 3), dtype=np.float32)
    return field


def test_balanced_assignment_is_reproducible_and_position_shuffled():
    first = balanced_scale_assignment(101, 9, 123)
    second = balanced_scale_assignment(101, 9, 123)
    np.testing.assert_array_equal(first, second)
    counts = np.bincount(first, minlength=9)
    assert counts.max() - counts.min() == 1
    assert not np.array_equal(first, np.arange(101) % 9)


def test_variable_scales_keep_fixed_tensor_shape_and_exact_initial_offsets():
    scales = parse_scale_table([
        {"name": "small", "offset_grid_scale": 0.5,
         "dt_scale": 0.25, "integration_steps": 31},
        {"name": "large", "offset_grid_scale": 1.5,
         "dt_scale": 0.125, "integration_steps": 40},
    ], sampled_steps=32)
    seeds = np.asarray([
        [0.35, 0.35, 0.35], [0.65, 0.35, 0.35],
        [0.35, 0.65, 0.35], [0.65, 0.65, 0.35],
        [0.35, 0.35, 0.65], [0.65, 0.65, 0.65],
    ])
    assignment = np.asarray([0, 1, 0, 1, 0, 1])
    result = integrate_multiscale_primitives_3d(
        _zero_field(), seeds, 0.0, scales, assignment, 32, chunk_size=64
    )
    assert result["valid_mask"].all()
    assert result["primitives"].shape == (6, 7, 32, 4)
    assert result["line_lengths"].shape == (6, 7)
    np.testing.assert_array_equal(result["scale_id"], assignment)

    starts = result["primitives"][:, :, 0, :3]
    distances = np.linalg.norm(starts[:, 1:] - starts[:, :1], axis=-1)
    expected = np.repeat(result["primitive_offset"][:, None], 6, axis=1)
    np.testing.assert_allclose(distances, expected, rtol=0, atol=5e-8)
    np.testing.assert_allclose(
        result["integration_horizon"],
        result["physical_dt"] * result["integration_steps"],
    )


def test_scale_table_rejects_too_few_integration_steps():
    try:
        parse_scale_table([
            {"name": "bad", "offset_grid_scale": 1.0,
             "dt_scale": 0.25, "integration_steps": 20},
        ], sampled_steps=32)
    except ValueError as error:
        assert "cannot provide" in str(error)
    else:
        raise AssertionError("invalid scale table was accepted")
