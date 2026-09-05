import numpy as np

from FMT_Utils.Task4B_CrossFlow_3D import (
    NonPeriodicStructuredVelocityField3D,
    apply_source_normalization,
    dimensionless_primitive_features,
    finite_difference_curl_zyx,
    fit_source_normalization,
)


def test_nonuniform_grid_curl_matches_quadratic_analytic_solution():
    zs = np.asarray([0.0, 0.15, 0.7, 1.6, 2.4], dtype=np.float64)
    ys = np.asarray([-0.8, -0.1, 0.35, 1.2], dtype=np.float64)
    xs = np.asarray([-1.0, -0.35, 0.2, 1.1, 2.0], dtype=np.float64)
    z, y, x = np.meshgrid(zs, ys, xs, indexing="ij")

    # u = y^2 + 2z, v = z^2 + 3x, w = x^2 + 4y.
    velocity = np.stack(
        (y * y + 2.0 * z, z * z + 3.0 * x, x * x + 4.0 * y),
        axis=-1,
    ).astype(np.float32)
    expected = np.stack(
        (
            np.broadcast_to(4.0 - 2.0 * z, z.shape),
            np.broadcast_to(2.0 - 2.0 * x, x.shape),
            np.broadcast_to(3.0 - 2.0 * y, y.shape),
        ),
        axis=-1,
    ).astype(np.float32)

    observed = finite_difference_curl_zyx(velocity, (zs, ys, xs))
    np.testing.assert_allclose(observed, expected, rtol=3.0e-5, atol=3.0e-5)


def test_nonperiodic_interpolation_returns_nan_outside_each_axis():
    zs = np.asarray([0.0, 0.4, 1.2], dtype=np.float64)
    ys = np.asarray([-1.0, -0.2, 0.7], dtype=np.float64)
    xs = np.asarray([2.0, 2.5, 3.4], dtype=np.float64)
    z, y, x = np.meshgrid(zs, ys, xs, indexing="ij")
    velocity = np.stack((x, y, z), axis=-1).astype(np.float32)
    vorticity = np.stack((z - y, x - z, y - x), axis=-1).astype(np.float32)
    omega_y_prime = (x + 2.0 * y - z).astype(np.float32)
    field = NonPeriodicStructuredVelocityField3D(
        axes_zyx=(zs, ys, xs),
        velocity_zyx3=velocity,
        vorticity_zyx3=vorticity,
        omega_y_prime_zyx=omega_y_prime,
    )

    inside = np.asarray([[2.6, 0.0, 0.5]], dtype=np.float64)
    outside = np.asarray(
        [
            [xs[-1] + 0.01, 0.0, 0.5],
            [2.6, ys[0] - 0.01, 0.5],
            [2.6, 0.0, zs[-1] + 0.01],
        ],
        dtype=np.float64,
    )
    assert np.isfinite(field.velocity(inside)).all()
    assert np.isfinite(field.vorticity(inside)).all()
    assert np.isfinite(field.omega_y_prime(inside)).all()
    assert np.isnan(field.velocity(outside)).all()
    assert np.isnan(field.vorticity(outside)).all()
    assert np.isnan(field.omega_y_prime(outside)).all()


def test_dimensionless_raw_and_fmt_are_invariant_to_translation_and_matched_scale():
    rng = np.random.default_rng(7068)
    increments = rng.integers(-3, 4, size=(3, 7, 9, 3)).astype(np.float32) / 8.0
    primitives = np.cumsum(increments, axis=2, dtype=np.float32)
    cross_offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, -0.25, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, -0.25],
        ],
        dtype=np.float32,
    )
    primitives += cross_offsets[None, :, None, :]
    options = {
        "num_freq": 4,
        "neighbor_scale": 2.0,
        "neighbor_pool": "sort",
        "mode": "gram",
        "include_chirality": True,
    }

    raw, fmt = dimensionless_primitive_features(
        primitives, spatial_step=0.25, **options
    )
    scale = np.float32(4.0)
    transformed = primitives * scale + np.asarray(
        [8.0, -12.0, 20.0], dtype=np.float32
    )
    transformed_raw, transformed_fmt = dimensionless_primitive_features(
        transformed, spatial_step=0.25 * float(scale), **options
    )

    np.testing.assert_allclose(transformed_raw, raw, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(transformed_fmt, fmt, rtol=1.0e-6, atol=1.0e-6)


def test_source_only_normalization_is_unchanged_by_target_application():
    sampled_steps = 3
    source_raw = (
        np.arange(4 * 7 * sampled_steps * 3, dtype=np.float32)
        .reshape(4, -1)
        / 11.0
    )
    source_fmt = (
        np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 7.0
    )
    normalization = fit_source_normalization(
        source_raw, source_fmt, sampled_steps=sampled_steps
    )
    frozen = {name: value.copy() for name, value in normalization.items()}
    source_before = apply_source_normalization(
        source_raw,
        source_fmt,
        normalization,
        sampled_steps=sampled_steps,
        fmt_base_width=3,
        neighbor_weight=0.25,
    )

    target_raw = np.full((2, source_raw.shape[1]), 1.0e6, dtype=np.float32)
    target_fmt = np.full((2, source_fmt.shape[1]), -1.0e6, dtype=np.float32)
    apply_source_normalization(
        target_raw,
        target_fmt,
        normalization,
        sampled_steps=sampled_steps,
        fmt_base_width=3,
        neighbor_weight=0.25,
    )
    source_after = apply_source_normalization(
        source_raw,
        source_fmt,
        normalization,
        sampled_steps=sampled_steps,
        fmt_base_width=3,
        neighbor_weight=0.25,
    )

    for name, expected in frozen.items():
        np.testing.assert_array_equal(normalization[name], expected)
    np.testing.assert_array_equal(source_after[0], source_before[0])
    np.testing.assert_array_equal(source_after[1], source_before[1])
