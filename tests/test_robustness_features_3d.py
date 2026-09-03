import unittest

import numpy as np

from FMT_Utils.RobustnessFeatures_3D import (
    FMT_KIN4_WIDTH,
    GEOMETRIC_STATISTICS_PER_LINE,
    corrupt_pathline_primitives_3d,
    fmt_kin4_ablation_mask,
    non_fmt_feature_matrix,
    pair_distance_dft_features_3d,
    pathline_geometric_quantities_3d,
    pathline_geometric_sequences_3d,
    pathline_geometric_statistics_3d,
    plain_pathline_dft_features_3d,
    reshape_cached_primitives,
)


class RobustnessFeatures3DTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        base = rng.normal(size=(9, 7, 32, 3)).astype(np.float32)
        base[:, :, 0] += np.arange(7, dtype=np.float32)[None, :, None]
        self.pathlines = base

    def test_baseline_feature_shapes_are_fixed_and_finite(self):
        complex_dft = plain_pathline_dft_features_3d(
            self.pathlines, num_freq=6, mode="complex"
        )
        magnitude_dft = plain_pathline_dft_features_3d(
            self.pathlines, num_freq=6, mode="magnitude"
        )
        pair_dft = pair_distance_dft_features_3d(self.pathlines, num_freq=6)
        self.assertEqual(complex_dft.shape, (9, 7 * 3 * 11))
        self.assertEqual(magnitude_dft.shape, (9, 7 * 3 * 6))
        self.assertEqual(pair_dft.shape, (9, 21 * 11))
        self.assertTrue(np.isfinite(complex_dft).all())
        self.assertTrue(np.isfinite(magnitude_dft).all())
        self.assertTrue(np.isfinite(pair_dft).all())

    def test_pair_distance_dft_is_rigid_motion_invariant(self):
        angle = 0.7
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        transformed = self.pathlines @ rotation.T + np.asarray(
            [3.0, -2.0, 0.5], dtype=np.float32
        )
        expected = pair_distance_dft_features_3d(self.pathlines)
        actual = pair_distance_dft_features_3d(transformed)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_component_masks_keep_width_and_remove_expected_counts(self):
        expected_zeros = {
            "full": 0,
            "without_chirality": 35,
            "without_cosine": 42,
            "without_neighbor_fourier": 138,
            "without_center_fourier": 23,
            "without_kinematic": 28,
            "kinematic_only": 161,
        }
        for name, zero_count in expected_zeros.items():
            with self.subTest(name=name):
                mask = fmt_kin4_ablation_mask(name)
                self.assertEqual(mask.shape, (FMT_KIN4_WIDTH,))
                self.assertEqual(int(np.count_nonzero(mask == 0.0)), zero_count)
                self.assertTrue(np.all(np.isin(mask, (0.0, 1.0))))

    def test_corruptions_are_deterministic_and_preserve_contract(self):
        clean = corrupt_pathline_primitives_3d(
            self.pathlines, "clean", 0.0, spatial_scale=0.2,
            random_state=4,
        )
        np.testing.assert_array_equal(clean, self.pathlines)
        gaussian_a = corrupt_pathline_primitives_3d(
            self.pathlines, "gaussian", 0.1, spatial_scale=0.2,
            random_state=4,
        )
        gaussian_b = corrupt_pathline_primitives_3d(
            self.pathlines, "gaussian", 0.1, spatial_scale=0.2,
            random_state=4,
        )
        np.testing.assert_array_equal(gaussian_a, gaussian_b)
        self.assertFalse(np.array_equal(gaussian_a, self.pathlines))
        dropped = corrupt_pathline_primitives_3d(
            self.pathlines, "frame_dropout", 0.4, spatial_scale=0.2,
            random_state=9,
        )
        np.testing.assert_array_equal(dropped[:, :, 0], self.pathlines[:, :, 0])
        np.testing.assert_array_equal(dropped[:, :, -1], self.pathlines[:, :, -1])
        shortened = corrupt_pathline_primitives_3d(
            self.pathlines, "short_track", 0.5, spatial_scale=0.2,
            random_state=9,
        )
        self.assertEqual(shortened.shape, self.pathlines.shape)
        np.testing.assert_allclose(
            shortened[:, :, -1],
            0.5 * (self.pathlines[:, :, 15] + self.pathlines[:, :, 16]),
            rtol=1e-6, atol=1e-6,
        )
        self.assertEqual(
            reshape_cached_primitives(shortened.reshape(9, -1)).shape,
            self.pathlines.shape,
        )

    def _rigid(self, pathlines, reflect=False):
        angle = 0.7
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, -1.0 if reflect else 1.0],
        ], dtype=np.float64)
        moved = pathlines.astype(np.float64) @ rotation.T + np.asarray([3.0, -2.0, 0.5])
        return moved.astype(np.float32)

    def test_geometric_baselines_have_fixed_width_and_are_finite(self):
        statistics = pathline_geometric_statistics_3d(self.pathlines)
        sequences = pathline_geometric_sequences_3d(self.pathlines)
        self.assertEqual(statistics.shape, (9, 7 * GEOMETRIC_STATISTICS_PER_LINE))
        self.assertEqual(sequences.shape, (9, 7 * (31 + 30 + 29)))
        self.assertTrue(np.isfinite(statistics).all())
        self.assertTrue(np.isfinite(sequences).all())
        np.testing.assert_array_equal(
            non_fmt_feature_matrix(self.pathlines.reshape(9, -1), "geometric_statistics"),
            statistics,
        )
        np.testing.assert_array_equal(
            non_fmt_feature_matrix(self.pathlines.reshape(9, -1), "geometric_sequences"),
            sequences,
        )

    def test_geometric_quantities_match_analytic_helix_and_are_rigid_invariant(self):
        # Helix x=cos t, y=sin t, z=c t has curvature 1/(1+c^2), torsion c/(1+c^2).
        c = 0.5
        t = np.linspace(0.0, 0.6, 32)
        helix = np.stack((np.cos(t), np.sin(t), c * t), axis=-1)
        pathlines = np.repeat(helix[None, None], 7, axis=1).astype(np.float32)
        quantities = pathline_geometric_quantities_3d(pathlines)
        np.testing.assert_allclose(
            quantities["curvature"], 1.0 / (1.0 + c * c), rtol=2e-3, atol=2e-3
        )
        # Torsion uses one more finite difference than curvature, so its
        # discretisation error is first order in the sample spacing.
        np.testing.assert_allclose(
            quantities["torsion"], c / (1.0 + c * c), rtol=3e-2, atol=0.0
        )
        moved = pathline_geometric_statistics_3d(self._rigid(self.pathlines))
        np.testing.assert_allclose(
            moved, pathline_geometric_statistics_3d(self.pathlines), rtol=1e-3, atol=1e-3
        )

    def test_signed_torsion_flips_under_reflection(self):
        reference = pathline_geometric_quantities_3d(self.pathlines)
        reflected = pathline_geometric_quantities_3d(self._rigid(self.pathlines, reflect=True))
        np.testing.assert_allclose(reflected["speed"], reference["speed"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            reflected["curvature"], reference["curvature"], rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            reflected["torsion"], -reference["torsion"], rtol=1e-2, atol=1e-2
        )

    def test_stagnant_lines_give_zero_curvature_and_torsion(self):
        still = np.zeros((2, 7, 32, 3), dtype=np.float32)
        still[:, 0, :, 0] = np.linspace(0.0, 1.0, 32)  # only the center line moves
        quantities = pathline_geometric_quantities_3d(still)
        self.assertTrue(np.isfinite(quantities["curvature"]).all())
        np.testing.assert_array_equal(quantities["curvature"][:, 1:], 0.0)
        np.testing.assert_array_equal(quantities["torsion"], 0.0)


if __name__ == "__main__":
    unittest.main()

