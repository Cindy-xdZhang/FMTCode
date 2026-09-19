"""Behavioral checks of radius boundaries, expansion, ties and frozen feature compatibility."""
import unittest
import numpy as np
import torch
from FMT_Utils import Task4C_BallQuery_2_2 as ball
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as baseline
from FMT_Utils import Task4C_FPS16_1_1 as encoder


class BallQueryTests(unittest.TestCase):
    def test_minimum_spacing_is_not_geometric_mean(self):
        self.assertAlmostEqual(ball.minimum_grid_spacing([np.array([0, 2, 4]), np.array([0, .1, .5]), np.array([0, 1])]), .1)

    def test_sparse_ball_expands_only_as_far_as_needed(self):
        points = np.column_stack([np.arange(30.), np.zeros(30), np.zeros(30)])
        result = ball.ball_tables(points, .25)
        self.assertTrue(np.all(result['initial_count'] == 0))
        self.assertEqual(result['effective_radius6'][0], 6)
        self.assertEqual(result['effective_radius16'][0], 16)
        self.assertEqual(set(result['order6'][0]), set(range(1, 7)))
        self.assertEqual(set(result['order16'][0]), set(range(1, 17)))

    def test_dense_ball_fps_and_boundary_inclusion(self):
        rng = np.random.default_rng(3); points = rng.random((70, 3))
        result = ball.ball_tables(points, 2.)
        self.assertTrue(np.all(result['initial_count'] == 69))
        self.assertFalse(result['expanded6'].any())
        expected = ball.select_fps(points, points[0], np.arange(1, len(points)), 16)
        np.testing.assert_array_equal(result['order16'][0], expected)
        np.testing.assert_array_equal(result['order6'][0], expected[:6])
        points = np.column_stack([np.arange(30.), np.zeros(30), np.zeros(30)])
        result = ball.ball_tables(points, 6.)
        self.assertEqual(result['initial_count'][0], 6)
        self.assertFalse(result['expanded6'][0])
        self.assertTrue(result['expanded16'][0])

    def test_selection_is_deterministic_and_cannot_cross_effective_ball(self):
        points = np.random.default_rng(9).normal(size=(100, 3))
        a = ball.ball_tables(points, .8); b = ball.ball_tables(points, .8, workers=1)
        for k in (6, 16):
            np.testing.assert_array_equal(a[f'order{k}'], b[f'order{k}'])
            ids = a[f'order{k}']
            self.assertTrue(all(len(set(row)) == k for row in ids))
            self.assertTrue(np.all(np.linalg.norm(points[ids]-points[:, None], axis=2) <= a[f'effective_radius{k}'][:, None]+1e-12))

    def test_duplicate_seeds_fail(self):
        points = np.zeros((30, 3))
        with self.assertRaises(ValueError): ball.ball_tables(points, 1.)

    def test_per_sample_radius_and_nearest_neighbor_equivalence_when_sparse(self):
        from scipy.spatial import cKDTree
        points = np.random.default_rng(7).random((60, 3))
        radius = np.linspace(.001, .002, len(points))
        result = ball.ball_tables(points, radius)
        _, expected = cKDTree(points).query(points, k=17)
        for k in (6, 16):
            np.testing.assert_array_equal(np.sort(result[f'order{k}'], axis=1), np.sort(expected[:, 1:k+1], axis=1))
        self.assertTrue(np.all(result['expanded16']))

    def test_selected_clusters_keep_frozen_single_center_features(self):
        points = np.random.default_rng(13).random((40, 3))
        tables = ball.ball_tables(points, .3)
        t = np.linspace(-1, 1, 32)
        curves = points[:, None]+np.stack([.03*t, .02*t*t, .01*t**3], axis=1)[None]
        for k in (6, 16):
            ids = np.column_stack([np.arange(4), tables[f'order{k}'][:4]])
            geometry, seeds, counts = baseline.normalize_bundle(torch.tensor(curves[ids], dtype=torch.float32), torch.tensor(points[ids]))
            for name in ('p35_h0', 'c156'):
                token, used = encoder.encode(geometry, seeds, counts, torch.zeros(4, dtype=torch.long), dict(base_method=name, neighbors=k))
                self.assertEqual(tuple(token.shape), (4, 1, 142))
                self.assertTrue(torch.isfinite(token).all())
                torch.testing.assert_close(used.sort(1).values, torch.arange(1, k+1)[None].expand(4, k))


if __name__ == '__main__': unittest.main()
