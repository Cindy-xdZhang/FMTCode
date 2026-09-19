"""Exact neighborhood equivalence, 6/16 export correctness, and single-seed scope."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from FMT_Utils import Task4C_BallQuery_2_2 as reference
from FMT_Utils import Task4C_BallQuery_2_5 as ball
from experiments import Task4C_BallQueryBaselines_2_5 as baseline


class BallQuery25Tests(unittest.TestCase):
    def test_acceleration_preserves_full_ball_fps_and_expansion(self):
        points = np.random.default_rng(96611).normal(size=(180, 3))
        for radius in (.01, .9, 10.):
            a = reference.ball_tables(points, radius); b = ball.ball_tables(points, radius)
            for key in a: np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        points = np.column_stack([np.arange(40.), np.zeros(40), np.zeros(40)])
        for radius in (6., 16.):
            a = reference.ball_tables(points, radius); b = ball.ball_tables(points, radius)
            for key in a: np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_baselines_export_same_selected_sample_ids_at_each_length(self):
        rng = np.random.default_rng(5); seeds = rng.random((40, 3)); curves = rng.normal(size=(40, 3, 32, 3)).astype(np.float32)
        meta = dict(label=np.arange(40)%2, component=np.arange(40), instance=np.arange(40)//2,
            half_arc_lengths=np.ones((40,3,2)), half_step_counts=np.ones((40,3,2), np.int32), local_grid_scale=np.ones(40))
        tables = ball.ball_tables(seeds, .5); rows = np.array([0, 4, 10]); lengths = np.array([0, 1, 2])
        for k in (6, 16):
            neighbors = tables[f'order{k}']; ids = np.column_stack([rows, neighbors[rows]])
            g, s, m = baseline.bundle_function(k)(curves, seeds, meta, neighbors, rows, lengths, [70,100,130], .001)
            expected, es, _ = baseline.frozen.v2.normalize_bundle(torch.tensor(curves[ids, lengths[:,None]]), torch.tensor(seeds[ids], dtype=torch.float32))
            np.testing.assert_array_equal(g[:, :k+1], expected.numpy()); np.testing.assert_array_equal(s[:, :k+1], es.numpy())
            np.testing.assert_array_equal(m['labels'], meta['label'][rows]); np.testing.assert_array_equal(m['center_number'], rows)
            self.assertTrue(np.all(m['counts'] == k+1)); self.assertTrue(np.all(g[:, k+1:] == 0))

    def test_only_ten_declared_fits_with_latest_h(self):
        from experiments.Task4C_BallQuery_2_5 import CONFIG
        spec = json.loads(Path(CONFIG).read_text()); self.assertEqual(spec['final']['seeds'], [96611])
        self.assertEqual(spec['neighbors']['radius_h'], 3)
        self.assertEqual([f['h_value'] for f in spec['flows']], [.003909492, .117239990])
        self.assertEqual(len(spec['candidates']), 4)
        total = 4
        for k, config in baseline.CONFIGS.items():
            for family in baseline.FAMILIES:
                s = baseline.load_spec(config, family)
                self.assertEqual(s['neighbor_count'], k); self.assertEqual(s['training']['seeds'], [96611])
                self.assertEqual(s['encoding']['voxel_resolutions'], [16]); total += len(s['methods'])
        self.assertEqual(total, 10)


if __name__ == '__main__': unittest.main()
