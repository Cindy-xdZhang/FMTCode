"""Cluster and split contracts of the v2 FMT data interface (no VTK, no training)."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as v2
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import normalize


class FixedDatasetFMTv2(unittest.TestCase):
    def setUp(self): self.spec = json.loads(Path('config/mainExp_Task4C_FixedDatasetFMT_2.1.json').read_text())

    def test_fps_order_is_nested_distinct_and_starts_far_from_the_center(self):
        rng = np.random.default_rng(1); seeds = rng.random((500, 3))
        order16, candidates = v2.neighbor_table(seeds, 64, 16)
        self.assertEqual(order16.shape, (500, 16)); self.assertTrue(np.all(order16 != np.arange(500)[:, None]))
        self.assertTrue(all(len(np.unique(r)) == 16 for r in order16))
        order6 = v2.fps_order(seeds, candidates, 6); self.assertTrue(np.array_equal(order6, order16[:, :6]))      # k=6 is the FPS16 prefix
        i = 7; d = np.linalg.norm(seeds[candidates[i]]-seeds[i], axis=1)
        self.assertEqual(order16[i, 0], candidates[i][d.argmax()])                                            # first pick = farthest candidate
        self.assertTrue(set(order16[i]) <= set(candidates[i]))

    def test_split_masks_partition_samples_and_keep_the_test_fold(self):
        rng = np.random.default_rng(2); n = 10000; meta = dict(fold=rng.integers(0, 5, n), label=rng.integers(0, 2, n))
        masks = v2.split_masks(meta, 0, self.spec)
        self.assertTrue(np.array_equal(masks['test'], meta['fold'] == 0))
        total = masks['train'].astype(int)+masks['validation'].astype(int)+masks['test'].astype(int); self.assertTrue(np.all(total == 1))
        self.assertAlmostEqual(masks['validation'].sum()/(~masks['test']).sum(), .1, delta=.002)
        again = v2.split_masks(meta, 0, self.spec); self.assertTrue(np.array_equal(again['validation'], masks['validation']))
        other = v2.split_masks(meta, 1, self.spec); self.assertFalse(np.array_equal(other['validation'], masks['validation']))

    def test_normalize_bundle_matches_the_frozen_normalization_for_lines_and_seeds(self):
        torch.manual_seed(0); lines = torch.rand(4, 7, 32, 3, dtype=torch.float64)*3+1; seeds = lines[:, :, 15].clone()+.01
        geometry, s, counts = v2.normalize_bundle(lines.float(), seeds.float())
        reference, _ = normalize(lines.float(), counts)
        torch.testing.assert_close(geometry, reference); self.assertEqual(counts.tolist(), [7]*4)
        center = lines.mean((1, 2), keepdim=True); radius = (lines-center).square().sum(-1).amax((1, 2)).sqrt()
        torch.testing.assert_close(s, ((seeds-center[:, :, 0])/radius[:, None, None]).float(), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.all(geometry.norm(dim=-1).amax((1, 2)) <= 1+1e-5))

    def test_config_arms_and_row_counts(self):
        ids = [c['id'] for c in self.spec['candidates']]; self.assertEqual(ids, ['p35_h0_fps6', 'p35_h0_fps16', 'c156_fps6', 'c156_fps16'])
        self.assertTrue(all(c['neighbor_selection'] == 'fps' and c['neighbors'] in (6, 16) for c in self.spec['candidates']))
        self.assertEqual(self.spec['neighbors'], dict(self.spec['neighbors'], pool=64, max_k=16))
        self.assertEqual(sum(self.spec['expected_counts'].values()), (189416+198413)*3)


if __name__ == '__main__': unittest.main()
