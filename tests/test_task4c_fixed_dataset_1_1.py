"""Rule contracts of the Task4-c fixed dataset builder 1.1 (no VTK integration needed)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_FixedDataset_1_1 as data


def bare_builder(**attributes):
    b = data.Builder.__new__(data.Builder); b.rejections = {}; b.reserved = {r: [] for r in data.ROLES}; b.reserved_cache = {}
    b.dist = dict(eval_min_to_any_train_over_h=3., test_max_to_same_instance_train_over_h=6., test_min_to_validation_over_h=.5, heldout_exclusion_margin_over_h=1.)
    b.rule = dict(seed=1, minimum_valid_lines=17, normal_attempts=4, relaxed_attempts=4, integration_batch_templates=8)
    b.final_trees = {}; b.instance_trees = {}; b.exclusion = []; b.heldout = set()
    for k, v in attributes.items(): setattr(b, k, v)
    return b


class FixedDatasetRules(unittest.TestCase):
    def setUp(self): self.config = json.loads(Path('config/mainExp_Task4C_FixedDataset_1.1.json').read_text())

    def test_instance_groups_hold_out_exactly_twenty_percent_and_keep_overlaps_together(self):
        ids = np.arange(10); low = np.array([[i*10., 0, 0] for i in range(10)]); high = low+4
        low[3] = [21., 0, 0]; high[3] = [26., 4, 4]            # instance 3 overlaps instance 2
        g = data.instance_groups(ids, low, high, 5, .2)
        self.assertEqual(g['target_heldout'], 2); self.assertEqual(len(g['heldout']), g['achieved_heldout']); self.assertLessEqual(g['achieved_heldout'], 2)
        self.assertEqual(sorted(g['heldout']+g['covered']), list(range(10)))
        self.assertEqual((2 in g['heldout']), (3 in g['heldout']))
        for seed in range(20):
            h = data.instance_groups(ids, low, high, seed, .2)['heldout']; self.assertEqual((2 in h), (3 in h)); self.assertLessEqual(len(h), 2)

    def test_legal_center_applies_split_distance_rules(self):
        b = bare_builder(); train = np.array([[0., 0, 0], [10., 0, 0]]); b.final_trees['train'] = cKDTree(train)
        b.final_trees['validation'] = cKDTree(np.array([[5., 20, 0]])); b.instance_trees = {7: cKDTree(train[:1])}; b.heldout = {9}
        b.exclusion = [(np.array([40., -1, -1]), np.array([50., 1, 1]))]
        self.assertTrue(b.legal_center('train', np.array([1., 0, 0]), 1., 7, 1))
        self.assertFalse(b.legal_center('train', np.array([45., 0, 0]), 1., -1, 0)); self.assertIn('inside_heldout_exclusion', b.rejections)
        self.assertFalse(b.legal_center('validation', np.array([2., 0, 0]), 1., 7, 1)); self.assertIn('too_close_to_train', b.rejections)
        self.assertTrue(b.legal_center('validation', np.array([5., 0, 0]), 1., 7, 1))            # 5h from both training centers
        self.assertTrue(b.legal_center('test', np.array([5., 0, 0]), 1., 7, 1))                  # >=3h, <=6h from instance 7
        self.assertFalse(b.legal_center('test', np.array([5., 7, 0]), 1., 7, 1)); self.assertIn('too_far_from_same_instance_train', b.rejections)
        self.assertTrue(b.legal_center('test', np.array([5., 7, 0]), 1., 9, 1))                  # held-out instance: no <=6h rule
        self.assertTrue(b.legal_center('test', np.array([45., 0, 0]), 1., -1, 0))               # exclusion only binds train/validation
        self.assertFalse(b.legal_center('test', np.array([5., 20.2, 0]), 1., -1, 0)); self.assertIn('too_close_to_validation', b.rejections)
        b.reserved['test'] = [np.array([5., 0, 0])]
        self.assertFalse(b.legal_center('test', np.array([5., 0, 0]), 1., 7, 1)); self.assertIn('duplicate_center', b.rejections)

    def test_fill_relaxes_a_source_after_normal_attempts_and_labels_relaxed_rows(self):
        b = bare_builder(flow='x', index=0, plan=[{'neighbor_grid_scale': 1.}], relaxed={r: 0 for r in data.ROLES}, attempts={r: 0 for r in data.ROLES}, trace_calls=0)
        calls = []
        def head_sample(role, head, number, scale, sid, relaxed, rng):
            calls.append(relaxed)
            if not relaxed: return None
            return dict(center=np.array([number % 97, 0., 0.]), seeds=np.zeros((27, 3)), label=0, instance=-1, local_grid_scale=1., scale_id=sid, relaxed=True)
        def trace(samples, sid):
            return [dict(s, geometry=None, line_count=17) for s in samples]
        with patch.object(b, 'head_sample', head_sample), patch.object(b, 'trace', trace), patch.object(b, 'legal_center', return_value=True):
            rows = b.fill('train', [('head', dict(head_component=3))], 3, 'unit')
        self.assertEqual(len(rows), 3); self.assertTrue(all(r['relaxed'] for r in rows)); self.assertEqual(b.relaxed['train'], 3)
        self.assertEqual(calls[:4], [False]*4); self.assertTrue(all(calls[4:]))

    def test_fill_raises_when_all_sources_exhaust_relaxed_proposals(self):
        b = bare_builder(flow='x', index=0, plan=[{'neighbor_grid_scale': 1.}], relaxed={r: 0 for r in data.ROLES}, attempts={r: 0 for r in data.ROLES})
        with patch.object(b, 'head_sample', return_value=None), patch.object(b, 'trace', return_value=[]):
            with self.assertRaises(ValueError): b.fill('train', [('head', dict(head_component=3))], 1, 'unit')

    def test_save_writes_metadata_index_and_line_attributes(self):
        axes = [np.arange(0., 400., 50.) for _ in range(3)]; lam = np.full((8, 8, 8), -5.); oyf = np.ones((8, 8, 8))
        b = bare_builder(axes=axes, lambda2=lam, oyf=oyf, threshold=-1., instances=[7, 9], heldout={9}, groups=dict(covered=[7], heldout=[9]),
                         scene=dict(gt=None, locator=None, velocity=np.zeros((8, 8, 8, 3)), omega=np.zeros((8, 8, 8, 3))))
        def row(center, label, instance, kind, relaxed, sid):
            grid = data.stencil(np.array(center, float), 1.); n = 20
            r = dict(center=np.array(center, float), label=label, instance=instance, head_component=5 if kind == 0 else -instance-1, scale_id=sid, center_number=len(center),
                     source_cell=0, centroid=np.zeros(3), radius=1., bounds=np.zeros((2, 3)), neighbor_distance=1., seed_rms_distance=1., measured_mean_arc_length=1.,
                     half_arc_lengths=np.ones((27, 2)), half_step_counts=np.ones((27, 2), np.int32), ds=.1, maxiteration=10, requested_half_length=1., local_grid_scale=1.,
                     nearest_train_center_distance=0., nearest_same_head_train_center_distance=0., line_count=n, geometry=np.zeros((27, 32, 3), np.float32),
                     normalized_seeds=np.zeros((27, 3), np.float32), kind=kind, relaxed=relaxed, nearest_instance=instance if instance >= 0 else 7)
            r['normalized_seeds'][:n] = grid[:n]; r['physical_seeds'], r['stencil_slots'] = data.physical_seeds_of_row(r); return r
        rows = [row([100., 50, 50], 1, 7, 1, False, 0), row([200., 50, 50], 0, -1, 0, False, 1), row([300., 50, 50], 1, 9, 0, True, 2)]
        owners = np.array([7, -1, 9]); angle = np.array([True, False, True])
        with tempfile.TemporaryDirectory() as temp, patch.object(data, 'sample_gt', return_value=(owners, None)), patch.object(data, 'head_mask', return_value=(angle, None)), \
             patch.object(data, 'vector_at', return_value=np.zeros((3, 3))):
            out = Path(temp); report = b.save('train', rows, out)
            with np.load(out/'metadata.npz') as z: m = {k: z[k] for k in z.files}
            with np.load(out/data.INDEX_FILE) as z: idx = {k: z[k] for k in z.files}
            with np.load(out/data.ATTRIBUTE_FILE) as z: at = {k: z[k] for k in z.files}
        self.assertEqual(set(m), set(data.METADATA_KEYS)); np.testing.assert_array_equal(m['labels'], [1, 0, 1]); np.testing.assert_array_equal(m['head_component'], [-8, 5, 5])
        np.testing.assert_array_equal(idx['sample_kind'], [1, 0, 0]); np.testing.assert_array_equal(idx['relaxed'], [False, False, True])
        np.testing.assert_array_equal(idx['neighbor_filter'], [1, 1, 2]); np.testing.assert_array_equal(idx['heldout_instance'], [False, False, True])
        np.testing.assert_array_equal(idx['center_in_gt_head'], [True, False, True]); np.testing.assert_array_equal(idx['original_center_id'], [0, 0, 0])
        self.assertEqual(report['coverage_head_centers_per_instance'], {'7': 1, '9': 1}); self.assertEqual(report['gt_head_rows'], 1); self.assertEqual(report['relaxed'], 1)
        self.assertEqual(at['seed_points'].shape, (3, 27, 3)); self.assertTrue(at['head_candidate'][:, :20].all() and not at['head_candidate'][:, 20:].any())
        np.testing.assert_allclose(at['seed_points'][0, :20], data.stencil(np.array([100., 50, 50]), 1.)[:20])

    def test_config_encodes_user_rules(self):
        c = self.config
        self.assertEqual(c['distances']['eval_min_to_any_train_over_h'], 3.); self.assertEqual(c['distances']['test_max_to_same_instance_train_over_h'], 6.)
        self.assertEqual(c['instance_split']['held_out_fraction'], .2); self.assertEqual(c['proposals']['minimum_valid_lines'], 17)
        self.assertEqual(c['proposals']['normal_attempts'], 1000); self.assertGreaterEqual(c['coverage']['min_head_centers_per_instance_per_split'], 2)
        self.assertEqual(c['dataset']['expected_counts'], {k: 2*v for k, v in c['dataset']['per_flow_counts'].items()})


if __name__ == '__main__': unittest.main()
