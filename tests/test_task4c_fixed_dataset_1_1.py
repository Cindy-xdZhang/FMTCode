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
    b.dist = dict(eval_min_to_any_train_over_h=1., test_max_to_same_instance_train_over_h=6., test_min_to_validation_over_h=.5, heldout_exclusion_margin_over_h=0.)
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

    def test_legal_center_applies_eval_first_distance_rules(self):
        b = bare_builder(); b.eval_points = np.array([[0., 0, 0], [10., 0, 0]]); b.eval_h = np.ones(2); b.eval_tree = cKDTree(b.eval_points)
        b.exclusion = [(np.array([40., -1, -1]), np.array([50., 1, 1]))]; b.heldout = {9}
        self.assertTrue(b.legal_center('test', np.array([0.1, 0, 0]), 1., 7, 1))                # test rows: only exact duplicates are forbidden
        self.assertTrue(b.legal_center('test', np.array([45., 0, 0]), 1., -1, 0))               # exclusion binds train/validation only
        b.reserved['test'] = [np.array([5., 0, 0])]
        self.assertFalse(b.legal_center('test', np.array([5., 0, 0]), 1., 7, 1)); self.assertIn('duplicate_center', b.rejections)
        self.assertFalse(b.legal_center('validation', np.array([0.3, 0, 0]), 1., -1, 0)); self.assertIn('too_close_to_test', b.rejections)
        self.assertTrue(b.legal_center('validation', np.array([0.7, 0, 0]), 1., -1, 0))          # >= 0.5 h_test
        self.assertFalse(b.legal_center('validation', np.array([45., 0, 0]), 1., -1, 0)); self.assertIn('inside_heldout_exclusion', b.rejections)
        self.assertFalse(b.legal_center('train', np.array([0.9, 0, 0]), 1., 7, 1)); self.assertIn('too_close_to_evaluation', b.rejections)
        self.assertTrue(b.legal_center('train', np.array([5., 0, 0]), 1., 7, 1))                 # 5h from both evaluation centers
        self.assertFalse(b.legal_center('train', np.array([45., 0, 0]), 1., -1, 0))

    def test_pair_sample_places_a_same_instance_training_center_in_the_3h_6h_shell(self):
        b = bare_builder(scene=dict(gt=None, locator=None, velocity=None, omega=None), axes=None)
        target = dict(center=np.array([10., 10, 10]), local_grid_scale=2., instance=7, test_index=42)
        captured = {}
        def fake_gt_head_sample(scene, center, instance, scale, sid, number, group, role, relaxed=False):
            captured['center'] = center; return dict(center=center, seeds=np.zeros((27, 3)), label=1, instance=instance, local_grid_scale=2., scale_id=sid)
        angle = lambda ok: patch.multiple(data, vector_at=lambda *a: np.zeros((1, 3)), head_mask=lambda *a: (np.array([ok]), None))
        with patch.object(data, 'sample_gt', return_value=(np.array([7]), None)), patch.object(data, 'gt_head_sample', fake_gt_head_sample), angle(True):
            s = b.pair_sample('train', target, 5, {'neighbor_grid_scale': 1.}, 0, False, np.random.default_rng(1))
        r = np.linalg.norm(captured['center']-target['center']); self.assertTrue(2. <= r <= 12.)          # 1h..6h with h = 2
        self.assertEqual((s['kind'], s['paired_test_center'], s['nearest_instance'], s['relaxed']), (data.KIND_GT_HEAD, 42, 7, False))
        with patch.object(data, 'sample_gt', return_value=(np.array([-1]), None)), angle(True):
            self.assertIsNone(b.pair_sample('train', target, 6, {'neighbor_grid_scale': 1.}, 0, False, np.random.default_rng(2)))
        self.assertIn('pair_outside_instance', b.rejections)
        with patch.object(data, 'sample_gt', return_value=(np.array([7]), None)), patch.object(data, 'gt_head_sample', fake_gt_head_sample), angle(False):
            self.assertIsNone(b.pair_sample('train', target, 7, {'neighbor_grid_scale': 1.}, 0, False, np.random.default_rng(3)))   # inside GT but not a head center
            self.assertIn('pair_head_angle', b.rejections)
            self.assertIsNotNone(b.pair_sample('train', target, 8, {'neighbor_grid_scale': 1.}, 0, True, np.random.default_rng(4)))  # relaxed pairs skip the angle

    def test_generate_train_tops_up_gt_head_rows_per_covered_instance_before_head_region_rows(self):
        b = bare_builder(spec=dict(dataset=dict(per_flow_counts=dict(train=10)), coverage=dict(min_head_centers_per_instance_per_split=1, target_head_centers_per_instance_per_split=2, paired_train_centers_per_test_positive=1)),
                         groups=dict(covered=[7, 8], heldout=[9]), heldout={9}, plan=[{'neighbor_grid_scale': 1.}], pools=dict(train=[dict(head_component=3)]))
        test_rows = [dict(label=1, instance=7), dict(label=1, instance=9), dict(label=0, instance=-1)]
        calls = []
        def fill(role, sources, quota, tag, fixed_sid=None, allow_short=False):
            calls.append((sources[0][0], sources[0][1] if sources[0][0] == 'gt' else tag, quota, allow_short))
            if sources[0][0] == 'pair': return [dict(kind=data.KIND_GT_HEAD, relaxed=False, instance=7, paired_test_center=0)]
            if sources[0][0] == 'gt': return [dict(kind=data.KIND_GT_HEAD, relaxed=False, instance=sources[0][1]) for _ in range(quota)]
            return [dict(kind=data.KIND_HEAD_REGION, relaxed=False, instance=-1) for _ in range(quota)]
        with patch.object(b, 'fill', fill):
            rows = b.generate_train(test_rows)
        self.assertEqual(calls[0], ('pair', 'pair0', 1, True))                       # only the covered-instance positive is paired
        self.assertEqual(calls[1:3], [('gt', 7, 1, True), ('gt', 8, 2, True)])        # top-up to the target of two head centers per covered instance
        self.assertEqual(sum(q for k, _, q, _ in calls if k == 'head'), 10-4); self.assertEqual(len(rows), 10); self.assertEqual(b.unpaired_test_rows, [])

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
        b = bare_builder(axes=axes, lambda2=lam, oyf=oyf, threshold=-1., instances=[7, 9], heldout={9}, groups=dict(covered=[7], heldout=[9]), bounds={9: (np.array([250., 0, 0]), np.array([350., 100, 100]))},
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

    def test_box_sample_takes_any_point_of_a_heldout_gt_box_and_labels_by_gt_membership(self):
        axes = [np.arange(0., 400., 50.) for _ in range(3)]
        b = bare_builder(axes=axes, lambda2=np.zeros((8, 8, 8)), heldout={9}, bounds={9: (np.array([100., 100, 100]), np.array([150., 150, 150]))},
                         scene=dict(gt=None, locator=None))
        with patch.object(data, 'sample_gt', return_value=(np.array([9]), None)):
            s = b.box_sample('test', 9, 1, {'neighbor_grid_scale': 1.}, 0, False, np.random.default_rng(0))
        self.assertTrue(np.all(s['center'] >= 100) and np.all(s['center'] <= 150)); self.assertEqual((s['kind'], s['label'], s['instance'], s['nearest_instance']), (data.KIND_GT_BOX, 1, 9, 9))
        with patch.object(data, 'sample_gt', return_value=(np.array([-1]), None)):
            s = b.box_sample('test', 9, 2, {'neighbor_grid_scale': 1.}, 0, False, np.random.default_rng(1))
        self.assertEqual((s['label'], s['instance'], s['nearest_instance']), (0, -1, 9)); self.assertEqual(len(s['seeds']), 27)

    def test_generate_test_uses_box_rows_for_heldout_and_gt_head_rows_for_covered_instances(self):
        b = bare_builder(spec=dict(dataset=dict(per_flow_counts=dict(test=6)), coverage=dict(min_head_centers_per_instance_per_split=2)), instances=[7, 9], heldout={9},
                         plan=[{'neighbor_grid_scale': 1.}], pools=dict(test=[dict(head_component=3)]))
        calls = []
        def fill(role, sources, quota, tag, fixed_sid=None, allow_short=False):
            calls.append((sources[0][0], sources[0][1] if sources[0][0] != 'head' else 'head', quota, allow_short)); return [dict()]*quota
        with patch.object(b, 'fill', fill): rows = b.generate_test()
        self.assertEqual(calls[:2], [('gt', 7, 2, True), ('box', 9, 2, True)]); self.assertEqual(len(rows), 6)

    def test_head_sample_labels_every_row_by_gt_membership_of_the_center(self):
        b = bare_builder(components=None, axes=None, oyf=None, index=0, scene=dict(gt=None, locator=None))
        head = dict(head_component=3, cell_ids=[0], label=0, instance=-1)
        fake = lambda *a: dict(center=np.array([1., 2, 3]), seeds=np.zeros((27, 3)), neighbor_distance=1.)
        with patch.object(data, 'head_center_and_neighbors', fake), patch.object(data, 'sample_gt', return_value=(np.array([5]), None)):
            s = b.head_sample('train', head, 1, {'neighbor_grid_scale': 1.}, 0, False, np.random.default_rng(0))
        self.assertEqual((s['label'], s['instance'], s['nearest_instance'], s['kind']), (1, 5, 5, data.KIND_HEAD_REGION))      # head catalogue said 0, center is inside GT 5
        with patch.object(data, 'head_center_and_neighbors', fake), patch.object(data, 'sample_gt', return_value=(np.array([-1]), None)):
            s = b.head_sample('train', dict(head, label=1, instance=7), 2, {'neighbor_grid_scale': 1.}, 0, False, np.random.default_rng(0))
        self.assertEqual((s['label'], s['instance'], s['nearest_instance']), (0, -1, -1))

    def test_config_encodes_user_rules(self):
        c = self.config
        self.assertEqual(c['distances']['test_max_to_same_instance_train_over_h'], 6.); self.assertEqual(c['distances']['test_min_to_validation_over_h'], .5)
        self.assertEqual(c['instance_split']['held_out_fraction'], .2); self.assertEqual(c['proposals']['minimum_valid_lines'], 17)
        self.assertEqual(c['proposals']['normal_attempts'], 100); self.assertEqual(c['coverage']['min_head_centers_per_instance_per_split'], 2)
        self.assertEqual(c['coverage']['target_head_centers_per_instance_per_split'], 10); self.assertEqual(c['distances']['eval_min_to_any_train_over_h'], 1.)
        self.assertEqual(c['dataset']['expected_counts'], {k: 2*v for k, v in c['dataset']['per_flow_counts'].items()}); self.assertEqual(c['distances']['heldout_exclusion_margin_over_h'], 0.)


if __name__ == '__main__': unittest.main()
