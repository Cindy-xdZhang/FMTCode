"""Rule contracts of the Task4-c fixed dataset builder 2.1 (no VTK integration needed)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_FixedDataset_2_1 as data

INTEGRATION = dict(points_per_curve=32, minimum_actual_half_arc_fraction=.95, maximum_actual_half_arc_fraction=1.002, out_of_domain_min_half_arc_fraction=.25, out_of_domain_max_half_arc_fraction=1.25)
FULL = np.array([3, 3])            # VTK reason 3 as observed for full-length halves; 1 = OUT_OF_DOMAIN


def straight_half(seed, direction, n, ds):
    return seed[None]+np.arange(n+1)[:, None]*ds*np.asarray(direction, float)[None]


class FixedDatasetV2Rules(unittest.TestCase):
    def setUp(self): self.config = json.loads(Path('config/mainExp_Task4C_FixedDataset_2.1.json').read_text())

    def test_config_half_steps_match_the_rules(self):
        steps = {f['name']: data.half_steps(f['half_lengths'], f['ds']) for f in self.config['flows']}
        self.assertEqual(steps, dict(channel=[70, 100, 130], tbl=[140, 200, 260]))
        self.assertEqual(self.config['sampling']['samples_per_flow']*2, 400000); self.assertEqual(self.config['sampling']['pool_per_flow']*2, 4000000)

    def test_candidate_cells_and_pool_stay_inside_the_trilinear_region(self):
        axes = [np.linspace(0, 1, 11), np.linspace(0, 2, 21), np.linspace(0, 1, 6)]
        nz, ny, nx = 6, 21, 11; lam = np.ones((nz, ny, nx)); lam[:, :, 3:7] = -1; oyf = np.ones((nz, ny, nx)); oyf[:, 15:] = -1
        mask = data.candidate_mask(lam, oyf, 0.)
        cells = data.candidate_cells(mask); low, size = data.cell_corners(cells, axes)
        self.assertTrue(np.all(low[:, 0] >= .2-1e-12) and np.all(low[:, 0]+size[:, 0] <= .7+1e-12))
        pool, stats = data.initial_pool(axes, lam, oyf, 0., cells, 500, 7, 400)
        self.assertEqual(pool.shape, (500, 3)); self.assertLess(stats['acceptance'], 1)
        self.assertTrue(np.all(pool[:, 0] > .25) and np.all(pool[:, 0] < .65) and np.all(pool[:, 1] < 1.45))   # trilinear zero crossings

    def test_poisson_disk_is_deterministic_maximal_and_separated(self):
        rng = np.random.default_rng(3); points = rng.random((20000, 3))
        selected, radius, log = data.poisson_disk(points, 500, .05, tolerance=1e-3)
        self.assertEqual(len(selected), 500); self.assertGreater(radius, 0)
        d, _ = cKDTree(points[selected]).query(points[selected], k=2); self.assertTrue(np.all(d[:, 1] >= radius))
        again, radius_again, _ = data.poisson_disk(points, 500, .05, tolerance=1e-3)
        self.assertEqual(radius, radius_again); self.assertTrue(np.array_equal(selected, again))
        self.assertLess(len(data.dart_throw(points, radius*1.01, 500)), 500)
        self.assertTrue(np.array_equal(data.dart_throw(points, radius, 500)[:500], selected))

    def test_cut_and_clean_uses_step_prefixes_and_rejects_short_halves(self):
        seed = np.array([1., 2., 3.]); ds = .01; steps = [7, 10, 13]
        back = straight_half(seed, (-1, 0, 0), 13, ds); front = straight_half(seed, np.array([1, .3, .2])/np.linalg.norm([1, .3, .2]), 13, ds)
        curves, valid, relaxed, arcs, counts, bounds, reasons = data.cut_and_clean(back, front, FULL, ds, steps, INTEGRATION)
        self.assertTrue(valid.all()); self.assertFalse(relaxed.any()); self.assertEqual(reasons, [])
        self.assertTrue(np.allclose(arcs[:, 0], np.array(steps)*ds)); self.assertTrue(np.array_equal(counts[:, 0], steps))
        self.assertTrue(np.allclose(curves[0, 0], back[7]) and np.allclose(curves[2, -1], front[13]))
        short_front = front[:9]                                   # 8 forward steps without leaving the domain: 10- and 13-step curves invalid
        curves, valid, relaxed, arcs, counts, bounds, reasons = data.cut_and_clean(back, short_front, FULL, ds, steps, INTEGRATION)
        self.assertEqual(valid.tolist(), [True, False, False]); self.assertEqual(reasons, ['insufficient_physical_length']*2); self.assertFalse(relaxed.any())

    def test_out_of_domain_half_is_relaxed_to_a_quarter_length(self):
        seed = np.array([1., 2., 3.]); ds = .01; steps = [7, 10, 13]
        back = straight_half(seed, (-1, 0, 0), 13, ds)
        left = np.array([3, data.OUT_OF_DOMAIN]); d = np.array([.3, 1, .2])/np.linalg.norm([.3, 1, .2])   # the forward half left the domain
        curves, valid, relaxed, arcs, counts, bounds, reasons = data.cut_and_clean(back, straight_half(seed, d, 8, ds), left, ds, steps, INTEGRATION)
        self.assertTrue(valid.all()); self.assertEqual(relaxed.tolist(), [False, True, True])     # 8 steps: full for L=7; 0.8L and 0.615L relaxed
        self.assertEqual(curves.shape, (3, 32, 3)); self.assertTrue(np.allclose(curves[2, -1], seed+.08*d))
        curves, valid, relaxed, arcs, counts, bounds, reasons = data.cut_and_clean(back, straight_half(seed, d, 3, ds), left, ds, steps, INTEGRATION)
        self.assertEqual(valid.tolist(), [True, True, False])    # 0.03 >= 0.25*0.07 and 0.25*0.10, but < 0.25*0.13
        self.assertEqual(reasons, ['insufficient_physical_length'])
        curves, valid, relaxed, arcs, counts, bounds, reasons = data.cut_and_clean(back, straight_half(seed, d, 3, ds), FULL, ds, steps, INTEGRATION)
        self.assertFalse(valid.any())                              # the same short half without leaving the domain stays invalid

    def test_assignment_kinds_follow_the_rule_order_and_labels_stay_separate(self):
        instances = [0, 5]; low = np.array([[0., 0, 0], [10., 0, 0]]); high = np.array([[1., 1, 1], [11., 1, 1]])
        gt_points = np.array([[.5, .5, .5], [10.5, .5, .5]]); gt_inst = np.array([0, 5])
        seeds = np.array([[.5, .5, .5], [10.2, .2, .2], [3., 0, 0], [8., 5, 5]])
        owner = np.array([0, -1, -1, -1]); seed_component = np.array([1, 1, 2, 3]); comp_instance = np.array([-1, -1, 5, -1])
        instance, kind = data.assign_instances(seeds, owner, instances, low, high, gt_points, gt_inst, seed_component, comp_instance)
        self.assertEqual(instance.tolist(), [0, 5, 5, 5]); self.assertEqual(kind.tolist(), [0, 1, 2, 3])
        self.assertEqual(((owner >= 0).astype(int)).tolist(), [1, 0, 0, 0])    # label is GT membership only

    def test_component_instance_mapping_takes_the_largest_overlap(self):
        # component labels: comp 1 overlaps instance 0 twice and instance 4 once -> instance 0.
        class FakeGT: pass
        mask = np.zeros((1, 1, 5), bool); mask[0, 0, :4] = True; labels = np.zeros((1, 1, 5), np.int64); labels[0, 0, :3] = 1; labels[0, 0, 3] = 2
        axes = [np.arange(5.), np.arange(1.), np.arange(1.)]
        owners = np.array([0, 0, 4, -1])
        original = data.sample_gt
        try:
            data.sample_gt = lambda gt, points, locator=None: (owners[:len(points)], None)
            mapping, overlap, points = data.component_instances(mask, labels, axes, None, None)
        finally: data.sample_gt = original
        self.assertEqual(mapping.tolist(), [-1, 0, -1]); self.assertEqual(overlap.tolist(), [0, 2, 0]); self.assertEqual(len(points), 4)

    def test_instance_folds_partition_balance_and_tie_overlaps(self):
        ids = list(range(10)); low = np.array([[i*10., 0, 0] for i in ids]); high = low+4
        low[3] = [21., 0, 0]; high[3] = [26., 4, 4]                # instance 3 overlaps instance 2
        counts = {i: 100+i for i in ids}; counts[7] = 1000
        fold_of, report = data.instance_folds(ids, low, high, counts, 5, 11)
        self.assertEqual(sorted(fold_of), ids); self.assertEqual(fold_of[2], fold_of[3]); self.assertEqual(sum(report['fold_samples']), sum(counts.values()))
        self.assertEqual(set(fold_of.values()), set(range(5)))
        alone = [g for g in report['groups'] if g['group'] == [7]][0]; self.assertEqual(report['fold_samples'][alone['fold']], 1000)

    def test_save_and_load_roundtrip(self):
        n = 6; rng = np.random.default_rng(0)
        meta = {k: rng.integers(0, 3, n) for k in ('label', 'instance', 'assignment_kind', 'gt_owner', 'fold', 'split', 'component', 'component_size', 'pool_index')}
        meta.update(lambda2=-rng.random(n), oyf=rng.random(n), local_grid_scale=rng.random(n), half_arc_lengths=rng.random((n, 3, 2)), half_step_counts=rng.integers(1, 5, (n, 3, 2)), half_termination=rng.integers(1, 4, (n, 2)), boundary_relaxed=rng.random((n, 3)) < .5,
                    curve_bounds=rng.random((n, 3, 2, 3)), requested_half_length=np.array([.07, .1, .13]), half_steps=np.array([70, 100, 130]), ds=np.float64(.001),
                    poisson_radius=np.float64(.01), lambda2_threshold=np.float64(-13.395))
        rows = dict(seeds=rng.random((n, 3)), curves=rng.random((n, 3, 32, 3)).astype(np.float32), metadata=meta, folds=dict(fold_samples=[1]*5), assignment=dict(per_kind={}))
        with tempfile.TemporaryDirectory() as tmp:
            record = data.save(rows, Path(tmp), lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
            seeds, curves, loaded = data.load(tmp)
            self.assertEqual(record['samples'], n); self.assertEqual(set(loaded), set(data.METADATA_KEYS))
            self.assertTrue(np.array_equal(seeds, rows['seeds']) and np.array_equal(np.asarray(curves), rows['curves']))


if __name__ == '__main__': unittest.main()
