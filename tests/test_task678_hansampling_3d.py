"""Dense-query sampling, strict temporal holdout, and complete executable smoke."""
import copy
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from tests.test_task678_flowmap_3d import analytic_field
from FMT_Utils.HanFlowMapData_3D import dense_window, support_features, partition_roles, sobol_queries, measured_trajectories
from FMT_Utils.HanFlowMapFit_3D import fit_dense
from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.FlowMapFit_3D import compose_direct
from experiments.Build_Task678_HanSampling_1_1 import DEFAULT_CONFIG, save_roles
from experiments.Run_Task678_HanSampling_1_1 import arrays_for, derangement_by_window, run


def smoke_spec(output):
    spec = json.loads(Path(DEFAULT_CONFIG).read_text())
    spec.update(experiment='Verify_Task678_HanSmoke_1.1', output_root=str(output),
                datasets=['analytic_unsteady'], families={'analytic_unsteady': 'analytic'}, seeds=[9110])
    spec['sampling'].update(train_queries=8, validation_queries=4, test_queries=8, query_candidates=64,
                            minimum_bundles=8, minimum_role_bundles=1)
    spec['decoder'].update(hidden_width=16, support_blocks=1, query_blocks=1,
                          bundle_batch=4, queries_per_bundle=4, smoke_steps=3)
    return spec


class HanSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.spec = smoke_spec('unused')
        centers = np.random.default_rng(19).uniform(-1, 1, (16, 3))
        cls.data, cls.audit = dense_window(analytic_field(), centers, .1, .5, cls.spec['sampling'], 910)
        cls.features = support_features(cls.data)

    def test_queries_cover_volume_and_are_deterministic(self):
        a = sobol_queries(512, 91)
        np.testing.assert_array_equal(a, sobol_queries(512, 91))
        self.assertGreater(np.std(np.linalg.norm(a, axis=-1)), .1)
        self.assertTrue((np.abs(a).sum(-1) < .95).all())
        self.assertFalse(np.array_equal(a, sobol_queries(512, 92)))

    def test_analytic_trajectory_and_gap_and_second_coverage(self):
        d = self.data
        t = np.linspace(0, 1, 125)
        displacement = np.stack([.1*t + .02*t*t, -.01*t*t, np.zeros_like(t)], -1)
        np.testing.assert_allclose(d['target_long'] - d['target_long'][:, :, :1],
                                  np.broadcast_to(displacement, d['target_long'].shape), atol=1e-7)
        self.assertGreater(self.audit['task7_minimum_seed_gap'], .1)
        arrival = (d['target_long'][:, :, 62] - d['origin1'][:, None]) / d['radius1'][:, None, None]
        self.assertTrue((np.abs(arrival).sum(-1) <= .95).all())
        np.testing.assert_allclose(d['radius1'], .2, atol=2e-7)

    def test_frozen_features_and_target_independence(self):
        from FMT_Utils.FlowMapModels_3D import raw_features
        old = np.zeros((len(self.data['support0']), 31, 32, 3), np.float32)
        old[:, :7] = self.data['support0']
        np.testing.assert_array_equal(raw_features(old, self.data['radius0'], 'fmt_all'), self.features['fmt_all__support0'])
        changed = {k: v.copy() for k, v in self.data.items()}
        for k in ('target0', 'target1', 'target_long'):
            changed[k][:] = 12345.
        for k, v in support_features(changed).items():
            np.testing.assert_array_equal(v, self.features[k])

    def test_roles_have_disjoint_material_queries_and_primitives(self):
        roles = partition_roles(self.data, 'train', self.spec['sampling'])
        fit, val, test = (roles[k] for k in ('fit', 'query_validation', 'query_test'))
        for k in ('query_ids0', 'query_ids1'):
            for i in range(len(fit[k])):
                self.assertFalse(set(fit[k][i]) & set(val[k][i]))
                self.assertFalse(set(fit[k][i]) & set(test[k][i]))
                self.assertFalse(set(val[k][i]) & set(test[k][i]))
        self.assertFalse(set(fit['material_bundle_ids']) & set(roles['primitive_test']['material_bundle_ids']))

    def test_odd_time_truth_cannot_affect_training(self):
        arrays = arrays_for({**self.data, **self.features}, 'Task6', 'fmt_all')
        changed = {k: v.copy() for k, v in arrays.items()}
        changed['target'][:, :, 1::2] += 9000.
        a, ia = fit_dense(arrays, self.spec['decoder'], 91, 'cpu')
        b, ib = fit_dense(changed, self.spec['decoder'], 91, 'cpu')
        self.assertEqual(ia['training_output_scale'], ib['training_output_scale'])
        self.assertEqual(ia['final_train_probe_nrmse'], ib['final_train_probe_nrmse'])
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_learning_can_fit_analytic_mapping(self):
        settings = {**self.spec['decoder'], 'smoke_steps': 400, 'learning_rate': .003}
        arrays = arrays_for({**self.data, **self.features}, 'Task6', 'fmt_all')
        model, info = fit_dense(arrays, settings, 93, 'cpu')
        self.assertLess(info['final_train_probe_nrmse'], .1 * info['initial_train_probe_nrmse'])

    def test_composition_never_reads_true_arrival(self):
        class Translation(torch.nn.Module):
            def encode_support(self, tokens, geometry):
                return tokens[:, 0]
            def query_context(self, context, query, tau, scales):
                v = torch.zeros_like(query); v[:, 0] = context[:, 0]
                return query + tau[:, None] * v
        arrays = dict(tokens=np.array([[[.1]], [[.2]]], np.float32), geometry=np.zeros((2, 1, 4), np.float32),
                      query=np.zeros((2, 8, 3), np.float32), scales=np.zeros((2, 2), np.float32),
                      target=np.full((2, 8, 5, 3), 999., np.float32),
                      origin=np.array([[0., 0., 0.], [.7, 0., 0.]], np.float32), radius=np.ones(2, np.float32))
        p, info = compose_direct(Translation(), arrays, 'cpu')
        np.testing.assert_allclose(p[:, :, -1, 0], .3, atol=1e-6)
        self.assertEqual(info['second_query_uses'], 'predicted_first_stage_endpoint')

    def test_derangement_keeps_source_window(self):
        d = {**self.data, 'window_ordinal': np.arange(len(self.data['origin0'])) // 4}
        p = derangement_by_window(d, 91)
        self.assertTrue((p != np.arange(len(p))).all())
        np.testing.assert_array_equal(d['window_ordinal'], d['window_ordinal'][p])

    def test_metrics_bounded_chunk_and_known_error(self):
        t = self.data['target0']; p = t.copy(); p[:, :, 1:, 0] += .1
        a = measured_trajectories(p, t, self.data['radius0'], chunk=1)
        b = measured_trajectories(p, t, self.data['radius0'], chunk=8)
        self.assertAlmostEqual(a['position_nrmse'], 1., places=6)
        self.assertLess(a['pair_distance_error'], 1e-6)
        for k, v in a.items():
            if not isinstance(v, str):
                np.testing.assert_allclose(v, b[k], atol=1e-12)


def run_smoke(output):
    torch.set_num_threads(2)
    spec = smoke_spec(output); root = Path(output); cfg = root / 'smoke_config.json'
    write_json(cfg, spec)
    records = []
    for i in range(8):
        data, audit = dense_window(analytic_field(i / 8), np.random.default_rng(i).uniform(-1, 1, (8, 3)),
                                  .1, .5, spec['sampling'], 900+i)
        row = dict(ordinal=i, role=next(k for k, ids in spec['splits'].items() if i in ids),
                   start_index=i*9, end_index=i*9+8, time_start=float(i), time_end=float(i+1))
        records.extend(save_roles(root, 'analytic_unsteady', data, row, spec['sampling'],
                                 {**row, **audit, 'config_sha256': sha256(cfg), 'status': 'PASS'}))
    write_json(root / 'build/analytic_unsteady.json', dict(status='PASS', config_sha256=sha256(cfg), records=records))
    from experiments.Audit_Task678_HanSampling_1_1 import audit_data, audit
    audit_data(spec, str(cfg))
    for arm in [*spec['neural_arms'], 'affine']:
        run(spec, str(cfg), 'analytic_unsteady', arm, 0 if arm == 'affine' else 9110, 'cpu')
    audit(spec, str(cfg))
    report = json.loads((root / 'independent_audit.json').read_text())
    assert report['metric_records'] == 102 and report['status'] == 'PASS'
    # Deliberately corrupt a scalar while leaving predictions unchanged.
    path = root / 'runs/analytic_unsteady/fmt_all/seed9110/Task6_fit_normal.json'
    original = path.read_bytes(); value = json.loads(original)
    value['metrics']['position_nrmse'] += 1.
    write_json(path, value)
    try:
        audit(spec, str(cfg))
    except AssertionError:
        print('CORRUPTION REJECTED', flush=True)
    else:
        raise AssertionError('Audit accepted corrupt reported metrics')
    finally:
        path.write_bytes(original)
    print('HAN SMOKE PASS: 102 replayed records, all arms, no checkpoint', flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--smoke-output':
        run_smoke(sys.argv[2])
    else:
        unittest.main()
