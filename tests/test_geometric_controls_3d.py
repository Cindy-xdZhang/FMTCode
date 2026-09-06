"""Standard-library unittest driver; numerical checks require only NumPy."""
import json
from pathlib import Path
import sys
import unittest
import importlib.util
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FMT_Utils.GeometricControls_3D import (
    validate_cache, perturb, geometric_sequences, pad_auxiliary, array_hash, stressed_seed_ivd,
)


def fixture():
    offsets = np.asarray([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
    x = np.broadcast_to(offsets[None,:,None,:], (3,7,32,3)).copy()
    x[...,0] += np.arange(32)[None,None,:] * 0.01
    return {'raw_features': x.reshape(3,-1), 'fmt_features': np.zeros((3,161)),
            'valid_mask': np.asarray([1,0,1,1],dtype=bool),
            'line_lengths': np.asarray([[49]*7,[12]*7,[49]*7,[49]*7]),
            'seeds': np.arange(9).reshape(3,3), 'reference': np.asarray([0,1,0]),
            'metadata_json': np.asarray(json.dumps({'valid_primitives':3,'source_time_step':0.1}))}


class GeometricControlsTests(unittest.TestCase):
    def test_zero_coordinates_are_valid(self):
        r = validate_cache(fixture())
        self.assertEqual(r['certificate']['invalid'],1)
        self.assertEqual(r['raw'].shape,(3,7,32,3))

    def test_negative_invalid_flags_rejected(self):
        c=fixture(); c['valid_mask']=np.asarray([1,-1,1,1])
        with self.assertRaisesRegex(ValueError,'sentinels'): validate_cache(c)

    def test_incomplete_retained_line_rejected(self):
        c=fixture(); c['line_lengths'][2,5]=48
        with self.assertRaisesRegex(ValueError,'incomplete'): validate_cache(c)

    def test_mixed_scale_lengths_and_times(self):
        c=fixture(); c['integration_steps']=np.asarray([32,48,64]); c['physical_dt']=np.asarray([.01,.02,.03])
        c['line_lengths'][c['valid_mask']]=c['integration_steps'][:,None]+1
        r=validate_cache(c)
        np.testing.assert_allclose(r['times'][:,-1],[.32,.96,1.92])

    def test_clean_is_exact_and_inputs_not_modified(self):
        r=validate_cache(fixture()); before=array_hash(r['raw'])
        x,t=perturb(r,{'id':'clean','kind':'clean','level':0},17)
        np.testing.assert_array_equal(x,r['raw']); np.testing.assert_array_equal(t,r['times'])
        self.assertEqual(before,array_hash(r['raw']))

    def test_paired_noise_is_deterministic_and_varies_with_repeat(self):
        r=validate_cache(fixture()); c={'id':'g','kind':'gaussian','level':.01}
        a,_=perturb(r,c,17); b,_=perturb(r,c,17); d,_=perturb(r,c,18)
        np.testing.assert_array_equal(a,b); self.assertFalse(np.array_equal(a,d))

    def test_dropout_preserves_endpoints_and_physical_grid(self):
        r=validate_cache(fixture()); x,t=perturb(r,{'id':'d','kind':'frame_dropout','level':.4},17)
        np.testing.assert_array_equal(x[:,:,[0,-1]],r['raw'][:,:,[0,-1]])
        np.testing.assert_array_equal(t,r['times'])

    def test_truncation_does_not_read_future_points(self):
        r=validate_cache(fixture()); q=dict(r); q['raw']=r['raw'].copy(); q['raw'][:,:,16:]=100000
        c={'id':'s','kind':'short_track','level':.5}
        a,t=perturb(r,c,17); b,_=perturb(q,c,17)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(t[:,-1]<r['times'][:,-1]))

    def test_uniform_translation_has_zero_vorticity(self):
        r=validate_cache(fixture()); s=geometric_sequences(r['raw'],r['times'],r['scale_id'])
        np.testing.assert_allclose(s[:,:,0],0,atol=1e-5)

    def test_rotations_recover_physical_curl_and_sample_mean(self):
        base=validate_cache(fixture())['raw']
        offsets=base[:,:,0].astype(np.float64)
        t=np.broadcast_to(np.linspace(0,.0001,32),(3,32))
        angular=np.asarray([0.,1.,3.]); angle=angular[:,None]*t
        x=np.broadcast_to(offsets[:,:,None,:],(3,7,32,3)).copy()
        x[:,:,:,0]=offsets[:,:,None,0]*np.cos(angle)[:,None]-offsets[:,:,None,1]*np.sin(angle)[:,None]
        x[:,:,:,1]=offsets[:,:,None,0]*np.sin(angle)[:,None]+offsets[:,:,None,1]*np.cos(angle)[:,None]
        s=geometric_sequences(x,t,np.zeros(3))
        np.testing.assert_allclose(s[:,0,0],np.abs(2*angular-2*angular.mean()),atol=1e-4)

    def test_padding_does_not_discard_information(self):
        x=np.arange(12).reshape(3,4); y=pad_auxiliary(x)
        np.testing.assert_array_equal(x,y[:,:4]); self.assertTrue(np.all(y[:,4:]==0))
        with self.assertRaises(ValueError): pad_auxiliary(np.zeros((3,269)))

    def test_stress_default_matches_reference_and_preserves_input(self):
        r = validate_cache(fixture())
        rng = np.random.default_rng(81)
        r['raw'][:, :, 1:] += rng.normal(0, .0001, size=r['raw'][:, :, 1:].shape).astype(np.float32)
        before = array_hash(r['raw'])
        expected = geometric_sequences(r['raw'], r['times'], r['scale_id'])[:, 0, :1]
        np.testing.assert_array_equal(stressed_seed_ivd(r), expected)
        np.testing.assert_array_equal(stressed_seed_ivd(r, rcond=1.01), np.zeros_like(expected))
        np.testing.assert_allclose(stressed_seed_ivd(r, feature_scale=100), 100*expected, rtol=1e-6)
        for lag in (4, 16, 31): self.assertTrue(np.isfinite(stressed_seed_ivd(r, lag=lag)).all())
        for window in (9, 31): self.assertTrue(np.isfinite(stressed_seed_ivd(r, smooth_window=window)).all())
        self.assertEqual(array_hash(r['raw']), before)

    def test_stress_rejects_invalid_parameter_grid(self):
        r = validate_cache(fixture())
        for kwargs in ({'lag': 0}, {'lag': 32}, {'smooth_window': 2}, {'rcond': -1}, {'feature_scale': float('nan')}):
            with self.assertRaises(ValueError): stressed_seed_ivd(r, **kwargs)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,'requires research PyTorch environment')
    def test_stress_vae_in_memory_training_is_reproducible(self):
        import torch
        from experiments.Run_GeometryParameterStress import train_vae, encode
        torch.set_num_threads(2)
        x = np.random.default_rng(8).normal(size=(12, 4)).astype(np.float32)
        spec = {'hidden_dims': [8], 'latent_dim': 2, 'learning_rate': .001,
                'weight_decay': .00001, 'batch_size': 4, 'optimizer_steps': 2, 'beta': .000001}
        a, loss = train_vae(x, spec, 40, torch.device('cpu'))
        b, _ = train_vae(x, spec, 40, torch.device('cpu'))
        self.assertEqual(loss['steps'], 2)
        np.testing.assert_array_equal(encode(a, x, 'cpu'), encode(b, x, 'cpu'))

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,'requires research PyTorch environment')
    def test_frozen_training_and_prediction_interface(self):
        import torch
        from FMT_Utils.PathlineClassifier_3D import PathlineBinaryClassifier3D
        from experiments.Run_Task135_GeometricControls import load_residual
        from experiments.Verify_Task3_FMTClassifier import _normalize_train_only
        from experiments.Verify_Task3_FMTResidual import _train_one
        torch.set_num_threads(2)
        rng=np.random.default_rng(123)
        raw=rng.normal(size=(24,7,32,3)).astype(np.float32)
        aux=rng.normal(size=(24,268)).astype(np.float32)
        y=np.tile([0,1],12).astype(np.float32)
        tr,va,_,stats=_normalize_train_only((raw[:16],aux[:16],y[:16]),(raw[16:],aux[16:],y[16:]))
        base_config={'model':{'temporal_width':32,'embedding_dim':128,'auxiliary_dim':64}}
        base=PathlineBinaryClassifier3D(variant='raw',fmt_dim=268,**base_config['model'])
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            torch.save({'variant':'raw','config':base_config,'state_dict':base.state_dict(),
                        'normalization':stats,'best_epoch':0,'threshold':.5},directory/'synthetic_raw_seed40.pt')
            spec={'model':{'embedding_dim':128,'auxiliary_dim':64,'residual_input':'geometry_fmt'},
                  'fusion':{'fixed_alpha':1.,'selection_metric':'average_precision'},
                  'training':{'batch_size':8,'learning_rate':.001,'weight_decay':.0001,
                              'max_epochs':1,'patience':1,'min_delta':.0001},
                  'raw_checkpoint_dir':str(directory),'raw_wide_parameter_count':148225,
                  'auxiliary_source':'fmt'}
            result=_train_one(spec,'synthetic',40,(tr,va,None),stats,torch.device('cpu'),directory)
            self.assertNotIn('test_f1',result)
            model,state=load_residual(result['checkpoint'],268,'cpu')
            self.assertEqual(state['best_epoch'],1)
            self.assertGreater(sum(p.numel() for p in model.parameters()),0)


if __name__=='__main__': unittest.main()
