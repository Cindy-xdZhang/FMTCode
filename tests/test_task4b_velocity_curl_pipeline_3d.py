"""End-to-end unit conversion and pooled-training smoke tests (CPU only)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from experiments.Verify_Task4B_VelocityCurlMemorization import DimensionlessField
from experiments.Verify_Task4B_PooledMemorization_3_1 import _train_one
from FMT_Utils.Task4A_StreamlineClustering_3D import integrate_bidirectional_cross_primitives
from FMT_Utils.Task4B_CrossFlow_3D import dimensionless_primitive_features


class ScaledField:
    def __init__(self, origin, length, speed):
        self.origin, self.length, self.speed = origin, length, speed

    def velocity(self, xyz):
        q = (xyz-self.origin)/self.length
        return np.stack((2-0.01*q[...,1],0.01*q[...,0],0.001*q[...,2]),axis=-1)*self.speed


class PipelineTest(unittest.TestCase):
    def test_length_speed_and_translation_units(self):
        encoder = dict(num_freq=6,neighbor_scale=100,neighbor_pool='sort',mode='gram',include_chirality=True)
        primitives, features = [], []
        for origin,length,speed in [(np.zeros(3),.005,1.),(np.array([250.,7.,30.]),.5,100.)]:
            field = DimensionlessField(ScaledField(origin,length,speed),origin,length,speed)
            p, valid = integrate_bidirectional_cross_primitives(field,np.array([[1.,2.,3.],[2.,4.,6.]]),1.,16,.1501458034894056)
            self.assertTrue(valid.all())
            self.assertEqual(p.shape,(2,7,33,3))
            raw,fmt = dimensionless_primitive_features(p,1.,**encoder)
            self.assertEqual(fmt.shape,(2,161))
            primitives.append(p)
            features.append(fmt)
        np.testing.assert_allclose(primitives[0],primitives[1],rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(features[0],features[1],rtol=1e-4,atol=1e-4)

    def test_shared_fit_loop_writes_matching_targets_without_model_files(self):
        torch.set_num_threads(2)
        spec=yaml.safe_load(Path('config/Verify_Task4B_VelocityCurlMemorization_4.1.yaml').read_text())
        spec['model'].update(temporal_width=8,embedding_dim=8,auxiliary_dim=8)
        spec['training'].update(max_epochs=2,batch_size=8,evaluation_batch_size=8)
        spec['pass_gate'].update(expected_sample_count=16,expected_class_support=[4,4,4,4])
        rng=np.random.default_rng(123)
        raw=rng.normal(size=(16,7,33,3)).astype(np.float32)
        fmt=rng.normal(size=(16,161)).astype(np.float32)
        labels=np.tile(np.arange(4),4)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=_train_one(spec,raw,fmt,labels,np.zeros(16,np.int8),np.ones(16,np.int32),
                np.arange(16),np.zeros((16,3),np.int32),np.zeros((16,3),np.float32),{},
                variant='fmt_only',seed=7068,device=torch.device('cpu'),output_dir=root)
            self.assertEqual(result['epochs_executed'],2)
            with np.load(root/'predictions/fmt_only_seed7068.npz') as p:
                np.testing.assert_array_equal(p['targets'],labels)
                self.assertEqual(p['logits'].shape,(16,4))
            self.assertFalse(any(p.suffix in {'.pt','.pth','.ckpt'} for p in root.rglob('*')))


if __name__=='__main__':
    unittest.main()
