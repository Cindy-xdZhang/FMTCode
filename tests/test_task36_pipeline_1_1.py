"""Run real label attachment, all controls, final evaluation and tamper detection."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import netCDF4 as nc
import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256,write_json
from FMT_Utils.Task36Labels_3D import load_labels
from FMT_Utils.Task36MultiGate_3D import VARIANTS
from experiments import Task36_MultiGate_1_1 as run
from tests.test_task6_direct_neural_5_1 import examples
from tests.test_task36_multigate_3d import joint_candidate


class TestTask36Pipeline(unittest.TestCase):
    def test_complete_label_and_joint_training_pipeline(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(temp)
            dataset='fixture'
            source,subset,root=[base/n for n in ('source','subset','result')]
            src,sub=source/'data'/dataset,subset/'data'/dataset
            for path in (src,sub,root):
                path.mkdir(parents=True)
            for path in (source/'config.frozen.json',subset/'config.frozen.json'):
                write_json(path,dict(fixture=True,path=str(path)))
            fieldpath=base/'flow.nc'
            with nc.Dataset(fieldpath,'w') as data:
                for name,values in [('t',np.arange(6)),('x',np.linspace(-1,1,21)),('y',np.linspace(-1,1,5)),('z',np.linspace(-1,1,5))]:
                    data.createDimension(name,len(values))
                    data.createVariable(name,'f8',(name,))[:]=values
                z,y,x=np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5),np.linspace(-1,1,21),indexing='ij')
                for name,value in [('u',x*0),('v',x*x),('w',z*0)]:
                    data.createVariable(name,'f4',('t','z','y','x'))[:]=np.broadcast_to(value,(6,5,5,21))
            files={}
            for i,role in enumerate(('train','validation','test','unseen_scale')):
                y=examples(48 if role=='train' else 12,100+i)
                start=0 if role=='train' else 2 if role=='validation' else 4
                origins=np.zeros((len(y),3))
                origins[:,0]=np.arange(len(y))%2
                np.savez(src/f'{role}.npz',geometry=y,scale_id=np.arange(len(y))%3,origin=origins,
                    seed_time=np.full(len(y),start),source_start=np.full(len(y),start),
                    primitive_id=np.arange(len(y))+i*100)
                files[role]=dict(sha256=sha256(src/f'{role}.npz'))
            write_json(src/'manifest.json',dict(files=files,source=dict(source=str(fieldpath),
                source_bytes=fieldpath.stat().st_size,source_mtime_ns=fieldpath.stat().st_mtime_ns)))
            write_json(src/'audit.json',dict(passed=True))
            small={}
            for seed in (91,92):
                ids=np.random.default_rng(seed).permutation(48)[:32]
                with np.load(src/'train.npz') as data:
                    np.savez(sub/f'train_{seed}.npz',geometry=data['geometry'][ids],source_indices=ids)
                small[f'train_{seed}.npz']=sha256(sub/f'train_{seed}.npz')
            with np.load(src/'validation.npz') as data:
                np.savez(sub/'validation.npz',geometry=data['geometry'])
            small['validation.npz']=sha256(sub/'validation.npz')
            write_json(sub/'manifest.json',dict(files=small,provenance=dict(config_sha256=sha256(subset/'config.frozen.json')),
                source_manifest_sha256=sha256(src/'manifest.json'),source_train_sha256=files['train']['sha256'],
                source_validation_sha256=files['validation']['sha256']))
            spec=dict(experiment='fixture',output_root=str(root),source_cache=str(source),subset_cache=str(subset),
                source_config_sha256=sha256(source/'config.frozen.json'),subset_config_sha256=sha256(subset/'config.frozen.json'),
                primary_train_size=24,datasets=[dataset],search_seed=91,seeds=[92],variants=list(VARIANTS),
                label_max_spatial_dim=96,candidate=joint_candidate(),validation_gate_rmse_r=3.,
                training=dict(updates=80,batch_size=8,warmup_updates=5,probe_every=40,gradient_clip=1.))
            config=base/'config.json'
            write_json(config,spec)
            (root/'config.frozen.json').write_bytes(config.read_bytes())
            with patch.object(run,'gpu',return_value=torch.device('cpu')), \
                 patch.object(run,'evaluate',side_effect=AssertionError('Premature test read')):
                run.prepare(spec,config,0)
                self.assertFalse((root/'labels'/dataset/'test.npz').exists())
                with np.load(sub/'train_91.npz') as data:
                    expected=data['source_indices'][:24]%2
                np.testing.assert_array_equal(load_labels(spec,config,dataset,'train_91'),expected)
                for index in range(len(VARIANTS)):
                    run.search(spec,config,index)
                run.select(spec,config)
            selection=run.read(root/'selection.json')
            self.assertTrue(selection['gate_passed'])
            self.assertFalse(selection['test_read'])
            run.prepare_test(spec,config,0)
            with patch.object(run,'gpu',return_value=torch.device('cpu')):
                for index in range(len(VARIANTS)):
                    run.final(spec,config,index)
            run.audit(spec,config,0)
            run.merge(spec,config)
            result=run.read(root/'final_audit.json')
            self.assertTrue(result['audit_passed'])
            self.assertFalse(result['failures'])
            self.assertEqual(len(result['summary']),14)
            self.assertTrue(all(row['complete'] for row in result['summary']))
            path=root/'final'/dataset/'92'/'joint_task_gates'/'test_predictions.npz'
            with np.load(path) as data:
                predictions={k:data[k].copy() for k in data.files}
            predictions['prediction'][0,0,1,0]+=1
            np.savez(path,**predictions)
            with self.assertRaises(AssertionError):
                run.audit(spec,config,0)
            self.assertFalse(list(root.rglob('*.pt')))


if __name__=='__main__':
    unittest.main()
