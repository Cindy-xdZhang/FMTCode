"""Synthetic pipeline fixtures; no research data or scheduler submission."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256,write_json
from experiments import Task6_PNNTuning_1_2 as run
from tests.test_task6_direct_neural_5_1 import examples
from tests.test_task6_pnn_tuning_1_2 import candidate


class TestPipeline(unittest.TestCase):
    def test_two_stage_selection_and_three_final_arms(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(temp)
            dataset='unit_fixture'
            source,subset,baseline,root=[base/name for name in ('source','subset','baseline','result')]
            src,sub=source/'data'/dataset,subset/'data'/dataset
            for path in (src,sub,baseline,root):
                path.mkdir(parents=True)
            for path in (source/'config.frozen.json',subset/'config.frozen.json',baseline/'config.frozen.json'):
                write_json(path,dict(unit_fixture=True,path=str(path)))
            files={}
            for i,role in enumerate(('train','validation','test','unseen_scale')):
                y=examples(24 if role=='train' else 8,100+i)
                np.savez(src/f'{role}.npz',geometry=y,scale_id=np.arange(len(y))%2)
                files[role]=dict(sha256=sha256(src/f'{role}.npz'))
            write_json(src/'manifest.json',dict(files=files))
            write_json(src/'audit.json',dict(passed=True))
            small_files={}
            for seed in (91,92):
                ids=np.random.default_rng(seed).permutation(24)[:16]
                with np.load(src/'train.npz') as data:
                    np.savez(sub/f'train_{seed}.npz',geometry=data['geometry'][ids],source_indices=ids)
                small_files[f'train_{seed}.npz']=sha256(sub/f'train_{seed}.npz')
            with np.load(src/'validation.npz') as data:
                np.savez(sub/'validation.npz',geometry=data['geometry'])
            small_files['validation.npz']=sha256(sub/'validation.npz')
            write_json(sub/'manifest.json',dict(files=small_files,provenance=dict(config_sha256=sha256(subset/'config.frozen.json')),
                source_manifest_sha256=sha256(src/'manifest.json'),source_train_sha256=files['train']['sha256'],
                source_validation_sha256=files['validation']['sha256']))
            for name in ('selection.json','selection.before_test.json'):
                write_json(baseline/name,dict(test_read=False,rows=[dict(dataset=dataset,candidate='c1',arm='raw_trans',validation_rmse_r=1.)]))
            c=candidate()
            c['geometry_mix_fraction']=.5
            training=dict(updates=100,batch_size=16,warmup_updates=10,probe_every=50,gradient_clip=1.)
            spec=dict(experiment='unit_fixture',output_root=str(root),source_cache=str(source),subset_cache=str(subset),
                source_config_sha256=sha256(source/'config.frozen.json'),subset_config_sha256=sha256(subset/'config.frozen.json'),
                baseline_cache=str(baseline),baseline_config_sha256=sha256(baseline/'config.frozen.json'),
                baseline_selection_sha256=sha256(baseline/'selection.json'),datasets=[dataset],screen_datasets=[dataset],
                arms=['pnn_trans','raw_trans','raw_frozen'],train_sizes=[16],primary_train_size=16,search_seed=91,seeds=[92],
                training=training,screening=training,candidates=[c],promote_count=1,validation_gate_rmse_r=3.,selection='fixture',
                baseline_candidate=dict(id='c1',learning_rate=.003,dropout=0.,weight_decay=0.),
                baseline_training=dict(training,architecture=c['architecture']))
            config=base/'config.json'
            write_json(config,spec)
            with patch.object(run,'gpu',return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name',return_value='CPU fixture'), \
                 patch.object(run,'evaluate',side_effect=AssertionError('Premature test read')):
                run.prepare(spec,config)
                run.screen(spec,config,0)
                run.promote(spec,config)
                run.refine(spec,config,0)
                run.select(spec,config)
            selection=run.read(root/'selection.json')
            self.assertFalse(selection['test_read'])
            self.assertEqual(selection['candidate']['id'],c['id'])
            self.assertEqual(sha256(root/'selection.json'),sha256(root/'selection.before_test.json'))
            with self.assertRaises(AssertionError):
                run.promote(spec,config)
            # The gate arithmetic is separately tested. Construct a positive gate fixture
            # to exercise all three training/evaluation code paths regardless of toy-model ranking.
            positive=dict(selection,gate_passed=True,unit_test_gate_fixture=True)
            for name in ('selection.json','selection.before_test.json'):
                write_json(root/name,positive)
            with patch.object(run,'gpu',return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name',return_value='CPU fixture'):
                run.final(spec,config,0)
            result=run.read(root/'final'/dataset/'16'/'92'/'result.json')
            self.assertEqual(sorted(r['arm'] for r in result['fits']),['pnn_trans','raw_frozen','raw_trans'])
            run.audit(spec,config,0)
            run.merge(spec,config)
            self.assertTrue(run.read(root/'final_audit.json')['audit_passed'])
            self.assertFalse(list(root.rglob('*.pt')))


if __name__=='__main__':
    unittest.main()
