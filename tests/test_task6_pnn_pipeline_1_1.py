"""Exercise source hashes, pre-test gate, final evaluation and metric corruption."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json
from experiments import Task6_PNNTrans_1_1 as run
from tests.test_task6_direct_neural_5_1 import examples


class TestPipeline(unittest.TestCase):
    def test_frozen_data_selection_gate_and_prediction_audit(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            base = Path(temp)
            dataset = 'unit_fixture'
            source, subset, root = base/'source', base/'subset', base/'result'
            src, sub = source/'data'/dataset, subset/'data'/dataset
            src.mkdir(parents=True)
            sub.mkdir(parents=True)
            root.mkdir()
            for path in (source/'config.frozen.json', subset/'config.frozen.json'):
                write_json(path, dict(unit_fixture=True, source=str(path.parent)))
            files = {}
            for index, role in enumerate(('train','validation','test','unseen_scale')):
                y = examples(24 if role == 'train' else 10, 100+index)
                np.savez(src/f'{role}.npz', geometry=y, scale_id=np.arange(len(y))%2,
                         primitive_id=np.arange(len(y))+index*1000)
                files[role] = dict(sha256=sha256(src/f'{role}.npz'))
            write_json(src/'manifest.json', dict(files=files))
            write_json(src/'audit.json', dict(passed=True))
            small_files = {}
            for seed in (91,92):
                ids = np.random.default_rng(seed).permutation(24)[:16]
                with np.load(src/'train.npz') as data:
                    np.savez(sub/f'train_{seed}.npz', geometry=data['geometry'][ids], source_indices=ids)
                small_files[f'train_{seed}.npz'] = sha256(sub/f'train_{seed}.npz')
            with np.load(src/'validation.npz') as data:
                np.savez(sub/'validation.npz', geometry=data['geometry'])
            small_files['validation.npz'] = sha256(sub/'validation.npz')
            write_json(sub/'manifest.json', dict(files=small_files,
                provenance=dict(config_sha256=sha256(subset/'config.frozen.json')),
                source_manifest_sha256=sha256(src/'manifest.json'), source_train_sha256=files['train']['sha256'],
                source_validation_sha256=files['validation']['sha256']))
            spec = dict(experiment='unit_fixture', output_root=str(root), source_cache=str(source), subset_cache=str(subset),
                source_config_sha256=sha256(source/'config.frozen.json'), subset_config_sha256=sha256(subset/'config.frozen.json'),
                datasets=[dataset], train_sizes=[16], primary_train_size=16, search_seed=91, seeds=[92],
                arms=['pnn_trans','raw_trans'], training=dict(architecture=dict(width=16, heads=2, layers=1,
                    latent_dim=8, decoder_width=24, decoder_blocks=1, frequencies=4), batch_size=16, updates=100,
                    warmup_updates=10, probe_every=50, gradient_clip=1.),
                candidates=[dict(id='c0', learning_rate=.003, dropout=0., weight_decay=0.)],
                fit_check=dict(samples=16, updates=350, batch_size=16, probe_every=175, learning_rate=.003,
                               maximum_error_ratio=.15, minimum_zero_latent_error_ratio=1.1),
                validation_gate_rmse_r=3., selection='unit fixture validation only')
            config = base/'config.json'
            write_json(config, spec)
            run.prepare(spec, config, 0)
            with patch.object(run, 'gpu', return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name', return_value='CPU fixture'), \
                 patch.object(run, 'load_truth', side_effect=AssertionError('Premature test read')):
                run.fit_check(spec, config, 0)
                run.search(spec, config, 0)
                run.select(spec, config)
            self.assertTrue(run.frozen_selection(spec, config)['gate_passed'])
            with self.assertRaises(FileExistsError):
                run.select(spec, config)
            # Even if a tampered selection and snapshot agree, a failed gate forbids final entry.
            selection_bytes = (root/'selection.json').read_bytes()
            failed = run.read(root/'selection.json')
            failed['gate_passed'] = False
            for name in ('selection.json','selection.before_test.json'):
                write_json(root/name, failed)
            with self.assertRaises(AssertionError):
                run.frozen_selection(spec, config)
            for name in ('selection.json','selection.before_test.json'):
                (root/name).write_bytes(selection_bytes)
            with patch.object(run, 'gpu', return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name', return_value='CPU fixture'):
                run.final(spec, config, 0)
            run.audit(spec, config, 0)
            run.merge(spec, config)
            self.assertTrue(run.read(root/'final_audit.json')['audit_passed'])
            self.assertTrue(all(r['complete'] for r in run.read(root/'final_audit.json')['summary']))
            self.assertFalse(list(root.rglob('*.pt')))
            path = root/'final'/dataset/'16'/'92'/'pnn_trans'/'test_predictions.npz'
            with np.load(path) as data:
                changed = data['prediction'].copy()
            changed[:,:,1:] += .1
            np.savez(path, prediction=changed)
            with self.assertRaises(AssertionError):
                run.audit(spec, config, 0)


if __name__ == '__main__':
    unittest.main()
