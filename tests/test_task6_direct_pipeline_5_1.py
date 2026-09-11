"""Tiny synthetic fixtures exercise the deployment pipeline, never research data."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments import Task6_DirectNeural_5_1 as run
from FMT_Utils.FlowMapData_3D import sha256, write_json
from tests.test_task6_direct_neural_5_1 import examples


class TestPipeline(unittest.TestCase):
    def test_end_to_end_selection_and_independent_prediction_audit(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            base = Path(temp)
            dataset = 'synthetic_unit_fixture'
            source, subset, root = base/'source', base/'subset', base/'result'
            src, sub = source/'data'/dataset, subset/'data'/dataset
            src.mkdir(parents=True)
            sub.mkdir(parents=True)
            root.mkdir()
            for path in (source/'config.frozen.json', subset/'config.frozen.json'):
                write_json(path, dict(unit_test_fixture=True, purpose=path.parent.name))
            files = {}
            for index, role in enumerate(('train', 'validation', 'test', 'unseen_scale')):
                y = examples(24 if role == 'train' else 10, 100+index)
                np.savez(src/f'{role}.npz', geometry=y, scale_id=np.arange(len(y))%2,
                    primitive_id=np.arange(len(y))+index*1000)
                files[role] = dict(sha256=sha256(src/f'{role}.npz'))
            write_json(src/'manifest.json', dict(files=files))
            write_json(src/'audit.json', dict(passed=True))
            small_files = {}
            for seed in (91, 92):
                ids = np.random.default_rng(seed).permutation(24)[:16]
                with np.load(src/'train.npz') as data:
                    np.savez(sub/f'train_{seed}.npz', geometry=data['geometry'][ids], source_indices=ids)
                small_files[f'train_{seed}.npz'] = sha256(sub/f'train_{seed}.npz')
            with np.load(src/'validation.npz') as data:
                np.savez(sub/'validation.npz', geometry=data['geometry'])
            small_files['validation.npz'] = sha256(sub/'validation.npz')
            write_json(sub/'manifest.json', dict(files=small_files,
                provenance=dict(config_sha256=sha256(subset/'config.frozen.json')),
                source_manifest_sha256=sha256(src/'manifest.json'),
                source_train_sha256=files['train']['sha256'], source_validation_sha256=files['validation']['sha256']))
            spec = dict(experiment='synthetic_unit_fixture', output_root=str(root), source_cache=str(source),
                subset_cache=str(subset), source_config_sha256=sha256(source/'config.frozen.json'),
                subset_config_sha256=sha256(subset/'config.frozen.json'), datasets=[dataset],
                train_sizes=[16], primary_train_size=16, search_seed=91, seeds=[92],
                arms=['signed_fmt16_neural', 'raw_neural'],
                training=dict(width=24, blocks=1, latent_dim=8, batch_size=16, updates=90,
                    probe_every=30, beta=1e-5, gradient_clip=1.),
                candidates=[dict(id='c0', learning_rate=.003, dropout=0., weight_decay=0.)],
                fit_check=dict(samples=16, updates=250, batch_size=16, probe_every=125,
                    learning_rate=.003, maximum_error_ratio=.1, minimum_zero_latent_error_ratio=1.1),
                validation_gate_rmse_r=3., selection='fixture validation only')
            config = base/'config.json'
            write_json(config, spec)
            run.prepare(spec, config, 0)
            # Fitting/search/selection must not open test or unseen-scale truth.
            with patch.object(run, 'gpu', return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name', return_value='CPU unit-test fixture'), \
                 patch.object(run, 'load_truth', side_effect=AssertionError('Premature test read')):
                run.fit_check(spec, config, 0)
                run.search(spec, config, 0)
                run.select(spec, config)
            self.assertTrue(run.read(root/'fit_check'/dataset/'result.json')['passed'])
            self.assertEqual(sha256(root/'selection.json'), sha256(root/'selection.before_test.json'))
            self.assertEqual(len(run.model_plan(run.frozen_selection(spec, config))), 2)
            with self.assertRaises(FileExistsError):
                run.select(spec, config)
            with patch.object(run, 'gpu', return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name', return_value='CPU unit-test fixture'):
                run.final(spec, config, 0)
            run.audit(spec, config, 0)
            run.merge(spec, config)
            result = run.read(root/'final_audit.json')
            self.assertTrue(result['passed'])
            self.assertEqual(result['models'], 3)  # FMT + deduplicated Raw + independent PCA
            self.assertEqual(result['rows'], 24)  # four comparison roles, two splits, three scale rows
            self.assertFalse(list(root.rglob('*.pt')))
            # Selection edits are rejected; prediction corruption is independently detected.
            original = (root/'selection.json').read_bytes()
            (root/'selection.json').write_bytes(original+b' ')
            with self.assertRaises(AssertionError):
                run.frozen_selection(spec, config)
            (root/'selection.json').write_bytes(original)
            target = root/'final'/dataset/'16'/'92'/'geometry_pca'/'test_predictions.npz'
            with np.load(target) as data:
                damaged = data['prediction'].copy()
            damaged[:, :, 1:] += .1
            np.savez(target, prediction=damaged)
            with self.assertRaises(AssertionError):
                run.audit(spec, config, 0)


if __name__ == '__main__':
    unittest.main()
