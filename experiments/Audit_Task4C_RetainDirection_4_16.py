"""Check the complete retained feature blocks, baseline, and saved predictions."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import Task4C_RetainDirection_4_16 as run
from experiments import Audit_Task4C_Encoders_4_15 as previous_audit
from FMT_Utils.Task4C_RetainDirection_4_16 import WIDTHS, BLOCKS, encode_lines, make_model
from FMT_Utils.Task4C_LinePooling_4_3 import LearnedLinePooling
from FMT_Utils.Task4C_Encoders_4_15 import encode_lines as local_encode


def preflight(spec, config):
    root = Path(spec['output']); device = torch.device('cuda')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    run.check_parent(spec)
    fixtures = {e: {} for e in spec['encoders']}; checks = []
    for flow in spec['flows']:
        name = flow['name']; parent = Path(spec['parent_output'])
        src = parent / 'physical' / name / 'train'
        with np.load(src / 'metadata.npz') as z:
            meta = {k: z[k] for k in z.files}
        chosen = np.concatenate([np.flatnonzero(meta['labels'] == y)[:32] for y in (0, 1)])
        if len(chosen) != 64:
            raise ValueError('Expected 64 stratified training-only examples')
        def tensor(path):
            return torch.as_tensor(np.array(np.load(path, mmap_mode='r')[chosen]), device=device)
        geometry = tensor(src / 'geometry.npy'); seeds = tensor(src / 'seeds.npy')
        original = tensor(parent / name / 'train/fmt.npy')
        counts = torch.as_tensor(meta['counts'][chosen].astype(np.int64), device=device)
        direct_v5 = local_encode(geometry, seeds, counts, 'fmt_v5')
        direct_obj = local_encode(geometry, seeds, counts, 'fmt_objective_ntod_v2')
        for encoder in spec['encoders']:
            result = encode_lines(geometry, seeds, counts, encoder, original)
            if encoder == 'fmt_4_14':
                assert torch.equal(result, original)
            elif encoder == 'fmt_v5_4_14':
                assert torch.equal(result[..., :233], original[..., :233])
                assert torch.equal(result[..., 233:398], direct_v5[..., 161:326])
            else:
                assert torch.equal(result[..., :231], direct_obj[..., :231])
                assert torch.equal(result[..., 231:303], original[..., 161:233])
                assert torch.equal(result[..., 303:468], direct_obj[..., 231:396])
            assert torch.equal(result[..., -1], original[..., -1])
            fixtures[encoder][name] = dict(features=result.cpu().numpy(),
                metadata={k: v[chosen] for k, v in meta.items() if v.ndim and len(v) == len(meta['labels'])})
        checks.append(dict(flow=name, training_rows=chosen.tolist(), original_233_exact=True,
            appended_165_exact=True, objective_original_direction_72_exact=True, mask_exact=True))
    parameters = {}
    for encoder in spec['encoders']:
        torch.manual_seed(96611); model = make_model(encoder, .15)
        torch.manual_seed(96611); original = LearnedLinePooling(.15)
        for key, value in original.state_dict().items():
            if encoder == 'fmt_4_14' or not key.startswith('line.0.'):
                assert torch.equal(value, model.state_dict()[key])
        parameters[encoder] = sum(p.numel() for p in model.parameters())
        pilot = copy.deepcopy(spec); pilot['output'] = str(root / 'engineering_pilot')
        pilot['training']['seeds'] = [96611]; pilot['training']['epochs'] = 2; pilot['training']['patience'] = 2
        pilot['expected_training_samples'] = 64
        local = run.variant_spec(pilot, encoder); base = Path(local['output']); base.mkdir(parents=True)
        manifest = dict(identity=run.identity(config), splits={})
        for flow in spec['flows']:
            fixture = fixtures[encoder][flow['name']]
            for split, offsets in (('train', np.r_[0:16, 32:48]), ('validation', np.r_[16:24, 48:56]), ('test', np.r_[24:32, 56:64])):
                key = flow['name'] + '/' + split; dest = base / key; dest.mkdir(parents=True)
                np.save(dest / 'fmt.npy', fixture['features'][offsets])
                np.savez_compressed(dest / 'metadata.npz', **{k: v[offsets] for k, v in fixture['metadata'].items()})
                manifest['splits'][key] = dict(files={n: run.pipeline.sha(dest / n) for n in ('fmt.npy', 'metadata.npz')})
        run.pipeline.write(base / 'encoding.json', manifest)
        run.training.identity = run.identity
        run.training.make_model = lambda method, candidate: make_model(encoder, candidate['dropout'])
        run.training.train(local, config, 0)
        result = json.loads((run.folder(pilot, encoder, 96611) / 'result.json').read_text())
        assert result['weights_written'] == 0 and result['test_examples'] == 32
    run.pipeline.write(root / 'preflight.json', dict(status='PASS', identity=run.identity(config), checks=checks,
        parameters=parameters, feature_blocks=BLOCKS, training_only_engineering_fixture=True,
        original_model_initialization_exact=True, unchanged_hidden_layer_initialization_exact=True,
        new_first_layer_default_initialization=True, frozen_training_function_reused=True))


def audit(spec, config):
    # Reuse the independent metric implementation, with the explicit new runner.
    previous_audit.run = run
    previous_audit.audit(spec, config)
    root = Path(spec['output']); parent = Path(spec['parent_output'])
    (root / 'independent_audit.json').rename(root / 'prediction_metrics_audit.json')
    retained = []
    for flow in spec['flows']:
        for split in run.SPLITS:
            key = flow['name'] + '/' + split
            old = np.load(parent / key / 'fmt.npy', mmap_mode='r')
            v5 = np.load(Path(run.variant_spec(spec, 'fmt_v5_4_14')['output']) / key / 'fmt.npy', mmap_mode='r')
            obj = np.load(Path(run.variant_spec(spec, 'fmt_objective_ntod_v2_4_14')['output']) / key / 'fmt.npy', mmap_mode='r')
            for start in range(0, len(old), 256):
                sl = slice(start, start+256)
                assert np.array_equal(old[sl, :, :233], v5[sl, :, :233])
                assert np.array_equal(old[sl, :, 161:233], obj[sl, :, 231:303])
                assert np.array_equal(v5[sl, :, 233:398], obj[sl, :, 303:468])
                assert np.array_equal(old[sl, :, -1], v5[sl, :, -1])
                assert np.array_equal(old[sl, :, -1], obj[sl, :, -1])
            retained.append(dict(split=key, rows=len(old), original_233_and_direction_72_exact=True))
    reference = []
    for seed in spec['training']['seeds']:
        fresh = json.loads((run.folder(spec, 'fmt_4_14', seed) / 'result.json').read_text())
        original = json.loads((parent / 'runs' / f'regularized_fmt_mlp_seed{seed}' / 'result.json').read_text())
        if fresh['normalization'] != original['normalization']:
            raise ValueError('Baseline normalization no longer reproduces 4.14')
        error = max(abs(fresh[s]['pooled']['f1']-original[s]['pooled']['f1']) for s in ('training', 'validation', 'test'))
        # Preserve differences as evidence rather than silently accepting a shifted baseline.
        if error > 1e-12:
            raise ValueError('Baseline failed exact 4.14 F1 reproduction: ' + str(error))
        for encoder, size in (('fmt_v5_4_14', 233),):
            extended = json.loads((run.folder(spec, encoder, seed) / 'result.json').read_text())
            for stat in ('mean', 'std'):
                assert np.array_equal(extended['normalization'][stat][:size], original['normalization'][stat])
        conv_path = parent / 'runs' / f'regularized_conv3d_mlp_seed{seed}' / 'result.json'
        conv = json.loads(conv_path.read_text())
        assert conv['predictions_sha256'] == run.pipeline.sha(conv_path.parent / 'predictions.npz')
        with np.load(conv_path.parent / 'predictions.npz') as z:
            for split, field in (('train', 'training'), ('validation', 'validation'), ('test', 'test')):
                mm = previous_audit.metric(z[split+'_labels'], z[split+'_probability'])
                assert abs(mm['f1']-conv[field]['pooled']['f1']) < 1e-12
        reference.append(dict(seed=seed, original_fmt_f1_max_error=error,
            conv3d_reused_result_sha256=run.pipeline.sha(conv_path), conv3d_predictions_sha256=conv['predictions_sha256']))
    result = json.loads((root / 'prediction_metrics_audit.json').read_text())
    result.update(retained_feature_checks=retained, original_4_14_reference_checks=reference,
        original_233_normalization_exact=True, original_fmt_baseline_reproduced=True,
        reused_conv3d_predictions_independently_recomputed=True)
    run.pipeline.write(root / 'independent_audit.json', result)
