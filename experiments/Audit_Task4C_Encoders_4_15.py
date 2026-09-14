"""Engineering checks and independent saved-prediction checks for Task4-c 4.15."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from experiments import Task4C_Encoders_4_15 as run
from FMT_Utils.Task4C_Encoders_4_15 import ENCODERS, encode_lines, local_indices, make_model
from FMT_Utils.Task4C_LinePooling_4_3 import line_fmt, make_model as old_model, bundle_voxels


def metric(y, p, threshold=.5):
    y = np.asarray(y).astype(bool); p = np.asarray(p)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid saved probability')
    predicted = p >= threshold
    tp = int(np.sum(y & predicted)); fp = int(np.sum(~y & predicted)); fn = int(np.sum(y & ~predicted))
    return dict(f1=2*tp/max(2*tp+fp+fn, 1), average_precision=float(average_precision_score(y, p)), tp=tp, fp=fp, fn=fn)


def preflight(spec, config):
    root = Path(spec['output']); device = torch.device('cuda')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    run.check_parent(spec)
    checks = []; fixtures = {e: {} for e in spec['encoders']}
    for flow in spec['flows']:
        name = flow['name']; src = Path(spec['parent_output']) / 'physical' / name / 'train'
        with np.load(src / 'metadata.npz') as z:
            meta = {k: z[k] for k in z.files}
        # Deterministic stratified training-only fixtures; never use test to debug.
        chosen = np.concatenate([np.flatnonzero(meta['labels'] == y)[:32] for y in (0, 1)])
        if len(chosen) != 64:
            raise ValueError('Engineering fixture needs 32 examples per class')
        gg = np.array(np.load(src / 'geometry.npy', mmap_mode='r')[chosen])
        ss = np.array(np.load(src / 'seeds.npy', mmap_mode='r')[chosen])
        geometry = torch.as_tensor(gg, device=device); seeds = torch.as_tensor(ss, device=device)
        counts = torch.as_tensor(meta['counts'][chosen].astype(np.int64), device=device)
        ids, mask = local_indices(geometry, seeds, counts)
        old = line_fmt(geometry, seeds, counts)
        fresh = encode_lines(geometry, seeds, counts, 'fmt_v5')
        if not torch.equal(fresh[..., :161], old[..., :161]):
            raise ValueError('Frozen 4.14 local FMT prefix changed')
        for encoder in spec['encoders']:
            if encoder == 'conv3d':
                value = bundle_voxels(geometry, counts).cpu().numpy().astype(np.float16)
                cached = np.load(Path(spec['parent_output']) / name / 'train/voxels.npy', mmap_mode='r')[chosen]
                if not np.array_equal(value, cached):
                    raise ValueError('Frozen Conv3D voxel input differs')
            else:
                token = encode_lines(geometry, seeds, counts, encoder)
                if torch.any(token[~mask] != 0) or not torch.equal(token[..., -1].bool(), mask):
                    raise ValueError('Invalid padding mask')
                bi, li = mask.nonzero(as_tuple=True)
                primitive = geometry[bi[:20, None], ids[bi[:20], li[:20]]]
                direct = ENCODERS[encoder][0](primitive, return_numpy=False)
                if not torch.equal(token[bi[:20], li[:20], :-1], direct):
                    raise ValueError('Hidden or changed feature block')
                value = token.cpu().numpy()
            fixtures[encoder][name] = dict(features=value, metadata={k: v[chosen] for k, v in meta.items() if v.ndim and len(v) == len(meta['labels'])})
        # Check scalar objectivity with fixed seven-line correspondence.
        bi, li = mask.nonzero(as_tuple=True)
        primitive = geometry[bi[:20, None], ids[bi[:20], li[:20]]].double()
        angle = torch.linspace(-.9, 1.2, 32, dtype=torch.float64, device=device)
        q = torch.zeros((32, 3, 3), dtype=torch.float64, device=device)
        q[:, 0, 0] = q[:, 1, 1] = angle.cos(); q[:, 0, 1] = -angle.sin(); q[:, 1, 0] = angle.sin(); q[:, 2, 2] = 1
        moved = torch.einsum('tij,bntj->bnti', q, primitive) + torch.stack((angle, angle.square(), angle.sin()), -1)[None, None]
        objective = spec['encoders'][1]; function = ENCODERS[objective][0]
        before, after = function(primitive, return_numpy=False), function(moved, return_numpy=False)
        error = float((before-after).abs().max())
        if not torch.allclose(before, after, atol=2e-9, rtol=2e-9):
            raise ValueError('Scalar encoder changed under a rigid observer with fixed correspondence')
        checks.append(dict(flow=name, training_rows=chosen.tolist(), frozen_fmt_prefix_exact=True,
            voxel_cache_exact=True, masked_padding_zero=True, direct_encoder_only=True, fixed_correspondence_objectivity_max_error=error))
    models = {}
    for encoder in spec['encoders']:
        torch.manual_seed(96611); model = make_model(encoder, .15).to(device)
        if encoder == 'conv3d':
            torch.manual_seed(96611)
            original = old_model('conv3d_mlp', dict(dropout=.15, architecture='learned_pool_wider_conv_4.3')).to(device)
            if any(not torch.equal(v, original.state_dict()[k]) for k, v in model.state_dict().items()):
                raise ValueError('Frozen Conv3D initialization changed')
        models[encoder] = sum(p.numel() for p in model.parameters())
        del model
        # Exercise the unchanged training function, selection lock and inference.
        pilot = copy.deepcopy(spec); pilot['output'] = str(root / 'engineering_pilot')
        pilot['training']['seeds'] = [96611]; pilot['training']['epochs'] = 2; pilot['training']['patience'] = 2
        pilot['expected_training_samples'] = 64
        local = run.variant_spec(pilot, encoder); base = Path(local['output']); base.mkdir(parents=True)
        manifest = dict(identity=run.identity(config), splits={})
        for flow in spec['flows']:
            fixture = fixtures[encoder][flow['name']]
            for split, offsets in (('train', np.r_[0:16, 32:48]), ('validation', np.r_[16:24, 48:56]), ('test', np.r_[24:32, 56:64])):
                key = flow['name'] + '/' + split; dest = base / key; dest.mkdir(parents=True)
                filename = 'voxels.npy' if encoder == 'conv3d' else 'fmt.npy'
                np.save(dest / filename, fixture['features'][offsets])
                np.savez_compressed(dest / 'metadata.npz', **{k: v[offsets] for k, v in fixture['metadata'].items()})
                manifest['splits'][key] = dict(files={n: run.pipeline.sha(dest / n) for n in (filename, 'metadata.npz')})
        run.pipeline.write(base / 'encoding.json', manifest)
        run.training.identity = run.identity
        run.training.make_model = lambda method, candidate: make_model(encoder, candidate['dropout'])
        run.training.train(local, config, 0)
        result = json.loads((run.folder(pilot, encoder, 96611) / 'result.json').read_text())
        if result['weights_written'] != 0 or result['test_examples'] != 32:
            raise ValueError('Engineering training/inference contract changed')
    run.pipeline.write(root / 'preflight.json', dict(status='PASS', identity=run.identity(config), checks=checks,
        parameters=models, training_only_engineering_fixture=True, scientific_metrics=False,
        frozen_training_function_reused=True, no_weight_files=True))


def audit(spec, config):
    root = Path(spec['output']); records = []
    # Full metadata identities are checked against physical 4.14 sources.
    metadata = {}
    for split in run.SPLITS:
        pieces = []
        for i, flow in enumerate(spec['flows']):
            path = Path(spec['parent_output']) / 'physical' / flow['name'] / split / 'metadata.npz'
            with np.load(path) as z:
                part = {k: z[k] for k in ('labels', 'head_component', 'scale_id')}
            part['flow_index'] = np.full(len(part['labels']), i, np.int8); pieces.append(part)
        metadata[split] = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    for encoder in spec['encoders']:
        local = run.variant_spec(spec, encoder); base = Path(local['output'])
        manifest = json.loads((base / 'encoding.json').read_text())
        if manifest['identity']['config_sha256'] != run.pipeline.sha(config):
            raise ValueError('Wrong encoding identity')
        for key, entry in manifest['splits'].items():
            for name, h in entry['files'].items():
                if run.pipeline.sha(base / key / name) != h:
                    raise ValueError('Encoded artifact changed')
            if run.pipeline.sha(base / key / 'metadata.npz') != run.pipeline.sha(Path(spec['parent_output']) / 'physical' / key / 'metadata.npz'):
                raise ValueError('Sample identity changed between methods')
        for seed in spec['training']['seeds']:
            folder = run.folder(spec, encoder, seed)
            r = json.loads((folder / 'result.json').read_text())
            lock = json.loads((folder / 'selection.lock.json').read_text())
            if r['predictions_sha256'] != run.pipeline.sha(folder / 'predictions.npz') or r['selection_lock_sha256'] != run.pipeline.sha(folder / 'selection.lock.json'):
                raise ValueError('Prediction or selection-lock checksum changed')
            if r['encoder'] != encoder or r['feature_blocks'] != (['voxels'] if encoder == 'conv3d' else [encoder]):
                raise ValueError('Encoder identity differs')
            history = [json.loads(x) for x in (folder / 'history.jsonl').read_text().splitlines()]
            best = max(history, key=lambda h: (h['validation_f1'], h['validation_average_precision']))
            if best['epoch'] != r['selected_epoch'] or lock['selected_epoch'] != best['epoch'] or lock['test_loaded']:
                raise ValueError('Validation-only model selection failed')
            rng = np.random.default_rng(seed)
            for h in history:
                order = rng.permutation(spec['expected_training_samples'])
                if h['training_examples'] != len(order) or h['permutation_sha256'] != hashlib.sha256(order.astype('<i8').tobytes()).hexdigest():
                    raise ValueError('Training row coverage or seed changed')
            errors = []
            with np.load(folder / 'predictions.npz') as z:
                for split, field in (('train', 'training'), ('validation', 'validation'), ('test', 'test')):
                    for key, expected in metadata[split].items():
                        if not np.array_equal(z[split+'_'+key], expected):
                            raise ValueError('Predictions do not match physical sample identities')
                    y, p = z[split+'_labels'], z[split+'_probability']
                    mm = metric(y, p)
                    for key in ('f1', 'average_precision'):
                        errors.append(abs(mm[key]-r[field]['pooled'][key]))
                    for i, flow in enumerate(spec['flows']):
                        mask = z[split+'_flow_index'] == i; fm = metric(y[mask], p[mask])
                        for key in ('f1', 'average_precision'):
                            errors.append(abs(fm[key]-r[field]['per_flow'][flow['name']][key]))
                    for scale in np.unique(z[split+'_scale_id']):
                        mask = z[split+'_scale_id'] == scale; sm = metric(y[mask], p[mask])
                        for key in ('f1', 'average_precision'):
                            errors.append(abs(sm[key]-r[field]['per_scale'][str(scale)][key]))
                supplemental = metric(z['test_labels'], z['test_probability'], r['validation_selected_threshold'])
                errors.append(abs(supplemental['f1']-r['test_with_validation_selected_threshold']['pooled']['f1']))
            if max(errors) > 1e-12 or r['weights_written'] != 0:
                raise ValueError('Independent metric or weight-file check failed')
            records.append(dict(encoder=encoder, seed=seed, max_metric_error=max(errors), selected_epoch=best['epoch'],
                parameters=r['parameters'], prediction_sha256=r['predictions_sha256']))
    summary = json.loads((root / 'summary.json').read_text())
    for row in summary['rows']:
        rr = [json.loads((run.folder(spec, row['encoder'], s) / 'result.json').read_text()) for s in spec['training']['seeds']]
        for split in ('training', 'validation', 'test'):
            values = [r[split]['pooled']['f1'] for r in rr]
            if abs(row[split]['f1_mean']-np.mean(values)) > 1e-12 or abs(row[split]['f1_std']-np.std(values, ddof=1)) > 1e-12:
                raise ValueError('Summary differs from all three registered seeds')
    weights = [str(p) for suffix in ('*.pt', '*.pth', '*.ckpt') for p in root.rglob(suffix)]
    if weights:
        raise ValueError('Unexpected saved model weights')
    run.pipeline.write(root / 'independent_audit.json', dict(status='PASS', identity=run.identity(config),
        runs=records, sample_counts={s: len(m['labels']) for s, m in metadata.items()},
        summary_sha256=run.pipeline.sha(root / 'summary.json'), unchanged_4_14_samples=True,
        weights_found=weights, validation_only_selection=True, all_training_rows_each_epoch=True))
