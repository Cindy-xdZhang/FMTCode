"""Independent direct-neural Task6 experiment; no analytic inverse in fitting."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import traceback

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.PrimitiveVAE_3D import geometry_metrics
from FMT_Utils.Task6DirectNeural_3D import direct_inputs, predict_direct, train_direct
from FMT_Utils.Task6PCABaseline_3D import GeometryPCABaseline

DEFAULT = 'config/Verify_Task6_DirectNeural_5.1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def provenance(config):
    return dict(config_sha256=sha256(config),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        node=socket.gethostname(), job_id=os.getenv('SLURM_JOB_ID'),
        array_id=os.getenv('SLURM_ARRAY_TASK_ID'),
        time=datetime.datetime.now().astimezone().isoformat())


def assert_sources(spec):
    for key in ('source', 'subset'):
        assert sha256(Path(spec[key+'_cache'])/'config.frozen.json') == spec[key+'_config_sha256']


def prepare(spec, config, index):
    assert_sources(spec)
    dataset = spec['datasets'][index]
    source = Path(spec['source_cache'])/'data'/dataset
    subset = Path(spec['subset_cache'])/'data'/dataset
    sm, small = read(source/'manifest.json'), read(subset/'manifest.json')
    assert read(source/'audit.json')['passed']
    assert small['source_manifest_sha256'] == sha256(source/'manifest.json')
    assert small['provenance']['config_sha256'] == spec['subset_config_sha256']
    folder = Path(spec['output_root'])/'data'/dataset
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in ['validation.npz']+[f'train_{seed}.npz' for seed in [spec['search_seed']]+spec['seeds']]:
        path = subset/name
        assert sha256(path) == small['files'][name]
        files[name] = dict(path=str(path), sha256=small['files'][name])
    assert small['source_train_sha256'] == sm['files']['train']['sha256']
    assert small['source_validation_sha256'] == sm['files']['validation']['sha256']
    write_json(folder/'manifest.json', dict(dataset=dataset, provenance=provenance(config), files=files,
        source_manifest_sha256=sha256(source/'manifest.json'),
        subset_manifest_sha256=sha256(subset/'manifest.json'),
        test_read=False, policy='Read-only reuse of frozen 4.1 nested training subsets and validation'))


def load_data(spec, config, dataset, seed, count, validation=True):
    manifest = read(Path(spec['output_root'])/'data'/dataset/'manifest.json')
    assert manifest['provenance']['config_sha256'] == sha256(config)
    arrays = []
    names = [f'train_{seed}.npz']+(['validation.npz'] if validation else [])
    for name in names:
        item = manifest['files'][name]
        assert sha256(item['path']) == item['sha256']
        with np.load(item['path']) as data:
            arrays.append(data['geometry'][:count].copy() if name.startswith('train') else data['geometry'].copy())
            if name.startswith('train'):
                ids = data['source_indices'][:count].copy()
    assert len(arrays[0]) == count and len(set(ids.tolist())) == count
    return arrays[0], arrays[1] if validation else None, hashlib.sha256(ids.tobytes()).hexdigest()


def gpu():
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '4')))
    if not torch.cuda.is_available():
        raise RuntimeError('Research training requires a GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda')


def fit_check(spec, config, index):
    dataset = spec['datasets'][index]
    folder = Path(spec['output_root'])/'fit_check'/dataset
    folder.mkdir(parents=True, exist_ok=False)
    device = gpu()
    y, _, subset_hash = load_data(spec, config, dataset, spec['search_seed'], spec['fit_check']['samples'], validation=False)
    training = dict(spec['training'], **{k: spec['fit_check'][k] for k in ('updates', 'batch_size', 'probe_every')})
    candidate = dict(id='memorization', learning_rate=spec['fit_check']['learning_rate'], dropout=0., weight_decay=0.)
    rows = []
    for arm in spec['arms']:
        model, result = train_direct(y, y, arm, candidate, training, spec['search_seed'], device, folder/arm)
        initial = result['curve'][0]['train_rmse_r']
        ratio = result['train_rmse_r']/max(initial, 1e-12)
        zero_ratio = result['train_zero_latent_rmse_r']/max(result['train_rmse_r'], 1e-12)
        passed = ratio < spec['fit_check']['maximum_error_ratio'] and zero_ratio > spec['fit_check']['minimum_zero_latent_error_ratio']
        rows.append(dict(arm=arm, initial_rmse_r=initial, trained_rmse_r=result['train_rmse_r'],
            error_ratio=ratio, zero_latent_error_ratio=zero_ratio, passed=passed))
        del model
        torch.cuda.empty_cache()
    result = dict(provenance=provenance(config), dataset=dataset, device=torch.cuda.get_device_name(0),
        subset_sha256=subset_hash, passed=all(r['passed'] for r in rows), rows=rows, test_read=False)
    write_json(folder/'result.json', result)
    assert result['passed'], 'Real-data memorization prerequisite failed; dependent jobs must stop'


def search(spec, config, index):
    dataset, candidate = [(d, c) for d in spec['datasets'] for c in spec['candidates']][index]
    root = Path(spec['output_root'])
    assert read(root/'fit_check'/dataset/'result.json')['passed']
    folder = root/'search'/dataset/candidate['id']
    folder.mkdir(parents=True, exist_ok=False)
    device = gpu()
    y, vy, subset_hash = load_data(spec, config, dataset, spec['search_seed'], spec['primary_train_size'])
    rows = []
    for arm in spec['arms']:
        model, result = train_direct(y, vy, arm, candidate, spec['training'], spec['search_seed'], device, folder/arm)
        rows.append(dict(arm=arm, candidate=candidate['id'], validation_rmse_r=result['validation_rmse_r'],
            train_rmse_r=result['train_rmse_r'], selected_step=result['selected_step']))
        del model
        torch.cuda.empty_cache()
    write_json(folder/'result.json', dict(provenance=provenance(config), dataset=dataset,
        device=torch.cuda.get_device_name(0), subset_sha256=subset_hash, rows=rows, test_read=False))


def select(spec, config, index=0):
    root = Path(spec['output_root'])
    if (root/'selection.json').exists() or (root/'selection.before_test.json').exists():
        raise FileExistsError('Selection may not be overwritten')
    rows, scores = [], {}
    for d in spec['datasets']:
        for c in spec['candidates']:
            result = read(root/'search'/d/c['id']/'result.json')
            assert result['test_read'] is False and result['provenance']['config_sha256'] == sha256(config)
            assert {r['arm'] for r in result['rows']} == set(spec['arms'])
            rows.extend(dict(dataset=d, **r) for r in result['rows'])
    for arm in spec['arms']:
        for c in spec['candidates']:
            values = [r['validation_rmse_r'] for r in rows if r['arm'] == arm and r['candidate'] == c['id']]
            assert len(values) == len(spec['datasets']) and np.isfinite(values).all()
            scores[arm+'_'+c['id']] = float(np.mean(values))
    choices = {arm: min(spec['candidates'], key=lambda c: (scores[arm+'_'+c['id']], c['id'])) for arm in spec['arms']}
    result = dict(provenance=provenance(config), fmt=choices['signed_fmt16_neural'],
        raw_matched=choices['signed_fmt16_neural'], raw_selected=choices['raw_neural'],
        scores=scores, rows=rows, test_read=False, rule=spec['selection'])
    write_json(root/'selection.json', result)
    shutil.copyfile(root/'selection.json', root/'selection.before_test.json')
    print(json.dumps(result, indent=2), flush=True)


def frozen_selection(spec, config):
    root = Path(spec['output_root'])
    assert sha256(root/'selection.json') == sha256(root/'selection.before_test.json')
    selection = read(root/'selection.json')
    assert selection['test_read'] is False and selection['provenance']['config_sha256'] == sha256(config)
    return selection


def model_plan(selection):
    unique = {}
    for role, arm in [('fmt', 'signed_fmt16_neural'), ('raw_matched', 'raw_neural'), ('raw_selected', 'raw_neural')]:
        candidate = selection[role]
        unique.setdefault((arm, candidate['id']), dict(arm=arm, candidate=candidate, roles=[]))['roles'].append(role)
    return list(unique.values())


def load_truth(spec, dataset, role):
    source = Path(spec['source_cache'])/'data'/dataset
    manifest = read(source/'manifest.json')
    frozen = read(Path(spec['output_root'])/'data'/dataset/'manifest.json')
    assert sha256(source/'manifest.json') == frozen['source_manifest_sha256']
    path = source/f'{role}.npz'
    assert sha256(path) == manifest['files'][role]['sha256']
    with np.load(path) as data:
        return data['geometry'].copy(), data['scale_id'].copy()


def evaluate(spec, dataset, folder, predictor):
    rows = []
    for role in ('test', 'unseen_scale'):
        y, scales = load_truth(spec, dataset, role)
        prediction = predictor(y)
        np.savez(folder/f'{role}_predictions.npz', prediction=prediction)
        for sid in [-1]+sorted(np.unique(scales).tolist()):
            mask = np.ones(len(y), bool) if sid == -1 else scales == sid
            rows.append(dict(role=role, scale_id=sid, **geometry_metrics(prediction[mask], y[mask])))
    return rows


def final(spec, config, index):
    root = Path(spec['output_root'])
    selection = frozen_selection(spec, config)
    dataset, n, seed = [(d, n, s) for d in spec['datasets'] for n in spec['train_sizes'] for s in spec['seeds']][index]
    folder = root/'final'/dataset/str(n)/str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    device = gpu()
    y, vy, subset_hash = load_data(spec, config, dataset, seed, n)
    fits, failures = [], []
    for item in model_plan(selection):
        name = item['arm']+'_'+item['candidate']['id']
        fit_dir = folder/name
        model, training = train_direct(y, vy, item['arm'], item['candidate'], spec['training'], seed, device, fit_dir)
        if training['validation_rmse_r'] >= spec['validation_gate_rmse_r']:
            failure = dict(name=name, roles=item['roles'], validation_rmse_r=training['validation_rmse_r'],
                reason='Validation engineering failure; no test read for this model')
            write_json(fit_dir/'validation_failed.json', failure)
            failures.append(failure)
        else:
            metrics = evaluate(spec, dataset, fit_dir,
                lambda truth: predict_direct(model, direct_inputs(truth, item['arm']), device))
            fits.append(dict(name=name, **item, metrics=metrics, selected_step=training['selected_step'],
                validation_rmse_r=training['validation_rmse_r'], train_rmse_r=training['train_rmse_r']))
        del model
        torch.cuda.empty_cache()
    pca_dir = folder/'geometry_pca'
    pca_dir.mkdir()
    pca = GeometryPCABaseline(y, spec['training']['latent_dim'])
    pca_metadata = dict(train_samples=len(y), latent_dimension=pca.basis.shape[1], neural_dependency=False,
        subset_sha256=subset_hash, source='train geometry only')
    write_json(pca_dir/'fit.json', pca_metadata)
    fits.append(dict(name='geometry_pca', roles=['pca'], metrics=evaluate(spec, dataset, pca_dir, pca.predict)))
    write_json(folder/'result.json', dict(provenance=provenance(config), dataset=dataset, train_size=n, seed=seed,
        device=torch.cuda.get_device_name(0), subset_sha256=subset_hash, selection_sha256=sha256(root/'selection.json'),
        fits=fits, failures=failures, complete=True, passed=not failures, model_files='none'))


def audit(spec, config, index):
    root = Path(spec['output_root'])
    frozen_selection(spec, config)
    dataset = spec['datasets'][index]
    rows, failures, model_count, expected_rows = [], [], 0, 0
    for n in spec['train_sizes']:
        for seed in spec['seeds']:
            folder = root/'final'/dataset/str(n)/str(seed)
            result = read(folder/'result.json')
            assert result['complete'] and result['provenance']['config_sha256'] == sha256(config)
            assert result['selection_sha256'] == sha256(root/'selection.json')
            _, _, subset_hash = load_data(spec, config, dataset, seed, n, validation=False)
            assert result['subset_sha256'] == subset_hash
            failures.extend(dict(dataset=dataset, train_size=n, seed=seed, **f) for f in result['failures'])
            roles = [r for f in result['fits']+result['failures'] for r in f['roles']]
            assert sorted(roles) == ['fmt', 'pca', 'raw_matched', 'raw_selected']
            model_count += len(result['fits'])
            for fit in result['fits']:
                for role in ('test', 'unseen_scale'):
                    y, scales = load_truth(spec, dataset, role)
                    with np.load(folder/fit['name']/f'{role}_predictions.npz') as data:
                        prediction = data['prediction']
                    records = [m for m in fit['metrics'] if m['role'] == role]
                    assert sorted(m['scale_id'] for m in records) == [-1]+sorted(np.unique(scales).tolist())
                    expected_rows += len(records)*len(fit['roles'])
                    for metric in records:
                        sid = metric['scale_id']
                        mask = np.ones(len(y), bool) if sid == -1 else scales == sid
                        fresh = geometry_metrics(prediction[mask], y[mask])
                        delta = prediction[mask].astype(np.float64)[:, :, 1:]-y[mask].astype(np.float64)[:, :, 1:]
                        independent = float(np.sqrt(np.sum(delta*delta)/(len(delta)*7*31)))
                        np.testing.assert_allclose(independent, fresh['position_rmse_r'], rtol=1e-12)
                        for k, v in fresh.items():
                            np.testing.assert_allclose(metric[k], v, rtol=1e-10, atol=1e-12)
                        for comparison in fit['roles']:
                            rows.append(dict(dataset=dataset, train_size=n, seed=seed, comparison=comparison,
                                role=role, scale_id=sid, **{k: v for k, v in fresh.items() if not isinstance(v, list)}))
    assert len(rows) == expected_rows
    output = root/'audit'
    output.mkdir(exist_ok=True)
    write_json(output/f'{dataset}.json', dict(provenance=provenance(config), dataset=dataset,
        audit_passed=True, experiment_complete_without_failures=not failures,
        models=model_count, rows=rows, failures=failures, selection_sha256=sha256(root/'selection.json')))


def merge(spec, config, index=0):
    root = Path(spec['output_root'])
    frozen_selection(spec, config)
    rows, failures, models = [], [], 0
    for dataset in spec['datasets']:
        data = read(root/'audit'/f'{dataset}.json')
        assert data['audit_passed'] and data['provenance']['config_sha256'] == sha256(config)
        assert data['selection_sha256'] == sha256(root/'selection.json')
        rows.extend(data['rows'])
        failures.extend(data['failures'])
        models += data['models']
    for suffix in ('*.pt', '*.pth', '*.ckpt'):
        assert not any(root.rglob(suffix))
    keys = [(r['dataset'], r['train_size'], r['seed'], r['comparison'], r['role'], r['scale_id']) for r in rows]
    assert len(keys) == len(set(keys))
    with (root/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root/'final_audit.json', dict(passed=not failures, audit_passed=True, failures=failures,
        provenance=provenance(config), models=models, rows=len(rows), selection_sha256=sha256(root/'selection.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'fit_check', 'search', 'select', 'final', 'audit', 'merge'])
    parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    spec = read(args.config)
    root = Path(spec['output_root'])
    assert sha256(args.config) == sha256(root/'config.frozen.json')
    events = root/'events'
    events.mkdir(exist_ok=True)
    event_path = events/f'{args.phase}_{args.index}.json'
    if event_path.exists():
        raise FileExistsError(event_path)
    event = dict(phase=args.phase, index=args.index, started=provenance(args.config), status='RUNNING')
    if args.phase in ('fit_check', 'search', 'final') and torch.cuda.is_available():
        event['gpu'] = torch.cuda.get_device_name(0)
    write_json(event_path, event)
    try:
        globals()[args.phase](spec, args.config, args.index)
        event['status'] = 'COMPLETED'
    except BaseException:
        event.update(status='FAILED', traceback=traceback.format_exc())
        raise
    finally:
        event['ended'] = datetime.datetime.now().astimezone().isoformat()
        write_json(event_path, event)


if __name__ == '__main__':
    main()
