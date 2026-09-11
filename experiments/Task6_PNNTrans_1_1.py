"""Versioned Task6 Point-NN/Transformer experiment with a pre-test validation gate."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.PrimitiveVAE_3D import geometry_metrics
from FMT_Utils.Task6PNNTrans_3D import predict, train_model
from experiments.Task6_DirectNeural_5_1 import (read, provenance, prepare, load_data,
                                              load_truth, gpu)

DEFAULT = 'config/Verify_Task6_PNNTrans_1.1.json'


def fit_check(spec, config, index):
    dataset = spec['datasets'][index]
    folder = Path(spec['output_root'])/'fit_check'/dataset
    folder.mkdir(parents=True, exist_ok=False)
    device = gpu()
    y, _, subset_hash = load_data(spec, config, dataset, spec['search_seed'], spec['fit_check']['samples'], validation=False)
    training = dict(spec['training'], **{k: spec['fit_check'][k] for k in ('updates', 'batch_size', 'probe_every')})
    candidate = dict(id='fit_only', learning_rate=spec['fit_check']['learning_rate'], dropout=0., weight_decay=0.)
    rows = []
    for arm in spec['arms']:
        model, result = train_model(y, y, arm, candidate, training, spec['search_seed'], device, folder/arm)
        ratio = result['train_rmse_r']/max(result['curve'][0]['train_rmse_r'], 1e-12)
        zero_ratio = result['train_zero_latent_rmse_r']/max(result['train_rmse_r'], 1e-12)
        rows.append(dict(arm=arm, train_rmse_r=result['train_rmse_r'], error_ratio=ratio,
            zero_latent_error_ratio=zero_ratio, passed=ratio < spec['fit_check']['maximum_error_ratio']
            and zero_ratio > spec['fit_check']['minimum_zero_latent_error_ratio']))
        del model
        torch.cuda.empty_cache()
    output = dict(provenance=provenance(config), dataset=dataset, subset_sha256=subset_hash,
        device=torch.cuda.get_device_name(0), test_read=False, rows=rows, passed=all(r['passed'] for r in rows))
    write_json(folder/'result.json', output)
    assert output['passed'], 'Training-only fitting check failed'


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
        model, result = train_model(y, vy, arm, candidate, spec['training'], spec['search_seed'], device, folder/arm)
        rows.append(dict(arm=arm, candidate=candidate['id'], train_rmse_r=result['train_rmse_r'],
                         validation_rmse_r=result['validation_rmse_r'], selected_step=result['selected_step']))
        del model
        torch.cuda.empty_cache()
    write_json(folder/'result.json', dict(provenance=provenance(config), dataset=dataset,
        device=torch.cuda.get_device_name(0), subset_sha256=subset_hash, rows=rows, test_read=False))


def select(spec, config, index=0):
    root = Path(spec['output_root'])
    if (root/'selection.json').exists() or (root/'selection.before_test.json').exists():
        raise FileExistsError('Frozen selection must not be replaced')
    rows = []
    for dataset in spec['datasets']:
        for candidate in spec['candidates']:
            result = read(root/'search'/dataset/candidate['id']/'result.json')
            assert not result['test_read'] and result['provenance']['config_sha256'] == sha256(config)
            assert sorted(r['arm'] for r in result['rows']) == sorted(spec['arms'])
            rows.extend(dict(dataset=dataset, **r) for r in result['rows'])
    scores = {c['id']: float(np.mean([r['validation_rmse_r'] for r in rows
        if r['candidate'] == c['id'] and r['arm'] == 'pnn_trans'])) for c in spec['candidates']}
    assert np.isfinite(list(scores.values())).all()
    candidate = min(spec['candidates'], key=lambda c: (scores[c['id']], c['id']))
    selected = [r for r in rows if r['candidate'] == candidate['id']]
    assert len(selected) == len(spec['datasets'])*len(spec['arms'])
    failures = [r for r in selected if not np.isfinite(r['validation_rmse_r'])
                or r['validation_rmse_r'] >= spec['validation_gate_rmse_r']]
    result = dict(provenance=provenance(config), candidate=candidate, scores=scores, rows=rows,
        gate_passed=not failures, failures=failures, test_read=False, rule=spec['selection'])
    write_json(root/'selection.json', result)
    shutil.copyfile(root/'selection.json', root/'selection.before_test.json')
    print(json.dumps(result, indent=2), flush=True)
    # A negative scientific result is recorded, not an exception hiding the report.
    # The final submission command separately requires gate_passed=True.


def frozen_selection(spec, config):
    root = Path(spec['output_root'])
    assert sha256(root/'selection.json') == sha256(root/'selection.before_test.json')
    selection = read(root/'selection.json')
    assert not selection['test_read'] and selection['provenance']['config_sha256'] == sha256(config)
    assert selection['gate_passed'], 'Validation gate failed: final training/testing forbidden'
    return selection


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
    selection = frozen_selection(spec, config)
    dataset, n, seed = [(d, n, s) for d in spec['datasets'] for n in spec['train_sizes'] for s in spec['seeds']][index]
    root = Path(spec['output_root'])
    folder = root/'final'/dataset/str(n)/str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    device = gpu()
    y, vy, subset_hash = load_data(spec, config, dataset, seed, n)
    fits, failures = [], []
    for arm in spec['arms']:
        fit_dir = folder/arm
        model, result = train_model(y, vy, arm, selection['candidate'], spec['training'], seed, device, fit_dir)
        fit = dict(arm=arm, train_rmse_r=result['train_rmse_r'], validation_rmse_r=result['validation_rmse_r'],
                   selected_step=result['selected_step'])
        if result['validation_rmse_r'] >= spec['validation_gate_rmse_r']:
            fit.update(reason='Validation failure; no test read', test_read=False)
            failures.append(fit)
        else:
            fit['metrics'] = evaluate(spec, dataset, fit_dir, lambda geometry: predict(model, geometry, device))
            fits.append(fit)
        del model
        torch.cuda.empty_cache()
    write_json(folder/'result.json', dict(provenance=provenance(config), dataset=dataset, train_size=n,
        seed=seed, device=torch.cuda.get_device_name(0), subset_sha256=subset_hash,
        selection_sha256=sha256(root/'selection.json'), fits=fits, failures=failures,
        complete=True, passed=not failures, model_files='none'))


def audit(spec, config, index):
    frozen_selection(spec, config)
    root, dataset = Path(spec['output_root']), spec['datasets'][index]
    rows, failures = [], []
    for n in spec['train_sizes']:
        for seed in spec['seeds']:
            folder = root/'final'/dataset/str(n)/str(seed)
            result = read(folder/'result.json')
            assert result['complete'] and result['provenance']['config_sha256'] == sha256(config)
            assert result['selection_sha256'] == sha256(root/'selection.json')
            assert sorted(r['arm'] for r in result['fits']+result['failures']) == sorted(spec['arms'])
            _, _, subset_hash = load_data(spec, config, dataset, seed, n, validation=False)
            assert result['subset_sha256'] == subset_hash
            failures.extend(dict(dataset=dataset, train_size=n, seed=seed, **f) for f in result['failures'])
            for failed in result['failures']:
                assert not list((folder/failed['arm']).glob('*predictions.npz'))
            for fit in result['fits']:
                for role in ('test', 'unseen_scale'):
                    truth, scales = load_truth(spec, dataset, role)
                    with np.load(folder/fit['arm']/f'{role}_predictions.npz') as data:
                        prediction = data['prediction'].copy()
                    records = [r for r in fit['metrics'] if r['role'] == role]
                    assert sorted(r['scale_id'] for r in records) == [-1]+sorted(np.unique(scales).tolist())
                    for record in records:
                        mask = np.ones(len(truth), bool) if record['scale_id'] == -1 else scales == record['scale_id']
                        delta = prediction[mask, :, 1:].astype(np.float64)-truth[mask, :, 1:].astype(np.float64)
                        independent = float(np.sqrt(np.sum(delta**2)/(len(delta)*7*31)))
                        np.testing.assert_allclose(record['position_rmse_r'], independent, rtol=1e-10, atol=1e-12)
                        fresh = geometry_metrics(prediction[mask], truth[mask])
                        for key, value in fresh.items():
                            np.testing.assert_allclose(record[key], value, rtol=1e-10, atol=1e-12)
                        rows.append(dict(dataset=dataset, train_size=n, seed=seed, arm=fit['arm'],
                            role=role, scale_id=record['scale_id'], **{k:v for k,v in fresh.items() if not isinstance(v, list)}))
    output = root/'audit'
    output.mkdir(exist_ok=True)
    write_json(output/f'{dataset}.json', dict(provenance=provenance(config), audit_passed=True, rows=rows, failures=failures))


def merge(spec, config, index=0):
    frozen_selection(spec, config)
    root = Path(spec['output_root'])
    rows, failures = [], []
    for dataset in spec['datasets']:
        audit_result = read(root/'audit'/f'{dataset}.json')
        assert audit_result['audit_passed'] and audit_result['provenance']['config_sha256'] == sha256(config)
        rows.extend(audit_result['rows'])
        failures.extend(audit_result['failures'])
    for suffix in ('*.pt', '*.pth', '*.ckpt'):
        assert not list(root.rglob(suffix))
    with (root/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for n in spec['train_sizes']:
        for arm in spec['arms']:
            for role in ('test', 'unseen_scale'):
                values = [r['position_rmse_r'] for r in rows if r['train_size'] == n
                          and r['arm'] == arm and r['role'] == role and r['scale_id'] == -1]
                complete = len(values) == len(spec['datasets'])*len(spec['seeds'])
                summary.append(dict(train_size=n, arm=arm, role=role, evaluated=len(values), complete=complete,
                    position_rmse_r=float(np.mean(values)) if complete else None))
    write_json(root/'final_audit.json', dict(provenance=provenance(config), audit_passed=True,
        complete_without_failures=not failures, failures=failures, summary=summary, rows=len(rows)))


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
