"""Deliberate geometric-baseline misconfiguration, never a replacement baseline.

Each shard fits every training variant before opening confirmation performance.
Geometry and threshold swaps then probe the frozen reference model separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from FMT_Utils.GeometricControls_3D import (
    array_hash, geometric_sequences, pad_auxiliary, stressed_seed_ivd,
)
from FMT_Utils.Task12Evaluation_3D import calibrate_vortex_cluster
from FMT_Utils.VAE_3D import FeatureVAE3D, vae_loss
from experiments.Run_Task135_GeometricControls import (
    load_records, sources, sha, write_json, write_csv, metrics, load_residual,
)
from experiments.Verify_Task3_FMTClassifier import _select_f1_threshold, _normalize_train_only, _loader
from experiments.Verify_Task3_FMTResidual import _train_one, _predict_components, _probabilities


def jobs(spec):
    return [(task, dataset, seed) for task in spec['tasks']
            for dataset in spec['datasets'] for seed in spec['seeds'][task]]


def records(spec, task, dataset, role):
    source_task = 'Task3' if task == 'Task2' else task
    phase = 'confirmation' if task == 'Task1' and role == 'confirmation' else 'development'
    return load_records(spec, source_task, dataset, phase, spec[task.lower()][role])


def feature_rows(rows, parameters):
    return np.concatenate([stressed_seed_ivd(r, **parameters) for r in rows])


def verify_sources(rows, certificate, task):
    known = {(c['task'], c['path']): c['source_sha256'] for c in certificate['certificates']}
    for row in rows:
        if sha(row['path']) != known[(task, row['path'])]:
            raise ValueError('source cache changed after preflight')


def verify_code():
    manifest = json.loads(Path('DEPLOYMENT.json').read_text())
    for path, expected in manifest['files'].items():
        if sha(path) != expected:
            raise ValueError(f'deployed source changed: {path}')
    return manifest['commit']


def preflight(spec, config):
    root = Path(spec['output_root'])
    if (root/'preflight.json').exists():
        raise FileExistsError('do not overwrite preflight evidence')
    commit = verify_code()
    selection = json.loads(Path(spec['vae']['reference_selection']).read_text())
    if selection['winner']['architecture'] != 'l512_l64_b1e-6':
        raise ValueError('historical frozen VAE architecture changed')
    certificates = []
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            seen = set()
            for role in ('train', 'validation', 'confirmation'):
                rr = records(spec, task, dataset, role)
                for r in rr:
                    if r['path'] in seen:
                        raise ValueError('source-file overlap across splits')
                    seen.add(r['path'])
                    certificates.append({'task': task, 'dataset': dataset, 'role': role,
                        'path': r['path'], 'source_sha256': sha(r['path']),
                        'identity': r['identity'], **r['certificate']})
                if role == 'train':
                    ref = geometric_sequences(rr[0]['raw'], rr[0]['times'], rr[0]['scale_id'])[:, 0, :1]
                    np.testing.assert_array_equal(ref, stressed_seed_ivd(rr[0]))
                    for variant in spec['geometry_variants']:
                        value = feature_rows(rr[:1], variant['parameters'])
                        if not np.isfinite(value).all():
                            raise ValueError('nonfinite diagnostic features')
            if task == 'Task3':
                _, _, directory, _ = sources(spec, task, dataset, 'development')
                for seed in spec['seeds'][task]:
                    if not (directory/f'{dataset}_raw_seed{seed}.pt').is_file():
                        raise FileNotFoundError('missing frozen Raw backbone')
            print(f'preflight {task}/{dataset}: PASS; no confirmation metrics', flush=True)
    write_json(root/'preflight.json', {'status': 'PASS', 'code_commit': commit,
        'config_sha256': sha(config), 'certificates': certificates,
        'vae_selection_sha256': sha(spec['vae']['reference_selection']),
        'confirmation_metrics_computed': False})


def cluster_score(model, cluster, x):
    distance = model.transform(x)
    score = (distance[:, 1-cluster]-distance[:, cluster]).astype(np.float64)
    if cluster == 1:
        score[score == 0] = np.nextafter(0., -1.)
    return score


@torch.no_grad()
def encode(model, x, device):
    model.eval()
    return np.concatenate([model.encode(torch.from_numpy(x[i:i+4096]).to(device))[0].cpu().numpy()
                           for i in range(0, len(x), 4096)])


def train_vae(x, settings, seed, device):
    """Same FeatureVAE3D/AdamW/batching/loss as the frozen nonlinear VAE.

    No relational or PCA-anchor loss is used by the frozen l512_l64 recipe.
    Models remain in memory; no VAE checkpoint is written.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    model = FeatureVAE3D(x.shape[1], settings['hidden_dims'], settings['latent_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=settings['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=False)
    sums = np.zeros(3); count = 0; step = 0; model.train()
    while step < settings['optimizer_steps']:
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, rec, kl = vae_loss(recon, batch, mu, logvar, settings['beta'])
            if not torch.isfinite(loss): raise ValueError('nonfinite VAE loss; retain failure, do not score as zero')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            sums += len(batch)*np.asarray([loss.item(), rec.item(), kl.item()])
            count += len(batch); step += 1
            if step >= settings['optimizer_steps']: break
        if step % 1000 < len(loader): print(f'VAE steps={step}/{settings["optimizer_steps"]}', flush=True)
    return model.eval(), {'steps': step, 'train_loss': sums[0]/count,
                         'train_reconstruction': sums[1]/count, 'train_kl': sums[2]/count,
                         'parameter_count': sum(p.numel() for p in model.parameters())}


def fit_model(spec, task, dataset, seed, variant, train, val, directory, device):
    tr_raw, tr_f, tr_y = train; va_raw, va_f, va_y = val
    name = variant['id']; started = time.perf_counter(); temporary = []
    metadata = {'id': name, 'overrides': variant['overrides']}
    if task in ('Task1', 'Task2'):
        scaler = StandardScaler().fit(tr_f)
        tr = scaler.transform(tr_f).astype(np.float32)
        va = scaler.transform(va_f).astype(np.float32)
        km_settings = dict(spec['kmeans'])
        if task == 'Task1':
            km_settings.update(variant['overrides'])
            transform = lambda f: scaler.transform(f).astype(np.float32)
        else:
            settings = {**spec['vae'], **variant['overrides']}
            tr = pad_auxiliary(tr, spec['task2_input_width'])
            va = pad_auxiliary(va, spec['task2_input_width'])
            vae, losses = train_vae(tr, settings, seed, device)
            tr, va = encode(vae, tr, device), encode(vae, va, device)
            transform = lambda f: encode(vae, pad_auxiliary(scaler.transform(f).astype(np.float32),
                                                        spec['task2_input_width']), device)
            metadata.update(settings=settings, **losses)
        kmseed = 7080+spec['seeds'][task].index(seed) if task == 'Task1' else 7068
        km = KMeans(n_clusters=2, random_state=kmseed, **km_settings).fit(tr)
        cluster = calibrate_vortex_cluster(va_y, km.predict(va))
        predict = lambda raw, f: cluster_score(km, cluster, transform(f))
        train_scores = cluster_score(km, cluster, tr)
        threshold = 0.
        metadata.update(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        kmeans_settings=km_settings, kmeans_seed=kmseed,
                        kmeans_iterations=int(km.n_iter_), cluster=cluster)
    else:
        _, _, backbone_dir, _ = sources(spec, task, dataset, 'development')
        backbone_path = backbone_dir/f'{dataset}_raw_seed{seed}.pt'
        backbone = torch.load(backbone_path, map_location='cpu', weights_only=False)
        width = spec['auxiliary_input_width']
        tr, va, _, stats = _normalize_train_only(
            (tr_raw, pad_auxiliary(tr_f, width), tr_y),
            (va_raw, pad_auxiliary(va_f, width), va_y), None, raw_stats=backbone['normalization'])
        training = {**spec['training'], **variant['overrides']}
        run_spec = {'experiment': spec['experiment'], 'model': spec['model'], 'fusion': spec['fusion'],
            'training': training, 'auxiliary_source': 'fmt', 'raw_checkpoint_dir': str(backbone_dir),
            'raw_wide_parameter_count': 148225, 'raw_backbone_seed': seed,
            'evaluation': {'test_enabled': False}, 'split': spec['task3']}
        directory.mkdir(parents=True)
        result = _train_one(run_spec, dataset, seed, (tr, va, None), stats, device, directory)
        model, state = load_residual(result['checkpoint'], width, device)
        temporary.append(Path(result['checkpoint']))
        def predict(raw, f):
            x = ((raw-np.asarray(stats['raw_mean']))/np.asarray(stats['raw_std'])).astype(np.float32)
            f = ((pad_auxiliary(f, width)-np.asarray(stats['fmt_mean']))/np.asarray(stats['fmt_std'])).astype(np.float32)
            loader = _loader((x, f, np.zeros(len(x), np.float32)), 1024, False, seed, True)
            _, raw_logits, aux_logits = _predict_components(model, loader, device)
            return _probabilities(raw_logits, aux_logits, float(state['alpha']), spec['model'])
        train_scores = predict(tr_raw, tr_f)
        threshold = float(state['threshold'])
        metadata.update(training=training, best_epoch=int(state['best_epoch']), normalization=stats,
                        parameter_count=int(state['total_parameter_count']),
                        backbone_sha256=sha(backbone_path), alpha=float(state['alpha']))
    metadata.update(threshold=threshold, score_std=max(float(np.std(train_scores)), 1e-8),
                    train_seconds=time.perf_counter()-started)
    return predict, metadata, temporary


def feature_stats(f, reference):
    flat = f[:, 0]; ref = reference[:, 0]
    correlation = float(np.corrcoef(flat, ref)[0, 1]) if flat.std() > 0 and ref.std() > 0 else None
    return {'feature_mean': float(flat.mean()), 'feature_std': float(flat.std()),
            'feature_zero_fraction': float(np.mean(flat == 0)), 'feature_correlation_reference': correlation}


def run(spec, config, index):
    task, dataset, seed = jobs(spec)[index]; commit = verify_code()
    root = Path(spec['output_root']); cert = json.loads((root/'preflight.json').read_text())
    if cert['status'] != 'PASS' or cert['config_sha256'] != sha(config): raise ValueError('preflight mismatch')
    target = root/'shards'/task/dataset/f'seed{seed}'
    target.mkdir(parents=True, exist_ok=False)
    device = torch.device('cpu' if task == 'Task1' else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('GPU allocation required')
    write_json(target/'started.json', {'job': os.environ.get('SLURM_JOB_ID'),
        'array_task': os.environ.get('SLURM_ARRAY_TASK_ID'), 'code_commit': commit,
        'node': socket.gethostname(), 'time': time.time(),
        'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})
    tr_records, va_records = [records(spec, task, dataset, role) for role in ('train', 'validation')]
    verify_sources(tr_records+va_records, cert, task)
    def stack(rr):
        return (np.concatenate([r['raw'] for r in rr]), feature_rows(rr, {}),
                np.concatenate([r['labels'] for r in rr]))
    train, val = stack(tr_records), stack(va_records)
    direct = {'threshold': _select_f1_threshold(val[2], val[1][:, 0]),
              'score_std': max(float(np.std(train[1][:, 0])), 1e-8)}
    fitted = {}; temporary = []
    for variant in spec['training_variants'][task]:
        print(f'fit {task}/{dataset}/seed{seed}/{variant["id"]}', flush=True)
        predict, meta, paths = fit_model(spec, task, dataset, seed, variant, train, val,
                                        target/'temporary_training'/variant['id'], device)
        fitted[variant['id']] = (predict, meta); temporary.extend(paths)
    frozen = {'config_sha256': sha(config), 'code_commit': commit, 'direct': direct,
              'training': {k: v[1] for k, v in fitted.items()},
              'confirmation_metrics_computed': False}
    write_json(target/'frozen_before_confirmation.json', frozen)
    del tr_records, va_records, train, val
    rr = records(spec, task, dataset, 'confirmation'); verify_sources(rr, cert, task)
    raw, reference, labels = stack(rr)
    names = []; score_arrays = []; thresholds = []; rows = []; reference_scores = {}
    ref_predict, ref_meta = fitted['reference']
    def add(case, arm, stage, scores, threshold, f, **extra):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != labels.shape or not np.isfinite(scores).all(): raise ValueError('invalid scores')
        row = {'task': task, 'dataset': dataset, 'seed': seed, 'case': case, 'arm': arm,
            'stage': stage, 'threshold': float(threshold), **feature_stats(f, reference),
            **metrics(labels, scores, threshold), **extra}
        decision = scores >= threshold; y = labels.astype(bool)
        row.update(tp=int(np.sum(decision & y)), fp=int(np.sum(decision & ~y)),
                   tn=int(np.sum(~decision & ~y)), fn=int(np.sum(~decision & y)))
        names.append(case+'/'+arm); score_arrays.append(scores); thresholds.append(threshold); rows.append(row)
    for variant in spec['geometry_variants']:
        f = feature_rows(rr, variant['parameters'])
        predicted = {'direct': f[:, 0], 'learned': ref_predict(raw, f)}
        for arm, score in predicted.items():
            meta = direct if arm == 'direct' else ref_meta
            add(variant['id'], arm, 'reference' if variant['id'] == 'reference' else 'inference_geometry_swap',
                score, meta['threshold'], f, parameters_json=json.dumps(variant['parameters'], sort_keys=True))
        if variant['id'] == 'reference': reference_scores = predicted
    for offset in spec['decision_offsets_in_train_score_std']:
        for arm, score in reference_scores.items():
            meta = direct if arm == 'direct' else ref_meta
            add(f'threshold_offset_{offset:g}', arm, 'decision_threshold_swap', score,
                meta['threshold']+offset*meta['score_std'], reference,
                threshold_offset=offset, training_score_std=meta['score_std'])
    for name, (predict, meta) in fitted.items():
        if name == 'reference': continue
        add(name, 'learned', 'training_parameter_swap', predict(raw, reference), meta['threshold'], reference,
            training_seconds=meta['train_seconds'], parameters_json=json.dumps(meta['overrides'], sort_keys=True))
    baselines = {r['arm']: r for r in rows if r['case'] == 'reference'}
    for row in rows:
        for metric in ('f1', 'average_precision', 'iou', 'balanced_accuracy'):
            row[metric+'_drop_from_reference'] = baselines[row['arm']][metric]-row[metric]
    predictions = target/'predictions.npz'
    np.savez_compressed(predictions, labels=labels, scores=np.stack(score_arrays),
        thresholds=np.asarray(thresholds), case_ids=np.asarray(names),
        identities_json=np.asarray(json.dumps([r['identity'] for r in rr])))
    write_csv(target/'per_run.csv', rows)
    for path in temporary:
        if not path.resolve().is_relative_to((target/'temporary_training').resolve()):
            raise ValueError('refuse deletion outside this shard')
        path.unlink()
    write_json(target/'complete.json', {'status': 'COMPLETE', 'config_sha256': sha(config),
        'code_commit': commit, 'rows': len(rows), 'predictions_sha256': sha(predictions),
        'per_run_sha256': sha(target/'per_run.csv'), 'temporary_checkpoints_remaining': 0,
        'node': socket.gethostname(), 'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})
    print(f'complete {task}/{dataset}/seed{seed}: {len(rows)} rows', flush=True)


def summarize(spec, config):
    root = Path(spec['output_root']); all_rows = []
    for task, dataset, seed in jobs(spec):
        target = root/'shards'/task/dataset/f'seed{seed}'
        complete = json.loads((target/'complete.json').read_text())
        if complete['status'] != 'COMPLETE' or complete['config_sha256'] != sha(config): raise ValueError('incomplete shard')
        if complete['per_run_sha256'] != sha(target/'per_run.csv'): raise ValueError('modified metrics')
        with (target/'per_run.csv').open(newline='') as handle: all_rows.extend(csv.DictReader(handle))
    fields = ['f1', 'average_precision', 'iou', 'precision', 'recall', 'balanced_accuracy',
              'predicted_positive_fraction', 'f1_drop_from_reference', 'average_precision_drop_from_reference',
              'iou_drop_from_reference', 'balanced_accuracy_drop_from_reference']
    summary = []
    keys = sorted({(r['task'], r['case'], r['arm'], r['stage']) for r in all_rows})
    for task, case, arm, stage in keys:
        subset = [r for r in all_rows if (r['task'], r['case'], r['arm'], r['stage']) == (task, case, arm, stage)]
        row = {'task': task, 'case': case, 'arm': arm, 'stage': stage}
        for field in fields:
            values = [np.mean([float(r[field]) for r in subset if r['dataset'] == d]) for d in spec['datasets']]
            row[field] = float(np.mean(values))
            row[field+'_dataset_std'] = float(np.std(values, ddof=1))
        summary.append(row)
    write_csv(root/'per_run.csv', all_rows); write_csv(root/'parameter_stress_table.csv', summary)
    write_json(root/'summary.json', {'status': 'COMPLETE_PENDING_INDEPENDENT_AUDIT',
        'purpose': spec['purpose'], 'shards': len(jobs(spec)), 'rows': len(all_rows),
        'config_sha256': sha(config), 'table_sha256': sha(root/'parameter_stress_table.csv')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/Verify_Task123_GeometryParameterStress_1.1.yaml')
    parser.add_argument('--mode', choices=['preflight', 'run', 'summarize'], required=True)
    parser.add_argument('--job-index', type=int)
    args = parser.parse_args(); spec = yaml.safe_load(Path(args.config).read_text())
    if args.mode == 'preflight': preflight(spec, args.config)
    elif args.mode == 'run': run(spec, args.config, args.job_index)
    else: summarize(spec, args.config)
