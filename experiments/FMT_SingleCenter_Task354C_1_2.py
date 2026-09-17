"""Single-seed, fixed-center P35 verification on frozen Task3/5/4-c data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

from FMT_Utils.FMT_SingleCenter_1_1 import encode, SingleCenterMLP
from FMT_Utils.FMT_P35_NormFrequency_3_1 import fit_normalizer, apply_normalizer
from FMT_Utils.FMTNoConvolution_1_1 import assert_no_fmt_convolution
from experiments.Run_Task1235_AngleFeatures_1_1 import records, source_paths, evidence
from experiments.Run_Task135_GeometricControls import sources

CONFIG = 'config/Verify_FMT_SingleCenter_Task354C_1.2.json'
CODE = (
    'FMT_Utils/FMT_SingleCenter_1_1.py', 'FMT_Utils/FMT_P35_NormFrequency_3_1.py',
    'FMT_Utils/DFT_FMT_3D.py', 'FMT_Utils/FMTNoConvolution_1_1.py',
    'FMT_Utils/GeometricControls_3D.py',
    'experiments/Run_Task1235_AngleFeatures_1_1.py', 'experiments/Run_Task135_GeometricControls.py',
    'experiments/FMT_SingleCenter_Task354C_1_2.py', 'tests/test_fmt_single_center_1_1.py',
    'tests/test_fmt_single_center_subset_1_2.py',
    'config/mainExp_Task135_FMTv8_1.1.json', 'config/Verify_Task4C_P35GTHead_1.1.json',
    'ibex_bash/fmt_single_center_task354c_1p2.sh', CONFIG,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def identity(config):
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                config_sha256=sha(config), code={p: sha(p) for p in CODE})


def load(config):
    spec = json.loads(Path(config).read_text())
    parent = json.loads(Path(spec['parent_task35']).read_text())
    source4 = json.loads(Path(spec['parent_task4']).read_text())
    assert spec['seed'] == 40 and spec['candidate'] == 'p35_n0_k06_single_center'
    assert spec['feature_dimensions'] == 141 and spec['neighbors'] == 'all_valid_except_fixed_center'
    source4["subset"] = spec["task4c_subset"]
    source4["expected_counts"] = spec["task4c_subset"]["counts"]
    return spec, parent, source4


def units(parent):
    return [(t, d) for t in ('Task3', 'Task5') for d in parent['datasets']] + [('Task4c', 'channel_tbl')]


def setup(device):
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device == 'cuda':
        assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name()


def metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p)
    assert np.isfinite(p).all() and np.isin(y, [0, 1]).all() and np.all((p >= 0) & (p <= 1))
    pred = p >= threshold
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    fn, tn = int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred).sum())
    return dict(f1=2*tp/max(2*tp+fp+fn, 1), precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                accuracy=(tp+tn)/len(y), average_precision=float(average_precision_score(y, p)) if y.any() else 0.,
                samples=len(y), positives=int(y.sum()), confusion_matrix=[[tn, fp], [fn, tp]])


def source35(parent, task, dataset, role, rows):
    result = evidence(rows)
    phase, rebuilt, _ = source_paths(parent, task, dataset, role)
    if not rebuilt:
        _, label_dir, _, _ = sources(parent, task, dataset, phase)
        for item, row in zip(result, rows):
            label = label_dir / Path(row['path']).name
            item.update(label_path=str(label), label_sha256=sha(label))
    return result


def read35(parent, task, dataset, role, device, batch):
    source_role = 'confirmation' if role == 'test' else role
    rows = records(parent, task, dataset, source_role)
    values = []
    for row in rows:
        assert row['raw'].shape[1:] == (7, 32, 3)
        for first in range(0, len(row['raw']), batch):
            values.append(encode(torch.as_tensor(row['raw'][first:first+batch], device=device)).cpu().numpy())
    ids = dict(labels=np.concatenate([r['labels'] for r in rows]).astype(np.int64),
               scale_id=np.concatenate([r['scale_id'] for r in rows]),
               row_id=np.concatenate([np.column_stack((np.full(len(r['raw']), r['ordinal']), np.arange(len(r['raw'])))) for r in rows]))
    ids['center_id'] = np.zeros(len(ids['labels']), dtype=np.int64)
    return np.concatenate(values), ids, source35(parent, task, dataset, source_role, rows)


def subset_rows(source, flow, role):
    definition = source['subset']['files'][flow][role]
    path = Path(source['subset']['root']) / 'subsets' / f'{flow}_{role}.npz'
    assert sha(path) == definition['sha256'], path
    with np.load(path) as z:
        rows, centers = z['row_ids'].copy(), z['center_ids'].copy()
    assert len(rows) == definition['retained'] and len(centers) == len(rows)
    assert np.all(rows >= 0) and np.all(np.diff(rows) > 0) and np.all(centers >= 0)
    return rows, centers


def verify_subset_centers(source, flow, role):
    """Verify the user's locked subset from geometry only, without test labels."""
    rows, centers = subset_rows(source, flow, role)
    folder = Path(source['source_output'])/'physical'/flow/role
    seeds = np.load(folder/'seeds.npy', mmap_mode='r')
    with np.load(folder/'metadata.npz') as z:
        center = (z['center']-z['centroid'])/z['radius'][:, None]
        counts, radius, spacing = z['counts'], z['radius'], z['neighbor_distance']
    distance = np.linalg.norm(np.asarray(seeds, dtype=np.float64)-center[:, None], axis=-1)
    distance[np.arange(seeds.shape[1])[None] >= counts[:, None]] = np.inf
    ids = distance.argmin(1)
    ratio = distance[np.arange(len(ids)), ids]*radius/spacing
    expected = np.flatnonzero(ratio < 1e-4)
    assert np.array_equal(rows, expected) and np.array_equal(centers, ids[rows])
    return dict(flow=flow, role=role, retained=len(rows), excluded=len(ids)-len(rows),
                rows_sha256=array_sha(rows), centers_sha256=array_sha(centers),
                max_seed_error_over_spacing=float(ratio[rows].max()), nearest_substitution=False)


def read4(source, role, device, batch):
    values, pieces, certificates = [], [], []
    root = Path(source['source_output']) / 'physical'
    for fi, flow in enumerate(('channel', 'tbl')):
        folder = root / flow / role
        expected = source['source_files'][flow][role]
        for name, digest in expected['files'].items():
            assert sha(folder/name) == digest, folder/name
        geometry = np.load(folder/'geometry.npy', mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:
            meta = {k: z[k] for k in z.files}
        rows, centers = subset_rows(source, flow, role)
        assert len(geometry) == expected['samples'] and geometry.shape[2:] == (32, 3)
        for first in range(0, len(rows), batch):
            take = rows[first:first+batch]
            x = torch.tensor(np.array(geometry[take]), device=device)
            values.append(encode(x, meta['counts'][take], centers[first:first+batch]).cpu().numpy())
        pieces.append(dict(labels=meta['labels'][rows].astype(np.int64), center_id=centers,
                           flow_index=np.full(len(rows), fi, dtype=np.int64),
                           row_in_split=rows, instance=meta['instance'][rows]))
        certificates.append(dict(flow=flow, role=role, files={str(folder/k): v for k, v in expected['files'].items()},
                                 samples=len(rows), excluded_missing_center=len(geometry)-len(rows),
                                 center_ids_sha256=array_sha(centers), original_row_ids_sha256=array_sha(rows),
                                 nearest_substitution=False))
    ids = {k: np.concatenate([part[k] for part in pieces]) for k in pieces[0]}
    assert len(ids['labels']) == source['expected_counts'][role]
    return np.concatenate(values), ids, certificates


def read_split(parent, source4, task, dataset, role, device, batch):
    return read4(source4, role, device, batch) if task == 'Task4c' else read35(parent, task, dataset, role, device, batch)


@torch.no_grad()
def predict(model, x, batch):
    model.eval()
    return torch.cat([model(x[first:first+batch]).softmax(-1)[:, 1] for first in range(0, len(x), batch)]).cpu().numpy()


@torch.no_grad()
def validation_prediction(model, x, y, batch):
    model.eval(); probabilities = []; total = 0.
    for first in range(0, len(x), batch):
        logits = model(x[first:first+batch])
        labels = torch.tensor(y[first:first+batch], device=x.device, dtype=torch.long)
        total += float(F.cross_entropy(logits, labels, reduction='sum'))
        probabilities.append(logits.softmax(-1)[:, 1])
    return torch.cat(probabilities).cpu().numpy(), total/len(x)


def subgroup_metrics(task, ids, probability, threshold):
    if task != 'Task4c':
        return {str(k): metrics(ids['labels'][ids['scale_id'] == k], probability[ids['scale_id'] == k], threshold)
                for k in np.unique(ids['scale_id'])}
    return {flow: metrics(ids['labels'][ids['flow_index'] == i], probability[ids['flow_index'] == i], threshold)
            for i, flow in enumerate(('channel', 'tbl'))}


def preflight(spec, parent, source4, config):
    import unittest
    from tests.test_fmt_single_center_1_1 import SingleCenterTests
    setup('cuda')
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SingleCenterTests))
    assert outcome.wasSuccessful()
    checks = []
    for task, dataset in units(parent)[:-1]:
        split_paths = [set(map(str, source_paths(parent, task, dataset, role)[2])) for role in ('train', 'validation', 'confirmation')]
        assert all(not split_paths[i] & split_paths[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        checks.append(dict(task=task, dataset=dataset, paths=[sorted(p) for p in split_paths]))
    assert sha(Path(source4['source_output'])/'data_audit.json') == source4['source_audit_sha256']
    # Check all immutable source bytes; no test labels or test metrics are used.
    for flow in ('channel', 'tbl'):
        for role, definition in source4['source_files'][flow].items():
            for name, digest in definition['files'].items():
                assert sha(Path(source4['source_output'])/'physical'/flow/role/name) == digest
    subset_checks = [verify_subset_centers(source4, flow, role) for flow in ("channel", "tbl") for role in ("train", "validation", "test")]
    pilots = []
    for task in ('Task3', 'Task5', 'Task4c'):
        if task == 'Task4c':
            folder = Path(source4['source_output'])/'physical'/'channel'/'train'
            rows, anchors = subset_rows(source4, 'channel', 'train')
            with np.load(folder/'metadata.npz') as z:
                present_labels = z['labels'][rows]
                local = np.r_[np.flatnonzero(present_labels == 0)[:16], np.flatnonzero(present_labels == 1)[:16]]
                selected = rows[local]
                labels, counts = z['labels'][selected].astype(np.int64), z['counts'][selected]
                centers = anchors[local]
            x = np.load(folder/'geometry.npy', mmap_mode='r')[selected]
        else:
            row = records(parent, task, 'channel', 'train')[0]
            selected = np.r_[np.flatnonzero(row['labels'] == 0)[:16], np.flatnonzero(row['labels'] == 1)[:16]]
            x, labels = row['raw'][selected], row['labels'][selected].astype(np.int64)
            counts, centers = np.full(32, 7), np.zeros(32, dtype=np.int64)
        assert len(labels) == 32
        cpu = encode(x, counts, centers).numpy()
        gpu = encode(torch.tensor(x, device='cuda'), counts, centers).cpu().numpy()
        np.testing.assert_allclose(cpu, gpu, atol=3e-5, rtol=3e-5)
        split = np.concatenate([encode(torch.tensor(x[i:i+8], device='cuda'), counts[i:i+8], centers[i:i+8]).cpu().numpy() for i in range(0, 32, 8)])
        np.testing.assert_allclose(gpu, split, atol=3e-5, rtol=3e-5)
        norm = fit_normalizer(gpu, None, 'zscore')
        features = torch.tensor(apply_normalizer(gpu, norm), device='cuda')
        y = torch.tensor(labels, device='cuda')
        torch.manual_seed(40)
        model = SingleCenterMLP().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        losses = []
        for step in range(180):
            model.train(); opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features), y)
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        score = metrics(labels, predict(model, features, 32), .5)
        assert score['f1'] >= .95 and losses[-1] < losses[0]*.5, (task, score, losses[0], losses[-1])
        pilots.append(dict(task=task, samples=32, f1=score['f1'], initial_loss=losses[0], final_loss=losses[-1],
                           cpu_gpu_max_error=float(np.abs(cpu-gpu).max()), batch_max_error=float(np.abs(gpu-split).max())))
        # Full training batch and backward pass, without additional fitting.
        loss = F.cross_entropy(model(features.repeat(16, 1)), y.repeat(16)); loss.backward()
    model = SingleCenterMLP()
    write(Path(spec['output'])/'preflight.json', dict(status='PASS', identity=identity(config), gpu=torch.cuda.get_device_name(),
          tests=outcome.testsRun, pilots=pilots, splits=checks, task4c_subsets=subset_checks, parameters=sum(p.numel() for p in model.parameters()),
          network=str(model), modules=[type(m).__name__ for m in model.modules()], formal_fits=21,
          test_labels_read=False, test_metrics_computed=False, timestamp=datetime.now(timezone.utc).isoformat()))
    print('PREFLIGHT PASS', flush=True)


def train(spec, parent, source4, config, index):
    setup('cuda')
    root = Path(spec['output'])
    pre = json.loads((root/'preflight.json').read_text())
    assert pre['status'] == 'PASS' and pre['identity'] == identity(config)
    task, dataset = units(parent)[index]
    folder = root/'runs'/task/dataset
    folder.mkdir(parents=True, exist_ok=False)
    settings = spec['training']['task4c' if task == 'Task4c' else 'task35']
    started_prepare = time.perf_counter()
    train_x, train_ids, train_source = read_split(parent, source4, task, dataset, 'train', 'cuda', spec['encode_batch'])
    val_x, val_ids, val_source = read_split(parent, source4, task, dataset, 'validation', 'cuda', spec['encode_batch'])
    norm = fit_normalizer(train_x, None, 'zscore')
    write(folder/'normalization.json', {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in norm.items()})
    write(folder/'source_train_validation.json', dict(train=train_source, validation=val_source))
    np.savez_compressed(folder/'training_ids.npz', **train_ids)
    x = torch.tensor(apply_normalizer(train_x, norm), device='cuda')
    vx = torch.tensor(apply_normalizer(val_x, norm), device='cuda')
    del train_x, val_x
    y, vy = train_ids['labels'], val_ids['labels']
    target = torch.tensor(y, device='cuda')
    assert np.isin(y, [0, 1]).all() and 0 < y.sum() < len(y)
    preparation_seconds = time.perf_counter()-started_prepare
    torch.manual_seed(spec['seed']); rng = np.random.default_rng(spec['seed'])
    model = assert_no_fmt_convolution(SingleCenterMLP(dropout=spec['dropout'])).cuda()
    assert sum(p.numel() for p in model.parameters()) == spec['parameters']
    write(folder/'architecture.json', dict(network=str(model), modules=[type(m).__name__ for m in model.modules()],
          parameters=spec['parameters'], feature_shape=list(x.shape), centers_per_sample=1, convolution_count=0, raw_branch=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=15, threshold=1e-4, min_lr=1e-6) if task == 'Task4c' else None
    positive_weight = torch.tensor((len(y)-float(y.sum()))/float(y.sum()), device='cuda')
    best, best_epoch, state = (-1., -1.), 0, None
    history = []; batch = settings['batch_size']; started = time.perf_counter()
    for epoch in range(1, settings['epochs']+1):
        order = rng.permutation(len(y)); model.train(); total = 0.
        for first in range(0, len(order), batch):
            ids = torch.tensor(order[first:first+batch], device='cuda')
            optimizer.zero_grad(set_to_none=True); logits = model(x[ids])
            loss = F.cross_entropy(logits, target[ids]) if task == 'Task4c' else F.binary_cross_entropy_with_logits(logits[:, 1]-logits[:, 0], target[ids].float(), pos_weight=positive_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if task == 'Task4c':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())*len(ids)
        p, val_loss = validation_prediction(model, vx, vy, batch)
        score = metrics(vy, p, .5)
        rank = (score['f1'], score['average_precision']) if task == 'Task4c' else (score['average_precision'], 0.)
        improved = rank > best if task == 'Task4c' else rank[0] > best[0]+settings['min_delta']
        if improved:
            best, best_epoch = rank, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = dict(epoch=epoch, training_loss=total/len(y), validation_f1=score['f1'], validation_average_precision=score['average_precision'],
                   validation_loss=val_loss, selected=improved, learning_rate=optimizer.param_groups[0]['lr'],
                   permutation_sha256=array_sha(order.astype('<i8')), seconds=time.perf_counter()-started)
        history.append(row)
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if scheduler:
            scheduler.step(val_loss)
        if epoch == 1 or epoch % 10 == 0:
            print(task, dataset, row, flush=True)
        if epoch-best_epoch >= settings['patience']:
            break
    assert state is not None
    model.load_state_dict(state); p = predict(model, vx, batch)
    if task == 'Task4c':
        threshold = .5
    else:
        precision, recall, thresholds = precision_recall_curve(vy, p)
        f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
        threshold = float(thresholds[int(np.nanargmax(f1))])
    training_seconds = time.perf_counter()-started
    lock = dict(identity=identity(config), task=task, dataset=dataset, candidate=spec['candidate'], seed=spec['seed'],
                selected_epoch=best_epoch, threshold=threshold, test_loaded=False,
                normalization_sha256=sha(folder/'normalization.json'), timestamp=datetime.now(timezone.utc).isoformat())
    write(folder/'selection.lock.json', lock)
    np.savez_compressed(folder/'validation_predictions.npz', **val_ids, probability=p, threshold=threshold)
    result = dict(complete=False, identity=identity(config), task=task, dataset=dataset, seed=spec['seed'], candidate=spec['candidate'],
                  parameters=spec['parameters'], selected_epoch=best_epoch, epochs=len(history), threshold=threshold,
                  training_seconds=training_seconds, preparation_seconds=preparation_seconds,
                  validation=metrics(vy, p, threshold), validation_by_group=subgroup_metrics(task, val_ids, p, threshold),
                  predictions={'validation': sha(folder/'validation_predictions.npz')}, gpu=torch.cuda.get_device_name(),
                  single_seed=True, historical_scores_reproduced=False, no_best_configuration_claim=True)
    del x, vx, target, state; torch.cuda.empty_cache()
    # No evaluation labels/features are opened until the selection is recorded.
    tx, test_ids, test_source = read_split(parent, source4, task, dataset, 'test', 'cuda', spec['encode_batch'])
    probability = predict(model, torch.tensor(apply_normalizer(tx, norm), device='cuda'), batch)
    np.savez_compressed(folder/'test_predictions.npz', **test_ids, probability=probability, threshold=threshold)
    write(folder/'source_test.json', test_source)
    result.update(test=metrics(test_ids['labels'], probability, threshold), test_by_group=subgroup_metrics(task, test_ids, probability, threshold), complete=True)
    if task == 'Task4c':
        for name, mask in (('old_test', test_ids['row_in_split'] < 5000), ('added_head', test_ids['row_in_split'] >= 5000)):
            result[name] = metrics(test_ids['labels'][mask], probability[mask], threshold)
    result['predictions']['test'] = sha(folder/'test_predictions.npz')
    write(folder/'result.json', result)
    print(json.dumps(result), flush=True)


def audit(spec, parent, source4, config):
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
    root = Path(spec['output']); results = []
    for task, dataset in units(parent):
        folder = root/'runs'/task/dataset
        r = json.loads((folder/'result.json').read_text())
        lock = json.loads((folder/'selection.lock.json').read_text())
        assert r['complete'] and r['identity'] == identity(config) == lock['identity']
        assert r['seed'] == spec['seed'] and r['parameters'] == spec['parameters'] and not lock['test_loaded']
        history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        best, best_epoch = (-1., -1.), 0
        for h in history:
            rank = (h['validation_f1'], h['validation_average_precision']) if task == 'Task4c' else (h['validation_average_precision'], 0.)
            better = rank > best if task == 'Task4c' else rank[0] > best[0]+spec['training']['task35']['min_delta']
            if better:
                best, best_epoch = rank, h['epoch']
        assert best_epoch == r['selected_epoch'] == lock['selected_epoch']
        for role in ('validation', 'test'):
            path = folder/f'{role}_predictions.npz'; assert sha(path) == r['predictions'][role]
            with np.load(path) as z:
                y, p, threshold = z['labels'], z['probability'], float(z['threshold'])
                assert threshold == lock['threshold'] == r['threshold']
                assert np.isfinite(p).all() and np.all((0 <= p) & (p <= 1))
                if task == 'Task4c':
                    assert len(y) == source4['expected_counts'][role]
                    for fi, flow in enumerate(('channel', 'tbl')):
                        sel = z['flow_index'] == fi; ids = z['row_in_split'][sel]
                        original = Path(source4['source_output'])/'physical'/flow/role
                        with np.load(original/'metadata.npz') as m:
                            expected_rows, expected_centers = subset_rows(source4, flow, role)
                            assert np.array_equal(ids, expected_rows)
                            assert np.array_equal(y[sel], m['labels'][ids])
                            assert np.array_equal(z['instance'][sel], m['instance'][ids])
                            assert np.array_equal(z['center_id'][sel], expected_centers)
                else:
                    rows = records(parent, task, dataset, 'confirmation' if role == 'test' else role)
                    assert np.array_equal(y, np.concatenate([row['labels'] for row in rows]))
                    assert np.array_equal(z['scale_id'], np.concatenate([row['scale_id'] for row in rows]))
                    expected_ids = np.concatenate([np.column_stack((np.full(len(row['raw']), row['ordinal']), np.arange(len(row['raw'])))) for row in rows])
                    assert np.array_equal(z['row_id'], expected_ids) and np.all(z['center_id'] == 0)
                pred = p >= threshold
                independent = dict(f1=f1_score(y, pred, zero_division=0), precision=precision_score(y, pred, zero_division=0),
                                   recall=recall_score(y, pred, zero_division=0), accuracy=accuracy_score(y, pred), average_precision=average_precision_score(y, p))
                for key, value in independent.items():
                    assert abs(value-r[role][key]) < 1e-12, (task, dataset, role, key)
                assert confusion_matrix(y, pred, labels=[0, 1]).tolist() == r[role]['confusion_matrix']
        results.append(r)
    report = dict(status='PASS', identity=identity(config), fits=len(results), seed=spec['seed'], results=results,
                  checkpoint_files=[], no_best_configuration_claim=True, single_seed=True)
    report['task_means'] = {task: {role: float(np.mean([r[role]['f1'] for r in results if r['task'] == task])) for role in ('validation', 'test')}
                            for task in ('Task3', 'Task5', 'Task4c')}
    report['equal_task_mean'] = {role: float(np.mean([v[role] for v in report['task_means'].values()])) for role in ('validation', 'test')}
    forbidden = [str(p) for p in root.rglob('*') if p.suffix in ('.pt', '.pth', '.ckpt')]
    assert not forbidden
    write(root/'independent_audit.json', report)
    lines = ['# Single-center P35 verification, seed 40', '', '| Task | Validation F1 | Test F1 |', '|---|---:|---:|']
    for task, scores in report['task_means'].items():
        lines.append(f"| {task} | {scores['validation']:.6f} | {scores['test']:.6f} |")
    lines += ['', 'One candidate and one seed; this does not establish the best compliant configuration.', 'Historical reused benchmark; Task4-c shares GT instances across splits.']
    (root/'performance.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['task_means']), flush=True)


def runtime(spec, args):
    row = dict(phase=args.runtime_phase, state=args.state, exit_code=args.exit_code, job=os.environ.get('SLURM_JOB_ID'),
               array_id=os.environ.get('SLURM_ARRAY_TASK_ID'), node=socket.gethostname(), utc=datetime.now(timezone.utc).isoformat())
    folder = Path(spec['output'])/'runtime'; folder.mkdir(parents=True, exist_ok=True)
    with (folder/f"{row['job']}_{row['array_id']}.jsonl").open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'train', 'audit', 'runtime'))
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--runtime-phase'); parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec, parent, source4 = load(args.config)
    if args.phase == 'runtime':
        runtime(spec, args)
    elif args.phase == 'preflight':
        preflight(spec, parent, source4, args.config)
    elif args.phase == 'train':
        train(spec, parent, source4, args.config, args.index)
    else:
        audit(spec, parent, source4, args.config)


if __name__ == '__main__':
    main()
