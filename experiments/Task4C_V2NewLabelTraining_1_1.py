"""Six single-seed fits; all four training folds; explicit test-guided stopping."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import FunctionType
import numpy as np
import torch
from FMT_Utils import Task4C_V2NewLabelTraining_1_1 as data
from experiments import Task4C_V3Training_3_1 as old
from experiments import Prepare_Task4C_BallQuery_3_1 as neighbors

CONFIG = 'config/mainExp_Task4C_V2NewLabelTraining_1.1.json'
NEIGHBOR_CONFIG = 'config/Verify_Task4C_V2NewLabelBallQuery_1.1.json'
FILES = tuple(dict.fromkeys(old.FILES + (CONFIG, NEIGHBOR_CONFIG,
    'FMT_Utils/Task4C_V2NewLabelTraining_1_1.py', 'experiments/Task4C_V2NewLabelTraining_1_1.py',
    'experiments/Submit_Task4C_V2NewLabelTraining_1_1.py', 'tests/test_task4c_v2_newlabel_training_1_1.py',
    'ibex_bash/task4c_v2_newlabel_training_1p1.sh', 'docs/Task4C_v2_newlabel_training_protocol_1.1.md')))
engine = old.base_old.frozen.engine
sha, write = old.sha, old.write


def identity():
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        config_sha256=sha(CONFIG), sources={p:sha(p) for p in FILES})


def spec():
    s = json.loads(Path(CONFIG).read_text())
    assert s['final']['seeds'] == [96611] and len(s['candidates']) == 6
    assert 'validation' not in s and s['expected_counts'] == dict(train=880530, test=290580)
    assert s['selection']['role'] == 'test' and s['selection']['test_labels_in_gradients'] is False
    return s


def folder(s, index):
    return Path(s['output'])/'runs'/s['candidates'][index]['id']/'seed96611'


def checked(path):
    r = json.loads(Path(path).read_text())
    assert r['complete'] and r['identity'] == identity(), str(path)
    return r


def build_neighbors(index):
    FunctionType(neighbors.build.__code__, dict(neighbors.__dict__, split_masks=data.split_masks))(NEIGHBOR_CONFIG, index)


def verify_source():
    s = spec(); root = Path(s['source_output']); audit = json.loads((root/'data_audit.json').read_text())
    assert audit['complete'] and audit['dataset_name'] == 'v2_newlabel_rule'
    assert sha(root/'data_audit.json') == s['source_audit_sha256']
    assert audit['labeling']['positive_minimum_hits'] == 17 and audit['validation_samples'] == 0
    counts = dict(train=0, test=0); records = {}
    for fi, f in enumerate(s['flows']):
        name = f['name']; src = root/'physical'/name
        for file, digest in audit['frozen_files'][name].items():
            assert sha(src/file) == digest
        with np.load(src/'metadata.npz') as z:
            masks = data.split_masks(z, fi, s)
            np.testing.assert_array_equal(z['label'], z['short_curve_gt_count'] >= 17)
            assert not np.any(masks['train'] & masks['test'])
            assert np.all(masks['train'] | masks['test'])
            assert not set(z['instance'][masks['train']]).intersection(z['instance'][masks['test']])
            record = {r:dict(samples=int(m.sum()), positive=int(z['label'][m].sum())) for r, m in masks.items()}
            for r, mask in masks.items(): counts[r] += int(mask.sum())*3
        nd = Path(s['neighbor_output'])/name; m = json.loads((nd/'manifest.json').read_text())
        assert m['complete'] and m['source_audit_sha256'] == s['source_audit_sha256']
        assert m['h'] == f['h_value'] and m['radius_h'] == 3 and m['independent_centers'] >= 30
        for file, digest in m['files'].items(): assert sha(nd/file) == digest
        records[name] = dict(splits=record, neighbor_manifest_sha256=sha(nd/'manifest.json'))
    assert counts == s['expected_counts']
    write(Path(s['output'])/'source_verification.json', dict(complete=True, identity=identity(),
        counts=counts, per_flow=records, validation_samples=0, source_audit_sha256=s['source_audit_sha256']))
    print(json.dumps(dict(source='PASS', counts=counts)), flush=True)


def make_model(c):
    model = engine.Conv915(.15) if c['base_method'] == 'conv16' else old.fmt_old.baseline.method.make_model(c['base_method'])
    assert sum(p.numel() for p in model.parameters()) == c['parameters']
    return model.cuda()


def standardize(c, values, norm):
    return old.fmt_old.baseline.environment(c)['standardize'](values, norm)


def preflight(index):
    s = spec(); c = s['candidates'][index]
    checked(Path(s['output'])/'source_verification.json')
    old.fmt_old.baseline.deterministic('cuda'); torch.manual_seed(96611)
    d = data.Dataset(s, 'train', c, limit=32)
    assert np.all(d.extra['fold'] != s['folds']['test_fold'])
    if c['base_method'] == 'conv16':
        g, counts = d.geometry(np.arange(len(d.labels)))
        gt = torch.tensor(g, device='cuda'); ct = torch.tensor(counts, device='cuda')
        x = data.old.bundle_voxels(gt, ct, 16, 4).half().float()
        off, ids, values = engine.sparse_voxels(gt, ct, 16)
        recovered = engine.dense_sparse_batch([dict(offsets=off, indices=ids, values=values)],
            [(0, i) for i in range(len(g))], 16, 'cuda')
        torch.testing.assert_close(x, recovered, atol=0, rtol=0)
        from experiments.Task4C_ConvResolution_4_22 import reference_voxel
        assert np.allclose(x[0].cpu().numpy(), reference_voxel(g[0], int(counts[0]), 16), atol=.0005, rtol=.002)
    else:
        d.encode(c); small = data.Dataset(s, 'train', c, limit=32); small.encode(c, batch=8)
        np.testing.assert_array_equal(d.selected_neighbors, small.selected_neighbors)
        torch.testing.assert_close(d.clean, small.clean, atol=2e-4, rtol=2e-4)
        norm = old.fit_normalizer(d, c); x = standardize(c, d.clean, norm)
    model = make_model(c); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    model.eval()
    with torch.no_grad(): initial = float(torch.nn.functional.cross_entropy(model(x), d.targets))
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for _ in range(30):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), d.targets); assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
    model.eval()
    with torch.no_grad(): final = float(torch.nn.functional.cross_entropy(model(x), d.targets))
    assert final < initial, (c['id'], initial, final)
    take = torch.arange(128, device='cuda') % len(d.labels)
    model.train(); optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x[take]), d.targets[take]).backward(); torch.cuda.synchronize()
    write(Path(s['output'])/'checks'/f'{index}.json', dict(complete=True, identity=identity(), engineering_only=True,
        method=c['id'], parameters=c['parameters'], initial_loss=initial, final_loss=final,
        batch128_backward=True, seconds=time.perf_counter()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name()))


def all_scores(s, d, probability):
    return dict(combined=engine.metrics(d.labels, probability), per_flow={f['name']:
        engine.metrics(d.labels[d.flows == fi], probability[d.flows == fi]) for fi, f in enumerate(s['flows'])})


def save_predictions(path, d, probability):
    temp = path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temp, labels=d.labels, probability=probability, flow_index=d.flows,
        instance=d.owners, sample_id=d.rows, length_id=d.lengths, threshold=.5, **d.extra)
    temp.replace(path)
    return sha(path)


@torch.no_grad()
def predict(model, dataset, get_batch, batch):
    model.eval(); probabilities = []; total = 0.
    for first in range(0, len(dataset.labels), batch):
        rows = np.arange(first, min(first+batch, len(dataset.labels)))
        logits = model(get_batch(dataset, rows))
        total += float(torch.nn.functional.cross_entropy(logits, dataset.targets[rows], reduction='sum'))
        probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities), total/len(dataset.labels)


def train(index):
    s = spec(); c = s['candidates'][index]; dest = folder(s, index)
    checked(Path(s['output'])/'source_verification.json'); checked(Path(s['output'])/'checks'/f'{index}.json')
    dest.mkdir(parents=True, exist_ok=False)
    old.fmt_old.baseline.deterministic('cuda'); torch.manual_seed(96611); rng = np.random.default_rng(96611)
    s['run_folder'] = str(dest); options = dict(s['training'], **s['final']); batch = options['batch_size']
    started_prepare = time.perf_counter()
    training = data.Dataset(s, 'train', c); test = data.Dataset(s, 'test', c)
    assert np.all(training.extra['fold'] != 0) and np.all(test.extra['fold'] == 0)
    norm = None
    if c['base_method'] == 'conv16':
        training.cache_voxels(engine); test.cache_voxels(engine)
        get_batch = lambda d, rows: d.voxel_batch(rows, engine)
    else:
        training.encode(c); test.encode(c); norm = old.fit_normalizer(training, c, batch)
        for d in (training, test):
            d.clean = torch.tensor(np.array(d.clean.array), device='cuda')
        get_batch = lambda d, rows: standardize(c, d.clean[rows], norm)
    preparation_seconds = time.perf_counter()-started_prepare
    model = make_model(c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5,
        patience=options['lr_patience'], threshold=1e-4, min_lr=1e-6)
    best = (-1., -1.); best_epoch = 0; state = None; history = []; best_digest = None
    started = time.perf_counter()
    for epoch in range(1, options['epochs']+1):
        order = rng.permutation(len(training.labels)); model.train(); total = 0.
        for first in range(0, len(order), batch):
            chosen = order[first:first+batch]; optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(get_batch(training, chosen)), training.targets[chosen])
            assert torch.isfinite(loss); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())*len(chosen)
        probability, test_loss = predict(model, test, get_batch, batch)
        scores = all_scores(s, test, probability); score = scores['combined']; rank = (score['f1'], score['average_precision'])
        row = dict(epoch=epoch, training_loss=total/len(order), test_loss=test_loss, test=scores,
            learning_rate=optimizer.param_groups[0]['lr'], samples=len(order),
            permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(), seconds=time.perf_counter()-started)
        history.append(row)
        old.fmt_old.baseline.append_line(dest/'history.jsonl', json.dumps(row)+'\n')
        if rank > best:
            best, best_epoch, state = rank, epoch, copy.deepcopy(model.state_dict())
            best_digest = save_predictions(dest/'test_predictions.npz', test, probability)
            write(dest/'best_test.json', dict(complete=False, identity=identity(), epoch=epoch, method=c['id'], seed=96611,
                test=scores, predictions_sha256=best_digest, test_used_for_early_stopping=True))
        write(dest/'progress.json', dict(complete=False, method=c['id'], seed=96611, latest=row, best_epoch=best_epoch,
            best_test_f1=best[0], preparation_seconds=preparation_seconds, test_used_for_early_stopping=True))
        print(json.dumps(dict(method=c['id'], **row)), flush=True)
        scheduler.step(test_loss)
        if epoch-best_epoch >= options['patience']: break
    model.load_state_dict(state)
    probability, _ = predict(model, test, get_batch, batch)
    assert abs(engine.metrics(test.labels, probability)['f1']-best[0]) < 1e-12
    with np.load(dest/'test_predictions.npz') as z:
        np.testing.assert_array_equal(z['probability'], probability)
    train_probability, _ = predict(model, training, get_batch, batch)
    train_digest = save_predictions(dest/'train_predictions.npz', training, train_probability)
    result = dict(complete=True, identity=identity(), version=s['version'], candidate=c, seed=96611, parameters=c['parameters'],
        epochs=len(history), selected_epoch=best_epoch, history=history, training_seconds=time.perf_counter()-started,
        preparation_seconds=preparation_seconds, test=all_scores(s, test, probability), train=all_scores(s, training, train_probability),
        predictions=dict(train=train_digest, test=best_digest), selection=s['selection'], validation_samples=0,
        normalization=None if norm is None else dict(mean=norm[0].cpu().tolist(), std=norm[1].cpu().tolist(), fit_tokens=norm[2], train_only=True),
        gpu=torch.cuda.get_device_name())
    write(dest/'result.json', result)
    audit_result(index)


def audit_result(index):
    s = spec(); dest = folder(s, index); r = checked(dest/'result.json')
    assert r['seed'] == 96611 and r['validation_samples'] == 0
    assert r['parameters'] == s['candidates'][index]['parameters'] and len(r['history']) == r['epochs']
    assert all(h['samples'] == s['expected_counts']['train'] for h in r['history'])
    best = max(r['history'], key=lambda h:(h['test']['combined']['f1'], h['test']['combined']['average_precision']))
    assert best['epoch'] == r['selected_epoch'] and r['selection'] == s['selection']
    for role in ('train', 'test'):
        path = dest/f'{role}_predictions.npz'; assert sha(path) == r['predictions'][role]
        with np.load(path) as z: p = {k:z[k] for k in z.files}
        assert len(p['labels']) == s['expected_counts'][role] and float(p['threshold']) == .5
        assert np.isfinite(p['probability']).all() and ((p['probability'] >= 0) & (p['probability'] <= 1)).all()
        for fi, f in enumerate(s['flows']):
            with np.load(Path(s['source_output'])/'physical'/f['name']/'metadata.npz') as z:
                ids = np.flatnonzero(data.split_masks(z, fi, s)[role]); take = p['flow_index'] == fi
                np.testing.assert_array_equal(p['sample_id'][take], np.repeat(ids, 3))
                np.testing.assert_array_equal(p['length_id'][take], np.tile(np.arange(3), len(ids)))
                for key, src in [('labels', 'label'), ('instance', 'instance'), ('fold', 'fold')]:
                    np.testing.assert_array_equal(p[key][take], np.repeat(z[src][ids], 3))
                assert engine.metrics(p['labels'][take], p['probability'][take]) == r[role]['per_flow'][f['name']]
        assert engine.metrics(p['labels'], p['probability']) == r[role]['combined']
        from sklearn.metrics import f1_score
        assert abs(f1_score(p['labels'], p['probability'] >= .5)-r[role]['combined']['f1']) < 1e-12
    write(dest/'result_audit.json', dict(complete=True, identity=identity(), method=r['candidate']['id'],
        result_sha256=sha(dest/'result.json'), test=r['test'], train=r['train'], test_used_for_early_stopping=True))
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['neighbors', 'verify-source', 'preflight', 'train', 'audit-results', 'runtime'])
    parser.add_argument('--index', type=int, default=0); parser.add_argument('--state'); parser.add_argument('--runtime-phase'); parser.add_argument('--exit-code', type=int)
    a = parser.parse_args(); s = spec()
    if a.phase == 'neighbors': build_neighbors(a.index)
    elif a.phase == 'verify-source': verify_source()
    elif a.phase == 'preflight': preflight(a.index)
    elif a.phase == 'train': train(a.index)
    elif a.phase == 'audit-results':
        results = [audit_result(i) for i in range(6)]
        write(Path(s['output'])/'summary.json', dict(complete=True, identity=identity(), methods=[dict(method=r['candidate']['id'],
            seed=96611, selected_epoch=r['selected_epoch'], test=r['test'], train=r['train']) for r in results], selection=s['selection']))
    else:
        Path(s['output']).mkdir(parents=True, exist_ok=True)
        old.fmt_old.baseline.append_line(Path(s['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(),
            phase=a.runtime_phase, state=a.state, exit_code=a.exit_code, index=a.index, job=os.environ.get('SLURM_JOB_ID'),
            host=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


if __name__ == '__main__': main()
