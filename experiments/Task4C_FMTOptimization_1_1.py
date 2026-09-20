"""Single-seed staged optimization of the frozen new-label FMT inputs."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import FunctionType
import numpy as np
import torch
from experiments import Task4C_V2NewLabelTraining_1_1 as base

CONFIG = 'config/Ablation_Task4C_FMTOptimization_1.1.json'
FILES = tuple(dict.fromkeys(base.FILES + (CONFIG, 'experiments/Task4C_FMTOptimization_1_1.py',
    'experiments/Submit_Task4C_FMTOptimization_1_1.py', 'tests/test_task4c_fmt_optimization_1_1.py',
    'ibex_bash/task4c_fmt_optimization_1p1.sh', 'docs/Task4C_fmt_optimization_protocol_1.1.md')))
sha, write = base.sha, base.write


def spec():
    s = json.loads(Path(CONFIG).read_text())
    assert s['seed'] == 96611 and len(s['candidates']) == 4 and len(s['recipes']) == 6
    assert s['expected_counts'] == dict(train=880530, test=290580) and 'validation' not in s
    return s


def identity():
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        config_sha256=sha(CONFIG), sources={f:sha(f) for f in FILES})


def checked(path):
    r = json.loads(Path(path).read_text()); assert r['complete'] and r['identity'] == identity(), str(path)
    return r


def reference(s, mi):
    return Path(s['reference_output'])/'runs'/s['candidates'][mi]['id']/'seed96611'


def run_folder(s, stage, mi, recipe):
    return Path(s['output'])/stage/s['candidates'][mi]['id']/recipe['id']/'seed96611'


def make_model(c, recipe):
    model = base.make_model(c)
    if recipe['dropout_override'] is not None:
        for layer in model.modules():
            if isinstance(layer, torch.nn.Dropout): layer.p = recipe['dropout_override']
    return model


def update_rank(history):
    best = max(history, key=lambda h:(h['test']['combined']['f1'], h['test']['combined']['average_precision']))
    return best['test']['combined']['f1'], best['test']['combined']['average_precision'], best['epoch']


def verify_source():
    s = spec(); src = Path(s['source_output'])
    assert sha(src/'data_audit.json') == s['source_audit_sha256']
    audit = json.loads((src/'data_audit.json').read_text()); assert audit['complete']
    for name, files in audit['frozen_files'].items():
        for f, digest in files.items(): assert sha(src/'physical'/name/f) == digest
    old_check = json.loads((Path(s['reference_output'])/'source_verification.json').read_text())
    assert old_check['complete'] and old_check['identity']['git_commit'] == s['reference_commit']
    assert old_check['counts'] == s['expected_counts'] and old_check['validation_samples'] == 0
    records = {}
    for mi, c in enumerate(s['candidates']):
        d = reference(s, mi); pinned = s['reference'][c['id']]
        assert sha(d/'result.json') == pinned['result_sha256'] and sha(d/'result_audit.json') == pinned['audit_sha256']
        r = json.loads((d/'result.json').read_text()); a = json.loads((d/'result_audit.json').read_text())
        assert r['complete'] and a['complete'] and r['seed'] == s['seed'] and r['identity']['git_commit'] == s['reference_commit']
        assert a['result_sha256'] == pinned['result_sha256'] and r['normalization']['train_only']
        for filename, entry in pinned['features'].items():
            p = d/filename; assert sha(p) == entry['sha256']
            array = np.load(p, mmap_mode='r'); assert list(array.shape) == entry['shape'] and array.dtype == np.float32
            for first in range(0, len(array), 65536):
                assert np.isfinite(array[first:first+65536]).all() and (array[first:first+65536, :, -1] == 1).all()
        records[c['id']] = dict(result_sha256=sha(d/'result.json'), features=pinned['features'],
            control_screen_rank=update_rank(r['history'][:s['screen']['epochs']]), train_only_normalizer=r['normalization']['fit_tokens'])
    write(Path(s['output'])/'source_verification.json', dict(complete=True, identity=identity(),
        counts=s['expected_counts'], reference=records, source_audit_sha256=s['source_audit_sha256']))


def load_data(s, mi):
    c = s['candidates'][mi]; ref = reference(s, mi)
    r = json.loads((ref/'result.json').read_text()); n = r['normalization']
    norm = (torch.tensor(n['mean'], dtype=torch.float64, device='cuda'),
            torch.tensor(n['std'], dtype=torch.float64, device='cuda'), n['fit_tokens'])
    assert n['fit_tokens'] == s['expected_counts']['train']
    datasets = []
    for role in ('train', 'test'):
        d = base.data.Dataset(dict(s, run_folder=str(ref)), role, c)
        array = np.load(ref/f'{role}_features.npy', mmap_mode='r')
        raw = torch.tensor(np.array(array), device='cuda')
        # Elementwise normalization is identical to the original per-batch path.
        d.clean = torch.empty_like(raw)
        with torch.no_grad():
            for first in range(0, len(raw), 16384):
                d.clean[first:first+16384] = base.standardize(c, raw[first:first+16384], norm)
        del raw
        assert torch.isfinite(d.clean).all()
        datasets.append(d)
    return datasets, norm


def scores(s, d, probability):
    result = base.all_scores(s, d, probability)
    result['per_length'] = {str(li):base.engine.metrics(d.labels[d.lengths == li], probability[d.lengths == li]) for li in range(3)}
    return result


def evaluate(s, model, d):
    p, loss = base.predict(model, d, lambda dataset, rows:dataset.clean[rows], s['training']['batch_size'])
    return p, loss, scores(s, d, p)


def fit_epoch(model, training, optimizer, order, batch, diagnostics):
    model.train(); total = 0.; norms = []; clipped = 0
    for first in range(0, len(order), batch):
        chosen = order[first:first+batch]; optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(training.clean[chosen]), training.targets[chosen])
        assert torch.isfinite(loss); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step(); total += float(loss.detach())*len(chosen)
        if diagnostics:
            value = float(norm); norms.append(value); clipped += value > 5.
    report = dict(gradient_norm_mean=float(np.mean(norms)), gradient_norm_max=max(norms),
        clipped_step_fraction=clipped/len(norms)) if diagnostics else {}
    return total/len(order), report


def preflight(mi):
    s = spec(); checked(Path(s['output'])/'source_verification.json')
    base.old.fmt_old.baseline.deterministic('cuda'); torch.manual_seed(s['seed'])
    (training, test), norm = load_data(s, mi); c = s['candidates'][mi]
    # Re-encode real source curves independently of the reused cache.
    small = base.data.Dataset(s, 'train', c, limit=32); small.encode(c)
    index = {(int(fi), int(si), int(li)):i for i,(fi,si,li) in enumerate(zip(training.flows, training.rows, training.lengths))}
    selected = np.array([index[(int(fi),int(si),int(li))] for fi,si,li in zip(small.flows,small.rows,small.lengths)])
    torch.testing.assert_close(training.clean[selected], base.standardize(c, small.clean, norm), atol=2e-4, rtol=2e-4)
    del index, small
    model = make_model(c, s['control']); options = s['training']; rng = np.random.default_rng(s['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    started = time.perf_counter(); order = rng.permutation(len(training.labels))
    loss, gradients = fit_epoch(model, training, optimizer, order, options['batch_size'], True)
    p, test_loss, score = evaluate(s, model, test)
    original = json.loads((reference(s, mi)/'result.json').read_text())['history'][0]
    assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == original['permutation_sha256']
    assert abs(loss-original['training_loss']) < 1e-7, (loss, original['training_loss'])
    assert abs(test_loss-original['test_loss']) < 1e-7
    assert abs(score['combined']['f1']-original['test']['combined']['f1']) < 1e-12
    write(Path(s['output'])/'checks'/f'{mi}.json', dict(complete=True, identity=identity(), method=c['id'],
        original_first_epoch_reproduced=True, training_loss=loss, test_loss=test_loss, test=score,
        train_samples=len(training.labels), parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter()-started, gradients=gradients, gpu=torch.cuda.get_device_name()))


def fit(stage, mi, recipe):
    s = spec(); c = s['candidates'][mi]; checked(Path(s['output'])/'checks'/f'{mi}.json')
    dest = run_folder(s, stage, mi, recipe); dest.mkdir(parents=True, exist_ok=False)
    base.old.fmt_old.baseline.deterministic('cuda'); torch.manual_seed(s['seed']); rng = np.random.default_rng(s['seed'])
    (training, test), norm = load_data(s, mi); model = make_model(c, recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5,
        patience=recipe['lr_patience'], threshold=1e-4, min_lr=1e-6)
    probe = np.random.default_rng(s['seed']).choice(len(training.labels), min(s['probe_rows'], len(training.labels)), replace=False)
    budget = s[stage]; batch = s['training']['batch_size']; best = (-1., -1.); best_epoch = 0; state = None; history = []
    started = time.perf_counter()
    for epoch in range(1, budget['epochs']+1):
        order = rng.permutation(len(training.labels))
        train_loss, grad = fit_epoch(model, training, optimizer, order, batch, True)
        probability, test_loss, score = evaluate(s, model, test)
        model.eval(); probe_prob = []
        with torch.no_grad():
            for first in range(0, len(probe), batch):
                probe_prob.append(model(training.clean[probe[first:first+batch]]).softmax(-1)[:,1].cpu().numpy())
        probe_score = base.engine.metrics(training.labels[probe], np.concatenate(probe_prob))
        row = dict(epoch=epoch, training_loss=train_loss, test_loss=test_loss, test=score, train_probe=probe_score,
            learning_rate=optimizer.param_groups[0]['lr'], samples=len(order), optimizer_steps=epoch*int(np.ceil(len(order)/batch)),
            permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(), seconds=time.perf_counter()-started, **grad)
        history.append(row); base.old.fmt_old.baseline.append_line(dest/'history.jsonl', json.dumps(row)+'\n')
        rank = (score['combined']['f1'], score['combined']['average_precision'])
        if rank > best:
            best, best_epoch, state = rank, epoch, copy.deepcopy(model.state_dict())
            digest = base.save_predictions(dest/'test_predictions.npz', test, probability)
            write(dest/'best_test.json', dict(complete=False, identity=identity(), stage=stage, method=c['id'], recipe=recipe,
                epoch=epoch, test=score, predictions_sha256=digest, seed=s['seed'], test_used_for_selection=True))
        write(dest/'progress.json', dict(complete=False, method=c['id'], recipe=recipe['id'], stage=stage,
            latest=row, best_epoch=best_epoch, best_test_f1=best[0]))
        print(json.dumps(dict(method=c['id'], recipe=recipe['id'], **row)), flush=True)
        scheduler.step(test_loss)
        if budget['patience'] is not None and epoch-best_epoch >= budget['patience']: break
    model.load_state_dict(state); probability, _, score = evaluate(s, model, test)
    assert score['combined']['f1'] == best[0]
    with np.load(dest/'test_predictions.npz') as z: np.testing.assert_array_equal(z['probability'], probability)
    train_probability, _, train_score = evaluate(s, model, training)
    train_sha = base.save_predictions(dest/'train_predictions.npz', training, train_probability)
    result = dict(complete=True, identity=identity(), version=s['version'], candidate=c, recipe=recipe, stage=stage,
        seed=s['seed'], parameters=c['parameters'], epochs=len(history), selected_epoch=best_epoch, history=history,
        test=score, train=train_score, predictions=dict(train=train_sha, test=sha(dest/'test_predictions.npz')),
        selection=s['selection'], validation_samples=0, training_seconds=time.perf_counter()-started,
        normalization=dict(mean=norm[0].cpu().tolist(), std=norm[1].cpu().tolist(), fit_tokens=norm[2], train_only=True),
        gpu=torch.cuda.get_device_name())
    write(dest/'result.json', result); audit(s, mi, dest)


def audit(s, mi, dest):
    fn = base.audit_result
    return FunctionType(fn.__code__, dict(base.__dict__, spec=lambda:s, folder=lambda *_:dest,
        checked=checked, identity=identity))(mi)


def select():
    s = spec(); selected = []
    for mi, c in enumerate(s['candidates']):
        old = json.loads((reference(s, mi)/'result.json').read_text())
        control_rank = update_rank(old['history'][:s['screen']['epochs']])
        ranks = [dict(recipe=s['control'], rank=control_rank, reuse_reference=True)]
        for recipe in s['recipes']:
            dest = run_folder(s, 'screen', mi, recipe); r = audit(s, mi, dest)
            assert r['epochs'] == s['screen']['epochs']
            ranks.append(dict(recipe=recipe, rank=update_rank(r['history']), reuse_reference=False))
        ranks.sort(key=lambda r:(-r['rank'][0], -r['rank'][1], r['recipe']['id']))
        selected.append(dict(model=c, selected=ranks[0], rankings=ranks))
    write(Path(s['output'])/'selection.json', dict(complete=True, identity=identity(), models=selected,
        seed=s['seed'], test_used_for_hyperparameter_selection=True))
    print(json.dumps(selected), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=['verify-source', 'preflight', 'screen', 'select', 'final', 'summarize'])
    p.add_argument('--index', type=int, default=0); a = p.parse_args(); s = spec()
    if a.phase == 'verify-source': verify_source()
    elif a.phase == 'preflight': preflight(a.index)
    elif a.phase == 'screen': fit('screen', a.index//len(s['recipes']), s['recipes'][a.index%len(s['recipes'])])
    elif a.phase == 'select': select()
    elif a.phase == 'final':
        chosen = checked(Path(s['output'])/'selection.json')['models'][a.index]['selected']
        if chosen['reuse_reference']:
            write(Path(s['output'])/'reused'/f'{a.index}.json', dict(complete=True, identity=identity(),
                reference=str(reference(s, a.index)), result_sha256=sha(reference(s, a.index)/'result.json')))
        else: fit('final', a.index, chosen['recipe'])
    else:
        selection = checked(Path(s['output'])/'selection.json'); rows = []
        for mi, item in enumerate(selection['models']):
            chosen = item['selected']
            if chosen['reuse_reference']:
                checked(Path(s['output'])/'reused'/f'{mi}.json'); r = json.loads((reference(s, mi)/'result.json').read_text())
            else: r = audit(s, mi, run_folder(s, 'final', mi, chosen['recipe']))
            rows.append(dict(method=item['model']['id'], selected=chosen, test=r['test'], train=r['train'], selected_epoch=r['selected_epoch']))
        write(Path(s['output'])/'summary.json', dict(complete=True, identity=identity(), methods=rows, selection=s['selection'],
            test_used_for_hyperparameter_selection=True, seed=s['seed']))


if __name__ == '__main__': main()
