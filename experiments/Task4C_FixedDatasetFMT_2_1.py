"""Fixed-center FMT arms (p35/h0, c156; FPS6 / FPS16 clusters) on the Task4-c fixed dataset v2.

mainExp_Task4C_FixedDatasetFMT_2.1.  Phases: verify-source (frozen hashes + FPS neighbour
tables) -> gpu-check -> train (arm x seed) -> audit-results; ``run-local`` chains them on this
host.  The frozen optimizer / early-stopping / scoring loop of Task4C_FPSAugmentSearch_1_1 is
reused unchanged; only the data interface (Task4C_FixedDatasetFMT_2_1.Dataset) is new.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace
import numpy as np
import torch
from FMT_Utils import Task4C_FPS16_1_1 as method
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as v2
from FMT_Utils import Task4C_FixedDataset_2_1 as data
from experiments import Task4C_FixedDatasetFMT_1_1 as base
from experiments import Task4C_OriginalCenter_1_1 as previous

frozen, p35 = previous.frozen, previous.p35
sha, write = frozen.sha, frozen.write
CONFIG = 'config/mainExp_Task4C_FixedDatasetFMT_2.1.json'
FILES = base.FILES+('experiments/Task4C_FixedDatasetFMT_2_1.py', 'FMT_Utils/Task4C_FixedDatasetFMT_2_1.py', 'FMT_Utils/Task4C_FixedDataset_2_1.py',
                    'config/mainExp_Task4C_FixedDataset_2.1.json', 'docs/Task4C_fixed_dataset_rules_v2.md')


def append_line(path, text):
    """frozen.append_locked needs fcntl (Linux); on this Windows host each file has a single writer."""
    try: frozen.append_locked(path, text)
    except ModuleNotFoundError:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf8') as stream: stream.write(text)


def identity(config):
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), config_sha256=sha(config),
                sources={p: sha(p) for p in FILES}, host=socket.gethostname())


def verify_source(spec, config):
    """Read-only data: every frozen file must match data_audit.json; then build the FPS tables and count rows."""
    source = Path(spec['source_output']); audit = json.loads((source/'data_audit.json').read_text())
    assert audit['complete'] and sha(source/'data_audit.json') == spec['source_audit_sha256']
    checked = []; counts = {r: 0 for r in v2.ROLES}; samples = {}
    for fi, flow in enumerate(spec['flows']):
        folder = source/'physical'/flow['name']
        for filename, digest in audit['frozen_files'][flow['name']].items(): assert sha(folder/filename) == digest, (flow['name'], filename); checked.append(str(folder/filename))
        with np.load(folder/'metadata.npz') as z: meta = {k: z[k] for k in ('fold', 'label', 'requested_half_length')}
        assert np.allclose(meta['requested_half_length'], flow['half_lengths'])
        masks = v2.split_masks(meta, fi, spec); K = len(flow['half_lengths'])
        samples[flow['name']] = {r: dict(samples=int(m.sum()), rows=int(m.sum())*K, hairpin=int(meta['label'][m].sum())) for r, m in masks.items()}
        for r, m in masks.items(): counts[r] += int(m.sum())*K
        assert sum(int(m.sum()) for m in masks.values()) == len(meta['fold']) and not np.any(masks['train'] & masks['validation'])
    assert counts == spec['expected_counts'], counts
    tables = v2.build_neighbor_tables(spec, sha)
    write(Path(spec['output'])/'source_verification.json', dict(complete=True, identity=identity(config), counts=counts, per_flow=samples, files=checked,
                                                               neighbor_tables=tables, read_only_source=True))


def arm_spec(spec, index):
    seeds = spec['final']['seeds']; arm = copy.deepcopy(spec); arm['_root_output'] = spec['output']
    arm['candidate'] = spec['candidates'][index//len(seeds)]; arm['_active_seed'] = seeds[index % len(seeds)]
    arm['final'] = dict(spec['final'], seeds=[arm['_active_seed']]); arm['output'] = str(Path(spec['output'])/'arms'/arm['candidate']['id']/f"seed{arm['_active_seed']}")
    return arm


def save_predictions(folder, role, dataset, probability):
    file = folder/f'{role}_predictions.npz'
    np.savez_compressed(file, labels=dataset.labels, probability=probability, instance=dataset.owners, flow_index=dataset.flows, sample_id=dataset.rows,
                        length_id=dataset.lengths, threshold=.5, **dataset.extra)
    return sha(file)


def deterministic(device):
    """Frozen determinism flags without the V100-only guard (the cluster is chosen in float64 outside the encoder)."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8'); torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    assert device == 'cuda' and torch.cuda.is_available()


def environment(candidate):
    standardize = p35.standardize if candidate['base_method'] == 'p35_h0' else frozen.standardize
    predict = FunctionType(frozen.predict.__wrapped__.__code__, dict(frozen.__dict__, standardize=standardize), argdefs=frozen.predict.__wrapped__.__defaults__)
    constructor = lambda pool, architecture, profile: method.make_model(candidate['base_method'])
    engine = SimpleNamespace(**dict(frozen.engine.__dict__, deterministic=deterministic))
    return dict(frozen.__dict__, Dataset=v2.Dataset, identity=identity, fit_normalizer=previous.fit_normalizer, engine=engine, append_locked=append_line,
                method=SimpleNamespace(**dict(frozen.method.__dict__, FourierClassifier=constructor)),
                standardize=standardize, predict=torch.no_grad()(predict), save_predictions=save_predictions)


def gpu_check(spec, config):
    deterministic('cuda'); checks = {}
    for index, c in enumerate(spec['candidates']):
        torch.manual_seed(97811); arm = arm_spec(spec, index*len(spec['final']['seeds'])); d = v2.Dataset(arm, 'train', c, limit=32); d.encode(c)
        small = v2.Dataset(arm, 'train', c, limit=32); small.encode(c, batch=8)
        assert np.array_equal(d.selected_neighbors, small.selected_neighbors); torch.testing.assert_close(d.clean, small.clean, atol=2e-4, rtol=2e-4)
        env = environment(c); norm = previous.fit_normalizer(d, c); xx = env['standardize'](d.clean, norm)
        model = method.make_model(c['base_method']).cuda(); optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001); losses = []
        for step in range(150):
            model.train(); optim.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(xx), d.targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True); optim.step(); losses.append(float(loss.detach()))
        probability, _ = env['predict'](model, d, norm, 128); score = frozen.metrics(d.labels, probability)
        assert score['f1'] >= .95 and losses[-1] < losses[0], (c['id'], score['f1'])
        checks[c['id']] = dict(parameters=sum(p.numel() for p in model.parameters()), shape=list(d.clean.shape), neighbors=c['neighbors'], pilot_f1=score['f1'])
    write(Path(spec['output'])/'gpu_check.json', dict(complete=True, identity=identity(config), gpu=torch.cuda.get_device_name(), checks=checks, engineering_only=True))


def train(spec, config, index):
    for filename in ('gpu_check.json', 'source_verification.json'):
        result = json.loads((Path(spec['output'])/filename).read_text()); assert result['complete'] and result['identity'] == identity(config), filename
    arm = arm_spec(spec, index); write(Path(arm['output'])/'selection.lock.json', dict(complete=True, selected=arm['candidate'], seed=arm['_active_seed'], identity=identity(config)))
    env = environment(arm['candidate']); FunctionType(frozen.train.__code__, env, argdefs=frozen.train.__defaults__)(arm, config, 0, 'final')


def audit_results(spec, config):
    root = Path(spec['output']); records = []; paired = {}
    from sklearn.metrics import f1_score, precision_score, recall_score
    for index in range(len(spec['candidates'])*len(spec['final']['seeds'])):
        arm = arm_spec(spec, index); c = arm['candidate']; seed = arm['_active_seed']; folder = Path(arm['output'])/'final'/c['id']/f'seed{seed}'
        r = json.loads((folder/'result.json').read_text()); assert r['complete'] and r['identity'] == identity(config) and r['test_loaded'] and r['seed'] == seed
        assert r['candidate'] == c and r['parameters'] == c['parameters']; scores = {}
        history = r['history']; assert all(h['samples'] == spec['expected_counts']['train'] for h in history)
        best = max(history, key=lambda h: (h['validation_f1'], h['validation_average_precision'])); assert best['epoch'] == r['selected_epoch']
        lock = json.loads((folder/'selection.lock.json').read_text()); assert lock['test_loaded'] is False and lock['selected_epoch'] == r['selected_epoch']
        for role in ('validation', 'test'):
            file = folder/f'{role}_predictions.npz'; assert sha(file) == r['predictions'][role]
            with np.load(file) as z: pred = {k: z[k] for k in z.files}
            assert len(pred['labels']) == spec['expected_counts'][role] and float(pred['threshold']) == .5
            key = np.column_stack([pred[k] for k in ('labels', 'instance', 'flow_index', 'sample_id', 'length_id', 'fold', 'assignment_kind')])
            if role in paired: assert np.array_equal(key, paired[role])
            else: paired[role] = key
            for fi, flow in enumerate(spec['flows']):
                mask = pred['flow_index'] == fi; _, _, meta = data.load(Path(spec['source_output'])/'physical'/flow['name'])
                expected = v2.split_masks(meta, fi, spec)[role]; ids = pred['sample_id'][mask]
                assert np.array_equal(np.unique(ids), np.flatnonzero(expected)) and np.array_equal(pred['labels'][mask], meta['label'][ids]) and np.array_equal(pred['instance'][mask], meta['instance'][ids])
                if role == 'test': assert np.all(meta['fold'][ids] == spec['folds']['test_fold'])
            y = pred['labels']; p = pred['probability']; assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            score = dict(f1=float(f1_score(y, p >= .5)), precision=float(precision_score(y, p >= .5)), recall=float(recall_score(y, p >= .5)))
            expected = r['validation'] if role == 'validation' else r['test']['combined']
            for k, v in score.items(): assert abs(v-expected[k]) < 1e-12
            per_instance = {}
            for fi, flow in enumerate(spec['flows']):
                for i in np.unique(pred['instance'][(pred['flow_index'] == fi) & (y == 1)]):
                    take = (pred['flow_index'] == fi) & (y == 1) & (pred['instance'] == i)
                    per_instance[f"{flow['name']}/{int(i)}"] = dict(positive_rows=int(take.sum()), recall=float(np.mean(p[take] >= .5)))
            scores[role] = dict(combined=frozen.metrics(y, p),
                                per_flow={flow['name']: frozen.metrics(y[pred['flow_index'] == fi], p[pred['flow_index'] == fi]) for fi, flow in enumerate(spec['flows'])},
                                per_length={str(k): frozen.metrics(y[pred['length_id'] == k], p[pred['length_id'] == k]) for k in np.unique(pred['length_id'])},
                                per_instance_recall=per_instance)
        records.append(dict(candidate=c, seed=seed, parameters=r['parameters'], epochs=r['epochs'], selected_epoch=r['selected_epoch'], gpu=r['gpu'],
                            training_seconds=r['training_seconds'], preparation_seconds=r['preparation_seconds'], scores=scores, result_sha256=sha(folder/'result.json')))
    summary = {}
    for c in spec['candidates']:
        rows = [r for r in records if r['candidate']['id'] == c['id']]
        def agg(get):
            values = np.array([get(r) for r in rows], float)
            return dict(mean=float(values.mean()), sample_std=float(values.std(ddof=1)) if len(values) > 1 else None, values=values.tolist())
        summary[c['id']] = dict(parameters=c['parameters'], seeds=[r['seed'] for r in rows],
                                test_f1=agg(lambda r: r['scores']['test']['combined']['f1']), test_ap=agg(lambda r: r['scores']['test']['combined']['average_precision']),
                                test_f1_channel=agg(lambda r: r['scores']['test']['per_flow']['channel']['f1']), test_f1_tbl=agg(lambda r: r['scores']['test']['per_flow']['tbl']['f1']),
                                test_f1_per_length={k: agg(lambda r, k=k: r['scores']['test']['per_length'][k]['f1']) for k in ('0', '1', '2')},
                                validation_f1=agg(lambda r: r['scores']['validation']['combined']['f1']), training_minutes=agg(lambda r: r['training_seconds']/60))
    assert not list((root/'arms').rglob('*.pt')) and not list((root/'arms').rglob('*.pth'))
    write(root/'summary.json', dict(complete=True, identity=identity(config), counts=spec['expected_counts'], methods=records, summary=summary, paired_rows_verified=True, no_weight_files=True))
    print(json.dumps({k: dict(test_f1=v['test_f1']['mean'], std=v['test_f1']['sample_std']) for k, v in summary.items()}), flush=True)


def runtime(spec, config, phase, state, code):
    append_line(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(config), phase=phase, state=state, exit_code=code,
        job_id=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def run_local(spec, config):
    """verify-source -> gpu-check -> the arm x seed trainings (``concurrency`` at a time) -> audit-results, on this host."""
    root = Path(spec['output']); logs = root/'logs'; logs.mkdir(parents=True, exist_ok=True)
    def step(phase, extra=(), log=None):
        command = [sys.executable, '-m', 'experiments.Task4C_FixedDatasetFMT_2_1', phase, '--config', config, *extra]
        runtime(spec, config, phase+(''.join(extra)), 'STARTED', None)
        with open(logs/((log or phase)+'.log'), 'a', encoding='utf8') as stream: process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        return process
    for phase in ('verify-source', 'gpu-check'):
        code = step(phase).wait(); runtime(spec, config, phase, 'ENDED', code); assert code == 0, phase
    pending = list(range(len(spec['candidates'])*len(spec['final']['seeds']))); running = {}
    while pending or running:
        while pending and len(running) < spec['concurrency']:
            i = pending.pop(0); running[i] = step('train', ['--index', str(i)], log=f'train_{i}')
        for i, process in list(running.items()):
            code = process.poll()
            if code is not None:
                runtime(spec, config, f'train--index{i}', 'ENDED', code); del running[i]
                if code != 0: raise RuntimeError(f'training index {i} exited with {code}; see {logs}/train_{i}.log')
        time.sleep(20)
    code = step('audit-results').wait(); runtime(spec, config, 'audit-results', 'ENDED', code); assert code == 0
    print(json.dumps(dict(complete=True, summary=str(root/'summary.json'))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['verify-source', 'gpu-check', 'train', 'audit-results', 'run-local', 'runtime'])
    p.add_argument('--config', default=CONFIG); p.add_argument('--index', type=int, default=0); p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    a = p.parse_args(); spec = json.loads(Path(a.config).read_text())
    if a.phase == 'verify-source': verify_source(spec, a.config)
    elif a.phase == 'gpu-check': gpu_check(spec, a.config)
    elif a.phase == 'train': train(spec, a.config, a.index)
    elif a.phase == 'audit-results': audit_results(spec, a.config)
    elif a.phase == 'run-local': run_local(spec, a.config)
    else: runtime(spec, a.config, a.runtime_phase, a.state, a.exit_code)


if __name__ == '__main__': main()
