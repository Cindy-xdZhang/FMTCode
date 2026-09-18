"""Fixed-center FMT arms on the Task4-c fixed bundle dataset v1 (mainExp_Task4C_FixedDatasetFMT_1.1).

Four arms from Ablation_Task4C_FPS16_1.x (p35/h0 and c156 with FPS16, plus their six-neighbour
controls), three seeds each, trained on the audited read-only files of
mainExp_Task4C_FixedDataset_1.1.  Test scores are reported for the whole test set, the
held-out-instance rows, the covered-instance offset positives and per GT instance.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from types import FunctionType, SimpleNamespace
import numpy as np
import torch
from FMT_Utils import Task4C_FPS16_1_1 as method
from FMT_Utils import Task4C_FixedDataset_1_1 as data
from experiments import Task4C_FPS16_1_2 as base
from experiments import Task4C_OriginalCenter_1_1 as previous

frozen, p35 = previous.frozen, previous.p35
sha, write = frozen.sha, frozen.write
CONFIG = 'config/mainExp_Task4C_FixedDatasetFMT_1.1.json'
FILES = base.FILES+('experiments/Task4C_FixedDatasetFMT_1_1.py', 'FMT_Utils/Task4C_FixedDataset_1_1.py', 'config/mainExp_Task4C_FixedDataset_1.1.json')


def identity(config):
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                config_sha256=sha(config), sources={p: sha(p) for p in FILES})


def verify_source(spec, config):
    """The data set is read-only: every file must match the frozen audit hashes."""
    source = Path(spec['source_output']); audit = json.loads((source/'data_audit.json').read_text())
    assert audit['complete'] and sha(source/'data_audit.json') == spec['source_audit_sha256']
    checked = []; counts = {r: 0 for r in data.ROLES}
    for flow in spec['flows']:
        for role in data.ROLES:
            folder = source/'physical'/flow['name']/role
            for filename, digest in audit['frozen_files'][f"{flow['name']}/{role}"].items():
                assert sha(folder/filename) == digest, (flow['name'], role, filename); checked.append(str(folder/filename))
            with np.load(folder/'metadata.npz') as z: counts[role] += len(z['labels'])
    assert counts == spec['expected_counts'] and len(checked) == 30
    write(Path(spec['output'])/'source_verification.json', dict(complete=True, identity=identity(config), counts=counts, files=checked, read_only_source=True))


class Dataset:
    def __init__(self, spec, role, candidate, device='cuda', limit=None):
        self.device = device; self.parts = []; self.role = role
        if role == 'test':
            lock = Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"/'selection.lock.json'
            assert json.loads(lock.read_text())['test_loaded'] is False
        self.evidence = None if limit else Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"
        labels = []; owners = []; flows = []; rows = []; centers = []; extra = {k: [] for k in ('relaxed', 'heldout_instance', 'offset_test', 'sample_kind', 'center_in_gt_head')}
        for fi, flow in enumerate(spec['flows']):
            folder = Path(spec['source_output'])/'physical'/flow['name']/role; m = data.rules.metadata(folder)
            with np.load(folder/data.INDEX_FILE) as z: idx = {k: z[k] for k in z.files}
            ids = np.arange(len(m['labels']))
            if limit: ids = np.concatenate([np.flatnonzero(m['labels'] == cls)[:limit//4] for cls in (0, 1)])
            anchor = idx['original_center_id'][ids]; assert np.all(anchor == 0) and np.all(m['counts'][ids] >= 17)
            self.parts.append((folder, ids, m['counts'][ids], anchor))
            labels.append(m['labels'][ids]); owners.append(m['instance'][ids]); flows.append(np.full(len(ids), fi)); rows.append(ids); centers.append(anchor)
            for k in extra: extra[k].append(idx[k][ids])
        self.labels = np.concatenate(labels).astype(np.int64); self.owners = np.concatenate(owners); self.flows = np.concatenate(flows)
        self.rows = np.concatenate(rows); self.anchor = np.concatenate(centers); self.extra = {k: np.concatenate(v) for k, v in extra.items()}
        assert limit or len(self.labels) == spec['expected_counts'][role]
        self.targets = torch.tensor(self.labels, device=device); self.neighbors = torch.empty((len(self.labels), 0), device=device, dtype=torch.long); self.clean = None

    def encode(self, candidate, batch=128):
        self.clean = torch.empty((len(self.labels), 1, 142), device=self.device); offset = 0; neighbors = []
        for folder, rows, counts, anchor in self.parts:
            g = np.load(folder/'geometry.npy', mmap_mode='r'); s = np.load(folder/'seeds.npy', mmap_mode='r')
            for first in range(0, len(rows), batch):
                ids = rows[first:first+batch]
                tokens, n = method.encode(torch.tensor(np.array(g[ids]), device=self.device), torch.tensor(np.array(s[ids]), device=self.device),
                                          torch.tensor(counts[first:first+batch], device=self.device, dtype=torch.long),
                                          torch.tensor(anchor[first:first+batch], device=self.device, dtype=torch.long), candidate)
                if candidate['base_method'] == 'p35_h0':
                    values = tokens.cpu().numpy(); values[..., :141] = p35.transform_fmt(values[..., :141], True); tokens = torch.from_numpy(values).to(self.device)
                self.clean[offset:offset+len(ids)] = tokens; offset += len(ids); neighbors.append(n.cpu().numpy())
        assert offset == len(self.labels) and torch.isfinite(self.clean).all(); self.selected_neighbors = np.concatenate(neighbors)
        assert self.selected_neighbors.shape == (len(self.labels), candidate['neighbors']) and np.all(self.selected_neighbors != 0)
        assert all(len(np.unique(row)) == candidate['neighbors'] for row in self.selected_neighbors)
        if self.evidence:
            write(self.evidence/f'{self.role}_encoding.json', dict(candidate=candidate, shape=list(self.clean.shape), samples=len(self.labels),
                  center_policy='slot0_original_center_only', neighbors=candidate['neighbors'],
                  neighbor_ids_sha256=hashlib.sha256(self.selected_neighbors.astype('<i8').tobytes()).hexdigest()))


def arm_spec(spec, index):
    seeds = spec['final']['seeds']; arm = copy.deepcopy(spec); arm['_root_output'] = spec['output']
    arm['candidate'] = spec['candidates'][index//len(seeds)]; arm['_active_seed'] = seeds[index % len(seeds)]
    arm['final'] = dict(spec['final'], seeds=[arm['_active_seed']]); arm['output'] = str(Path(spec['output'])/'arms'/arm['candidate']['id']/f"seed{arm['_active_seed']}")
    return arm


def save_predictions(folder, role, dataset, probability):
    file = folder/f'{role}_predictions.npz'
    np.savez_compressed(file, labels=dataset.labels, probability=probability, instance=dataset.owners, flow_index=dataset.flows, row_in_split=dataset.rows,
                        original_center_id=dataset.anchor, threshold=.5, **dataset.extra)
    return sha(file)


def environment(candidate):
    standardize = p35.standardize if candidate['base_method'] == 'p35_h0' else frozen.standardize
    predict = FunctionType(frozen.predict.__wrapped__.__code__, dict(frozen.__dict__, standardize=standardize), argdefs=frozen.predict.__wrapped__.__defaults__)
    constructor = lambda pool, architecture, profile: method.make_model(candidate['base_method'])
    return dict(frozen.__dict__, Dataset=Dataset, identity=identity, fit_normalizer=previous.fit_normalizer,
                method=SimpleNamespace(**dict(frozen.method.__dict__, FourierClassifier=constructor)),
                standardize=standardize, predict=torch.no_grad()(predict), save_predictions=save_predictions)


def gpu_check(spec, config):
    frozen.engine.deterministic('cuda'); assert 'V100' in torch.cuda.get_device_name(); torch.set_num_threads(4); checks = {}
    for index, c in enumerate(spec['candidates']):
        torch.manual_seed(97811); arm = arm_spec(spec, index*len(spec['final']['seeds'])); d = Dataset(arm, 'train', c, limit=32); d.encode(c)
        small = Dataset(arm, 'train', c, limit=32); small.encode(c, batch=8)
        assert np.array_equal(d.selected_neighbors, small.selected_neighbors); torch.testing.assert_close(d.clean, small.clean, atol=2e-4, rtol=2e-4)
        env = environment(c); norm = previous.fit_normalizer(d, c); xx = env['standardize'](d.clean, norm)
        model = method.make_model(c['base_method']).cuda(); optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001); losses = []
        for step in range(150):
            model.train(); optim.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(xx), d.targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True); optim.step(); losses.append(float(loss.detach()))
        probability, _ = env['predict'](model, d, norm, 128); score = frozen.metrics(d.labels, probability)
        assert score['f1'] >= .95 and losses[-1] < losses[0]
        checks[c['id']] = dict(parameters=sum(p.numel() for p in model.parameters()), shape=list(d.clean.shape), neighbors=c['neighbors'], pilot_f1=score['f1'],
                               modules=sorted(set(type(m).__name__ for m in model.modules())))
    write(Path(spec['output'])/'gpu_check.json', dict(complete=True, identity=identity(config), gpu=torch.cuda.get_device_name(), checks=checks, engineering_only=True))


def train(spec, config, index):
    for filename in ('gpu_check.json', 'source_verification.json'):
        result = json.loads((Path(spec['output'])/filename).read_text()); assert result['complete'] and result['identity'] == identity(config)
    arm = arm_spec(spec, index); write(Path(arm['output'])/'selection.lock.json', dict(complete=True, selected=arm['candidate'], seed=arm['_active_seed'], identity=identity(config)))
    env = environment(arm['candidate']); FunctionType(frozen.train.__code__, env, argdefs=frozen.train.__defaults__)(arm, config, 0, 'final')


def audit_results(spec, config):
    root = Path(spec['output']); source = Path(spec['source_output']); records = []; paired = {}
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
            key = np.column_stack([pred[k] for k in ('labels', 'instance', 'flow_index', 'row_in_split', 'heldout_instance', 'offset_test', 'relaxed')])
            if role in paired: assert np.array_equal(key, paired[role])
            else: paired[role] = key
            for fi, flow in enumerate(spec['flows']):
                mask = pred['flow_index'] == fi; m = data.rules.metadata(source/'physical'/flow['name']/role)
                with np.load(source/'physical'/flow['name']/role/data.INDEX_FILE) as z: idx = {k: z[k] for k in z.files}
                assert np.array_equal(pred['row_in_split'][mask], np.arange(len(m['labels']))) and np.array_equal(pred['labels'][mask], m['labels'])
                assert np.array_equal(pred['instance'][mask], m['instance']) and np.array_equal(pred['heldout_instance'][mask], idx['heldout_instance'])
            y = pred['labels']; p = pred['probability']; assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            score = dict(f1=float(f1_score(y, p >= .5)), precision=float(precision_score(y, p >= .5)), recall=float(recall_score(y, p >= .5)))
            expected = r['validation'] if role == 'validation' else r['test']['combined']
            for k, v in score.items(): assert abs(v-expected[k]) < 1e-12
            subsets = dict(heldout_instances=pred['heldout_instance'], covered_instances=~pred['heldout_instance'],
                           covered_offset_positive=pred['offset_test'], gt_head_rows=pred['sample_kind'] == data.KIND_GT_HEAD, relaxed=pred['relaxed'])
            per_instance = {}
            for fi, flow in enumerate(spec['flows']):
                for i in np.unique(pred['instance'][(pred['flow_index'] == fi) & (y == 1)]):
                    take = (pred['flow_index'] == fi) & (y == 1) & (pred['instance'] == i)
                    per_instance[f"{flow['name']}/{int(i)}"] = dict(positives=int(take.sum()), recall=float(np.mean(p[take] >= .5)), heldout=bool(pred['heldout_instance'][take][0]))
            scores[role] = dict(combined=frozen.metrics(y, p),
                                per_flow={flow['name']: frozen.metrics(y[pred['flow_index'] == fi], p[pred['flow_index'] == fi]) for fi, flow in enumerate(spec['flows'])},
                                **{k: (frozen.metrics(y[v], p[v]) if v.any() and len(np.unique(y[v])) == 2 else dict(samples=int(v.sum()), recall=float(np.mean(p[v] >= .5)) if v.any() else None)) for k, v in subsets.items()},
                                per_instance_recall=per_instance)
        records.append(dict(candidate=c, seed=seed, parameters=r['parameters'], epochs=r['epochs'], selected_epoch=r['selected_epoch'],
                            training_seconds=r['training_seconds'], preparation_seconds=r['preparation_seconds'], scores=scores, result_sha256=sha(folder/'result.json')))
    summary = {}
    for c in spec['candidates']:
        rows = [r for r in records if r['candidate']['id'] == c['id']]
        def agg(path):
            values = np.array([eval('r'+path, {}, dict(r=r)) for r in rows], float)
            return dict(mean=float(values.mean()), sample_std=float(values.std(ddof=1)) if len(values) > 1 else None, values=values.tolist())
        summary[c['id']] = dict(parameters=c['parameters'], seeds=[r['seed'] for r in rows],
                                test_f1=agg("['scores']['test']['combined']['f1']"), test_ap=agg("['scores']['test']['combined']['average_precision']"),
                                heldout_f1=agg("['scores']['test']['heldout_instances']['f1']"), covered_f1=agg("['scores']['test']['covered_instances']['f1']"),
                                validation_f1=agg("['scores']['validation']['combined']['f1']"), training_minutes=agg("['training_seconds']/60"))
    assert not list((root/'arms').rglob('*.pt')) and not list((root/'arms').rglob('*.pth'))
    write(root/'summary.json', dict(complete=True, identity=identity(config), counts=spec['expected_counts'], methods=records, summary=summary,
                                    all_centers_fixed=True, paired_rows_verified=True, no_weight_files=True))


def runtime(spec, config, phase, state, code):
    frozen.append_locked(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(config), phase=phase, state=state, exit_code=code,
        job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        hostname=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def submit(spec, config):
    root = Path(spec['output']); (root/'logs').mkdir(parents=True, exist_ok=True); assert not (root/'submission.json').exists(); jobs = {}
    runs = len(spec['candidates'])*len(spec['final']['seeds'])
    phases = [('verify-source', [], None, False, '00:30:00', '32G'), ('gpu-check', ['verify-source'], None, True, '00:15:00', '32G'),
              ('train', ['gpu-check'], f'0-{runs-1}%{spec["concurrency"]}', True, '06:00:00', '64G'), ('audit-results', ['train'], None, False, '00:30:00', '32G')]
    for phase, deps, array, gpu, wall, memory in phases:
        command = ['sbatch', '--parsable', '--propagate=NONE', '--job-name=FixedFMT11_'+phase, '--cpus-per-task=4', '--mem='+memory, '--time='+wall,
                   '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps: command += ['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        if array: command += ['--array='+array]
        if gpu: command += ['--gres=gpu:1', '--constraint=v100', '--exclude=gpu213-18']
        command += ['ibex_bash/task4c_fixed_dataset_fmt_1p1.sh', phase, config]
        jobs[phase] = subprocess.check_output(command, text=True).strip().split(';')[0]
        frozen.append_locked(root/'submissions.jsonl', json.dumps(dict(phase=phase, job=jobs[phase], command=command, submitted_at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        write(root/'submission.json', dict(identity=identity(config), jobs=jobs))
    print(json.dumps(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['verify-source', 'gpu-check', 'train', 'audit-results', 'submit', 'runtime'])
    p.add_argument('--config', default=CONFIG); p.add_argument('--index', type=int, default=0); p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    a = p.parse_args(); spec = json.loads(Path(a.config).read_text())
    if a.phase == 'verify-source': verify_source(spec, a.config)
    elif a.phase == 'gpu-check': gpu_check(spec, a.config)
    elif a.phase == 'train': train(spec, a.config, a.index)
    elif a.phase == 'audit-results': audit_results(spec, a.config)
    elif a.phase == 'submit': submit(spec, a.config)
    else: runtime(spec, a.config, a.runtime_phase, a.state, a.exit_code)


if __name__ == '__main__': main()
