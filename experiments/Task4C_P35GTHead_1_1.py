"""Frozen original Task4-c p35/h0 on the audited expanded GT-head dataset."""
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
from types import FunctionType

import numpy as np
import torch

from FMT_Utils.Task4C_LinePooling_4_3 import line_fmt
from FMT_Utils.FMT_V8_Search_2_1 import pooling_candidates, select_features
from experiments import Task4C_FPSAugmentSearch_1_1 as frozen
from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt

CONFIG = 'config/Verify_Task4C_P35GTHead_1.1.json'
POOL = next(p for p in pooling_candidates() if p['id'] == 'p35')
FILES = (
    'experiments/Task4C_P35GTHead_1_1.py',
    'experiments/Audit_Task4C_P35GTHead_1_1.py',
    'ibex_bash/task4c_p35_gt_head_1p1.sh',
    'FMT_Utils/DFT_FMT_3D.py', 'FMT_Utils/Task4C_LinePooling_4_3.py',
    'FMT_Utils/Task4C_Encoders_4_15.py', 'FMT_Utils/FMT_V8_Search_2_1.py',
    'FMT_Utils/Task4C_FPSAugmentSearch_1_1.py',
    'experiments/Task4C_FPSAugmentSearch_1_1.py',
    'FMT_Utils/Task4C_Multiscale_4_1.py',
)
sha, write = frozen.sha, frozen.write


def identity(config):
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                config_sha256=sha(config), sources={p: sha(p) for p in FILES})


def load_spec(config):
    spec = json.loads(Path(config).read_text())
    assert spec['candidate'] == dict(id='p35_h0', pool='p35', profile='h0', architecture='original',
                                    points=32, sampling='uniform', augmentation='none', parameters=76738)
    assert spec['encoding'] == dict(neighbors='nearest6', neighbor_scale=100., neighbor_weight=.5,
                                   frequencies=6, geometry='frozen_normalized_32_points',
                                   features='signed_log_zscore_clip8')
    assert spec['final']['seeds'] == [96721, 96722, 96723]
    return spec


def verify_source(spec, config):
    source = Path(spec['source_output'])
    assert sha(source/'data_audit.json') == spec['source_audit_sha256']
    previous = json.loads((source/'data_audit.json').read_text())
    assert previous['complete'] and previous['counts'] == spec['expected_counts']
    checked = []
    counts = {role: 0 for role in ('train', 'validation', 'test')}
    for flow in spec['flows']:
        name = flow['name']
        for role in counts:
            record = spec['source_files'][name][role]
            folder = source/'physical'/name/role
            for filename, digest in record['files'].items():
                actual = sha(folder/filename)
                assert actual == digest, (name, role, filename)
                checked.append(dict(flow=name, role=role, file=filename, sha256=actual))
            with np.load(folder/'metadata.npz') as z:
                assert len(z['labels']) == record['samples']
                assert np.all((z['counts'] >= 10) & (z['counts'] <= 27))
                assert set(np.unique(z['labels'])) == {0, 1}
                counts[role] += len(z['labels'])
    assert counts == spec['expected_counts'] and len(checked) == 18
    root = Path(spec['output'])
    write(root/'source_verification.json', dict(complete=True, identity=identity(config), counts=counts,
          files=checked, source_audit_sha256=spec['source_audit_sha256'], read_only_source=True))
    write(root/'selection.lock.json', dict(complete=True, selected=spec['candidate'], threshold=.5,
          selected_by='user_fixed_original_Task4C_p35_h0', identity=identity(config)))


@torch.no_grad()
def encode_batch(geometry, seeds, counts):
    # Preserve the original scale=100, weight=.5 encoder and nearest-seed rule.
    original = line_fmt(geometry, seeds, counts)
    features = select_features(original[..., :233], POOL)
    return torch.cat((features, original[..., -1:]), -1)


@torch.no_grad()
def fixed_neighbor_reference(geometry, counts, ids):
    """CPU reference conditional on the same material neighbors, for numeric QA only."""
    from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
    from FMT_Utils.FMT_P35_NormFrequency_3_1 import direction_spectrum
    mask = torch.arange(27, device=geometry.device)[None] < counts[:, None]
    bi, li = mask.nonzero(as_tuple=True)
    primitive = geometry[bi[:, None], ids[bi, li]]
    base = pathline_dft_features_3d(primitive, num_freq=6, neighbor_scale=100., neighbor_weight=.5,
                                  neighbor_pool='sort', mode='gram', include_chirality=True, return_numpy=False)
    values = torch.cat((base[:, :23], direction_spectrum(primitive[:, 0], 6),
                        frozen.method.pool_neighbors(base[:, 23:], POOL)), -1)
    return values


class Dataset:
    """Stream the fixed geometry into tokens, avoiding any resampling or copying of source files."""
    def __init__(self, spec, role, candidate, device='cuda', limit=None):
        if role == 'test':
            lock = Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"/'selection.lock.json'
            assert json.loads(lock.read_text())['test_loaded'] is False
        self.role, self.device = role, device
        self.parts = []
        labels, owners, flows, rows, counts = [], [], [], [], []
        for fi, flow in enumerate(spec['flows']):
            folder = Path(spec['source_output'])/'physical'/flow['name']/role
            with np.load(folder/'metadata.npz') as z:
                y, owner, c = z['labels'].copy(), z['instance'].copy(), z['counts'].astype(np.int64)
            ids = np.arange(len(y))
            if limit:
                ids = np.concatenate([np.flatnonzero(y == cls)[:limit//4] for cls in (0, 1)])
            self.parts.append((folder, ids, c[ids]))
            labels.append(y[ids]); owners.append(owner[ids]); counts.append(c[ids])
            flows.append(np.full(len(ids), fi)); rows.append(ids)
        self.labels = np.concatenate(labels).astype(np.int64)
        assert limit or len(self.labels) == spec['expected_counts'][role]
        self.targets = torch.tensor(self.labels, device=device)
        self.owners, self.flows, self.rows = np.concatenate(owners), np.concatenate(flows), np.concatenate(rows)
        self.counts = torch.tensor(np.concatenate(counts), device=device)
        # The frozen generic trainer passes neighbors; the original classifier ignores them.
        self.neighbors = torch.empty((len(self.labels), 0), dtype=torch.long, device=device)
        self.clean = None

    def encode(self, candidate, batch=128):
        self.clean = torch.empty((len(self.labels), 27, 142), device=self.device)
        offset = 0
        for folder, ids, counts in self.parts:
            g = np.load(folder/'geometry.npy', mmap_mode='r')
            s = np.load(folder/'seeds.npy', mmap_mode='r')
            for start in range(0, len(ids), batch):
                selected = ids[start:start+batch]
                values = encode_batch(torch.tensor(np.array(g[selected]), device=self.device),
                                      torch.tensor(np.array(s[selected]), device=self.device),
                                      torch.tensor(counts[start:start+batch], device=self.device))
                # Use the original NumPy log1p implementation before statistics.
                a = values.cpu().numpy()
                a[..., :-1] = transform_fmt(a[..., :-1], True)
                self.clean[offset:offset+len(selected)] = torch.from_numpy(a).to(self.device)
                offset += len(selected)
        assert offset == len(self.labels) and torch.isfinite(self.clean).all()


def fit_normalizer(dataset, candidate, batch=128):
    valid = dataset.clean[..., :-1][dataset.clean[..., -1] > .5].cpu().numpy()
    mean, std = valid.mean(0, dtype=np.float64), valid.std(0, dtype=np.float64)
    std[std < 1e-8] = 1.
    return (torch.tensor(mean, device=dataset.device), torch.tensor(std, device=dataset.device), len(valid))


def standardize(x, norm):
    v = ((x[..., :-1].double()-norm[0])/norm[1]).float().clamp(-8, 8)*x[..., -1:]
    return torch.cat((v, x[..., -1:]), -1)


# Reuse the frozen optimizer, scheduler, early stop, scoring, and checkpoint-in-memory logic.
predict = FunctionType(frozen.predict.__wrapped__.__code__, dict(frozen.__dict__, standardize=standardize),
                       argdefs=frozen.predict.__wrapped__.__defaults__)
predict = torch.no_grad()(predict)


def train(spec, config, index):
    root = Path(spec['output'])
    assert json.loads((root/'source_verification.json').read_text())['complete']
    assert json.loads((root/'gpu_check.json').read_text())['complete']
    spec = dict(spec, _active_seed=spec['final']['seeds'][index])
    environment = dict(frozen.__dict__, Dataset=Dataset, identity=identity, fit_normalizer=fit_normalizer,
                       standardize=standardize, predict=predict)
    FunctionType(frozen.train.__code__, environment, argdefs=frozen.train.__defaults__)(spec, config, index, 'final')


def gpu_check(spec, config):
    frozen.engine.deterministic('cuda'); torch.manual_seed(97011)
    assert json.loads((Path(spec['output'])/'source_verification.json').read_text())['complete']
    d = Dataset(spec, 'train', spec['candidate'], limit=32); d.encode(spec['candidate'])
    norm = fit_normalizer(d, spec['candidate'])
    model = frozen.method.FourierClassifier('p35', 'original', 'h0').cuda()
    assert sum(p.numel() for p in model.parameters()) == 76738
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    x = standardize(d.clean, norm); losses = []
    for _ in range(120):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), d.targets)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step(); losses.append(float(loss.detach()))
    p, _ = predict(model, d, norm, 128); score = frozen.metrics(d.labels, p)
    assert score['f1'] >= .95 and losses[-1] < losses[0]
    model.eval(); original = model(x).detach()
    changed = x.clone(); changed[..., :-1][changed[..., -1] < .5] = 12345.
    assert torch.allclose(model(changed), original, atol=1e-6, rtol=1e-6)
    model.train(); optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x.repeat(4, 1, 1)), d.targets.repeat(4)).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    # Native float32 nearest-neighbor ties may resolve differently on CPU and GPU.
    # Keep that full-pipeline discrepancy visible; compare arithmetic on fixed IDs.
    cpu = Dataset(spec, 'train', spec['candidate'], device='cpu', limit=32); cpu.encode(spec['candidate'], batch=8)
    a, b = cpu.clean.numpy(), d.clean.cpu().numpy()
    gpu_small = Dataset(spec, 'train', spec['candidate'], limit=32); gpu_small.encode(spec['candidate'], batch=8)
    assert torch.equal(gpu_small.clean, d.clean), 'V100 encoding changed with batch size'
    from FMT_Utils.Task4C_Encoders_4_15 import local_indices
    conditional_errors, changed_sets, radius_errors = [], 0, []
    for folder, selected, counts in d.parts:
        g = torch.tensor(np.array(np.load(folder/'geometry.npy', mmap_mode='r')[selected]))
        s = torch.tensor(np.array(np.load(folder/'seeds.npy', mmap_mode='r')[selected]))
        c = torch.tensor(counts)
        gpu_ids, _ = local_indices(g.cuda(), s.cuda(), c.cuda())
        cpu_ids, mask = local_indices(g, s, c)
        different = (cpu_ids[..., 1:].sort(-1).values != gpu_ids.cpu()[..., 1:].sort(-1).values).any(-1) & mask
        changed_sets += int(different.sum())
        distances = torch.cdist(s.double(), s.double())
        for bi, li in different.nonzero().tolist():
            radius_errors.append(float(abs(distances[bi, li, cpu_ids[bi, li, 1:]].max()
                                            - distances[bi, li, gpu_ids[bi, li, 1:].cpu()].max())))
        reference = fixed_neighbor_reference(g, c, gpu_ids.cpu()).numpy()
        actual = fixed_neighbor_reference(g.cuda(), c.cuda(), gpu_ids).cpu().numpy()
        assert np.allclose(reference, actual, atol=2e-4, rtol=2e-4)
        conditional_errors.append(float(np.max(np.abs(reference-actual))))
    write(Path(spec['output'])/'gpu_check.json', dict(complete=True, identity=identity(config),
          samples=32, parameters=76738, gpu=torch.cuda.get_device_name(), pilot_f1=score['f1'],
          initial_loss=losses[0], final_loss=losses[-1], batch128_backward=True,
          cpu_gpu_full_pipeline_allclose=bool(np.allclose(a, b, atol=2e-4, rtol=2e-4)),
          cpu_gpu_log_features_max_error=float(np.max(np.abs(a-b))),
          cpu_gpu_different_neighbor_sets=changed_sets,
          max_selected_radius_difference_float64=max(radius_errors, default=0.),
          fixed_neighbor_cpu_gpu_max_error=max(conditional_errors),
          fixed_neighbor_cpu_gpu_pass=True, v100_batch_exact=True, scientific_metric=False))


def runtime(spec, config, phase, state, exit_code):
    frozen.append_locked(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(config),
        phase=phase, state=state, exit_code=exit_code, hostname=socket.gethostname(),
        job_id=os.environ.get('SLURM_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def submit(spec, config):
    root = Path(spec['output']); (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(), 'Already submitted'
    jobs = {}
    for phase, dependency, array, gpu, walltime in (
        ('verify-source', None, None, False, '00:20:00'),
        ('gpu-check', 'verify-source', None, True, '00:20:00'),
        ('train', 'gpu-check', '0-2%3', True, '03:00:00'),
        ('audit', 'train', None, False, '00:20:00')):
        command = ['sbatch', '--parsable', '--propagate=NONE', '--job-name=P35GThead_'+phase,
            '--cpus-per-task=4', '--mem=16G', '--time='+walltime,
            '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if dependency: command += ['--dependency=afterok:'+jobs[dependency]]
        if array: command += ['--array='+array]
        if gpu: command += ['--gres=gpu:1', '--constraint=v100', '--exclude=gpu213-18']
        command += ['ibex_bash/task4c_p35_gt_head_1p1.sh', phase, config]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        assert job.isdigit(); jobs[phase] = job
        record = dict(phase=phase, job=job, command=command, identity=identity(config),
                      submitted_at_utc=datetime.now(timezone.utc).isoformat(), expected_device='V100' if gpu else 'CPU')
        frozen.append_locked(root/'submissions.jsonl', json.dumps(record)+'\n')
        write(root/'submission.json', dict(identity=identity(config), jobs=jobs))
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['verify-source', 'gpu-check', 'train', 'audit', 'submit', 'runtime'])
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--runtime-phase'); parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec = load_spec(args.config)
    if args.phase == 'verify-source': verify_source(spec, args.config)
    elif args.phase == 'gpu-check': gpu_check(spec, args.config)
    elif args.phase == 'train': train(spec, args.config, args.index)
    elif args.phase == 'submit': submit(spec, args.config)
    elif args.phase == 'audit':
        from experiments.Audit_Task4C_P35GTHead_1_1 import audit
        audit(spec, args.config)
    else: runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)


if __name__ == '__main__': main()
