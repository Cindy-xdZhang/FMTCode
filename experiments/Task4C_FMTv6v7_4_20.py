"""Rerun user-specified FMT V6/V7 on frozen Task4-c 4.14 with three seeds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments import Task4C_RetainDirection_4_16 as parent
from experiments import Task4C_NearbySampling_4_13 as training
from experiments.Record_Task1235_AngleFeatures_1_1 import append_record
from FMT_Utils.Task4C_FMTv6v7_4_20 import WIDTHS, VERSIONS, BLOCKS, make_model, select_cached, fmt_V6, fmt_V7
from FMT_Utils.Task4C_LinePooling_4_3 import LearnedLinePooling

DEFAULT_CONFIG = 'config/Ablation_Task4C_FMTv6v7_4.20.json'
SPLITS = parent.SPLITS
pipeline = parent.pipeline
identity = parent.identity
variant_spec = parent.variant_spec
folder = parent.folder
plan = parent.plan
check_parent = parent.check_parent


def load_spec(config):
    spec = json.loads(Path(config).read_text())
    if pipeline.sha(spec['parent_config']) != spec['parent_config_sha256']:
        raise ValueError('Frozen 4.14 configuration changed')
    base = json.loads(Path(spec['parent_config']).read_text())
    if pipeline.sha(base['base_config']) != base['base_config_sha256']:
        raise ValueError('Frozen physical configuration changed')
    for key in ('training', 'candidates', 'expected_training_samples'):
        if spec[key] != base[key]:
            raise ValueError('Training differs from 4.14: ' + key)
    if spec['encoders'] != list(WIDTHS):
        raise ValueError('Unexpected encoder list')
    spec['flows'] = json.loads(Path(base['base_config']).read_text())['flows']
    return spec


def preflight(spec, config):
    """Check real training fixtures, then copy exact frozen cache columns."""
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    if not torch.cuda.is_available() or 'V100' not in torch.cuda.get_device_name(0):
        raise ValueError('Prespecified V100 device required')
    source = Path(spec['parent_output']); root = Path(spec['output'])
    checked = check_parent(spec, full=True)
    checks = []
    for flow in spec['flows']:
        key = flow['name'] + '/train'; physical = source / 'physical' / key
        with np.load(physical / 'metadata.npz') as z:
            ids = np.concatenate([np.flatnonzero(z['labels'] == c)[:16] for c in (0, 1)])
            counts = torch.as_tensor(z['counts'][ids].astype(np.int64), device='cuda')
        g = torch.as_tensor(np.array(np.load(physical / 'geometry.npy', mmap_mode='r')[ids]), device='cuda')
        s = torch.as_tensor(np.array(np.load(physical / 'seeds.npy', mmap_mode='r')[ids]), device='cuda')
        cached = torch.as_tensor(np.array(np.load(source / key / 'fmt.npy', mmap_mode='r')[ids]), device='cuda')
        exact = fmt_V6(g, s, counts)
        if not torch.allclose(exact, cached, atol=1e-5, rtol=1e-5) or not torch.equal(fmt_V7(g, s, counts), select_cached(exact, 'fmt_V7')):
            raise ValueError('Public encoder differs from frozen cached coefficients')
        models = {}
        for encoder in spec['encoders']:
            token = select_cached(cached, encoder)
            mask = token[..., -1].bool()
            if token.shape[-1] != WIDTHS[encoder]+1 or torch.any(token[~mask] != 0):
                raise ValueError('Incorrect width or padded values')
            torch.manual_seed(96611); model = make_model(encoder, .15).cuda()
            torch.manual_seed(96611); old = LearnedLinePooling(.15).cuda()
            for name, value in model.state_dict().items():
                if encoder == 'fmt_V6' or not name.startswith('line.0.'):
                    if not torch.equal(value, old.state_dict()[name]):
                        raise ValueError('Unexpected initialization change: ' + name)
            if encoder == 'fmt_V6':
                model.eval(); old.eval()
                with torch.no_grad():
                    if not torch.equal(model(token), old(cached)):
                        raise ValueError('V6 original forward differs')
            model.train(); loss = model(token).square().mean(); loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Invalid model forward/backward')
            count = sum(p.numel() for p in model.parameters())
            if count != {'fmt_V6': 88514, 'fmt_V7': 70850}[encoder]:
                raise ValueError('Unexpected parameter count')
            models[encoder] = count
        checks.append(dict(flow=flow['name'], training_rows=ids.tolist(), recomputation_max_error=float((exact-cached).abs().max()),
            recomputation_tolerance=dict(atol=1e-5, rtol=1e-5), cache_column_selection_exact=True,
            v6_initialization_forward_exact=True, v7_hidden_initialization_exact=True, parameters=models))
    manifests = {}
    for encoder in spec['encoders']:
        dest = Path(variant_spec(spec, encoder)['output']); dest.mkdir(parents=True, exist_ok=False)
        manifests[encoder] = dict(identity=identity(config), encoder=encoder, encoder_version=VERSIONS[encoder],
            line_width=WIDTHS[encoder], feature_blocks=BLOCKS[encoder], splits={})
    for flow in spec['flows']:
        for split in SPLITS:
            key = flow['name'] + '/' + split
            old = np.load(source / key / 'fmt.npy', mmap_mode='r')
            for encoder in spec['encoders']:
                dest = Path(variant_spec(spec, encoder)['output']) / key; dest.mkdir(parents=True)
                (dest / 'metadata.npz').symlink_to((source / 'physical' / key / 'metadata.npz').resolve())
                if encoder == 'fmt_V6':
                    (dest / 'fmt.npy').symlink_to((source / key / 'fmt.npy').resolve())
                else:
                    target = np.lib.format.open_memmap(dest / 'fmt.npy', mode='w+', dtype=old.dtype,
                        shape=(*old.shape[:2], 96))
                    for start in range(0, len(old), 256):
                        sl = slice(start, start+256)
                        target[sl] = np.concatenate((old[sl, :, :23], old[sl, :, 161:234]), -1)
                        if not np.array_equal(target[sl, :, :23], old[sl, :, :23]) or not np.array_equal(target[sl, :, 23:], old[sl, :, 161:234]):
                            raise ValueError('Changed V7 retained feature values')
                    target.flush(); del target
                manifests[encoder]['splits'][key] = dict(samples=len(old), files={
                    n: pipeline.sha(dest / n) for n in ('fmt.npy', 'metadata.npz')})
    for encoder, manifest in manifests.items():
        pipeline.write(Path(variant_spec(spec, encoder)['output']) / 'encoding.json', manifest)
    pipeline.write(root / 'preflight.json', dict(status='PASS', identity=identity(config), checks=checks,
        parent_files=checked, exact_cache_column_selection=True, no_kinematic_or_angle_inputs=True,
        no_test_labels_or_metrics_used=True, frozen_training_function_reused=True))


def train(spec, config, index):
    encoder, seed = plan(spec)[index]
    if not torch.cuda.is_available() or 'V100' not in torch.cuda.get_device_name(0):
        raise ValueError('Prespecified V100 device required')
    local = variant_spec(spec, encoder)
    training.identity = identity
    training.make_model = lambda method, candidate: make_model(encoder, candidate['dropout'])
    training.train(local, config, spec['training']['seeds'].index(seed))
    path = folder(spec, encoder, seed) / 'result.json'; result = json.loads(path.read_text())
    result.update(encoder=encoder, encoder_version=VERSIONS[encoder], feature_blocks=[encoder],
        actual_feature_blocks=BLOCKS[encoder], input_features=WIDTHS[encoder], parent_commit=spec['parent_commit'],
        test_scope=spec['test_scope'], no_kinematic_or_angle_inputs=True)
    pipeline.write(path, result)


def merge(spec, config):
    parent.merge(spec, config)
    from experiments import Audit_Task4C_Encoders_4_15 as audit
    import sys
    audit.run = sys.modules[__name__]
    audit.audit(spec, config)
    comparisons = []
    for seed in spec['training']['seeds']:
        new = json.loads((folder(spec, 'fmt_V6', seed) / 'result.json').read_text())
        old_folder = Path(spec['parent_output']) / 'runs' / ('regularized_fmt_mlp_seed' + str(seed))
        old = json.loads((old_folder / 'result.json').read_text())
        if new['normalization'] != old['normalization'] or new['selected_epoch'] != old['selected_epoch']:
            raise ValueError('V6 normalization or selected epoch differs from frozen baseline')
        errors = {}
        with np.load(folder(spec, 'fmt_V6', seed) / 'predictions.npz') as a, np.load(old_folder / 'predictions.npz') as b:
            for split in SPLITS:
                key = split + '_probability'
                errors[split] = float(np.abs(a[key]-b[key]).max())
                if not np.array_equal(a[key], b[key]):
                    raise ValueError('V6 predictions differ from frozen baseline: ' + split)
        comparisons.append(dict(seed=seed, selected_epoch=new['selected_epoch'], prediction_max_error=errors))
    pipeline.write(Path(spec['output']) / 'baseline_reproduction.json', dict(status='PASS', comparisons=comparisons,
        all_three_seeds_fresh_training=True, exact_predictions=True, baseline_commit=spec['parent_commit']))


def runtime(spec, config, phase, state, code=None):
    root = Path(spec['output']); events = root / 'lifecycle_events'; events.mkdir(parents=True, exist_ok=True)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    row = dict(version=spec['version'], phase=phase, state=state, exit_code=code,
        time=datetime.now(timezone.utc).isoformat(), device=device, **identity(config))
    pipeline.write(events / f"{row['job']}_{state}_{time.time_ns()}.json", row)
    append_record(root / 'runtime_events.jsonl', row, 'Task4C 4.20 runtime')
    append_record('docs/ibex_run_registry.md', row, 'Task4C 4.20 runtime')


def submit(spec, config, phase, dependency=None):
    root = Path(spec['output']); logs = root / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    gpu = phase in ('preflight', 'train')
    command = ['sbatch', '--parsable', '--cpus-per-task=4', '--mem=32G', '--time=01:00:00',
        '--job-name=t4c420-' + phase, '--output=' + str(logs / (phase + '.%A_%a.out')),
        '--error=' + str(logs / (phase + '.%A_%a.err'))]
    if gpu: command += ['--gres=gpu:1', '--constraint=v100']
    if phase == 'train': command += ['--array=0-5%6']
    if dependency: command += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
    command += ['ibex_bash/task4c_fmt_v6v7_4p20.sh', phase, config]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    row = dict(version=spec['version'], phase=phase, job_id=job, command=command,
        submitted_at_utc=datetime.now(timezone.utc).isoformat(), config=config,
        expected_device='one Tesla V100 GPU' if gpu else 'CPU', **identity(config))
    append_record(root / 'submissions.jsonl', row, 'Task4C 4.20 submitted')
    append_record('docs/ibex_run_registry.md', row, 'Task4C 4.20 submitted')
    print(json.dumps({k:v for k,v in row.items() if k != 'source_sha256'}), flush=True)
    return job


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('preflight', 'train', 'merge', 'runtime', 'submit'))
    p.add_argument('--config', default=DEFAULT_CONFIG); p.add_argument('--index', type=int, default=0)
    p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    p.add_argument('--submit-phase'); p.add_argument('--dependency')
    a = p.parse_args(); spec = load_spec(a.config)
    if a.phase == 'train': train(spec, a.config, a.index)
    elif a.phase == 'runtime': runtime(spec, a.config, a.runtime_phase, a.state, a.exit_code)
    elif a.phase == 'submit': submit(spec, a.config, a.submit_phase, a.dependency)
    else: {'preflight': preflight, 'merge': merge}[a.phase](spec, a.config)


if __name__ == '__main__': main()
