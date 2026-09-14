"""Retain all Task4-c 4.14 directional features when adding new encoder blocks."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch

from experiments import Task4C_NearbySampling_4_13 as training
from experiments import Task4C_ManifoldMixup_4_4 as pipeline
from FMT_Utils.Task4C_RetainDirection_4_16 import WIDTHS, VERSIONS, BLOCKS, encode_lines, make_model, local_indices

DEFAULT_CONFIG = 'config/Verify_Task4C_RetainDirection_4.16.json'
SPLITS = ('train', 'validation', 'test')


def load_spec(config):
    spec = json.loads(Path(config).read_text())
    base_path = Path(spec['parent_config'])
    if pipeline.sha(base_path) != spec['parent_config_sha256']:
        raise ValueError('Frozen 4.14 configuration changed')
    base = json.loads(base_path.read_text())
    root = json.loads(Path(base['base_config']).read_text())
    if pipeline.sha(base['base_config']) != base['base_config_sha256']:
        raise ValueError('Frozen physical configuration changed')
    for key in ('training', 'candidates', 'expected_training_samples'):
        if spec[key] != base[key]:
            raise ValueError('Training settings differ from 4.14: ' + key)
    if spec['encoders'] != list(WIDTHS):
        raise ValueError('Unexpected prespecified method list')
    spec['flows'] = root['flows']
    return spec


def identity(config):
    manifest = json.loads(Path('SOURCE_MANIFEST.json').read_text())
    for name, expected in manifest.items():
        if pipeline.sha(name) != expected:
            raise ValueError('Scientific source changed: ' + name)
    return dict(commit=Path('SOURCE_COMMIT.txt').read_text().strip(), source_sha256=manifest,
                config_sha256=pipeline.sha(config), host=socket.gethostname(),
                job=os.environ.get('SLURM_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'))


def variant_spec(spec, encoder):
    result = copy.deepcopy(spec)
    result['output'] = str(Path(spec['output']) / 'variants' / encoder)
    result['methods'] = ['conv3d_mlp' if encoder == 'conv3d' else 'fmt_mlp']
    result['encoder'] = encoder
    return result


def plan(spec):
    return [(encoder, seed) for encoder in spec['encoders'] for seed in spec['training']['seeds']]


def folder(spec, encoder, seed):
    local = variant_spec(spec, encoder)
    return training.run_folder(local, local['methods'][0], spec['candidates'][0], seed)


def check_parent(spec, full=False):
    parent = Path(spec['parent_output'])
    summary = json.loads((parent / 'summary.json').read_text())
    if pipeline.sha(parent / 'summary.json') != spec['parent_summary_sha256']:
        raise ValueError('Frozen 4.14 summary changed')
    if summary['identity']['commit'] != spec['parent_commit']:
        raise ValueError('Wrong 4.14 scientific commit')
    for name, expected in summary['identity']['source_sha256'].items():
        if pipeline.sha(name) != expected:
            raise ValueError('Frozen parent implementation changed: ' + name)
    encoded = json.loads((parent / 'encoding.json').read_text())
    if encoded['identity']['config_sha256'] != spec['parent_config_sha256']:
        raise ValueError('Wrong encoded parent configuration')
    checked = {}
    for flow in spec['flows']:
        root = parent / 'physical' / flow['name']
        prepared = json.loads((root / 'preparation.json').read_text())
        if prepared['identity']['config_sha256'] != spec['parent_config_sha256']:
            raise ValueError('Wrong physical parent configuration')
        for split in SPLITS:
            key = flow['name'] + '/' + split
            paths = [(root / split / name, value) for name, value in prepared['splits'][split]['files'].items()]
            paths += [(parent / key / name, encoded['splits'][key]['files'][name]) for name in ('metadata.npz', 'voxels.npy', 'fmt.npy')]
            if full:
                for path, expected in paths:
                    if pipeline.sha(path) != expected:
                        raise ValueError('Parent data changed: ' + str(path))
            checked[key] = {str(p): h for p, h in paths}
    return checked


def encode(spec, config):
    parent = Path(spec['parent_output']); out = Path(spec['output'])
    checked = check_parent(spec, full=True)
    device = torch.device('cuda')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    manifests = {}
    for encoder in spec['encoders']:
        local = variant_spec(spec, encoder)
        manifests[encoder] = dict(version=spec['version'], encoder=encoder, identity=identity(config),
            splits={}, parent_data=checked, no_kinematic_or_IVD_inputs=True,
            line_width=WIDTHS[encoder], feature_blocks=BLOCKS[encoder],
            extra_direction_spectrum=True, reused_4_14_samples=True)
        Path(local['output']).mkdir(parents=True, exist_ok=False)
    old = json.loads((parent / 'encoding.json').read_text())
    for flow in spec['flows']:
        for split in SPLITS:
            key = flow['name'] + '/' + split; source = parent / 'physical' / key
            with np.load(source / 'metadata.npz') as z:
                counts = z['counts'].copy(); labels = z['labels'].copy()
            g = np.load(source / 'geometry.npy', mmap_mode='r'); seeds = np.load(source / 'seeds.npy', mmap_mode='r')
            original = np.load(parent / key / 'fmt.npy', mmap_mode='r')
            n = len(counts); files = {}
            for encoder in spec['encoders']:
                dest = Path(variant_spec(spec, encoder)['output']) / key
                dest.mkdir(parents=True)
                (dest / 'metadata.npz').symlink_to((source / 'metadata.npz').resolve())
                if encoder == 'fmt_4_14':
                    (dest / 'fmt.npy').symlink_to((parent / key / 'fmt.npy').resolve())
                else:
                    files[encoder] = np.lib.format.open_memmap(dest / 'fmt.npy', mode='w+', dtype='float32',
                        shape=(n, 27, WIDTHS[encoder] + 1))
            for start in range(0, n, 32):
                sl = slice(start, start+32)
                geometry = torch.as_tensor(np.array(g[sl]), device=device)
                ss = torch.as_tensor(np.array(seeds[sl]), device=device)
                cc = torch.as_tensor(counts[sl].astype(np.int64), device=device)
                for encoder, target in files.items():
                    target[sl] = encode_lines(geometry, ss, cc, encoder, torch.as_tensor(np.array(original[sl]), device=device)).cpu().numpy()
            for target in files.values():
                target.flush()
            for encoder in spec['encoders']:
                dest = Path(variant_spec(spec, encoder)['output']) / key
                name = 'fmt.npy'
                h = pipeline.sha(dest / name)
                manifests[encoder]['splits'][key] = dict(samples=n, class_counts=np.bincount(labels, minlength=2).tolist(),
                    files={name: h, 'metadata.npz': pipeline.sha(dest / 'metadata.npz')})
            print(json.dumps(dict(encoded=key, samples=n)), flush=True)
    for encoder, manifest in manifests.items():
        pipeline.write(Path(variant_spec(spec, encoder)['output']) / 'encoding.json', manifest)
    pipeline.write(out / 'encoding_audit.json', dict(identity=identity(config), parent_files=checked,
        encoders=spec['encoders'], unchanged_samples=True, full_parent_hash_check=True))


def train(spec, config, index):
    encoder, seed = plan(spec)[index]
    local = variant_spec(spec, encoder)
    training.identity = identity
    training.make_model = lambda method, candidate: make_model(encoder, candidate['dropout'])
    training.train(local, config, spec['training']['seeds'].index(seed))
    path = folder(spec, encoder, seed) / 'result.json'
    result = json.loads(path.read_text())
    result.update(encoder=encoder, encoder_version=VERSIONS[encoder],
        feature_blocks=[encoder], actual_feature_blocks=BLOCKS[encoder],
        test_scope='frozen_4.14_known_heads_local_spatial_blocks_already_used_benchmark',
        parent_commit=spec['parent_commit'], raw_coordinate_or_direction_bypass=True,
        no_kinematic_or_IVD_inputs=True)
    pipeline.write(path, result)


def merge(spec, config):
    rows = []
    for encoder in spec['encoders']:
        records = [json.loads((folder(spec, encoder, s) / 'result.json').read_text()) for s in spec['training']['seeds']]
        for r in records:
            if r['encoder'] != encoder or r['identity']['config_sha256'] != pipeline.sha(config):
                raise ValueError('Wrong result identity')
        row = dict(encoder=encoder, parameters=records[0]['parameters'], seeds=spec['training']['seeds'])
        for split in ('training', 'validation', 'test'):
            values = [r[split]['pooled']['f1'] for r in records]
            row[split] = dict(f1_mean=float(np.mean(values)), f1_std=float(np.std(values, ddof=1)), per_seed_f1=values,
                per_flow_f1_mean={flow['name']: float(np.mean([r[split]['per_flow'][flow['name']]['f1'] for r in records])) for flow in spec['flows']},
                average_precision_mean=float(np.mean([r[split]['pooled']['average_precision'] for r in records])))
        rows.append(row)
    parent = json.loads((Path(spec['parent_output']) / 'summary.json').read_text())
    result = dict(version=spec['version'], identity=identity(config), rows=rows, fixed_threshold=.5,
        historical_4_14_reference=parent['rows'], conv3d_reference_reused=True, same_4_14_samples=True, no_test_selection=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    pipeline.write(Path(spec['output']) / 'summary.json', result)
    print(json.dumps(rows), flush=True)


def runtime(spec, config, phase, state, code=None):
    from experiments.Record_Task1235_AngleFeatures_1_1 import append_record
    root = Path(spec['output']); events = root / 'lifecycle_events'; events.mkdir(parents=True, exist_ok=True)
    device = 'CPU'
    if phase in ('preflight', 'encode', 'train'):
        probe = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], capture_output=True, text=True)
        device = probe.stdout.strip()
    row = dict(version=spec['version'], phase=phase, state=state, exit_code=code,
        time=datetime.now(timezone.utc).isoformat(), device=device, **identity(config))
    pipeline.write(events / f"{row['job']}_{state}_{time.time_ns()}.json", row)
    append_record(root / 'runtime_events.jsonl', row, 'Task4C 4.16 runtime')
    append_record('docs/ibex_run_registry.md', row, 'Task4C 4.16 runtime')
    print(json.dumps(row), flush=True)


def submit(spec, config, phase, dependency=None):
    from experiments.Record_Task1235_AngleFeatures_1_1 import append_record
    root = Path(spec['output']); logs = root / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    gpu = phase in ('preflight', 'encode', 'train')
    command = ['sbatch', '--parsable', '--cpus-per-task=4', '--mem=64G',
        '--time=04:00:00' if phase == 'train' else '--time=01:00:00',
        '--job-name=t4c416-' + phase, '--output=' + str(logs / (phase + '.%A_%a.out')),
        '--error=' + str(logs / (phase + '.%A_%a.err'))]
    if gpu:
        command += ['--gres=gpu:1', '--constraint=a100|v100']
    if phase == 'train':
        command += ['--array=0-' + str(len(plan(spec))-1) + '%6']
    if dependency:
        command += ['--dependency=afterok:' + dependency]
    command += ['ibex_bash/task4c_retain_direction_4p16.sh', phase, config]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    row = dict(version=spec['version'], phase=phase, job_id=job, command=command,
        submitted_at_utc=datetime.now(timezone.utc).isoformat(), config=config,
        expected_device='one A100 or V100 GPU' if gpu else 'CPU', **identity(config))
    append_record(root / 'submissions.jsonl', row, 'Task4C 4.16 submitted')
    append_record('docs/ibex_run_registry.md', row, 'Task4C 4.16 submitted')
    print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('preflight', 'encode', 'train', 'merge', 'audit', 'submit', 'runtime'))
    p.add_argument('--config', default=DEFAULT_CONFIG); p.add_argument('--index', type=int, default=0)
    p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    p.add_argument('--submit-phase'); p.add_argument('--dependency')
    a = p.parse_args(); spec = load_spec(a.config)
    if a.phase == 'train': train(spec, a.config, a.index)
    elif a.phase == 'runtime': runtime(spec, a.config, a.runtime_phase, a.state, a.exit_code)
    elif a.phase == 'submit': submit(spec, a.config, a.submit_phase, a.dependency)
    elif a.phase in ('preflight', 'audit'):
        from experiments.Audit_Task4C_RetainDirection_4_16 import preflight, audit
        (preflight if a.phase == 'preflight' else audit)(spec, a.config)
    else: {'encode': encode, 'merge': merge}[a.phase](spec, a.config)


if __name__ == '__main__':
    main()
