"""Change only voxel resolution for the frozen 83,918-parameter Conv3D."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments import Task4C_ConvCapacity_4_18 as capacity
from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_voxels

pipeline = capacity.pipeline
training = capacity.training
CONFIG = 'config/Ablation_Task4C_ConvResolution_4.22.json'
CAPACITY_IDENTITY = capacity.identity


def load_spec(config):
    spec = capacity.load_spec(config)
    assert pipeline.sha(spec['capacity_config']) == spec['capacity_config_sha256']
    assert pipeline.sha('experiments/Task4C_ConvCapacity_4_18.py') == spec['capacity_runner_sha256']
    assert spec['voxel_resolutions'] == [32, 36, 48] and spec['voxel_subdivisions'] == 4
    assert spec['training']['seeds'] == [96611, 96612, 96613]
    assert spec['training']['batch_size'] == 128
    return spec


def identity(config):
    result = CAPACITY_IDENTITY(config)
    spec = json.loads(Path(config).read_text())
    for name in ('experiments/Task4C_ConvResolution_4_22.py',
                 'ibex_bash/task4c_conv_resolution_4p22.sh', spec['capacity_config']):
        result['source_sha256'][name] = pipeline.sha(name)
    return result


def require_data_confirmation(spec):
    if spec['data_revision_status'] != 'user_confirmed_original_4.14_data_and_evaluator':
        raise ValueError('Clarify the user-corrected 24-cubed data/evaluator before production runs')


def resolution_spec(spec, resolution):
    assert resolution in spec['voxel_resolutions']
    local = copy.deepcopy(spec)
    local['output'] = str(Path(spec['output']) / f'r{resolution}')
    local['voxel_resolution'] = resolution
    return local


@contextmanager
def capacity_identity():
    original = capacity.identity
    try:
        capacity.identity = identity
        yield
    finally:
        capacity.identity = original


def reference_voxel(geometry, count, resolution):
    """Independent NumPy trilinear scatter for one valid, normalized bundle."""
    # Match float32 interpolation coordinates, including exact grid-plane hits;
    # accumulate independently in float64 to check the scatter sums.
    lines = np.asarray(geometry[:count], dtype=np.float32)
    delta = np.diff(lines, axis=1)
    tangent = delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-12)
    fractions = np.arange(5, dtype=np.float32) / 4
    points = (lines[:, :-1, None] + delta[:, :, None] * fractions[None, None, :, None]).reshape(-1, 3)
    features = np.concatenate((np.ones((len(points), 1)), np.repeat(tangent.reshape(-1, 3), 5, axis=0)), axis=1)
    coordinates = np.clip((points + 1) * ((resolution - 1) / 2), 0, resolution - 1)
    lower = np.floor(coordinates).astype(np.int64)
    fraction = coordinates - lower
    grid = np.zeros((4, resolution, resolution, resolution), dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.array([dx, dy, dz])
                index = np.minimum(lower + offset, resolution - 1)
                weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=-1)
                for channel in range(4):
                    np.add.at(grid[channel], (index[:, 2], index[:, 1], index[:, 0]), features[:, channel] * weight)
    mass = grid[0].copy()
    grid[1:] /= np.maximum(mass[None], 1e-12)
    grid[0] = np.minimum(mass, 1)
    return grid


def preflight(spec, resolution, device):
    """Synthetic geometry only: operator, independent splat and batch-memory checks."""
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    torch.manual_seed(422)
    t = torch.linspace(-.7, .7, 32, device=device)
    offsets = torch.linspace(-.15, .15, 27, device=device)[:, None]
    geometry = torch.stack((t.expand(27, -1), .35*torch.sin(3*t)+offsets,
                            .25*torch.cos(4*t)+offsets/2), dim=-1)[None]
    counts = torch.tensor([19], device=device)
    voxel = bundle_voxels(geometry, counts, resolution=resolution, subdivisions=4)
    reference = reference_voxel(geometry[0].cpu().numpy(), 19, resolution)
    difference = np.abs(voxel[0].cpu().numpy() - reference)
    assert np.allclose(voxel[0].cpu().numpy(), reference, atol=3e-5, rtol=2e-4)
    model = capacity.BudgetConv3D(.15).to(device)
    batch = spec['training']['batch_size'] if device.type == 'cuda' else 2
    x = voxel.expand(batch, -1, -1, -1, -1).contiguous()
    y = torch.arange(batch, device=device) % 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        assert logits.shape == (batch, 2) and torch.isfinite(logits).all()
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
    if device.type == 'cuda': torch.cuda.synchronize()
    report = dict(status='PASS', resolution=resolution, batch_size=batch, **capacity.parameter_check(spec),
                  numpy_voxel_max_error=float(difference.max()), three_training_steps_seconds=time.perf_counter()-start,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                  all_40000_voxels_gib=40000*4*resolution**3*2/2**30,
                  peak_train_input_copy_gib=2*27000*4*resolution**3*2/2**30,
                  synthetic_only=True, device=str(device))
    del optimizer, model, x, voxel
    if device.type == 'cuda': torch.cuda.empty_cache()
    return report


def prepare(spec, config, index):
    require_data_confirmation(spec)
    capacity.parent_check(spec)
    resolution = spec['voxel_resolutions'][index]
    local = resolution_spec(spec, resolution)
    root = Path(local['output']); root.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    assert torch.cuda.get_device_name(0) == spec['required_device']
    check = preflight(spec, resolution, device)
    pipeline.write(root/'preflight.json', dict(identity=identity(config), **check))
    start = time.perf_counter(); source = Path(spec['parent_output'])
    old = json.loads((source/'encoding.json').read_text())
    assert old['identity']['config_sha256'] == spec['parent_config_sha256']
    manifest = dict(version=spec['version'], identity=identity(config), resolution=resolution, splits={},
                    subdivisions=4, regenerated_from_frozen_geometry=True, interpolated_old_voxels=False)
    for flow in spec['flows']:
        physical = source/'physical'/flow['name']
        prepared = json.loads((physical/'preparation.json').read_text())
        assert prepared['identity']['config_sha256'] == spec['parent_config_sha256']
        for split in capacity.SPLITS:
            key = flow['name']+'/'+split; src = physical/split; dst = root/key
            dst.mkdir(parents=True, exist_ok=False)
            for name, digest in prepared['splits'][split]['files'].items():
                assert pipeline.sha(src/name) == digest, str(src/name)
            assert pipeline.sha(src/'metadata.npz') == old['splits'][key]['files']['metadata.npz']
            with np.load(src/'metadata.npz') as z:
                counts = z['counts'].astype(np.int64); labels = z['labels']
            assert np.all((counts >= 10) & (counts <= 27))
            n = len(labels); assert n == spec['sampling']['per_flow_counts'][split]
            geometry = np.load(src/'geometry.npy', mmap_mode='r')
            assert geometry.shape == (n, 27, 32, 3)
            voxels = np.lib.format.open_memmap(dst/'voxels.npy', mode='w+', dtype='float16',
                                              shape=(n, 4, resolution, resolution, resolution))
            for first in range(0, n, spec['encoding_batch_size']):
                sl = slice(first, first + spec['encoding_batch_size'])
                g = torch.as_tensor(np.array(geometry[sl]), device=device)
                count = torch.as_tensor(counts[sl], device=device)
                value = bundle_voxels(g, count, resolution=resolution, subdivisions=4)
                assert torch.isfinite(value).all() and value[:, :1].min() >= 0 and value.abs().max() <= 1.00001
                voxels[sl] = value.cpu().numpy().astype(np.float16)
            voxels.flush(); del voxels
            (dst/'metadata.npz').symlink_to((src/'metadata.npz').resolve())
            manifest['splits'][key] = dict(samples=n, class_counts=np.bincount(labels, minlength=2).tolist(),
                voxel_shape=[n, 4, resolution, resolution, resolution], voxel_dtype='float16',
                physical_files=prepared['splits'][split]['files'],
                files={name:pipeline.sha(dst/name) for name in ('voxels.npy','metadata.npz')})
            print(json.dumps(dict(resolution=resolution, split=key, samples=n)), flush=True)
    manifest['encoding_seconds'] = time.perf_counter()-start
    pipeline.write(root/'encoding.json', manifest)
    pipeline.write(root/'input_and_capacity_checks.json', dict(identity=identity(config), status='PASS',
        resolution=resolution, encoding_sha256=pipeline.sha(root/'encoding.json'),
        parent_geometry_and_metadata_hashes_checked=True, **capacity.parameter_check(spec)))


def train(spec, config, index):
    require_data_confirmation(spec)
    count = len(training.plan(spec))
    resolution = spec['voxel_resolutions'][index // count]
    local = resolution_spec(spec, resolution)
    root = Path(local['output'])
    check = json.loads((root/'input_and_capacity_checks.json').read_text())
    assert check['resolution'] == resolution and check['encoding_sha256'] == pipeline.sha(root/'encoding.json')
    manifest = json.loads((root/'encoding.json').read_text())
    assert manifest['resolution'] == resolution
    current = identity(config)
    for key in ('commit', 'config_sha256', 'source_sha256'):
        assert manifest['identity'][key] == current[key]
    # Validate declared shapes without opening test arrays before selection.lock.
    # The frozen loader checks each actual file hash when its split is loaded.
    for entry in manifest['splits'].values():
        assert entry['voxel_dtype'] == 'float16'
        assert entry['voxel_shape'] == [entry['samples'], 4, resolution, resolution, resolution]
    with capacity_identity(): capacity.train(local, config, index % count)
    method, candidate, seed = training.plan(local)[index % count]
    path = training.run_folder(local, method, candidate, seed)/'result.json'
    result = json.loads(path.read_text())
    result.update(voxel_resolution=resolution, geometry_protocol='frozen_4.14_geometry_revoxelized',
                  encoding_sha256=pipeline.sha(root/'encoding.json'))
    pipeline.write(path, result)


def audit_inputs(spec, resolution):
    local = resolution_spec(spec, resolution); root = Path(local['output'])
    manifest = json.loads((root/'encoding.json').read_text())
    checks = []
    for flow in spec['flows']:
        for split in capacity.SPLITS:
            key = flow['name']+'/'+split; entry = manifest['splits'][key]
            physical = Path(spec['parent_output'])/'physical'/key
            for name, digest in entry['files'].items(): assert pipeline.sha(root/key/name) == digest
            for name, digest in entry['physical_files'].items(): assert pipeline.sha(physical/name) == digest
            geometry = np.load(physical/'geometry.npy', mmap_mode='r')
            voxels = np.load(root/key/'voxels.npy', mmap_mode='r')
            assert list(voxels.shape) == entry['voxel_shape'] and voxels.dtype == np.float16
            with np.load(physical/'metadata.npz') as z: counts = z['counts']
            for row in (0, len(geometry)//2, len(geometry)-1):
                expected = reference_voxel(geometry[row], int(counts[row]), resolution)
                actual = np.asarray(voxels[row], dtype=np.float32)
                # Float16 rounding plus floating-point trilinear accumulation.
                assert np.allclose(actual, expected, atol=6e-4, rtol=1e-3), (key, row)
                checks.append(dict(split=key, row=row, maximum_error=float(np.abs(actual-expected).max())))
    return checks


def merge(spec, config):
    require_data_confirmation(spec)
    rows = []; audits = []
    for resolution in spec['voxel_resolutions']:
        local = resolution_spec(spec, resolution)
        voxel_checks = audit_inputs(spec, resolution)
        for method, candidate, seed in training.plan(local):
            result = json.loads((training.run_folder(local, method, candidate, seed)/'result.json').read_text())
            assert result['voxel_resolution'] == resolution
            assert result['encoding_sha256'] == pipeline.sha(Path(local['output'])/'encoding.json')
        with capacity_identity(): capacity.merge(local, config)
        path = Path(local['output'])/'summary.json'; summary = json.loads(path.read_text())
        row = dict(summary['rows'][0], resolution=resolution, timing=summary['timing'])
        row['training_minutes_mean'] = float(np.mean([r['training_through_last_validation_seconds'] for r in row['timing']])/60)
        row['epoch_seconds_median_mean'] = float(np.mean([r['epoch_seconds_after_first_median'] for r in row['timing']]))
        row['encoding_seconds'] = json.loads((path.parent/'encoding.json').read_text())['encoding_seconds']
        rows.append(row)
        audits.append(dict(resolution=resolution, summary_sha256=pipeline.sha(path),
                           predictions_audit_sha256=pipeline.sha(path.parent/'independent_predictions_audit.json'),
                           voxel_checks=voxel_checks))
    root = Path(spec['output'])
    summary = dict(version=spec['version'], identity=identity(config), rows=rows, reference_24=spec['reference_24'],
        comparison_scope=spec['comparison_scope'], selection_policy=spec['selection_policy'],
        parameter_check=capacity.parameter_check(spec), completed_at_utc=datetime.now(timezone.utc).isoformat())
    pipeline.write(root/'summary.json', summary)
    pipeline.write(root/'independent_audit.json', dict(status='PASS', identity=identity(config), resolutions=audits,
        summary_sha256=pipeline.sha(root/'summary.json'), user_corrected_24_reference_not_reaudited=True, weights_written=0))


def runtime(spec, config, phase, state, code=None):
    row = dict(time=datetime.now(timezone.utc).isoformat(), phase=phase, state=state, exit_code=code,
               device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', **identity(config))
    pipeline.append_locked(Path(spec['output'])/'runtime_events.jsonl', json.dumps(row)+'\n')
    pipeline.append_locked('docs/ibex_run_registry.md', '\n- Task4-c 4.22 runtime '+json.dumps(row)+'\n')


def submit(spec, config):
    require_data_confirmation(spec)
    root = Path(spec['output']).resolve(); root.mkdir(parents=True, exist_ok=False); (root/'logs').mkdir()
    pipeline.write(root/'config.frozen.json', spec); previous = None
    for phase in ('prepare', 'train', 'merge'):
        command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=64G',
                   '--time=10:00:00' if phase == 'train' else '--time=02:00:00', '--job-name=t4c422-'+phase,
                   f'--output={root}/logs/{phase}_%A_%a.out', f'--error={root}/logs/{phase}_%A_%a.err']
        if previous: command += ['--dependency=afterok:'+previous, '--kill-on-invalid-dep=yes']
        if phase in ('prepare', 'train'):
            command += ['--gres=gpu:1', '--constraint=v100', '--array=0-2%3' if phase == 'prepare' else '--array=0-8%9']
        command += ['ibex_bash/task4c_conv_resolution_4p22.sh', phase, config]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        row = dict(job_id=job, phase=phase, version=spec['version'], command=command, dependency=previous,
                   submitted_at_utc=datetime.now(timezone.utc).isoformat(),
                   expected_device=spec['required_device'] if phase != 'merge' else 'CPU', **identity(config))
        pipeline.append_locked(root/'submissions.jsonl', json.dumps(row)+'\n')
        pipeline.append_locked('docs/ibex_run_registry.md', '\n- Task4-c 4.22 SUBMITTED '+json.dumps(row)+'\n')
        print(json.dumps(row), flush=True); previous = job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'train', 'merge', 'submit', 'runtime', 'preflight'))
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--runtime-phase'); parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec = load_spec(args.config)
    if args.phase in ('prepare', 'train'): globals()[args.phase](spec, args.config, args.index)
    elif args.phase == 'runtime': runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    elif args.phase == 'preflight':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        reports = [preflight(spec, r, device) for r in spec['voxel_resolutions']]
        path = Path('outputs/Verify_Task4C_ConvResolutionCode_4.22/checks.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.write(path, dict(status='PASS', identity=identity(args.config), checks=reports))
        print(json.dumps(reports), flush=True)
    else: globals()[args.phase](spec, args.config)


if __name__ == '__main__': main()
