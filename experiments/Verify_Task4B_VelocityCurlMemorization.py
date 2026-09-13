"""Build, fit and audit the versioned Task4-b velocity-curl memorization task."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch
import yaml
from scipy.interpolate import RegularGridInterpolator

from FMT_Utils.Task4A_StreamlineClustering_3D import ChannelVelocityField3D, integrate_bidirectional_cross_primitives
from FMT_Utils.Task4B_CrossFlow_3D import NonPeriodicStructuredVelocityField3D, finite_difference_curl_zyx, dimensionless_primitive_features
from FMT_Utils.Task4B_ProxyLabels_3D import pad_periodic_xy
from FMT_Utils.Task4B_VelocityCurlLabels_3D import CLASS_NAMES, coverage_threshold, velocity_curl_labels, spatial_mean
from FMT_Utils.VoxelSegmentation_3D import (
    VoxelGrid3D, _read_vtk_dataset, _sample_cell_labels, _copy_selected_array,
    _select_label_array, _label_array_descriptions, suggest_native_resolution, cap_resolution_by_voxels,
)
from experiments.Train_Task4B_FourClassClassifier_1_1 import _normalize_train_only
from experiments.Verify_Task4B_PooledMemorization_3_1 import _train_one, _fit_metrics


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class DimensionlessField:
    def __init__(self, field, origin, length, speed):
        self.field, self.origin, self.length, self.speed = field, origin, length, speed

    def velocity(self, points):
        return self.field.velocity(np.asarray(points) * self.length + self.origin) / self.speed


def load_field(path, name):
    if name == 'channel':
        old, metadata = ChannelVelocityField3D.from_vtk(path)
        axes = (old.axes_zyx[0], old.axes_zyx[1][:-1], old.axes_zyx[2][:-1])
        velocity = old.velocity_zyx3[:, :-1, :-1]
        curl = finite_difference_curl_zyx(velocity, axes)
        zs, ys, xs = axes
        dx, dy = np.diff(xs).mean(), np.diff(ys).mean()
        ddx = lambda v: (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx)
        ddy = lambda v: (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
        curl[..., 0] = ddy(velocity[..., 2]) - np.gradient(velocity[..., 1], zs, axis=0, edge_order=2)
        curl[..., 1] = np.gradient(velocity[..., 0], zs, axis=0, edge_order=2) - ddx(velocity[..., 2])
        curl[..., 2] = ddx(velocity[..., 1]) - ddy(velocity[..., 0])
        reference = old.vorticity_zyx3[:, :-1, :-1]
        metadata['recomputed_curl_component_correlations'] = [float(np.corrcoef(curl[..., i].ravel()[::17], reference[..., i].ravel()[::17])[0,1]) for i in range(3)]
        field = ChannelVelocityField3D(old.axes_zyx, old.velocity_zyx3, old.wall_bounds_z, old.periods_xy, pad_periodic_xy(curl))
    else:
        field, metadata = NonPeriodicStructuredVelocityField3D.from_vtk(path)
        axes, velocity, curl = field.axes_zyx, field.velocity_zyx3, field.vorticity_zyx3
    mean = spatial_mean(curl, axes, periodic_xy=name == 'channel')
    speed_rms = float(np.sqrt(np.mean(np.sum(velocity.astype(np.float64) ** 2, axis=-1))))
    metadata.update(global_volume_mean_curl=mean.tolist(), velocity_rms=speed_rms,
                    curl_source='finite_difference_of_velocity_periodic_xy_for_channel')
    return field, metadata, mean, speed_rms


def build(spec, config_path, input_root):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    output = Path(spec['output_dir'])
    if (output / 'cache.npz').exists():
        raise RuntimeError('Frozen cache already exists; do not overwrite it')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config_snapshot.yaml').write_bytes(Path(config_path).read_bytes())
    arrays, reports = [], {}
    for volume_code, (name, item) in enumerate(spec['flows'].items()):
        flow_path, gt_path = Path(input_root) / item['flow'], Path(input_root) / item['gt']
        print(f'Loading {name}: {flow_path}', flush=True)
        field, metadata, mean, speed_rms = load_field(flow_path, name)
        dataset = _read_vtk_dataset(gt_path)
        selected = _select_label_array(_label_array_descriptions(dataset), 'VortexIds', 'cell')
        working, source_labels = _copy_selected_array(dataset, selected)
        source_ids = np.asarray(vtk_to_numpy(source_labels), dtype=np.int32)
        centers_filter = vtk.vtkCellCenters()
        centers_filter.SetInputData(dataset)
        centers_filter.Update()
        gt_centers = np.asarray(vtk_to_numpy(centers_filter.GetOutput().GetPoints().GetData()), dtype=np.float64)[source_ids > 0]
        gt_ids = source_ids[source_ids > 0]
        # The regular target grid covers the full velocity domain, not just GT bounds.
        low = np.array([axis[0] for axis in field.axes_zyx[::-1]])
        high = np.array([axis[-1] for axis in field.axes_zyx[::-1]])
        gt_bounds = np.array(dataset.GetBounds()).reshape(3, 2)
        native = np.array(suggest_native_resolution(dataset, 'cell', max_axis_resolution=1024))
        expanded = np.maximum(1, np.ceil(native * (high-low) / np.diff(gt_bounds, axis=1).ravel()).astype(int))
        resolution = cap_resolution_by_voxels(tuple(expanded), spec['voxelization']['max_voxels'])
        grid = VoxelGrid3D(resolution, low, high)
        ids_grid, probe_valid = _sample_cell_labels(working, grid)
        ids_grid = np.where(probe_valid & (ids_grid > 0), ids_grid, 0).astype(np.int32)
        nz, ny, nx = grid.shape_zyx
        flat = np.arange(nx * ny * nz)
        iz, remainder = np.divmod(flat, ny * nx)
        iy, ix = np.divmod(remainder, nx)
        indices = np.column_stack((ix,iy,iz)).astype(np.int32)
        points = low + (indices + 0.5) * grid.voxel_size_xyz
        count = len(points)
        ivd = np.empty(count, np.float64)
        labels = np.empty(count, np.int8)
        cosines = np.empty(count, np.float64)
        ids = ids_grid.ravel()
        chunk = spec['streamlines']['chunk_size']
        for start in range(0, count, chunk):
            end = min(count, start + chunk)
            omega = field.vorticity(points[start:end])
            ivd[start:end] = np.linalg.norm(omega.astype(np.float64) - mean, axis=1)
            labels[start:end], cosines[start:end], _ = velocity_curl_labels(field.velocity(points[start:end]), omega, ids[start:end] > 0)
        gt_ivd = np.empty(len(gt_centers), np.float64)
        for start in range(0, len(gt_centers), chunk):
            end = min(len(gt_centers), start + chunk)
            gt_ivd[start:end] = np.linalg.norm(field.vorticity(gt_centers[start:end]).astype(np.float64) - mean, axis=1)
        thresholds = coverage_threshold(np.r_[gt_ivd, ivd[ids > 0]])
        vortex = ivd > thresholds['vortex_threshold']
        assert np.all(vortex[ids > 0]) and np.all(gt_ivd > thresholds['a'])
        if not np.isfinite(ivd).all():
            raise RuntimeError('Nonfinite IVD in the full target domain')
        labels[~vortex] = -1
        np.savez_compressed(output / f'{name}_label_volume.npz', labels=labels.reshape(grid.shape_zyx), vortex_ids=ids_grid,
                            ivd=ivd.reshape(grid.shape_zyx), abs_cosine=cosines.reshape(grid.shape_zyx),
                            domain_min_xyz=low, domain_max_xyz=high, resolution_xyz=resolution,
                            original_gt_cell_ivd=gt_ivd, original_gt_cell_ids=gt_ids)
        rng = np.random.default_rng(spec['sampling']['seed'] + volume_code)
        chosen = []
        for class_id in range(4):
            members = np.flatnonzero(labels == class_id)
            if class_id < 2 and len(members) > spec['sampling']['ordinary_per_class_per_volume']:
                members = rng.choice(members, spec['sampling']['ordinary_per_class_per_volume'], replace=False)
            chosen.append(members)
        chosen = np.sort(np.concatenate(chosen))
        step = float(np.mean(grid.voxel_size_xyz)) * spec['streamlines']['spatial_step_mean_voxel_scale']
        wrapper = DimensionlessField(field, low, step, speed_rms)
        chunks, valid_sources = [], []
        for start in range(0, len(chosen), chunk):
            source = chosen[start:start+chunk]
            primitive, valid = integrate_bidirectional_cross_primitives(
                wrapper, (points[source] - low) / step, spatial_step=1.,
                steps_per_direction=spec['streamlines']['steps_per_direction'],
                offset=spec['streamlines']['offset_over_step'],
                minimum_speed=spec['streamlines']['minimum_speed_relative_rms'], chunk_size=chunk)
            if valid.any():
                raw, fmt = dimensionless_primitive_features(primitive[valid], 1., **{k:spec['encoder'][k] for k in ('num_freq','neighbor_scale','neighbor_pool','mode','include_chirality')})
                chunks.append((raw,fmt))
                valid_sources.append(source[valid])
            print(f'{name}: integrated {min(start+chunk,len(chosen))}/{len(chosen)}', flush=True)
        sources = np.concatenate(valid_sources)
        raw, fmt = (np.concatenate([c[i] for c in chunks]) for i in (0,1))
        support = np.bincount(labels[sources], minlength=4)
        if np.any(support < spec['sampling']['minimum_per_class_per_volume']):
            raise RuntimeError(f'Insufficient support in {name}: {support}')
        arrays.append(dict(raw_features=raw, fmt_features=fmt, labels=labels[sources], volume_codes=np.full(len(sources),volume_code,np.int8),
                           source_candidate_indices=sources, voxel_indices_xyz=indices[sources], seeds_xyz=points[sources].astype(np.float32), vortex_ids=ids[sources]))
        reports[name] = dict(metadata=metadata, flow_path=str(flow_path), gt_path=str(gt_path), flow_sha256=sha(flow_path), gt_sha256=sha(gt_path),
                             thresholds=thresholds, full_grid_count=count, full_grid_vortex_fraction=float(vortex.mean()),
                             original_gt_cell_count=len(gt_ivd), original_gt_coverage=float(np.mean(gt_ivd > thresholds['a'])),
                             grid_hairpin_count=int((ids>0).sum()), grid_hairpin_coverage=float(vortex[ids>0].mean()),
                             candidate_class_support=np.bincount(labels[labels>=0],minlength=4).tolist(),
                             selected_class_support=np.bincount(labels[chosen],minlength=4).tolist(), valid_class_support=support.tolist(),
                             invalid_primitive_count=int(len(chosen)-len(sources)),
                             original_gt_ids=np.unique(gt_ids).tolist(), voxelized_gt_ids=np.unique(ids[ids>0]).tolist(),
                             valid_gt_ids=np.unique(ids[sources][ids[sources]>0]).tolist(),
                             spatial_step=step, offset=step*spec['streamlines']['offset_over_step'],
                             label_volume_sha256=sha(output / f'{name}_label_volume.npz'))
        write_json(output / f'{name}_build_report.json', reports[name])
        del field, working, dataset, chunks, raw, fmt, primitive
    combined = {key:np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    np.savez_compressed(output / 'cache.npz', **combined)
    report = dict(experiment=spec['experiment'], classes=list(CLASS_NAMES), config_sha256=sha(config_path),
                  cache_sha256=sha(output/'cache.npz'), sample_count=len(combined['labels']),
                  class_support=np.bincount(combined['labels'],minlength=4).tolist(), flows=reports,
                  hostname=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(output / 'build_summary.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='flows'}, indent=2), flush=True)


def train(spec, config_path):
    output = Path(spec['output_dir'])
    build_report = json.loads((output/'build_summary.json').read_text())
    if sha(config_path) != build_report['config_sha256'] or sha(output/'cache.npz') != build_report['cache_sha256']:
        raise RuntimeError('Build identity mismatch')
    if (output/'fit_summary.json').exists():
        raise RuntimeError('Fit already exists; use a new version for changed training')
    if not torch.cuda.is_available():
        raise RuntimeError('This production fit requires an allocated GPU')
    with np.load(output/'cache.npz') as cache:
        data = {k:cache[k] for k in cache.files}
    spec = copy.deepcopy(spec)
    spec['pass_gate'].update(expected_sample_count=len(data['labels']), expected_class_support=np.bincount(data['labels'],minlength=4).tolist())
    codes = np.zeros(len(data['labels']), np.int8)
    raw, fmt, normalization = _normalize_train_only(data['raw_features'], data['fmt_features'], codes,
        sampled_steps=2*spec['streamlines']['steps_per_direction']+1, encoder_spec=spec['encoder'])
    np.savez_compressed(output/'normalization_all_fit_rows.npz', **normalization)
    identity = dict(config_sha256=sha(config_path), cache_sha256=build_report['cache_sha256'],
                    deployment_sha256=sha('DEPLOYMENT_MANIFEST_4p1.sha256'), base_commit=Path('DEPLOYMENT_BASE_COMMIT.txt').read_text().strip(),
                    build_summary_sha256=sha(output/'build_summary.json'))
    # Reuse the existing tested epoch loop without changing the frozen older entry point.
    result = _train_one(spec, raw, fmt, data['labels'].astype(np.int64), codes, data['vortex_ids'],
        data['source_candidate_indices'], data['voxel_indices_xyz'], data['seeds_xyz'], identity,
        variant='fmt_only', seed=spec['training']['seed'], device=torch.device('cuda'), output_dir=output)
    predictions = output/'predictions'/f"fmt_only_seed{spec['training']['seed']}.npz"
    with np.load(predictions) as p:
        per_volume = {name:_fit_metrics(data['labels'][data['volume_codes']==i],p['logits'][data['volume_codes']==i]) for i,name in enumerate(spec['flows'])}
    write_json(output/'fit_summary.json', dict(experiment=spec['experiment'], classes=list(CLASS_NAMES), pooled=result, per_volume=per_volume,
        hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(), job_id=os.environ.get('SLURM_JOB_ID'), prediction_sha256=sha(predictions)))


def audit(spec, config_path):
    from sklearn.metrics import confusion_matrix, f1_score
    output = Path(spec['output_dir'])
    report = json.loads((output/'build_summary.json').read_text())
    fit = json.loads((output/'fit_summary.json').read_text())
    checks = {}
    checks['config_hash'] = sha(config_path) == report['config_sha256']
    checks['cache_hash'] = sha(output/'cache.npz') == report['cache_sha256']
    with np.load(output/'cache.npz') as c:
        y, volume = c['labels'], c['volume_codes']
        keys = np.column_stack((volume,c['source_candidate_indices']))
        checks['unique_volume_voxel_pairs'] = len(np.unique(keys,axis=0)) == len(y)
        for i,name in enumerate(spec['flows']):
            r = report['flows'][name]
            with np.load(output/f'{name}_label_volume.npz') as v:
                ivd, ids, cosine, labels = v['ivd'], v['vortex_ids'], v['abs_cosine'], v['labels']
                # Independent dot-angle decision expressed in angle space.
                angle = np.degrees(np.arccos(np.clip(cosine,0,1)))
                parallel = angle <= 45.0 + 1e-12
                expected = np.where(ids>0,np.where(parallel,3,2),np.where(parallel,0,1))
                minimum = min(float(ivd[ids>0].min()),float(v['original_gt_cell_ivd'].min()))
                checks[f'{name}_threshold'] = r['thresholds']['a'] == float(np.nextafter(minimum,-np.inf))
                valid = np.isfinite(cosine) & (ivd > r['thresholds']['vortex_threshold'])
                expected[~valid] = -1
                checks[f'{name}_labels'] = bool(np.array_equal(expected, labels))
                checks[f'{name}_coverage'] = bool(np.all(ivd[ids>0]>r['thresholds']['a']) and np.all(v['original_gt_cell_ivd']>r['thresholds']['a']))
                rows = volume == i
                checks[f'{name}_cache_targets'] = bool(np.array_equal(y[rows],labels.ravel()[c['source_candidate_indices'][rows]]))
                checks[f'{name}_label_hash'] = sha(output/f'{name}_label_volume.npz') == r['label_volume_sha256']
        _, expected_fmt, norm = _normalize_train_only(c['raw_features'],c['fmt_features'],np.zeros(len(y),np.int8),sampled_steps=33,encoder_spec=spec['encoder'])
        with np.load(output/'normalization_all_fit_rows.npz') as n:
            checks['normalization'] = all(np.array_equal(n[k],norm[k]) for k in n.files)
    pred_path = output/'predictions'/f"fmt_only_seed{spec['training']['seed']}.npz"
    with np.load(pred_path) as p:
        checks['prediction_hash'] = sha(pred_path) == fit['prediction_sha256']
        checks['prediction_targets'] = bool(np.array_equal(y,p['targets']))
        pred = p['logits'].argmax(1)
        checks['reported_predictions'] = bool(np.array_equal(pred,p['predicted_labels']))
        independent = {}
        for name,mask in [('pooled',np.ones(len(y),bool))] + [(name,volume==i) for i,name in enumerate(spec['flows'])]:
            values = dict(count=int(mask.sum()),errors=int(np.sum(pred[mask]!=y[mask])),accuracy=float(np.mean(pred[mask]==y[mask])),
                          macro_f1=float(f1_score(y[mask],pred[mask],labels=list(range(4)),average='macro',zero_division=0)),
                          confusion_matrix=confusion_matrix(y[mask],pred[mask],labels=list(range(4))).tolist())
            independent[name] = values
            observed = fit['pooled'] if name=='pooled' else fit['per_volume'][name]
            checks[f'{name}_metrics'] = values['errors']==observed['error_count'] and abs(values['macro_f1']-observed['macro_f1'])<1e-12
        wrong = p['logits'].copy()
        wrong[np.arange(len(y)), y] = -np.inf
        margin = p['logits'][np.arange(len(y)),y] - wrong.max(1)
        exact = bool(np.all(pred==y) and np.min(margin)>0)
    history = np.genfromtxt(output/'histories'/f"fmt_only_seed{spec['training']['seed']}.csv",delimiter=',',names=True)
    history = np.atleast_1d(history)
    terminal = history[-spec['pass_gate']['required_consecutive_zero_error_epochs']:]
    streak = len(terminal)==3 and np.all(terminal['fit_error_count']==0) and np.all(terminal['minimum_true_logit_margin']>0)
    checks['epoch_visits'] = bool(np.all(history['unique_count']==len(y)) and np.all(history['missing_count']==0) and np.all(history['duplicate_count']==0))
    checks['no_checkpoint'] = not any(p.suffix in {'.pt','.pth','.ckpt'} for p in output.rglob('*'))
    checks['reported_pass'] = bool(fit['pooled']['passed']) == bool(exact and streak)
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', memorization_pass=bool(exact and streak),checks=checks,independent_metrics=independent)
    write_json(output/'independent_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='PASS' or not result['memorization_pass']:
        raise RuntimeError('Audit or memorization gate failed; preserve evidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['build','train','audit'])
    parser.add_argument('--config',default='config/Verify_Task4B_VelocityCurlMemorization_4.1.yaml')
    parser.add_argument('--input-root')
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    torch.set_num_threads(min(8,int(os.environ.get('SLURM_CPUS_PER_TASK','8'))))
    if args.phase=='build':
        build(spec,args.config,args.input_root or spec['input_root_local'])
    elif args.phase=='train':
        train(spec,args.config)
    else:
        audit(spec,args.config)
