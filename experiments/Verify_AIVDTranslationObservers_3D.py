"""Apply the unchanged scalar AIVD feature to seven observed-field integrations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    root = Path(spec['source_root'])
    frozen_source = root / spec['classifier_source']
    # Freeze the scientific dependencies to the already-run transfer snapshot.
    sys.path.insert(0, str(frozen_source))
    import torch
    from experiments.Verify_AIVDTransfer_3D import records, features, labels
    from FMT_Utils.Task12Data_3D import feature_matrix, _anchored_recipe
    from FMT_Utils.DFT_FMT_3D import pathline_anchored_kinematic_dft_features_3d
    from FMT_Utils.Task12Evaluation_3D import (
        fit_kmeans_transform, calibrate_vortex_cluster, binary_cluster_metrics,
    )
    from FMT_Utils.ObservedFieldWorldline_3D import (
        TranslatingObserverField, ReferenceFrameFromWorldline,
    )
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    device = torch.device('cpu')
    output = root / spec['output_root']
    output.mkdir(parents=True, exist_ok=False)
    started = dict(experiment=spec['experiment'], base_commit=spec['base_commit'],
                   config_sha256=sha(args.config), source_sha256=sha(__file__),
                   encoder_sha256=sha(Path(sys.modules['FMT_Utils.DFT_FMT_3D'].__file__)),
                   source_manifest_sha256=sha(frozen_source/'SOURCE_MANIFEST.sha256'),
                   job_id=os.environ.get('SLURM_JOB_ID'), node=socket.gethostname(),
                   started=time.time(), device='CPU', torch=torch.__version__)
    write_json(output/'started.json', started)
    transfer = json.loads((frozen_source/spec['classifier_config']).read_text())
    train = records(transfer, 'Task1', spec['dataset'], 'train')
    validation = records(transfer, 'Task1', spec['dataset'], 'validation')
    x = features(train, spec['feature'], device)
    val = features(validation, spec['feature'], device)
    model = fit_kmeans_transform(x, None, spec['classifier_seed'], 20)
    vortex = calibrate_vortex_cluster(labels(validation), model.predict(val))
    calibration = binary_cluster_metrics(labels(validation), model.predict(val), vortex)
    prior_model_path = (Path(transfer['output_root'])/'shards/Task1'/spec['dataset']/
                        f"seed{spec['classifier_seed']}"/'frozen_models.json')
    prior_model = json.loads(prior_model_path.read_text())['aivd']
    assert vortex == prior_model['vortex_cluster']
    for key, value in calibration.items():
        assert abs(value-prior_model['calibration'][key]) < 1e-12, key
    model_info = {
        'feature': spec['feature'], 'seed': spec['classifier_seed'], 'vortex_cluster': vortex,
        'calibration_matches_registered_transfer': True, 'pca_dimension': None,
        'scaler_mean': model.scaler.mean_.tolist(), 'scaler_scale': model.scaler.scale_.tolist(),
        'cluster_centres_standardized': model.model.cluster_centers_.tolist(),
        'training_slices': [r['metadata']['source_time'] for r in train],
        'calibration_slices': [r['metadata']['source_time'] for r in validation],
        'training_cache_sha256': [sha(r['path']) for r in train],
        'calibration_cache_sha256': [sha(r['path']) for r in validation],
        'frozen_before_display_loaded': time.time(),
    }
    print('Classifier frozen; calibration reproduces registered scalar run.', flush=True)

    source = root/spec['cohort_source']
    old_audit = json.loads((source/'audit.json').read_text())
    with np.load(source/'observed_pathlines.npz') as archive:
        paths = archive['pathlines']
        times = archive['sample_times']
    assert paths.shape == (7, spec['cohort_count'], 7, spec['display_steps']+1, 3)
    assert np.isfinite(paths).all()
    assert times[0] == spec['initial_time'] and times[0] >= spec['minimum_time']
    indices = np.asarray(old_audit['classification_sample_indices'])
    assert np.array_equal(indices, np.linspace(0, 48, 32).round().astype(int))
    np.testing.assert_allclose(old_audit['levels'], spec['levels'], rtol=0, atol=1e-15)
    assert max(old_audit['trajectory_correspondence_max_error']) < old_audit['correspondence_tolerance']

    def encode(coordinates, dtype):
        xyz = np.asarray(coordinates, dtype=dtype)
        # Fixed initial origin; not time-dependent recentering of paths.
        local = xyz - xyz[:, :1, :1]
        if dtype == np.float32:
            return feature_matrix({'raw': local.reshape(len(local), -1), 'features': {}},
                                  spec['feature'], device)
        return pathline_anchored_kinematic_dft_features_3d(
            torch.from_numpy(local), **_anchored_recipe(spec['feature']))

    def separation(xyz):
        return np.stack((xyz[:, 1]-xyz[:, 2], xyz[:, 3]-xyz[:, 4],
                         xyz[:, 5]-xyz[:, 6]), axis=-1)

    all_features, all_labels, details = [], [], []
    exact_all_features, double_all_features = [], []
    # Basic slicing preserves [N,7,L,3] regardless of NumPy advanced indexing rules.
    original = paths[0][:, :, indices]
    base_d = separation(original)
    base = encode(original, np.float32)
    base64 = encode(original, np.float64)
    base_labels = model.predict(base) == vortex
    base64_labels = model.predict(base64) == vortex
    for level, alpha in enumerate(spec['levels']):
        observer = TranslatingObserverField(old_audit['observer_velocity_times'],
                    alpha*np.asarray(old_audit['observer_velocity_samples']))
        frame = ReferenceFrameFromWorldline(observer, times[0])
        displacement = np.stack([frame.camera_position(t)-frame.start for t in times])
        exact = paths[0] - displacement[None, None]
        error = float(np.max(np.linalg.norm(paths[level]-exact, axis=-1)))
        assert error < old_audit['correspondence_tolerance']
        current = paths[level][:, :, indices]
        actual = encode(current, np.float32)
        exact_feature = encode(exact[:, :, indices], np.float32)
        double_feature = encode(current, np.float64)
        exact_double = encode(exact[:, :, indices], np.float64)
        prediction = model.predict(actual) == vortex
        exact_prediction = model.predict(exact_feature) == vortex
        distances = model.model.transform(model.transform(actual))
        row = {
            'alpha': alpha, 'vortex_count': int(prediction.sum()),
            'changed_labels': int(np.count_nonzero(prediction != base_labels)),
            'changed_fraction': float(np.mean(prediction != base_labels)),
            'feature_max_absolute_change': float(np.max(np.abs(actual-base))),
            'feature_relative_l2_change': float(np.linalg.norm(actual-base)/np.linalg.norm(base)),
            'exact_float32_changed_labels': int(np.count_nonzero(exact_prediction != base_labels)),
            'exact_float32_max_absolute_change': float(np.max(np.abs(exact_feature-base))),
            'integration_vs_exact_changed_labels': int(np.count_nonzero(prediction != exact_prediction)),
            'float64_max_absolute_change': float(np.max(np.abs(double_feature-base64))),
            'float64_changed_labels': int(np.count_nonzero((model.predict(double_feature)==vortex) != base64_labels)),
            'exact_float64_max_absolute_change': float(np.max(np.abs(exact_double-base64))),
            'trajectory_correspondence_max_error': error,
            'same_time_separation_max_error': float(np.max(np.abs(separation(current)-base_d))),
            'minimum_cluster_distance_margin': float(np.min(np.abs(distances[:, 0]-distances[:, 1]))),
            'camera_displacement_x': float(displacement[-1, 0]),
            'camera_displacement_y': float(displacement[-1, 1]),
            'camera_displacement_z': float(displacement[-1, 2]),
        }
        print(json.dumps(row), flush=True)
        details.append(row); all_features.append(actual); all_labels.append(prediction)
        exact_all_features.append(exact_feature); double_all_features.append(double_feature)
    predictions = np.stack(all_labels)
    audit = {
        **started, 'completed_computation': time.time(), 'classifier': model_info,
        'input_conversion': spec['input_conversion'], 'mean_context': spec['mean_context'],
        'source_paths_sha256': sha(source/'observed_pathlines.npz'),
        'source_audit_sha256': sha(source/'audit.json'),
        'trajectory_reuse': spec['trajectory_policy'], 'levels': spec['levels'],
        'cohort_count': len(original), 'initial_time': float(times[0]),
        'sample_times': times.tolist(), 'classification_sample_indices': indices.tolist(),
        'observer_velocity_times': old_audit['observer_velocity_times'],
        'observer_velocity_samples': old_audit['observer_velocity_samples'],
        'changed_labels_vs_original': [r['changed_labels'] for r in details],
        'vortex_counts': [r['vortex_count'] for r in details], 'details': details,
        'float32_vs_float64_original_labels': int(np.count_nonzero(base_labels != base64_labels)),
        'feature_original_min': float(base.min()), 'feature_original_max': float(base.max()),
        'scope': 'translation-only numerical verification; does not establish time-dependent rotation objectivity',
        'checkpoints_written': 0,
    }
    write_json(output/'summary.json', audit)
    with (output/'observer_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    # Keep independent prediction evidence on the compute host; not a model checkpoint.
    np.savez_compressed(output/'feature_evidence.npz', feature=np.stack(all_features),
                        predictions=predictions, exact_float32=np.stack(exact_all_features),
                        float64=np.stack(double_all_features))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from Plot_AIVDTranslationObservers_3D import render
    for medium in ['paper', 'slides']:
        render(paths, predictions, np.arange(len(original)), audit, output/'final',
               'Re160 3D | aivd1w3_dft | t0 = 10.5', medium, pdf_collision_audit=False)
    write_json(output/'complete.json', {**started, 'ended': time.time(), 'status': 'COMPLETE'})


if __name__ == '__main__':
    main()
