"""Apply the final 10h rule to saved preview cores, preserving upstream results."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import vtk
from scipy.spatial import cKDTree
from FMT_Utils.Task6VortexCore_3D import polylines
from FMT_Utils.Task6CoreLength_3D import mean_grid_spacing, filter_core_lengths
from experiments.Extract_Task6_VortexCore_1_2 import write_lines

BASE = ROOT / 'outputs/Verify_Task6_VortexCore_1.4'
VIEWER = ROOT / 'outputs/Verify_Task6_VortexCore_1.2/viewer'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_cores(path):
    if not path.exists():
        raise FileNotFoundError(path)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return polylines(reader.GetOutput())


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    backup = BASE / 'previous_viewer_1_4'
    if not backup.exists():
        backup.mkdir()
        for name in ['manifest.json', 'index.html']:
            shutil.copy2(VIEWER / name, backup / name)
    manifest = read_json(backup / 'manifest.json')
    config = read_json(ROOT / 'config/Verify_Task6_VortexCore_1.4.json')
    audit_frames = []
    for row in manifest['frames']:
        for flow, item in row['flows'].items():
            index = row['index']
            parent = ROOT / 'outputs' / config['parents'][flow]
            source = parent / flow / f'frame_{index:03d}'
            target = BASE / flow / f'frame_{index:03d}'
            target.mkdir(parents=True, exist_ok=True)
            payload = read_json(VIEWER / item['file'])
            with np.load(source / 'coordinates.npz') as data:
                axes = [data[a] for a in 'xyz']
            spacing = mean_grid_spacing(axes)
            h = float(spacing.min())
            geometry = {'grid_counts_xyz': [len(a) for a in axes],
                        'bounds_xyz': [[float(a.min()), float(a.max())] for a in axes],
                        'mean_spacing_xyz': spacing.tolist(), 'h': h,
                        'minimum_length_h': 10, 'minimum_length': 10 * h,
                        'keep_comparison': '>=', 'stage': 'after IVD and winding fragmentation'}
            # Historical snapshots with no accepted core have no sampling method.
            selected = payload['sampling'].get('selected_method')
            if selected is None:
                selected = payload['cleaning'].get('selected_key', payload['cleaning'].get('selected_method', 0))
            selected = str(selected if selected is not None else 0)
            method_audits = []
            maps = {}
            retained_by_method = {}
            source_hashes = {}
            for report in payload['cleaning']['methods']:
                key = report['key']
                path = source / f'vtk_higher{key}_clean.vtp'
                original = read_cores(path)
                source_hashes[str(path.relative_to(ROOT))] = sha256(path)
                retained, lengths, keep = filter_core_lengths(original, h)
                assert len(original) == report['accepted_lines']
                ids = np.flatnonzero(keep)
                mapping = np.full(len(original), -1, dtype=np.int32)
                mapping[ids] = np.arange(len(ids), dtype=np.int32)
                maps[key] = mapping
                retained_by_method[key] = retained
                output = target / f'vtk_higher{key}_clean.vtp'
                write_lines(output, retained)
                saved = read_cores(output)
                assert len(saved) == len(retained)
                # Verify the final file preserves every original retained vertex exactly.
                for before, after in zip(retained, saved):
                    np.testing.assert_array_equal(before, after)
                report.update({'winding_accepted_lines': len(original),
                               'winding_total_length': float(lengths.sum()),
                               'accepted_lines': len(saved), 'accepted_points': sum(map(len, saved)),
                               'total_length': float(lengths[keep].sum()),
                               'longest_length': float(lengths[keep].max()) if keep.any() else 0.,
                               'length_rejected_lines': int((~keep).sum()),
                               'minimum_core_length': 10 * h, 'status': 'winding_and_10h_length_checked'})
                payload['clean_core'][key] = [np.round(c, 6).tolist() for c in saved]
                method_audits.append({'key': key, 'before': len(original), 'retained': len(saved),
                                      'removed': int((~keep).sum()), 'lengths_before': lengths.tolist(),
                                      'retained_source_ids': ids.tolist(),
                                      'removed_source_ids': np.flatnonzero(~keep).tolist(),
                                      'retained_total_length': float(lengths[keep].sum()),
                                      'output_sha256': sha256(output)})
            payload['cleaning']['selected_key'] = selected
            payload['cleaning']['status'] = 'length_filter_complete'
            save_json(target / 'cleaning.json', payload['cleaning'])
            with np.load(source / 'sample_seeds.npz') as data:
                arrays = {name: data[name] for name in data.files}
            source_ids = arrays['near_core_id'].copy()
            mapping = maps[selected]
            assigned = source_ids >= 0
            assert not assigned.any() or int(source_ids[assigned].max()) < len(mapping)
            new_ids = np.full(len(source_ids), -1, dtype=np.int32)
            new_ids[assigned] = mapping[source_ids[assigned]]
            arrays['source_near_core_id'] = source_ids
            arrays['near_core_id'] = new_ids
            np.savez_compressed(target / 'sample_seeds.npz', **arrays)
            seeds = arrays['seeds']
            assert len(seeds) == 400000
            tree = cKDTree(seeds)
            coverage = []
            for cid, core in enumerate(retained_by_method[selected]):
                counts = tree.query_ball_point(core, 2 * h, return_length=True)
                dedicated = int((new_ids == cid).sum())
                assert dedicated > 0 and counts.min() > 0
                coverage.append({'core': cid, 'assigned_near_core_samples': dedicated,
                                 'minimum_seeds_within_2h_of_each_vertex': int(counts.min())})
            previous_sampling = copy.deepcopy(payload['sampling'])
            core_file = target / f'vtk_higher{selected}_clean.vtp'
            sampling = {'count': len(seeds), 'selected_method': selected,
                        'accepted_corelines': len(coverage), 'field': previous_sampling['field'],
                        'sampling_core_sha256': sha256(core_file),
                        'near_core_sample_counts': [c['assigned_near_core_samples'] for c in coverage],
                        'former_removed_core_samples': int((assigned & (new_ids < 0)).sum()),
                        'current_general_candidate_samples': int((new_ids < 0).sum()),
                        'status': 'reuse validated candidate samples; source core assignments remapped',
                        'source_sampling': previous_sampling}
            save_json(target / 'sampling.json', sampling)
            dataset_reference = {'source_scientific_version': config['parents'][flow],
                                 'sample_streamlines': str((source / 'sample_streamlines.npy').relative_to(ROOT)),
                                 'sample_seeds': str((target / 'sample_seeds.npz').relative_to(ROOT)),
                                 'corelines': str(core_file.relative_to(ROOT)),
                                 'relative_velocity': str((source / 'relative_velocity.npy').relative_to(ROOT)),
                                 'coordinates': str((source / 'coordinates.npz').relative_to(ROOT)),
                                 'source_audit': str((parent / 'preview_audit.json').relative_to(ROOT))}
            for path in [source / 'sample_seeds.npz', source / 'sample_streamlines.npy',
                         source / 'coordinates.npz', parent / 'preview_audit.json']:
                source_hashes[str(path.relative_to(ROOT))] = sha256(path)
            save_json(target / 'dataset_reference.json', dataset_reference)
            example = payload.get('winding_example')
            if example:
                cores = retained_by_method[str(example.get('method', '0'))]
                center = np.asarray(example['center'])
                belongs = any(np.linalg.norm(c - center, axis=1).min() < 1e-8 for c in cores)
                if not belongs:
                    payload['winding_example'] = None
                payload['winding_example_retained_by_length'] = belongs
            payload.update({'length_filter_version': '1.4', 'length_filter': geometry, 'sampling': sampling})
            filename = f'{flow}_{index}_length1_4.json'
            (VIEWER / filename).write_text(json.dumps(payload, separators=(',', ':'), ensure_ascii=False), encoding='utf-8')
            item.update({'file': filename, 'clean_lines': len(coverage), 'minimum_core_length': 10 * h,
                         'length_filter_version': '1.4'})
            report = {'flow': flow, 'index': index, 'time': payload['time'], **geometry,
                      'selected_method': selected, 'methods': method_audits, 'coverage': coverage,
                      'samples_reused': len(seeds), 'former_removed_core_samples': sampling['former_removed_core_samples'],
                      'source_hashes': source_hashes, 'passed': True}
            save_json(target / 'length_audit.json', report)
            audit_frames.append(report)
            print(json.dumps({'flow': flow, 'index': index, 'h': h, 'retained': len(coverage),
                              'methods': [{k: m[k] for k in ['key', 'before', 'retained', 'removed']} for m in method_audits]}), flush=True)
    manifest.update({'viewer_version': 'Verify_Task6_Viewer_1.5',
                     'new_scientific_version': 'Verify_Task6_VortexCore_1.4 (10h filter on saved previews)',
                     'core_length_rule': config['length_filter']})
    save_json(VIEWER / 'manifest.json', manifest)
    shutil.copy2(ROOT / 'experiments/templates/task6_vortex_viewer_1_5.html', VIEWER / 'index.html')
    save_json(BASE / 'length_audit.json', {'version': config['experiment'], 'frames': audit_frames,
                                        'passed': True, 'training_enabled': False,
                                        'samples_reintegrated': 0, 'samples_reused': 2400000})


if __name__ == '__main__':
    main()
