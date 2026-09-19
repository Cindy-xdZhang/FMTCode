"""Read-only radius counts and neighborhood diagnostics for frozen Task4-c v2."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_BallQuery_2_2 as ball
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as baseline

OUTPUT = 'outputs/Ablation_Task4C_BallQuery_2.2'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def grid_evidence(dataset_config, out, h_mode):
    from experiments.Build_Task4C_CandidateHeadRegion_1_2 import load_fields
    physical = json.loads(Path(dataset_config['physical_config']).read_text())
    root = next(Path(p) for p in dataset_config['input_roots'] if Path(p).is_dir())
    result = {}
    for flow in physical['flows']:
        path = root/flow['flow']; assert sha(path) == flow['flow_sha256']
        axes, _, _ = load_fields(path)
        seeds = np.load(Path(dataset_config['output'])/'physical'/flow['name']/'seeds.npy')
        if h_mode == 'local_min':
            from FMT_Utils.Task4C_GTHeadCoverage_1_1 import spacing_at
            h_values = spacing_at(seeds, [a.astype(np.float64) for a in axes]).min(1)
        else:
            h_values = np.full(len(seeds), ball.minimum_grid_spacing(axes))
        h_path = out/f"h_{flow['name']}.npy"; np.save(h_path, h_values)
        result[flow['name']] = dict(h=ball.minimum_grid_spacing(axes), source=str(path), source_sha256=sha(path),
            h_mode=h_mode, h_values=ball.describe(h_values), h_sha256=sha(h_path),
            definition='minimum edge of the sample grid cell' if h_mode == 'local_min' else 'minimum adjacent-grid spacing over all x/y/z axes of this flow',
            axis_spacing=[dict(minimum=float(np.diff(a.astype(float)).min()), maximum=float(np.diff(a.astype(float)).max())) for a in axes])
    return result


def run(radii, output=OUTPUT, h_mode='global_min'):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    dataset_config = json.loads(Path('config/mainExp_Task4C_FixedDataset_2.1.json').read_text())
    spec = json.loads(Path('config/mainExp_Task4C_FixedDatasetFMT_2.1.json').read_text())
    source = Path(spec['source_output']); audit = json.loads((source/'data_audit.json').read_text())
    assert audit['complete'] and sha(source/'data_audit.json') == spec['source_audit_sha256']
    grids = grid_evidence(dataset_config, out, h_mode)
    report = dict(version='Verify_Task4C_BallQueryCounts_2.2', dataset_audit_sha256=sha(source/'data_audit.json'), grids=grids,
                  radii_h=radii, h_mode=h_mode, count_unit='other seed points; one neighbor curve per length; center excluded', flows={})
    for fi, flow in enumerate(spec['flows']):
        name = flow['name']; folder = source/'physical'/name
        for file, digest in audit['frozen_files'][name].items(): assert sha(folder/file) == digest
        seeds = np.load(folder/'seeds.npy')
        with np.load(folder/'metadata.npz') as z: meta = {k: z[k] for k in ('fold', 'label', 'instance', 'component', 'local_grid_scale')}
        masks = dict(all=np.ones(len(seeds), bool), **baseline.split_masks(meta, fi, spec))
        tree = cKDTree(seeds); h = np.load(out/f'h_{name}.npy'); counts = []
        records = {}
        for radius in radii:
            n = tree.query_ball_point(seeds, np.nextafter(radius*h, np.inf), return_length=True, workers=4)-1
            counts.append(n)
            records[str(radius)] = {role: {label: ball.count_summary(n[mask & lm]) for label, lm in
                (('all', np.ones(len(seeds), bool)), ('hairpin', meta['label'] == 1), ('non_hairpin', meta['label'] == 0))} for role, mask in masks.items()}
        np.savez_compressed(out/f'counts_{name}.npz', radius_h=radii, counts=np.asarray(counts), h=h, seeds_sha256=sha(folder/'seeds.npy'))
        old_path = Path(spec['output'])/'neighbors'/f'neighbors_{name}.npz'
        old = {}
        if old_path.exists():
            with np.load(old_path) as z:
                assert str(z['seeds_sha256']) == sha(folder/'seeds.npy'); order = z['order']
            for k in (6, 16):
                ids = order[:, :k]; dist = np.linalg.norm(seeds[ids]-seeds[:, None], axis=2)
                positives = meta['label'] == 1
                old[str(k)] = dict(max_distance_over_h=ball.describe(dist.max(1)/h),
                    fraction_selected_edges_outside_1h=float(np.mean(dist > h[:, None])),
                    fraction_positive_centers_with_all_negative_neighbors=float(np.mean(np.all(meta['label'][ids[positives]] == 0, axis=1))),
                    fraction_edges_cross_test_boundary=float(np.mean((meta['fold'][ids] == 0) != (meta['fold'][:, None] == 0))))
        report['flows'][name] = dict(samples=len(seeds), h=ball.describe(h), fixed_radius_counts=records, old_fps=old,
                                    geometric_mean_grid_scale=ball.describe(meta['local_grid_scale']), frozen_files=audit['frozen_files'][name])
        print(json.dumps(dict(flow=name, h=ball.describe(h), at_1h=records.get('1.0', records.get('1', {})).get('all', {}).get('all'), old_fps=old)), flush=True)
    (out/'radius_statistics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radii', type=float, nargs='+', default=[.25, .5, 1, 1.5, 2, 3, 4, 6, 8, 10, 16])
    p.add_argument('--output', default=OUTPUT)
    p.add_argument('--h-mode', choices=['global_min', 'local_min'], default='global_min')
    a = p.parse_args()
    if any(not np.isfinite(r) or r <= 0 for r in a.radii): p.error('Radii must be positive and finite')
    run(sorted(set(a.radii)), a.output, a.h_mode)
