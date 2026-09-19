"""Exact neighbor counts with user-fixed Channel dy and TBL dz on frozen v2 r3."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.spatial import cKDTree
from experiments.Analyze_Task4C_BallQuery_2_2 import sha
from FMT_Utils import Task4C_BallQuery_2_2 as ball
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as baseline

CONFIG = 'config/Verify_Task4C_BallQueryCounts_2.3.json'


def fixed_grids(config, out, spec):
    previous = json.loads(Path(config['grid_evidence']).read_text())
    assert previous['dataset_audit_sha256'] == spec['source_audit_sha256']
    grids = {}
    for name, choice in config['fixed_h'].items():
        old = previous['grids'][name]
        source = Path(old['source'])
        assert sha(source) == old['source_sha256'], source
        h = float(choice['value'])
        axis_minimum = old['axis_spacing'][choice['axis_index']]['minimum']
        assert np.isfinite(h) and h > 0 and abs(h-axis_minimum) < 1e-10
        seeds = np.load(Path(spec['source_output'])/'physical'/name/'seeds.npy', mmap_mode='r')
        values = np.full(len(seeds), h, dtype=np.float64)
        file = out/f'h_{name}.npy'; np.save(file, values)
        grids[name] = dict(h=h, h_mode=config['h_mode'], h_values=ball.describe(values), h_sha256=sha(file),
            definition=choice['label'], selected_axis=choice['axis'], axis_minimum_from_vtk=axis_minimum,
            source=str(source), source_sha256=old['source_sha256'], axis_spacing=old['axis_spacing'],
            previous_global_minimum_h=old['h'], authority='User-specified fixed decimal constant, 2026-09-19')
    return grids


def check_counts(seeds, h, radii, counts):
    """Independent Euclidean scan at fixed random centers and every radius's extrema."""
    assert counts.shape == (len(radii), len(seeds))
    assert np.all(counts >= 0) and np.all(counts < len(seeds))
    assert np.all(np.diff(counts, axis=0) >= 0)
    rng = np.random.default_rng(98304)
    ids = np.unique(np.r_[rng.choice(len(seeds), min(32, len(seeds)), replace=False),
                          counts.argmin(1), counts.argmax(1)])
    for i in ids:
        distance = np.sqrt(np.sum((seeds-seeds[i])**2, axis=1))
        for j, radius in enumerate(radii):
            expected = np.count_nonzero(distance <= np.nextafter(radius*h, np.inf))-1
            assert int(counts[j, i]) == expected, (int(i), radius, expected, int(counts[j, i]))
    return dict(complete=True, independent_method='direct float64 Euclidean distance scan of all seeds',
                centers=ids.tolist(), radii_checked=len(radii), monotonic_counts=True)


def run(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text(encoding='utf8'))
    spec = json.loads(Path(config['source_config']).read_text())
    radii = np.asarray(config['radii_h'], np.float64)
    assert np.all(np.isfinite(radii)) and np.all(radii > 0) and np.all(np.diff(radii) > 0)
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    source = Path(spec['source_output']); audit = json.loads((source/'data_audit.json').read_text())
    assert audit['complete'] and sha(source/'data_audit.json') == spec['source_audit_sha256']
    grids = fixed_grids(config, out, spec)
    report = dict(version=config['version'], config_sha256=sha(config_path), dataset_audit_sha256=sha(source/'data_audit.json'),
        generated_at_utc=datetime.now(timezone.utc).isoformat(), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={p: sha(p) for p in (config_path, __file__, 'experiments/Analyze_Task4C_BallQuery_2_2.py',
            'FMT_Utils/Task4C_BallQuery_2_2.py', 'FMT_Utils/Task4C_FixedDatasetFMT_2_1.py', config['source_config'], config['grid_evidence'])},
        grids=grids, radii_h=radii.tolist(), h_mode=config['h_mode'], definition=config['definition'],
        count_unit='other seed points; one neighbor curve per chosen length; center excluded', flows={})
    for fi, flow in enumerate(spec['flows']):
        started = time.perf_counter(); name = flow['name']; folder = source/'physical'/name
        for file, digest in audit['frozen_files'][name].items():
            assert sha(folder/file) == digest, (name, file)
        seeds = np.load(folder/'seeds.npy')
        with np.load(folder/'metadata.npz') as z:
            meta = {k: z[k] for k in ('fold', 'label')}
        masks = dict(all=np.ones(len(seeds), bool), **baseline.split_masks(meta, fi, spec))
        tree = cKDTree(seeds); h = grids[name]['h']; counts = []; records = {}
        for radius in radii:
            n = tree.query_ball_point(seeds, np.nextafter(radius*h, np.inf), return_length=True, workers=config['workers'])-1
            counts.append(n)
            records[str(float(radius))] = {role: {label: ball.count_summary(n[mask & lm]) for label, lm in
                (('all', np.ones(len(seeds), bool)), ('hairpin', meta['label'] == 1), ('non_hairpin', meta['label'] == 0))} for role, mask in masks.items()}
            print(json.dumps(dict(flow=name, radius_h=float(radius), physical_radius=float(radius*h),
                minimum=int(n.min()), mean=float(n.mean()), maximum=int(n.max()), fewer16=int((n < 16).sum()))), flush=True)
        counts = np.asarray(counts, np.int64)
        verification = check_counts(seeds, h, radii, counts)
        file = out/f'counts_{name}.npz'
        np.savez_compressed(file, radius_h=radii, counts=counts, h=h, seeds_sha256=sha(folder/'seeds.npy'))
        report['flows'][name] = dict(samples=len(seeds), h=grids[name]['h_values'], fixed_radius_counts=records,
            frozen_files=audit['frozen_files'][name], count_file_sha256=sha(file), verification=verification,
            seconds=time.perf_counter()-started)
        (out/'radius_statistics.partial.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    report['complete'] = True
    (out/'radius_statistics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    run(parser.parse_args().config)
