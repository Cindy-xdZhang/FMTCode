"""Build shared v3 ball6/16 indices; this CPU stage never trains a model."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import numpy as np
from FMT_Utils.Task4C_GTHeadCoverage_1_1 import sha
from FMT_Utils import Task4C_BallQuery_3_1 as ball
from FMT_Utils.Task4C_BallQuery_2_2 import count_summary, describe, select_fps
from FMT_Utils.Task4C_FixedDatasetFMT_2_1 import split_masks

CONFIG = 'config/Verify_Task4C_BallQuery_3.1.json'


def write(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(path)


def build(config, index):
    spec = json.loads(Path(config).read_text())
    root = Path(spec['output']); flow = spec['flows'][index]; name = flow['name']
    source = Path(spec['source_output']); folder = source/'physical'/name
    assert sha(source/'data_audit.json') == spec['source_audit_sha256']
    audit = json.loads((source/'data_audit.json').read_text()); assert audit['complete']
    for filename, digest in audit['frozen_files'][name].items():
        assert sha(folder/filename) == digest
    seeds = np.load(folder/'seeds.npy', mmap_mode='r')
    out = root/name; out.mkdir(parents=True, exist_ok=False)
    arrays = ball.ball_tables(seeds, 3*flow['h_value'], chunk=spec['chunk'], progress=True)
    # Independent full-domain scans, with the original NumPy FPS implementation.
    probes = np.unique(np.r_[0, len(seeds)-1, np.random.default_rng(96611).integers(len(seeds), size=30)])
    for i in probes:
        distance = np.linalg.norm(seeds-seeds[i], axis=1)
        assert int(np.sum(distance <= np.nextafter(3*flow['h_value'], np.inf)))-1 == arrays['initial_count'][i]
        for k in (6, 16):
            radius = arrays[f'effective_radius{k}'][i]
            ids = np.flatnonzero((distance <= np.nextafter(radius, np.inf)) & (np.arange(len(seeds)) != i))
            if arrays[f'expanded{k}'][i]:
                ids = np.flatnonzero((distance <= radius+8*np.finfo(float).eps*max(1., radius)) & (np.arange(len(seeds)) != i))
            expected = select_fps(seeds, seeds[i], ids, k) if len(ids) > k else ids
            np.testing.assert_array_equal(expected, arrays[f'order{k}'][i])
    for key, array in arrays.items():
        np.save(out/f'{key}.npy', array)
    with np.load(folder/'metadata.npz') as z:
        masks = split_masks({'fold': z['fold']}, index, spec)
        counts = {r: int(mask.sum())*3 for r, mask in masks.items()}
    write(out/'manifest.json', dict(complete=True, flow=name, samples=len(seeds), counts=counts,
        source_audit_sha256=spec['source_audit_sha256'], seeds_sha256=audit['frozen_files'][name]['seeds.npy'],
        h=flow['h_value'], radius_h=3, rule=spec['rule'], initial_count=count_summary(arrays['initial_count']),
        neighbors={str(k): dict(expanded=int(arrays[f'expanded{k}'].sum()), radius=describe(arrays[f'effective_radius{k}'])) for k in (6,16)},
        independent_centers=len(probes), files={key+'.npy':sha(out/(key+'.npy')) for key in arrays},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha(config), completed_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(flow=name, status='PASS', samples=len(seeds), counts=counts)), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default=CONFIG); p.add_argument('--index',type=int,required=True)
    a=p.parse_args(); build(a.config,a.index)


if __name__ == '__main__': main()
