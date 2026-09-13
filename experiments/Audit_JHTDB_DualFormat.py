"""Audit every time, coordinate and velocity in both JHTDB storage formats."""
import argparse
import json
from pathlib import Path

import numpy as np

from FLowUtils.flowDatasetUtils.JHTDB_NetCDF import load_jhtdb_netcdf
from FLowUtils.flowDatasetUtils.JHTDB_VTK import read_frame
from experiments.Audit_JHTDB_VTK import audit as audit_vtk
from experiments.JHTDB_DualFormat import verify_netcdf
from experiments.Download_JHTDB_Channel import sha, write_json


def audit(root, reference_loader=None):
    root = Path(root)
    audit_vtk(root, reference_loader)
    report = json.loads((root / 'audit.json').read_text())
    # The combined audit is published only after both formats pass.
    (root / 'audit.json').replace(root / 'vtk_audit.json')
    for name in ('channel', 'isotropic'):
        out = root / name
        m = json.loads((out / 'manifest.json').read_text())
        distinct = len({r['velocity_sha256'] for r in m['frames']})
        assert distinct == m['config']['time_res'], 'Unexpected duplicate turbulent velocity frames'
        report['flows'][name]['distinct_velocity_arrays'] = distinct
        path = out / m['netcdf']['file']
        assert sha(path) == m['netcdf']['sha256']
        verify_netcdf(path, m['config'], out, m['frames'])
        field = load_jhtdb_netcdf(path)
        c = m['config']
        assert field.field.shape == (c['time_res'], c['z_res'], c['y_res'], c['x_res'], 3)
        np.testing.assert_array_equal(field.jhtdb_times, np.linspace(c['time_start'], c['time_end'], c['time_res']))
        for r in m['frames']:
            arr, _ = read_frame(out / r['file'])
            np.testing.assert_array_equal(field.field[r['index']], arr)
        del field
        report['flows'][name]['netcdf'] = dict(status='pass', frames=len(m['frames']),
                                             all_coordinates_times_velocities_equal_vtk=True,
                                             fmt_reader='all frames including the last exactly match VTK')
    report['dual_format_audit_source_sha256'] = sha(Path(__file__))
    write_json(root / 'audit.json', report)
    print('Both complete NetCDF time series exactly match all VTK frames.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='outputs/Verify_JHTDB_DualFormatDownload_1.2')
    p.add_argument('--reference-loader', type=Path)
    args = p.parse_args()
    audit(args.root, args.reference_loader)
