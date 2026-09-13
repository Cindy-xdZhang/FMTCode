"""Independently check the downloaded arrays, NetCDF copy, and FMT reader."""
import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
from FLowUtils.flowDatasetUtils.JHTDB_NetCDF import load_jhtdb_netcdf


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def audit(directory):
    out=Path(directory)
    m=json.loads((out/'manifest.json').read_text())
    c=m['config']
    assert m['status']=='complete'
    records=sorted(m['frames'],key=lambda r:r['index'])
    assert [r['index'] for r in records]==list(range(c['time_res']))
    times=np.linspace(c['time_start'],c['time_end'],c['time_res'])
    np.testing.assert_array_equal([r['time'] for r in records],times)
    assert digest(out/'channel.nc')==m['netcdf_sha256']
    shape=(c['time_res'],c['z_res'],c['y_res'],c['x_res'],3)
    field=load_jhtdb_netcdf(out/'channel.nc')
    assert field.field.shape==shape
    assert field.field.dtype==np.float32
    np.testing.assert_array_equal(field.jhtdb_times,times)
    with netCDF4.Dataset(out/'channel.nc') as ds:
        np.testing.assert_array_equal(ds.variables['time'][:],times)
        for a in 'xyz':
            np.testing.assert_array_equal(ds.variables[a][:],np.linspace(*c[f'{a}_range'],c[f'{a}_res']))
        for r in records:
            path=out/r['file']
            assert digest(path)==r['sha256']
            arr=np.load(path,mmap_mode='r')
            assert arr.shape==shape[1:] and arr.dtype==np.float32
            assert np.isfinite(arr).all()
            assert not np.any(arr==np.float32(-999.9))
            np.testing.assert_array_equal(field.field[r['index']],arr)
            for j,a in enumerate('uvw'):
                np.testing.assert_array_equal(ds.variables[a][r['index']],arr[...,j])
    report={'status':'pass','shape':list(shape),'dtype':'float32','frame_hashes_verified':len(records),
            'netcdf_vs_npy':'exactly equal, every value','fmt_reader_vs_npy':'exactly equal, every value',
            'time_start':float(times[0]),'time_end':float(times[-1]),'time_step':float(times[1]-times[0]),
            'online_verification':m['verification'],'audit_source_sha256':digest(Path(__file__)),
            'final_source_sha256':{p:digest(Path(p)) for p in [
                'FLowUtils/flowDatasetUtils/JHTDB_Lodader.py','FLowUtils/flowDatasetUtils/JHTDB_NetCDF.py',
                'experiments/Download_JHTDB_Channel.py','experiments/Visualize_JHTDB_Channel.py',
                'tests/test_jhtdb_loader.py','tests/test_jhtdb_netcdf.py']}}
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',nargs='?',default='outputs/Verify_JHTDB_ChannelDownload_1.1')
    audit(p.parse_args().directory)
