"""Audit all VTK velocity values and optionally test the user's PyFlowVis reader."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from FLowUtils.flowDatasetUtils.JHTDB_VTK import read_grid,read_frame,check_geometry
from experiments.Download_JHTDB_Channel import sha,write_json
from experiments.Download_JHTDB_VTK import array_sha,axes_for


def audit(root,reference_loader=None,old_channel=None):
    root=Path(root)
    reports={}
    reference=None
    if reference_loader:
        name='FLowUtils.flowDatasetUtils._jhtdb_reference_vtk_loader'
        spec=importlib.util.spec_from_file_location(name,reference_loader)
        reference=importlib.util.module_from_spec(spec)
        sys.modules[name]=reference
        spec.loader.exec_module(reference)
    for name in ('channel','isotropic'):
        out=root/name
        m=json.loads((out/'manifest.json').read_text())
        assert m['status']=='complete'
        c=m['config']
        records=sorted(m['frames'],key=lambda r:r['index'])
        assert [r['index'] for r in records]==list(range(c['time_res']))
        times=np.linspace(c['time_start'],c['time_end'],c['time_res'])
        np.testing.assert_array_equal([r['time'] for r in records],times)
        axes=axes_for(c)
        combined_path=out/m['combined']['file']
        assert sha(combined_path)==m['combined']['sha256']
        combined=read_grid(combined_path)
        check_geometry(combined,axes)
        series=json.loads((out/f'{name}.vtk.series').read_text())
        assert series['files']==[{'name':r['file'],'time':r['time']} for r in records]
        field=None
        if reference:
            field=reference.VTKLoader.load_vector_field3d(str(combined_path))
            np.testing.assert_allclose(field.vtk_time_values,times,rtol=0,atol=1e-12)
            assert field.field.shape==(c['time_res'],c['z_res'],c['y_res'],c['x_res'],3)
        same_old=0
        for r in records:
            path=out/r['file']
            assert sha(path)==r['sha256']
            arr,grid=read_frame(path)
            assert arr.dtype==np.float32 and np.isfinite(arr).all()
            assert array_sha(arr)==r['velocity_sha256']
            check_geometry(grid,axes)
            np.testing.assert_array_equal(vtk_to_numpy(combined.GetPointData().GetArray(f'velocity_{r["time"]:.7f}')),arr.reshape(-1,3))
            if field is not None:
                np.testing.assert_array_equal(field.field[r['index']],arr)
            if old_channel and name=='channel':
                previous=Path(old_channel)/f'frame_{r["index"]:03d}.npy'
                np.testing.assert_array_equal(np.load(previous,mmap_mode='r'),arr)
                same_old+=1
        reports[name]={'status':'pass','frames':len(records),'combined_time_arrays':combined.GetPointData().GetNumberOfArrays(),
                       'vtk_coordinates_and_every_velocity':'exactly equal',
                       'pyflowvis_reader': 'all frames, times and values match' if reference else 'not tested',
                       'online_verification':m['online_verification'],
                       'fresh_download_equal_old_frames':same_old}
    result={'status':'pass','flows':reports,'audit_source_sha256':sha(Path(__file__)),
            'reference_loader':str(reference_loader) if reference_loader else None,
            'reference_loader_sha256':sha(Path(reference_loader)) if reference_loader else None}
    write_json(root/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='outputs/Verify_JHTDB_VTKDownload_1.1')
    p.add_argument('--reference-loader',type=Path)
    p.add_argument('--old-channel',type=Path)
    a=p.parse_args()
    audit(a.root,a.reference_loader,a.old_channel)
