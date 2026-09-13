"""Download channel/isotropic directly to binary legacy VTK, with roundtrip audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
from FLowUtils.flowDatasetUtils.JHTDB_Lodader import JHTDBLoader
from FLowUtils.flowDatasetUtils.JHTDB_VTK import write_frame,read_frame,read_grid,check_geometry,geometry,add_velocity,save
from experiments.Download_JHTDB_Channel import read_token,sha,write_json


def array_sha(a):
    return hashlib.sha256(np.asarray(a,dtype='<f4').tobytes(order='C')).hexdigest()


def axes_for(c):
    return [np.linspace(*c[f'{a}_range'],c[f'{a}_res']) for a in 'xyz']


def finish_flow(loader,c,out,manifest):
    from givernylocal.turbulence_toolkit import getData
    axes=axes_for(c)
    records=sorted(manifest['frames'],key=lambda r:r['index'])
    assert [r['index'] for r in records]==list(range(c['time_res']))
    errors=[]
    for r in (records[0],records[-1]):
        arr,_=read_frame(out/r['file'])
        y_index=c['y_res']//2
        plane=loader.load_2d_unsteadyFlow(c['dataset_name'],'xz',c['x_res'],c['z_res'],axes[1][y_index],
                c['x_range'],c['z_range'],r['time'],r['time'],1,temporal_method=c['temporal_method'],return_array=True)[0]
        difference=float(np.max(np.abs(plane-arr[:,y_index,:,:][...,[0,2]])))
        np.testing.assert_allclose(plane,arr[:,y_index,:,:][...,[0,2]],rtol=1e-6,atol=1e-7)
        errors.append(difference)
    rng=np.random.default_rng(20260906)
    ids=np.column_stack([rng.integers(0,len(axis),17) for axis in axes])
    points=np.column_stack([axis[ids[:,i]] for i,axis in enumerate(axes)])
    direct=np.asarray(getData(loader._get_or_create_dataset(c['dataset_name']),'velocity',records[-1]['time'],
                     c['temporal_method'],c['spatial_method'],'field',points,verbose=False)[0],dtype=np.float32)
    last,_=read_frame(out/records[-1]['file'])
    selected=last[ids[:,2],ids[:,1],ids[:,0]]
    np.testing.assert_allclose(direct,selected,rtol=1e-6,atol=1e-7)
    manifest['online_verification']={'xz_plane_max_abs_errors':errors,'random_point_max_abs_error':float(np.max(np.abs(direct-selected))),
                                      'random_seed':20260906,'random_points':17}
    combined=geometry(axes)
    series=[]
    for r in records:
        path=out/r['file']
        assert sha(path)==r['sha256']
        a,g=read_frame(path)
        assert array_sha(a)==r['velocity_sha256']
        check_geometry(g,axes)
        assert float(vtk_to_numpy(g.GetFieldData().GetArray('TimeValue'))[0])==r['time']
        add_velocity(combined,a,f'velocity_{r["time"]:.7f}')
        series.append({'name':r['file'],'time':r['time']})
    time_values=numpy_to_vtk(np.array([r['time'] for r in records]),deep=True)
    time_values.SetName('TimeValues')
    combined.GetFieldData().AddArray(time_values)
    combined_path=out/f'{c["name"]}.vtk'
    save(combined,combined_path)
    result=read_grid(combined_path)
    check_geometry(result,axes)
    assert result.GetPointData().GetNumberOfArrays()==len(records)
    for r in records:
        values=vtk_to_numpy(result.GetPointData().GetArray(f'velocity_{r["time"]:.7f}'))
        assert array_sha(values)==r['velocity_sha256']
    write_json(out/f'{c["name"]}.vtk.series',{'file-series-version':'1.0','files':series})
    manifest['combined']={'file':combined_path.name,'sha256':sha(combined_path),'time_arrays':len(records),'all_values_equal':True}
    manifest['status']='complete'
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(out/'manifest.json',manifest)


def run(config_path,token):
    config=json.loads(Path(config_path).read_text())
    root=Path(config['output_dir'])
    root.mkdir(parents=True,exist_ok=True)
    manifests={}
    kwargs={}
    sources={p:sha(Path(p)) for p in [__file__,'FLowUtils/flowDatasetUtils/JHTDB_VTK.py',
                                      'FLowUtils/flowDatasetUtils/JHTDB_Lodader.py','experiments/Download_JHTDB_Channel.py']}
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    for c in config['flows']:
        out=root/c['name']
        out.mkdir(exist_ok=True)
        signature=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        path=out/'manifest.json'
        if path.exists():
            m=json.loads(path.read_text())
            if m['config_sha256']!=signature:
                raise ValueError('Existing VTK output has a different sampling config')
        else:
            m={'version':config['version'],'config':c,'config_sha256':signature,'frames':[],
               'source_sha256':sources,'base_commit':commit,'status':'running',
               'started_utc':datetime.now(timezone.utc).isoformat(),
               'packages':{p:importlib.metadata.version(p) for p in ['vtk','givernylocal','numpy']},
               'format':'binary legacy STRUCTURED_GRID; PointData velocity; physical xyz; x fastest',
               'fresh_download':True}
        manifests[c['name']]=m
        kwargs[c['name']]={k:v for k,v in c.items() if k!='name'}
        write_json(path,m)
    local=threading.local()
    def download(c,i,t):
        name=c['name']
        out=root/name
        if not hasattr(local,'loader'):
            local.loader=JHTDBLoader(token,root,max_query_points=config['max_query_points'])
        start=time.perf_counter()
        values=local.loader.load_3d_unsteadyFlow(**dict(kwargs[name],time_start=float(t),time_end=float(t),time_res=1),return_array=True)[0]
        path=out/f'{name}_{i:03d}.vtk'
        axes=axes_for(c)
        write_frame(path,values,axes,float(t))
        restored,grid=read_frame(path)
        np.testing.assert_array_equal(restored,values)
        check_geometry(grid,axes)
        return name,{'index':i,'time':float(t),'file':path.name,'sha256':sha(path),'velocity_sha256':array_sha(values),
                     'finite':bool(np.isfinite(values).all()),'roundtrip_exact':True,'seconds':time.perf_counter()-start,
                     'min':values.min(axis=(0,1,2)).tolist(),'max':values.max(axis=(0,1,2)).tolist()}
    jobs=[]
    for i in range(max(c['time_res'] for c in config['flows'])):
        for c in config['flows']:
            if i>=c['time_res']:
                continue
            t=np.linspace(c['time_start'],c['time_end'],c['time_res'])[i]
            prior=next((r for r in manifests[c['name']]['frames'] if r['index']==i),None)
            if prior and (root/c['name']/prior['file']).exists() and sha(root/c['name']/prior['file'])==prior['sha256']:
                continue
            jobs.append((c,i,t))
    failures=[]
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(download,*job) for job in jobs]
        for future in as_completed(futures):
            try:
                name,r=future.result()
            except Exception as exc:
                failures.append(str(exc).replace(token,'<redacted>'))
                print('Frame failed; other frames continue',flush=True)
                continue
            m=manifests[name]
            m['frames']=[p for p in m['frames'] if p['index']!=r['index']]+[r]
            m['frames'].sort(key=lambda r:r['index'])
            write_json(root/name/'manifest.json',m)
            print(f'{name} {len(m["frames"])}/{m["config"]["time_res"]}; t={r["time"]:.7f}; {r["seconds"]:.1f}s; VTK exact',flush=True)
    if failures:
        write_json(root/'failure.json',{'errors':failures})
        raise RuntimeError('Some downloads failed; rerun to resume')
    loader=JHTDBLoader(token,root,max_query_points=config['max_query_points'])
    for c in config['flows']:
        try:
            finish_flow(loader,c,root/c['name'],manifests[c['name']])
            print(f'{c["name"]}: all VTK files and online queries verified',flush=True)
        except Exception as exc:
            m=manifests[c['name']]
            m['status']='verification_failed'
            m['error']=str(exc).replace(token,'<redacted>')
            write_json(root/c['name']/'manifest.json',m)
            raise RuntimeError(m['error']) from None


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/Verify_JHTDB_VTKDownload_1.1.json')
    p.add_argument('--token-source',type=Path,required=True)
    args=p.parse_args()
    run(args.config,read_token(args.token_source))
