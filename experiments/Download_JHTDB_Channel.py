"""Resumable, per-frame JHTDB download and independent 2D/point verification.

Run with python -m experiments.Download_JHTDB_Channel --config CONFIG.
Credentials come from JHTDB_AUTH_TOKEN or an explicitly supplied existing source.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import threading

import netCDF4
import numpy as np
from FLowUtils.flowDatasetUtils.JHTDB_Lodader import JHTDBLoader


def read_token(path):
    """Read only a literal credential assignment, without executing source code."""
    if path is None:
        return None
    for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8-sig'))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('my_api_key', 'auth_token') for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise ValueError('No literal token assignment found in the supplied source')


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    temp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def run(config, token, probe=False, workers=1):
    c = json.loads(Path(config).read_text())
    if workers not in range(1,5):
        raise ValueError('workers must be between 1 and 4')
    if any(not 1 <= c[f'{a}_res'] <= 512 for a in 'xyz'):
        raise ValueError('Download dimensions must be in [1,512]')
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    loader = JHTDBLoader(token, out, max_query_points=c['max_query_points'])
    kwargs = {k:v for k,v in c.items() if k not in ('version','output_dir','max_query_points')}
    if probe:
        kwargs.update(x_res=3,y_res=4,z_res=5,time_res=1,time_end=c['time_start'])
        start=time.perf_counter()
        arr=loader.load_3d_unsteadyFlow(**kwargs,return_array=True)
        record={'shape':list(arr.shape),'finite':bool(np.isfinite(arr).all()),'min':arr.min(axis=(0,1,2,3)).tolist(),
                'max':arr.max(axis=(0,1,2,3)).tolist(),'seconds':time.perf_counter()-start}
        write_json(out/'probe.json',record)
        print(json.dumps(record),flush=True)
        return
    signature=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['config_sha256']!=signature:
            raise ValueError('Existing output belongs to a different config')
    else:
        commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        manifest={'version':c['version'],'config':c,'config_sha256':signature,'base_commit':commit,
                  'source_sha256':{p:sha(Path(p)) for p in ['FLowUtils/flowDatasetUtils/JHTDB_Lodader.py',__file__]},
                  'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','frames':[],
                  'packages':{x:importlib.metadata.version(x) for x in ['numpy','givernylocal','netCDF4']},
                  'frame':'stationary walls; fixed physical coordinates',
                  'array_order':'time,z,y,x,component; components u,v,w'}
    manifest.setdefault('attempts',[])
    if not manifest['attempts'] and manifest['frames']:
        manifest['attempts'].append({'workers':1,'status':'interrupted to enable bounded concurrency',
                                    'completed_frames':[r['index'] for r in manifest['frames']],
                                    'source_sha256':manifest['source_sha256'],
                                    'source_snapshot':'source_attempt_001'})
    attempt={'started_utc':datetime.now(timezone.utc).isoformat(),'workers':workers,'status':'running',
             'source_sha256':{p:sha(Path(p)) for p in ['FLowUtils/flowDatasetUtils/JHTDB_Lodader.py',__file__]}}
    manifest['attempts'].append(attempt)
    manifest['status']='running'
    manifest.pop('error',None)
    write_json(manifest_path,manifest)
    times,_=loader._generate_times(c['temporal_method'],c['time_start'],c['time_res'],c['time_end'])
    local=threading.local()
    def download_frame(i,t):
        if not hasattr(local,'loader'):
            local.loader=JHTDBLoader(token,out,max_query_points=c['max_query_points'])
        file=out/f'frame_{i:03d}.npy'
        start=time.perf_counter()
        frame_args=dict(kwargs,time_start=float(t),time_res=1,time_end=float(t))
        arr=local.loader.load_3d_unsteadyFlow(**frame_args,return_array=True)[0]
        temp=file.with_suffix('.partial')
        with temp.open('wb') as f:
            np.save(f,arr)
        temp.replace(file)
        return {'index':i,'time':float(t),'file':file.name,'sha256':sha(file),'seconds':time.perf_counter()-start,
                'min':arr.min(axis=(0,1,2)).tolist(),'max':arr.max(axis=(0,1,2)).tolist(),
                'mean':arr.mean(axis=(0,1,2),dtype=np.float64).tolist(),'finite':bool(np.isfinite(arr).all()),
                'attempt':len(manifest['attempts'])}
    try:
        pending=[]
        for i,t in enumerate(times):
            file=out/f'frame_{i:03d}.npy'
            prior=next((r for r in manifest['frames'] if r['index']==i),None)
            if prior and file.exists() and sha(file)==prior['sha256']:
                continue
            pending.append((i,t))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(download_frame,i,t) for i,t in pending]
            errors=[]
            for future in as_completed(futures):
                try:
                    r=future.result()
                except Exception as exc:
                    errors.append(str(exc).replace(loader.auth_token,'<redacted>'))
                    continue
                manifest['frames']=[p for p in manifest['frames'] if p['index']!=r['index']]+[r]
                manifest['frames'].sort(key=lambda r:r['index'])
                write_json(manifest_path,manifest)
                print(f'Completed {len(manifest["frames"])}/{len(times)}; frame {r["index"]} t={r["time"]:.7f}: {r["seconds"]:.1f}s, finite={r["finite"]}',flush=True)
            if errors:
                raise RuntimeError('; '.join(errors))
        verify(loader,c,out,times,manifest)
        export_nc(c,out,times)
        manifest['status']='complete'
        attempt['status']='complete'
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        manifest['netcdf_sha256']=sha(out/'channel.nc')
        write_json(manifest_path,manifest)
    except Exception as exc:
        manifest['status']='failed'
        attempt['status']='failed'
        manifest['error']=str(exc).replace(loader.auth_token,'<redacted>')
        write_json(manifest_path,manifest)
        raise RuntimeError(manifest['error']) from None


def verify(loader,c,out,times,manifest):
    """Compare full xz planes via the 2D API and random points via direct getData."""
    from givernylocal.turbulence_toolkit import getData
    axes=[np.linspace(*c[f'{a}_range'],c[f'{a}_res']) for a in 'xyz']
    iy=c['y_res']//2
    errs=[]
    for i in (0,len(times)-1):
        arr=np.load(out/f'frame_{i:03d}.npy',mmap_mode='r')
        plane=loader.load_2d_unsteadyFlow('channel','xz',c['x_res'],c['z_res'],axes[1][iy],
                    c['x_range'],c['z_range'],float(times[i]),float(times[i]),1,
                    temporal_method=c['temporal_method'],return_array=True)[0]
        expected=arr[:,iy,:,:][...,[0,2]]
        np.testing.assert_allclose(plane,expected,rtol=1e-6,atol=1e-7)
        errs.append(float(np.max(np.abs(plane-expected))))
    rng=np.random.default_rng(20260906)
    ids=np.column_stack([rng.integers(0,n,size=17) for n in (c['x_res'],c['y_res'],c['z_res'])])
    pts=np.column_stack([axis[ids[:,i]] for i,axis in enumerate(axes)])
    response=getData(loader._get_or_create_dataset('channel'),'velocity',float(times[-1]),
                     c['temporal_method'],c['spatial_method'],'field',pts,verbose=False)
    direct=np.asarray(response[0],dtype=np.float32)
    last=np.load(out/f'frame_{len(times)-1:03d}.npy',mmap_mode='r')
    selected=last[ids[:,2],ids[:,1],ids[:,0]]
    np.testing.assert_allclose(direct,selected,rtol=1e-6,atol=1e-7)
    first=np.load(out/'frame_000.npy',mmap_mode='r')
    manifest['verification']={'xz_plane_max_abs_errors':errs,'direct_point_max_abs_error':float(np.max(np.abs(direct-selected))),
                              'first_last_rms_change':float(np.sqrt(np.mean((last-first)**2))),
                              'random_seed':20260906,'random_point_count':17}


def export_nc(c,out,times):
    temp=out/'channel.partial.nc'
    with netCDF4.Dataset(temp,'w') as ds:
        for a in 'xyz':
            ds.createDimension(a,c[f'{a}_res'])
            ds.createVariable(a,'f8',(a,))[:]=np.linspace(*c[f'{a}_range'],c[f'{a}_res'])
        ds.createDimension('time',len(times))
        ds.createVariable('time','f8',('time',))[:]=times
        v=[ds.createVariable(a,'f4',('time','z','y','x'),zlib=True,complevel=1) for a in 'uvw']
        ds.dataset='JHTDB channel'
        ds.sampling_frame='stationary walls; uniform physical query grid, spatially interpolated'
        ds.temporal_method=c['temporal_method']
        ds.spatial_method=c['spatial_method']
        ds.config_json=json.dumps(c)
        for i in range(len(times)):
            frame=np.load(out/f'frame_{i:03d}.npy',mmap_mode='r')
            for j,var in enumerate(v):
                var[i]=frame[...,j]
    temp.replace(out/'channel.nc')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/Verify_JHTDB_ChannelDownload_1.1.json')
    parser.add_argument('--token-source',type=Path)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    run(args.config,read_token(args.token_source),args.probe,args.workers)
