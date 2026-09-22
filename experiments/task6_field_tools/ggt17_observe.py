"""Generate one instantaneous observed field using the frozen C++ Objective solver."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,time
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load_source(recipe,source,index):
    if recipe['format']=='amira':
        from amira_reader import read_series,read_amira
        paths,times=read_series(source)
        if not 0<index<len(times)-1:raise ValueError('Unsteady input requires the preceding and following frames')
        loaded=[read_amira(paths[i]) for i in (index-1,index,index+1)]
        for a in loaded[1:]:
            for x,y in zip(a[1],loaded[0][1]):np.testing.assert_array_equal(x,y)
        series=np.stack([a[0] for a in loaded]);axes=loaded[0][1]
        inputs={p.name:sha(p) for p in [source/'SquareCylinder.fileseries',*(paths[i] for i in (index-1,index,index+1))]}
    else:
        import netCDF4
        with netCDF4.Dataset(source) as ds:
            mapping=recipe['coordinate_variables'];axes=[np.asarray(ds[mapping[a]][:],np.float64) for a in 'zyx']
            if recipe['steady']:
                if index!=0:raise ValueError('Steady field has index 0 only')
                series=np.stack([np.asarray(ds[c][:],np.float32) for c in 'uvw'],-1)[None]
                times=np.array([0.],np.float64)
            else:
                times=np.asarray(ds[mapping['t']][:],np.float64)
                if not 0<index<len(times)-1:raise ValueError('Unsteady input requires the preceding and following frames')
                series=np.stack([np.stack([np.asarray(ds[c][i],np.float32) for c in 'uvw'],-1) for i in (index-1,index,index+1)])
        # Hash the actually consumed velocity slices, without scanning the whole time series.
        inputs={source.name:dict(bytes=source.stat().st_size,selected_velocity_sha256=hashlib.sha256(series.tobytes()).hexdigest())}
    if series.shape[1:]!=tuple(map(len,axes))+(3,):raise ValueError('Velocity must have storage order time,z,y,x,xyz components')
    if not np.isfinite(series).all():raise ValueError('Non-finite velocity')
    for a in axes:
        if len(a)<2 or not np.all(np.diff(a)>0) or not np.allclose(a,np.linspace(a[0],a[-1],len(a)),rtol=1e-5,atol=1e-6):
            raise ValueError('The frozen observer supports increasing uniform spatial axes only; no implicit resampling')
    if recipe['steady']:return series,axes,None,None,inputs
    dt=float((times[-1]-times[0])/(len(times)-1))
    if dt<=0 or not np.allclose(times,times[0]+np.arange(len(times))*dt,rtol=1e-6,atol=max(1e-6,abs(dt)*1e-4)):
        raise ValueError('The frozen observer requires a uniform time series')
    return series,axes,(float(times[0])+(index-1)*dt,dt),float(times[index]),inputs

def compute(series,axes,t0,dt,library,threads=None):
    aligned=series.shape[2]*series.shape[3]%64==0
    threads=(8 if aligned else 1) if threads is None else threads
    if threads<1 or (not aligned and threads!=1):
        raise ValueError('Use one thread for planes not aligned to 64 cells, to avoid the upstream packed validity-mask race')
    lib=ctypes.CDLL(str(Path(library).resolve()))
    fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn=lib.task6_cpp_reference
    fn.argtypes=[fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,dp,dp,ctypes.c_double,ctypes.c_double,
                 ctypes.c_int,ctypes.c_int,ctypes.c_int,fp]
    fn.restype=ctypes.c_int
    out=np.zeros(series.shape[1:],np.float32)
    lower=np.array([a[0] for a in axes[::-1]],np.float64);upper=np.array([a[-1] for a in axes[::-1]],np.float64)
    code=fn(np.ascontiguousarray(series,np.float32),*series.shape[1:4][::-1],lower,upper,t0,dt,11,1,threads,out)
    if code or not np.isfinite(out).all():raise RuntimeError(f'GGT17 returned {code} or non-finite values')
    return out,threads

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int);p.add_argument('--library',type=Path)
    args=p.parse_args();recipes=json.loads((HERE/'flow_recipes.json').read_text())
    if args.flow not in recipes:raise ValueError('Known flows: '+', '.join(recipes))
    if args.out.exists():raise FileExistsError('Output must be a new directory; curated labels are never overwritten')
    recipe=recipes[args.flow];started=time.perf_counter()
    series,axes,timing,physical_time,inputs=load_source(recipe,args.source,args.index)
    library=args.library or HERE/'bin'/('reference.dll' if os.name=='nt' else 'libreference.so')
    if recipe['steady']:field=series[0];threads=0;library_sha=None
    else:
        if not library.is_file():raise FileNotFoundError('Build the bundled native library or supply --library: '+str(library))
        library_sha=sha(library)
        if args.library is None and os.name=='nt':
            expected=json.loads((HERE/'native_identity.json').read_text())['windows_library_sha256']
            if library_sha!=expected:raise ValueError('Bundled Windows solver hash mismatch; use --library for an explicitly rebuilt solver')
        field,threads=compute(series,axes,*timing,library,threads=args.threads)
    spacing=[float((a[-1]-a[0])/(len(a)-1)) for a in axes[::-1]]
    args.out.mkdir(parents=True)
    np.save(args.out/'relative_velocity.npy',field)
    np.savez(args.out/'coordinates.npz',**dict(zip('zyx',axes)))
    report=dict(flow=args.flow,index=args.index,time=physical_time,steady=recipe['steady'],
        velocity_definition='v' if recipe['steady'] else 'v-u',storage_order='z,y,x,component_xyz',
        shape=list(field.shape),spacing_xyz=spacing,h=min(spacing),bounds_xyz=[[float(a[0]),float(a[-1])] for a in axes[::-1]],
        observer=dict(objective='EInvariance::Objective',NeighborhoodU=11,UseSummedAreaTables=True,
          implementation='frozen original computeWithOutFilter, middle of three input frames',
          temporal_origin=None if timing is None else timing[0],dt=None if timing is None else timing[1],
          library_sha256=library_sha,threads=threads,computed=not recipe['steady']),
        input_files=inputs,selected_velocity_sha256=hashlib.sha256((series[0] if recipe['steady'] else series).tobytes()).hexdigest(),
        files={n:sha(args.out/n) for n in ('relative_velocity.npy','coordinates.npz')},
        seconds=time.perf_counter()-started,coreline_labels_generated=False,streamlines_saved=False)
    (args.out/'generation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
