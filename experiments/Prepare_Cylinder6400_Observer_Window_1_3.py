"""Lossless source-grid window extraction for a remote cache with a local source.

Preserve original physical times and indices, and the frozen spatial stride.
Only frames 105..118 are stored. Other frames are explicitly missing.
The observer mean is calculated from each FULL original grid before striding.
"""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/task123_plotdeps'),str(ROOT)]
import numpy as np
import netCDF4 as nc
from FMT_Utils.GlobalMeanObserver_3D import volume_mean

source=Path('C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D/halfcylinderRe6400.nc')
output=ROOT/'outputs/Other_FMTObjectivityTranslation_1.3/source_window'
output.mkdir(parents=True,exist_ok=True)
target=output/'halfcylinderRe6400_late_window.nc'
if target.exists():raise FileExistsError(target)
aliases={'x':{'x','xdim'},'y':{'y','ydim'},'z':{'z','zdim'},'t':{'t','time','tdim'}}
means=[]; missing=[]; hashes=[]
with nc.Dataset(source) as ds, nc.Dataset(target,'w') as dest:
    dims={a:next(d for d in ds.dimensions if d.lower() in aliases[a]) for a in 'tzyx'}
    coords={}
    for a,d in dims.items():
        candidates=[d]+[n for n in ds.variables if n.lower() in aliases[a]]
        var=next((ds.variables[n] for n in candidates if n in ds.variables and ds.variables[n].dimensions==(d,)),None)
        coords[a]=np.arange(len(ds.dimensions[d]),dtype=float) if var is None else np.asarray(var[:],dtype=float)
    strides={'x':7,'y':3,'z':1,'t':1}
    for a in 'tzyx':
        values=coords[a][::strides[a]]
        dest.createDimension(a,len(values))
        dest.createVariable(a,'f8',(a,))[:]=values
    names=next(n for n in [('u','v','w'),('velocity_x','velocity_y','velocity_z'),('Component1','Component2','Component3')]
               if all(k in ds.variables for k in n))
    outvars=[dest.createVariable(n,'f4',('t','z','y','x'),zlib=True,complevel=4,fill_value=np.float32(np.nan)) for n in ['u','v','w']]
    for index in range(105,119):
        fields=[]; h=hashlib.sha256()
        for name,outvar in zip(names,outvars):
            var=ds.variables[name];spatial=[d for d in var.dimensions if d!=dims['t']]
            data=np.ma.transpose(var[tuple(index if d==dims['t'] else slice(None) for d in var.dimensions)],
                                 [spatial.index(dims[a]) for a in 'zyx'])
            fields.append(data)
            h.update(np.asarray(data.filled(np.nan),dtype=np.float32).tobytes())
            outvar[index]=data[::1,::3,::7]
        mean, count=volume_mean(np.ma.stack(fields,-1),[coords[a] for a in 'zyx'])
        means.append(mean.tolist());missing.append(count);hashes.append(h.hexdigest())
        print(json.dumps({'index':index,'time':float(coords['t'][index]),'mean':means[-1]}),flush=True)
    dest.original_source=str(source)
    dest.extraction_rule='Original indices 105..118 only; xyz strides 7,3,1; all other frames missing'
    times=coords['t'][105:119].tolist()
info={'time_origin':'physical','times':times,'velocities':means,
      'mean_definition':'Full original 640x240x80 grid spatial volume mean per frame; tensor-product trapezoidal weights',
      'missing_nodes_per_frame':missing,'source_path':str(source),'source_size_bytes':source.stat().st_size,
      'source_frame_sha256':hashes,'stored_frame_indices':list(range(105,119)),
      'spatial_strides_xyz':[7,3,1],'window_file_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(output/'observer.json').write_text(json.dumps(info,indent=2))
print('Window bytes',target.stat().st_size,flush=True)
