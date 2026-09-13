"""Extend the verified Re6400 source window to 26 frames for 2x path duration."""
from pathlib import Path
import sys,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/task123_plotdeps'),str(ROOT)]
import numpy as np
import netCDF4 as nc
from FMT_Utils.GlobalMeanObserver_3D import volume_mean

previous=ROOT/'outputs/Other_FMTObjectivityTranslation_1.3/source_window'
output=ROOT/'outputs/Other_FMTObjectivityTranslation_1.4/source_window'
output.mkdir(parents=True,exist_ok=True)
info=json.loads((previous/'observer.json').read_text())
source=Path(info['source_path']);target=output/'halfcylinderRe6400_long_window.nc'
if target.exists():raise FileExistsError(target)
shutil.copyfile(previous/'halfcylinderRe6400_late_window.nc',target)
aliases={'x':{'x','xdim'},'y':{'y','ydim'},'z':{'z','zdim'},'t':{'t','time','tdim'}}
with nc.Dataset(source) as ds,nc.Dataset(target,'r+') as dest:
    dims={a:next(d for d in ds.dimensions if d.lower() in aliases[a]) for a in 'tzyx'}
    coords={}
    for a,d in dims.items():
        candidates=[d]+[n for n in ds.variables if n.lower() in aliases[a]]
        var=next((ds.variables[n] for n in candidates if n in ds.variables and ds.variables[n].dimensions==(d,)),None)
        coords[a]=np.arange(len(ds.dimensions[d]),dtype=float) if var is None else np.asarray(var[:],dtype=float)
    names=next(n for n in [('u','v','w'),('velocity_x','velocity_y','velocity_z'),('Component1','Component2','Component3')] if all(k in ds.variables for k in n))
    for index in range(119,131):
        fields=[];h=hashlib.sha256()
        for name,outname in zip(names,['u','v','w']):
            var=ds.variables[name];spatial=[d for d in var.dimensions if d!=dims['t']]
            data=np.ma.transpose(var[tuple(index if d==dims['t'] else slice(None) for d in var.dimensions)],
                                 [spatial.index(dims[a]) for a in 'zyx'])
            fields.append(data);h.update(np.asarray(data.filled(np.nan),dtype=np.float32).tobytes())
            dest.variables[outname][index]=data[:,::3,::7]
        mean,count=volume_mean(np.ma.stack(fields,-1),[coords[a] for a in 'zyx'])
        info['times'].append(float(coords['t'][index]));info['velocities'].append(mean.tolist())
        info['missing_nodes_per_frame'].append(count);info['source_frame_sha256'].append(h.hexdigest())
        info['stored_frame_indices'].append(index)
        print(json.dumps({'index':index,'time':info['times'][-1],'mean':mean.tolist()}),flush=True)
    dest.extraction_rule='Original indices105..130 only; xyz strides7,3,1; all other frames missing'
info['window_file_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
info['previous_observer_sha256']=hashlib.sha256((previous/'observer.json').read_bytes()).hexdigest()
(output/'observer.json').write_text(json.dumps(info,indent=2))
print('Window bytes',target.stat().st_size,flush=True)
