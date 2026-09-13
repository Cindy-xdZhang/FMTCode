"""Export only late Re6400 frames at the frozen 96-grid sampling resolution."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import netCDF4 as nc
from FMT_Utils.NetCDF_window_3D import _axis_dimension, _coordinate

root=Path(__file__).resolve().parents[1]
source=Path('C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D/halfcylinderRe6400.nc')
output=root/'outputs/Verify_AIVDLongtime_1.1/source';output.mkdir(parents=True,exist_ok=True)
target=output/'halfcylinderRe6400_late96.nc'
if target.exists():raise FileExistsError(target)
with nc.Dataset(source) as src,nc.Dataset(target,'w') as dst:
    dims={a:_axis_dimension(src,a) for a in 'tzyx'}
    indices={a:np.arange(len(src.dimensions[dims[a]]))[::max(1,int(np.ceil(len(src.dimensions[dims[a]])/96)))] for a in 'zyx'}
    indices['t']=np.arange(len(src.dimensions[dims['t']]))
    times=_coordinate(src,dims['t'],indices['t'])
    assert len(times)>=150 and abs(times[75]-7.5)<1e-6
    for a in 'tzyx':
        dst.createDimension(a,len(indices[a]));dst.createVariable(a,'f8',(a,))[:]=_coordinate(src,dims[a],indices[a])
    names=next(v for v in [('u','v','w'),('velocity_x','velocity_y','velocity_z'),('Component1','Component2','Component3')] if all(n in src.variables for n in v))
    outputs=[dst.createVariable(a,'f4',tuple('tzyx'),zlib=True,complevel=1,fill_value=np.nan) for a in 'uvw']
    hashes=[]
    for i in range(75,150):
        h=hashlib.sha256()
        for name,var in zip(names,outputs):
            original=src.variables[name];remaining=[d for d in original.dimensions if d!=dims['t']]
            selectors=tuple(i if d==dims['t'] else indices[next(a for a in 'zyx' if dims[a]==d)] for d in original.dimensions)
            data=np.ma.asarray(original[selectors]).filled(0.)
            data=np.transpose(np.asarray(data,np.float32),[remaining.index(dims[a]) for a in 'zyx'])
            var[i]=data;h.update(data.tobytes())
        hashes.append({'index':i,'sha256':h.hexdigest()})
        if i%10==0:print('exported',i,flush=True)
    dst.extraction='Original frames75..149; original coordinate indices retained; exact frozen ceil(size/96) spatial strides; early frames missing'
digest=hashlib.sha256(target.read_bytes()).hexdigest()
(output/'source_export.json').write_text(json.dumps({'source':str(source),'source_bytes':source.stat().st_size,
    'export':target.name,'sha256':digest,'frame_hashes':hashes,'spatial_indices':{a:indices[a].tolist() for a in 'zyx'}},indent=2))
print(target,target.stat().st_size,digest,flush=True)
