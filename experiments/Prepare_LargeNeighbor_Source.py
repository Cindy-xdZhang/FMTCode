"""Export exact pre-strided source frames; never compute local training metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import netCDF4 as nc
import numpy as np
from FMT_Utils.NetCDF_window_3D import _axis_dimension, _coordinate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True)
    parser.add_argument('--config',default='config/Verify_LargeNeighbor_1.1.json')
    args=parser.parse_args();spec=json.loads(Path(args.config).read_text())
    entry=spec['source_exports'][args.dataset];source=Path(entry['local_path'])
    out=Path('outputs')/spec['experiment']/'source';out.mkdir(parents=True,exist_ok=True)
    target=out/(args.dataset+'_windows96.nc')
    if target.exists():raise FileExistsError(target)
    selected=sorted({i for w in entry['windows'] for i in range(w['index'],w['index']+w['frame_count'])})
    hashes=[]
    with nc.Dataset(source) as src,nc.Dataset(target,'w') as dst:
        dims={a:_axis_dimension(src,a) for a in 'tzyx'}
        indices={a:np.arange(len(src.dimensions[dims[a]]))[::max(1,int(np.ceil(len(src.dimensions[dims[a]])/96)))] for a in 'zyx'}
        indices['t']=np.arange(len(src.dimensions[dims['t']]))
        for a in 'tzyx':
            dst.createDimension(a,len(indices[a]));dst.createVariable(a,'f8',(a,))[:]=_coordinate(src,dims[a],indices[a])
        names=next(v for v in [('u','v','w'),('velocity_x','velocity_y','velocity_z'),('Component1','Component2','Component3')] if all(n in src.variables for n in v))
        outputs=[dst.createVariable(a,'f4',tuple('tzyx'),zlib=True,complevel=1,fill_value=np.nan) for a in 'uvw']
        for ordinal,i in enumerate(selected):
            h=hashlib.sha256()
            for name,var in zip(names,outputs):
                original=src.variables[name];remaining=[d for d in original.dimensions if d!=dims['t']]
                selectors=tuple(i if d==dims['t'] else indices[next(a for a in 'zyx' if dims[a]==d)] for d in original.dimensions)
                data=np.ma.asarray(original[selectors]).filled(0.)
                data=np.transpose(np.asarray(data,np.float32),[remaining.index(dims[a]) for a in 'zyx'])
                var[i]=data;h.update(data.tobytes())
            hashes.append({'index':i,'sha256':h.hexdigest()})
            if ordinal%20==0:print(args.dataset,'exported',ordinal+1,'/',len(selected),flush=True)
        dst.selected_original_indices_json=json.dumps(selected)
        dst.extraction='Exact original float32 values with ceil(size/96) spatial strides; original time indices preserved; other frames missing.'
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    meta={'dataset':args.dataset,'source':str(source),'source_bytes':source.stat().st_size,
          'export':target.name,'sha256':digest,'bytes':target.stat().st_size,'frames':hashes,
          'spatial_indices':{a:indices[a].tolist() for a in 'zyx'}}
    (out/(args.dataset+'_export.json')).write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k not in ['frames','spatial_indices']}),flush=True)


if __name__=='__main__':main()
