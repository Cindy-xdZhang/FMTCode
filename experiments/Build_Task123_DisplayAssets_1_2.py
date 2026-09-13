"""Build reference meshes and Mode-a paths from local original flow fields.

No predictions or trained models are consumed. Reuse the historical geometry
transforms and IVD-only seed-selection functions, with a NumPy/SciPy RK4 display
integrator. All paths are explanatory display samples, not new model inputs.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/task123_plotdeps'))
sys.path.insert(0,str(ROOT/'experiments'))
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import Visualize_Task123_PaperTriptychs_1_1 as base

ASSETS = ROOT/'outputs/Other_Task123_PaperTriptychs_1.2/display_assets'
NAMES = ['cylinder3d','halfcylinderRe640','halfcylinderRe6400','tangaroa','boeing747','deltaWing_LBM']
LEGACY = ROOT/'experiments/Visualize_Task1_3D_PaperCandidates.py'


def legacy_helpers():
    names={'_source_coordinate_axes','_lattice_to_physical','_read_binary_stl_triangles',
           '_vertex_cluster_triangles','_cpp_integer_ratio','_load_simulation_geometry',
           '_blue_noise_like_indices','_stratified_blue_noise_indices'}
    tree=ast.parse(LEGACY.read_text())
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    scope={'np':np,'nc':nc,'Path':Path,'struct':struct,'DISPLAY_BLUE_NOISE_SEED':7068,
           'BOEING_STL_PATH':Path('C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/FluidX3DSTLAssets/stl/Boeing747/files/techtris_airplane.stl')}
    exec(compile(selected,str(LEGACY),'exec'),scope)
    return scope


def display_paths(meta,seeds,labels,helpers,aircraft):
    requested_steps=192 if aircraft else 96
    selected,npos,nneg=helpers['_stratified_blue_noise_indices'](seeds,labels,480 if aircraft else 240,.7)
    with nc.Dataset(meta['source_path']) as ds:
        aliases={'x':{'x','xdim'},'y':{'y','ydim'},'z':{'z','zdim'},'t':{'t','time','tdim'}}
        dims={a:next(n for n in ds.dimensions if n.lower() in aliases[a]) for a in 'xyzt'}
        sizes={a:len(ds.dimensions[dims[a]]) for a in 'xyzt'}
        start=meta['source_start_index']
        steps=min(requested_steps,4*(sizes['t']-start-2))
        assert steps>=48
        frames=int(np.ceil(.25*steps))+2
        slices={'t':slice(start,start+frames),**{a:slice(0,sizes[a],meta['spatial_strides'][a]) for a in 'xyz'}}
        canonical=[dims[a] for a in 'tzyx']
        names=next(v for v in [('u','v','w'),('velocity_x','velocity_y','velocity_z'),('Component1','Component2','Component3')] if all(n in ds.variables for n in v))
        arrays=[]
        for name in names:
            var=ds.variables[name]
            ix=tuple(slices['tzyx'[canonical.index(dim)]] for dim in var.dimensions)
            a=np.ma.asarray(var[ix]).filled(0)
            arrays.append(np.transpose(np.asarray(a,np.float32),[var.dimensions.index(dim) for dim in canonical]))
        values=np.stack(arrays,axis=-1)
        coords={}
        for a in 'xyzt':
            candidates=[dims[a]]+[n for n in ds.variables if n.lower() in aliases[a]]
            var=next((ds.variables[n] for n in candidates if n in ds.variables and ds.variables[n].dimensions==(dims[a],)),None)
            full=np.arange(sizes[a]) if var is None else np.asarray(var[:])
            coords[a]=full[slices[a]]
    bounds=np.asarray([[coords[a][0] for a in 'xyz'],[coords[a][-1] for a in 'xyz']],np.float32)
    spatial=[np.linspace(bounds[0,i],bounds[1,i],values.shape[3-i]) for i in range(3)]
    times=np.linspace(0,float(coords['t'][-1]-coords['t'][0]),frames)
    dt=float(times[1])*.25
    interpolate=RegularGridInterpolator((times,*spatial[::-1]),values,bounds_error=False,fill_value=np.nan)
    paths=np.full((len(selected),steps+1,4),np.nan,np.float32)
    points=np.asarray(seeds[selected],np.float64)
    paths[:,0,:3]=points;paths[:,0,3]=0
    active=np.ones(len(points),bool)
    lengths=np.ones(len(points),np.int32)
    def velocity(x,t):
        return interpolate(np.column_stack((np.full(len(x),t),x[:,[2,1,0]])))
    for k in range(steps):
        ids=np.flatnonzero(active)
        if not len(ids):break
        x=points[ids];t=k*dt
        k1=velocity(x,t);k2=velocity(x+.5*dt*k1,t+.5*dt)
        k3=velocity(x+.5*dt*k2,t+.5*dt);k4=velocity(x+dt*k3,t+dt)
        next_x=x+dt*(k1+2*k2+2*k3+k4)/6
        valid=np.isfinite(next_x).all(axis=1)&(next_x>=bounds[0]).all(axis=1)&(next_x<=bounds[1]).all(axis=1)
        active[ids[~valid]]=False
        ids=ids[valid];points[ids]=next_x[valid]
        paths[ids,k+1,:3]=points[ids];paths[ids,k+1,3]=(k+1)*dt;lengths[ids]+=1
    return paths,lengths,{'selected_indices':selected.tolist(),'count':len(selected),'vortex_count':npos,
          'background_count':nneg,'requested_steps':requested_steps,'steps':steps,'dt':dt,
          'duration':dt*steps,'minimum_points':int(lengths.min()),'maximum_points':int(lengths.max()),
          'integrator':'RK4, linear interpolation of original time-dependent velocity; terminate on first out-of-domain stage',
          'sampling':'historical IVD-only stratified deterministic maximin, seed7068; target70% vortex',
          'legacy_helper_sha256':hashlib.sha256(LEGACY.read_bytes()).hexdigest()}


def build(dataset):
    ASSETS.mkdir(parents=True,exist_ok=True)
    helpers=legacy_helpers()
    for task in ['task1','task23']:
        target=ASSETS/f'{dataset}_{task}.npz'
        if target.exists():
            print('EXISTS',target,flush=True);continue
        if dataset=='boeing747':
            folder=ROOT/'outputs/mainExp_Task123NewFlows_1.1'/('confirmation_cache' if task=='task1' else 'development_cache')/dataset
        else:
            folder=ROOT/('outputs/mainExp_Task3Universality_2.2/confirmation_cache' if task=='task1' else 'outputs/Verify_Task2Universality_1.1/cache')/dataset
        cache=sorted(folder.glob('slice_*.npz'))[0 if task=='task1' else 8]
        with np.load(cache) as data:
            seeds=data['seeds'];labels=data['reference'].astype(bool);meta=json.loads(str(data['metadata_json']))
        vertices,faces,bounds,audit=base.load_reference(meta,seeds,labels,ASSETS/'reference_meshes'/f'{dataset}_{task}.npz')
        paths=np.empty((0,0,4),np.float32);lengths=np.empty(0,np.int32);path_meta={}
        geometry=None
        if task=='task1':
            paths,lengths,path_meta=display_paths(meta,seeds,labels,helpers,dataset in ['boeing747','deltaWing_LBM'])
            kind={'boeing747':'boeing747_stl','deltaWing_LBM':'delta_wing_triangle'}.get(dataset)
            geometry=helpers['_load_simulation_geometry']({'geometry':kind},meta['source_path'])
        info={'metadata':meta,'surface_audit':audit,'paths':path_meta,
              'geometry':geometry['metadata'] if geometry else None,
              'geometry_note':'Verified original simulation geometry' if geometry else 'No validated geometry in historical flow specification; no invented obstacle',
              'cache_sha256':base.sha(cache),'cache_path':str(cache),'builder_sha256':base.sha(__file__)}
        triangles=geometry['triangles'] if geometry else np.empty((0,3,3))
        np.savez_compressed(target,seeds=seeds,reference=labels,vertices=vertices,faces=faces,bounds=bounds,
                            paths=paths,lengths=lengths,geometry=triangles,info_json=json.dumps(info))
        print('BUILT',dataset,task,'MB',round(target.stat().st_size/1e6,2),'paths',len(paths),'geometry',len(triangles),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=NAMES)
    for ds in p.parse_args().datasets:build(ds)
