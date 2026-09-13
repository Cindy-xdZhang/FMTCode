"""Construct new shells on frozen seeds and keep a shared complete cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
import torch
from FMT_Utils.GeometricControls_3D import validate_cache,array_hash
from FMT_Utils.FMT_3D_pipeline import generate_seeding_grid_3d,integrate_cross_primitives_3d
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from FMT_Utils.LargeNeighbor_3D import integrate_shells,large_neighbor_fmt
from FMT_Utils.FMTAllV2_3D import fmt_all_v2
from FMT_Utils.Task12Data_3D import feature_matrix
from experiments.Build_Task5_Multiscale_Cache import _channel_fields
from experiments.Run_Task135_GeometricControls import sources,sha,write_json


def ordinals(spec,task,dataset,role):
    return spec.get('dataset_splits',{}).get(dataset,{}).get(task,{}).get(role,spec[task.lower()][role])


def required_sources(spec,dataset):
    result={}
    for task in spec['tasks']:
        for role in ('train','validation','confirmation'):
            phase='confirmation' if task=='Task1' and role=='confirmation' else 'development'
            directory,_,_,expected=sources(spec,'Task1',dataset,phase)
            files=sorted(directory.glob('slice_*.npz'));assert len(files)==expected
            for ordinal in ordinals(spec,task,dataset,role):result[(phase,ordinal)]=files[ordinal]
    return sorted(result.items())


def cache_file(spec,dataset,phase,ordinal):
    return Path(spec['output_root'])/'cache'/dataset/phase/f'slice_{ordinal:02d}.npz'


def build(spec,config_path,dataset,smoke=False):
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    started=time.time();pairs=required_sources(spec,dataset)
    if smoke:
        first=ordinals(spec,'Task1',dataset,'train')[0]
        pairs=[item for item in pairs if item[0]==('development',first)]
    descriptors=[]
    for (phase,ordinal),path in pairs:
        with np.load(path,allow_pickle=False) as archive:
            old=validate_cache(archive);mask=np.asarray(archive['valid_mask'],bool)
            stored_seeds=np.asarray(archive['seeds'])
        descriptors.append((phase,ordinal,path,old,mask,stored_seeds))
    source_path=Path(spec['source_fields'][dataset]);assert source_path.is_file()
    indices=[int(item[3]['metadata']['source_start_index']) for item in descriptors]
    fields=_channel_fields(source_path,spec,indices,14) if dataset=='channel' else None
    audits=[]
    for phase,ordinal,path,old,mask,stored_seeds in descriptors:
        target=cache_file(spec,dataset,phase,ordinal)
        target.parent.mkdir(parents=True,exist_ok=True)
        oldmeta=old['metadata'];source_index=int(oldmeta['source_start_index'])
        if fields is not None:field,meta=next(fields)
        else:field,meta=load_netcdf_window_3d(source_path,source_index,14,96)
        assert np.isfinite(field.field).all()
        assert meta['loaded_shape_TZYXC']==oldmeta['loaded_shape_TZYXC']
        assert abs(meta['source_time']-oldmeta['source_time'])<1e-6
        assert abs(meta['source_time_step']-oldmeta['source_time_step'])<1e-7
        if dataset in spec['cylinder_datasets']:assert meta['source_time']>=7.5
        r=float(np.min(field.gridInterval[field.gridInterval>0]))*.5
        original_seeds,_=generate_seeding_grid_3d(field,[16,16,16],.08,r,grid_phase=oldmeta.get('seed_grid_phase'))
        assert len(original_seeds)==len(mask)
        seeds=original_seeds[mask]
        np.testing.assert_array_equal(seeds.astype(np.float32),stored_seeds)
        dt=float(field.timeInterval)*.25
        # Verify the source window and integrator against actual old geometry.
        check=np.arange(min(64,len(seeds)))
        cross,crossmask,_=integrate_cross_primitives_3d(field,seeds[check],0.,dt,48,32,r)
        assert crossmask.all()
        cross_local=(cross[...,:3]-cross[:,:1,:1,:3]).astype(np.float32)
        error=float(np.max(np.abs(cross_local-old['raw'][check])))
        tolerance=max(1e-5*r,2e-6)
        if error>tolerance:raise ValueError(f'{dataset}/{phase}/{ordinal}: source replay error {error} > {tolerance}')
        if target.exists():
            meta_saved=json.loads(target.with_suffix('.json').read_text())
            assert meta_saved['config_sha256']==sha(config_path) and meta_saved['old_cache_sha256']==sha(path)
            assert meta_saved['cache_sha256']==sha(target)
            audits.append(meta_saved);continue
        primitives,retained,lengths=integrate_shells(field,seeds,dt,r,chunk_size=spec['sampling']['chunk_size'])
        if retained.sum()<100:raise ValueError('Too few common complete primitives')
        raw25=(primitives[...,:3]-primitives[:,:1,:1,:3]).astype(np.float32)
        raw7=old['raw'][retained];reference=old['labels'][retained]
        center_error=float(np.max(np.abs(raw25[:,0]-raw7[:,0])))
        if center_error>tolerance:raise ValueError(f'Center material correspondence error {center_error}')
        times=old['times'][retained]
        np.testing.assert_allclose(primitives[:,0,:,3],times,rtol=1e-5,atol=2e-6)
        adapter={'raw':raw7.reshape(len(raw7),-1),'fmt':old['cached_fmt'][retained],'features':{}}
        values={'old_fmt':feature_matrix(adapter,'fmt_all+kin4','cpu'),
                'short':feature_matrix(adapter,'aivd1w3_dft','cpu'),
                'gram7':fmt_all_v2(raw7),'large':large_neighbor_fmt(raw25)}
        meta_saved={**oldmeta,'experiment':spec['experiment'],'dataset':dataset,'phase':phase,'ordinal':ordinal,
          'valid_primitives':int(retained.sum()),'old_valid_primitives':len(raw7)+int((~retained).sum()),
          'excluded_new_shell_boundary':int((~retained).sum()),'radius':r,'config_sha256':sha(config_path),
          'source_manifest_sha256':sha('SOURCE_MANIFEST.sha256'),'old_cache_path':str(path),'old_cache_sha256':sha(path),
          'source_field_path':str(source_path),'source_field_bytes':source_path.stat().st_size,
          'source_window_sha256':array_hash(field.field),'cross_replay_max_abs_error':error,
          'center_max_abs_error':center_error,'source_replay_tolerance':tolerance,
          'retained_indices_sha256':array_hash(np.flatnonzero(retained)),
          'labels_sha256':array_hash(reference),'time_sha256':array_hash(times),
          'job':os.environ.get('SLURM_JOB_ID'),'node':socket.gethostname(),'device':'CPU'}
        np.savez_compressed(target,raw7=raw7,raw25=raw25,times=times,labels=reference,
             retained_indices=np.flatnonzero(retained),seeds=seeds[retained].astype(np.float32),
             line_lengths=lengths,**values,metadata_json=np.asarray(json.dumps(meta_saved)))
        meta_saved['cache_sha256']=sha(target);meta_saved['end_time']=time.time()
        write_json(target.with_suffix('.json'),meta_saved);audits.append(meta_saved)
        print(f'BUILT {dataset}/{phase}/{ordinal}: {retained.sum()}/{len(retained)}, replay={error:g}',flush=True)
        del field,primitives,values
    write_json(Path(spec['output_root'])/'build_audits'/f'{dataset}{"_smoke" if smoke else ""}.json',
        {'status':'PASS','dataset':dataset,'config_sha256':sha(config_path),'slices':audits,
         'job':os.environ.get('SLURM_JOB_ID'),'node':socket.gethostname(),'start':started,'end':time.time()})


def load_records(spec,task,dataset,role):
    phase='confirmation' if task=='Task1' and role=='confirmation' else 'development'
    result=[]
    for ordinal in ordinals(spec,task,dataset,role):
        path=cache_file(spec,dataset,phase,ordinal)
        with np.load(path,allow_pickle=False) as archive:
            data={k:np.asarray(archive[k]) for k in ('raw7','raw25','times','labels','old_fmt','short','gram7','large')}
            meta=json.loads(str(archive['metadata_json']))
        assert meta['config_sha256']==sha(spec['_config_path'])
        n=len(data['labels']);assert data['raw7'].shape==(n,7,32,3) and data['raw25'].shape==(n,25,32,3)
        assert all(np.isfinite(a).all() for a in data.values()) and np.isin(data['labels'],[0,1]).all()
        data.update(raw=data['raw7'],large_raw=data['raw25'],path=str(path),ordinal=ordinal,metadata=meta,
          identity=meta['retained_indices_sha256']+':'+meta['labels_sha256'],
          certificate={'total':meta['old_valid_primitives'],'valid':n,'invalid':meta['excluded_new_shell_boundary'],
            'raw_sha256':array_hash(data['raw7']),'times_sha256':array_hash(data['times'])})
        result.append(data)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config/Verify_LargeNeighbor_1.1.json')
    parser.add_argument('--index',type=int);parser.add_argument('--dataset');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();spec=json.loads(Path(args.config).read_text())
    build(spec,args.config,args.dataset or spec['datasets'][args.index],args.smoke)
