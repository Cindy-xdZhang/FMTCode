"""Build optional, reproducible 21-streamline bundles from canonical Task6 fields."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, sys, time, traceback, shutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from numba import set_num_threads
from FMT_Utils.Task6IcosaBundles_1_1 import Candidate, directions, frame_seed, integrate, integrate_filtered
from FMT_Utils.Task6Sampling_2_1 import SegmentQuery
from experiments.Task6_FieldStore import read, sha, atomic

CONFIG = ROOT/'config/mainExp_Task6_IcosaBundles_1.1.json'
EVIDENCE = ROOT/'outputs/Verify_Task6_IcosaBundles_1.1'


def load_frame(config, record):
    root = Path(config['dataset_root']); folder = root/record['folder']
    manifest = read(root/'dataset.json')
    # Geometry may be edited later; fields must remain the exact input snapshot.
    for name in ['relative_velocity.npy', 'coordinates.npz', 'ivd.npy']:
        expected = manifest['files_sha256'][record['folder']+'/'+name]
        if sha(folder/name) != expected: raise ValueError('Input checksum mismatch: '+name)
    with np.load(folder/'coordinates.npz') as z: axes = [z[a].astype(float) for a in 'xyz']
    h = min((a[-1]-a[0])/(len(a)-1) for a in axes)
    if not np.isclose(h, record['h'], rtol=1e-12): raise ValueError('Physical grid h mismatch')
    field = np.load(folder/'relative_velocity.npy', mmap_mode='r')
    ivd = np.load(folder/'ivd.npy', mmap_mode='r')
    offsets = directions(config['geometry'])
    fraction=config['candidate'].get('per_flow_fraction',{}).get(record['flow'],config['candidate']['threshold_fraction_of_maximum'])
    trim=config['candidate'].get('per_flow_trim_xyz',{}).get(record['flow'])
    candidate_ivd,candidate_axes=ivd,axes
    if trim:
        from FMT_Utils.Task6CandidateRegion_1_1 import crop_candidate_region
        candidate_ivd,candidate_axes=crop_candidate_region(ivd,axes,trim)
    sampler = Candidate(candidate_ivd, candidate_axes, h*offsets, frame_seed(record['key'], config['seed']),fraction)
    return field, ivd, axes, h, offsets, sampler


def pilot_one(config, record):
    set_num_threads(config['threads_per_worker'])
    start = time.monotonic(); key = record['key']
    try:
        field, ivd, axes, h, offsets, sampler = load_frame(config, record)
        centers = sampler.draw(32)
        # A separate warmup removes JIT startup from throughput estimates.
        integrate(field, centers[:1], axes, h, offsets)
        clock = time.monotonic(); result = integrate(field, centers, axes, h, offsets)
        seconds = time.monotonic()-clock
        good = int(result['valid'].sum())
        row = dict(key=key,passed=good>0,centers=32,valid=good,seconds=seconds,
                   seconds_per_accepted=seconds/good if good else None,
                   candidate_grid_nodes=int((ivd>sampler.threshold).sum()),threshold=sampler.threshold,
                   proposed=sampler.proposed,candidate_cells=len(sampler.low),h=h,
                   maximum_retained_error=float(result['errors'][result['valid']].max()) if good else None)
        if not good: row['error']='No fully valid 21-line bundle in the 32-center pilot; investigate before bulk build'
    except Exception as e:
        row = dict(key=key,passed=False,error=str(e),traceback=traceback.format_exc())
    row['elapsed_seconds']=time.monotonic()-start
    atomic(EVIDENCE/'pilots'/f"{record['flow']}_{record['index']:03d}.json",row)
    return row


def fingerprint(config,flow):
    scientific={k:config[k] for k in ['version','geometry','centers_per_frame','neighbors','seed','integration']}
    scientific['candidate']={k:v for k,v in config['candidate'].items() if k not in ('per_flow_fraction','per_flow_trim_xyz')}
    scientific['candidate']['threshold_fraction_of_maximum']=config['candidate'].get('per_flow_fraction',{}).get(flow,config['candidate']['threshold_fraction_of_maximum'])
    paths=['FMT_Utils/Task6IcosaBundles_1_1.py','FMT_Utils/Task6AdaptiveStreamlines_3D.py',
           'FMT_Utils/Task6Streamlines_3D.py','FMT_Utils/Task6Refinement_1_3.py']
    trim=config['candidate'].get('per_flow_trim_xyz',{}).get(flow)
    if trim:
        scientific['candidate']['trim_xyz']=trim;paths.append('FMT_Utils/Task6CandidateRegion_1_1.py')
    scientific['sources']={p:sha(ROOT/p) for p in paths}
    return hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()


def core_labels(root, record, centers):
    folder=root/record['folder']; manifest_sha=sha(root/'dataset.json');manifest=read(root/'dataset.json')
    files={n:sha(folder/n) for n in ['corelines.vtp','core_points.npy','core_offsets.npy']}
    if any(files[n]!=manifest['files_sha256'][record['folder']+'/'+n] for n in files):
        raise RuntimeError('Coreline publication in progress; retry after publication')
    core_sha=files['corelines.vtp'];points=np.load(folder/'core_points.npy'); offsets=np.load(folder/'core_offsets.npy')
    cores=[points[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    labels=(SegmentQuery(cores).distance(centers,record['h'],workers=1)<record['h']).astype(np.uint8)
    if manifest_sha!=sha(root/'dataset.json') or any(files[n]!=sha(folder/n) for n in files):
        raise RuntimeError('Corelines changed while deriving center labels; retry')
    return labels,core_sha


def build_one(config,record):
    """Resumable per-frame output; no writes to canonical field or annotation files."""
    set_num_threads(config['threads_per_worker']); root=Path(config['dataset_root'])
    dest=Path(config['output_root'])/'frames'/record['flow']/f"frame_{record['index']:03d}"
    dest.mkdir(parents=True,exist_ok=True); signature=fingerprint(config,record['flow'])
    if (dest/'complete.json').exists():
        previous=read(dest/'complete.json')
        if previous['fingerprint']!=signature: raise ValueError('Existing bundle uses a different protocol')
        for n,h in previous['files'].items():
            if sha(dest/n)!=h: raise ValueError('Completed bundle checksum mismatch: '+n)
        return dict(key=record['key'],state='COMPLETE',reused=True)
    field,ivd,axes,h,d,sampler=load_frame(config,record)
    count=config['centers_per_frame']; k=len(d)
    if k!=config['neighbors']+1: raise ValueError('Neighbor count mismatch')
    shapes={'curves':((count,k,65,3),np.float32),'centers':((count,3),np.float64),
            'half_lengths':((count,k,2),np.float32),'termination':((count,k,2),np.int8),
            'refinement_errors':((count,k),np.float32)}
    progress_path=dest/'progress.json'
    if progress_path.exists():
        state=read(progress_path)
        if state['fingerprint']!=signature: raise ValueError('Partial bundle uses a different protocol')
        sampler.rng.bit_generator.state=state['rng_state']
        sampler.proposed=state['candidate_proposals'];sampler.candidate_accepted=state['candidate_accepted']
        arrays={n:np.load(dest/(n+'.npy'),mmap_mode='r+') for n in shapes}
        for n,(shape,dtype) in shapes.items():
            if arrays[n].shape!=shape or arrays[n].dtype!=dtype: raise ValueError('Resume shape mismatch')
    else:
        expected_bytes=sum(int(np.prod(shape))*np.dtype(dtype).itemsize for shape,dtype in shapes.values())
        if shutil.disk_usage(dest).free<expected_bytes+5*2**30: raise RuntimeError('Less than 5 GiB free-space reserve')
        # Profile rejection order only; it never changes acceptance or stored positions.
        probe=sampler.draw(32); full=integrate(field,probe,axes,h,d)
        bad=(full['half_lengths'].sum(-1)<=h)|(full['errors']>h/20)|(full['termination']>=3).any(-1)
        order=np.argsort(-bad.sum(0),kind='stable').tolist()
        arrays={n:np.lib.format.open_memmap(dest/(n+'.npy'),mode='w+',dtype=dtype,shape=shape)
                for n,(shape,dtype) in shapes.items()}
        state=dict(key=record['key'],fingerprint=signature,done=0,total=count,integrated_proposals=0,
                   validation_order=order,compute_seconds=0.,source_files={n:sha(root/record['folder']/n)
                   for n in ['relative_velocity.npy','coordinates.npz','ivd.npy']})
    clock=time.monotonic(); previous_compute=state['compute_seconds']; empty_batches=0
    while state['done']<count:
        # Candidate groups remain small, avoiding simultaneous giant temporary arrays.
        batch=config['chunk_centers']; centers=sampler.draw(batch)
        result=integrate_filtered(field,centers,axes,h,d,state['validation_order'])
        selected=np.flatnonzero(result['valid'])[:count-state['done']]
        begin=state['done']; end=begin+len(selected)
        for name,key in [('curves','curves'),('half_lengths','half_lengths'),('termination','termination'),('refinement_errors','errors')]:
            arrays[name][begin:end]=result[key][selected]
        arrays['centers'][begin:end]=centers[selected]
        state['done']=end;state['integrated_proposals']+=batch
        empty_batches=empty_batches+1 if not len(selected) else 0
        for a in arrays.values():a.flush()
        state.update(rng_state=sampler.rng.bit_generator.state,candidate_proposals=sampler.proposed,
                     candidate_accepted=sampler.candidate_accepted,compute_seconds=previous_compute+time.monotonic()-clock)
        atomic(progress_path,state)
        if empty_batches>=128:
            raise RuntimeError('8192 consecutive proposed bundles had no valid result; preserve partial output and investigate')
    # Full saved arrays are independently checked before the completed marker appears.
    centers=np.asarray(arrays['centers']);assert np.unique(centers,axis=0).shape==centers.shape
    values=sampler.interpolate(centers[:,::-1]);assert (values>sampler.threshold).all()
    lower=np.array([a[0] for a in axes]);upper=np.array([a[-1] for a in axes])
    assert np.all(centers[:,None]+h*d>=lower) and np.all(centers[:,None]+h*d<=upper)
    for begin in range(0,count,256):
        curves=np.asarray(arrays['curves'][begin:begin+256]);assert np.isfinite(curves).all()
        assert np.all(curves>=lower-1e-5) and np.all(curves<=upper+1e-5)
    assert (arrays['half_lengths'].sum(-1)>h).all() and (arrays['termination']<3).all()
    assert np.isfinite(arrays['refinement_errors']).all() and arrays['refinement_errors'].max()<=h/20
    from FMT_Utils.Task6AdaptiveStreamlines_3D import integrate_samples_adaptive
    spacing=np.array([(a[-1]-a[0])/(len(a)-1) for a in axes]);ids=np.linspace(0,count-1,16).astype(int)
    seeds=np.ascontiguousarray((centers[ids,None]+h*d).reshape(-1,3))
    strict,_,termination=integrate_samples_adaptive(field,seeds,lower,spacing,.25,h/64,h*1e-10,65)
    errors=np.linalg.norm(strict-arrays['curves'][ids].reshape(-1,65,3),axis=-1).max(1).reshape(len(ids),k)
    failed=(~np.isfinite(strict).all((1,2)).reshape(len(ids),k) | (errors>h/20) |
            (termination.reshape(len(ids),k,2)>=3).any(-1)).any(-1)
    # Reject the entire sample when the independent check fails, just as for
    # earlier refinement levels. Never relax tolerances or patch one neighbor.
    for slot in np.flatnonzero(failed):
        sample_index=int(ids[slot]); replacement=None
        for attempt in range(128):
            proposals=sampler.draw(config['chunk_centers'])
            result=integrate_filtered(field,proposals,axes,h,d,state['validation_order'])
            state['integrated_proposals']+=len(proposals)
            for candidate_index in np.flatnonzero(result['valid']):
                proposed=proposals[candidate_index]
                if np.any(np.all(arrays['centers']==proposed,axis=1)):continue
                check_seeds=np.ascontiguousarray(proposed[None]+h*d)
                checked,_,reasons=integrate_samples_adaptive(field,check_seeds,lower,spacing,.25,h/64,h*1e-10,65)
                checked_error=np.linalg.norm(checked-result['curves'][candidate_index],axis=-1).max(1)
                if (not np.isfinite(checked).all() or (reasons>=3).any() or
                        not np.isfinite(checked_error).all() or checked_error.max()>h/20):continue
                replacement=(candidate_index,checked_error);break
            if replacement is not None:break
        if replacement is None:raise RuntimeError('Independent-check replacement budget exhausted; preserve partial output')
        candidate_index,checked_error=replacement
        for name,key in [('curves','curves'),('half_lengths','half_lengths'),('termination','termination'),('refinement_errors','errors')]:
            arrays[name][sample_index]=result[key][candidate_index]
        arrays['centers'][sample_index]=proposals[candidate_index]
        for a in arrays.values():a.flush()
        state.setdefault('independent_check_replacements',[]).append(dict(sample_index=sample_index,
            rejected_max_error=float(errors[slot].max()),replacement_max_error=float(checked_error.max())))
        errors[slot]=checked_error
        state.update(rng_state=sampler.rng.bit_generator.state,candidate_proposals=sampler.proposed,
                     candidate_accepted=sampler.candidate_accepted,compute_seconds=previous_compute+time.monotonic()-clock)
        atomic(progress_path,state)
    difference=float(errors.max())
    current=next(r for r in read(root/'dataset.json')['frames'] if r['key']==record['key'])
    labels,core_sha=core_labels(root,current,centers);np.save(dest/'labels.npy',labels)
    metadata=dict(key=record['key'],count=count,streamlines_per_sample=k,points_per_streamline=65,
                  coreline_sha256=core_sha,label_rule='center distance to current continuous coreline segments < h',
                  positive=int(labels.sum()),h=h,threshold=sampler.threshold,threshold_rule='IVD(v) > fraction * maximum',
                  threshold_fraction=config['candidate'].get('per_flow_fraction',{}).get(record['flow'],config['candidate']['threshold_fraction_of_maximum']),
                  candidate_bounds_xyz=[[float(a[0]),float(a[-1])] for a in sampler.axes],
                  candidate_applies_to_center_only=True,all_neighbor_seeds_inside_domain=True,
                  geometry=config['geometry'],integration=config['integration'],source_files=state['source_files'],
                  numerical_rejections=state['integrated_proposals']-count,independent_max_error=difference,
                  independent_centers=ids.tolist(),independent_check_replacements=state.get('independent_check_replacements',[]),
                  unit_offsets=d.tolist(),compute_seconds=state['compute_seconds'])
    atomic(dest/'metadata.json',metadata)
    files={n+'.npy':sha(dest/(n+'.npy')) for n in [*shapes,'labels']};files['metadata.json']=sha(dest/'metadata.json')
    atomic(dest/'complete.json',dict(passed=True,key=record['key'],fingerprint=signature,files=files))
    return dict(key=record['key'],state='COMPLETE',samples=count,positive=int(labels.sum()),seconds=state['compute_seconds'])


def guarded_build(config,record):
    try:return build_one(config,record)
    except Exception as e:
        row=dict(key=record['key'],state='FAILED',error=str(e),traceback=traceback.format_exc())
        atomic(EVIDENCE/'failures'/f"{record['flow']}_{record['index']:03d}.json",row)
        return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['pilot','build'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--keys-file',type=Path)
    parser.add_argument('--run-name')
    args=parser.parse_args(); config=read(args.config)
    run_name=args.run_name or args.phase
    if Path(run_name).name!=run_name:raise ValueError('Run name must be a filename component')
    assert config['geometry_user_confirmed'] and config['centers_per_frame']>0
    records=read(Path(config['dataset_root'])/'dataset.json')['frames']
    if args.keys_file:
        keys=read(args.keys_file); records=[r for r in records if r['key'] in keys]
        if len(records)!=len(keys):raise ValueError('Unknown or duplicated requested frame keys')
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    # Prevent nested BLAS/OpenMP pools from consuming all machine threads.
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS=str(config['threads_per_worker']))
    rows=[];start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=config['workers']) as pool:
        function=pilot_one if args.phase=='pilot' else guarded_build
        futures={pool.submit(function,config,r):r['key'] for r in records}
        for future in concurrent.futures.as_completed(futures):
            row=future.result();rows.append(row)
            atomic(EVIDENCE/(run_name+'_progress.json'),dict(done=len(rows),total=len(records),rows=rows))
            print(json.dumps(row),flush=True)
    atomic(EVIDENCE/(run_name+'.json'),dict(passed=all(r.get('passed',r.get('state')=='COMPLETE') for r in rows),rows=rows,
                                    elapsed_seconds=time.monotonic()-start))


if __name__=='__main__': main()
