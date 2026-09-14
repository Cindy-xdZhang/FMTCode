"""Physical-length vortex bundles with local spatial blocks and fixed classifiers."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments import Task4C_NearbySampling_4_13 as training
from experiments import Task4C_Multiscale_4_1 as physical
from experiments import Task4C_ManifoldMixup_4_4 as pipeline
from FMT_Utils.Task4C_Multiscale_4_1 import center_and_neighbors
from FMT_Utils.Task4C_PaperBundles_3_1 import normalize_view, cell_centers
from FMT_Utils.Task4C_LinePooling_4_3 import line_fmt, bundle_voxels
from FMT_Utils.Task4C_Bundles_1_1 import resample_line

DEFAULT_CONFIG = 'config/mainExp_Task4C_PhysicalLength_4.14.json'
SPLITS = ('train', 'validation', 'test')


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    if pipeline.sha(definition['base_config']) != definition['base_config_sha256']:
        raise ValueError('Frozen physical base configuration changed')
    base = json.loads(Path(definition['base_config']).read_text())
    spec = {key: base[key] for key in ('flows', 'input_root', 'labels', 'bundles')}
    spec.update(definition)
    for flow, bounds, length in (('channel', (.001,.005), .1), ('tbl', (.01,.05), 5.)):
        for item in spec['integration'][flow]:
            ds, n = item['ds'], item['maxiteration']
            if not bounds[0] <= ds <= bounds[1] or not isinstance(n,int) or n < 1:
                raise ValueError('Integration parameters violate user ranges')
            if not .8*length-1e-12 <= ds*n <= 1.2*length+1e-12:
                raise ValueError('Physical propagation is outside the declared length band')
    return spec


def identity(config):
    result = pipeline.identity(config)
    for name in ('experiments/Task4C_NearbySampling_4_13.py',
                 'experiments/Task4C_PhysicalLength_4_14.py',
                 'ibex_bash/task4c_physical_length_4p14.sh'):
        result['source_sha256'][name] = pipeline.sha(name)
    definition=json.loads(Path(config).read_text())
    result['source_sha256'][definition['base_config']]=pipeline.sha(definition['base_config'])
    return result


def scale_plan(spec, flow):
    return [dict(step,neighbor_grid_scale=radius) for radius in spec['sampling']['neighbor_grid_scales']
            for step in spec['integration'][flow]]


def spatial_blocks(cell_ids, shape, axes, head, spec):
    points=cell_centers(cell_ids,shape,axes);centered=points-points.mean(0)
    _,vectors=np.linalg.eigh(centered.T@centered);axis=vectors[:,-1]
    if axis[np.argmax(np.abs(axis))]<0:axis=-axis
    order=np.lexsort((cell_ids,centered@axis))
    roles=np.random.default_rng(np.random.SeedSequence([spec['sampling']['seed'],int(head),17])).permutation(
        spec['local_split']['block_roles'])
    result={split:[] for split in SPLITS}
    for ids,role in zip(np.array_split(np.asarray(cell_ids)[order],len(roles)),roles):result[role].extend(ids)
    return {key:np.asarray(value,dtype=np.int64) for key,value in result.items()}


def build_catalog(axes, heads, gt, spec):
    from scipy import ndimage
    from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt
    components,count=ndimage.label(heads)
    flat=np.flatnonzero(heads);order=np.argsort(components.ravel()[flat],kind='stable');flat=flat[order]
    labels,_=sample_gt(gt,cell_centers(flat,heads.shape,axes))
    regions=components.ravel()[flat];starts=np.r_[0,np.flatnonzero(np.diff(regions))+1,len(flat)]
    catalog=[];excluded=[]
    for first,stop in zip(starts[:-1],starts[1:]):
        head=int(regions[first]);ids=flat[first:stop];targets=labels[first:stop]
        if len(ids)<spec['bundles']['minimum_head_cells']:
            excluded.append(dict(head=head,reason='fewer_than_10_head_cells'));continue
        instances,counts=np.unique(targets[targets>=0],return_counts=True)
        majority=float(counts.max()/len(ids)) if len(counts) else 0.
        if len(counts) and majority<spec['labels']['positive_minimum_single_instance_fraction']:
            excluded.append(dict(head=head,reason='ambiguous_partial_GT_overlap'));continue
        catalog.append(dict(head_component=head,cell_ids=ids,head_cell_count=len(ids),
            label=int(len(counts)>0),instance=int(instances[counts.argmax()]) if len(counts) else -1,
            all_instances=instances.tolist(),dominant_instance_fraction=majority,
            split_cells=spatial_blocks(ids,heads.shape,axes,head,spec)))
    return components,catalog,dict(head_components=count,eligible_labeled_heads=len(catalog),excluded_heads=excluded,
        whole_head_GT_rule_unchanged=True,shared_heads_instances_and_native_support=True)


def trace_lines(grid, seeds, scale, spec):
    """RK45 with fixed physical step bounds and an explicit per-direction step cap."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.asarray(seeds,np.float64),deep=True))
    source=vtk.vtkPolyData();source.SetPoints(points)
    ds=float(scale['ds']);n=int(scale['maxiteration']);limit=ds*n
    halves=[];reasons={};half_arcs=np.zeros((len(seeds),2));half_steps=np.zeros((len(seeds),2),np.int32)
    for direction in range(2):
        tracer=vtk.vtkStreamTracer();tracer.SetInputData(grid);tracer.SetSourceData(source)
        tracer.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,'vorticity')
        tracer.SetIntegratorTypeToRungeKutta45();tracer.SetInterpolatorTypeToCellLocator()
        tracer.SetIntegrationStepUnit(vtk.vtkStreamTracer.LENGTH_UNIT)
        tracer.SetInitialIntegrationStep(ds);tracer.SetMinimumIntegrationStep(ds);tracer.SetMaximumIntegrationStep(ds)
        # A small internal guard prevents VTK from dropping the final endpoint
        # at exact propagation equality. Retained curves still have at most N steps.
        tracer.SetMaximumPropagation(limit+ds*spec['integration']['vtk_internal_propagation_guard_steps'])
        tracer.SetMaximumNumberOfSteps(n)
        tracer.SetMaximumError(ds*spec['integration']['maximum_error_over_ds'])
        tracer.SetTerminalSpeed(1e-12);tracer.SetComputeVorticity(False)
        if direction:tracer.SetIntegrationDirectionToForward()
        else:tracer.SetIntegrationDirectionToBackward()
        tracer.Update();output=tracer.GetOutput();curves={}
        if output.GetNumberOfCells():
            coords=vtk_to_numpy(output.GetPoints().GetData())
            ids=vtk_to_numpy(output.GetCellData().GetArray('SeedIds'))
            termination=vtk_to_numpy(output.GetCellData().GetArray('ReasonForTermination'))
            for j in range(output.GetNumberOfCells()):
                cell=output.GetCell(j)
                # VTK's documented extra step depends on propagation termination;
                # retain at most N segments and independently check actual arc length.
                curve=coords[[cell.GetPointId(k) for k in range(min(cell.GetNumberOfPoints(),n+1))]].copy()
                seed=int(ids[j]);curves[seed]=curve;half_steps[seed,direction]=len(curve)-1
                half_arcs[seed,direction]=np.linalg.norm(np.diff(curve,axis=0),axis=1).sum()
                key=str(int(termination[j]));reasons[key]=reasons.get(key,0)+1
        halves.append(curves)
    if np.any(half_steps>n):raise ValueError('VTK exceeded the requested one-direction step count')
    cleaned=np.zeros((len(seeds),32,3),np.float32);valid=np.zeros(len(seeds),bool)
    stats=dict(seeds=len(seeds),valid=0,short_or_nonfinite=0,axis_degenerate=0,insufficient_physical_length=0,termination_reasons=reasons)
    for i in range(len(seeds)):
        back=halves[0].get(i,seeds[i:i+1]);front=halves[1].get(i,seeds[i:i+1])
        curve=np.concatenate((back[:0:-1],front))
        if not np.isfinite(curve).all() or len(curve)<=2:stats['short_or_nonfinite']+=1;continue
        lo=spec['integration']['minimum_actual_half_arc_fraction']*limit
        hi=spec['integration']['maximum_actual_half_arc_fraction']*limit
        if np.any(half_arcs[i]<lo) or np.any(half_arcs[i]>hi):stats['insufficient_physical_length']+=1;continue
        curve=curve[np.r_[True,np.linalg.norm(np.diff(curve,axis=0),axis=1)>0]]
        if len(curve)<=2 or np.any(np.ptp(curve,axis=0)==0):stats['axis_degenerate']+=1;continue
        sampled=resample_line(curve,32)
        if sampled is None or np.any(np.ptp(sampled,axis=0)==0):stats['axis_degenerate']+=1;continue
        cleaned[i]=sampled;valid[i]=True
    stats['valid']=int(valid.sum())
    return cleaned,valid,half_arcs,half_steps,stats


def trace_batch(grid, samples, scale, spec):
    boundaries=np.r_[0,np.cumsum([len(r['seeds']) for r in samples])]
    lines,valid,arcs,steps,stats=trace_lines(grid,np.concatenate([r['seeds'] for r in samples]),scale,spec)
    rows=[];rejected=[]
    for j,sample in enumerate(samples):
        sl=slice(boundaries[j],boundaries[j+1]);good=valid[sl]
        if good.sum()<spec['bundles']['minimum_valid_lines']:
            rejected.append(dict(reason='fewer_than_10_full_length_clean_lines',label=sample['label']));continue
        physical_lines=lines[sl][good]
        geometry,seeds,centroid,radius=normalize_view(physical_lines,sample['seeds'][good],max_lines=27)
        arc=np.full((27,2),np.nan);count=np.zeros((27,2),np.int32)
        arc[:good.sum()]=arcs[sl][good];count[:good.sum()]=steps[sl][good]
        row={key:value for key,value in sample.items() if key!='seeds'}
        row.update(geometry=geometry,normalized_seeds=seeds,line_count=int(good.sum()),centroid=centroid,radius=radius,
            bounds=np.stack((physical_lines.min((0,1)),physical_lines.max((0,1)))),
            measured_mean_arc_length=float(np.linalg.norm(np.diff(physical_lines,axis=1),axis=-1).sum(1).mean()),
            half_arc_lengths=arc,half_step_counts=count,ds=scale['ds'],maxiteration=scale['maxiteration'],
            requested_half_length=scale['ds']*scale['maxiteration'])
        rows.append(row)
    return rows,rejected,stats


def distance_check(sample, train_tree, head_trees, validation_tree, spec):
    h=sample['local_grid_scale'];center=sample['center'];head=sample['head_component']
    near=float(train_tree.query(center)[0]);same=float(head_trees[head].query(center)[0]) if head in head_trees else float('inf')
    rules=spec['local_split']
    if near<h*rules['evaluation_min_distance_to_any_train_over_local_grid_scale']:return False,'too_close_to_train'
    if same>h*rules['evaluation_max_distance_to_same_head_train_over_local_grid_scale']:return False,'too_far_from_same_head_train'
    val=float(validation_tree.query(center)[0]) if validation_tree is not None else float('inf')
    if val<h*rules['test_min_distance_to_validation_over_local_grid_scale']:return False,'too_close_to_validation'
    sample.update(nearest_train_center_distance=near,nearest_same_head_train_center_distance=same)
    return True,None


def prepare(spec, config, index, preflight=False, input_root=None):
    from scipy.spatial import cKDTree
    from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow
    from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset
    if preflight:
        spec=copy.deepcopy(spec);spec['output']=str(Path(spec['output'])/'preflight')
        spec['sampling']['per_flow_counts']={'train':90,'validation':18,'test':36};spec['sampling']['minimum_class_count']=0
    flow=spec['flows'][index];root=Path(input_root or spec['input_root']);out=Path(spec['output'])/'physical'/flow['name']
    out.mkdir(parents=True,exist_ok=False)
    for role in ('flow','gt'):
        if pipeline.sha(root/flow[role])!=flow[role+'_sha256']:raise ValueError('Source snapshot or GT changed')
    axes,_,heads,oyf,grid,source=load_flow(root/flow['flow'],flow['lambda2_threshold'])
    components,catalog,details=build_catalog(axes,heads,read_dataset(root/flow['gt']),spec)
    report=dict(version=spec['version'],identity=identity(config),flow=flow,source=source,catalog=details,
        splits={},preflight=preflight,scale_plan=scale_plan(spec,flow['name']),shared_heads_instances_and_native_support=True,
        started_at_utc=datetime.now(timezone.utc).isoformat())
    # Persist every eligible native cell and its role, independent of integration acceptance.
    np.savez_compressed(out/'cell_split.npz',**{split:np.concatenate([r['split_cells'][split] for r in catalog]) for split in SPLITS})
    report['cell_split_sha256']=pipeline.sha(out/'cell_split.npz')
    pipeline.write(out/'candidate_manifest.json',report)
    train_tree=validation_tree=None;head_trees={}
    for split_index,split in enumerate(SPLITS):
        total=spec['sampling']['per_flow_counts'][split];all_rows=[];generation=[]
        for scale_id,scale in enumerate(report['scale_plan']):
            quota=total//len(report['scale_plan'])+int(scale_id<total%len(report['scale_plan']))
            accepted=[];pending=[];rejections={};stats=[];attempts=0
            eligible=[dict(r,cell_ids=r['split_cells'][split]) for r in catalog if len(r['split_cells'][split])]
            rng=np.random.default_rng(np.random.SeedSequence([spec['sampling']['seed'],index,split_index,scale_id]))
            def flush():
                nonlocal pending
                if not pending:return
                good,bad,cleaning=trace_batch(grid,pending,scale,spec)
                accepted.extend(good[:max(0,quota-len(accepted))]);stats.append(cleaning)
                for item in bad:rejections[item['reason']]=rejections.get(item['reason'],0)+1
                pending=[]
            for number in range(spec['sampling']['maximum_center_rounds']):
                for chosen in rng.permutation(len(eligible)):
                    head=eligible[chosen];attempts+=1
                    sample=center_and_neighbors(head,components,axes,oyf,number,scale,
                        spec['sampling']['seed']+index*100000+split_index*10000)
                    if sample is None or len(sample['seeds'])<spec['bundles']['minimum_valid_lines']:
                        rejections['fewer_than_10_head_seeds']=rejections.get('fewer_than_10_head_seeds',0)+1;continue
                    sample.update(head_component=head['head_component'],instance=head['instance'],label=head['label'],
                        scale_id=scale_id,center_number=number,local_grid_scale=sample['neighbor_distance']/scale['neighbor_grid_scale'],
                        nearest_train_center_distance=0.,nearest_same_head_train_center_distance=0.)
                    if split!='train':
                        valid,reason=distance_check(sample,train_tree,head_trees,validation_tree if split=='test' else None,spec)
                        if not valid:rejections[reason]=rejections.get(reason,0)+1;continue
                    pending.append(sample)
                    if len(pending)>=spec['sampling']['trace_batch_primitives']:flush()
                    if len(accepted)>=quota:break
                flush()
                if len(accepted)>=quota:break
            if len(accepted)!=quota:raise ValueError(f'Insufficient valid {flow["name"]}/{split}/scale{scale_id}: {len(accepted)}/{quota}; no distance relaxation')
            all_rows.extend(accepted);generation.append(dict(scale_id=scale_id,accepted=len(accepted),attempts=attempts,rejections=rejections,line_cleaning=stats))
            pipeline.write(out/(split+'_progress.json'),dict(scales=generation,accepted=len(all_rows),target=total))
            print(json.dumps(dict(flow=flow['name'],split=split,scale=scale_id,accepted=len(all_rows),target=total)),flush=True)
        rng.shuffle(all_rows)
        fields=('half_arc_lengths','half_step_counts','ds','maxiteration','requested_half_length','local_grid_scale',
                'nearest_train_center_distance','nearest_same_head_train_center_distance')
        extra={key:np.asarray([r[key] for r in all_rows]) for key in fields}
        entry=physical.save_split(all_rows,out/split,spec)
        with np.load(out/split/'metadata.npz') as z:meta={key:z[key] for key in z.files}
        meta.update(extra);np.savez_compressed(out/split/'metadata.npz',**meta)
        entry['files']['metadata.npz']=pipeline.sha(out/split/'metadata.npz');entry['generation']=generation
        entry['actual_half_length_quantiles']=np.nanquantile(meta['half_arc_lengths'],[0,.5,1]).tolist()
        if split=='train':
            train_tree=cKDTree(meta['center'])
            head_trees={int(h):cKDTree(meta['center'][meta['head_component']==h]) for h in np.unique(meta['head_component'])}
        else:
            entry['nearest_train_distance_over_grid_scale_quantiles']=np.quantile(meta['nearest_train_center_distance']/meta['local_grid_scale'],[0,.5,1]).tolist()
            if split=='validation':validation_tree=cKDTree(meta['center'])
        report['splits'][split]=entry;pipeline.write(out/'preparation.partial.json',report)
    audit_physical(out,spec,report)
    report['completed_at_utc']=datetime.now(timezone.utc).isoformat();pipeline.write(out/'preparation.json',report)


def audit_physical(folder,spec,report):
    from scipy.spatial import cKDTree
    meta={}
    with np.load(folder/'cell_split.npz') as z:cells={s:z[s] for s in SPLITS}
    if any(np.intersect1d(cells[a],cells[b]).size for i,a in enumerate(SPLITS) for b in SPLITS[:i]):raise ValueError('Native seed center cells cross splits')
    for split in SPLITS:
        with np.load(folder/split/'metadata.npz') as z:meta[split]={k:z[k] for k in z.files}
        m=meta[split];valid=np.arange(27)[None,:]<m['counts'][:,None]
        if not np.isin(m['source_cell'],cells[split]).all():raise ValueError('Center cell outside its fixed spatial block')
        ratio=m['half_arc_lengths']/m['requested_half_length'][:,None,None]
        if np.any(ratio[valid]<spec['integration']['minimum_actual_half_arc_fraction']) or np.any(ratio[valid]>spec['integration']['maximum_actual_half_arc_fraction']):raise ValueError('Actual physical line length guard failed')
        if np.any((m['half_step_counts']>m['maxiteration'][:,None,None])[valid]):raise ValueError('Step budget exceeded')
    tree=cKDTree(meta['train']['center']);validation=cKDTree(meta['validation']['center'])
    heads={int(h):cKDTree(meta['train']['center'][meta['train']['head_component']==h]) for h in np.unique(meta['train']['head_component'])}
    for split in ('validation','test'):
        m=meta[split]
        for i in range(len(m['labels'])):
            sample={k:m[k][i] for k in ('head_component','center','local_grid_scale')}
            ok,reason=distance_check(sample,tree,heads,validation if split=='test' else None,spec)
            if not ok:raise ValueError('Saved center distance audit failed: '+reason)
            if not np.isclose(sample['nearest_train_center_distance'],m['nearest_train_center_distance'][i],rtol=1e-12,atol=1e-12):raise ValueError('Saved center distance changed')
    report['spatial_cell_and_physical_length_audit_passed']=True


def encode(spec,config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=Path(spec['output'])
    if os.environ.get('SLURM_JOB_ID') and device.type!='cuda':raise ValueError('Encoding GPU unavailable')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    report=dict(version=spec['version'],identity=identity(config),splits={},all_geometry_newly_integrated=True)
    for flow in spec['flows']:
        physical_root=out/'physical'/flow['name'];prepared=json.loads((physical_root/'preparation.json').read_text())
        if prepared['identity']['config_sha256']!=pipeline.sha(config):raise ValueError('Preparation configuration changed')
        for split in SPLITS:
            src=physical_root/split;key=flow['name']+'/'+split;dst=out/key;dst.mkdir(parents=True,exist_ok=False)
            for name,value in prepared['splits'][split]['files'].items():
                if pipeline.sha(src/name)!=value:raise ValueError('Physical sample artifact changed')
            with np.load(src/'metadata.npz') as z:meta={k:z[k] for k in z.files}
            n=len(meta['labels']);g=np.load(src/'geometry.npy',mmap_mode='r');s=np.load(src/'seeds.npy',mmap_mode='r')
            fmt=np.lib.format.open_memmap(dst/'fmt.npy',mode='w+',dtype='float32',shape=(n,27,234))
            voxels=np.lib.format.open_memmap(dst/'voxels.npy',mode='w+',dtype='float16',shape=(n,4,24,24,24))
            for first in range(0,n,32):
                sl=slice(first,first+32);geometry=torch.as_tensor(np.array(g[sl]),device=device)
                seeds=torch.as_tensor(np.array(s[sl]),device=device);counts=torch.as_tensor(meta['counts'][sl].astype(np.int64),device=device)
                fmt[sl]=line_fmt(geometry,seeds,counts).cpu().numpy();voxels[sl]=bundle_voxels(geometry,counts).cpu().numpy().astype(np.float16)
            fmt.flush();voxels.flush()
            import shutil
            shutil.copyfile(src/'metadata.npz',dst/'metadata.npz')
            report['splits'][key]=dict(samples=n,class_counts=np.bincount(meta['labels'],minlength=2).tolist(),
                files={name:pipeline.sha(dst/name) for name in ('fmt.npy','voxels.npy','metadata.npz')})
    pipeline.write(out/'encoding.json',report)


def train(spec,config,index):
    training.identity=identity
    training.train(spec,config,index)
    method,candidate,seed=training.plan(spec)[index]
    folder=training.run_folder(spec,method,candidate,seed);result=json.loads((folder/'result.json').read_text())
    result['test_scope']='local_spatial_cell_blocks_with_1_to_4_grid_scale_train_distance'
    result['geometry_protocol']='physical_ds_times_maxiteration_4.14_all_splits_new'
    pipeline.write(folder/'result.json',result)


def merge(spec,config):
    training.identity=identity;training.merge(spec,config)
    path=Path(spec['output'])/'summary.json';result=json.loads(path.read_text())
    result['comparison']='both_frozen_regularized_models_all_three_prespecified_seeds'
    result['test_scope']='local_spatial_cell_blocks_with_1_to_4_grid_scale_train_distance'
    pipeline.write(path,result)


def runtime(spec,config,phase,state,code=None):
    row=dict(time=datetime.now(timezone.utc).isoformat(),phase=phase,state=state,exit_code=code,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',**identity(config))
    pipeline.append_locked(Path(spec['output'])/'runtime_events.jsonl',json.dumps(row)+'\n')
    pipeline.append_locked('docs/ibex_run_registry.md','\n- Task4-c 4.14 runtime '+json.dumps(row)+'\n')


def submit(spec,config):
    out=Path(spec['output']).resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    pipeline.write(out/'config.frozen.json',spec);previous=None
    for phase,count in (('preflight',2),('prepare',2),('encode',None),('train',len(training.plan(spec))),('merge',None)):
        gpu=phase in ('encode','train')
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G',
            '--time=04:00:00' if phase in ('prepare','train') else '--time=01:00:00',f'--job-name=t4c414-{phase}',
            f'--output={out}/logs/{phase}_%A_%a.out',f'--error={out}/logs/{phase}_%A_%a.err']
        if previous:command.extend((f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes'))
        if gpu:command.extend(('--gres=gpu:1','--constraint=a100|v100'))
        if count is not None:command.append(f'--array=0-{count-1}%6')
        command.extend(('ibex_bash/task4c_physical_length_4p14.sh',phase,config))
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,version=spec['version'],submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            command=command,dependency=previous,expected_device='A100/V100' if gpu else 'CPU',**identity(config))
        pipeline.append_locked(out/'submissions.jsonl',json.dumps(row)+'\n')
        pipeline.append_locked('docs/ibex_run_registry.md','\n- Task4-c 4.14 SUBMITTED '+json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);previous=job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('preflight','prepare','encode','train','merge','submit','runtime'))
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0);p.add_argument('--input-root')
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=load_spec(a.config)
    if a.phase in ('preflight','prepare'):prepare(spec,a.config,a.index,a.phase=='preflight',a.input_root)
    elif a.phase=='train':train(spec,a.config,a.index)
    elif a.phase=='runtime':runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:{'encode':encode,'merge':merge,'submit':submit}[a.phase](spec,a.config)


if __name__=='__main__':main()
