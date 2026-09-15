"""Corrected v2: frozen 4.14 labels/splits, more lengths and lower-wall sampling."""
from __future__ import annotations
import copy
from pathlib import Path
from types import FunctionType
import numpy as np
from experiments import Task4C_PhysicalLength_4_14 as baseline
from FMT_Utils.Task4C_PaperBundles_3_1 import normalize_view, cell_centers
from FMT_Utils.Task4C_InstanceCoverage_9_15 import write, sparse_voxels, dense_sparse_batch

def validate_spec(spec):
    assert spec['user_version']=='task4-c-v9.15_v2'
    assert spec['execution_revision']=='restored_4.14'
    assert spec['whole_instance_holdout_count']==0
    assert spec['integration']['length_definition']=='per_direction'
    assert spec['sampling']['neighbor_grid_scales']==[.25,.5,1.]
    assert spec['labels']['source']=='whole_head_center_GT_membership_including_instance_zero_not_paper_human_bundle_labels'
    assert spec['local_split']['shared_heads_instances_and_native_support']
    assert spec['expected_counts']==dict(train=89100,validation=9900,test=33000,total=132000)
    assert spec['training']['epochs']==500 and spec['training']['patience']==50
    assert spec['training']['selection']=='validation_fixed_0.5_F1_then_average_precision'
    for flow,limits in [('channel',(.001,.005)),('tbl',(.01,.05))]:
        lo,hi=spec['integration']['length_ranges'][flow]
        for step in spec['integration'][flow]:
            assert limits[0]-1e-12<=step['ds']<=limits[1]+1e-12
            assert lo<=step['ds']*step['maxiteration']<=hi

def center_and_neighbors(row,component_grid,axes,oyf,number,scale,seed):
    """Only positive center positions change; negative sampling calls old code verbatim."""
    if row['label']==1 and number%2:
        row=dict(row)
        points=cell_centers(row['cell_ids'],component_grid.shape,axes)
        row['cell_ids']=row['cell_ids'][points[:,2]<=np.median(points[:,2])]
    return baseline.center_and_neighbors(row,component_grid,axes,oyf,number,scale,seed)

def trace_batch(grid,samples,scale,spec,flow):
    rows,failures,stats=baseline.trace_batch(grid,samples,scale,spec)
    lower,upper=spec['integration']['length_ranges'][flow];eps=2e-6*upper
    accepted=[]
    for row in rows:
        n=row['line_count'];arcs=row['half_arc_lengths'][:n]
        world=row['geometry'][:n].astype(np.float64)*row['radius']+row['centroid']
        seeds=row['normalized_seeds'][:n].astype(np.float64)*row['radius']+row['centroid']
        sampled=np.linalg.norm(np.diff(world,axis=1),axis=-1).sum(1)
        good=((arcs>=lower-eps)&(arcs<=upper+eps)).all(1)&(sampled>=2*lower-eps)&(sampled<=2*upper+eps)
        if good.sum()<10:
            failures.append(dict(reason='fewer_than_10_lines_in_user_length_range',label=row['label']));continue
        physical=world[good]
        geometry,normalized_seeds,centroid,radius=normalize_view(physical,seeds[good],max_lines=27)
        count=int(good.sum());half_arcs=np.full((27,2),np.nan);steps=np.zeros((27,2),np.int32)
        half_arcs[:count]=arcs[good];steps[:count]=row['half_step_counts'][:n][good]
        row.update(geometry=geometry,normalized_seeds=normalized_seeds,centroid=centroid,radius=radius,
            line_count=count,bounds=np.stack((physical.min((0,1)),physical.max((0,1)))),
            half_arc_lengths=half_arcs,half_step_counts=steps,measured_mean_arc_length=float(sampled[good].mean()))
        accepted.append(row)
    return accepted,failures,stats

def build_catalog(axes,heads,gt,spec):
    components,catalog,details=baseline.build_catalog(axes,heads,gt,spec)
    details['eligible_labels']=[{k:r[k] for k in ('head_component','label','instance','head_cell_count')} for r in catalog]
    return components,catalog,details

def prepare(spec,config,index,identity,pilot=False,input_root=None):
    working=copy.deepcopy(spec)
    if pilot:
        working['output']=str(Path(spec['output'])/'pilot')
        count=len(baseline.scale_plan(spec,spec['flows'][index]['name']))
        working['sampling']['per_flow_counts']=dict(train=10*count,validation=2*count,test=4*count)
        working['sampling']['minimum_class_count']=0
    # Scoped globals preserve the frozen baseline module and its other callers.
    env=dict(baseline.__dict__,identity=identity,center_and_neighbors=center_and_neighbors,build_catalog=build_catalog,
        trace_batch=lambda grid,samples,scale,settings:trace_batch(grid,samples,scale,settings,settings['flows'][index]['name']))
    execute=FunctionType(baseline.prepare.__code__,env,argdefs=baseline.prepare.__defaults__)
    execute(working,config,index,preflight=False,input_root=input_root)
