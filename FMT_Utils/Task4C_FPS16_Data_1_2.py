"""Fixed original center with >=16 in-domain neighbours; replace only failing rows (1.2).

Differences from 1.1 (user decisions of 2026-09-18):
* Template is the complete GT-head-coverage dataset (196,960 / 3,000 / 11,320 rows),
  not the original-center common subset.
* A row is replaced when its original center line is missing OR it has fewer than
  17 valid lines (center + 16 neighbours), or when the final training set no longer
  provides a same-head training center within 4h for a retained evaluation row.
* Only the center seed keeps the lambda2 / oyf>0 / same-head filter.  The 26 stencil
  neighbours are integrated whenever they lie inside the domain; the old code also
  required lambda2 / oyf / same-head for every neighbour, which made 17 seeds
  structurally impossible for thin head regions.
* After ``normal_attempts`` failed proposals for one slot the center filter is relaxed
  (``relaxed_attempts`` further proposals).  A relaxed center is labelled by the GT
  membership of the center point itself and flagged ``relaxed`` in the index.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_GTHeadCoverage_1_1 as coverage
from FMT_Utils import Task4C_BottomDensity_1_2 as expansion
from FMT_Utils import Task4C_FPS16_Data_1_1 as previous
from FMT_Utils.Task4C_OriginalCenter_1_1 import original_center_indices
from FMT_Utils.Task4C_InstanceCoverage_9_15 import vector_at,head_mask
from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt
from experiments import Task4C_PhysicalLength_4_14 as physical

ROLES=previous.ROLES
metadata=previous.metadata
row_metadata=previous.row_metadata
FILTER_OLD,FILTER_IN_DOMAIN,FILTER_RELAXED=0,1,2


def stencil(center,distance):
    """27 cubic offsets with the center first; identical ordering to the frozen 4.14 code."""
    offsets=np.array([[x,y,z] for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1)],float)
    middle=int(np.flatnonzero((offsets==0).all(1))[0]);offsets[[0,middle]]=offsets[[middle,0]]
    return center+offsets*distance


def cell_of(point,axes):
    xyz=[int(np.clip(np.searchsorted(a,point[d],side='right')-1,0,len(a)-2)) for d,a in enumerate(axes)]
    return int(np.ravel_multi_index(xyz[::-1],tuple(len(a)-1 for a in axes[::-1])))


def failing_rows(seeds,m,minimum):
    """Rows whose original center is absent or that have fewer than ``minimum`` valid lines."""
    anchor,_=original_center_indices(seeds,m)
    return anchor,(anchor<0)|(m['counts']<minimum)


def head_center_and_neighbors(row,component_grid,axes,oyf,center_number,scale,base_seed):
    """Frozen 4.14 center rule (same RNG, cell, jitter, oyf>0 and same-head check); all in-domain neighbours."""
    rng=np.random.default_rng(np.random.SeedSequence([base_seed,row['head_component'],center_number]))
    cells=row['cell_ids'];cell=int(rng.choice(cells))
    iz,iy,ix=np.unravel_index(cell,component_grid.shape)
    spacing=np.array([axes[0][ix+1]-axes[0][ix],axes[1][iy+1]-axes[1][iy],axes[2][iz+1]-axes[2][iz]])
    center=np.array([axes[0][ix],axes[1][iy],axes[2][iz]])+rng.uniform(.25,.75,3)*spacing
    distance=float(np.prod(spacing)**(1/3))*scale['neighbor_grid_scale']
    seeds=stencil(center,distance)
    fluct,inside,cell_id=interpolate_scalar(seeds,axes,oyf)
    if not (inside[0] and fluct[0]>0 and component_grid.ravel()[cell_id[0]]==row['head_component']):return None
    valid=inside.copy();valid[0]=True
    return dict(center=center,seeds=seeds[valid],source_cell=cell,offset_grid_ids=np.flatnonzero(valid),neighbor_distance=distance,
        seed_rms_distance=float(np.sqrt(np.mean(np.sum((seeds[valid]-center)**2,axis=1)))))


def gt_head_sample(scene,center,instance,scale,sid,number,group,role,relaxed=False):
    """GT-head positive sample; the center keeps lambda2/oyf unless relaxed, neighbours only need the domain."""
    axes=scene['axes'];h=float(np.prod(coverage.spacing_at(center[None],axes))**(1/3))
    distance=h*scale['neighbor_grid_scale'];seeds=stencil(center,distance)
    lam,inside,_=interpolate_scalar(seeds,axes,scene['lambda2']);fluct,_,_=interpolate_scalar(seeds,axes,scene['oyf'])
    if not inside[0]:return None
    if not relaxed and not (lam[0]<scene['flow']['lambda2_threshold'] and fluct[0]>0):return None
    valid=inside.copy();valid[0]=True;seeds=seeds[valid]
    return dict(center=center,seeds=seeds,label=1,instance=instance,head_component=-instance-1,source_cell=cell_of(center,axes),
        scale_id=sid,center_number=number,group=group,added_role=role,local_grid_scale=h,neighbor_distance=distance,
        seed_rms_distance=float(np.linalg.norm(seeds-center,axis=1).mean()),
        nearest_train_center_distance=0.,nearest_same_head_train_center_distance=0.)


class Builder(previous.Builder):
    """Reuses the frozen 1.1 pools, lower-pool provenance and center legality rules."""

    def __init__(self,spec,index):
        self.spec=spec;self.index=index;self.flow=spec['flows'][index]['name'];self.rule=spec['replacement']
        self.physical=json.loads(Path(spec['physical_config']).read_text())
        self.previous=json.loads(Path(spec['expansion_config']).read_text())
        self.scene=coverage.load_scene(self.physical,index)
        self.source=Path(spec['source_output'])/'physical'/self.flow
        self.original={r:metadata(self.source/r) for r in ROLES}
        self.ids={};self.good={};self.centers={}
        for role in ROLES:
            m=self.original[role]
            anchor,bad=failing_rows(np.load(self.source/role/'seeds.npy',mmap_mode='r'),m,self.rule['minimum_valid_lines'])
            self.ids[role]=np.arange(len(m['labels']));self.centers[role]=anchor;self.good[role]=~bad
        from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow
        flow=self.physical['flows'][index]
        axes,_,heads,oyf,_,_=load_flow(Path(self.physical['input_root'])/flow['flow'],flow['lambda2_threshold'])
        self.components,self.catalog,_=physical.build_catalog(axes,heads,self.scene['gt'],self.physical)
        self.by_head={h['head_component']:h for h in self.catalog};self.axes=axes;self.oyf=oyf
        self.plan=physical.scale_plan(self.physical,self.flow)
        n=self.previous['expansion']['source_rows_per_flow'][self.flow]
        first={k:v[:n] for k,v in self.original['train'].items()}
        self.lower_original=expansion.lower_pool_flags(first,self.previous);self.nbase=n
        self.train_pools=expansion.candidate_pools(self.catalog,self.components,axes,first,self.previous)
        self.pool_cache={};self.gt_points={};self.gt_cells=None;self.box_cache={}
        self.rejections={};self.actual_trace_calls=0;self.relaxed_accepted={r:0 for r in ROLES}
        self.reserved={r:[] for r in ROLES};self.reserved_cache={r:(None,0) for r in ROLES}
        self.final_train=None;self.final_train_heads={}
        self.all_old=np.concatenate([m['center'] for m in self.original.values()]);self.old_tree=cKDTree(self.all_old)
        self.original_trees={r:cKDTree(m['center']) for r,m in self.original.items()}
        self.eval_points=np.concatenate([self.original[r]['center'] for r in ('validation','test')])
        self.eval_h=np.concatenate([self.original[r]['local_grid_scale'] for r in ('validation','test')])
        self.eval_tree=cKDTree(self.eval_points)

    def stage(self,attempt):
        r=self.rule
        if attempt<r['prefer_original_head_attempts']:return 'original_head'
        if attempt<r['normal_attempts']:return 'any_head'
        if attempt<r['normal_attempts']+r['relaxed_attempts']:return 'relaxed'
        return None

    def gt_cell_centers(self,instance):
        """Unfiltered GT cell centers of one annotated instance (relaxed GT-head proposals only)."""
        if self.gt_cells is None:
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy
            gt=self.scene['gt'];labels=vtk_to_numpy(gt.GetCellData().GetArray('VortexIds')).astype(np.int64)
            centers=vtk.vtkCellCenters();centers.SetInputData(gt);centers.Update()
            points=vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).copy()
            self.gt_cells={int(i):points[labels==i] for i in np.unique(labels)}
        return self.gt_cells.get(int(instance),np.zeros((0,3)))

    def propose(self,role,slot,attempt):
        rid=int(self.ids[role][slot]);m=self.original[role]
        label=int(m['labels'][rid]);instance=int(m['instance'][rid]);head=int(m['head_component'][rid]);sid=int(m['scale_id'][rid])
        stage=self.stage(attempt)
        if stage is None:return None
        rng=np.random.default_rng([self.rule['seed'],self.index,ROLES.index(role),rid,attempt])
        number=500000000+ROLES.index(role)*100000+slot+attempt*400000
        scale=self.plan[sid];relaxed=stage=='relaxed'
        if head<0:sample=self.propose_gt_head(role,slot,rng,number,instance,scale,sid,relaxed)
        else:sample=self.propose_head_region(role,rid,rng,number,label,instance,head,scale,sid,stage)
        if sample is None:return None
        if len(sample['seeds'])<self.rule['minimum_valid_lines']:self.reject('fewer_than_17_in_domain_seed_points');return None
        if not self.legal_center(sample['center'],sample['local_grid_scale'],role,sample['head_component']):return None
        sample.update(replacement_slot=int(slot),replaced_source_row=rid,replacement_attempt=attempt+1,relaxed=relaxed,
                      neighbor_filter=FILTER_RELAXED if relaxed else FILTER_IN_DOMAIN,stratum_label=label,stratum_instance=instance)
        return sample

    def propose_gt_head(self,role,slot,rng,number,instance,scale,sid,relaxed):
        if relaxed:points=self.gt_cell_centers(instance)
        else:
            if instance not in self.gt_points:
                points,_=coverage.candidate_points(self.scene,self.original,instance,self.rule['seed']+self.index*100000+instance,
                                                  count=self.rule['gt_candidate_points'])
                self.gt_points[instance]=points
            points=self.gt_points[instance]
        if not len(points):self.reject('empty_gt_pool');return None
        center=points[int(rng.integers(len(points)))].copy()
        # A fresh random location, not a copied qualifying bundle.
        center+=rng.uniform(-.08,.08,3)*coverage.spacing_at(center[None],self.axes)[0]
        owner,_=sample_gt(self.scene['gt'],center[None],self.scene['locator'])
        if owner[0]!=instance:self.reject('GT_head_membership');return None
        if not relaxed:
            angle,_=head_mask(vector_at(center[None],self.axes,self.scene['velocity']),vector_at(center[None],self.axes,self.scene['omega']))
            if not angle[0]:self.reject('GT_head_membership');return None
        sample=gt_head_sample(self.scene,center,instance,scale,sid,number,slot,role,relaxed)
        if sample is None:self.reject('gt_center_failed_domain_lambda2_oyf')
        return sample

    def propose_head_region(self,role,rid,rng,number,label,instance,head,scale,sid,stage):
        pool=self.pools(role,label,instance,self.lower(role,rid))
        if stage=='relaxed':return self.relaxed_head_region(rng,number,pool,head,scale,sid,role,label,instance)
        matching=[p for p in pool if p['head_component']==head]
        choices=matching if matching and stage=='original_head' else pool
        sizes=np.array([len(p['cell_ids']) for p in choices],float)
        selected=choices[int(rng.choice(len(choices),p=sizes/sizes.sum()))]
        sample=head_center_and_neighbors(selected,self.components,self.axes,self.oyf,number,scale,self.rule['seed']+self.index*100000)
        if sample is None:self.reject('center_failed_oyf_or_head');return None
        sample.update(head_component=selected['head_component'],instance=instance,label=label,scale_id=sid,center_number=number,
            local_grid_scale=sample['neighbor_distance']/scale['neighbor_grid_scale'],
            nearest_train_center_distance=0.,nearest_same_head_train_center_distance=0.)
        return sample

    def relaxed_head_region(self,rng,number,pool,head,scale,sid,role,label,instance):
        """Center anywhere in the stratum's cell box (one spacing margin) without lambda2/oyf; GT membership labels it."""
        from FMT_Utils.Task4C_PaperBundles_3_1 import cell_centers
        key=(role,label,instance,head)
        if key not in self.box_cache:
            points=np.concatenate([cell_centers(p['cell_ids'],self.components.shape,self.axes) for p in pool])
            spacing=coverage.spacing_at(points,self.axes)
            self.box_cache[key]=((points-spacing).min(0),(points+spacing).max(0))
        low,high=self.box_cache[key];center=rng.uniform(low,high)
        h=float(np.prod(coverage.spacing_at(center[None],self.axes))**(1/3));distance=h*scale['neighbor_grid_scale']
        seeds=stencil(center,distance);_,inside,_=interpolate_scalar(seeds,self.axes,self.oyf)
        if not inside[0]:self.reject('relaxed_center_outside_domain');return None
        valid=inside.copy();valid[0]=True
        owner,_=sample_gt(self.scene['gt'],center[None],self.scene['locator'])
        new_label=int(owner[0]>=0);new_instance=int(owner[0]) if new_label else -1
        return dict(center=center,seeds=seeds[valid],source_cell=cell_of(center,self.axes),offset_grid_ids=np.flatnonzero(valid),
            neighbor_distance=distance,seed_rms_distance=float(np.sqrt(np.mean(np.sum((seeds[valid]-center)**2,axis=1)))),
            head_component=head,instance=new_instance,label=new_label,scale_id=sid,center_number=number,local_grid_scale=h,
            nearest_train_center_distance=0.,nearest_same_head_train_center_distance=0.)

    def trace(self,samples,sid):
        if not samples:return []
        rows,rejected,_=physical.trace_batch(self.scene['grid'],samples,self.plan[sid],self.physical)
        self.actual_trace_calls+=1
        for r in rejected:self.reject(r['reason'])
        accepted=[]
        for row in rows:
            if row['line_count']<self.rule['minimum_valid_lines']:self.reject('fewer_than_17_clean_lines');continue
            m=dict(center=row['center'][None],centroid=row['centroid'][None],radius=np.array([row['radius']]),
                   counts=np.array([row['line_count']]),neighbor_distance=np.array([row['neighbor_distance']]))
            center,_=original_center_indices(row['normalized_seeds'][None],m)
            if center[0]<0:self.reject('original_center_removed');continue
            row['original_center_id']=int(center[0]);accepted.append(row)
        return accepted

    def generate(self,role,slots,pilot=False):
        slots=np.asarray(slots);all_rows=[];batch=self.rule['integration_batch_templates']
        for first in range(0,len(slots),batch):
            waiting=slots[first:first+batch].tolist();accepted={};attempts={int(s):0 for s in waiting}
            while waiting:
                pending={sid:[] for sid in range(len(self.plan))}
                for slot in waiting:
                    attempt=attempts[slot];attempts[slot]+=1
                    if self.stage(attempt) is None:
                        raise ValueError(f'{self.flow}/{role}/slot{slot} exhausted {attempt} proposals including relaxed centers; {self.rejections}')
                    proposal=self.propose(role,slot,attempt)
                    if proposal is not None:pending[proposal['scale_id']].append(proposal)
                for sid,samples in pending.items():
                    for row in self.trace(samples,sid):
                        slot=row['replacement_slot']
                        if slot in accepted:continue
                        if not self.legal_center(row['center'],row['local_grid_scale'],role,row['head_component']):continue
                        accepted[slot]=row;self.reserved[role].append(row['center'])
                        if row['relaxed']:self.relaxed_accepted[role]+=1
                waiting=[s for s in waiting if s not in accepted]
                if attempts and max(attempts.values())%64==0:
                    print(json.dumps(dict(flow=self.flow,role=role,remaining=len(waiting),attempt=max(attempts.values()),
                                          relaxed_accepted=self.relaxed_accepted[role],rejections=self.rejections)),flush=True)
            all_rows.extend(accepted[int(s)] for s in slots[first:first+batch])
            print(json.dumps(dict(flow=self.flow,role=role,pilot=pilot,done=len(all_rows),total=len(slots),
                                  relaxed_accepted=self.relaxed_accepted[role])),flush=True)
        return all_rows


def lost_same_head_proximity(builder,role):
    """Retained evaluation rows whose final training set no longer has a same-head center within 4h."""
    m=builder.original[role];lost=np.zeros(len(m['labels']),bool)
    for head in np.unique(m['head_component']):
        take=np.flatnonzero(m['head_component']==head);tree=builder.final_train_heads.get(int(head))
        if tree is None:lost[take]=True;continue
        lost[take]=tree.query(m['center'][take])[0]>4*m['local_grid_scale'][take]+1e-12
    return lost


def assemble(builder,role,folder,keep,rows,old,sha,pilot):
    """Copy retained rows, overwrite replaced slots, freeze provenance; quotas change only for relaxed rows."""
    position={int(k):i for i,k in enumerate(keep)}
    m={k:v[keep].copy() for k,v in old.items()}
    g_old=np.load(builder.source/role/'geometry.npy',mmap_mode='r');s_old=np.load(builder.source/role/'seeds.npy',mmap_mode='r')
    g=np.lib.format.open_memmap(folder/'geometry.npy',mode='w+',dtype=np.float32,shape=(len(keep),27,32,3))
    seeds=np.lib.format.open_memmap(folder/'seeds.npy',mode='w+',dtype=np.float32,shape=(len(keep),27,3))
    for first in range(0,len(keep),512):
        take=keep[first:first+512];g[first:first+len(take)]=g_old[take];seeds[first:first+len(take)]=s_old[take]
    attempt=np.zeros(len(keep),np.int32);relaxed=np.zeros(len(keep),bool);filt=np.full(len(keep),FILTER_OLD,np.int8)
    for row in rows:
        i=position[int(row['replacement_slot'])];values=row_metadata(row)
        g[i]=row['geometry'];seeds[i]=row['normalized_seeds']
        for key in m:m[key][i]=values[key]
        attempt[i]=row['replacement_attempt'];relaxed[i]=bool(row['relaxed']);filt[i]=row['neighbor_filter']
    changed=attempt>0
    for key in ('labels','instance'):assert np.array_equal(m[key][~relaxed],old[key][keep][~relaxed]),key
    assert np.array_equal(m['scale_id'],old['scale_id'][keep])
    assert np.array_equal(m['head_component']<0,old['head_component'][keep]<0)
    anchor,_=original_center_indices(seeds,m)
    assert np.all(anchor>=0) and np.all(m['counts']>=builder.rule['minimum_valid_lines'])
    assert np.all(anchor[changed]==0),'replaced rows keep the center in slot 0'
    if not pilot:
        if role=='train':
            builder.final_train=cKDTree(m['center'])
            builder.final_train_heads={int(h):cKDTree(m['center'][m['head_component']==h]) for h in np.unique(m['head_component'])}
        else:
            m['nearest_train_center_distance']=builder.final_train.query(m['center'])[0]
            for h in np.unique(m['head_component']):
                take=m['head_component']==h;tree=builder.final_train_heads.get(int(h))
                if tree is None:raise ValueError(f'No final training head {h} for {builder.flow}/{role}')
                m['nearest_same_head_train_center_distance'][take]=tree.query(m['center'][take])[0]
            assert np.all(m['nearest_train_center_distance']>=m['local_grid_scale']-1e-12)
            assert np.all(m['nearest_same_head_train_center_distance']<=4*m['local_grid_scale']+1e-12)
    g.flush();seeds.flush();del g,seeds
    np.savez_compressed(folder/'metadata.npz',**m)
    np.savez_compressed(folder/'replacement_index.npz',source_row=keep,replaced=changed,attempts=attempt,original_center_id=anchor,
        source_head=old['head_component'][keep],lower_pool=np.array([builder.lower(role,int(i)) for i in keep]),
        relaxed=relaxed,neighbor_filter=filt,source_label=old['labels'][keep],source_instance=old['instance'][keep])
    return dict(samples=len(keep),replaced=int(changed.sum()),unchanged=int((~changed).sum()),relaxed=int(relaxed.sum()),
        label_changes=int(np.sum(m['labels']!=old['labels'][keep])),classes=np.bincount(m['labels'],minlength=2).tolist(),
        minimum_lines=int(m['counts'].min()),files={file.name:sha(file) for file in folder.iterdir()})


def prepare(spec,index,write,sha,identity,pilot=False):
    builder=Builder(spec,index);flow=builder.flow
    root=Path(spec['output'])/('pilot' if pilot else 'physical')/flow;root.mkdir(parents=True,exist_ok=False)
    reports={}
    for role in ROLES:
        ids=builder.ids[role];old=builder.original[role];bad=~builder.good[role];lost=0
        if role!='train' and not pilot:
            lost_rows=lost_same_head_proximity(builder,role)&~bad;lost=int(lost_rows.sum());bad=bad|lost_rows
        bad=np.flatnonzero(bad)
        if pilot:
            # Cover every affected (class, instance, label rule) and every (class, scale) stratum once.
            strata=np.column_stack([old[k][bad] for k in ('labels','instance')]+[(old['head_component'][bad]<0)])
            scales=np.column_stack([old[k][bad] for k in ('labels','scale_id')])
            selected=np.union1d(np.unique(strata,axis=0,return_index=True)[1],np.unique(scales,axis=0,return_index=True)[1])
            bad=bad[selected]
        rows=builder.generate(role,bad,pilot)
        folder=root/role;folder.mkdir()
        report=assemble(builder,role,folder,bad if pilot else ids,rows,old,sha,pilot)
        report.update(required_replacements=int((~builder.good[role]).sum()),replaced_for_lost_same_head_train=lost,
                      pilot_slots=len(bad) if pilot else None)
        reports[role]=report
    report=dict(complete=True,pilot=pilot,identity=identity(),flow=flow,reports=reports,rejections=builder.rejections,
                relaxed_accepted=builder.relaxed_accepted,trace_calls=builder.actual_trace_calls,source_files_unchanged=True,
                neighbor_filter='center_only_lambda2_oyf_same_head; neighbours in domain only',
                relaxation='center filter dropped after normal_attempts; label by GT membership of the center')
    write(root/'preparation.json',report)
    return report
