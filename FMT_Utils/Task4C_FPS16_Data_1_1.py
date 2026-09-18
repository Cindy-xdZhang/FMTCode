"""Replace short bundles with fresh physical integrations in frozen sampling strata."""
from pathlib import Path
import copy
import json
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_GTHeadCoverage_1_1 as coverage
from FMT_Utils import Task4C_BottomDensity_1_2 as expansion
from FMT_Utils.Task4C_OriginalCenter_1_1 import original_center_indices
from FMT_Utils.Task4C_InstanceCoverage_9_15 import vector_at,head_mask
from experiments import Task4C_PhysicalLength_4_14 as physical

ROLES=('train','validation','test')


def metadata(folder):
    with np.load(Path(folder)/'metadata.npz') as z:return {k:z[k] for k in z.files}


def clear(point,h,centers,scales,tree,factor=1.):
    near=tree.query_ball_point(point,float(scales.max())*factor)
    return not near or bool(np.all(np.linalg.norm(centers[near]-point,axis=1)>=scales[near]*factor-1e-12))


def row_metadata(row):
    values=dict(row)
    values['labels']=values.pop('label');values['counts']=values.pop('line_count')
    return values


class Builder:
    def __init__(self,spec,index):
        self.spec=spec;self.index=index;self.flow=spec['flows'][index]['name']
        self.physical=json.loads(Path(spec['physical_config']).read_text())
        self.previous=json.loads(Path(spec['expansion_config']).read_text())
        self.scene=coverage.load_scene(self.physical,index)
        self.source=Path(spec['source_output'])/'physical'/self.flow
        self.original={r:metadata(self.source/r) for r in ROLES}
        self.ids={};self.good={};self.centers={}
        for role in ROLES:
            file=Path(spec['subset_source'])/f'{self.flow}_{role}.npz'
            with np.load(file) as z:self.ids[role]=z['row_ids'].copy()
            m=self.original[role];ids=self.ids[role]
            anchor,_=original_center_indices(np.load(self.source/role/'seeds.npy',mmap_mode='r'),m)
            assert np.all(anchor[ids]>=0)
            self.centers[role]=anchor;self.good[role]=m['counts'][ids]>=17
        axes=self.scene['axes']
        # Reproduce the original connected-head labeling and spatial block assignment.
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
        self.pool_cache={};self.gt_points={};self.rejections={};self.actual_trace_calls=0
        self.reserved={r:[] for r in ROLES}
        self.reserved_cache={r:(None,0) for r in ROLES}
        self.final_train=None;self.final_train_heads={}
        self.all_old=np.concatenate([m['center'] for m in self.original.values()]);self.old_tree=cKDTree(self.all_old)
        self.original_trees={r:cKDTree(m['center']) for r,m in self.original.items()}
        self.eval_points=np.concatenate([self.original[r]['center'] for r in ('validation','test')])
        self.eval_h=np.concatenate([self.original[r]['local_grid_scale'] for r in ('validation','test')])
        self.eval_tree=cKDTree(self.eval_points)

    def reject(self,reason):self.rejections[reason]=self.rejections.get(reason,0)+1

    def reserved_distance(self,role,point):
        points=self.reserved[role];tree,n=self.reserved_cache[role]
        if len(points)-n>=256:
            tree=cKDTree(points);n=len(points);self.reserved_cache[role]=(tree,n)
        result=float(tree.query(point)[0]) if tree is not None else float('inf')
        if len(points)>n:result=min(result,float(np.linalg.norm(np.asarray(points[n:])-point,axis=1).min()))
        return result

    def lower(self,role,row):
        if role!='train' or row>=self.nbase*5:return False
        parent=row if row<self.nbase else (row-self.nbase)%self.nbase
        return bool(self.lower_original[parent])

    def pools(self,role,label,instance,lower):
        key=(role,label,instance,lower)
        if key not in self.pool_cache:
            pool=[]
            for h in self.catalog:
                if h['label']!=label or h['instance']!=instance:continue
                item=self.train_pools[(h['head_component'],lower)] if role=='train' else dict(h,cell_ids=h['split_cells'][role])
                if len(item['cell_ids']):pool.append(item)
            assert pool,('No original cell pool',key)
            self.pool_cache[key]=pool
        return self.pool_cache[key]

    def legal_center(self,point,h,role,head):
        if self.old_tree.query(point)[0]<=1e-10:self.reject('old_center_duplicate');return False
        if self.reserved_distance(role,point)<=1e-10:
            self.reject('new_center_duplicate');return False
        if role=='train':
            if not clear(point,h,self.eval_points,self.eval_h,self.eval_tree):self.reject('original_evaluation_clearance');return False
        else:
            # Keep original train centers reserved even when their rows are replaced.
            if self.original_trees['train'].query(point)[0]<h-1e-12:self.reject('old_train_clearance');return False
            if self.final_train is not None:
                if self.final_train.query(point)[0]<h-1e-12:self.reject('new_train_clearance');return False
                tree=self.final_train_heads.get(head)
                if tree is None or tree.query(point)[0]>4*h+1e-12:self.reject('same_head_training_proximity');return False
            if role=='validation':
                m=self.original['test']
                if not clear(point,h,m['center'],m['local_grid_scale'],self.original_trees['test'],.5):
                    self.reject('old_test_clearance');return False
            else:
                if self.original_trees['validation'].query(point)[0]<.5*h-1e-12:self.reject('old_validation_clearance');return False
                if self.reserved_distance('validation',point)<.5*h-1e-12:
                    self.reject('new_validation_clearance');return False
        return True

    def propose(self,role,slot,attempt):
        rid=int(self.ids[role][slot]);m=self.original[role]
        label=int(m['labels'][rid]);instance=int(m['instance'][rid]);head=int(m['head_component'][rid]);sid=int(m['scale_id'][rid])
        rng=np.random.default_rng([self.spec['replacement']['seed'],self.index,ROLES.index(role),rid,attempt])
        number=500000000+ROLES.index(role)*100000+slot+attempt*400000
        scale=self.plan[sid]
        if head<0:
            if instance not in self.gt_points:
                points,h=coverage.candidate_points(self.scene,self.original,instance,self.spec['replacement']['seed']+self.index*100000+instance,
                                                  count=self.spec['replacement']['gt_candidate_points'])
                self.gt_points[instance]=points
            points=self.gt_points[instance];center=points[int(rng.integers(len(points)))].copy()
            # A fresh random location, not a copied qualifying bundle.
            center+=rng.uniform(-.08,.08,3)*coverage.spacing_at(center[None],self.axes)[0]
            from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt
            owner,_=sample_gt(self.scene['gt'],center[None],self.scene['locator'])
            angle,_=head_mask(vector_at(center[None],self.axes,self.scene['velocity']),vector_at(center[None],self.axes,self.scene['omega']))
            if owner[0]!=instance or not angle[0]:self.reject('GT_head_membership');return None
            sample=coverage.make_sample(self.scene,center,instance,scale,sid,number,slot,role)
        else:
            pool=self.pools(role,label,instance,self.lower(role,rid))
            matching=[p for p in pool if p['head_component']==head]
            choices=matching if matching and attempt<self.spec['replacement']['prefer_original_head_attempts'] else pool
            sizes=np.array([len(p['cell_ids']) for p in choices],float)
            selected=choices[int(rng.choice(len(choices),p=sizes/sizes.sum()))]
            sample=physical.center_and_neighbors(selected,self.components,self.axes,self.oyf,number,scale,
                                                self.spec['replacement']['seed']+self.index*100000)
            if sample is not None:sample.update(head_component=selected['head_component'],instance=instance,label=label,scale_id=sid,
                center_number=number,local_grid_scale=sample['neighbor_distance']/scale['neighbor_grid_scale'],
                nearest_train_center_distance=0.,nearest_same_head_train_center_distance=0.)
        if sample is None or len(sample['seeds'])<17:self.reject('fewer_than_17_seed_points');return None
        if not self.legal_center(sample['center'],sample['local_grid_scale'],role,sample['head_component']):return None
        sample.update(replacement_slot=int(slot),replaced_source_row=rid,replacement_attempt=attempt+1)
        return sample

    def trace(self,samples,sid):
        if not samples:return []
        rows,rejected,stats=physical.trace_batch(self.scene['grid'],samples,self.plan[sid],self.physical)
        self.actual_trace_calls+=1
        for r in rejected:self.reject(r['reason'])
        accepted=[]
        for row in rows:
            if row['line_count']<17:self.reject('fewer_than_17_clean_lines');continue
            m=dict(center=row['center'][None],centroid=row['centroid'][None],radius=np.array([row['radius']]),
                   counts=np.array([row['line_count']]),neighbor_distance=np.array([row['neighbor_distance']]))
            center,_=original_center_indices(row['normalized_seeds'][None],m)
            if center[0]<0:self.reject('original_center_removed');continue
            row['original_center_id']=int(center[0]);accepted.append(row)
        return accepted

    def generate(self,role,slots,pilot=False):
        slots=np.asarray(slots);all_rows=[];batch=self.spec['replacement']['integration_batch_templates']
        for first in range(0,len(slots),batch):
            waiting=slots[first:first+batch].tolist();accepted={};attempts={int(s):0 for s in waiting}
            while waiting:
                pending={sid:[] for sid in range(len(self.plan))}
                for slot in waiting:
                    attempt=attempts[slot];attempts[slot]+=1
                    if attempt>=self.spec['replacement']['maximum_attempts_per_sample']:
                        raise ValueError(f'{self.flow}/{role}/slot{slot} exhausted {attempt} proposals; no quota/length/label relaxation; {self.rejections}')
                    proposal=self.propose(role,slot,attempt)
                    if proposal is not None:pending[proposal['scale_id']].append(proposal)
                for sid,samples in pending.items():
                    for row in self.trace(samples,sid):
                        slot=row['replacement_slot']
                        if not self.legal_center(row['center'],row['local_grid_scale'],role,row['head_component']):continue
                        accepted[slot]=row;self.reserved[role].append(row['center'])
                waiting=[s for s in waiting if s not in accepted]
                if attempts and max(attempts.values())%128==0:
                    print(json.dumps(dict(flow=self.flow,role=role,remaining=len(waiting),attempt=max(attempts.values()),rejections=self.rejections)),flush=True)
            all_rows.extend(accepted[int(s)] for s in slots[first:first+batch])
            print(json.dumps(dict(flow=self.flow,role=role,pilot=pilot,done=len(all_rows),total=len(slots))),flush=True)
        return all_rows


def prepare(spec,index,write,sha,identity,pilot=False):
    builder=Builder(spec,index);flow=builder.flow
    root=Path(spec['output'])/('pilot' if pilot else 'physical')/flow;root.mkdir(parents=True,exist_ok=False)
    reports={}
    for role in ROLES:
        ids=builder.ids[role];old=builder.original[role];bad=np.flatnonzero(~builder.good[role])
        if pilot:
            # Cover all affected instances / label rules, and all class / scale combinations.
            strata=np.column_stack([old[k][ids[bad]] for k in ('labels','instance')]+[(old['head_component'][ids[bad]]<0)])
            scales=np.column_stack([old[k][ids[bad]] for k in ('labels','scale_id')])
            selected=np.union1d(np.unique(strata,axis=0,return_index=True)[1],np.unique(scales,axis=0,return_index=True)[1])
            bad=bad[selected]
        rows=builder.generate(role,bad,pilot)
        folder=root/role;folder.mkdir();attempt=np.zeros(len(ids),np.int32)
        if pilot:
            reports[role]=dict(required_replacements=int((~builder.good[role]).sum()),pilot_replacements=len(rows),
                strata_passed=True,original_center_all_present=True,minimum_lines=min([r['line_count'] for r in rows],default=17))
            continue
        m={k:v[ids].copy() for k,v in old.items()}
        g_old=np.load(builder.source/role/'geometry.npy',mmap_mode='r');s_old=np.load(builder.source/role/'seeds.npy',mmap_mode='r')
        g=np.lib.format.open_memmap(folder/'geometry.npy',mode='w+',dtype=np.float32,shape=(len(ids),27,32,3))
        seeds=np.lib.format.open_memmap(folder/'seeds.npy',mode='w+',dtype=np.float32,shape=(len(ids),27,3))
        for first in range(0,len(ids),512):g[first:first+512]=g_old[ids[first:first+512]];seeds[first:first+512]=s_old[ids[first:first+512]]
        for row in rows:
            slot=row['replacement_slot'];values=row_metadata(row)
            g[slot]=row['geometry'];seeds[slot]=row['normalized_seeds']
            for key in m:m[key][slot]=values[key]
            attempt[slot]=row['replacement_attempt']
        # Quotas remain identical for class, GT instance, scale and old/new label rule.
        for key in ('labels','instance','scale_id'):assert np.array_equal(m[key],old[key][ids])
        assert np.array_equal(m['head_component']<0,old['head_component'][ids]<0)
        anchor,_=original_center_indices(seeds,m);assert np.all(anchor>=0) and np.all(m['counts']>=17)
        if role=='train':
            builder.final_train=cKDTree(m['center'])
            builder.final_train_heads={int(h):cKDTree(m['center'][m['head_component']==h]) for h in np.unique(m['head_component'])}
        else:
            m['nearest_train_center_distance']=builder.final_train.query(m['center'])[0]
            for h in np.unique(m['head_component']):
                take=m['head_component']==h;tree=builder.final_train_heads.get(int(h))
                if tree is None:raise ValueError(f'No final training head {h} for {flow}/{role}')
                m['nearest_same_head_train_center_distance'][take]=tree.query(m['center'][take])[0]
            assert np.all(m['nearest_train_center_distance']>=m['local_grid_scale']-1e-12)
            assert np.all(m['nearest_same_head_train_center_distance']<=4*m['local_grid_scale']+1e-12)
        g.flush();seeds.flush();np.savez_compressed(folder/'metadata.npz',**m)
        np.savez_compressed(folder/'replacement_index.npz',source_row=ids,replaced=attempt>0,attempts=attempt,
                            original_center_id=anchor,source_head=old['head_component'][ids],lower_pool=np.array([builder.lower(role,int(i)) for i in ids]))
        reports[role]=dict(samples=len(ids),replaced=len(rows),unchanged=len(ids)-len(rows),
            classes=np.bincount(m['labels'],minlength=2).tolist(),minimum_lines=int(m['counts'].min()),
            changed_head_count=int(np.sum(m['head_component']!=old['head_component'][ids])),
            files={file.name:sha(file) for file in folder.iterdir()})
        del g,seeds
    report=dict(complete=True,pilot=pilot,identity=identity(),flow=flow,reports=reports,rejections=builder.rejections,
                trace_calls=builder.actual_trace_calls,source_files_unchanged=True)
    write(root/'preparation.json',report)
    return report
