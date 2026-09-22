"""Persistent manual core edits and atomic, training-ready dataset revisions."""
from collections import OrderedDict
from pathlib import Path
import copy
import datetime
import hashlib
import json
import os
import threading
import uuid
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from FMT_Utils.Task6CorelineDataset_1_1 import coreline_labels
from FMT_Utils.Task6Sampling_2_1 import SegmentQuery
from experiments.Filter_Task6_CoreLength_1_4 import read_cores, sha256
from experiments.Extract_Task6_VortexCore_1_2 import write_lines
from experiments.Filter_Task6_Dataset_2_3 import reuse, remap_ids

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'config/Other_Task6_DatasetInspector_1.7.json').read_text(encoding='utf-8'))
BASE = ROOT/CONFIG['base_dataset']
BASE_SHA = CONFIG['base_frozen_sha256']
VERSION = 'mainExp_Task6_CorelineDataset_2.7'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def merge_curves(a,b,h):
    """Nearest endpoint pair; preserve every original vertex, add a straight bridge."""
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    pairs=[(float(np.linalg.norm(a[i]-b[j])),i,j) for i in (0,-1) for j in (0,-1)]
    gap,i,j=min(pairs,key=lambda x:x[0])
    left=a[::-1] if i==0 else a
    right=b[::-1] if j==-1 else b
    # Subdivision changes no geometry; it bounds independent segment-tree queries.
    steps=max(1,int(np.ceil(gap/h)))
    bridge=np.linspace(left[-1],right[0],steps+1)
    line=np.concatenate((left,bridge[1:-1],right))
    return line,dict(gap=gap,gap_h=gap/h,first_endpoint=i,second_endpoint=j,
                     bridge=bridge.tolist(),bridge_subsegments=steps)


def replay(cores,operations,h):
    entries={i:dict(points=np.asarray(c,np.float64),parents=[i]) for i,c in enumerate(cores)}
    bridges=[]
    for op in operations:
        kind=op['kind'];a=op['first']
        if type(a) is not int or a not in entries:raise ValueError('核线序号不存在或已删除')
        if kind=='delete':
            del entries[a]
            bridges=[b for b in bridges if b['owner']!=a]
        elif kind=='merge':
            b=op['second']
            if type(b) is not int or b==a or b not in entries:raise ValueError('请选择两条不同的现有核线')
            merged,report=merge_curves(entries[a]['points'],entries[b]['points'],h)
            entries[a]=dict(points=merged,parents=entries[a]['parents']+entries[b]['parents'])
            del entries[b]
            for bridge in bridges:
                if bridge['owner']==b:bridge['owner']=a
            bridges.append(dict(owner=a,first=a,second=b,**report))
        else:raise ValueError('Unsupported edit operation')
    ids=sorted(entries)
    return dict(ids=ids,cores=[entries[i]['points'] for i in ids],
                parents=[entries[i]['parents'] for i in ids],bridges=bridges)


class Conflict(ValueError):
    pass


class Editor:
    def __init__(self,workspace,base=BASE,expected=BASE_SHA):
        self.base=Path(base).resolve();self.workspace=Path(workspace).resolve()
        if not self.workspace.is_relative_to((ROOT/'outputs').resolve()):
            raise ValueError('Editor workspace must be inside project outputs')
        assert sha256(self.base/'dataset_frozen.json')==expected
        self.expected=expected;self.frozen=read(self.base/'dataset_frozen.json')
        self.records={f"{f['flow']}:{f['index']}":f for f in self.frozen['frames']}
        self.workspace.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.cache=OrderedDict();self.job=None
        state=self.workspace/'draft.json'
        self.state=read(state) if state.exists() else dict(base_sha256=expected,revision=0,operations={})
        assert self.state['base_sha256']==expected
        if (self.workspace/'job.json').exists():
            self.job=read(self.workspace/'job.json')
            if self.job['state']=='RUNNING':
                active,frozen,applied=self._active()
                published=(active!=self.base and read(active/'manual_edits.json')['draft_revision']==self.job['draft_revision'])
                if published:
                    self.job.update(state='COMPLETED',dataset_path=str(active),
                        frozen_sha256=sha256(active/'dataset_frozen.json'),manual_revision=frozen['manual_revision'],
                        recovered_after_restart=True)
                else:
                    self.job.update(state='FAILED',error='服务重启中断应用；上一已应用版本保留，可重新应用草稿。')
                write(self.workspace/'job.json',self.job)
        self._active()

    def _active(self):
        pointer=self.workspace/'current.json'
        if not pointer.exists():return self.base,self.frozen,{}
        p=read(pointer);dest=(self.workspace/p['relative_path']).resolve()
        assert dest.is_relative_to(self.workspace/'revisions')
        assert sha256(dest/'dataset_frozen.json')==p['frozen_sha256']
        return dest,read(dest/'dataset_frozen.json'),read(dest/'manual_edits.json')['operations']

    def active_folder(self,key):
        f=self.record(key);active,_,_=self._active()
        return active/f['flow']/f"frame_{f['index']:03d}"

    def record(self,key):
        if key not in self.records:raise ValueError('Unknown frame')
        return self.records[key]

    def base_folder(self,key):
        f=self.record(key)
        return self.base/f['flow']/f"frame_{f['index']:03d}"

    def status(self):
        with self.lock:
            active,frozen,applied=self._active()
            pending=[key for key in self.records if self.state['operations'].get(key,[])!=applied.get(key,[])]
            return dict(draft_revision=self.state['revision'],pending_frames=pending,
                        pending_count=len(pending),dataset=frozen['version'],dataset_path=str(active),
                        dataset_sha256=sha256(active/'dataset_frozen.json'),
                        applied_revision=frozen.get('manual_revision',0),job=copy.deepcopy(self.job))

    def preview(self,key):
        with self.lock:
            f=self.record(key);ops=self.state['operations'].get(key,[]);signature=(key,digest(ops))
            if signature in self.cache:
                self.cache.move_to_end(signature);return self.cache[signature]
            src=self.base_folder(key)
            assert sha256(src/'corelines.vtp')==f['core_sha256']
            original=read_cores(src/'corelines.vtp')
            value=replay(original,ops,f['h'])
            if ops:
                seed=np.load(src/'seeds.npy',mmap_mode='r')
                result=coreline_labels(seed,value['cores'],f['h'])
                arrays={'labels.npy':result['label'],'positive_distance.npy':result['positive_distance'],
                        'positive_core_id.npy':result['positive_core_id']}
            else:
                arrays={name:np.load(src/name,mmap_mode='r') for name in
                        ('labels.npy','positive_distance.npy','positive_core_id.npy')}
            value.update(arrays=arrays,lengths=[float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum()) for c in value['cores']],
                         operations=copy.deepcopy(ops),signature=signature[1])
            self.cache[signature]=value
            while len(self.cache)>4:self.cache.popitem(last=False)
            return value

    def edit(self,key,operation,revision):
        with self.lock:
            if self.job and self.job['state']=='RUNNING':raise Conflict('正在应用修改，请完成后继续编辑')
            if revision!=self.state['revision']:raise Conflict('草稿已更新，请刷新后再操作')
            self.record(key);ops=copy.deepcopy(self.state['operations'].get(key,[]))
            kind=operation.get('kind')
            if kind in ('delete','merge'):
                op={'kind':kind,'first':operation.get('first')}
                if kind=='merge':op['second']=operation.get('second')
                ops.append(op)
            elif kind=='undo':
                if not ops:raise ValueError('没有可撤销的操作')
                ops.pop()
            elif kind=='reset':ops=[]
            else:raise ValueError('Unsupported edit operation')
            replay(read_cores(self.base_folder(key)/'corelines.vtp'),ops,self.record(key)['h'])
            # Geometry validation precedes the durable write; no partially valid draft.
            next_state=copy.deepcopy(self.state)
            next_state['operations'][key]=ops;next_state['revision']+=1;next_state['updated_utc']=utc()
            write(self.workspace/'draft.json',next_state);self.state=next_state
            return self.status()

    def start_apply(self,revision):
        with self.lock:
            if self.job and self.job['state']=='RUNNING':raise Conflict('正在应用修改')
            if revision!=self.state['revision']:raise Conflict('草稿已更新，请刷新后再应用')
            status=self.status()
            if not status['pending_count']:raise ValueError('没有尚未应用的修改')
            self.job=dict(id=uuid.uuid4().hex,state='RUNNING',done=0,total=status['pending_count'],
                          started_utc=utc(),draft_revision=revision)
            write(self.workspace/'job.json',self.job)
            snapshot=copy.deepcopy(self.state)
            threading.Thread(target=self._apply,args=(snapshot,status['pending_frames']),daemon=True).start()
            return copy.deepcopy(self.job)

    def _apply_frame(self,key,dest,snapshot):
        f=self.record(key);src=self.base_folder(key);dst=dest/f['flow']/f"frame_{f['index']:03d}"
        dst.mkdir(parents=True,exist_ok=True);v=self.preview(key)
        assert v['operations']==snapshot['operations'].get(key,[])
        cores=v['cores'];h=f['h'];arrays=v['arrays'];labels=arrays['labels.npy'];owners=arrays['positive_core_id.npy']
        completed=read(src/'completed.json')
        changed={'corelines.vtp','labels.npy','positive_distance.npy','positive_core_id.npy','sample_seeds.npz',
                 'length_filter.json','sampling.json','candidate_core_support.json','audit.json','completed.json','frame_result.json'}
        for name,sha in completed['files'].items():
            assert sha256(src/name)==sha,(key,name)
            if name not in changed:reuse(src/name,dst/name)
        for name in ('seeds.npy','sample_streamlines.npy'):
            if not (dst/name).exists():reuse(src/name,dst/name)
        seeds=np.load(src/'seeds.npy',mmap_mode='r')
        independent=SegmentQuery(cores).distance(seeds,h,workers=2)
        np.testing.assert_array_equal(labels,independent<h)
        np.testing.assert_allclose(arrays['positive_distance.npy'][labels==1],independent[labels==1],rtol=1e-10,atol=1e-12)
        assert all(length>=16*h for length in v['lengths'])
        write_lines(dst/'corelines.vtp',cores)
        for before,after in zip(cores,read_cores(dst/'corelines.vtp')):np.testing.assert_array_equal(before,after)
        for name,array in arrays.items():np.save(dst/name,array)
        mapping=np.full(f['cores'],-1,np.int32)
        for current,parents in enumerate(v['parents']):mapping[parents]=current
        with np.load(src/'sample_seeds.npz') as z:attributes={name:z[name] for name in z.files}
        for name in ('sampling_target_core_id','near_core_id'):
            attributes['manual_parent_'+name]=attributes[name].copy()
            attributes[name]=remap_ids(attributes[name],mapping)
        np.savez_compressed(dst/'sample_seeds.npz',**attributes)
        prep=read(src/'prepared_input.json');prepared=Path(prep['source'])
        for name in ('coordinates.npz','ivd.npy','ivd.json'):assert sha256(prepared/name)==prep['files'][name]
        with np.load(prepared/'coordinates.npz') as z:axes=tuple(z[k] for k in ('z','y','x'))
        ivd=np.load(prepared/'ivd.npy',mmap_mode='r');ivdmeta=read(prepared/'ivd.json')
        interp=RegularGridInterpolator(axes,ivd,bounds_error=False,fill_value=0.)
        tree=cKDTree(seeds);coverage=[];support=[]
        old_support=read(src/'candidate_core_support.json')['cores']
        for cid,c in enumerate(cores):
            counts=tree.query_ball_point(c,2*h,return_length=True,workers=2)
            mask=interp(c[:,::-1])>ivdmeta['threshold']
            coverage.append(dict(core=cid,ui_id=v['ids'][cid],all_vertex_min=int(counts.min()),
                candidate_vertices=int(mask.sum()),candidate_vertex_min=int(counts[mask].min()) if mask.any() else None,
                count_quantiles=np.quantile(counts,[0,.5,1]).tolist(),includes_manual_bridge_vertices=True))
            # Existing witness points remain valid after a merge; lack of a new
            # witness is unknown, not a proof that the candidate tube is empty.
            inherited=any(old_support[i]['candidate_support'] is True for i in v['parents'][cid])
            witness=inherited or bool(mask.any()) or bool(np.any(owners==cid))
            support.append(dict(core=cid,ui_id=v['ids'][cid],parent_cores=v['parents'][cid],
                candidate_support=True if witness else None,evidence='retained witness / candidate vertex / saved positive seed' if witness else 'not recomputed; no witness found'))
        write(dst/'candidate_core_support.json',dict(cores=support,eligible=[r['core'] for r in support if r['candidate_support']],
            absent=[],unknown=[r['core'] for r in support if r['candidate_support'] is None],no_new_sampling=True))
        length=dict(version=VERSION,minimum_length_h=16.,minimum_length=16*h,keep_comparison='>=',h=h,
            lengths=v['lengths'],retained_count=len(cores),ui_core_ids=v['ids'],
            stage='Manual delete or nearest-endpoint straight bridge; no automatic gap rejection',
            total_retained_length=sum(v['lengths']),corelines_sha256=sha256(dst/'corelines.vtp'))
        write(dst/'length_filter.json',length)
        parent_labels=np.load(src/'labels.npy')
        audit=read(src/'audit.json');audit.update(version=VERSION,cores=len(cores),positive=int(labels.sum()),
            negative=int(len(labels)-labels.sum()),coverage=coverage,manual_refinement=True,
            all_labels_independently_recomputed=True,labels_independently_checked_rows=len(seeds),
            positive_to_negative=int(((parent_labels==1)&(labels==0)).sum()),
            negative_to_positive=int(((parent_labels==0)&(labels==1)).sum()),
            integration_reused_without_recomputation=True,unique_source_segments=True,
            parent_audit_sha256=sha256(src/'audit.json'))
        write(dst/'audit.json',audit)
        sampling=read(src/'sampling.json');sampling.update(version=VERSION,manual_refinement=True,
            sampling_reused=True,source_dataset=self.frozen['version'],quota_reallocated=False,
            eligible_cores=[r['core'] for r in support if r['candidate_support']],
            near_core_quota=int((attributes['sampling_target_core_id']>=0).sum()),
            note='Original seeds and curves retained; core target IDs remapped after manual edits.')
        write(dst/'sampling.json',sampling)
        write(dst/'manual_edit.json',dict(base_sha256=self.expected,operations=v['operations'],
            ui_core_ids=v['ids'],parent_core_ids=v['parents'],bridges=v['bridges']))
        result=copy.deepcopy(f)
        for stale in ('retained_parent_core_ids','candidate_vertices_without_2h_coverage_cores','still_positive_after_removed_nearest_core'):
            result.pop(stale,None)
        result.update(version=VERSION,cores=len(cores),positive=int(labels.sum()),negative=int(len(labels)-labels.sum()),
            parent_cores=f['cores'],removed_cores=f['cores']-len(cores),manual_refinement=True,
            ui_core_ids=v['ids'],manual_base_positive=f['positive'],retained_arc_length=sum(v['lengths']),
            positive_to_negative=audit['positive_to_negative'],negative_to_positive=audit['negative_to_positive'],
            positive_per_core=np.bincount(owners[owners>=0],minlength=len(cores)).tolist(),
            source_dataset=self.frozen['version'],candidate_sampler_revision='manual_refinement_reuse_2.7_base',
            candidate_unsupported_cores=[],candidate_support_unknown_cores=[r['core'] for r in support if r['candidate_support'] is None])
        for name,field in [('corelines.vtp','core_sha256'),('labels.npy','labels_sha256'),('audit.json','audit_sha256')]:
            result[field]=sha256(dst/name)
        write(dst/'completed.json',dict(passed=True,frame=completed['frame'],neighbors_complete=True,
            source_completed_sha256=sha256(src/'completed.json'),manual_refinement=True,
            files={p.name:sha256(p) for p in dst.iterdir() if p.is_file() and p.name not in ('completed.json','frame_result.json')}))
        write(dst/'frame_result.json',result)
        return result

    def _apply(self,snapshot,changed):
        try:
            previous,old_frozen,_=self._active();revisions=self.workspace/'revisions'
            revisions.mkdir(exist_ok=True)
            number=max([int(p.name[1:]) for p in revisions.glob('r[0-9]*') if p.name[1:].isdigit()]+[0])+1
            dest=revisions/f'r{number:04d}';dest.mkdir()
            inventory=read(self.base/'frame_inventory.json');inventory.update(version=VERSION,manual_revision=number)
            write(dest/'frame_inventory.json',inventory)
            for filename in ('deployment_source_sha256.json',):reuse(self.base/filename,dest/filename)
            # Later revision frames can still name an earlier frozen source manifest.
            for manifest_name in old_frozen.get('source_manifests',{}).values():
                if manifest_name!='manual_source_sha256.json':reuse(previous/manifest_name,dest/manifest_name)
            records=[]
            for f in old_frozen['frames']:
                key=f"{f['flow']}:{f['index']}";suffix=Path(f['flow'])/f"frame_{f['index']:03d}"
                if key in changed:
                    result=self._apply_frame(key,dest,snapshot)
                    with self.lock:
                        self.job['done']+=1;self.job['current_frame']=key;write(self.workspace/'job.json',self.job)
                else:
                    for p in (previous/suffix).iterdir():
                        if p.is_file():reuse(p,dest/suffix/p.name)
                    result=copy.deepcopy(f)
                for p in (self.base/'neighbors'/suffix).iterdir():
                    if p.is_file():reuse(p,dest/'neighbors'/suffix/p.name)
                records.append(result)
            source_files=['experiments/Task6_LabelEditor_1_5.py','experiments/Serve_Task6_DatasetInspector_1_7.py',
                          'experiments/Task6_DisplayStreamlines_1_6.py',
                          'FMT_Utils/Task6CorelineDataset_1_1.py','FMT_Utils/Task6Sampling_2_1.py',
                          'experiments/templates/task6_dataset_inspector_1_7.html',
                          'experiments/templates/task6_dataset_inspector_1_7.js',
                          'config/Other_Task6_DatasetInspector_1.7.json',
                          'docs/Task6_manual_label_editor_protocol_1.5.md']
            write(dest/'manual_source_sha256.json',{p:sha256(ROOT/p) for p in source_files})
            for name in source_files:
                source_copy=dest/'source_snapshot'/name;source_copy.parent.mkdir(parents=True,exist_ok=True)
                source_copy.write_bytes((ROOT/name).read_bytes())
            write(dest/'manual_edits.json',dict(base_sha256=self.expected,operations=snapshot['operations'],
                draft_revision=snapshot['revision'],created_utc=utc()))
            frozen=copy.deepcopy(self.frozen)
            frozen.update(version=VERSION,manual_revision=number,frames=records,created_utc=utc(),
                parent_version=old_frozen['version'],parent_frozen_sha256=sha256(previous/'dataset_frozen.json'),
                manual_base_frozen_sha256=self.expected,manual_edits_sha256=sha256(dest/'manual_edits.json'),
                manual_source_sha256=sha256(dest/'manual_source_sha256.json'),
                inventory_sha256=sha256(dest/'frame_inventory.json'),
                total_corelines=sum(f['cores'] for f in records),total_positive=sum(f['positive'] for f in records))
            write(dest/'dataset_frozen.json',frozen)
            write(dest/'completion_summary.json',dict(complete=True,manual_revision=number,changed_frames=changed,
                samples=frozen['total_samples'],corelines=frozen['total_corelines'],positive=frozen['total_positive'],
                frozen_sha256=sha256(dest/'dataset_frozen.json'),no_resampling=True,no_remote_training_changes=True))
            # Publication occurs only after every changed frame and complete dataset succeeds.
            with self.lock:
                assert self.state['revision']==snapshot['revision']
                write(self.workspace/'current.json',dict(relative_path=f'revisions/r{number:04d}',
                    frozen_sha256=sha256(dest/'dataset_frozen.json'),published_utc=utc()))
                self.state['revision']+=1;write(self.workspace/'draft.json',self.state)
                self.job.update(state='COMPLETED',completed_utc=utc(),dataset_path=str(dest),
                    frozen_sha256=sha256(dest/'dataset_frozen.json'),manual_revision=number)
                write(self.workspace/'job.json',self.job)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with self.lock:
                self.job.update(state='FAILED',error=str(exc),failed_utc=utc())
                write(self.workspace/'job.json',self.job)
