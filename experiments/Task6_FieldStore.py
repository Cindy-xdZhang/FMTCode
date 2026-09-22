"""One canonical field dataset with durable core drafts and recoverable publication."""
from pathlib import Path
from collections import OrderedDict
from contextlib import contextmanager
import copy,json,os,shutil,threading,uuid,time
import numpy as np
from experiments.Task6_LabelEditor_1_5 import replay,Conflict,digest,utc
from experiments.Export_Task6_FieldPackage_1_1 import read,sha
from experiments.Extract_Task6_VortexCore_1_2 import write_lines
from experiments.Filter_Task6_CoreLength_1_4 import read_cores

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    # Windows readers and OneDrive can briefly hold a file without delete sharing.
    for attempt in range(50):
        try:os.replace(temp,path);break
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)

class Store:
    def __init__(self,root):
        self.root=Path(root).resolve();self.workspace=self.root/'annotations/coreline'
        self.workspace.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.cache=OrderedDict();self.job=None
        with self.writer():
            self.recover()
            self.reload()
            if not (self.workspace/'draft.json').exists():
                edits=read(self.root/'manual_edits.json')
                atomic(self.workspace/'draft.json',dict(revision=0,operations=edits['operations']))
            self.state=read(self.workspace/'draft.json')

    @contextmanager
    def writer(self):
        with self.lock:
            lockfile=self.workspace/'writer.lock'
            if lockfile.exists() and os.name=='nt':
                import ctypes
                pid=int(lockfile.read_text());kernel=ctypes.windll.kernel32
                kernel.OpenProcess.restype=ctypes.c_void_p
                handle=kernel.OpenProcess(0x1000,False,pid)
                if handle:
                    code=ctypes.c_ulong();kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code));kernel.CloseHandle(ctypes.c_void_p(handle))
                    alive=code.value==259
                else:alive=kernel.GetLastError()!=87
                if not alive:lockfile.unlink()
            try:fd=os.open(lockfile,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            except FileExistsError:raise Conflict('另一个发布正在进行，请稍后重试；如服务异常退出，先核查并恢复事务。')
            os.write(fd,str(os.getpid()).encode());os.close(fd)
            try:yield
            finally:lockfile.unlink()

    def recover(self):
        marker=self.workspace/'transaction.json'
        if not marker.exists():return
        transaction=read(marker)
        for entry in transaction['files']:
            if entry.get('delete'):
                target=self.root/entry['target']
                if not target.resolve().is_relative_to(self.root):raise ValueError('Invalid deletion path')
                target.unlink(missing_ok=True);continue
            target=self.root/entry['target'];stage=self.root/entry['stage']
            if stage.exists():
                if sha(stage)!=entry['sha256']:raise ValueError('Invalid publication staging file')
                target.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,target)
            elif not target.exists() or sha(target)!=entry['sha256']:raise ValueError('Incomplete publication cannot be recovered')
        atomic(self.workspace/'last_publication.json',transaction);marker.unlink()

    def reload(self):
        self.manifest=read(self.root/'dataset.json');self.expected=sha(self.root/'dataset.json')
        self.records={r['key']:r for r in self.manifest['frames']}
        self.split=read(self.root/'default_split.json')
        if self.split['dataset_sha256']!=self.expected or read(self.root/'current.json')['sha256']!=self.expected:raise ValueError('Published pointer/split mismatch')
        self.applied=read(self.root/'manual_edits.json')['operations']

    def record(self,key):return self.records[key]
    def active_folder(self,key):return self.root/self.record(key)['folder']
    def base_folder(self,key):
        f=self.record(key);return self.root/'provenance/automatic_corelines'/f['flow']/f"frame_{f['index']:03d}"

    def preview(self,key):
        with self.lock:
            ops=self.state['operations'].get(key,[]);signature=(key,digest(ops))
            if signature not in self.cache:
                value=replay(read_cores(self.base_folder(key)/'corelines.vtp'),ops,self.record(key)['h'])
                value.update(lengths=[float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum()) for c in value['cores']],operations=copy.deepcopy(ops))
                self.cache[signature]=value
                while len(self.cache)>4:self.cache.popitem(last=False)
            return self.cache[signature]

    def status(self):
        with self.lock:
            pending=[k for k in self.records if self.state['operations'].get(k,[])!=self.applied.get(k,[])]
            return dict(draft_revision=self.state['revision'],pending_frames=pending,pending_count=len(pending),dataset=self.manifest['version'],dataset_path=str(self.root),dataset_sha256=self.expected,applied_revision=self.manifest.get('manual_revision',1),job=self.job)

    def edit(self,key,operation,revision):
        with self.writer():
            if revision!=self.state['revision']:raise Conflict('草稿已更新，请刷新')
            self.record(key);state=copy.deepcopy(self.state);ops=state['operations'].setdefault(key,[]);kind=operation.get('kind')
            if kind in ('delete','merge'):ops.append({k:operation[k] for k in (('kind','first','second') if kind=='merge' else ('kind','first'))})
            elif kind=='undo':
                if not ops:raise ValueError('没有可撤销的操作')
                ops.pop()
            elif kind=='reset':state['operations'][key]=[]
            else:raise ValueError('Unknown operation')
            replay(read_cores(self.base_folder(key)/'corelines.vtp'),state['operations'][key],self.record(key)['h'])
            state['revision']+=1;state['updated_utc']=utc();atomic(self.workspace/'draft.json',state);self.state=state
            return self.status()

    def commit(self,changed_files,manifest,split,description):
        """Caller owns writer lock. Keep small before-images and roll forward after crash."""
        if sha(self.root/'dataset.json')!=self.expected:raise Conflict('数据集已更新，请重载后发布')
        used=[int(p.name[1:]) for p in (self.workspace/'revisions').glob('r[0-9]*') if p.name[1:].isdigit()]
        rid='r'+str(max(used+[manifest.get('publication_revision',0)])+1).zfill(4)
        history=self.workspace/'revisions'/rid
        if history.exists():raise ValueError('Publication history already exists')
        history.mkdir(parents=True);stage=history/'staged';stage.mkdir()
        manifest=copy.deepcopy(manifest);manifest['publication_revision']=int(rid[1:]);manifest['updated_utc']=utc()
        entries=[]
        for target,source in changed_files.items():
            if source is None:
                manifest['files_sha256'].pop(target,None);continue
            source=Path(source);manifest['files_sha256'][target]=sha(source)
            destination=stage/target;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        atomic(stage/'dataset.json',manifest);newsha=sha(stage/'dataset.json')
        split=copy.deepcopy(split);split['dataset_sha256']=newsha;atomic(stage/'default_split.json',split)
        atomic(stage/'current.json',dict(manifest='dataset.json',sha256=newsha,publication_revision=rid))
        ordered=[*changed_files,'dataset.json','default_split.json','current.json']
        for target in ordered:
            old=self.root/target
            if not old.resolve().is_relative_to(self.root):raise ValueError('Publication path outside dataset')
            if old.exists():
                backup=history/'before'/target;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,backup)
            if target in changed_files and changed_files[target] is None:entries.append(dict(target=target,delete=True))
            else:entries.append(dict(target=target,stage=(stage/target).relative_to(self.root).as_posix(),sha256=sha(stage/target)))
        transaction=dict(revision=rid,utc=utc(),description=description,previous_manifest_sha256=self.expected,manifest_sha256=newsha,files=entries)
        atomic(history/'publication.json',transaction);atomic(self.workspace/'transaction.json',transaction)
        self.recover();self.reload()
        config=Path(__file__).resolve().parents[1]/'config/Task6_current_dataset.json'
        if config.exists():
            settings=read(config)
            if Path(settings['dataset_root']).resolve()==self.root:
                settings['manifest_sha256']=self.expected;atomic(config,settings)

    def start_apply(self,revision):
        with self.writer():
            if revision!=self.state['revision']:raise Conflict('草稿已更新，请刷新')
            changed=self.status()['pending_frames']
            if not changed:raise ValueError('没有尚未应用的修改')
            temp=self.workspace/('work-'+uuid.uuid4().hex);temp.mkdir();files={};manifest=copy.deepcopy(self.manifest)
            records={r['key']:r for r in manifest['frames']}
            for key in changed:
                record=records[key];v=self.preview(key)
                if not all(x>=16*record['h'] for x in v['lengths']):raise ValueError('核线弧长不足16h')
                dest=temp/record['folder'];dest.mkdir(parents=True)
                write_lines(dest/'corelines.vtp',v['cores'])
                loaded=read_cores(dest/'corelines.vtp');assert len(loaded)==len(v['cores'])
                for a,b in zip(loaded,v['cores']):np.testing.assert_array_equal(a,b)
                np.save(dest/'core_points.npy',np.concatenate(loaded) if loaded else np.empty((0,3),np.float64))
                np.save(dest/'core_offsets.npy',np.r_[0,np.cumsum([len(c) for c in loaded])].astype(np.int64))
                record.update(core_count=len(loaded),ui_core_ids=v['ids'],source_corelines_sha256=sha(dest/'corelines.vtp'))
                for p in dest.iterdir():files[p.relative_to(temp).as_posix()]=p
            edits=read(self.root/'manual_edits.json');edits.update(operations=self.state['operations'],draft_revision=revision)
            atomic(temp/'manual_edits.json',edits);files['manual_edits.json']=temp/'manual_edits.json'
            manifest['manual_revision']=manifest.get('manual_revision',1)+1
            self.commit(files,manifest,self.split,dict(kind='coreline_edits',frames=changed,draft_revision=revision))
            self.job=dict(state='COMPLETED',done=len(changed),total=len(changed),dataset_path=str(self.root),frozen_sha256=self.expected)
            return self.job

    def add_frames(self,staging,rows,role='unused'):
        with self.writer():
            if role not in ('train','test','unused'):raise ValueError('Invalid frame role')
            if self.status()['pending_count']:raise Conflict('先应用核线草稿再扩展帧')
            manifest=copy.deepcopy(self.manifest);split=copy.deepcopy(self.split);files={}
            for row in rows:
                key=f"{row['flow']}:{row['index']}"
                if key in self.records:raise ValueError('Frame already exists: '+key)
                source=Path(staging)/row['flow']/f"frame_{row['index']:03d}";complete=read(source/'complete.json')
                for n,h in complete['files'].items():assert sha(source/n)==h
                record=complete['frame'];manifest['frames'].append(record);split[role].append(key)
                names=['relative_velocity.npy','corelines.vtp','coordinates.npz','core_points.npy','core_offsets.npy','ivd.npy','ivd.json','observer_provenance.json']
                for name in names:files[record['folder']+'/'+name]=source/name
                files[f"provenance/automatic_corelines/{row['flow']}/frame_{row['index']:03d}/corelines.vtp"]=source/'corelines.vtp'
            manifest['frames'].sort(key=lambda r:(r['flow'],r['index']))
            self.commit(files,manifest,split,dict(kind='add_frames',role=role,keys=[f"{r['flow']}:{r['index']}" for r in rows]))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['add'])
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--role',choices=['train','test','unused'],default='unused')
    args=parser.parse_args();plan=read(args.plan)
    store=Store(plan['dataset_root']);store.add_frames(plan['staging'],plan['frames'],args.role)
    print(json.dumps(store.status()))
