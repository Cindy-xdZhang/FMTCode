"""Wait for a local builder, verify completed bundles and publish their catalog."""
from pathlib import Path
import argparse,ctypes,json,os,shutil,sys,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.Build_Task6_IcosaBundles_1_1 import core_labels,guarded_build,CONFIG,EVIDENCE
from experiments.Task6_FieldStore import read,sha,atomic


def wait_process(pid):
    if os.name!='nt':raise RuntimeError('This local process waiter is Windows-specific')
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x100000,False,pid)
    if not handle:return
    try:
        while kernel.WaitForSingleObject(ctypes.c_void_p(handle),30000)==258:pass
    finally:kernel.CloseHandle(ctypes.c_void_p(handle))


def independent_labels(root,record,centers):
    p=np.load(root/record['folder']/'core_points.npy');o=np.load(root/record['folder']/'core_offsets.npy')
    aa=[p[a:b-1] for a,b in zip(o[:-1],o[1:])];bb=[p[a+1:b] for a,b in zip(o[:-1],o[1:])]
    if not aa:return np.zeros(len(centers),np.uint8)
    a=np.concatenate(aa).astype(float);b=np.concatenate(bb).astype(float);minimum=np.full(len(centers),np.inf)
    for start in range(0,len(a),4096):
        left=a[start:start+4096];v=b[start:start+4096]-left;w=centers[:,None,:]-left
        denominator=np.sum(v*v,axis=1);t=np.sum(w*v,axis=-1)/np.where(denominator>0,denominator,1)
        delta=w-np.clip(t,0,1)[...,None]*v
        minimum=np.minimum(minimum,np.sum(delta*delta,axis=-1).min(1))
    return (minimum<record['h']**2).astype(np.uint8)


def publish(config):
    root=Path(config['dataset_root']);out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True)
    marker=out/'publish.lock';fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        manifest_sha=sha(root/'dataset.json');manifest=read(root/'dataset.json');rows=[];pending=[]
        for record in manifest['frames']:
            dest=out/'frames'/record['flow']/f"frame_{record['index']:03d}";finished=dest/'complete.json'
            if not finished.exists():pending.append(record['key']);continue
            complete=read(finished)
            for n,h in complete['files'].items():
                if sha(dest/n)!=h:raise ValueError('Bundle checksum mismatch: '+str(dest/n))
            meta=read(dest/'metadata.json');count=config['centers_per_frame'];k=config['neighbors']+1
            curves=np.load(dest/'curves.npy',mmap_mode='r');centers=np.load(dest/'centers.npy',mmap_mode='r')
            assert curves.shape==(count,k,65,3) and curves.dtype==np.float32
            assert centers.shape==(count,3) and centers.dtype==np.float64
            for first in range(0,count,512):assert np.isfinite(curves[first:first+512]).all()
            for n,h in meta['source_files'].items():
                assert manifest['files_sha256'][record['folder']+'/'+n]==h
                assert sha(root/record['folder']/n)==h
            labels,core_sha=core_labels(root,record,centers)
            selected=np.linspace(0,count-1,min(count,64)).astype(int)
            np.testing.assert_array_equal(labels[selected],independent_labels(root,record,centers[selected]))
            # Human geometry may have changed during a long integration run.
            if meta['coreline_sha256']!=core_sha or not np.array_equal(np.load(dest/'labels.npy'),labels):
                finished.unlink()
                temp=dest/'labels.next.npy';np.save(temp,labels);os.replace(temp,dest/'labels.npy')
                meta.update(coreline_sha256=core_sha,positive=int(labels.sum()));atomic(dest/'metadata.json',meta)
                complete['files'].update({n:sha(dest/n) for n in ['labels.npy','metadata.json']});atomic(finished,complete)
            rows.append(dict(key=record['key'],folder=dest.relative_to(out).as_posix(),samples=count,
                             positive=int(labels.sum()),coreline_sha256=core_sha,files=complete['files']))
            del curves,centers
        if sha(root/'dataset.json')!=manifest_sha:raise RuntimeError('Human labels changed during final verification; rerun publication')
        source=ROOT/'experiments/task6_field_tools/task6_bundles.py';shutil.copyfile(source,out/'task6_bundles.py')
        catalog=dict(version=config['version'],state='COMPLETE' if not pending else 'PARTIAL',
                     source_manifest_sha256=manifest_sha,centers_per_frame=config['centers_per_frame'],
                     requested_frames=len(manifest['frames']),completed_frames=len(rows),pending_frames=pending,
                     frames=rows,configuration=config,reader_sha256=sha(out/'task6_bundles.py'))
        atomic(out/'bundles.json',catalog);atomic(EVIDENCE/'publication.json',catalog)
        status='全部完成' if not pending else f'尚未全部完成：{len(rows)}/{len(manifest["frames"])} 帧可读取'
        description=f'''# Task6 预积分流线簇

状态：**{status}**。实际可读取帧及文件哈希见 `bundles.json`；缺少 `complete.json` 的帧是构建中间结果，不能当完整样本读取。

这是用户明确授权的可重算附加数据。原 `frames/` 中完整速度场、物理坐标和人工核线仍是核心数据。本轮每帧 {config['centers_per_frame']} 个中心，每个中心有20个半径h的正二十面体面心方向邻居，共21条流线。中心必须满足原v的IVD严格大于规定比例的帧最大值；具体阈值及有效区见每帧metadata。其余邻居不要求满足IVD阈值，但所有种子须在可用物理域内。

`curves.npy` 是 float32 `[10000,21,65,3]`，第0条为中心流线，后20条为邻居。归一化瞬时速度的双向弧长积分，每方向目标0.25，最多20000次自适应尝试；保存h/16结果，容限h×1e-8，并通过h/8和h/32误差比较≤h/20。合并双向原始点后等弧长重采样65点，原始积分点不保存。真实中心在 `centers.npy`，65点的中间点不一定是中心。

`labels.npy` 为中心到当前人工核线连续线段距离严格小于h的二分类标签。人工修改核线后仅需更新这些标签，不重算流线；读取器会拒绝旧标签。`half_lengths.npy`、`termination.npy`、`refinement_errors.npy` 分别记录每条线双向实际长度、终止原因和数值核验误差。全部几何使用真实物理xyz坐标。

在共享包根目录运行：

```python
import sys
sys.path.insert(0, 'preintegrated/icosa_bundles_1.1')
from task6_bundles import Bundles, integrate_bundle
from task6_fields import Dataset
bundles = Bundles('.')
sample = bundles.frame('deltaWing_resampled:9', verify=True)
x = sample['curves'][0]  # [21,65,3]，只读内存映射
y = sample['labels'][0]
field = Dataset('.').frame('deltaWing_resampled:9')
recomputed = integrate_bundle(field, sample['centers'][0], sample['metadata']['unit_offsets'])
assert recomputed['valid']
```

101帧的流线坐标预计16.5438GB（15.408GiB）；所有附加数组约16.87GB，原完整场和核线另计。40%阈值仅是被用户否决的SquareCylinder试算，不是正式协议。
'''
        (out/'README.md').write_text(description,encoding='utf-8')
        print(json.dumps(dict(state=catalog['state'],complete=len(rows),pending=pending)),flush=True)
    finally:marker.unlink()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--wait-pid',type=int,action='append',default=[]);args=parser.parse_args()
    for pid in args.wait_pid:wait_process(pid)
    config=read(args.config)
    # Recover interrupted/checkpoint-I/O frames once before final verification.
    records=read(Path(config['dataset_root'])/'dataset.json')['frames'];out=Path(config['output_root'])
    pending=[r for r in records if not (out/'frames'/r['flow']/f"frame_{r['index']:03d}"/'complete.json').exists()]
    if pending:
        os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS=str(config['threads_per_worker']))
        repaired=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(config['workers'],len(pending))) as pool:
            futures=[pool.submit(guarded_build,config,r) for r in pending]
            for future in concurrent.futures.as_completed(futures):
                row=future.result();repaired.append(row);print(json.dumps(row),flush=True)
                atomic(EVIDENCE/'completion_retries.json',dict(rows=repaired,total=len(pending)))
    for attempt in range(3):
        try:publish(config);break
        except RuntimeError as e:
            if 'changed' not in str(e) or attempt==2:raise
            time.sleep(2)


if __name__=='__main__':main()
