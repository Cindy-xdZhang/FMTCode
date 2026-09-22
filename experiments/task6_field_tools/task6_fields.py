"""Frame-level observed fields and coreline labels; integrations stay in memory."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
class Frame:
    def __init__(self,root,record):
        self.root=Path(root)/record['folder'];self.record=record
        self.velocity=np.load(self.root/'relative_velocity.npy',mmap_mode='r')
        with np.load(self.root/'coordinates.npz') as z:self.axes_xyz=[z[a].copy() for a in 'xyz']
        self.bounds=np.array([[a[0],a[-1]] for a in self.axes_xyz])
        self.spacing=np.array([(a[-1]-a[0])/(len(a)-1) for a in self.axes_xyz]);self.h=float(min(self.spacing))
        if self.velocity.shape!=tuple(map(len,self.axes_xyz[::-1]))+(3,):raise ValueError('Axis order mismatch')
    def corelines(self):
        p=np.load(self.root/'core_points.npy');o=np.load(self.root/'core_offsets.npy')
        return [p[a:b] for a,b in zip(o[:-1],o[1:])]
    def integrate(self,seeds,total_length,points=65,threads=8):
        """Bidirectional arc-length integration; no curve files are written."""
        from numba import set_num_threads,get_num_threads
        from tools.adaptive_streamlines import integrate_samples_adaptive
        seeds=np.ascontiguousarray(seeds,np.float64)
        if seeds.ndim!=2 or seeds.shape[1]!=3 or not np.isfinite(seeds).all():raise ValueError('Seeds must be finite physical xyz points')
        if not np.isfinite(total_length) or total_length<=0 or points<2 or threads<1:raise ValueError('Invalid integration parameters')
        if np.any(seeds<self.bounds[:,0]) or np.any(seeds>self.bounds[:,1]):raise ValueError('Seeds outside the physical field')
        old=get_num_threads();set_num_threads(min(old,threads))
        try:
            coarse,_,coarse_reason=integrate_samples_adaptive(self.velocity,seeds,self.bounds[:,0],self.spacing,
                total_length/2,self.h/4,self.h*1e-7,points)
            fine,lengths,reasons=integrate_samples_adaptive(self.velocity,seeds,self.bounds[:,0],self.spacing,
                total_length/2,self.h/8,self.h*1e-8,points)
        finally:set_num_threads(old)
        error=np.linalg.norm(fine-coarse,axis=-1).max(axis=1)
        valid=np.isfinite(fine).all(axis=(1,2))&(reasons<=1).all(axis=1)&(coarse_reason<=1).all(axis=1)&(error<=self.h/20)&(lengths.sum(1)>self.h/100)
        return dict(curves=fine,seeds=seeds,valid=valid,half_lengths=lengths,reasons=reasons,error=error,
                    total_length=total_length,saved_to_disk=False)
class Dataset:
    def __init__(self,root='.',split='default_split.json'):
        self.root=Path(root).resolve();self.manifest=read(self.root/'dataset.json')
        self.records={f['key']:f for f in self.manifest['frames']};self.split=read(self.root/split)
        if self.split['dataset_sha256']!=sha(self.root/'dataset.json'):raise ValueError('Split refers to another snapshot')
        keys=[k for role in ('train','test','unused') for k in self.split[role]]
        if len(keys)!=len(set(keys)) or set(keys)!=set(self.records):raise ValueError('Frame splits must be disjoint and cover every frame')
    def frames(self,role=None):return list(self.records.values()) if role is None else [self.records[k] for k in self.split[role]]
    def frame(self,key):return Frame(self.root,self.records[key])
    def verify(self):
        for name,digest in self.manifest['files_sha256'].items():
            if sha(self.root/name)!=digest:raise ValueError('Hash mismatch: '+name)
        return dict(verified=True,frames=len(self.records),files=len(self.manifest['files_sha256']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','list']);p.add_argument('--root',default=str(Path(__file__).resolve().parent));a=p.parse_args()
    ds=Dataset(a.root);print(json.dumps(ds.verify() if a.command=='verify' else ds.frames(),indent=2))
