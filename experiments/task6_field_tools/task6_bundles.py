"""Read optional Task6 preintegrated bundles without loading all curves into RAM."""
from pathlib import Path
import hashlib,json
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class Bundles:
    def __init__(self, dataset_root, bundle_folder='preintegrated/icosa_bundles_1.1'):
        self.dataset_root=Path(dataset_root)
        self.root=self.dataset_root/bundle_folder
        self.dataset=json.loads((self.dataset_root/'dataset.json').read_text(encoding='utf-8'))
        self.records={r['key']:r for r in self.dataset['frames']}

    def frame(self,key,verify=False):
        r=self.records[key];folder=self.root/'frames'/r['flow']/f"frame_{r['index']:03d}"
        complete=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        if not complete.get('passed'):raise ValueError('Frame bundles are incomplete')
        if verify:
            for name,digest in complete['files'].items():
                if sha(folder/name)!=digest:raise ValueError('Bundle checksum mismatch: '+name)
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        for name,digest in meta['source_files'].items():
            if self.dataset['files_sha256'][r['folder']+'/'+name]!=digest:
                raise ValueError('Bundles were integrated in a different field or grid')
        current_core=self.dataset['files_sha256'][r['folder']+'/corelines.vtp']
        if meta['coreline_sha256']!=current_core:
            raise ValueError('Center labels predate the current human corelines; refresh labels before training')
        return {name:np.load(folder/(name+'.npy'),mmap_mode='r') for name in
                ['curves','centers','labels','half_lengths','termination','refinement_errors']} | {'metadata':meta}


def integrate_bundle(frame,center,unit_offsets,threads=8):
    """Reproduce the frozen 21-line integration using the full field package."""
    from numba import set_num_threads,get_num_threads
    from tools.adaptive_streamlines import integrate_samples_adaptive
    center=np.asarray(center,float);offsets=np.asarray(unit_offsets,float)
    seeds=np.ascontiguousarray(center[None,:]+frame.h*offsets)
    if np.any(seeds<frame.bounds[:,0]) or np.any(seeds>frame.bounds[:,1]):
        raise ValueError('A neighbor seed is outside the physical field')
    old=get_num_threads();set_num_threads(min(old,threads))
    try:
        results=[integrate_samples_adaptive(frame.velocity,seeds,frame.bounds[:,0],frame.spacing,
                                           .25,frame.h/divisor,frame.h*tol,65)
                 for divisor,tol in [(8,1e-7),(16,1e-8),(32,1e-9)]]
    finally:set_num_threads(old)
    coarse,fine,finest=[r[0] for r in results]
    error=np.maximum(np.linalg.norm(coarse-fine,axis=-1).max(1),np.linalg.norm(fine-finest,axis=-1).max(1))
    valid=all(np.isfinite(r[0]).all() and (r[2]<3).all() for r in results)
    valid=bool(valid and (results[1][1].sum(1)>frame.h).all() and (error<=frame.h/20).all())
    return dict(curves=fine,valid=valid,seeds=seeds,half_lengths=results[1][1],errors=error)
