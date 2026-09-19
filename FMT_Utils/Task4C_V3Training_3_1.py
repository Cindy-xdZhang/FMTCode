"""Streaming v3 data adapters; retain frozen encoders, normalization and model inputs."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as v2
from FMT_Utils import Task4C_FPS16_1_1 as fmt
from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt
from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_voxels


class Source:
    def __init__(self, spec, fi, k):
        name = spec['flows'][fi]['name']
        folder = Path(spec['source_output'])/'physical'/name
        self.seeds = np.load(folder/'seeds.npy', mmap_mode='r')
        self.curves = np.load(folder/'curves.npy', mmap_mode='r')
        self.neighbors = np.load(Path(spec['neighbor_output'])/name/f'order{k}.npy', mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:
            self.meta = {key:z[key] for key in ('label','instance','fold','assignment_kind')}
        self.masks = v2.split_masks(self.meta, fi, spec)
        self.k = k

    def gather(self, samples, lengths):
        ids = np.column_stack((samples, self.neighbors[samples]))
        return self.curves[ids, lengths[:, None]], self.seeds[ids], ids


class HostTokens:
    """Only the requested batch is transferred to the frozen CUDA training loop."""
    def __init__(self, array, device):
        self.array, self.device, self.shape = array, torch.device(device), array.shape

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor): index=index.cpu().numpy()
        return torch.as_tensor(np.array(self.array[index]), device=self.device)


class Dataset:
    def __init__(self, spec, role, candidate, device='cuda', limit=None):
        if role == 'test':
            lock = Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"/'selection.lock.json'
            assert json.loads(lock.read_text())['test_loaded'] is False
        self.device, self.role, self.limit = device, role, limit
        self.evidence = None if limit else Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"
        self.parts=[]; fields={key:[] for key in ('labels','owners','flows','rows','lengths','fold','assignment_kind')}
        for fi, flow in enumerate(spec['flows']):
            source=Source(spec,fi,candidate['neighbors']); ids=np.flatnonzero(source.masks[role])
            if limit: ids=np.concatenate([ids[source.meta['label'][ids]==c][:limit//4] for c in (0,1)])
            samples=np.repeat(ids,3); lengths=np.tile(np.arange(3),len(ids))
            self.parts.append((source,samples,lengths))
            for key, values in dict(labels=source.meta['label'][samples], owners=source.meta['instance'][samples],
                flows=np.full(len(samples),fi), rows=samples, lengths=lengths,
                fold=source.meta['fold'][samples], assignment_kind=source.meta['assignment_kind'][samples]).items(): fields[key].append(values)
        for key in ('labels','owners','flows','rows','lengths'): setattr(self,key,np.concatenate(fields[key]))
        self.labels=self.labels.astype(np.int64)
        self.extra={key:np.concatenate(fields[key]) for key in ('fold','assignment_kind')}
        assert limit or len(self.labels)==spec['expected_counts'][role]
        self.targets=torch.tensor(self.labels,device=device)
        self.neighbors=torch.empty((len(self.labels),0),device=device,dtype=torch.long)
        self.clean=None

    def encode(self,candidate,batch=128):
        started=time.perf_counter()
        shape=(len(self.labels),1,142)
        if self.limit: values=np.empty(shape,np.float32)
        else:
            self.evidence.mkdir(parents=True,exist_ok=True)
            values=np.lib.format.open_memmap(self.evidence/f'{self.role}_features.npy',mode='w+',dtype=np.float32,shape=shape)
        offset=0; digest=hashlib.sha256(); selected=[]
        with torch.no_grad():
            for source,samples,lengths in self.parts:
                for first in range(0,len(samples),batch):
                    lines,seeds,ids=source.gather(samples[first:first+batch],lengths[first:first+batch])
                    g,s,c=v2.normalize_bundle(torch.tensor(lines,device=self.device),torch.tensor(seeds,device=self.device))
                    tokens,used=fmt.encode(g,s,c,torch.zeros(len(g),device=self.device,dtype=torch.long),candidate)
                    assert torch.equal(used.sort(1).values,torch.arange(1,candidate['neighbors']+1,device=self.device)[None].expand(len(g),-1))
                    result=tokens.cpu().numpy()
                    if candidate['base_method']=='p35_h0': result[...,:141]=transform_fmt(result[...,:141],True)
                    assert np.isfinite(result).all()
                    values[offset:offset+len(g)]=result;offset+=len(g)
                    digest.update(ids[:,1:].astype('<i8').tobytes())
                    if self.limit: selected.append(ids[:,1:])
                    elif offset % 131072 < batch:
                        print(json.dumps(dict(stage='encode',role=self.role,completed=offset,total=len(self.labels),seconds=time.perf_counter()-started)),flush=True)
        assert offset==len(self.labels)
        if self.limit:
            self.clean=torch.tensor(values,device=self.device);self.selected_neighbors=np.concatenate(selected)
        else:
            values.flush();self.clean=HostTokens(values,self.device)
            (self.evidence/f'{self.role}_encoding.json').write_text(json.dumps(dict(shape=list(shape),samples=len(values),
                neighbor_ids_sha256=digest.hexdigest(),center_policy='original_seed_only',neighbors=candidate['neighbors']),indent=2)+'\n')


def baseline_parts(spec,role,method,limit=None):
    parts=[]; pairs=[]; labels=[]; flows=[]; owners=[]
    for fi,flow in enumerate(spec['flows']):
        source=Source(spec,fi,spec['neighbor_count']); samples=np.flatnonzero(source.masks[role])
        if limit: samples=np.concatenate([samples[source.meta['label'][samples]==c][:limit//4] for c in (0,1)])
        n=len(samples)*3
        parts.append(dict(source=source,samples=samples))
        pairs.append(np.column_stack((np.full(n,fi),np.arange(n))))
        labels.append(np.repeat(source.meta['label'][samples],3)); owners.append(np.repeat(source.meta['instance'][samples],3));flows.append(np.full(n,fi))
    return parts,np.concatenate(pairs),np.concatenate(labels).astype(np.int64),np.concatenate(flows),np.concatenate(owners)


def baseline_geometry(parts,ids):
    # Match the old CPU export exactly, including 27 slots and float32 seed coordinates.
    geometry=np.zeros((len(ids),27,32,3),np.float32);counts=np.empty(len(ids),np.int64)
    for fi in np.unique(ids[:,0]):
        take=np.flatnonzero(ids[:,0]==fi);p=parts[int(fi)];rows=ids[take,1]
        source=p['source']; samples=p['samples'][rows//3]; lengths=rows%3
        lines,seeds,_=source.gather(samples,lengths)
        g,_,c=v2.normalize_bundle(torch.tensor(lines),torch.tensor(seeds,dtype=torch.float32))
        geometry[take,:source.k+1]=g.numpy();counts[take]=c.numpy()
    return geometry,counts


def baseline_batch(parts,ids,method,norm,device='cuda'):
    assert norm is None
    g,c=baseline_geometry(parts,ids)
    g=torch.tensor(g,device=device);c=torch.tensor(c,device=device)
    if method=='conv16':
        # Identical float16 round-trip as frozen sparse-voxel cache; computed per batch.
        return bundle_voxels(g,c,16,4).half().float()
    assert method in ('baseline2','pointnet_small')
    return g,c
