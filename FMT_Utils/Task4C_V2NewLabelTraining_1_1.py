"""Four complete training folds and one test fold; frozen model inputs."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from FMT_Utils import Task4C_V3Training_3_1 as old


def split_masks(meta, flow_index, spec):
    fold = np.asarray(meta['fold'])
    assert np.isin(fold, np.arange(5)).all()
    test = fold == spec['folds']['test_fold']
    return dict(train=~test, test=test)


class Source(old.Source):
    def __init__(self, spec, fi, k):
        name = spec['flows'][fi]['name']
        folder = Path(spec['source_output'])/'physical'/name
        self.seeds = np.load(folder/'seeds.npy', mmap_mode='r')
        self.curves = np.load(folder/'curves.npy', mmap_mode='r')
        self.neighbors = np.load(Path(spec['neighbor_output'])/name/f'order{k}.npy', mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:
            self.meta = {key:z[key] for key in ('label', 'instance', 'fold', 'assignment_kind')}
        self.masks = split_masks(self.meta, fi, spec)
        self.k = k


class Dataset(old.Dataset):
    def __init__(self, spec, role, candidate, device='cuda', limit=None):
        assert role in ('train', 'test'), 'No validation partition is permitted'
        self.device, self.role, self.limit = device, role, limit
        self.evidence = None if limit else Path(spec['run_folder'])
        self.parts = []
        fields = {key:[] for key in ('labels', 'owners', 'flows', 'rows', 'lengths', 'fold', 'assignment_kind')}
        for fi, flow in enumerate(spec['flows']):
            source = Source(spec, fi, candidate['neighbors'])
            ids = np.flatnonzero(source.masks[role])
            if limit:
                ids = np.concatenate([ids[source.meta['label'][ids] == c][:limit//4] for c in (0, 1)])
            samples = np.repeat(ids, 3); lengths = np.tile(np.arange(3), len(ids))
            self.parts.append((source, samples, lengths))
            for key, values in dict(labels=source.meta['label'][samples], owners=source.meta['instance'][samples],
                    flows=np.full(len(samples), fi), rows=samples, lengths=lengths,
                    fold=source.meta['fold'][samples], assignment_kind=source.meta['assignment_kind'][samples]).items():
                fields[key].append(values)
        for key in ('labels', 'owners', 'flows', 'rows', 'lengths'):
            setattr(self, key, np.concatenate(fields[key]))
        self.labels = self.labels.astype(np.int64)
        self.extra = {key:np.concatenate(fields[key]) for key in ('fold', 'assignment_kind')}
        assert limit or len(self.labels) == spec['expected_counts'][role]
        self.targets = torch.tensor(self.labels, device=device)
        self.neighbors = torch.empty((len(self.labels), 0), device=device, dtype=torch.long)
        self.clean = None

    def geometry(self, rows):
        parts = []
        for source, samples, lengths in self.parts:
            parts.append(dict(source=source, samples=samples[::3]))
        boundaries = np.cumsum([len(p[1]) for p in self.parts])
        rows = np.asarray(rows)
        flows = np.searchsorted(boundaries, rows, side='right')
        starts = np.r_[0, boundaries[:-1]]
        pairs = np.column_stack((flows, rows-starts[flows]))
        return old.baseline_geometry(parts, pairs)

    def cache_voxels(self, engine, batch=128):
        """Compute frozen float16 sparse voxels once, then keep sparse arrays in RAM."""
        dest = self.evidence/f'{self.role}_voxels'; dest.mkdir(parents=True, exist_ok=False)
        offsets = [0]; started = time.perf_counter()
        with (dest/'indices.bin').open('wb') as ix, (dest/'values.bin').open('wb') as val:
            for first in range(0, len(self.labels), batch):
                g, c = self.geometry(np.arange(first, min(first+batch, len(self.labels))))
                off, ids, values = engine.sparse_voxels(torch.tensor(g, device=self.device), torch.tensor(c, device=self.device), 16)
                offsets.extend((off[1:]+offsets[-1]).tolist())
                np.asarray(ids, dtype=np.int32).tofile(ix)
                np.asarray(values, dtype=np.float16).tofile(val)
                if first % 131072 == 0:
                    print(json.dumps(dict(stage='voxel_cache', role=self.role, completed=first+len(g),
                        total=len(self.labels), seconds=time.perf_counter()-started)), flush=True)
        offsets = np.asarray(offsets, np.int64); np.save(dest/'offsets.npy', offsets)
        self.voxels = dict(offsets=offsets, indices=np.fromfile(dest/'indices.bin', np.int32),
            values=np.fromfile(dest/'values.bin', np.float16).reshape(-1, 4))
        assert offsets[-1] == len(self.voxels['indices']) == len(self.voxels['values'])
        (dest/'manifest.json').write_text(json.dumps(dict(complete=True, samples=len(self.labels),
            occupied=int(offsets[-1]), resolution=16, dtype='float16', seconds=time.perf_counter()-started)))

    def voxel_batch(self, rows, engine):
        ids = np.column_stack((np.zeros(len(rows), np.int64), rows))
        return engine.dense_sparse_batch([self.voxels], ids, 16, self.device)
