"""Fixed-center FMT (p35/h0 and c156) on the Task4-c fixed dataset v2 (2.1): FPS neighbour clusters.

Dataset v2 stores one seed point and three vortex lines (three half lengths) per sample and no
neighbour lines.  This module builds the local cluster of a sample from the OTHER samples of the
same flow: the 64 nearest sample points (KD-tree on the seeds) are the candidate pool, farthest
point sampling (FPS) started at the center orders them, and the first 6 or 16 form the cluster.
Every (sample, half length) pair is one classification row; the neighbour lines use the same half
length as the center line.  The frozen encoders of Task4C_FPS16_1_1 / Task4C_OriginalCenter_1_1
receive exactly the k chosen neighbours, so their internal selection is the identity.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_FPS16_1_1 as method
from FMT_Utils import Task4C_FixedDataset_2_1 as data
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import normalize
from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt

ROLES = ('train', 'validation', 'test')
NEIGHBOR_FILE = 'neighbors_{flow}.npz'


# ---------------------------------------------------------------- neighbour clusters
def fps_order(seeds, candidates, k, chunk=2048):
    """FPS over each row's candidate points, started at the row's own seed; returns sample ids [n, k]."""
    seeds = np.asarray(seeds, np.float64); candidates = np.asarray(candidates, np.int64); n, K = candidates.shape
    assert k <= K; out = np.empty((n, k), np.int64)
    for first in range(0, n, chunk):
        ids = candidates[first:first+chunk]; b = np.arange(len(ids)); points = seeds[ids]
        minimum = np.linalg.norm(points-seeds[first:first+chunk][:, None], axis=-1)
        pairwise = np.linalg.norm(points[:, :, None]-points[:, None], axis=-1)
        excluded = np.zeros((len(ids), K), bool)
        for j in range(k):
            pick = np.where(excluded, -np.inf, minimum).argmax(1)
            out[first+b, j] = ids[b, pick]; excluded[b, pick] = True
            minimum = np.minimum(minimum, pairwise[b, pick])
    return out


def neighbor_table(seeds, pool, k):
    """Candidate pool = ``pool`` nearest other samples of the same flow; FPS order of length ``k``."""
    seeds = np.asarray(seeds, np.float64)
    _, idx = cKDTree(seeds).query(seeds, k=pool+1)
    assert np.all(idx[:, 0] == np.arange(len(seeds))), 'a seed is not its own nearest point (duplicate seeds?)'
    candidates = idx[:, 1:]
    order = fps_order(seeds, candidates, k)
    assert np.all(order != np.arange(len(seeds))[:, None]) and all(len(np.unique(r)) == k for r in order[:: max(1, len(order)//5000)])
    return order, candidates


def build_neighbor_tables(spec, sha):
    """Compute (or verify) the per-flow FPS tables next to the training output; they depend only on the frozen seeds."""
    root = Path(spec['output'])/'neighbors'; root.mkdir(parents=True, exist_ok=True); records = {}
    for flow in spec['flows']:
        folder = Path(spec['source_output'])/'physical'/flow['name']; path = root/NEIGHBOR_FILE.format(flow=flow['name'])
        seeds_sha = sha(folder/'seeds.npy')
        if path.exists():
            with np.load(path) as z: assert str(z['seeds_sha256']) == seeds_sha and int(z['pool']) == spec['neighbors']['pool'] and int(z['k']) == spec['neighbors']['max_k']
        else:
            seeds = np.load(folder/'seeds.npy'); order, candidates = neighbor_table(seeds, spec['neighbors']['pool'], spec['neighbors']['max_k'])
            np.savez_compressed(path, order=order, candidates=candidates, seeds_sha256=seeds_sha, pool=spec['neighbors']['pool'], k=spec['neighbors']['max_k'])
        records[flow['name']] = dict(file=path.name, sha256=sha(path), seeds_sha256=seeds_sha)
    return records


def load_neighbors(spec, flow):
    with np.load(Path(spec.get('_root_output', spec['output']))/'neighbors'/NEIGHBOR_FILE.format(flow=flow)) as z: return z['order'].astype(np.int64)


# ---------------------------------------------------------------- splits and rows
def split_masks(meta, flow_index, spec):
    """test = fold ``test_fold``; validation = a seeded fraction of the training samples; train = the rest."""
    test = meta['fold'] == spec['folds']['test_fold']; training = ~test
    rng = np.random.default_rng([spec['validation']['seed'], flow_index]); ids = np.flatnonzero(training); rng.shuffle(ids)
    validation = np.zeros(len(test), bool); validation[ids[:int(round(len(ids)*spec['validation']['fraction']))]] = True
    return dict(train=training & ~validation, validation=validation, test=test)


def normalize_bundle(lines, seeds):
    """Frozen whole-bundle normalization (centroid, maximum radius) applied to lines and their seed points."""
    counts = torch.full((len(lines),), lines.shape[1], device=lines.device, dtype=torch.long)
    geometry, _ = normalize(lines, counts)
    work = lines.double(); center = work.mean((1, 2), keepdim=True); radius = (work-center).square().sum(-1).amax((1, 2)).sqrt()
    torch.testing.assert_close(geometry, ((work-center)/radius[:, None, None, None]).float(), atol=1e-6, rtol=1e-6)
    return geometry, ((seeds.double()-center[:, :, 0])/radius[:, None, None]).float(), counts


class Dataset:
    """Rows = (sample, half length) of one role; clusters come from the frozen FPS tables."""
    def __init__(self, spec, role, candidate, device='cuda', limit=None):
        self.device = device; self.role = role; self.parts = []
        if role == 'test':
            lock = Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"/'selection.lock.json'
            assert json.loads(lock.read_text())['test_loaded'] is False
        self.evidence = None if limit else Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"
        labels, owners, flows, samples, lengths, extra = [], [], [], [], [], {k: [] for k in ('assignment_kind', 'fold')}
        K = len(spec['flows'][0]['half_lengths'])
        for fi, flow in enumerate(spec['flows']):
            folder = Path(spec['source_output'])/'physical'/flow['name']; seeds, curves, meta = data.load(folder)
            assert curves.shape[1] == K; mask = split_masks(meta, fi, spec)[role]; ids = np.flatnonzero(mask)
            if limit: ids = np.concatenate([ids[meta['label'][ids] == cls][:limit//4] for cls in (0, 1)])
            sample_ids = np.repeat(ids, K); length_ids = np.tile(np.arange(K), len(ids))
            self.parts.append(dict(curves=torch.tensor(curves, device=device), seeds=torch.tensor(seeds, device=device),
                                   neighbors=torch.tensor(load_neighbors(spec, flow['name']), device=device), sample_ids=sample_ids, length_ids=length_ids))
            labels.append(meta['label'][sample_ids]); owners.append(meta['instance'][sample_ids]); flows.append(np.full(len(sample_ids), fi))
            samples.append(sample_ids); lengths.append(length_ids)
            for k in extra: extra[k].append(meta[k][sample_ids])
        self.labels = np.concatenate(labels).astype(np.int64); self.owners = np.concatenate(owners); self.flows = np.concatenate(flows)
        self.rows = np.concatenate(samples); self.lengths = np.concatenate(lengths); self.extra = {k: np.concatenate(v) for k, v in extra.items()}
        assert limit or len(self.labels) == spec['expected_counts'][role], (role, len(self.labels))
        self.targets = torch.tensor(self.labels, device=device); self.neighbors = torch.empty((len(self.labels), 0), device=device, dtype=torch.long); self.clean = None

    def encode(self, candidate, batch=128):
        k = candidate['neighbors']; self.clean = torch.empty((len(self.labels), 1, 142), device=self.device); offset = 0; chosen = []
        for part in self.parts:
            n = len(part['sample_ids'])
            for first in range(0, n, batch):
                s = torch.tensor(part['sample_ids'][first:first+batch], device=self.device); l = torch.tensor(part['length_ids'][first:first+batch], device=self.device)
                ids = torch.cat((s[:, None], part['neighbors'][s, :k]), 1)
                lines = part['curves'][ids, l[:, None]]; points = part['seeds'][ids]
                geometry, seeds, counts = normalize_bundle(lines, points)
                tokens, used = method.encode(geometry, seeds, counts, torch.zeros(len(ids), device=self.device, dtype=torch.long), candidate)
                assert torch.equal(used.sort(1).values, torch.arange(1, k+1, device=self.device)[None].expand(len(ids), k)), 'the frozen encoder must use exactly the provided neighbours'
                if candidate['base_method'] == 'p35_h0':
                    values = tokens.cpu().numpy(); values[..., :141] = transform_fmt(values[..., :141], True); tokens = torch.from_numpy(values).to(self.device)
                self.clean[offset:offset+len(ids)] = tokens; offset += len(ids); chosen.append(ids[:, 1:].cpu().numpy())
        assert offset == len(self.labels) and torch.isfinite(self.clean).all(); self.selected_neighbors = np.concatenate(chosen)
        if self.evidence:
            self.evidence.mkdir(parents=True, exist_ok=True)
            (self.evidence/f'{self.role}_encoding.json').write_text(json.dumps(dict(candidate=candidate, shape=list(self.clean.shape), samples=len(self.labels),
                cluster='center sample + first k of the FPS order over the 64 nearest samples of the same flow; same half length for every line',
                neighbor_ids_sha256=hashlib.sha256(self.selected_neighbors.astype('<i8').tobytes()).hexdigest()), indent=2)+'\n', encoding='utf8')
