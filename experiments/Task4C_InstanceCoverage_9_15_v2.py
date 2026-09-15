"""Task4-c v9.15 v2: dense length variation, mandatory head bundles and exact analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from FMT_Utils.Task4C_InstanceCoverage_9_15_v2 import (
    write, load_scene, scene_report, generate_instance, save_instance, sparse_voxels,
    dense_sparse_batch, head_mask, local_label,
)
from FMT_Utils.FMT_V8_Search_2_1 import task4_model
from FMT_Utils.FMT_P35_NormFrequency_3_1 import candidates, encode as encode_p35, apply_normalizer
from experiments.Task4C_HairpinBinary_2_1 import append_locked

CONFIG = 'config/mainExp_Task4C_InstanceCoverage_9.15_v2.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(config):
    paths = [__file__, 'FMT_Utils/Task4C_InstanceCoverage_9_15_v2.py', 'FMT_Utils/Task4C_InstanceCoverage_9_15.py',
        'FMT_Utils/FMT_P35_NormFrequency_3_1.py', 'FMT_Utils/FMT_V8_Search_2_1.py',
        'FMT_Utils/Task4C_Encoders_4_15.py', 'FMT_Utils/DFT_FMT_3D.py',
        'FMT_Utils/Task4C_PaperBundles_3_1.py', 'FMT_Utils/Task4C_Multiscale_4_1.py',
        'FMT_Utils/Task4C_HairpinBinary_2_1.py', 'experiments/Task4C_PhysicalLength_4_14.py',
        'experiments/Task4C_ConvResolution_4_22.py',
        'experiments/Visualize_Task4C_Bundles_3D.py', 'experiments/templates/task4c_bundles.html',
        'ibex_bash/task4c_instance_9p15_v2.sh']
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        config_sha256=sha(config), sources={str(p): sha(p) for p in paths},
        hostname=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
        array_index=os.environ.get('SLURM_ARRAY_TASK_ID'))


def load_spec(config, allow_pending_length=False):
    spec = json.loads(Path(config).read_text(encoding='utf-8'))
    assert spec['labels']['confirmed_by_user']
    assert spec['training']['selection'] == 'fixed_last_epoch_no_test_selection'
    assert spec['encoding']['batch_size'] == 32 and spec['sampling']['mandatory_head_bundles'] == 3
    assert spec['encoding']['fmt_definition'] == next(c for c in candidates() if c['id'] == 'n0_k06')
    from FMT_Utils.Task4C_InstanceCoverage_9_15_v2 import validate_spec
    if not (allow_pending_length and spec['integration']['length_definition']=='pending_user_confirmation'):
        validate_spec(spec)
    return spec


def deterministic(device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        if not torch.cuda.is_available() or 'V100' not in torch.cuda.get_device_name():
            raise ValueError('The frozen p35 nearest-neighbor encoding requires a V100')


class DeterministicPool2(nn.Module):
    """Same 2x2x2 adaptive averaging bins; no CUDA atomic backward reduction."""
    def forward(self, x):
        import itertools
        shape=x.shape[-3:]
        bins=[]
        for position in itertools.product(range(2),repeat=3):
            slices=tuple(slice(i*n//2,((i+1)*n+1)//2) for i,n in zip(position,shape))
            bins.append(x[(slice(None),slice(None),*slices)].mean(dim=(-3,-2,-1)))
        return torch.stack(bins,-1).reshape(*x.shape[:2],2,2,2)


class DeterministicMaxPool2(nn.Module):
    """Nonoverlapping 2x2x2 max bins, preserving the first-index tie rule."""
    def forward(self,x):
        d,h,w=(n//2 for n in x.shape[-3:])
        cells=x[...,:2*d,:2*h,:2*w].reshape(*x.shape[:2],d,2,h,2,w,2)
        return cells.permute(0,1,2,4,6,3,5,7).flatten(5).max(-1).values


class Conv915(nn.Module):
    """Same spatial blocks; 70-wide head keeps 90-95% of current p35 parameters."""
    def __init__(self, dropout=.15):
        super().__init__()
        layers = []
        for i, (cin, cout) in enumerate(((4, 12), (12, 24), (24, 48))):
            layers.extend((nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(),
                nn.Dropout3d(dropout/2), DeterministicMaxPool2() if i < 2 else DeterministicPool2()))
        self.network = nn.Sequential(*layers, nn.Flatten(), nn.Linear(384, 70), nn.LayerNorm(70),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(70, 64), nn.Linear(64, 2))

    def forward(self, x):
        return self.network(x)


def metrics(y, p):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, bool); pred = np.asarray(p) >= .5
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    return dict(f1=2*tp/max(1, 2*tp+fp+fn), precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn),
        accuracy=(tp+tn)/len(y), average_precision=float(average_precision_score(y, p)),
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn, samples=len(y))


def prepare(spec, config, index, pilot=False, input_root=None):
    if not pilot:
        capacity_report = json.loads((Path(spec['output'])/'capacity.json').read_text())
        assert capacity_report['complete'] and capacity_report['identity']['config_sha256'] == sha(config)
    shards = spec['sampling']['shards_per_flow']; flow_index, shard = divmod(index, shards)
    scene = load_scene(spec, flow_index, input_root)
    name = scene['flow']['name']; root = Path(spec['output']); report = scene_report(scene)
    report['identity'] = identity(config)
    write(root/('pilot_catalog' if pilot else 'catalog')/f'{name}_{shard}.json', report)
    if report['missing_candidate_pools']:
        raise ValueError(f'Missing instance pools: {report["missing_candidate_pools"]}')
    results = []
    for j, item in enumerate(scene['instances']):
        if j % shards != shard:
            continue
        rows, result = generate_instance(scene, item, spec, 3 if pilot else None)
        result.update(flow=name, gt_bounds=item['gt_bounds'].tolist(), box_bounds=item['box_bounds'].tolist())
        directory = root/('pilot' if pilot else 'physical')/name/str(item['instance'])
        save_instance(directory, rows, result)
        result['files'] = {f.name: sha(f) for f in directory.iterdir() if f.suffix in ('.npy', '.npz')}
        write(directory/'coverage.json', result)
        results.append(result)
        print(json.dumps(dict(flow=name, instance=item['instance'], role=item['role'], bundles=len(rows),
            mandatory_heads=result['mandatory_head_bundles'], seconds=result['seconds'])), flush=True)
    write(root/('pilot_completed' if pilot else 'prepared')/f'{index}.json',
          dict(complete=True, identity=identity(config), instances=results))


def manifests(spec):
    root = Path(spec['output']); result = []
    for index in range(2*spec['sampling']['shards_per_flow']):
        report = json.loads((root/'prepared'/f'{index}.json').read_text())
        assert report['complete']
        for instance in report['instances']:
            assert instance['bundles'] == 2*spec['sampling']['bundles_per_class_per_instance']
            assert instance['bundles'] >= spec['sampling']['minimum_bundles_per_instance']
            assert instance['mandatory_head_bundles'] == 3
            result.append(instance)
    assert len(result) == 132
    assert len({(r['flow'], r['instance']) for r in result}) == 132
    return sorted(result, key=lambda r: (r['flow'], r['instance']))


def encode(spec, config, index):
    deterministic('cuda'); root = Path(spec['output']); name = spec['flows'][index]['name']
    audit = json.loads((root/'data_audit.json').read_text())
    assert audit['complete'] and audit['identity']['config_sha256'] == sha(config)
    rows = [r for r in manifests(spec) if r['flow'] == name]
    definition = spec['encoding']['fmt_definition']
    batch = spec['encoding']['batch_size']; results = []
    for row in rows:
        if shutil.disk_usage(root).free < 10*2**30:
            raise RuntimeError('Encoding stopped before the next instance: less than 10 GiB reserve')
        src = root/'physical'/name/str(row['instance'])
        for filename, digest in row['files'].items():
            assert sha(src/filename) == digest
        g = np.load(src/'geometry.npy', mmap_mode='r'); seeds = np.load(src/'seeds.npy', mmap_mode='r')
        with np.load(src/'metadata.npz') as z: counts = z['counts']
        dest = root/'encoded'/name/str(row['instance']); dest.mkdir(parents=True, exist_ok=False)
        tokens = np.lib.format.open_memmap(dest/'p35.npy', mode='w+', dtype=np.float32, shape=(len(g), 27, 142))
        sparse = {r: dict(offsets=[np.array([0], np.int64)], indices=[], values=[]) for r in spec['encoding']['voxel_resolutions']}
        for start in range(0, len(g), batch):
            sl = slice(start, start+batch)
            gg = torch.as_tensor(np.array(g[sl]), device='cuda')
            ss = torch.as_tensor(np.array(seeds[sl]), device='cuda')
            cc = torch.as_tensor(counts[sl], device='cuda')
            selected = encode_p35(gg, definition, seeds=ss, counts=cc)
            mask = (torch.arange(27, device='cuda')[None] < cc[:, None]).to(gg.dtype)
            tokens[sl] = torch.cat((selected, mask[..., None]), -1).cpu().numpy()
            for resolution, part in sparse.items():
                offset, ids, values = sparse_voxels(gg, cc, resolution)
                part['offsets'].append(offset[1:]+part['offsets'][-1][-1])
                part['indices'].append(ids); part['values'].append(values)
        tokens.flush(); del tokens
        for resolution, part in sparse.items():
            folder = dest/f'r{resolution}'; folder.mkdir()
            for key, arrays in part.items():
                np.save(folder/f'{key}.npy', np.concatenate(arrays))
        record = dict(flow=name, instance=row['instance'], role=row['role'], samples=len(g),
            files={str(f.relative_to(dest)): sha(f) for f in dest.rglob('*.npy')})
        write(dest/'manifest.json', record); results.append(record)
        print('encoded', name, row['instance'], len(g), flush=True)
    write(root/'encoding'/f'{index}.json', dict(complete=True, identity=identity(config), instances=results))


def capacity(spec, config):
    """Measure sparse storage on every pilot instance before the full allocation."""
    deterministic('cuda'); root = Path(spec['output']); sampled = total_sparse = 0
    for index in range(2*spec['sampling']['shards_per_flow']):
        report = json.loads((root/'pilot_completed'/f'{index}.json').read_text())
        assert report['complete']
        for item in report['instances']:
            source = root/'pilot'/item['flow']/str(item['instance'])
            g = torch.as_tensor(np.load(source/'geometry.npy'),device='cuda')
            with np.load(source/'metadata.npz') as m: counts = torch.as_tensor(m['counts'],device='cuda')
            sampled += len(g)
            for resolution in spec['encoding']['voxel_resolutions']:
                arrays = sparse_voxels(g, counts, resolution)
                total_sparse += sum(a.nbytes for a in arrays)
    assert sampled == 132*6
    total = spec['expected_counts']['total']
    estimated_sparse = total_sparse/sampled*total
    # Includes physical geometry, seeds, uncompressed metadata, tokens, viewer copy and margin.
    other_bytes = total*(27*32*3*4*2 + 27*3*4 + 27*142*4 + 12000)
    required = 2*estimated_sparse + other_bytes + 10*2**30
    available = shutil.disk_usage(root).free
    report = dict(complete=available>=required, identity=identity(config), pilot_samples=sampled,
        estimated_sparse_bytes=estimated_sparse, required_bytes=required, available_bytes=available,
        sparse_safety_factor=2, reserve_gib=10)
    write(root/'capacity.json',report)
    if not report['complete']:
        raise RuntimeError(f'Pilot storage estimate requires {required/2**30:.1f} GiB; available {available/2**30:.1f} GiB')
    print(json.dumps(report),flush=True)


def audit(spec, config):
    """Re-read every saved sample before any representation or model is fitted."""
    root = Path(spec['output']); coverage = manifests(spec)
    totals = dict(train=0, test=0); seen = set(); reports = []
    assert spec['integration']['length_definition'] == 'per_direction'
    for item in coverage:
        src = root/'physical'/item['flow']/str(item['instance'])
        for filename, digest in item['files'].items():
            assert sha(src/filename) == digest
        g = np.load(src/'geometry.npy', mmap_mode='r')
        with np.load(src/'metadata.npz') as z: m = {k:z[k] for k in z.files}
        n = m['counts']; valid = np.arange(27)[None] < n[:,None]
        assert len(g) == 1000 and np.all((n >= 10) & (n <= 27))
        assert np.isfinite(g).all() and np.all(g[~valid] == 0)
        center = (g.astype(np.float64)*valid[:,:,None,None]).sum((1,2))/(n[:,None]*32)
        assert np.allclose(center, 0, atol=2e-6)
        assert np.allclose(np.linalg.norm(g, axis=-1).max((1,2)), 1, atol=2e-6)
        lower, upper = spec['integration']['length_ranges'][item['flow']]
        eps = 2e-6*upper
        half = m['half_arc_lengths'][valid]
        assert np.all((half >= lower-eps) & (half <= upper+eps))
        steps = m['half_step_counts']
        assert np.all(steps[valid] <= np.broadcast_to(m['maxiteration'][:,None,None], steps.shape)[valid])
        assert np.allclose(m['requested_half_length'], m['ds']*m['maxiteration'])
        arcs = np.linalg.norm(np.diff(g.astype(np.float64),axis=2),axis=-1).sum(2)*m['radius'][:,None]
        assert np.allclose(arcs[valid], m['resampled_total_arc_lengths'][valid], atol=eps)
        assert np.all((arcs[valid] >= 2*lower-eps) & (arcs[valid] <= 2*upper+eps))
        assert np.allclose(m['raw_total_arc_lengths'][valid], half.sum(1))
        assert np.sum(m['labels']==1) == 500 and np.sum(m['labels']==0) == 500
        heads = m['mandatory_head']; assert heads.sum() == 3
        assert np.all(m['labels'][heads] == 1) and m['center_is_head'][heads].all()
        assert np.all(m['center_abs_cosine'][heads]**2 < .5)
        assert len(np.unique(m['center'][heads],axis=0)) == 3
        for i in range(len(g)):
            label, instance = local_label(m['seed_gt_ids'][i,:n[i]])
            assert label == m['labels'][i] and instance == m['instance'][i]
            assert not label or instance == item['instance']
            digest = hashlib.sha256(g[i].tobytes()).hexdigest()
            assert digest not in seen, 'Duplicated geometry in the dataset'
            seen.add(digest)
        totals[item['role']] += len(g)
        reports.append(dict(flow=item['flow'],instance=item['instance'],role=item['role'],samples=len(g),
            half_length_minmax=[float(half.min()),float(half.max())],
            integration_index_counts=item['integration_index_counts']))
    assert totals == {k:spec['expected_counts'][k] for k in ('train','test')}
    write(root/'data_audit.json',dict(complete=True,identity=identity(config),counts=totals,
        distinct_geometry=len(seen),mandatory_heads=396,instances=reports))
    print(json.dumps(dict(data_audit='PASS',counts=totals,mandatory_heads=396)),flush=True)


def load_parts(spec, role, method):
    root = Path(spec['output']); parts = []; indices = []; labels = []; flow_ids = []; owners = []
    for item in manifests(spec):
        if item['role'] != role:
            continue
        source = root/'physical'/item['flow']/str(item['instance'])
        cache = root/'encoded'/item['flow']/str(item['instance'])
        report = json.loads((cache/'manifest.json').read_text())
        assert report['role'] == role
        with np.load(source/'metadata.npz') as z: y = z['labels'].copy()
        if method == 'p35':
            assert sha(cache/'p35.npy') == report['files']['p35.npy']
            part = dict(tokens=np.load(cache/'p35.npy', mmap_mode='r'))
        else:
            folder = cache/f'r{int(method[4:])}'
            part = {}
            for key in ('offsets', 'indices', 'values'):
                file = folder/f'{key}.npy'; assert sha(file) == report['files'][str(file.relative_to(cache))]
                part[key] = np.load(file, mmap_mode='r')
        indices.extend((len(parts), i) for i in range(len(y))); labels.append(y)
        flow_ids.extend([0 if item['flow'] == 'channel' else 1]*len(y)); owners.extend([item['instance']]*len(y))
        parts.append(part)
    return parts, np.asarray(indices, np.int64), np.concatenate(labels), np.array(flow_ids), np.array(owners)


def normalization(parts):
    count = 0; mean = np.zeros(141, np.float64); m2 = mean.copy()
    for part in parts:
        x = np.asarray(part['tokens']); valid = x[..., -1] > .5
        values = x[..., :-1][valid]
        local_mean = values.mean(0, dtype=np.float64)
        local_m2 = ((values-local_mean)**2).sum(0, dtype=np.float64)
        delta = local_mean-mean; n = len(values)
        m2 += local_m2+delta*delta*count*n/(count+n)
        mean += delta*n/(count+n); count += n
    std = np.sqrt(m2/count); std[std < 1e-8] = 1
    return mean, std


def get_batch(parts, ids, method, norm, device='cuda'):
    if method != 'p35':
        return dense_sparse_batch(parts, ids, int(method[4:]), device)
    x = np.stack([parts[p]['tokens'][i] for p, i in ids])
    value = apply_normalizer(x[..., :-1], dict(kind='zscore', mean=norm[0], std=norm[1]), x[..., -1:])
    return torch.as_tensor(np.concatenate((value, x[..., -1:]), -1), device=device)


def train(spec, config, index):
    deterministic('cuda'); seeds = spec['training']['seeds']; method = spec['methods'][index//len(seeds)]
    seed = seeds[index % len(seeds)]; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    root = Path(spec['output']); directory = root/'runs'/method/f'seed{seed}'; directory.mkdir(parents=True, exist_ok=False)
    parts, ids, y, flows, owners = load_parts(spec, 'train', method)
    assert len(y) == spec['expected_counts']['train']
    norm = normalization(parts) if method == 'p35' else None
    model = (task4_model(141, 'h0') if method == 'p35' else Conv915(spec['training']['dropout'])).cuda()
    options = spec['training']; batch = options['batch_size']; epochs = options['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=options['minimum_learning_rate'])
    started = time.perf_counter(); history = []
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(y)); model.train(); loss_sum = 0.; lr = optimizer.param_groups[0]['lr']
        for start in range(0, len(y), batch):
            rows = order[start:start+batch]
            x = get_batch(parts, ids[rows], method, norm); yy = torch.as_tensor(y[rows], device='cuda')
            optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model(x), yy)
            assert torch.isfinite(loss); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
            loss_sum += float(loss.detach())*len(rows)
        scheduler.step()
        row = dict(epoch=epoch, loss=loss_sum/len(y), learning_rate=lr, samples=len(y),
            permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(), seconds=time.perf_counter()-started)
        history.append(row); append_locked(directory/'history.jsonl', json.dumps(row)+'\n')
        if epoch == 1 or epoch % 10 == 0:
            print(method, seed, row, flush=True)
    torch.cuda.synchronize(); training_seconds = time.perf_counter()-started
    outputs = {}
    for role in ('train', 'test'):
        if role == 'test':
            # First test load happens after the prespecified final optimizer step.
            parts, ids, y, flows, owners = load_parts(spec, 'test', method)
            assert len(y) == spec['expected_counts']['test']
        model.eval(); probability = []
        with torch.no_grad():
            for start in range(0, len(y), batch):
                probability.append(model(get_batch(parts, ids[start:start+batch], method, norm)).softmax(-1)[:, 1].cpu().numpy())
        probability = np.concatenate(probability)
        np.savez_compressed(directory/f'{role}_predictions.npz', labels=y, probability=probability,
                            flow_index=flows, owner_instance=owners, row_in_instance=ids[:, 1], threshold=.5)
        outputs[role] = dict(combined=metrics(y, probability),
            per_flow={name: metrics(y[flows == i], probability[flows == i]) for i, name in enumerate(('channel', 'tbl'))})
    result = dict(complete=True, version=spec['version'], identity=identity(config), method=method, seed=seed,
        parameters=sum(p.numel() for p in model.parameters()), training_seconds=training_seconds, epochs=epochs,
        selection='fixed_last_epoch_no_test_selection', threshold=.5, metrics=outputs,
        normalization=None if norm is None else dict(mean=norm[0].tolist(), std=norm[1].tolist()),
        predictions={role: sha(directory/f'{role}_predictions.npz') for role in ('train', 'test')}, history=history,
        gpu=torch.cuda.get_device_name(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(directory/'result.json', result)
    print(json.dumps({k: result[k] for k in ('method', 'seed', 'parameters', 'training_seconds', 'metrics')}), flush=True)


def preflight(spec, config, device):
    deterministic(device); torch.manual_seed(91500)
    pool_checks={}
    for size in (8,9,12):
        values=torch.randn(2,3,size,size,size,dtype=torch.float64)
        reference=values.clone().requires_grad_(True)
        expected=F.adaptive_avg_pool3d(reference,2)
        weight=torch.randn_like(expected);(expected*weight).sum().backward()
        actual=values.to(device).requires_grad_(True)
        output=DeterministicPool2()(actual);(output*weight.to(device)).sum().backward()
        assert torch.allclose(output.cpu(),expected.detach(),atol=1e-12,rtol=1e-12)
        assert torch.allclose(actual.grad.cpu(),reference.grad,atol=1e-12,rtol=1e-12)
        pool_checks[str(size)]=dict(forward_max_error=float((output.cpu()-expected.detach()).abs().max().detach()),
            backward_max_error=float((actual.grad.cpu()-reference.grad).abs().max()))
        # Rounded values deliberately exercise equal maxima and odd-size cropping.
        values=torch.randn(2,3,size,size,size,dtype=torch.float64).round()
        reference=values.clone().requires_grad_(True);expected=F.max_pool3d(reference,2)
        weight=torch.randn_like(expected);(expected*weight).sum().backward()
        actual=values.to(device).requires_grad_(True);output=DeterministicMaxPool2()(actual)
        (output*weight.to(device)).sum().backward()
        assert torch.equal(output.cpu(),expected.detach()) and torch.equal(actual.grad.cpu(),reference.grad)
        pool_checks[str(size)]['max_forward_and_backward_exact_including_ties']=True
    from experiments.Task4C_ConvResolution_4_22 import reference_voxel
    from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_voxels
    u = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 0., 0.]], dtype=torch.float64).numpy()
    w = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 1., 0.]], dtype=torch.float64).numpy()
    assert head_mask(u, w)[0].tolist() == [True, False, False]
    assert local_label([0, 0, -1]) == (1, 0) and local_label([-1, -1]) == (0, -1)
    assert local_label([0, 1, -1])[0] is None and local_label([0, 1])[0] is None
    t = torch.linspace(-1, 1, 32, device=device); geometry = torch.zeros((2, 27, 32, 3), device=device)
    for i in range(27):
        geometry[:, i, :, 0] = .7*t
        geometry[:, i, :, 1] = .17*torch.sin(3*t+i*.03)+i*.008
        geometry[:, i, :, 2] = .18*torch.cos(2*t+i*.04)-i*.006
    counts = torch.tensor([27, 19], device=device); geometry[1, 19:] = 0
    seeds = geometry[:, :, 16].clone()
    tokens = encode_p35(geometry, spec['encoding']['fmt_definition'], seeds=seeds, counts=counts)
    mask = (torch.arange(27, device=device)[None] < counts[:, None]).to(geometry.dtype)
    packed = torch.cat((tokens, mask[..., None]), -1).cpu().numpy()
    parts = [dict(tokens=packed)]
    norm = normalization(parts)
    from FMT_Utils.FMT_P35_NormFrequency_3_1 import fit_normalizer
    reference = fit_normalizer(packed[..., :-1], packed[..., -1:], 'zscore')
    assert np.allclose(norm[0], reference['mean'], atol=1e-12)
    assert np.allclose(norm[1], reference['std'], atol=1e-12)
    fmt_input = get_batch(parts, [(0,0),(0,1)], 'p35', norm, device)
    expected = apply_normalizer(packed[..., :-1], reference, packed[..., -1:])
    assert np.allclose(fmt_input[..., :-1].cpu().numpy(), expected, atol=1e-6)
    fmt_model = task4_model(141, 'h0').to(device)
    fmt_optimizer = torch.optim.AdamW(fmt_model.parameters(), lr=.001)
    for _ in range(2):
        fmt_optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(fmt_model(fmt_input), torch.tensor([0, 1], device=device))
        assert torch.isfinite(loss); loss.backward(); fmt_optimizer.step()
    fmt_model.eval(); changed = fmt_input.clone(); changed[1, 19:, :-1] = 1234.
    with torch.no_grad():
        assert torch.equal(fmt_model(changed), fmt_model(fmt_input))
    del fmt_model, fmt_optimizer
    parameters = dict(p35=sum(p.numel() for p in task4_model(141, 'h0').parameters()),
                      conv=sum(p.numel() for p in Conv915().parameters()))
    assert parameters['p35'] == 76738 and .90 <= parameters['conv']/parameters['p35'] <= .95
    report = dict(parameters=parameters, resolutions={}, deterministic_pool_checks=pool_checks,
        fmt_definition=spec['encoding']['fmt_definition'], train_only_zscore_reference=True,
        device=device, identity=identity(config))
    for r in spec['encoding']['voxel_resolutions']:
        offsets, indices, values = sparse_voxels(geometry, counts, r)
        recovered = dense_sparse_batch([dict(offsets=offsets, indices=indices, values=values)], [(0, 0), (0, 1)], r, device)
        expected = bundle_voxels(geometry, counts, r).half().float()
        assert torch.equal(expected, recovered)
        independent = reference_voxel(geometry[0].cpu().numpy(), 27, r)
        assert np.allclose(expected[0].cpu().numpy(), independent, atol=.0005, rtol=.002)
        model = Conv915().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        xb = recovered[:1].expand(spec['training']['batch_size'] if device == 'cuda' else 2, -1, -1, -1, -1).contiguous()
        start = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model(xb), torch.zeros(len(xb), device=device, dtype=torch.long))
            loss.backward(); optimizer.step()
        if device == 'cuda': torch.cuda.synchronize()
        report['resolutions'][str(r)] = dict(roundtrip_exact=True, loss=float(loss.detach()),
            two_steps_seconds=time.perf_counter()-start, max_voxel_error=float(np.max(np.abs(expected[0].cpu().numpy()-independent))))
        del model, optimizer, xb, recovered, expected
    report['complete'] = True
    write(Path(spec['output'])/f'preflight_{device}.json', report)
    print(json.dumps(report), flush=True)


def export_viewer_package(spec, config, coverage):
    """Export every real row, including all head flags and all four predictions."""
    root=Path(spec['output']);output=root/'viewer_package';output.mkdir(exist_ok=False)
    seed=spec['training']['seeds'][0];models=[];predictions={}
    for method in spec['methods']:
        folder=root/'runs'/method/f'seed{seed}'
        result=json.loads((folder/'result.json').read_text())
        models.append(dict(id=method,label='p35 FMT (141D)' if method=='p35' else f'Conv3D {method[4:]}³',
            seed=seed,parameters=result['parameters'],metrics=result['metrics'],result_sha256=sha(folder/'result.json'),
            version=spec['version']))
        for role in ('train','test'):
            with np.load(folder/f'{role}_predictions.npz') as z:
                predictions[method,role]={(int(f),int(o),int(i)):float(p) for f,o,i,p in zip(
                    z['flow_index'],z['owner_instance'],z['row_in_instance'],z['probability'])}
    entries={}
    keys=('counts','labels','center','head_component','instance','owner_instance','mandatory_head',
          'center_is_head','center_abs_cosine','scale_id','radius','neighbor_distance','ds','maxiteration',
          'requested_half_length','requested_total_length')
    for fi,flow in enumerate(spec['flows']):
        for role in ('train','test'):
            packs=[];population=0
            for item in coverage:
                if item['flow']!=flow['name'] or item['role']!=role:continue
                source=root/'physical'/flow['name']/str(item['instance'])
                with np.load(source/'metadata.npz') as z:m={k:z[k] for k in z.files}
                take=np.arange(len(m['labels']))
                geometry=np.load(source/'geometry.npy').astype(np.float64)
                world=geometry*m['radius'][:,None,None,None]+m['centroid'][:,None,None,:]
                world[np.arange(27)[None]>=m['counts'][:,None]]=0
                pack={k:m[k] for k in keys};pack.update(geometry=world.astype(np.float32),row_ids=take+population)
                for method in spec['methods']:
                    pack['p_'+method]=np.array([predictions[method,role][fi,item['instance'],int(i)] for i in take])
                population+=len(take);packs.append(pack)
            packed={key:np.concatenate([p[key] for p in packs]) for key in packs[0]}
            assert np.array_equal(packed['row_ids'],np.arange(population))
            filename=flow['name']+'_'+role+'.npz';np.savez_compressed(output/filename,**packed)
            entries[flow['name']+'/'+role]=dict(file=filename,sha256=sha(output/filename),selected=population,
                population=population,selection_uses_prediction_scores=False,
                mandatory_head_bundles=int(packed['mandatory_head'].sum()),
                all_head_center_bundles=int(((packed['labels']==1)&packed['center_is_head']).sum()),
                class_counts={method:dict(hairpin=int((packed['p_'+method]>=.5).sum()),
                    non_hairpin=int((packed['p_'+method]<.5).sum())) for method in spec['methods']})
    write(output/'manifest.json',dict(version=spec['version'],identity=identity(config),
        config=dict(display_bundles=200,splits=['train','test'],geometry_chunk_size=128,
            default_hairpin_count=1500,default_nonhairpin_count=100,default_view_mode='heads'),
        flows=spec['flows'],models=models,pending=[],splits=entries,
        evidence_note='v9.15 v2完整集合，未抽样或复制。Head模式按GT和速度–涡量夹角选择，保留漏检；分析色比较局部束标签与0.5阈值预测。F1使用完整集合。坐标保持物理尺度，z向上。'))


def merge(spec, config):
    coverage = manifests(spec); root = Path(spec['output']); results = []
    for method in spec['methods']:
        for seed in spec['training']['seeds']:
            folder = root/'runs'/method/f'seed{seed}'; result = json.loads((folder/'result.json').read_text())
            assert result['complete'] and result['epochs'] == spec['training']['epochs']
            owners_by_role = {}
            for role in ('train', 'test'):
                file = folder/f'{role}_predictions.npz'; assert sha(file) == result['predictions'][role]
                with np.load(file) as z:
                    computed = metrics(z['labels'], z['probability'])
                    assert computed == result['metrics'][role]['combined']
                    from sklearn.metrics import f1_score
                    assert abs(f1_score(z['labels'], z['probability'] >= .5)-computed['f1']) < 1e-12
                    owners_by_role[role] = set(zip(z['flow_index'].tolist(), z['owner_instance'].tolist()))
                    assert len(set(zip(z['flow_index'], z['owner_instance'], z['row_in_instance']))) == len(z['labels'])
                    for flow, owner in owners_by_role[role]:
                        source = root/'physical'/('channel' if flow == 0 else 'tbl')/str(owner)
                        with np.load(source/'metadata.npz') as m:
                            selected = (z['flow_index'] == flow) & (z['owner_instance'] == owner)
                            assert selected.sum() == len(m['labels'])
                            assert np.array_equal(z['labels'][selected], m['labels'][z['row_in_instance'][selected]])
            assert not owners_by_role['train'] & owners_by_role['test']
            results.append(result)
    summary = []
    for method in spec['methods']:
        rows = [r for r in results if r['method'] == method]
        f1 = [r['metrics']['test']['combined']['f1'] for r in rows]
        summary.append(dict(method=method, parameters=rows[0]['parameters'], test_f1_mean=float(np.mean(f1)),
            test_f1_std=float(np.std(f1, ddof=1)), mean_training_minutes=float(np.mean([r['training_seconds'] for r in rows])/60)))
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth')) and not any(root.rglob('*.ckpt'))
    write(root/'summary.json', dict(complete=True, identity=identity(config), coverage=coverage, methods=summary))
    export_viewer_package(spec, config, coverage)
    print(json.dumps(summary), flush=True)


def runtime(spec, config, phase, state, code=None):
    row = dict(version=spec['version'], phase=phase, state=state, exit_code=code, time_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity(config))
    if torch.cuda.is_available(): row['gpu'] = torch.cuda.get_device_name()
    root = Path(spec['output']); root.mkdir(parents=True, exist_ok=True)
    append_locked(root/'runtime.jsonl', json.dumps(row)+'\n')


def submit(spec, config):
    if shutil.disk_usage(Path.cwd()).free < 10*2**30:
        raise RuntimeError('Need at least 10 GiB reserve for preflight/pilot; full capacity is checked after pilot')
    root = Path(spec['output']).resolve(); root.mkdir(parents=True, exist_ok=False); (root/'logs').mkdir()
    write(root/'config.frozen.json', spec); previous = None
    phases = [('preflight', None, True, '00:20:00'), ('pilot', '0-15%8', False, '02:00:00'),
              ('capacity', None, True, '00:30:00'),
              ('prepare', '0-15%8', False, '16:00:00'), ('audit', None, False, '00:30:00'),
              ('encode', '0-1%2', True, '08:00:00'),
              ('train', '0-11%6', True, '2-00:00:00'), ('merge', None, False, '00:30:00')]
    for phase, array, gpu, limit in phases:
        cmd = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=32G',
            '--time='+limit, '--job-name=t4c915v2-'+phase, f'--output={root}/logs/{phase}_%A_%a.out',
            f'--error={root}/logs/{phase}_%A_%a.err']
        if gpu: cmd += ['--gres=gpu:1', '--constraint=v100']
        if array: cmd += ['--array='+array]
        if previous: cmd += ['--dependency=afterok:'+previous, '--kill-on-invalid-dep=yes']
        cmd += ['ibex_bash/task4c_instance_9p15_v2.sh', phase, config]
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        row = dict(job_id=job, version=spec['version'], phase=phase, dependency=previous,
            submitted_at_utc=datetime.now(timezone.utc).isoformat(), expected_device='V100' if gpu else 'CPU',
            command=cmd, identity=identity(config))
        append_locked(root/'submissions.jsonl', json.dumps(row)+'\n')
        append_locked('docs/ibex_run_registry.md', '\n- Task4-c-v9.15_v2 SUBMITTED '+json.dumps(row)+'\n')
        print(json.dumps(row), flush=True); previous = job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pilot', 'capacity', 'prepare', 'audit', 'encode', 'train', 'merge', 'preflight', 'runtime', 'submit'))
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--input-root'); parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument('--runtime-phase'); parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec = load_spec(args.config,args.phase in ('preflight','runtime'))
    if args.phase in ('pilot', 'prepare'):
        prepare(spec, args.config, args.index, args.phase == 'pilot', args.input_root)
    elif args.phase == 'runtime': runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    elif args.phase == 'preflight': preflight(spec, args.config, args.device)
    elif args.phase in ('encode', 'train'): globals()[args.phase](spec, args.config, args.index)
    else: globals()[args.phase](spec, args.config)


if __name__ == '__main__':
    main()
