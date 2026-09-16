"""Validation-only parallel search; locked three-seed final evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from FMT_Utils import Task4C_FPSAugmentSearch_1_1 as method
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine
from experiments import Task4C_BottomDensity_1_3 as reuse_source
from FMT_Utils.Task4C_NeighborSelection_1_1 import encode as old_encode
from FMT_Utils.FMT_V8_Search_2_1 import PROFILES

CONFIG = 'config/Ablation_Task4C_FPSAugmentSearch_1.1.json'
write, sha, append_locked, metrics = engine.write, engine.sha, engine.append_locked, engine.metrics


def load_spec(config):
    spec = json.loads(Path(config).read_text())
    assert sha(spec['base_config']) == spec['base_config_sha256']
    base = json.loads(Path(spec['base_config']).read_text())
    for key in ('flows', 'expected_counts'):
        spec[key] = base[key]
    assert spec['sampling_source'] == 'frozen_32_point_polylines'
    assert spec['jitter_fraction']==.15 and spec['curvature_uniform_mass']==.5
    assert spec['curvature_clip_over_line_mean']==4. and spec['frequencies']==6
    assert len(spec['old_top_five']) == 5
    assert spec['screen']['test_enabled'] is False and spec['refine']['test_enabled'] is False
    return spec


def candidates(spec):
    models = [dict(pool=r['pool'], profile=r['profile'], architecture='original') for r in spec['old_top_five']]
    if not any(r['pool']=='p35' and r['profile']=='h0' for r in models):
        models.append(dict(pool='p35', profile='h0', architecture='original'))
    models += [dict(pool='p35', profile='h0', architecture=a) for a in method.ARCHITECTURES[1:]]
    result = []
    for model, points, sampling, aug in itertools.product(models, spec['points'], spec['sampling'], spec['augmentations']):
        row = dict(model, points=points, sampling=sampling, augmentation=aug)
        row['id'] = f"c{len(result):03d}"
        row['parameters'] = method.parameter_count(row['pool'], row['architecture'], row['profile'])
        result.append(row)
    return result


def reference_candidate():
    return dict(id='fps_reference', pool='p35', profile='h0', architecture='original',
                points=32, sampling='uniform', augmentation='none', parameters=76738)


def identity(config):
    result = engine.identity(config)
    for p in ('FMT_Utils/Task4C_FPSAugmentSearch_1_1.py', 'experiments/Task4C_FPSAugmentSearch_1_1.py',
              'FMT_Utils/Task4C_NeighborSelection_1_1.py', 'experiments/Task4C_BottomDensity_1_3.py',
              'ibex_bash/task4c_fps_augment_search_1p1.sh'):
        result['sources'][p] = sha(p)
    return result


def reuse(spec, config):
    fn = reuse_source.reuse
    FunctionType(fn.__code__, dict(reuse_source.__dict__, identity=identity), argdefs=fn.__defaults__)(spec, config)


class Dataset:
    """Load frozen rows; test access explicitly requires the final selection lock."""
    def __init__(self, spec, role, candidate, device='cuda', limit=None):
        root = Path(spec['output'])
        if role == 'test':
            assert json.loads((root/'selection.lock.json').read_text())['complete']
        self.role = role
        self.parts = []
        labels, owners, flow_ids, row_ids, geometry, neighbors, counts = [], [], [], [], [], [], []
        for fi, flow in enumerate(spec['flows']):
            folder = root/'physical'/flow['name']/role
            g = np.load(folder/'geometry.npy', mmap_mode='r')
            s = np.load(folder/'seeds.npy', mmap_mode='r')
            with np.load(folder/'metadata.npz') as z:
                y, owner, c = z['labels'].copy(), z['instance'].copy(), z['counts'].astype(np.int64)
            ids = np.arange(len(y))
            if limit:
                ids = np.concatenate([np.flatnonzero(y==cls)[:limit//4] for cls in (0, 1)])
            labels.append(y[ids]); owners.append(owner[ids]); counts.append(c[ids])
            flow_ids.append(np.full(len(ids), fi)); row_ids.append(ids)
            chunks, chosen = [], []
            for first in range(0, len(ids), 512):
                rows = ids[first:first+512]
                gg = torch.tensor(np.array(g[rows]), device=device)
                ss = torch.tensor(np.array(s[rows]), device=device)
                cc = torch.tensor(c[rows], device=device)
                # FPS on the frozen seed coordinates is a distance-preserving
                # rigid-rotation operation; keep its original float32 tie choices.
                chosen.append(method.neighbor_indices(ss, cc, 'fps6'))
                chunks.append(method.resample(gg, candidate['points'], candidate['sampling']))
            geometry.extend(chunks); neighbors.extend(chosen)
        self.geometry = torch.cat(geometry)
        self.neighbors = torch.cat(neighbors)
        self.counts = torch.tensor(np.concatenate(counts), device=device)
        self.labels = np.concatenate(labels).astype(np.int64)
        self.targets = torch.tensor(self.labels, device=device)
        self.owners, self.flows, self.rows = np.concatenate(owners), np.concatenate(flow_ids), np.concatenate(row_ids)
        assert limit or len(self.labels)==spec['expected_counts'][role]
        self.mask = method.line_mask(self.counts, 27)
        self.clean = None

    def encode(self, candidate, batch=128):
        self.clean = torch.empty((len(self.labels), 27, method.POOLS[candidate['pool']]['feature_dimensions']+1),
                                 device=self.geometry.device)
        for first in range(0, len(self.labels), batch):
            sl = slice(first, first+batch)
            self.clean[sl] = method.fourier_tokens(self.geometry[sl], self.counts[sl], self.neighbors[sl], candidate['pool'])


def fit_normalizer(dataset, candidate, batch=128):
    """All clean training tokens plus one fixed train-only augmented view."""
    sums = torch.zeros(dataset.clean.shape[-1]-1, device=dataset.clean.device, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    n = 0
    generator = torch.Generator(device=dataset.clean.device).manual_seed(91901)
    views = (False, True) if candidate['augmentation']!='none' else (False,)
    for augmented in views:
        for first in range(0, len(dataset.labels), batch):
            sl = slice(first, first+batch)
            if augmented:
                g = method.augment(dataset.geometry[sl], dataset.counts[sl], candidate['augmentation'], generator)
                x = method.fourier_tokens(g, dataset.counts[sl], dataset.neighbors[sl], candidate['pool'])
            else:
                x = dataset.clean[sl]
            values = x[..., :-1][x[..., -1]>.5].double()
            sums += values.sum(0); squares += values.square().sum(0); n += len(values)
    mean = sums/n
    std = (squares/n-mean.square()).clamp_min(0).sqrt()
    std[std<1e-8] = 1.
    return mean, std, n


def standardize(x, norm):
    values = ((x[..., :-1].double()-norm[0])/norm[1]).float()*x[..., -1:]
    return torch.cat((values, x[..., -1:]), -1)


@torch.no_grad()
def predict(model, dataset, norm, batch):
    model.eval(); probabilities = []; losses = []
    for first in range(0, len(dataset.labels), batch):
        sl = slice(first, first+batch)
        logits = model(standardize(dataset.clean[sl], norm), dataset.neighbors[sl])
        probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy())
        losses.append(float(F.cross_entropy(logits, dataset.targets[sl], reduction='sum')))
    return np.concatenate(probabilities), sum(losses)/len(dataset.labels)


def save_predictions(folder, role, dataset, probability):
    path = folder/f'{role}_predictions.npz'
    np.savez_compressed(path, labels=dataset.labels, probability=probability, flow_index=dataset.flows,
                        instance=dataset.owners, row_in_split=dataset.rows, threshold=.5)
    return sha(path)


def train(spec, config, index, phase):
    engine.deterministic('cuda')
    root = Path(spec['output'])
    if phase == 'screen':
        candidate = candidates(spec)[index]; seed = spec['screen']['seed']
    elif phase == 'refine':
        locked = json.loads((root/'shortlist.json').read_text())
        seeds = spec['refine']['seeds']; candidate = locked['candidates'][index//len(seeds)]; seed = seeds[index%len(seeds)]
    else:
        locked = json.loads((root/'selection.lock.json').read_text()); assert locked['complete']
        seeds = spec['final']['seeds']
        candidate = locked['selected'] if index < len(seeds) else reference_candidate()
        seed = seeds[index%len(seeds)]
    options = dict(spec['training']); options.update(spec[phase])
    # Preserve any old h1/h2 optimizer profile when that profile was selected.
    options.update({k:v for k,v in PROFILES[candidate['profile']]['task4'].items() if k in options})
    folder = root/phase/candidate['id']/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed+200000)
    batch = options['batch_size']
    started_prepare = time.perf_counter()
    training = Dataset(spec, 'train', candidate); validation = Dataset(spec, 'validation', candidate)
    training.encode(candidate, batch); validation.encode(candidate, batch)
    norm = fit_normalizer(training, candidate, batch)
    if candidate['augmentation']!='none':
        # The clean training copy is not needed during online augmentation.
        training.clean = None
        torch.cuda.empty_cache()
    preparation_seconds = time.perf_counter()-started_prepare
    model = method.FourierClassifier(candidate['pool'], candidate['architecture'], candidate['profile']).cuda()
    assert sum(p.numel() for p in model.parameters()) == candidate['parameters']
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5,
                    patience=options['lr_patience'], threshold=1e-4, min_lr=1e-6)
    best = (-1., -1.); best_epoch = 0; state = None; history = []
    started = time.perf_counter()
    for epoch in range(1, options['epochs']+1):
        order = rng.permutation(len(training.labels)); model.train(); total = 0.
        for first in range(0, len(order), batch):
            ids = torch.tensor(order[first:first+batch], device='cuda')
            if candidate['augmentation']=='none':
                tokens = training.clean[ids]
            else:
                geometry = method.augment(training.geometry[ids], training.counts[ids], candidate['augmentation'], generator)
                tokens = method.fourier_tokens(geometry, training.counts[ids], training.neighbors[ids], candidate['pool'])
            xx = standardize(tokens, norm)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xx, training.neighbors[ids]), training.targets[ids])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())*len(ids)
        probability, val_loss = predict(model, validation, norm, batch)
        score = metrics(validation.labels, probability); rank = (score['f1'], score['average_precision'])
        row = dict(epoch=epoch, training_loss=total/len(order), validation_loss=val_loss,
                   validation_f1=score['f1'], validation_average_precision=score['average_precision'],
                   learning_rate=optimizer.param_groups[0]['lr'], samples=len(order),
                   permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(), seconds=time.perf_counter()-started)
        history.append(row); append_locked(folder/'history.jsonl', json.dumps(row)+'\n')
        if rank > best:
            best, best_epoch, state = rank, epoch, copy.deepcopy(model.state_dict())
        scheduler.step(val_loss)
        if epoch==1 or epoch%10==0:
            print(candidate['id'], seed, row, flush=True)
        if epoch-best_epoch >= options['patience']:
            break
    model.load_state_dict(state)
    probability, _ = predict(model, validation, norm, batch)
    assert abs(metrics(validation.labels, probability)['f1']-best[0])<1e-12
    torch.cuda.synchronize(); seconds = time.perf_counter()-started
    write(folder/'selection.lock.json', dict(identity=identity(config), candidate=candidate, selected_epoch=best_epoch,
                                            threshold=.5, test_loaded=False))
    validation_digest = save_predictions(folder, 'validation', validation, probability)
    result = dict(complete=True, identity=identity(config), version=spec['version'], phase=phase,
                  candidate=candidate, seed=seed, parameters=candidate['parameters'], selected_epoch=best_epoch,
                  epochs=len(history), history=history, training_seconds=seconds, preparation_seconds=preparation_seconds,
                  validation=metrics(validation.labels, probability), predictions={'validation':validation_digest},
                  normalization=dict(mean=norm[0].cpu().tolist(), std=norm[1].cpu().tolist(), fit_tokens=norm[2], train_only=True),
                  gpu=torch.cuda.get_device_name(), test_loaded=False)
    if phase == 'final':
        del training
        torch.cuda.empty_cache()
        test = Dataset(spec, 'test', candidate); test.encode(candidate, batch)
        p, _ = predict(model, test, norm, batch)
        result['test'] = dict(combined=metrics(test.labels, p), per_flow={flow['name']:metrics(test.labels[test.flows==fi], p[test.flows==fi]) for fi, flow in enumerate(spec['flows'])})
        result['predictions']['test'] = save_predictions(folder, 'test', test, p)
        result['test_loaded'] = True
    write(folder/'result.json', result)
    print(json.dumps({k:result[k] for k in ('candidate', 'seed', 'epochs', 'selected_epoch', 'training_seconds', 'validation', 'test_loaded')}), flush=True)


def check_result(spec, config, phase, candidate, seed, role='validation'):
    folder = Path(spec['output'])/phase/candidate['id']/f'seed{seed}'
    r = json.loads((folder/'result.json').read_text())
    assert r['complete'] and r['candidate']==candidate and r['seed']==seed
    assert r['identity']['git_commit']==identity(config)['git_commit'] and r['identity']['config_sha256']==sha(config)
    assert phase=='final' or not r['test_loaded']
    assert sha(folder/f'{role}_predictions.npz')==r['predictions'][role]
    with np.load(folder/f'{role}_predictions.npz') as z:
        assert len(z['labels'])==spec['expected_counts'][role]
        for fi, flow in enumerate(spec['flows']):
            with np.load(Path(spec['output'])/'physical'/flow['name']/role/'metadata.npz') as original:
                sel = z['flow_index']==fi; rows = z['row_in_split'][sel]
                assert np.array_equal(np.sort(rows), np.arange(len(original['labels'])))
                assert np.array_equal(z['labels'][sel], original['labels'][rows])
                assert np.array_equal(z['instance'][sel], original['instance'][rows])
        metric = metrics(z['labels'], z['probability'])
    expected = r['validation'] if role=='validation' else r['test']['combined']
    for key, value in metric.items():
        assert abs(value-expected[key])<1e-12, key
    best = max(r['history'], key=lambda h:(h['validation_f1'], h['validation_average_precision']))
    assert best['epoch']==r['selected_epoch']
    return r


def shortlist(spec, config):
    all_candidates = candidates(spec)
    results = [check_result(spec, config, 'screen', c, spec['screen']['seed']) for c in all_candidates]
    ordered = sorted(results, key=lambda r:(-r['validation']['f1'], -r['validation']['average_precision'], r['candidate']['id']))
    rows = [dict(candidate=r['candidate'], f1=r['validation']['f1'], average_precision=r['validation']['average_precision']) for r in ordered]
    write(Path(spec['output'])/'shortlist.json', dict(complete=True, identity=identity(config), rankings=rows,
          candidates=[r['candidate'] for r in ordered[:spec['refine']['candidates']]], test_read=False))


def select(spec, config):
    root = Path(spec['output']); locked = json.loads((root/'shortlist.json').read_text())
    rankings = []
    for c in locked['candidates']:
        results = [check_result(spec, config, 'refine', c, s) for s in spec['refine']['seeds']]
        rankings.append(dict(candidate=c, f1_mean=float(np.mean([r['validation']['f1'] for r in results])),
              f1_std=float(np.std([r['validation']['f1'] for r in results], ddof=1)),
              average_precision_mean=float(np.mean([r['validation']['average_precision'] for r in results]))))
    rankings.sort(key=lambda r:(-r['f1_mean'], -r['average_precision_mean'], r['candidate']['id']))
    write(root/'selection.lock.json', dict(complete=True, identity=identity(config), selected=rankings[0]['candidate'],
          rankings=rankings, threshold=.5, test_read=False, final_seeds=spec['final']['seeds']))


def merge(spec, config):
    root = Path(spec['output']); selected = json.loads((root/'selection.lock.json').read_text())['selected']
    rows = []
    for c in (selected, reference_candidate()):
        results = [check_result(spec, config, 'final', c, s, 'test') for s in spec['final']['seeds']]
        f1 = [r['test']['combined']['f1'] for r in results]
        rows.append(dict(candidate=c, f1_mean=float(np.mean(f1)), f1_std=float(np.std(f1, ddof=1)),
                         training_minutes=float(np.mean([r['training_seconds']/60 for r in results])), per_seed=results))
    events = [json.loads(line) for line in (root/'runtime.jsonl').read_text().splitlines() if line.strip()]
    write(root/'summary.json', dict(complete=True, identity=identity(config), methods=rows,
          target=.93, frozen_conv24_mean=.93319786, frozen_bilstm_mean=.919499358,
          exceeds_both_frozen_means=rows[0]['f1_mean']>max(.93319786,.919499358),
          target_reached=rows[0]['f1_mean']>=.93, checkpoint_policy='memory_only',
          runtime_events_at_merge=len(events)))


def preflight(spec, config, device):
    engine.deterministic(device); torch.manual_seed(91900)
    axis = torch.arange(-1, 2, device=device, dtype=torch.float32)
    seeds = torch.cartesian_prod(axis, axis, axis)[None].repeat(2, 1, 1)*.03
    counts = torch.tensor([27, 10], device=device)
    t = torch.linspace(-1, 1, 32, device=device)
    g = seeds[:, :, None, :]+torch.stack((t, .2*t*t, .1*torch.sin(3*t)), -1)[None, None]
    g *= method.line_mask(counts, 27)[..., None, None]
    n = method.neighbor_indices(seeds, counts, 'fps6')
    old = old_encode(g, seeds, counts, dict(geometry='max_radius', frequencies=6, feature_dimensions=141), 'fps6')
    new = method.fourier_tokens(g, counts, n, 'p35')
    assert torch.equal(old, new[..., :-1]), '32-point no-augmentation encoder must exactly match frozen FPS'
    assert torch.equal(method.resample(g, 32, 'uniform'), g)
    for points, sampling in itertools.product(spec['points'], spec['sampling']):
        out = method.resample(g, points, sampling)
        assert out.shape==(2,27,points,3) and torch.isfinite(out).all()
        assert torch.equal(out[..., 0, :],g[..., 0, :]) and torch.equal(out[..., -1, :],g[..., -1, :])
        # Independent NumPy interpolation for uniform arclength.
        if sampling=='uniform' and points!=32:
            line=g[0,0].cpu().numpy().astype(np.float64); arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
            expected=np.stack([np.interp(np.linspace(0,arc[-1],points),arc,line[:,d]) for d in range(3)],-1)
            assert np.allclose(out[0,0].cpu(),expected,atol=1e-7)
        for aug in spec['augmentations']:
            gen = torch.Generator(device=device).manual_seed(42)
            changed = method.augment(out, counts, aug, gen)
            assert torch.isfinite(changed).all() and torch.all(changed[1,10:]==0)
            if aug=='jitter':
                assert torch.equal(changed[...,0,:],out[...,0,:]) and torch.equal(changed[...,-1,:],out[...,-1,:])
                assert torch.all(changed[0,:,1:,0]>changed[0,:,:-1,0])
            if aug in ('rotate15','rotate_so3'):
                assert torch.allclose(torch.cdist(out[0,0].double(),out[0,0].double()),
                                      torch.cdist(changed[0,0].double(),changed[0,0].double()),atol=5e-7,rtol=5e-6)
            tokens=method.fourier_tokens(changed, counts, n, 'p35')
            assert torch.isfinite(tokens).all()
    for rotation_mode in ('rotate15','rotate_so3'):
        r=method.rotation_matrices(32,rotation_mode,torch.Generator(device=device).manual_seed(3),device)
        assert torch.allclose(r@r.transpose(-1,-2),torch.eye(3,device=device,dtype=r.dtype),atol=1e-12)
        assert torch.allclose(torch.det(r),torch.ones(32,device=device,dtype=r.dtype),atol=1e-12)
    checked = {}
    unique = {(c['pool'],c['architecture'],c['profile']) for c in candidates(spec)}
    for pool, arch, profile in sorted(unique):
        tokens = method.fourier_tokens(g, counts, n, pool)
        model = method.FourierClassifier(pool, arch, profile).to(device)
        model.eval(); result = model(tokens,n)
        padded = tokens.clone(); padded[1,10:,:-1]=1000
        assert torch.allclose(result,model(padded,n),atol=1e-6,rtol=1e-5)
        assert torch.allclose(result,torch.cat([model(tokens[i:i+1],n[i:i+1]) for i in range(2)]),atol=3e-6,rtol=1e-5)
        permutation = torch.randperm(27, device=device)
        inverse = torch.argsort(permutation)
        remapped = inverse[n[:, permutation]]
        assert torch.allclose(result,model(tokens[:,permutation],remapped),atol=3e-6,rtol=1e-5)
        model.train(); loss=F.cross_entropy(model(tokens,n),torch.tensor([0,1],device=device));loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checked[f'{pool}/{arch}/{profile}']=sum(p.numel() for p in model.parameters())
    write(Path(spec['output'])/f'preflight_{device}.json',dict(complete=True,identity=identity(config),
          frozen_FPS_features_exact=True,parameters=checked,augmentation_geometry_checks=True,
          padding_batch_gradient_checks=True,device=device,candidates=len(candidates(spec))))
    print(json.dumps(checked),flush=True)


def pilot(spec, config):
    engine.deterministic('cuda'); root=Path(spec['output']); results=[]
    assert json.loads((root/'preflight_cuda.json').read_text())['complete']
    reference_checks={}
    for flow in spec['flows']:
        folder=root/'physical'/flow['name']/'train'
        g=torch.tensor(np.array(np.load(folder/'geometry.npy',mmap_mode='r')[:32]),device='cuda')
        s=torch.tensor(np.array(np.load(folder/'seeds.npy',mmap_mode='r')[:32]),device='cuda')
        with np.load(folder/'metadata.npz') as z:c=torch.tensor(z['counts'][:32].astype(np.int64),device='cuda')
        n=method.neighbor_indices(s,c,'fps6');tokens=method.fourier_tokens(g,c,n,'p35')
        cached=Path(spec['fps_reference_output'])/'encoded'/flow['name']/'train'/'fps6.npy'
        manifest=json.loads((cached.parent/'manifest.json').read_text())
        assert sha(cached)==manifest['files']['fps6.npy']
        assert np.array_equal(tokens.cpu().numpy(),np.load(cached,mmap_mode='r')[:32])
        reference_checks[flow['name']]=dict(frozen_fps_tokens_exact=True,cache_sha256=sha(cached))
    for arch in method.ARCHITECTURES:
        torch.manual_seed(91920)
        c=dict(reference_candidate(),architecture=arch,points=48,sampling='curvature',augmentation='jitter_rotate_so3')
        dataset=Dataset(spec,'train',c,limit=128);dataset.encode(c)
        norm=fit_normalizer(dataset,c);model=method.FourierClassifier('p35',arch).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=standardize(dataset.clean[:32],norm);neighbors=dataset.neighbors[:32];y=dataset.targets[:32]
        # Balanced first 32 examples within each source are selected explicitly.
        ids=torch.cat([(dataset.targets==v).nonzero().flatten()[:16] for v in (0,1)])
        x=standardize(dataset.clean[ids],norm);neighbors=dataset.neighbors[ids];y=dataset.targets[ids]
        model.eval(); initial=float(F.cross_entropy(model(x,neighbors),y))
        for _ in range(100):
            model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x,neighbors),y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        model.eval();final=float(F.cross_entropy(model(x,neighbors),y));assert final<initial
        gen=torch.Generator(device='cuda').manual_seed(100)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(3):
            model.train();optimizer.zero_grad(set_to_none=True)
            geometry=method.augment(dataset.geometry,dataset.counts,c['augmentation'],gen)
            tokens=method.fourier_tokens(geometry,dataset.counts,dataset.neighbors,'p35')
            loss=F.cross_entropy(model(standardize(tokens,norm),dataset.neighbors),dataset.targets)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize();step=(time.perf_counter()-start)/3
        results.append(dict(architecture=arch,parameters=sum(p.numel() for p in model.parameters()),
            initial_loss=initial,final_loss=final,batch=len(dataset.labels),seconds_per_batch=step,
            estimated_epoch_seconds=step*np.ceil(193000/128),peak_bytes=torch.cuda.max_memory_allocated()))
        del dataset,model,optimizer,x,tokens,geometry;torch.cuda.empty_cache()
    write(root/'pilot.json',dict(complete=True,identity=identity(config),train_only=True,models=results,reference_checks=reference_checks))
    print(json.dumps(results),flush=True)


def submit(spec, config):
    root=Path(spec['output']).resolve();root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    grid=candidates(spec);write(root/'candidates.json',grid);write(root/'config.frozen.json',spec)
    phases=[('preflight',None,True,'00:30:00'),('reuse',None,False,'00:30:00'),
            ('pilot',None,True,'01:00:00'),
            ('screen',f"0-{len(grid)-1}%{spec['parallel_gpus']}",True,'12:00:00'),
            ('shortlist',None,False,'00:30:00'),
            ('refine',f"0-{spec['refine']['candidates']*len(spec['refine']['seeds'])-1}%{spec['parallel_gpus']}",True,'1-12:00:00'),
            ('select',None,False,'00:30:00'),
            ('final',f"0-{2*len(spec['final']['seeds'])-1}%6",True,'1-12:00:00'),('merge',None,False,'00:30:00')]
    dependency=None
    for phase,array,gpu,limit in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G','--time='+limit,
                 '--job-name=t4c-fpsaug-'+phase,f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100']
        if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        command+=['ibex_bash/task4c_fps_augment_search_1p1.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,version=spec['version'],phase=phase,submitted_at_utc=datetime.now(timezone.utc).isoformat(),
                 command=command,dependency=dependency,identity=identity(config),expected_device='V100' if gpu else 'CPU')
        append_locked(root/'submissions.jsonl',json.dumps(row)+'\n');print(json.dumps(row),flush=True);dependency=job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','reuse','pilot','screen','shortlist','refine','select','final','merge','submit','runtime'))
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--device',default='cuda');parser.add_argument('--runtime-phase');parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=load_spec(args.config)
    if args.phase=='preflight':preflight(spec,args.config,args.device)
    elif args.phase in ('screen','refine','final'):train(spec,args.config,args.index,args.phase)
    elif args.phase=='runtime':
        fn=engine.runtime
        FunctionType(fn.__code__,dict(engine.__dict__,identity=identity),argdefs=fn.__defaults__)(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    else:globals()[args.phase](spec,args.config)


if __name__=='__main__':
    main()
