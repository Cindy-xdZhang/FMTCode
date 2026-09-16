"""Three fixed-data, capacity-matched Task4-c geometry baselines."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
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
from FMT_Utils.Task4C_GeometricBaselines_1_1 import PARAMETERS, PointNNEncoder, make_model, tangent_curvature
from experiments import Task4C_BottomDensity_1_2 as source
from experiments import Task4C_BottomDensity_1_3 as reuse_source
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine

CONFIG = 'config/Ablation_Task4C_GeometricBaselines_1.1.json'
SPLITS = engine.SPLITS
write, sha, append_locked, metrics, deterministic = engine.write, engine.sha, engine.append_locked, engine.metrics, engine.deterministic


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    assert sha(definition['base_config']) == definition['base_config_sha256']
    spec = source.load_spec(definition['base_config'])
    spec.update(definition)
    assert spec['methods'] == list(PARAMETERS)
    assert spec['dataset_policy'] == 'reuse_all_source_geometry_seeds_metadata_without_sampling'
    for method, n in PARAMETERS.items():
        assert spec['baseline_definitions'][method]['parameters'] == n
        assert abs(n/spec['reference_fmt_parameters']-1) < .001
    return spec


def identity(config):
    result = source.identity(config)
    for name in ('experiments/Task4C_GeometricBaselines_1_1.py', 'FMT_Utils/Task4C_GeometricBaselines_1_1.py',
                 'ibex_bash/task4c_geometric_baselines_1p1.sh', 'experiments/Task4C_BottomDensity_1_3.py',
                 'config/Ablation_Task4C_BottomDensity_1.2.json'):
        result['sources'][name] = sha(name)
    return result


def run_engine(name, *args):
    fn = getattr(engine, name)
    return FunctionType(fn.__code__, dict(engine.__dict__, identity=identity), argdefs=fn.__defaults__)(*args)


def reuse(spec, config):
    fn = reuse_source.reuse
    FunctionType(fn.__code__, dict(reuse_source.__dict__, identity=identity), argdefs=fn.__defaults__)(spec, config)


def pointnn_bundles(geometry, counts, batch_size=4, device='cuda'):
    result = np.empty((len(geometry), 1152), np.float32)
    encoder = PointNNEncoder().to(device)
    for count in np.unique(counts):
        rows = np.flatnonzero(counts == count)
        for first in range(0, len(rows), batch_size):
            selected = rows[first:first+batch_size]
            xyz = torch.as_tensor(np.array(geometry[selected, :count]), device=device).reshape(len(selected), -1, 3)
            result[selected] = encoder(xyz).cpu().numpy()
    assert np.isfinite(result).all()
    return result


def encode(spec, config, index):
    deterministic('cuda')
    root = Path(spec['output']); name = spec['flows'][index]['name']
    report = engine.physical_reports(spec)[name]
    assert json.loads((root/'data_audit.json').read_text())['complete']
    records = {}; started = time.perf_counter()
    for split in SPLITS:
        folder = root/'physical'/name/split
        for filename, digest in report['splits'][split]['files'].items():
            assert sha(folder/filename) == digest
        geometry = np.load(folder/'geometry.npy', mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:
            counts = z['counts'].astype(np.int64)
        dest = root/'encoded'/name/split; dest.mkdir(parents=True, exist_ok=False)
        features0 = np.lib.format.open_memmap(dest/'baseline0.npy', mode='w+', dtype='float32', shape=(len(geometry), 8))
        features1 = np.lib.format.open_memmap(dest/'baseline1.npy', mode='w+', dtype='float32', shape=(len(geometry), 1152))
        for first in range(0, len(geometry), 256):
            sl = slice(first, first+256)
            g = torch.as_tensor(np.array(geometry[sl]), device='cuda')
            c = torch.as_tensor(counts[sl], device='cuda')
            features0[sl] = tangent_curvature(g, c).cpu().numpy()
        # Grouping by valid line count avoids artificial cloud padding or point duplication.
        completed = 0
        for count in np.unique(counts):
            rows = np.flatnonzero(counts == count)
            for first in range(0, len(rows), 256):
                chosen = rows[first:first+256]
                features1[chosen] = pointnn_bundles(geometry[chosen], counts[chosen], spec['pointnn_encoding_batch_size'])
                completed += len(chosen)
                if completed//5000 != (completed-len(chosen))//5000:
                    print('encode', name, split, completed, len(geometry), 'seconds', time.perf_counter()-started, flush=True)
        assert np.isfinite(features0).all() and np.isfinite(features1).all()
        features0.flush(); features1.flush(); del features0, features1
        records[split] = dict(samples=len(geometry), files={f.name: sha(f) for f in dest.glob('*.npy')})
        write(dest/'manifest.json', records[split])
    write(root/'encoding'/f'{index}.json', dict(complete=True, identity=identity(config),
        splits=records, feature_encoding_seconds=time.perf_counter()-started))


def load_parts(spec, role, method):
    root = Path(spec['output']); parts = []; ids = []; labels = []; flows = []; instances = []
    for fi, flow in enumerate(spec['flows']):
        src = root/'physical'/flow['name']/role
        with np.load(src/'metadata.npz') as z:
            y = z['labels'].astype(np.int64); owner = z['instance'].copy(); counts = z['counts'].astype(np.int64)
        if method == 'baseline2':
            expected = json.loads((src.parent/'preparation.json').read_text())['splits'][role]['files']['geometry.npy']
            assert sha(src/'geometry.npy') == expected
            part = dict(geometry=np.load(src/'geometry.npy', mmap_mode='r'), counts=counts)
        else:
            cache = root/'encoded'/flow['name']/role
            manifest = json.loads((cache/'manifest.json').read_text())
            file = cache/f'{method}.npy'; assert sha(file) == manifest['files'][file.name]
            part = dict(features=np.load(file, mmap_mode='r'))
        ids.extend((fi, i) for i in range(len(y))); labels.append(y); instances.append(owner)
        flows.extend([fi]*len(y)); parts.append(part)
    return parts, np.asarray(ids, np.int64), np.concatenate(labels), np.asarray(flows), np.concatenate(instances)


def normalization(parts):
    count = 0; width = parts[0]['features'].shape[1]
    mean = np.zeros(width, np.float64); m2 = mean.copy()
    for part in parts:
        for first in range(0, len(part['features']), 4096):
            values = np.asarray(part['features'][first:first+4096], np.float64)
            n = len(values); local_mean = values.mean(0); delta = local_mean-mean
            m2 += ((values-local_mean)**2).sum(0)+delta*delta*count*n/(count+n)
            mean += delta*n/(count+n); count += n
    std = np.sqrt(m2/count); std[std < 1e-8] = 1
    return mean, std


def get_batch(parts, ids, method, norm, device='cuda'):
    if method == 'baseline2':
        x = np.stack([parts[p]['geometry'][i] for p, i in ids])
        counts = np.asarray([parts[p]['counts'][i] for p, i in ids], np.int64)
        return torch.as_tensor(x, device=device), torch.as_tensor(counts, device=device)
    x = np.stack([parts[p]['features'][i] for p, i in ids])
    x = ((x-norm[0])/norm[1]).astype(np.float32)
    return torch.as_tensor(x, device=device)


def predict(model, dataset, method, norm, batch):
    fn = engine.predict
    return FunctionType(fn.__code__, dict(engine.__dict__, get_batch=get_batch), argdefs=fn.__defaults__)(model, dataset, method, norm, batch)


def preflight(spec, config, device):
    deterministic(device); torch.manual_seed(91701)
    t = torch.linspace(-1, 1, 32, device=device)
    circle = torch.stack((.5*torch.cos(t), .5*torch.sin(t), torch.zeros_like(t)), -1)
    straight = torch.stack((t, t*0, t*0), -1)
    g = torch.stack((circle, straight))[:, None].repeat(1, 27, 1, 1)
    counts = torch.tensor([10, 27], device=device)
    f = tangent_curvature(g, counts)
    assert torch.allclose(f[0, [3, 7]], torch.tensor([2., 2.], device=device), atol=3e-4, rtol=0)
    assert torch.equal(f[1], torch.tensor([1., 0., 0., 0., 1., 0., 0., 0.], device=device))
    changed = g.clone(); changed[0, 10:] = torch.randn_like(changed[0, 10:])*100
    assert torch.equal(tangent_curvature(changed, counts), f)
    p = torch.rand((2, 64, 3), device=device)
    encoder = PointNNEncoder().to(device)
    a = encoder(p); perm = torch.randperm(64, device=device)
    assert torch.equal(a, encoder(p[:, perm]))
    assert torch.allclose(a[:1], encoder(p[:1]), atol=2e-5, rtol=2e-5)
    assert not list(encoder.parameters()) and torch.isfinite(a).all()
    parameter_counts = {}
    for method in spec['methods']:
        model = make_model(method, .15).to(device)
        model.eval()
        if method == 'baseline2':
            sequences = torch.rand_like(g)
            x = (sequences, counts)
            padded = sequences.clone(); padded[0, 10:] = torch.randn_like(padded[0, 10:])*100
            assert torch.allclose(model(x), model((padded, counts)), atol=1e-6, rtol=1e-6)
            swapped = sequences.clone(); swapped[0, :10] = swapped[0, :10].flip(0); swapped[1] = swapped[1].flip(0)
            assert torch.allclose(model(x), model((swapped, counts)), atol=1e-6, rtol=1e-6)
            assert not torch.allclose(model(x), model((sequences.flip(2), counts)), atol=1e-6, rtol=1e-6)
        else:
            x = f if method == 'baseline0' else a
        model.train(); loss = F.cross_entropy(model(x), torch.tensor([0, 1], device=device)); loss.backward()
        assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        parameter_counts[method] = sum(v.numel() for v in model.parameters())
    result = dict(complete=True, identity=identity(config), device=device, parameters=parameter_counts,
        analytic_straight_line_and_circle=True, padding_exclusion=True, point_permutation_invariance=True,
        point_batch_independence=True, finite_forward_backward=True)
    write(Path(spec['output'])/f'preflight_{device}.json', result); print(json.dumps(result), flush=True)


def pilot(spec, config):
    """Training-only finite-gradient, real-bundle fit and throughput checks; no tuning."""
    deterministic('cuda'); torch.manual_seed(91702); root = Path(spec['output'])
    geometry = []; counts = []; labels = []
    for flow in spec['flows']:
        folder = root/'physical'/flow['name']/'train'
        g = np.load(folder/'geometry.npy', mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:
            rows = np.concatenate([np.flatnonzero(z['labels'] == cls)[:8] for cls in (0, 1)])
            geometry.append(np.array(g[rows])); counts.append(z['counts'][rows]); labels.append(z['labels'][rows])
    g = np.concatenate(geometry); c = np.concatenate(counts).astype(np.int64); y = np.concatenate(labels)
    geometry_tensor = torch.as_tensor(g, device='cuda'); count_tensor = torch.as_tensor(c, device='cuda')
    torch.cuda.synchronize(); start = time.perf_counter()
    fixed = dict(baseline0=tangent_curvature(geometry_tensor, count_tensor).cpu().numpy(),
                 baseline1=pointnn_bundles(g, c, spec['pointnn_encoding_batch_size']))
    torch.cuda.synchronize(); encode_seconds = time.perf_counter()-start
    rows = []
    for method in spec['methods']:
        torch.manual_seed(91702)
        model = make_model(method, .15).cuda()
        if method == 'baseline2':
            x = (geometry_tensor, count_tensor)
        else:
            values = fixed[method]; std = values.std(0, dtype=np.float64); std[std < 1e-8] = 1
            x = torch.as_tensor(((values-values.mean(0, dtype=np.float64))/std).astype(np.float32), device='cuda')
        yy = torch.as_tensor(y, device='cuda', dtype=torch.long)
        model.eval()
        with torch.no_grad(): initial = float(F.cross_entropy(model(x), yy))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        start = time.perf_counter()
        for _ in range(100):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), yy); assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x); final = float(F.cross_entropy(prediction, yy)); p = prediction.softmax(-1)[:, 1].cpu().numpy()
        assert final < initial, (method, initial, final)
        # One actual training-size backward pass before committing the full training budget.
        big = (x[0].repeat(4, 1, 1, 1), x[1].repeat(4)) if method == 'baseline2' else x.repeat(4, 1)
        model.train(); optimizer.zero_grad(set_to_none=True); F.cross_entropy(model(big), yy.repeat(4)).backward()
        torch.cuda.synchronize()
        rows.append(dict(method=method, initial_loss=initial, final_loss=final, training_metrics=metrics(y, p),
                         seconds=time.perf_counter()-start, batch128_backward=True))
    write(root/'pilot.json', dict(complete=True, identity=identity(config), train_only=True, samples=len(y),
          feature_encoding_seconds=encode_seconds, methods=rows)); print(json.dumps(rows), flush=True)


def train(spec,config,index):
    deterministic('cuda');seeds=spec['training']['seeds'];method=spec['methods'][index//len(seeds)];seed=seeds[index%len(seeds)]
    torch.manual_seed(seed);rng=np.random.default_rng(seed);options=spec['training'];batch=options['batch_size']
    root=Path(spec['output']);folder=root/'runs'/method/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    training=load_parts(spec,'train',method);validation=load_parts(spec,'validation',method)
    parts,ids,y,_,_=training;assert len(y)==spec['expected_counts']['train']
    norm=normalization(parts) if method!='baseline2' else None
    model=make_model(method,options['dropout']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=options['lr_patience'],threshold=1e-4,min_lr=1e-6)
    best=(-1.,-1.);best_epoch=0;state=None;history=[];started=time.perf_counter()
    for epoch in range(1,options['epochs']+1):
        order=rng.permutation(len(y));model.train();total=0.;lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(y),batch):
            chosen=order[first:first+batch];xx=get_batch(parts,ids[chosen],method,norm)
            yy=torch.as_tensor(y[chosen],device='cuda',dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(xx),yy)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(chosen)
        probability,val_loss=predict(model,validation,method,norm,batch);score=metrics(validation[2],probability)
        rank=(score['f1'],score['average_precision'])
        row=dict(epoch=epoch,training_loss=total/len(y),validation_loss=val_loss,validation_f1=score['f1'],
            validation_average_precision=score['average_precision'],learning_rate=lr,samples=len(y),
            permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),seconds=time.perf_counter()-started)
        history.append(row);append_locked(folder/'history.jsonl',json.dumps(row)+'\n')
        if rank>best:best=rank;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        scheduler.step(val_loss)
        if epoch==1 or epoch%10==0:print(method,seed,row,flush=True)
        if epoch-best_epoch>=options['patience']:break
    model.load_state_dict(state);vp,_=predict(model,validation,method,norm,batch)
    assert abs(metrics(validation[2],vp)['f1']-best[0])<1e-12
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    write(folder/'selection.lock.json',dict(identity=identity(config),selected_epoch=best_epoch,threshold=.5,test_loaded=False))
    outputs={}
    for role,dataset in [('train',training),('validation',validation),('test',None)]:
        if role=='test':dataset=load_parts(spec,'test',method)
        p,_=predict(model,dataset,method,norm,batch);_,rowids,yy,flows,owners=dataset
        assert len(yy)==spec['expected_counts'][role]
        np.savez_compressed(folder/f'{role}_predictions.npz',labels=yy,probability=p,flow_index=flows,
            instance=owners,row_in_split=rowids[:,1],threshold=.5)
        outputs[role]=dict(combined=metrics(yy,p),per_flow={f['name']:metrics(yy[flows==i],p[flows==i]) for i,f in enumerate(spec['flows'])})
    result=dict(complete=True,version=spec['version'],execution_revision=spec['execution_revision'],identity=identity(config),
        method=method,seed=seed,parameters=sum(p.numel() for p in model.parameters()),epochs=len(history),selected_epoch=best_epoch,
        training_seconds=elapsed,metrics=outputs,threshold=.5,selection=options['selection'],history=history,
        normalization=None if norm is None else dict(mean=norm[0].tolist(),std=norm[1].tolist()),
        predictions={role:sha(folder/f'{role}_predictions.npz') for role in SPLITS},gpu=torch.cuda.get_device_name())
    write(folder/'result.json',result);print(json.dumps(outputs),flush=True)

def merge(spec, config):
    fn = reuse_source.merge
    FunctionType(fn.__code__, dict(reuse_source.__dict__, identity=identity, run_engine=run_engine), argdefs=fn.__defaults__)(spec, config)
    file = Path(spec['output'])/'viewer_package/manifest.json'; manifest = json.loads(file.read_text())
    manifest['evidence_note'] = 'Three geometry baselines on exactly the frozen 193000/3000/10000 fivefold 1.2 dataset; seed 96611.'
    write(file, manifest)


def submit(spec, config):
    root = Path(spec['output']).resolve(); assert shutil.disk_usage(Path.cwd()).free > 15*2**30
    root.mkdir(parents=True, exist_ok=False); (root/'logs').mkdir(); write(root/'config.frozen.json', spec)
    dependency = None
    phases = [('preflight', None, True, '00:20:00'), ('reuse', None, False, '00:30:00'),
              ('pilot', None, True, '00:30:00'), ('encode', '0-1%2', True, '08:00:00'),
              ('train', '0-8%9', True, '24:00:00'), ('merge', None, False, '01:00:00')]
    for phase, array, gpu, limit in phases:
        command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=32G', '--time='+limit,
                   '--job-name=t4c-geometry-'+phase, f'--output={root}/logs/{phase}_%A_%a.out', f'--error={root}/logs/{phase}_%A_%a.err']
        if array: command += ['--array='+array]
        if gpu: command += ['--gres=gpu:1', '--constraint=v100']
        if dependency: command += ['--dependency=afterok:'+dependency, '--kill-on-invalid-dep=yes']
        command += ['ibex_bash/task4c_geometric_baselines_1p1.sh', phase, config]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        row = dict(job_id=job, phase=phase, submitted_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                   dependency=dependency, expected_device='V100' if gpu else 'CPU', identity=identity(config))
        append_locked(root/'submissions.jsonl', json.dumps(row)+'\n'); print(json.dumps(row), flush=True); dependency = job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'reuse', 'pilot', 'encode', 'train', 'merge', 'submit', 'runtime'))
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--runtime-phase')
    parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec = load_spec(args.config)
    if args.phase == 'preflight': preflight(spec, args.config, args.device)
    elif args.phase in ('encode', 'train'): globals()[args.phase](spec, args.config, args.index)
    elif args.phase == 'runtime': run_engine('runtime', spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else: globals()[args.phase](spec, args.config)


if __name__ == '__main__': main()
