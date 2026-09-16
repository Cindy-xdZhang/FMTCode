"""Append a capacity-matched classic PointNet to the frozen Task4-c comparison."""
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
from FMT_Utils.Task4C_PointNet_1_1 import PARAMETERS, make_model
from experiments import Task4C_GeometricBaselines_1_1 as geometry
from experiments import Task4C_BottomDensity_1_2 as source
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine

CONFIG = 'config/Ablation_Task4C_PointNet_1.1.json'
SPLITS = engine.SPLITS
write, sha, metrics, append_locked, deterministic = engine.write, engine.sha, engine.metrics, engine.append_locked, engine.deterministic


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    assert sha(definition['base_config']) == definition['base_config_sha256']
    spec = source.load_spec(definition['base_config']); spec.update(definition)
    assert spec['methods'] == ['pointnet_small']
    assert spec['pointnet']['parameters'] == PARAMETERS
    assert (spec['pointnet']['width'], spec['pointnet']['global_width']) == (11, 168)
    assert spec['pointnet']['transform_hidden'] == [88, 44] and spec['pointnet']['classifier_hidden'] == [88, 39]
    assert spec['pointnet']['feature_transform_regularization_weight'] == .001
    assert spec['pointnet']['dropout'] == spec['training']['dropout'] == .15
    assert spec['dataset_policy'] == 'reuse_all_source_geometry_seeds_metadata_without_sampling'
    return spec


def identity(config):
    result = geometry.identity(config)
    for file in ('FMT_Utils/Task4C_PointNet_1_1.py', 'experiments/Task4C_PointNet_1_1.py',
                 'ibex_bash/task4c_pointnet_1p1.sh'):
        result['sources'][file] = sha(file)
    return result


def run_engine(name, *args):
    fn = getattr(engine, name)
    return FunctionType(fn.__code__, dict(engine.__dict__, identity=identity), argdefs=fn.__defaults__)(*args)


def reuse(spec, config):
    fn = geometry.reuse
    FunctionType(fn.__code__, dict(geometry.__dict__, identity=identity), argdefs=fn.__defaults__)(spec, config)


def load_parts(spec, role, method):
    assert method == 'pointnet_small'
    return geometry.load_parts(spec, role, 'baseline2')


def get_batch(parts, ids, method, norm, device='cuda'):
    assert method == 'pointnet_small' and norm is None
    return geometry.get_batch(parts, ids, 'baseline2', None, device)


def predict(model, dataset, method, norm, batch):
    fn = engine.predict
    return FunctionType(fn.__code__, dict(engine.__dict__, get_batch=get_batch), argdefs=fn.__defaults__)(model, dataset, method, norm, batch)


def preflight(spec, config, device):
    deterministic(device); torch.manual_seed(91711)
    model = make_model(.15).to(device)
    counts = torch.tensor([10, 15, 22, 27], device=device)
    g = torch.randn((4, 27, 32, 3), device=device)*.1
    valid = torch.arange(27, device=device)[None] < counts[:, None]
    changed = g.clone(); changed[~valid] = 12345.
    shuffled = g.clone()
    for i, count in enumerate(counts.tolist()):
        cloud = g[i, :count].reshape(-1, 3)
        shuffled[i, :count] = cloud[torch.randperm(len(cloud), device=device)].reshape(count, 32, 3)
    model.eval()
    with torch.no_grad():
        a = model((g, counts)); initial_transform = model.last_feature_transform.clone()
        assert torch.equal(initial_transform, torch.eye(11, device=device).expand(4, -1, -1))
        assert float(model.orthogonality_penalty()) == 0
        assert torch.allclose(a, model((changed, counts)), atol=1e-6, rtol=1e-6)
        assert torch.allclose(a, model((shuffled, counts)), atol=1e-6, rtol=1e-6)
        padded = F.pad(g, (0, 0, 0, 0, 0, 5))
        assert torch.allclose(a, model((padded, counts)), atol=1e-6, rtol=1e-6)
    # Training BatchNorm statistics and outputs must also exclude padding completely.
    first = copy.deepcopy(model).train(); second = copy.deepcopy(model).train()
    torch.manual_seed(91712); out1 = first((g, counts))
    torch.manual_seed(91712); out2 = second((changed, counts))
    assert torch.equal(out1, out2)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key
    del first, second
    model.train(); yy = torch.tensor([0, 1, 0, 1], device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model((g, counts)), yy)+.001*model.orthogonality_penalty()
        assert torch.isfinite(loss); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optim.step()
    assert model.input_transform.output.weight.abs().sum() > 0
    assert model.feature_transform.output.weight.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        learned = model((g, counts))
        assert torch.allclose(learned, model((shuffled, counts)), atol=2e-6, rtol=2e-6)
        assert torch.allclose(learned[:1], model((g[:1], counts[:1])), atol=2e-6, rtol=2e-6)
    synthetic = (2*torch.eye(11, device=device)[None].repeat(4, 1, 1)).requires_grad_()
    model.last_feature_transform = synthetic
    penalty = model.orthogonality_penalty(); assert abs(float(penalty.detach())-198) < 1e-5
    penalty.backward(); assert torch.isfinite(synthetic.grad).all() and synthetic.grad.abs().sum() > 0
    result = dict(complete=True, identity=identity(config), device=device, parameters=PARAMETERS,
                  point_permutation_invariance=True, training_and_eval_padding_excluded=True,
                  both_transforms_identity_initialized_and_trainable=True, analytic_regularizer=True,
                  finite_forward_backward=True)
    write(Path(spec['output'])/f'preflight_{device}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'identity'}), flush=True)


def pilot(spec, config):
    deterministic('cuda'); torch.manual_seed(91713)
    root = Path(spec['output']); geometries = []; counts = []; labels = []
    for flow in spec['flows']:
        folder = root/'physical'/flow['name']/'train'
        g = np.load(folder/'geometry.npy', mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:
            rows = np.concatenate([np.flatnonzero(z['labels'] == cls)[:8] for cls in (0, 1)])
            geometries.append(np.array(g[rows])); counts.append(z['counts'][rows]); labels.append(z['labels'][rows])
    x = (torch.as_tensor(np.concatenate(geometries), device='cuda'),
         torch.as_tensor(np.concatenate(counts).astype(np.int64), device='cuda'))
    y = np.concatenate(labels); yy = torch.as_tensor(y, device='cuda', dtype=torch.long)
    model = make_model(.15).cuda(); model.eval()
    with torch.no_grad(): initial = float(F.cross_entropy(model(x), yy))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    start = time.perf_counter()
    for _ in range(100):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), yy)+.001*model.orthogonality_penalty()
        assert torch.isfinite(loss); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(x); final = float(F.cross_entropy(logits, yy)); p = logits.softmax(-1)[:, 1].cpu().numpy()
    assert final < initial, (initial, final)
    model.train(); optimizer.zero_grad(set_to_none=True)
    big = (x[0].repeat(4, 1, 1, 1), x[1].repeat(4))
    loss = F.cross_entropy(model(big), yy.repeat(4))+.001*model.orthogonality_penalty(); loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    torch.cuda.synchronize()
    result = dict(complete=True, identity=identity(config), train_only=True, samples=32,
                  initial_loss=initial, final_loss=final, training_metrics=metrics(y, p),
                  seconds=time.perf_counter()-start, batch128_backward=True, parameters=PARAMETERS)
    write(root/'pilot.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'identity'}), flush=True)


def train(spec,config,index):
    deterministic('cuda');seeds=spec['training']['seeds'];method=spec['methods'][index//len(seeds)];seed=seeds[index%len(seeds)]
    torch.manual_seed(seed);rng=np.random.default_rng(seed);options=spec['training'];batch=options['batch_size']
    root=Path(spec['output']);folder=root/'runs'/method/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    training=load_parts(spec,'train',method);validation=load_parts(spec,'validation',method)
    parts,ids,y,_,_=training;assert len(y)==spec['expected_counts']['train']
    norm=None
    model=make_model(options['dropout']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=options['lr_patience'],threshold=1e-4,min_lr=1e-6)
    best=(-1.,-1.);best_epoch=0;state=None;history=[];started=time.perf_counter()
    for epoch in range(1,options['epochs']+1):
        order=rng.permutation(len(y));model.train();total=0.;lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(y),batch):
            chosen=order[first:first+batch];xx=get_batch(parts,ids[chosen],method,norm)
            yy=torch.as_tensor(y[chosen],device='cuda',dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(xx),yy)+spec['pointnet']['feature_transform_regularization_weight']*model.orthogonality_penalty()
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
    fn = geometry.reuse_source.merge
    FunctionType(fn.__code__, dict(geometry.reuse_source.__dict__, identity=identity, run_engine=run_engine), argdefs=fn.__defaults__)(spec, config)
    file = Path(spec['output'])/'viewer_package/manifest.json'; manifest = json.loads(file.read_text())
    manifest['evidence_note'] = 'Classic PointNet with both T-Nets, reduced widths, same frozen 193000/3000/10000 dataset; seed 96611.'
    write(file, manifest)


def submit(spec, config):
    root = Path(spec['output']).resolve(); assert shutil.disk_usage(Path.cwd()).free > 10*2**30
    root.mkdir(parents=True, exist_ok=False); (root/'logs').mkdir(); write(root/'config.frozen.json', spec)
    dependency = None
    phases = [('preflight', None, True, '00:20:00'), ('reuse', None, False, '00:30:00'),
              ('pilot', None, True, '00:30:00'), ('train', '0-2%3', True, '24:00:00'), ('merge', None, False, '01:00:00')]
    for phase, array, gpu, limit in phases:
        command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=32G', '--time='+limit,
                   '--job-name=t4c-pointnet-'+phase, f'--output={root}/logs/{phase}_%A_%a.out', f'--error={root}/logs/{phase}_%A_%a.err']
        if array: command += ['--array='+array]
        if gpu: command += ['--gres=gpu:1', '--constraint=v100']
        if dependency: command += ['--dependency=afterok:'+dependency, '--kill-on-invalid-dep=yes']
        command += ['ibex_bash/task4c_pointnet_1p1.sh', phase, config]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        row = dict(job_id=job, phase=phase, submitted_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                   dependency=dependency, expected_device='V100' if gpu else 'CPU', identity=identity(config))
        append_locked(root/'submissions.jsonl', json.dumps(row)+'\n'); print(json.dumps(row), flush=True); dependency = job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'reuse', 'pilot', 'train', 'merge', 'submit', 'runtime'))
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--runtime-phase')
    parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec = load_spec(args.config)
    if args.phase == 'preflight': preflight(spec, args.config, args.device)
    elif args.phase == 'train': train(spec, args.config, args.index)
    elif args.phase == 'runtime': run_engine('runtime', spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else: globals()[args.phase](spec, args.config)


if __name__ == '__main__': main()
