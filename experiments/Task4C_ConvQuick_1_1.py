"""Two voxel resolutions on the exact frozen 0.584912 FMT comparison rows."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.Task4C_LabelSplitQuick_1_1 import sha, write, verify_sources
from FMT_Utils import Task4C_LabelSplitQuick_1_1 as core
from FMT_Utils import Task4C_V2NewLabelTraining_1_1 as data
from experiments import Task4C_V2NewLabelTraining_1_1 as original
from experiments.Task4C_ConvResolution_4_22 import reference_voxel

CONFIG = 'config/Verify_Task4C_ConvQuick_1.1.json'


def spec():
    return json.loads(Path(CONFIG).read_text())


def inputs():
    s = spec(); verify_sources(); root = Path(s['baseline_output'])
    assert sha(root/'dataset.json') == s['baseline_dataset_sha256']
    assert sha(root/'result_audit.json') == s['baseline_audit_sha256']
    record = json.loads((root/'dataset.json').read_text())
    rows = {}
    for role, source in s['roles'].items():
        rows[role] = {}
        for key in ('new_label', 'sample', 'flow', 'length', 'instance', 'fold'):
            name = f'dataset/{source}/{key}.npy'
            assert sha(root/name) == record['files'][name]
            rows[role][key] = np.load(root/name)
    assert len(rows['train']['sample']) == 72000 and len(rows['test']['sample']) == 18000
    assert (rows['train']['fold'] != 0).all() and (rows['test']['fold'] == 0).all()
    for fi in (0, 1):
        tr, te = rows['train'], rows['test']
        assert not set(tr['instance'][tr['flow']==fi]) & set(te['instance'][te['flow']==fi])
    baseline = json.loads((root/'runs/v2/result.json').read_text())
    assert abs(baseline['primary']['test']['f1'] - .5849116362293324) < 1e-6
    return s, record, rows, baseline


def geometry(sources, rows, take):
    # Retain the frozen baseline's padded CPU normalization path.
    parts = [dict(source=source, samples=np.arange(len(source.seeds))) for source in sources]
    ids = np.column_stack((rows['flow'][take], rows['sample'][take]*3+rows['length'][take]))
    return data.old.baseline_geometry(parts, ids)


def train(index):
    s, record, rows, baseline = inputs(); resolution = s['resolutions'][index]
    dest = Path(s['output'])/'runs'/f'conv{resolution}'; dest.mkdir(parents=True, exist_ok=False)
    source_spec = json.loads(Path(s['training_config']).read_text())
    source_spec.update({k:record['config'][k] for k in ('source_output', 'neighbor_output')})
    physical = Path(source_spec['source_output'])
    assert sha(physical/'data_audit.json') == record['config']['source_audit_sha256']
    frozen = json.loads((physical/'data_audit.json').read_text())
    sources = [data.Source(source_spec, fi, 6) for fi in (0, 1)]
    evidence = {}
    for fi, name in enumerate(('channel', 'tbl')):
        for file in ('seeds.npy', 'curves.npy', 'metadata.npz'):
            assert sha(physical/'physical'/name/file) == frozen['frozen_files'][name][file]
        for role, d in rows.items():
            take = d['flow']==fi; ids = d['sample'][take]
            for key, source_key in (('new_label','label'), ('instance','instance'), ('fold','fold')):
                np.testing.assert_array_equal(d[key][take], sources[fi].meta[source_key][ids])
        neighbor_root = Path(source_spec['neighbor_output'])/name
        manifest = json.loads((neighbor_root/'manifest.json').read_text())
        assert manifest['complete'] and manifest['source_audit_sha256']==record['config']['source_audit_sha256']
        assert sha(neighbor_root/'order6.npy')==manifest['files']['order6.npy']
        evidence[name] = dict(neighbors_sha256=sha(neighbor_root/'order6.npy'))
    # Preserve exact selected IDs from the same frozen neighbour table used by FMT.
    for role, d in rows.items():
        neighbor_ids = np.empty((len(d['sample']),6),np.int64)
        for fi in (0,1):
            mask = d['flow']==fi; neighbor_ids[mask] = sources[fi].neighbors[d['sample'][mask]]
        np.save(dest/f'{role}_neighbors.npy', neighbor_ids)
        assert np.all(neighbor_ids != d['sample'][:,None])
    original.old.fmt_old.baseline.deterministic('cuda')
    probes = np.concatenate([np.flatnonzero(rows['train']['flow']==fi)[:6] for fi in (0,1)])
    # Re-encode actual source bundles and compare the FMT baseline's frozen rows.
    candidate=source_spec['candidates'][0]
    for role,d in rows.items():
        raw=np.load(Path(s['baseline_output'])/'dataset'/s['roles'][role]/'features.npy',mmap_mode='r')
        assert sha(Path(s['baseline_output'])/'dataset'/s['roles'][role]/'features.npy')==record['files'][f"dataset/{s['roles'][role]}/features.npy"]
        for fi in (0,1):
            take=np.flatnonzero(d['flow']==fi)[:6]
            lines,seeds,_=sources[fi].gather(d['sample'][take],d['length'][take])
            with torch.no_grad():
                gg,ss,cc=data.old.v2.normalize_bundle(torch.tensor(lines,device='cuda'),torch.tensor(seeds,device='cuda'))
                tokens,_=data.old.fmt.encode(gg,ss,cc,torch.zeros(len(take),device='cuda',dtype=torch.long),candidate)
                values=tokens.cpu().numpy();values[...,:141]=data.old.transform_fmt(values[...,:141],True)
            np.testing.assert_allclose(values,raw[take],atol=2e-4,rtol=2e-4)
    g, counts = geometry(sources, rows['train'], probes)
    gt, ct = torch.tensor(g,device='cuda'), torch.tensor(counts,device='cuda')
    vox = data.old.bundle_voxels(gt,ct,resolution,4).half().float()
    for i in range(len(g)):
        np.testing.assert_allclose(vox[i].cpu(), reference_voxel(g[i],int(counts[i]),resolution),atol=.0005,rtol=.002)
    torch.manual_seed(s['seed']); model = original.engine.Conv915(s['dropout']).cuda()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(vox),torch.cat([model(x) for x in vox.split(3)]),atol=2e-5,rtol=2e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s['learning_rate'],weight_decay=s['weight_decay'])
    target = torch.tensor(rows['train']['new_label'][probes].astype(np.int64),device='cuda')
    with torch.no_grad(): initial_loss = float(torch.nn.functional.cross_entropy(model(vox),target))
    for _ in range(20):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(vox),target); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),s['clip'],error_if_nonfinite=True); optimizer.step()
    model.eval()
    with torch.no_grad(): final_loss = float(torch.nn.functional.cross_entropy(model(vox),target))
    assert final_loss < initial_loss
    model.train(); model(vox[torch.arange(128,device='cuda')%len(vox)]).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    write(dest/'preflight.json',dict(complete=True,engineering_only=True,initial_loss=initial_loss,final_loss=final_loss,
        reference_voxel_rows=len(g),batch128_backward=True,source=evidence,gpu=torch.cuda.get_device_name()))
    cache = {}; started = time.perf_counter(); input_hashes = {}
    for role,d in rows.items():
        path = dest/f'{role}_voxels.npy'; n = len(d['sample'])
        values = np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(n,4,resolution,resolution,resolution))
        for first in range(0,n,128):
            ids = np.arange(first,min(first+128,n)); g,c = geometry(sources,d,ids)
            with torch.no_grad(): values[ids] = data.old.bundle_voxels(torch.tensor(g,device='cuda'),torch.tensor(c,device='cuda'),resolution,4).half().cpu().numpy()
            if first%16384==0: print(json.dumps(dict(stage='voxels',role=role,done=first+len(ids),total=n)),flush=True)
        values.flush(); input_hashes[role]=sha(path);cache[role]=np.load(path,mmap_mode='r')
    write(dest/'encoding.json',dict(complete=True,files=input_hashes,seconds=time.perf_counter()-started))
    def batch(role, ids): return torch.tensor(np.array(cache[role][ids]),device='cuda',dtype=torch.float32)
    torch.manual_seed(s['seed']); np.random.seed(s['seed']); rng=np.random.default_rng(s['seed'])
    model=original.engine.Conv915(s['dropout']).cuda()
    assert sum(p.numel() for p in model.parameters())==72192
    init_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    labels=torch.tensor(rows['train']['new_label'].astype(np.int64),device='cuda')
    weight=core.loss_weights(rows['train']['new_label']);np.testing.assert_allclose(weight,baseline['class_weights'],atol=0,rtol=0)
    weights=torch.tensor(weight,device='cuda',dtype=torch.float32)
    def predict(role):
        model.eval(); probabilities=[]
        with torch.no_grad():
            for first in range(0,len(rows[role]['sample']),256):
                probabilities.append(model(batch(role,slice(first,first+256))).softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(probabilities)
    history=[];started=time.perf_counter()
    for epoch in range(1,s['epochs']+1):
        order=rng.permutation(len(labels));model.train();total=0
        permutation_hash=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
        assert permutation_hash==baseline['history'][epoch-1]['permutation_sha256']
        for first in range(0,len(order),s['batch_size']):
            ids=order[first:first+s['batch_size']];optimizer.zero_grad(set_to_none=True)
            losses=torch.nn.functional.cross_entropy(model(batch('train',ids)),labels[ids],reduction='none')
            loss=(losses*weights[labels[ids]]).mean();assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),s['clip'],error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(ids)
        probability=predict('test'); training_probability=predict('train');te=rows['test']
        row=dict(epoch=epoch,loss=total/len(labels),train=core.score(rows['train']['new_label'],training_probability),
            test=core.score(te['new_label'],probability),per_flow={name:core.score(te['new_label'][te['flow']==fi],probability[te['flow']==fi]) for fi,name in enumerate(('channel','tbl'))},
            permutation_sha256=permutation_hash,seconds=time.perf_counter()-started)
        history.append(row);write(dest/'progress.json',dict(complete=False,latest=row))
        np.savez_compressed(dest/f'predictions_epoch{epoch:02}.npz',probability=probability,**te)
        print(json.dumps(row),flush=True)
    torch.save(model.state_dict(),dest/'final_model.pt')
    write(dest/'result.json',dict(complete=True,resolution=resolution,seed=s['seed'],history=history,primary=history[-1],
        parameters=72192,class_weights=weight.tolist(),initial_parameters_sha256=init_hash,dataset_sha256=s['baseline_dataset_sha256'],selection=s['selection']))


def audit():
    s,record,rows,baseline=inputs();out=Path(s['output']);results={};initial=[]
    for resolution in s['resolutions']:
        dest=out/'runs'/f'conv{resolution}';r=json.loads((dest/'result.json').read_text());assert r['complete']
        initial.append(r['initial_parameters_sha256'])
        for epoch in r['history']:
            with np.load(dest/f"predictions_epoch{epoch['epoch']:02}.npz") as z:
                for key,value in rows['test'].items():np.testing.assert_array_equal(z[key],value)
                scores=core.score(z['new_label'],z['probability'])
                for key,value in scores.items():
                    if value is not None:assert np.isclose(value,epoch['test'][key],atol=1e-12,rtol=0)
                y=z['new_label'].astype(bool);p=z['probability']>=.5
                assert np.isclose(2*(y&p).sum()/max(y.sum()+p.sum(),1),scores['f1'],atol=1e-12)
            assert epoch['permutation_sha256']==baseline['history'][epoch['epoch']-1]['permutation_sha256']
        results[f'conv{resolution}']=r['primary']
    assert len(set(initial))==1
    for role in rows:
        np.testing.assert_array_equal(np.load(out/'runs/conv8'/f'{role}_neighbors.npy'),np.load(out/'runs/conv16'/f'{role}_neighbors.npy'))
    write(out/'result_audit.json',dict(complete=True,results=results,fmt_baseline=baseline['primary'],
        dataset_sha256=s['baseline_dataset_sha256'],all_20_predictions_checked=True,same_shuffle_as_fmt=True,same_conv_initialization=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('train','audit'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    train(a.index) if a.phase=='train' else audit()
