"""Parameter-budget Conv3D ablation on the exact frozen Task4-c 4.14 inputs."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

from experiments import Task4C_PhysicalLength_4_14 as parent_run
from experiments import Task4C_NearbySampling_4_13 as training
from experiments import Task4C_ManifoldMixup_4_4 as pipeline
from FMT_Utils.Task4C_LinePooling_4_3 import LearnedLinePooling, WiderConv3D

CONFIG = 'config/Ablation_Task4C_ConvCapacity_4.18.json'
SPLITS = ('train', 'validation', 'test')


class BudgetConv3D(nn.Module):
    """Same operators as 4.14; only convolution and first head widths shrink."""
    def __init__(self, dropout):
        super().__init__()
        layers = []
        for i, (cin, cout) in enumerate(((4,12), (12,24), (24,48))):
            layers.extend((nn.Conv3d(cin,cout,3,padding=1), nn.GroupNorm(4,cout), nn.GELU(),
                nn.Dropout3d(dropout/2), nn.MaxPool3d(2) if i<2 else nn.AdaptiveAvgPool3d(2)))
        self.network = nn.Sequential(*layers, nn.Flatten(), nn.Linear(384,96), nn.LayerNorm(96),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(96,64), nn.Linear(64,2))

    def forward(self, x):
        return self.network(x)


def parameter_check(spec):
    new = sum(p.numel() for p in BudgetConv3D(.15).parameters())
    fmt = sum(p.numel() for p in LearnedLinePooling(.15).parameters())
    old = sum(p.numel() for p in WiderConv3D(.15).parameters())
    assert fmt == spec['fmt_reference_parameters'] == 88514
    assert new == spec['conv_parameters'] == 83918 and old == 219602
    assert spec['parameter_ratio_range'][0] <= new/fmt <= spec['parameter_ratio_range'][1]
    assert spec['conv_channels'] == [12,24,48] and spec['head_width'] == 96
    return dict(new_conv=new, original_fmt=fmt, original_conv=old, ratio_to_fmt=new/fmt)


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    assert pipeline.sha(definition['parent_config']) == definition['parent_config_sha256']
    spec = parent_run.load_spec(definition['parent_config'])
    spec.update(definition)
    parameter_check(spec)
    return spec


def identity(config):
    definition = json.loads(Path(config).read_text())
    result = parent_run.identity(definition['parent_config'])
    result['config_sha256'] = pipeline.sha(config)
    for name in ('experiments/Task4C_ConvCapacity_4_18.py', 'ibex_bash/task4c_conv_capacity_4p18.sh',
                 definition['parent_config']):
        result['source_sha256'][name] = pipeline.sha(name)
    return result


def parent_check(spec):
    root = Path(spec['parent_output'])
    assert pipeline.sha(root/'summary.json') == spec['parent_summary_sha256']
    old = json.loads((root/'summary.json').read_text())
    assert old['identity']['commit'] == spec['parent_commit']
    for name, sha in old['identity']['source_sha256'].items():
        assert pipeline.sha(name) == sha, name
    return old


def prepare(spec, config):
    parent_check(spec)
    root = Path(spec['output']); source = Path(spec['parent_output'])
    old = json.loads((source/'encoding.json').read_text())
    assert old['identity']['config_sha256'] == spec['parent_config_sha256']
    manifest = dict(version=spec['version'], identity=identity(config), splits={},
        parent_summary_sha256=spec['parent_summary_sha256'], exact_cached_voxels=True)
    for flow in spec['flows']:
        for split in SPLITS:
            key = flow['name']+'/'+split; dst = root/key; dst.mkdir(parents=True, exist_ok=False)
            files = {}
            for name in ('voxels.npy', 'metadata.npz'):
                src = source/key/name
                assert pipeline.sha(src) == old['splits'][key]['files'][name], str(src)
                (dst/name).symlink_to(src.resolve()); files[name] = pipeline.sha(dst/name)
            manifest['splits'][key] = dict(files=files, samples=old['splits'][key]['samples'])
    pipeline.write(root/'encoding.json', manifest)
    pipeline.write(root/'input_and_capacity_checks.json', dict(identity=identity(config),
        status='PASS', **parameter_check(spec), all_six_input_splits_hash_identical=True))


def train(spec, config, index):
    assert torch.cuda.get_device_name(0) == spec['required_device']
    assert json.loads((Path(spec['output'])/'input_and_capacity_checks.json').read_text())['status'] == 'PASS'
    originals = training.identity, training.make_model, training.load_data
    loads = []
    def timed_load(local, split, method, manifest, allow_test=False):
        start = time.perf_counter()
        result = originals[2](local,split,method,manifest,allow_test=allow_test)
        loads.append(dict(split=split,seconds=time.perf_counter()-start))
        return result
    try:
        training.identity = identity
        training.make_model = lambda method,candidate: BudgetConv3D(candidate['dropout'])
        training.load_data = timed_load
        training.train(spec,config,index)
    finally:
        training.identity, training.make_model, training.load_data = originals
    method,candidate,seed = training.plan(spec)[index]
    folder = training.run_folder(spec,method,candidate,seed)
    result = json.loads((folder/'result.json').read_text())
    assert result['parameters'] == spec['conv_parameters']
    assert [r['split'] for r in loads] == list(SPLITS)
    elapsed = np.array([r['seconds'] for r in result['history']])
    result.update(architecture='BudgetConv3D_4.18', parent_commit=spec['parent_commit'],
        parameter_ratio_to_fmt=result['parameters']/spec['fmt_reference_parameters'],
        test_scope='frozen_4.14_known_heads_local_spatial_blocks_already_used_benchmark',
        geometry_protocol='exact_frozen_4.14_voxel_cache', data_loading_times=loads,
        training_through_last_validation_seconds=float(elapsed[-1]),
        epoch_seconds_after_first_median=float(np.median(np.diff(elapsed))),
        epoch_seconds_after_first_mean=float(np.mean(np.diff(elapsed))))
    pipeline.write(folder/'result.json',result)


def independent_metrics(y,p,threshold=.5):
    y=np.asarray(y).astype(bool); p=np.asarray(p); pred=p>=threshold
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    tp=int((y&pred).sum()); fn=int((y&~pred).sum()); tn=int((~y&~pred).sum()); fp=int((~y&pred).sum())
    order=np.argsort(-p,kind='stable'); yy=y[order]; pp=p[order]
    ends=np.r_[np.flatnonzero(np.diff(pp)),len(y)-1]
    hits=np.cumsum(yy)[ends]; precision=hits/(ends+1)
    ap=float(np.sum(np.diff(np.r_[0,hits])*precision)/max(int(y.sum()),1))
    recalls=[num/den for num,den in ((tp,tp+fn),(tn,tn+fp)) if den]
    return dict(f1=2*tp/max(2*tp+fp+fn,1), average_precision=ap, balanced_accuracy=float(np.mean(recalls)),
        confusion_matrix=[[tn,fp],[fn,tp]], class_counts=[tn+fp,tp+fn])


def check_metrics(y,p,saved,threshold=.5):
    largest=0.
    for field,value in independent_metrics(y,p,threshold).items():
        if isinstance(value,list): assert value==saved[field],field
        else:
            delta=abs(value-saved[field]); assert delta<1e-12,(field,delta); largest=max(largest,delta)
    return largest


def audit(spec,config):
    parent_check(spec); root=Path(spec['output']); current=identity(config); rows=[]; largest=0.
    expected={}
    for split in SPLITS:
        entries=[]
        for fi,flow in enumerate(spec['flows']):
            with np.load(Path(spec['parent_output'])/flow['name']/split/'metadata.npz') as z:
                entries.append(np.column_stack((z['labels'],np.full(len(z['labels']),fi),z['head_component'],z['scale_id'])))
        expected[split]=np.concatenate(entries)
    for method,candidate,seed in training.plan(spec):
        folder=training.run_folder(spec,method,candidate,seed); path=folder/'result.json'
        r=json.loads(path.read_text()); lock=json.loads((folder/'selection.lock.json').read_text())
        assert r['identity']['commit']==current['commit'] and r['identity']['config_sha256']==current['config_sha256']
        assert r['identity']['source_sha256']==current['source_sha256']
        assert r['parameters']==spec['conv_parameters'] and r['device']==spec['required_device']
        assert pipeline.sha(folder/'predictions.npz')==r['predictions_sha256']
        assert pipeline.sha(folder/'selection.lock.json')==r['selection_lock_sha256']
        assert not lock['test_loaded'] and lock['selected_at_utc']<r['finished_at_utc']
        assert lock['fixed_threshold']==r['threshold']==.5 and lock['selected_epoch']==r['selected_epoch']
        best=max(r['history'],key=lambda h:(h['validation_f1'],h['validation_average_precision']))
        assert best['epoch']==r['selected_epoch'] and best['validation_f1']==r['validation']['pooled']['f1']
        rng=np.random.default_rng(seed)
        for i,h in enumerate(r['history'],1):
            assert h['epoch']==i and h['training_examples']==27000
            assert hashlib.sha256(rng.permutation(27000).astype('<i8').tobytes()).hexdigest()==h['permutation_sha256']
        with np.load(folder/'predictions.npz') as z:
            for split,field in (('train','training'),('validation','validation'),('test','test')):
                y,p=z[split+'_labels'],z[split+'_probability']; flow=z[split+'_flow_index']
                head,scale=z[split+'_head_component'],z[split+'_scale_id']
                assert np.array_equal(np.column_stack((y,flow,head,scale)),expected[split])
                if split=='validation':
                    order=np.argsort(p,kind='stable')[::-1]; pp=p[order]; yy=y[order]
                    ends=np.r_[np.flatnonzero(np.diff(pp)),len(y)-1]; hits=np.cumsum(yy)[ends]
                    precision=hits/(ends+1); recall=hits/y.sum()
                    score=2*precision*recall/np.maximum(precision+recall,1e-15)
                    threshold=float(pp[ends][np.flatnonzero(score==score.max())[0]])
                    assert threshold==r['validation_selected_threshold']==lock['validation_selected_threshold']
                scopes=[(r[field],.5)]
                if split=='test': scopes.append((r['test_with_validation_selected_threshold'],r['validation_selected_threshold']))
                for metrics,t in scopes:
                    groups=[(y,p,metrics['pooled'])]
                    for fi,entry in enumerate(spec['flows']):
                        mask=flow==fi; name=entry['name']; groups.append((y[mask],p[mask],metrics['per_flow'][name]))
                        hy=[]; hp=[]
                        for hd in np.unique(head[mask]):
                            m=mask&(head==hd); assert len(np.unique(y[m]))==1
                            hy.append(y[m][0]); hp.append(p[m].mean())
                        assert len(hy)==metrics['per_head'][name]['head_count']
                        groups.append((hy,hp,metrics['per_head'][name]))
                    groups.extend((y[scale==s],p[scale==s],metrics['per_scale'][str(s)]) for s in np.unique(scale))
                    for yy,pp,saved in groups:largest=max(largest,check_metrics(yy,pp,saved,t))
        assert r['weights_written']==0 and not r['original_spatial_test_loaded']
        rows.append(dict(seed=seed,result_sha256=pipeline.sha(path),predictions_sha256=r['predictions_sha256']))
    assert not any(p.suffix in ('.pt','.pth','.ckpt') for p in root.rglob('*'))
    report=dict(status='PASS',identity=current,runs=rows,max_numeric_error=largest,
        parent_samples_and_all_grouped_metrics_checked=True,weights_written=0)
    pipeline.write(root/'independent_predictions_audit.json',report)
    return report


def merge(spec,config):
    report=audit(spec,config)
    original=training.identity
    try:
        training.identity=identity; training.merge(spec,config)
    finally: training.identity=original
    path=Path(spec['output'])/'summary.json'; summary=json.loads(path.read_text()); refs=parent_check(spec)
    summary.update(comparison=spec['comparison_scope'],historical_4_14_reference=refs['rows'],
        parameter_check=parameter_check(spec), test_scope='same_already_used_4.14_benchmark',timing=[])
    for method,candidate,seed in training.plan(spec):
        r=json.loads((training.run_folder(spec,method,candidate,seed)/'result.json').read_text())
        summary['timing'].append({k:r[k] for k in ('seed','selected_epoch','training_through_last_validation_seconds',
            'epoch_seconds_after_first_median','epoch_seconds_after_first_mean','seconds','data_loading_times')})
        summary['timing'][-1]['epochs']=len(r['history'])
    pipeline.write(path,summary)
    report['summary_sha256']=pipeline.sha(path); pipeline.write(path.parent/'independent_predictions_audit.json',report)


def runtime(spec,config,phase,state,code=None):
    root=Path(spec['output']); device='CPU'
    if phase=='train':
        probe=subprocess.run(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],capture_output=True,text=True)
        device=probe.stdout.strip()
    row=dict(time=datetime.now(timezone.utc).isoformat(),phase=phase,state=state,exit_code=code,device=device,**identity(config))
    pipeline.append_locked(root/'runtime_events.jsonl',json.dumps(row)+'\n')
    pipeline.append_locked('docs/ibex_run_registry.md','\n- Task4-c 4.18 runtime '+json.dumps(row)+'\n')


def submit(spec,config):
    root=Path(spec['output']).resolve();root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    pipeline.write(root/'config.frozen.json',spec); previous=None
    for phase in ('prepare','train','merge'):
        command=['sbatch','--parsable','--cpus-per-task=4','--mem=32G','--time=04:00:00' if phase=='train' else '--time=01:00:00',
            '--job-name=t4c418-'+phase,f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if previous: command += ['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        if phase=='train': command += ['--gres=gpu:1','--constraint=v100','--array=0-2%3']
        command += ['ibex_bash/task4c_conv_capacity_4p18.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,version=spec['version'],submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            command=command,dependency=previous,expected_device=spec['required_device'] if phase=='train' else 'CPU',**identity(config))
        pipeline.append_locked(root/'submissions.jsonl',json.dumps(row)+'\n')
        pipeline.append_locked('docs/ibex_run_registry.md','\n- Task4-c 4.18 SUBMITTED '+json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);previous=job


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','train','merge','audit','submit','runtime'))
    p.add_argument('--config',default=CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=load_spec(a.config)
    if a.phase=='train':train(spec,a.config,a.index)
    elif a.phase=='runtime':runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:{'prepare':prepare,'merge':merge,'audit':audit,'submit':submit}[a.phase](spec,a.config)


if __name__=='__main__':main()
