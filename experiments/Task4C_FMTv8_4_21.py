"""Three-seed FMT v8.1 comparison on the frozen Task4-c 4.14 benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from FMT_Utils.fmt_v8 import from_task4c_cache, pool_neighbor_features
from FMT_Utils.Task4C_Encoders_4_15 import EncoderLinePooling
from FMT_Utils.Task4C_LinePooling_4_3 import LearnedLinePooling
from experiments import Task4C_FMTv6v7_4_20 as base
from experiments import Task4C_RetainDirection_4_16 as parent
from experiments.Record_Task1235_AngleFeatures_1_1 import append_record

DEFAULT_CONFIG = 'config/mainExp_Task4C_FMTv8_4.21.json'
pipeline, training, identity = base.pipeline, base.training, base.identity
variant_spec, folder, plan, check_parent = base.variant_spec, base.folder, base.plan, base.check_parent
SPLITS = base.SPLITS
WIDTHS = {'fmt_161':161,'fmt_direction_233':233,'fmt_v5_326':326,'fmt_v5_direction_398':398,'fmt_v8_100':100}


def make_model(encoder,dropout):
    return LearnedLinePooling(dropout) if encoder=='fmt_direction_233' else EncoderLinePooling(WIDTHS[encoder],dropout)


def select_tokens(original,v5,encoder):
    if encoder=='fmt_161':return torch.cat((original[...,:161],original[...,-1:]),-1)
    if encoder=='fmt_direction_233':return original
    if encoder=='fmt_v5_326':return torch.cat((original[...,:161],v5[...,233:398],original[...,-1:]),-1)
    if encoder=='fmt_v5_direction_398':return v5
    if encoder=='fmt_v8_100':return from_task4c_cache(original)
    raise ValueError(encoder)


def load_spec(config):
    spec = json.loads(Path(config).read_text())
    frozen = json.loads(Path(spec['parent_config']).read_text())
    assert pipeline.sha(spec['parent_config']) == spec['parent_config_sha256']
    assert pipeline.sha(frozen['base_config']) == frozen['base_config_sha256']
    assert spec['encoders'] == list(WIDTHS)
    for key in ('training','candidates','expected_training_samples'):
        assert spec[key] == frozen[key], key
    spec['flows'] = json.loads(Path(frozen['base_config']).read_text())['flows']
    return spec


def preflight(spec, config):
    root, source = Path(spec['output']), Path(spec['parent_output'])
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    checked = check_parent(spec, full=True)
    test = torch.zeros((2,138));test[:, :6] = torch.tensor([-9.,9.,-8.,7.,-6.,5.])
    actual = pool_neighbor_features(test)
    assert torch.equal(actual[:, :4], test[:, :4])
    assert torch.equal(actual[:, 4], test.mean(-1))
    v5root=Path(spec['v5_reference']['output'])/'variants/fmt_v5_4_14'
    assert pipeline.sha(Path(spec['v5_reference']['output'])/'summary.json')==spec['v5_reference']['summary_sha256']
    v5manifest=json.loads((v5root/'encoding.json').read_text())
    manifests={}
    for encoder,width in WIDTHS.items():
        manifests[encoder]=dict(identity=identity(config),encoder=encoder,line_width=width,splits={})
        Path(variant_spec(spec,encoder)['output']).mkdir(parents=True,exist_ok=False)
    fixtures = []
    for flow in spec['flows']:
        for split in SPLITS:
            key = flow['name']+'/'+split;old = np.load(source/key/'fmt.npy',mmap_mode='r')
            assert pipeline.sha(v5root/key/'fmt.npy')==v5manifest['splits'][key]['files']['fmt.npy']
            v5=np.load(v5root/key/'fmt.npy',mmap_mode='r');targets={}
            for encoder,width in WIDTHS.items():
                dest=Path(variant_spec(spec,encoder)['output'])/key;dest.mkdir(parents=True)
                (dest/'metadata.npz').symlink_to((source/'physical'/key/'metadata.npz').resolve())
                targets[encoder]=np.lib.format.open_memmap(dest/'fmt.npy',mode='w+',dtype='float32',shape=(*old.shape[:2],width+1))
            for start in range(0,len(old),256):
                sl=slice(start,start+256);x=np.array(old[sl]);new=from_task4c_cache(torch.from_numpy(x)).numpy()
                vv=np.array(v5[sl]);assert np.array_equal(x[...,:233],vv[...,:233]) and np.array_equal(x[...,-1],vv[...,-1])
                nb=x[...,23:161];order=np.argsort(-np.abs(nb),axis=-1,kind='stable')[...,:4]
                expected=np.concatenate((x[...,:23],x[...,161:233],np.take_along_axis(nb,order,-1),
                    nb.mean(-1,keepdims=True),x[...,-1:]),-1)
                assert np.array_equal(new[...,:99],expected[...,:99]) and np.array_equal(new[...,-1],x[...,-1])
                np.testing.assert_allclose(new[...,99],expected[...,99],atol=1e-6,rtol=1e-6)
                assert not np.any(new[x[...,-1]==0] != 0)
                for encoder,target in targets.items():target[sl]=select_tokens(torch.from_numpy(x),torch.from_numpy(vv),encoder).numpy()
            for encoder,target in targets.items():
                target.flush()
                if split=='train':
                    token=torch.as_tensor(np.array(target[:16]),device='cuda');torch.manual_seed(96611)
                    model=make_model(encoder,.15).cuda();loss=model(token).square().mean();loss.backward()
                    assert sum(p.numel() for p in model.parameters())==88514+(WIDTHS[encoder]-233)*128
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    fixtures.append(dict(flow=flow['name'],encoder=encoder,samples=16,finite_gradients=True))
                dest=Path(variant_spec(spec,encoder)['output'])/key
                manifests[encoder]['splits'][key]=dict(samples=len(old),files={n:pipeline.sha(dest/n) for n in ('fmt.npy','metadata.npz')})
            targets.clear()
    for encoder,manifest in manifests.items():pipeline.write(Path(variant_spec(spec,encoder)['output'])/'encoding.json',manifest)
    pipeline.write(root/'preflight.json',dict(status='PASS',identity=identity(config),parent_files=checked,
        numpy_pooling_audit_all_rows=True,first95_and_mask_exact=True,signed_ties_checked=True,
        gradient_checks_training_only=fixtures,parameters=71490))


def train(spec,config,index):
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    encoder,seed=plan(spec)[index];local=variant_spec(spec,encoder)
    training.identity=identity;training.make_model=lambda method,candidate:make_model(encoder,candidate['dropout'])
    training.train(local,config,spec['training']['seeds'].index(seed))
    p=folder(spec,encoder,seed)/'result.json';r=json.loads(p.read_text())
    r.update(encoder=encoder,encoder_version='8.1' if encoder=='fmt_v8_100' else 'frozen_blocks',feature_blocks=[encoder],input_dimension=WIDTHS[encoder],
        actual_feature_blocks=spec['encoder_definitions'][encoder],
        no_kinematic_inputs=True,test_scope=spec['test_scope'])
    pipeline.write(p,r)


def timing(records):
    return dict(training_minutes_mean=float(np.mean([r['history'][-1]['seconds']/60 for r in records])),
        epoch_seconds_median_mean=float(np.mean([np.median(np.diff([h['seconds'] for h in r['history']])) for r in records])),
        scope='data_loading_and_training_through_last_validation; excludes_geometry_encoding_and_final_test',
        per_seed=[dict(seed=r['seed'],selected_epoch=r['selected_epoch'],epochs=len(r['history']),
            through_last_validation_seconds=r['history'][-1]['seconds'],device=r['device']) for r in records])


def merge(spec,config):
    parent.merge(spec,config)
    from experiments import Audit_Task4C_Encoders_4_15 as auditor
    auditor.run=sys.modules[__name__];auditor.audit(spec,config)
    root=Path(spec['output']);summary=json.loads((root/'summary.json').read_text());comparison=[]
    for row in summary['rows']:
        encoder=row['encoder'];records=[json.loads((folder(spec,encoder,s)/'result.json').read_text()) for s in spec['training']['seeds']]
        comparison.append(dict(encoder=encoder,input_dimension=WIDTHS[encoder],origin='new_three_seed_runs',metrics=row,timing=timing(records)))
    pipeline.write(root/'comparison.json',dict(version=spec['version'],rows=comparison,
        conv_capacity_reference=spec['conv_capacity_reference'],identity=identity(config),
        prediction_audit_sha256=pipeline.sha(root/'independent_audit.json'),all_five_encoders_retrained=True))
    print(json.dumps(comparison),flush=True)


def runtime(spec,config,phase,state,code=None):
    root=Path(spec['output']);events=root/'lifecycle_events';events.mkdir(parents=True,exist_ok=True)
    row=dict(version=spec['version'],phase=phase,state=state,exit_code=code,
        time=datetime.now(timezone.utc).isoformat(),device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',**identity(config))
    pipeline.write(events/f"{row['job']}_{state}_{time.time_ns()}.json",row)
    append_record(root/'runtime_events.jsonl',row,'Task4C 4.21 runtime');append_record('docs/ibex_run_registry.md',row,'Task4C 4.21 runtime')


def submit(spec,config,phase,dependency=None):
    root=Path(spec['output']);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True);gpu=phase in ('preflight','train')
    cmd=['sbatch','--parsable','--cpus-per-task=4','--mem=32G','--time=01:00:00','--job-name=t4c421-'+phase,
        '--output='+str(logs/(phase+'.%A_%a.out')),'--error='+str(logs/(phase+'.%A_%a.err'))]
    if gpu:cmd+=['--gres=gpu:1','--constraint=v100']
    if phase=='train':cmd+=['--array=0-14%6']
    if dependency:cmd+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    cmd+=['ibex_bash/task4c_fmt_v8_4p21.sh',phase,config]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    row=dict(version=spec['version'],phase=phase,job_id=job,command=cmd,
        submitted_at_utc=datetime.now(timezone.utc).isoformat(),config=config,expected_device='V100' if gpu else 'CPU',**identity(config))
    append_record(root/'submissions.jsonl',row,'Task4C 4.21 submitted');append_record('docs/ibex_run_registry.md',row,'Task4C 4.21 submitted')
    print(json.dumps({k:v for k,v in row.items() if k!='source_sha256'}),flush=True);return job


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','train','merge','runtime','submit'])
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--submit-phase');p.add_argument('--dependency');p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();s=load_spec(a.config)
    if a.phase=='runtime':runtime(s,a.config,a.runtime_phase,a.state,a.exit_code)
    elif a.phase=='submit':submit(s,a.config,a.submit_phase,a.dependency)
    elif a.phase=='train':train(s,a.config,a.index)
    else:{'preflight':preflight,'merge':merge}[a.phase](s,a.config)


if __name__=='__main__':main()
