"""Recompute complete Task6 MoE scalar summaries without reading trajectories."""
import csv
import hashlib
import json
from pathlib import Path
import argparse
from collections import Counter

import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--root',default='outputs/Verify_Task6_FMTGeometryMoE_1.1')
parser.add_argument('--config',default='config/Verify_Task6_FMTGeometryMoE_1.1.json')
args=parser.parse_args()
root=Path(args.root).resolve()
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
config=read(root/'config.frozen.json')
assert sha(root/'config.frozen.json')==sha(args.config)
assert sha(root/'selection.json')==sha(root/'selection.before_test.json')
selection=read(root/'selection.json')
audit=read(root/'final_audit.json')
status=read(root/'status_summary.json')
assert audit['audit_passed'] and not audit['failures']
assert status['completed_models']==dict(fit_check=18,screen=24,refine=108,final=189)
assert not status['incomplete_models']
records=[]
with (root/'metrics.csv').open(encoding='utf-8') as f:
    for row in csv.DictReader(f):
        row['seed']=int(row['seed'])
        row['scale_id']=int(row['scale_id'])
        row['position_rmse_r']=float(row['position_rmse_r'])
        records.append(row)
summary=[]
for row in audit['summary']:
    chosen=[r for r in records if r['family']==row['family'] and r['arm']==row['arm']
            and r['role']==row['role'] and r['scale_id']==-1]
    assert {(r['dataset'],r['seed']) for r in chosen}=={(d,s) for d in config['datasets'] for s in config['seeds']}
    assert len(chosen)==27 and row['complete'] and row['evaluated']==27
    value=float(np.mean([r['position_rmse_r'] for r in chosen]))
    np.testing.assert_allclose(value,row['position_rmse_r'],rtol=0,atol=1e-15)
    seed_means=[float(np.mean([r['position_rmse_r'] for r in chosen if r['seed']==s])) for s in config['seeds']]
    summary.append(dict(row,seed_macro_means=seed_means,seed_macro_std=float(np.std(seed_means,ddof=1))))
flowmeans={}
for d in config['datasets']:
    flowmeans[d]={}
    for row in summary:
        chosen=[r['position_rmse_r'] for r in records if r['dataset']==d and r['family']==row['family']
                and r['arm']==row['arm'] and r['role']==row['role'] and r['scale_id']==-1]
        assert len(chosen)==3
        flowmeans[d]['/'.join((row['family'],row['arm'],row['role']))]=float(np.mean(chosen))
diagnostics=[]
for family in config['families']:
    fits=[read(root/'final'/d/str(s)/family/'fmt_moe'/'fit.json') for d in config['datasets'] for s in config['seeds']]
    assert all(f['candidate']==selection['families'][family]['candidate'] for f in fits)
    diagnostics.append(dict(family=family,candidate=selection['families'][family]['candidate'],
        validation_rmse_r=float(np.mean([f['validation_rmse_r'] for f in fits])),
        gate_mean=float(np.mean([f['validation_gates']['a_mean'] for f in fits])),
        gate_range=[min(f['validation_gates']['a_mean'] for f in fits),max(f['validation_gates']['a_mean'] for f in fits)],
        branch={key:float(np.mean([f['validation_branch_diagnostics'][key] for f in fits])) for key in fits[0]['validation_branch_diagnostics']},
        selected_step_range=[min(f['selected_step'] for f in fits),max(f['selected_step'] for f in fits)],
        train_rmse_r=float(np.mean([f['train_rmse_r'] for f in fits]))))
runtime=[json.loads(line) for line in (root/'runtime_events.jsonl').read_text().splitlines()]
assert all(r['exit_code']==0 for r in runtime if r['state']=='ENDED')
starts={r['job_id']:r for r in runtime if r['state']=='STARTED'}
ends={r['job_id']:r for r in runtime if r['state']=='ENDED'}
assert set(starts)==set(ends)
comparisons=[]
for family in config['families']:
    for arm,f in [('geometry_only',family),('geometry_moe',family),('raw_frozen','raw_frozen')]:
        for role in ('test','unseen_scale'):
            ours=next(r['position_rmse_r'] for r in summary if (r['family'],r['arm'],r['role'])==(family,'fmt_moe',role))
            other=next(r['position_rmse_r'] for r in summary if (r['family'],r['arm'],r['role'])==(f,arm,role))
            wins=sum(flowmeans[d][f'{family}/fmt_moe/{role}']<flowmeans[d][f'{f}/{arm}/{role}'] for d in config['datasets'])
            comparisons.append(dict(family=family,comparison=arm,role=role,relative_change_percent=100*(ours/other-1),flow_wins=wins))
out=dict(config_sha256=sha(root/'config.frozen.json'),selection_sha256=sha(root/'selection.json'),
    code_commit=audit['provenance']['git_commit'],summary=summary,per_flow=flowmeans,diagnostics=diagnostics,
    comparisons=comparisons,completed_jobs=len(starts),device_jobs=dict(Counter(r['device'] for r in starts.values())),
    first_start=min(r['time'] for r in runtime),last_end=max(r['time'] for r in runtime),status_counts=status['completed_models'])
(root/'analysis_verified.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('per_flow','summary','diagnostics')},indent=2))
