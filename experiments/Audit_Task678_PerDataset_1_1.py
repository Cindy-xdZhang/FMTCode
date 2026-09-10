"""Parallel independent replay of unchanged metric definitions, then ordered collection."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import numpy as np
from FMT_Utils.FlowMapData_3D import sha256,write_json
from experiments.Build_Task678_FlowMap_1_1 import provenance
from experiments.Submit_Task678_DirectFMTFit_1_1 import event


def shard(spec,config,root,index,kind,copy_inputs=False):
    dataset=spec['datasets'][index]
    destination=root/'audit_shards_fast'/dataset
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('runs','cache','build'):
        source=root/name
        if not source.exists():continue
        if copy_inputs:
            shutil.copytree(source,destination/name)
        else:
            (destination/name).symlink_to(source.resolve(),target_is_directory=True)
    if (root/'data_audit.json').exists():shutil.copy2(root/'data_audit.json',destination/'data_audit.json')
    local=dict(spec,output_root=str(destination),datasets=[dataset],families={dataset:spec['families'][dataset]})
    if kind=='han':
        from experiments.Audit_Task678_HanFast_1_1 import audit
    else:
        from experiments.Audit_Task678_VectorFast_1_1 import audit
    audit(local,config)
    record=json.loads((destination/'independent_audit.json').read_text())
    record.update(audit_subset=[dataset],execution='Original validation rules; all Euclidean pairs computed with scipy.pdist; one dataset per CPU allocation')
    write_json(destination/'independent_audit.json',record)


def collect(spec,config,root,destination=None):
    destination=root if destination is None else Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    assert not (destination/'independent_audit.json').exists(), 'Never replace an earlier canonical audit'
    rows=[];evidence=[];columns=None
    per_dataset=21+sum(len(spec['seeds'])*(30 if arm in ('fmt_all','raw_positions','vector_fmt6') else 21) for arm in spec['neural_arms'])
    for dataset in spec['datasets']:
        part=root/'audit_shards_fast'/dataset
        report=json.loads((part/'independent_audit.json').read_text())
        assert report['status']=='PASS' and report['audit_subset']==[dataset]
        assert report['config_sha256']==sha256(config) and report['metric_records']==per_dataset
        assert report['metrics_sha256']==sha256(part/'metrics.csv')
        with (part/'metrics.csv').open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f)
            if columns is None:columns=reader.fieldnames
            assert reader.fieldnames==columns
            values=list(reader)
        assert len(values)==per_dataset and all(r['dataset']==dataset for r in values)
        rows.extend(values)
        evidence.append(dict(dataset=dataset,audit_sha256=sha256(part/'independent_audit.json'),metrics_sha256=report['metrics_sha256'],metric_records=len(values)))
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth')) and not any(root.rglob('*.ckpt'))
    with (destination/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in ('arm','task','role','variant')),[]).append(row)
    summaries=[]
    for (arm,task,role,variant),values in groups.items():
        pd={d:float(np.mean([float(r['position_nrmse']) for r in values if r['dataset']==d])) for d in spec['datasets']}
        pf={family:float(np.mean([pd[d] for d in spec['datasets'] if spec['families'][d]==family])) for family in set(spec['families'].values())}
        ps={str(seed):float(np.mean([float(r['position_nrmse']) for r in values if int(r['seed'])==seed])) for seed in sorted({int(r['seed']) for r in values})}
        summaries.append(dict(arm=arm,task=task,role=role,variant=variant,per_dataset=pd,per_family=pf,
            dataset_macro_position_nrmse=float(np.mean(list(pd.values()))),family_macro_position_nrmse=float(np.mean(list(pf.values()))),
            seed_macro=ps,seed_macro_std=float(np.std(list(ps.values()),ddof=1)) if len(ps)>1 else None))
    training_commits=sorted({json.loads(p.read_text())['git_commit'] for p in (root/'runs').glob('*/*/*/completed.json')})
    write_json(destination/'summary.json',{**provenance(config),'status':'PASS','summary':summaries})
    write_json(destination/'independent_audit.json',{**provenance(config),'status':'PASS','metric_records':len(rows),
        'checkpoint_files':0,'metrics_sha256':sha256(destination/'metrics.csv'),'summary_sha256':sha256(destination/'summary.json'),
        'metric_implementation_sha256':sha256('FMT_Utils/FlowMapAuditMetrics_3D.py'),'training_git_commits':training_commits,'shard_audits':evidence,
        'execution':'Independent equivalent metric replay in parallel by dataset, followed by ordered collection; no sampling or metric-definition changes'})
    print('COLLECTED AUDIT PASS',len(rows),'records',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    p.add_argument('--kind',choices=['han','vector'],required=True);p.add_argument('--index',type=int)
    p.add_argument('--collect',action='store_true');p.add_argument('--copy-inputs',action='store_true');p.add_argument('--collect-output')
    p.add_argument('--event',choices=['RUNNING','COMPLETED','FAILED']);p.add_argument('--phase');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text());root=Path(a.root).resolve()
    if a.event:
        spec['output_root']=str(root);event(spec,a.config,a.phase,a.event,a.exit_code)
    elif a.collect:collect(spec,a.config,root,a.collect_output)
    else:shard(spec,a.config,root,a.index,a.kind,a.copy_inputs)


if __name__=='__main__':main()
