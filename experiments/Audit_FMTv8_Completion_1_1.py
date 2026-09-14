"""Reconcile V8 experiments, audit metrics, and package model-free evidence."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scheduler(root,task4):
    submissions=[json.loads(x) for x in (root/'submissions.jsonl').read_text().splitlines()]
    cols=['JobID','JobIDRaw','State','ExitCode','NodeList','Start','End','Elapsed','AllocTRES']
    text=subprocess.check_output(['sacct','-j',','.join(r['job_id'] for r in submissions),'--noheader','--parsable2','--format='+','.join(cols)],text=True)
    rows=[dict(zip(cols,line.split('|'))) for line in text.splitlines() if line.strip()]
    expected=set()
    for r in submissions:
        count=15 if r['phase']=='train' else 30 if r['phase'] in ('Task1','Task35') else None
        expected.update([r['job_id']+'_'+str(i) for i in range(count)] if count else [r['job_id']])
    actual={r['JobID']:r for r in rows if '.' not in r['JobID']}
    assert set(actual)==expected
    assert len(actual)==(17 if task4 else 63)
    events=[json.loads(x) for x in (root/'runtime_events.jsonl').read_text().splitlines()]
    durable=[json.loads(p.read_text()) for p in (root/'lifecycle_events').glob('*.json')]
    key=lambda items:collections.Counter(json.dumps(x,sort_keys=True) for x in items)
    assert key(events)==key(durable) and len(events)==2*len(actual)
    groups=collections.defaultdict(list)
    for event in events:groups[event['job']].append(event)
    for r in actual.values():
        assert (r['State'],r['ExitCode'])==('COMPLETED','0:0')
        pair=groups[r['JobIDRaw']]
        assert len(pair)==2 and {e['state'] for e in pair}=={'STARTED','ENDED'}
        assert all(e['host']==r['NodeList'] for e in pair)
        assert next(e for e in pair if e['state']=='ENDED')['exit_code']==0
        if pair[0]['phase'] in (('preflight','train') if task4 else ('smoke','Task35')):
            assert all('V100' in e['device'] for e in pair)
    return dict(status='PASS',processes=len(actual),events=len(events),scheduler_rows=rows,all_events_matched=True)


def task4_extra(root):
    audit=json.loads((root/'independent_audit.json').read_text());assert audit['status']=='PASS' and len(audit['runs'])==15
    assert max(r['max_metric_error'] for r in audit['runs'])==0
    comparisons=json.loads((root/'comparison.json').read_text());assert comparisons['conv_capacity_reference']['test_f1_mean']==.7499
    sources={
        'fmt_direction_233':Path('/ibex/user/zhanx0o/FMT_Task4C_PhysicalLength_20260914/outputs/mainExp_Task4C_PhysicalLength_4.14/runs'),
        'fmt_v5_direction_398':Path('/ibex/user/zhanx0o/FMT_Task4C_RetainDirection_20260914/outputs/Verify_Task4C_RetainDirection_4.16/variants/fmt_v5_4_14/runs')}
    reproduced=[]
    for encoder,source in sources.items():
        for seed in (96611,96612,96613):
            name=f'regularized_fmt_mlp_seed{seed}';new=root/'variants'/encoder/'runs'/name
            r=json.loads((new/'result.json').read_text());old=json.loads((source/name/'result.json').read_text())
            assert r['normalization']==old['normalization'] and r['selected_epoch']==old['selected_epoch']
            with np.load(new/'predictions.npz') as a,np.load(source/name/'predictions.npz') as b:
                for split in ('train','validation','test'):
                    assert np.array_equal(a[split+'_probability'],b[split+'_probability'])
            reproduced.append(dict(encoder=encoder,seed=seed,probability_max_error=0))
    (root/'reference_reproduction.json').write_text(json.dumps(dict(status='PASS',runs=reproduced),indent=2)+'\n')
    return dict(prediction_sets=15,reproduced_reference_runs=6)


def task135_extra(root):
    # The unchanged independent auditor recomputes F1/IoU/AP, checks all 90
    # shards and confirms Task5's 18/6/9 scale tuples and per-scale metrics.
    subprocess.run([sys.executable,'-m','experiments.Audit_Task1235_AngleFeatures_1_1','--root',str(root)],check=True)
    audit=json.loads((root/'independent_audit.json').read_text());assert audit['status']=='PASS' and audit['shards']==90
    spec=json.loads((root/'run_config.json').read_text());costs=[];label_checks=[]
    from experiments import Run_Task1235_AngleFeatures_1_1 as frozen
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            reference=frozen.records(spec,task,dataset,'confirmation')
            labels=frozen.labels(reference)
            for seed in spec[task.lower()]['seeds']:
                folder=root/'shards'/task/dataset/f'seed{seed}'
                with np.load(folder/'predictions.npz') as z:assert np.array_equal(z['labels'],labels)
                label_checks.append(dict(task=task,dataset=dataset,seed=seed,samples=len(labels)))
                timing=json.loads((folder/'timing.json').read_text())['events'];models=json.loads((folder/'frozen_models.json').read_text())
                for arm in spec[task.lower()]['arms']:
                    m=models[arm]
                    if task=='Task1':
                        event=timing[[i for i,e in enumerate(timing) if e['kind']=='clustering_fit'][spec[task.lower()]['arms'].index(arm)]]
                        cost=dict(clustering_seconds=event['seconds'],parameters=0,trainable_parameters=0)
                    else:
                        raw=next(e['seconds'] for e in timing if e['kind']=='raw_training' and e['arm']=='raw')
                        if arm=='fixed_task3_raw':
                            old=json.loads((root/'shards/Task3'/dataset/f'seed{seed}/timing.json').read_text())['events']
                            own=0.;raw=next(e['seconds'] for e in old if e['kind']=='raw_training' and e['arm']=='raw')
                        elif arm in ('raw','raw_wide'):
                            raw=next(e['seconds'] for e in timing if e['kind']=='raw_training' and e['arm']==arm);own=0.
                        else:own=next(e['seconds'] for e in timing if e['kind']=='residual_training' and e['arm']==arm)
                        cost=dict(raw_backbone_seconds=raw,residual_seconds=own,total_training_seconds=raw+own,
                            parameters=m['parameter_count'],trainable_parameters=m.get('trainable_parameter_count',m['parameter_count']))
                    costs.append(dict(task=task,dataset=dataset,seed=seed,arm=arm,**cost))
    for name,rows in [('training_costs.csv',costs)]:
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
    (root/'source_label_audit.json').write_text(json.dumps(dict(status='PASS',checks=label_checks),indent=2)+'\n')
    return dict(prediction_sets=90,rows=audit['rows'],source_labels_exact=True,training_cost_rows=len(costs))


def package(root):
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth')) and not any(root.rglob('*.ckpt'))
    included=[];archive=root/'metrics_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            if p.suffix not in ('.json','.jsonl','.csv','.out','.err','.sha256') and not p.name.endswith('predictions.npz'):continue
            z.add(p,arcname=str(p.relative_to(root)));included.append(str(p.relative_to(root)))
    manifest=dict(files=included,archive_sha256=sha(archive),model_weights_included=False)
    (root/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--task4',action='store_true');a=p.parse_args();root=Path(a.root)
    runtime=scheduler(root,a.task4);extra=task4_extra(root) if a.task4 else task135_extra(root)
    (root/'completion_audit.json').write_text(json.dumps(dict(**runtime,**extra,auditor_sha256=sha(__file__)),indent=2)+'\n')
    manifest=package(root);print(json.dumps(dict(status='PASS',processes=runtime['processes'],events=runtime['events'],files=len(manifest['files']),archive_sha256=manifest['archive_sha256'])))


if __name__=='__main__':main()
