"""Independent saved-prediction reconciliation and final search evidence export."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
from sklearn.metrics import f1_score, average_precision_score

from experiments import FMTv8_Search_2_1 as run


def verify_predictions(folder, final=False, expected_labels=None):
    path=folder/('final_result.json' if final else 'result.json')
    row=json.loads(path.read_text());name='test_predictions.npz' if final else 'validation_predictions.npz'
    expected=row['test_predictions_sha256'] if final else row['predictions_sha256']
    assert run.old.sha(folder/name)==expected
    with np.load(folder/name) as z:
        y=z['labels'];p=z['probability'];threshold=float(z['threshold'])
        if expected_labels is not None:assert np.array_equal(y,expected_labels)
        assert len(y)==len(p) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        target=row['test' if final else 'validation']
        assert abs(f1_score(y,p>=threshold)-target['f1'])<1e-12
        assert abs(average_precision_score(y,p)-target['average_precision'])<1e-12
        if final:
            for key,groups in row['per_group'].items():
                for value,metrics in groups.items():
                    mask=z[key]==int(value)
                    assert abs(f1_score(y[mask],p[mask]>=threshold)-metrics['f1'])<1e-12
    return row


def merge(spec,config):
    root=Path(spec['output']);lock=json.loads((root/'selection.lock.json').read_text())
    ranking,search_rows=run.ranked(spec,['search','refine']);assert len(search_rows)==1239
    assert ranking==lock['ranking'] and ranking[0]==lock['selected']
    flat=[]
    for phase in ('search','refine'):
        for task,dataset in run.units(spec):
            labels=run.read_cache(spec,task,dataset,'validation')['labels']
            for p in (root/phase/task/dataset).glob('*/seed*/result.json'):
                row=verify_predictions(p.parent,expected_labels=labels)
                flat.append(dict(phase=phase,task=task,dataset=dataset,pool=row['pool'],profile=row['profile'],seed=row['seed'],
                    feature_dimensions=row['feature_dimensions'],parameters=row['parameters'],trainable_parameters=row['trainable_parameters'],
                    validation_f1=row['validation']['f1'],validation_average_precision=row['validation']['average_precision'],
                    training_seconds=row['seconds'],raw_training_seconds=row['raw_training_seconds'],raw_wide_selection_seconds=row['raw_wide_selection_seconds']))
    assert len(flat)==1239
    run.old.write_csv(root/'validation_search_results.csv',flat)
    chosen=list(dict.fromkeys([('p00','h0'),(lock['selected']['pool'],lock['selected']['profile'])]));finals=[]
    for task,dataset in run.units(spec):
        test,evidence=run.read_source(spec,task,dataset,'test',allow_test=True)
        for seed in spec['final_task4_seeds'] if task=='Task4C' else spec['final_seeds']:
            for pool,profile in chosen:
                folder=run.run_directory(spec,'final',task,dataset,pool,profile,seed)
                row=verify_predictions(folder,True,test['labels'])
                assert row['selection_lock_sha256']==run.old.sha(root/'selection.lock.json')
                assert row['test_source_evidence']==evidence
                assert row['models_frozen_before_test']>__import__('datetime').datetime.fromisoformat(lock['selected_at_utc']).timestamp()
                finals.append(row)
    assert len(finals)==63*len(chosen)
    table=[];per_flow=[];seed_rows=[]
    for pool,profile in chosen:
        for task in ('Task3','Task5','Task4C'):
            part=[r for r in finals if (r['pool'],r['profile'],r['task'])==(pool,profile,task)]
            seeds=spec['final_task4_seeds'] if task=='Task4C' else spec['final_seeds']
            values=[]
            for seed in seeds:
                rows=[r for r in part if r['seed']==seed]
                f1=float(np.mean([r['test']['f1'] for r in rows]));values.append(f1)
                seed_rows.append(dict(pool=pool,profile=profile,task=task,seed=seed,f1=f1))
            table.append(dict(pool=pool,profile=profile,task=task,f1_mean=float(np.mean(values)),f1_std=float(np.std(values,ddof=1)),
                input_dimensions=part[0]['feature_dimensions'],parameters=part[0]['parameters'],trainable_parameters=part[0]['trainable_parameters'],
                mean_training_seconds=float(np.mean([r['seconds']+r['raw_training_seconds'] for r in part])),
                mean_residual_or_classifier_seconds=float(np.mean([r['seconds'] for r in part]))))
            for dataset in sorted({r['dataset'] for r in part}):
                rr=[r for r in part if r['dataset']==dataset]
                per_flow.append(dict(pool=pool,profile=profile,task=task,dataset=dataset,
                    f1_mean=float(np.mean([r['test']['f1'] for r in rr])),f1_std=float(np.std([r['test']['f1'] for r in rr],ddof=1))))
    composite=[]
    for pool,profile in chosen:
        three=[r for r in table if (r['pool'],r['profile'])==(pool,profile)]
        composite.append(dict(pool=pool,profile=profile,f1_mean=float(np.mean([r['f1_mean'] for r in three]))))
    baseline=composite[0]['f1_mean'];selected=composite[-1]['f1_mean']
    # Count common backbones once in search cost; report it separately from deployment training cost.
    search_raw_seconds=0.
    for path in (root/'cache').glob('*/*/temporary_backbones/backbones.json'):
        search_raw_seconds+=sum(v['seconds'] for v in json.loads(path.read_text())['results'].values())
    for directory in [root/'cache',root/'final_backbones',root/'search',root/'refine',root/'final']:
        run.old.cleanup_checkpoints(directory)
    assert not [p for ext in ('*.pt','*.pth','*.ckpt') for p in root.rglob(ext)]
    run.old.write_csv(root/'summary.csv',table);run.old.write_csv(root/'per_dataset.csv',per_flow);run.old.write_csv(root/'per_seed_macro.csv',seed_rows)
    run.write(root/'summary.json',dict(version=spec['version'],selected=lock['selected'],rows=table,composite=composite,
        relative_test_gain=selected/baseline-1,absolute_test_gain=selected-baseline,
        target_relative_gain=[.15,.20],target_minimum_reached=selected/baseline-1>=.15,
        search_classifier_seconds=sum(r['training_seconds'] for r in flat),search_backbone_seconds=search_raw_seconds,
        final_test_runs=len(finals),validation_fits=len(flat),identity=run.identity(config)))
    run.write(root/'independent_audit.json',dict(status='PASS',validation_fits=len(flat),final_runs=len(finals),
        labels_exact=True,all_predictions_recomputed=True,selection_validation_only=True,no_weights=True))
    print(json.dumps(dict(status='PASS',relative_test_gain=selected/baseline-1,rows=table)),flush=True)


def complete(root):
    audit=json.loads((root/'independent_audit.json').read_text());assert audit['status']=='PASS'
    submissions=[json.loads(x) for x in (root/'submissions.jsonl').read_text().splitlines()]
    counts={'prepare':21,'search':105,'refine':63,'final':63}
    expected=set()
    for r in submissions:
        n=counts.get(r['phase']);expected.update([r['job_id']+'_'+str(i) for i in range(n)] if n else [r['job_id']])
    columns=['JobID','JobIDRaw','State','ExitCode','NodeList','Start','End','Elapsed','AllocTRES']
    output=subprocess.check_output(['sacct','-j',','.join(r['job_id'] for r in submissions),'--noheader','--parsable2','--format='+','.join(columns)],text=True)
    rows=[dict(zip(columns,line.split('|'))) for line in output.splitlines() if line.strip()]
    actual={r['JobID']:r for r in rows if '.' not in r['JobID']};assert set(actual)==expected
    events=[json.loads(x) for x in (root/'runtime_events.jsonl').read_text().splitlines()]
    durable=[json.loads(p.read_text()) for p in (root/'lifecycle_events').glob('*.json')]
    encode=lambda rr:collections.Counter(json.dumps(r,sort_keys=True) for r in rr)
    assert encode(events)==encode(durable) and len(events)==2*len(expected)
    for r in actual.values():
        pair=[e for e in events if e['job']==r['JobIDRaw']]
        assert len(pair)==2 and {e['state'] for e in pair}=={'STARTED','ENDED'}
        assert r['State']=='COMPLETED' and r['ExitCode']=='0:0'
        assert all(e['host']==r['NodeList'] for e in pair)
        assert next(e for e in pair if e['state']=='ENDED')['exit_code']==0
    run.write(root/'completion_audit.json',dict(status='PASS',processes=len(expected),events=len(events),scheduler_rows=rows,
        auditor_sha256=run.old.sha(__file__),independent_audit_sha256=run.old.sha(root/'independent_audit.json')))
    archive=root/'metrics_evidence.tar.gz';included=[]
    with tarfile.open(archive,'w:gz') as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink() or p.name=='evidence_manifest.json':continue
            if p.suffix not in ('.json','.jsonl','.csv','.out','.err','.sha256') and not p.name.endswith('predictions.npz'):continue
            z.add(p,arcname=str(p.relative_to(root)));included.append(str(p.relative_to(root)))
    run.write(root/'evidence_manifest.json',dict(files=included,archive_sha256=run.old.sha(archive),model_weights_included=False))
    print(json.dumps(dict(status='PASS',processes=len(expected),events=len(events),evidence_files=len(included))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();complete(Path(args.root))
