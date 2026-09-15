"""Read-only independent prediction/source/scheduler audit for P35 search 3.1."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


OLD_COMMIT='0d77e3e57957393d82008a5a0ce8befcf45134a2'
NEW_COMMIT='c61434ac65f71a595617606f978a8516d815e288'
OLD_CONFIG='6a87037d3bd5714fd3217c2c564d99c7a6bfa206307b2764477312de497ba82c'
NEW_CONFIG='72ec704f6c9b48cdcb4a0bc536628aa6d925e3f9a96d9f73ba091447285350f0'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def labels_sha(labels):
    assert np.isin(labels,[0,1]).all()
    return hashlib.sha256(np.asarray(labels,dtype=np.uint8).tobytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def source_check(root):
    """Check original source bytes and prepared labels afresh, without training imports."""
    result={};verified={}
    def check(path,expected):
        name=str(path)
        if name not in verified:verified[name]=sha(path)
        assert verified[name]==expected,(name,verified[name],expected)
    for manifest in sorted((root/'cache').glob('*/*/manifest.json')):
        m=json.loads(manifest.read_text());assert m['status']=='PASS' and m['test_read'] is False
        task,dataset=manifest.parent.relative_to(root/'cache').parts;entry={}
        for role in ('train','validation'):
            for row in m['evidence'][role]:check(row['path'],row['sha256'])
            for name,expected in m['files'][role].items():check(manifest.parent/role/name,expected)
            folder=manifest.parent/role;labels=np.load(folder/'labels.npy')
            r=dict(samples=len(labels),positive=int(labels.sum()),labels_sha256=labels_sha(labels))
            if task=='Task4C':
                mask=np.load(folder/'mask.npy');counts=np.load(folder/'counts.npy')
                assert np.array_equal(mask[...,0]>.5,np.arange(mask.shape[1])[None]<counts[:,None])
                r['valid_tokens']=int(counts.sum())
                r['group_sha256']={key:hashlib.sha256(np.ascontiguousarray(np.load(folder/(key+'.npy'))).tobytes()).hexdigest()
                                  for key in ('flow_index','head_component','scale_id','instance')}
            else:r['valid_tokens']=len(labels)
            entry[role]=r
        result[task+'/'+dataset]=entry
    assert len(result)==21
    baseline_root=Path('/ibex/user/zhanx0o/FMT_V8Search_Deterministic_20260915/outputs/Ablation_FMTv8_Search_2.2')
    for name in result:
        task,dataset=name.split('/');seed=96611 if task=='Task4C' else 40
        previous=baseline_root/'search'/task/dataset/'p35_h0'/f'seed{seed}'/'validation_predictions.npz'
        control=json.loads((root/'control_checks'/f'{task}_{dataset}.json').read_text())
        check(previous,control['historical_predictions_sha256'])
    report=dict(status='PASS',units=result,verified_files=verified,test_read=False)
    write(root/'independent_source_check.json',report)
    return report


def audit(root):
    source_path=root/'independent_source_check.json';source=json.loads(source_path.read_text())
    assert source['status']=='PASS' and source['test_read'] is False
    report=json.loads((root/'summary.json').read_text());assert report['status']=='PASS' and report['test_read'] is False
    replacement=json.loads((root/'task4_scheduling_replacement.json').read_text())
    assert replacement['original_task4_fits_started']==0 and replacement['new_candidate_count']==20
    cancelled_jobs=set(replacement['old_jobs'])
    assert cancelled_jobs=={'51906472_'+str(i) for i in range(80,84)}
    records=[];groups=collections.defaultdict(list);control=[]
    for phase in ('controls','search'):
        for path in sorted((root/phase).glob('*/*/*/result.json')):
            task,dataset,candidate=path.parent.relative_to(root/phase).parts
            r=json.loads(path.read_text());expected=source['units'][task+'/'+dataset]
            assert (r['task'],r['dataset'])==(task,dataset)
            assert candidate==(r['pool']+'_'+r['profile'] if phase=='controls' else r['pool'])
            assert r['test_read'] is False and r['seed']==(96611 if task=='Task4C' else 40)
            commit,config=(OLD_COMMIT,OLD_CONFIG) if phase=='controls' else (NEW_COMMIT,NEW_CONFIG)
            assert r['identity']['commit']==commit and r['identity']['config_sha256']==config
            predfile=path.parent/'validation_predictions.npz';assert sha(predfile)==r['predictions_sha256']
            with np.load(predfile) as z:
                y=z['labels'];p=z['probability'];threshold=float(z['threshold'])
                assert len(y)==expected['validation']['samples']
                assert labels_sha(y)==expected['validation']['labels_sha256']
                assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                prediction=p>=threshold
                tp=int(np.sum((y==1)&prediction));fp=int(np.sum((y==0)&prediction));fn=int(np.sum((y==1)&~prediction));tn=int(np.sum((y==0)&~prediction))
                f1=2*tp/max(2*tp+fp+fn,1)
                assert abs(f1-r['validation']['f1'])<1e-12
                assert threshold==float(r['threshold'])
                if task=='Task4C':
                    assert threshold==.5
                    for key,value in expected['validation']['group_sha256'].items():
                        assert hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest()==value
            if phase=='search':
                definition=r['definition'];k=int(candidate.split('_k')[1]);width=24*k-3
                assert k in (2,4,6,10,16) and r['feature_dimensions']==width==definition['feature_dimensions']
                recipe=int(candidate[1]);assert recipe in range(4)
                assert definition['id']==candidate and definition['frequencies']==k
                assert definition['geometry']==('max_radius' if recipe<2 else 'rms_radius')
                kind='zscore' if recipe%2==0 else 'signed_log_zscore_clip8'
                assert definition['features']==r['normalization']['kind']==kind
                assert r['parameters']==(58690+128*width if task=='Task4C' else 142914)
                assert r['normalization']['fit_tokens']==expected['train']['valid_tokens']
                if task=='Task4C':
                    adapter=r['execution_adapter'];declared=replacement['new_job']
                    assert adapter['adapter_sha256']==declared['adapter_sha256']
                    assert adapter['adapter_commit']==declared['adapter_commit']
                    assert adapter['manifest_sha256']==declared['adapter_manifest_sha256']
                    assert adapter['candidate_id']==candidate
                    assert adapter['candidate_index']==recipe*5+(2,4,6,10,16).index(k)
                mean=np.array(r['normalization']['mean']);std=np.array(r['normalization']['std'])
                assert len(mean)==width and len(std)==width and np.isfinite(mean).all() and np.all(std>0)
                assert r['feature_seconds']>=0 and r['seconds']>=0
                groups[candidate].append(r)
            else:
                check=json.loads((root/'control_checks'/f'{task}_{dataset}.json').read_text())
                assert check['status']=='PASS' and check['exact_probabilities'] and check['validation_f1']==f1
                assert check['new_predictions_sha256']==r['predictions_sha256']
                control.append(r)
            records.append(dict(phase=phase,task=task,dataset=dataset,candidate=candidate,tp=tp,fp=fp,fn=fn,tn=tn,f1=f1))
    assert len(records)==441 and len(control)==21 and len(groups)==20
    raw_hashes={(r['task'],r['dataset']):r['raw_checkpoint_sha256'] for r in control if r['task']!='Task4C'}
    for items in groups.values():
        for row in items:
            assert row['profile']=='h0'
            if row['task']!='Task4C':
                assert row['raw_checkpoint_sha256']==raw_hashes[(row['task'],row['dataset'])]
                assert row['trainable_parameters']==52993
    ranking=[]
    for candidate,rows in groups.items():
        assert len(rows)==21 and {r['task']+'/'+r['dataset'] for r in rows}==set(source['units'])
        scores={t:float(np.mean([r['validation']['f1'] for r in rows if r['task']==t])) for t in ('Task3','Task5','Task4C')}
        ranking.append(dict(id=candidate,tasks=scores,score=float(np.mean(list(scores.values()))),feature_dimensions=rows[0]['feature_dimensions']))
    ranking.sort(key=lambda r:(-r['score'],r['feature_dimensions'],r['id']))
    assert [r['id'] for r in ranking]==[r['id'] for r in report['ranking']]
    for a,b in zip(ranking,report['ranking']):
        np.testing.assert_allclose([a['score']]+list(a['tasks'].values()),[b['score']]+[b['tasks'][k] for k in a['tasks']],rtol=0,atol=1e-15)
    baseline={t:float(np.mean([r['validation']['f1'] for r in control if r['task']==t])) for t in ('Task3','Task5','Task4C')}
    baseline['score']=float(np.mean(list(baseline.values())))
    np.testing.assert_allclose(list(baseline.values()),[report['baseline'][k] for k in baseline],rtol=0,atol=1e-15)
    np.testing.assert_allclose([baseline[t] for t in ('Task3','Task5','Task4C','score')],
        [0.794475819517293,0.7377752578222153,0.7270560190703218,0.7531023654699434],rtol=0,atol=1e-15)
    scheduler=json.loads((root/'scheduler_final_status.json').read_text())
    events=[json.loads(s) for s in (root/'runtime_events.jsonl').read_text().splitlines()]
    by_job=collections.defaultdict(dict)
    for event in events:
        assert event['state'] not in by_job[event['job']]
        by_job[event['job']][event['state']]=event
    assert len(scheduler)==149 and len(by_job)==145 and len(events)==290
    gpu_seconds=0;cpu_seconds=0
    for job in scheduler:
        if job['JobID'] in cancelled_jobs:
            assert job['State'].startswith('CANCELLED') and int(job['ElapsedRaw'])==0
            assert job['JobIDRaw'] not in by_job
            continue
        assert job['State']=='COMPLETED' and job['ExitCode']=='0:0'
        pair=by_job[job['JobIDRaw']];assert set(pair)=={'STARTED','ENDED'} and pair['ENDED']['exit_code']==0
        assert pair['STARTED']['host']==job['NodeList']==pair['ENDED']['host']
        if pair['STARTED']['phase']=='merge':cpu_seconds+=int(job['ElapsedRaw'])
        else:
            assert 'V100' in pair['STARTED']['device'];gpu_seconds+=int(job['ElapsedRaw'])
    # Inference feature preparation and training have different old timing scopes; report explicitly.
    costs=[]
    for candidate,items in [('p35_h0',control)]+sorted(groups.items()):
        for task in ('Task3','Task5','Task4C'):
            rows=[r for r in items if r['task']==task]
            feature=[r.get('feature_seconds',0.) for r in rows]
            train=[r['seconds']-(r.get('feature_seconds',0.) if task=='Task4C' else 0.) for r in rows]
            costs.append(dict(candidate=candidate,task=task,mean_parameters=float(np.mean([r['parameters'] for r in rows])),
                mean_feature_seconds=float(np.mean(feature)),mean_fitting_seconds=float(np.mean(train)),
                mean_encoding_and_fitting_seconds=float(np.mean(np.array(feature)+train)),
                baseline_timing_note='Original control does not separate encoding; Task3/5 seconds excludes encoding, Task4C includes it' if candidate=='p35_h0' else None))
    outcome=dict(status='PASS',source_check_sha256=sha(source_path),auditor_sha256=sha(__file__),
        predictions_verified=441,scheduler_processes=149,executed_processes=145,cancelled_before_start=sorted(cancelled_jobs),runtime_events=290,ranking=ranking,baseline=baseline,
        gpu_allocated_hours=gpu_seconds/3600,cpu_allocated_seconds=cpu_seconds,costs=costs,counts=records,
        test_read=False,selected=ranking[0])
    write(root/'independent_audit.json',outcome)
    print(json.dumps({k:outcome[k] for k in ('status','predictions_verified','scheduler_processes','runtime_events','selected','gpu_allocated_hours')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source-check',action='store_true')
    args=parser.parse_args()
    if args.source_check:
        result=source_check(args.root);print(json.dumps(dict(status=result['status'],units=len(result['units']),verified_files=len(result['verified_files']))))
    else:audit(args.root)
