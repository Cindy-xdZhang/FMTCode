"""Independent prediction, source, scheduler and reporting audit; no training."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def scores(y,p,threshold):
    positive=p>=threshold
    tp=int(np.sum(positive & (y==1)));fp=int(np.sum(positive & (y==0)))
    fn=int(np.sum(~positive & (y==1)));tn=int(np.sum(~positive & (y==0)))
    return dict(f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),
        recall=tp/max(tp+fn,1),accuracy=(tp+tn)/len(y),
        average_precision=float(average_precision_score(y,p)) if np.any(y) else 0.,
        confusion_matrix=[[tn,fp],[fn,tp]],sample_count=len(y),positives=int(y.sum()))


def compare(actual,expected):
    for key,value in actual.items():
        if isinstance(value,float):assert np.isclose(value,expected[key],atol=1e-12,rtol=1e-12),(key,value,expected[key])
        else:assert value==expected[key],(key,value,expected[key])


def dump_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit(root):
    started=time.perf_counter()
    root=Path(root);summary=load(root/'summary.json')
    assert summary['status']=='PASS' and summary['training_runs']==180
    identity=summary['identity'];arms=('raw_c156','fmt_c156','asap_fmt')
    device_check=load(root/'device_encoding_verification.json')
    assert device_check['status']=='MEASURED' and device_check['identity']==identity
    assert not device_check['test_read'] and device_check['training_runs']==0 and len(device_check['checks'])==60
    assert device_check['started_at']<device_check['ended_at'] and 'V100' in device_check['gpu']
    submissions=load(root/'submissions.json');resumed=load(root/'quota_resume_submissions.json')
    assert all(x['identity']==identity for x in submissions+resumed)
    sources={};manifests={};norms={};objectivity=[];timing=[]
    def remember_source(item):
        for pathkey,hashkey in [('path','sha256'),('label_path','label_sha256')]:
            if pathkey in item:
                path=item[pathkey]
                assert path not in sources or sources[path]==item[hashkey]
                sources[path]=item[hashkey]
    for file in sorted((root/'cache').glob('*/*/manifest.json')):
        m=load(file);assert m['status']=='PASS' and m['identity']==identity and not m['test_read']
        key=(m['task'],m['dataset']);manifests[key]=m
        assert digest(file.parent/'normalization.json')==m['normalization_sha256']
        norms[key]=load(file.parent/'normalization.json')
        for role,record in m['roles'].items():
            for item in record['source']:remember_source(item)
            for name in ('labels.npy','scale_id.npy','row_id.npy'):
                assert digest(file.parent/role/name)==record['files'][name]
        obj=m['roles']['train']['objectivity'];assert obj['batch_independence']
        objectivity.append(dict(task=key[0],dataset=key[1],**obj))
        timing.append(dict(task=key[0],dataset=key[1],camera_seconds_per_primitive=m['roles']['train']['camera_seconds_per_primitive'],
            **{a+'_encode_seconds':m['roles']['train']['encoding_seconds'][a] for a in arms}))
        for norm in norms[key].values():
            assert norm['train_only'] and norm['fit_tokens']==m['roles']['train']['samples']*7
    assert len(manifests)==20
    rows=[];scale_rows=[];run_count=0;paired_ids={};histories={};training=[]
    for file in sorted((root/'runs').glob('*/*/seed*/result.json')):
        result=load(file);folder=file.parent;run_count+=1
        assert result['identity']==identity and result['status']=='PASS' and not result['checkpoint_files_created']
        assert 'V100' in result['gpu']
        assert digest(folder/'selection.lock.json')==result['selection_lock_sha256']
        lock=load(folder/'selection.lock.json');assert not lock['test_loaded'] and lock['identity']==identity
        key=(result['task'],result['dataset']);assert lock['normalization_sha256']==manifests[key]['normalization_sha256']
        for collection in result['source'].values():
            for item in collection:remember_source(item)
        for arm,record in result['results'].items():
            assert record['parameters']==(993154 if arm=='raw_c156' else 992386)
            history=[json.loads(line) for line in (folder/(arm+'_history.jsonl')).read_text().splitlines()]
            assert len(history)==record['epochs']<=100
            best=-np.inf;selected=0
            for item in history:
                if item['validation_average_precision']>best+1e-4:
                    best=item['validation_average_precision'];selected=item['epoch']
                assert item['epoch']-selected<20 or item['epoch']==len(history)
            assert len(history)==100 or len(history)-selected==20
            assert selected==record['selected_epoch']==lock['arms'][arm]['selected_epoch']
            compare({'average_precision':best},record['validation'])
            order_key=(*key,result['seed'])
            sequence=[x['permutation_sha256'] for x in history]
            if order_key in histories:
                old=histories[order_key];assert old[:min(len(old),len(sequence))]==sequence[:min(len(old),len(sequence))]
            histories[order_key]=sequence
            training.append(dict(task=key[0],dataset=key[1],seed=result['seed'],arm=arm,
                seconds=record['training_seconds'],epochs=len(history),selected_epoch=selected))
            for role,expected_hash in record['predictions'].items():
                file_npz=folder/(arm+'_'+role+'.npz');assert digest(file_npz)==expected_hash
                with np.load(file_npz) as z:
                    y=z['labels'];p=z['probability'];threshold=float(z['threshold'])
                    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                    assert threshold==record['threshold']==lock['arms'][arm]['threshold']
                    ids=np.column_stack((z['row_id'],z['scale_id'],y))
                    pairkey=(*key,role)
                    if pairkey in paired_ids:np.testing.assert_array_equal(ids,paired_ids[pairkey])
                    else:paired_ids[pairkey]=ids
                    score=scores(y,p,threshold);compare(score,record[role])
                    if role=='validation':
                        precision,recall,candidates=precision_recall_curve(y,p)
                        f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
                        assert threshold==float(candidates[np.nanargmax(f1)])
                        cached=root/'cache'/key[0]/key[1]/'validation'
                        np.testing.assert_array_equal(y,np.load(cached/'labels.npy'))
                        np.testing.assert_array_equal(z['row_id'],np.load(cached/'row_id.npy'))
                    common=dict(task=key[0],dataset=key[1],seed=result['seed'],arm=arm,role=role,threshold=threshold)
                    rows.append(dict(**common,**{k:v for k,v in score.items() if k!='confusion_matrix'}))
                    for scale in np.unique(z['scale_id']):
                        mask=z['scale_id']==scale;ss=scores(y[mask],p[mask],threshold)
                        compare(ss,record[role+'_by_scale'][str(scale)])
                        scale_rows.append(dict(**common,scale_id=int(scale),**{k:v for k,v in ss.items() if k!='confusion_matrix'}))
    assert run_count==60 and len(rows)==450 and len(training)==180
    assert len(list(root.rglob('*.pt')))==len(list(root.rglob('*.pth')))==0
    for task in ('Task3','Task5'):
        for arm in arms:
            for metric in ('f1','average_precision'):
                values=[float(np.mean([r[metric] for r in rows if r['task']==task and r['arm']==arm and r['role']=='test' and r['seed']==seed])) for seed in (40,41,42)]
                target=summary['summary'][task][arm][metric]
                np.testing.assert_allclose(values,target['seeds'],atol=1e-12,rtol=1e-12)
                compare(dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1))),target)
    for arm in arms:
        for metric in ('f1','average_precision'):
            values=[float(np.mean([r[metric] for r in rows if r['task']=='Task3' and r['arm']==arm and r['role']=='transfer_task5' and r['seed']==seed])) for seed in (40,41,42)]
            target=summary['summary']['fixed_task3_transfer_to_task5'][arm][metric]
            np.testing.assert_allclose(values,target['seeds'],atol=1e-12,rtol=1e-12)
            compare(dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1))),target)
    # Recheck original training, validation and test files after all computation.
    for path,expected in sources.items():assert digest(path)==expected,path
    jobs=[x['job_id'] for x in submissions+resumed]+[device_check['job_id']]+device_check['previous_failed_jobs']
    sacct=subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--parsable2',
        '--format=JobID,State,ExitCode,ElapsedRaw,NodeList'],text=True)
    (root/'scheduler_audit.psv').write_text(sacct)
    scheduler={r['JobID']:r for r in csv.DictReader(sacct.splitlines(),delimiter='|')}
    active=[submissions[0],*resumed]
    expected_jobs={active[0]['job_id'],active[-1]['job_id']}
    expected_jobs|={resumed[0]['job_id']+'_'+str(i) for i in (16,18,19)}
    expected_jobs|={resumed[1]['job_id']+'_'+str(i) for i in range(60)}
    expected_jobs|={submissions[1]['job_id']+'_'+str(i) for i in range(20) if i not in (16,18,19)}
    assert len(expected_jobs)==82
    assert scheduler[device_check['job_id']]['State']=='COMPLETED' and scheduler[device_check['job_id']]['ExitCode']=='0:0'
    for job in expected_jobs:assert scheduler[job]['State']=='COMPLETED' and scheduler[job]['ExitCode']=='0:0',scheduler[job]
    events=[load(p) for p in (root/'runtime').glob('*.json')]
    pairs={}
    for event in events:
        assert event['identity']==identity
        pairs.setdefault((event['job_id'],event['array_index']),{})[event['state']]=event
    complete={k:v for k,v in pairs.items() if 'ENDED' in v and v['ENDED']['exit_code']==0}
    expected_units={('preflight','none'),('merge','none')}
    expected_units|={('prepare',str(i)) for i in range(20)}|{('train',str(i)) for i in range(60)}
    found=set()
    for pair in complete.values():
        begin,end=pair['STARTED'],pair['ENDED'];found.add((end['phase'],end['array_index']))
        assert begin['time_utc']<=end['time_utc'] and begin['node']==end['node']
        if end['phase'] in ('train','preflight'):assert 'V100' in end['gpu']
    assert expected_units<=found and len(complete)==82
    dump_csv(root/'metrics_all_roles.csv',rows);dump_csv(root/'metrics_by_scale.csv',scale_rows)
    dump_csv(root/'encoding_times.csv',timing);dump_csv(root/'training_times.csv',training)
    evidence=dict(status='PASS',scientific_identity=identity,auditor_sha256=digest(__file__),
        scope='Prediction metrics, selection, source hashes and required scheduler completion; numerical tolerances reported separately',
        audited_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        predictions=len(rows),scale_metric_rows=len(scale_rows),fits=len(training),source_files=len(sources),
        original_source_sha256=sources,successful_required_processes=len(expected_jobs),
        complete_runtime_pairs=len(complete),runtime_events=len(events),
        incomplete_runtime_pairs=[list(k) for k,v in pairs.items() if 'ENDED' not in v],
        additional_device_check_job=device_check['job_id'],device_encoding_checks=60,
        device_strict_tolerance_pass=device_check['strict_tolerance_pass'],
        quota_failures_retained=True,objectivity=objectivity,checkpoint_files=0)
    (root/'independent_audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('original_source_sha256','objectivity','scientific_identity')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');audit(parser.parse_args().root)
