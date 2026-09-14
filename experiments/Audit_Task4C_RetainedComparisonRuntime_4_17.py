"""Reconcile the original audit failure and its explicit V100 verification."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    config=json.loads(Path('config/Verify_Task4C_BaselineDevice_4.17.json').read_text())
    roots=[Path(config['source_output']),Path(config['output'])]
    submissions=[json.loads(x) for root in roots for x in (root/'submissions.jsonl').read_text().splitlines()]
    columns=['JobID','JobIDRaw','State','ExitCode','NodeList','Start','End','Elapsed','AllocTRES']
    output=subprocess.check_output(['sacct','-j',','.join(r['job_id'] for r in submissions),
        '--noheader','--parsable2','--format='+','.join(columns)],text=True)
    rows=[dict(zip(columns,line.split('|'))) for line in output.splitlines() if line.strip()]
    expected=set()
    for r in submissions:
        expected.update([r['job_id']+'_'+str(i) for i in range(9)] if r['phase']=='train' else [r['job_id']])
    actual={r['JobID']:r for r in rows if '.' not in r['JobID']}
    assert set(actual)==expected and len(actual)==14
    events=[]
    for root in roots:
        aggregate=[json.loads(x) for x in (root/'runtime_events.jsonl').read_text().splitlines()]
        durable=[json.loads(p.read_text()) for p in (root/'lifecycle_events').glob('*.json')]
        canonical=lambda seq:collections.Counter(json.dumps(r,sort_keys=True) for r in seq)
        assert canonical(aggregate)==canonical(durable)
        events+=aggregate
    assert len(events)==28
    by_job=collections.defaultdict(list)
    for event in events:by_job[event['job']].append(event)
    for key,r in actual.items():
        failure=key==config['after_job']
        assert (r['State'],r['ExitCode'])==(('FAILED','1:0') if failure else ('COMPLETED','0:0'))
        pair=by_job[r['JobIDRaw']]
        assert len(pair)==2 and {e['state'] for e in pair}=={'STARTED','ENDED'}
        assert all(e['host']==r['NodeList'] for e in pair)
        assert next(e for e in pair if e['state']=='ENDED')['exit_code']==int(failure)
    validated=json.loads((roots[1]/'validated_comparison.json').read_text())
    assert validated['status']=='PASS' and validated['original_fmt_baseline_reproduced_on_original_device']
    metric_rows=[]
    for root in roots:
        for p in sorted(root.glob('variants/*/runs/*/result.json')):
            r=json.loads(p.read_text())
            assert r['predictions_sha256']==hashlib.sha256((p.parent/'predictions.npz').read_bytes()).hexdigest()
            for split in ('training','validation','test'):
                for flow,score in [('pooled',r[split]['pooled']),*r[split]['per_flow'].items()]:
                    metric_rows.append(dict(version=r['version'],encoder=r['encoder'],seed=r['seed'],
                        device=r['device'],split=split,flow=flow,f1=score['f1'],average_precision=score['average_precision']))
    assert len(metric_rows)==90
    with (roots[1]/'metrics_all_attempts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
    report=dict(status='PASS',scheduler_processes=rows,successful_processes=13,
        preserved_failed_audit=config['after_job'],runtime_events=28,
        original_a100_baseline_retained=True,all_events_match_per_event_files=True,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (roots[1]/'scheduler_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    packages=[]
    for root in roots:
        weights=[str(p) for pattern in ('*.pt','*.pth','*.ckpt') for p in root.rglob(pattern)]
        assert not weights
        included=[];archive=root/'metrics_evidence.tar.gz'
        with tarfile.open(archive,'w:gz') as package:
            for path in sorted(root.rglob('*')):
                if not path.is_file() or path.is_symlink():continue
                if path.suffix not in ('.json','.jsonl','.csv','.out','.err') and path.name!='predictions.npz':continue
                package.add(path,arcname=str(path.relative_to(root)));included.append(str(path.relative_to(root)))
        manifest=dict(files=included,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),model_weights_included=False)
        (root/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        packages.append(dict(root=str(root),files=len(included),sha256=manifest['archive_sha256']))
    print(json.dumps(dict(status='PASS',processes=14,events=28,packages=packages)))


if __name__=='__main__':main()
