"""Reconcile all fast-ablation Slurm processes and package model-free evidence."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    root=Path('outputs/Ablation_Task4C_FMTv5Blocks_4.19')
    submissions=[json.loads(x) for x in (root/'submissions.jsonl').read_text().splitlines()]
    columns=['JobID','JobIDRaw','State','ExitCode','NodeList','Start','End','Elapsed','AllocTRES']
    output=subprocess.check_output(['sacct','-j',','.join(r['job_id'] for r in submissions),
        '--noheader','--parsable2','--format='+','.join(columns)],text=True)
    rows=[dict(zip(columns,line.split('|'))) for line in output.splitlines() if line.strip()]
    expected=set()
    for r in submissions:
        expected.update([r['job_id']+'_'+str(i) for i in range(5)] if r['phase']=='train' else [r['job_id']])
    actual={r['JobID']:r for r in rows if '.' not in r['JobID']}
    assert set(actual)==expected and len(actual)==7
    events=[json.loads(x) for x in (root/'runtime_events.jsonl').read_text().splitlines()]
    durable=[json.loads(p.read_text()) for p in (root/'lifecycle_events').glob('*.json')]
    canonical=lambda seq:collections.Counter(json.dumps(r,sort_keys=True) for r in seq)
    assert canonical(events)==canonical(durable) and len(events)==14
    by_job=collections.defaultdict(list)
    for event in events:by_job[event['job']].append(event)
    for key,r in actual.items():
        assert (r['State'],r['ExitCode'])==('COMPLETED','0:0')
        pair=by_job[r['JobIDRaw']]
        assert len(pair)==2 and {e['state'] for e in pair}=={'STARTED','ENDED'}
        assert all(e['host']==r['NodeList'] for e in pair)
        assert next(e for e in pair if e['state']=='ENDED')['exit_code']==0
    audit=json.loads((root/'independent_audit.json').read_text())
    assert audit['status']=='PASS' and len(audit['runs'])==6 and not audit['test_loaded']
    assert audit['summary_sha256']==hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()
    summary=json.loads((root/'summary.json').read_text())
    exported=[]
    for r in summary['rows']:
        exported.append(dict(block=r['block'],train_f1=r['train_f1'],validation_f1=r['validation_f1'],
            importance_f1_drop=r['importance_f1_drop'],validation_average_precision=r['validation_average_precision'],
            channel_validation_f1=r['per_flow_validation']['channel'],tbl_validation_f1=r['per_flow_validation']['tbl'],
            selected_epoch=r['selected_epoch']))
    with (root/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(exported[0]));w.writeheader();w.writerows(exported)
    report=dict(status='PASS',scheduler_processes=rows,successful_processes=7,runtime_events=14,
        all_events_match_per_event_files=True,auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'scheduler_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth')) and not any(root.rglob('*.ckpt'))
    included=[];archive=root/'metrics_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as package:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            if p.suffix not in ('.json','.jsonl','.csv','.out','.err') and not p.name.endswith('predictions.npz'):continue
            package.add(p,arcname=str(p.relative_to(root)));included.append(str(p.relative_to(root)))
    manifest=dict(files=included,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),model_weights_included=False)
    (root/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',processes=7,events=14,files=len(included),archive_sha256=manifest['archive_sha256'])))


if __name__=='__main__':main()
