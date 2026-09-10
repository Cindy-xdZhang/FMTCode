"""Package audited tabular evidence and metadata, never models or trajectory arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--expected-records',type=int,required=True)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    audit=json.loads((root/'independent_audit.json').read_text())
    assert audit['status']=='PASS' and audit['metric_records']==args.expected_records
    assert audit['metrics_sha256']==digest(root/'metrics.csv')
    assert audit['summary_sha256']==digest(root/'summary.json')
    assert audit['config_sha256']==digest(root/'runtime_config.json')
    selected=set(root.glob('*.json'))|set(root.glob('*.jsonl'))|{root/'metrics.csv'}
    selected.update((root/'build').glob('*.json'))
    selected.update((root/'runs').glob('*/*/*/*.json'))
    selected.update((root/'runtime_recovery_checks').glob('*.json'))
    selected.update((root/'failed_attempts').glob('stalled_eval_*/*.json'))
    selected.update((root/'failed_attempts').glob('stalled_eval_*/*.jsonl'))
    for part in (root/'audit_shards_fast').glob('*'):
        if part.is_dir():
            selected.update(part.glob('*.json'))
            selected.update(part.glob('*.csv'))
    selected={p for p in selected if p.is_file() and p.name!='evidence_manifest.json'}
    assert all(p.suffix in ('.json','.jsonl','.csv') for p in selected)
    for path in selected:
        assert path.resolve().is_relative_to(root),path
    jobs=set()
    for path in root.glob('*submissions*.jsonl'):
        for line in path.read_text().splitlines():
            item=json.loads(line)
            if item.get('job_id'):
                jobs.add(str(item['job_id']))
    stats=None
    if jobs:
        command=['sacct','-X','-n','-P','-j',','.join(sorted(jobs)),
                 '--format=JobIDRaw,JobName,State,ExitCode,Start,End,Elapsed,NodeList,AllocTRES%160']
        stats=subprocess.check_output(command,text=True)
    manifest=dict(status='PASS',metric_records=args.expected_records,
        audit_sha256=digest(root/'independent_audit.json'),
        training_git_commits=audit.get('training_git_commits',[]),
        audit_git_commit=audit['git_commit'],
        packager_sha256=digest(Path(__file__)),
        source_root=str(root),jobs=sorted(jobs),
        scope='JSON/JSONL/CSV only. No model checkpoint, trajectory array, or raw flow field is included.',
        files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(selected)])
    target=root/'audited_evidence.zip'
    assert not target.exists(),'Never replace a previously packaged evidence artifact'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(selected):
            archive.write(path,path.relative_to(root).as_posix())
        archive.writestr('evidence_manifest.json',json.dumps(manifest,indent=2)+'\n')
        if stats is not None:
            archive.writestr('slurm_accounting.psv',stats)
    print(json.dumps(dict(path=str(target),files=len(selected),bytes=target.stat().st_size,sha256=digest(target))))


if __name__=='__main__':
    main()
