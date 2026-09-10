"""Collect scalar research records only; never package checkpoints or full caches."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/Verify_Task6_ScarceGeneralization_4.1'))
    p.add_argument('--final',action='store_true');args=p.parse_args();root=args.root
    ledgers=list(root.glob('submissions_*.jsonl'))
    jobs=[json.loads(line)['job_id'] for path in ledgers for line in path.read_text().splitlines()]
    assert jobs
    if args.final:
        assert json.loads((root/'final_audit.json').read_text())['passed']
    status=subprocess.check_output(['sacct','-j',','.join(jobs),
        '--format=JobID,State,ExitCode,Elapsed,Submit,Start,End,NodeList','-P','-X'],text=True)
    (root/('slurm_status_final.txt' if args.final else 'slurm_status_search.txt')).write_text(status)
    name='metadata_final.zip' if args.final else 'metadata_search.zip'
    with zipfile.ZipFile(root/name,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.txt','.out','.err'):
                archive.write(path,path.relative_to(root))
    print(root/name,(root/name).stat().st_size)
