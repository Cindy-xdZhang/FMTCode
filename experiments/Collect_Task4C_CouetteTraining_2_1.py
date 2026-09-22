"""Read-only Ibex status for incremental raw-curl Couette training."""
import json
from pathlib import Path
import shlex
import subprocess


def main():
    code='''import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/ibex/user/zhanx0o/FMT_Task4C_CouetteRawCurl_2p1_20260922')
out=root/'outputs/mainExp_Task4C_CouetteTraining_2.1'
sub=json.loads((out/'submission.json').read_text());jobs=','.join(sub['jobs'].values());files={}
for p in [out/'source_verification.json',out/'prepared.json',out/'result_audit.json',*out.glob('fold*/encoding/complete.json'),*out.glob('pilot/*/preflight.json'),*out.glob('fold*/runs/*/progress.json'),*out.glob('fold*/runs/*/result.json')]:
 if p.exists():files[str(p.relative_to(out))]=json.loads(p.read_text())
errors={}
for p in (out/'logs').glob('*.err'):
 if p.stat().st_size:
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));errors[p.name]=f.read().decode(errors='replace')
print(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(),submission=sub,files=files,errors=errors,
 status=subprocess.check_output(['sacct','-X','-j',jobs,'--format=JobID,State,ExitCode,Elapsed','-P','-n'],text=True),
 queue=subprocess.check_output(['squeue','-h','-j',jobs,'-o','%i %T %R'],text=True))))
'''
    result=subprocess.run(['ssh','-q','-o','BatchMode=yes','-x','zhanx0o@glogin.ibex.kaust.edu.sa','/home/zhanx0o/anaconda3/envs/deepvortex/bin/python -c '+shlex.quote(code)],capture_output=True,text=True,check=True)
    record=json.loads(result.stdout);out=Path('outputs/mainExp_Task4C_CouetteTraining_2.1/remote_evidence');out.mkdir(parents=True,exist_ok=True)
    (out/'latest_status.json').write_text(json.dumps(record,indent=2)+'\n')
    summary={k:record[k] for k in ('at_utc','submission','status','queue','errors')}
    summary['checks']={k:v.get('complete') for k,v in record['files'].items() if not ('progress' in k or 'result.json' in k)}
    summary['results']={k:v.get('latest',v.get('primary')) for k,v in record['files'].items() if 'progress' in k or 'result.json' in k}
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
