"""Submit the preregistered run once, recording each job immediately."""
import datetime
import json
import os
from pathlib import Path
import subprocess
from experiments.Run_Task135_GeometricControls import sha, write_json


def main():
    config='config/Verify_Task1235_ObjectiveFMTnTDO_1.1.json'
    spec=json.loads(Path(config).read_text());root=Path(spec['output_root'])
    if (root/'submissions.jsonl').exists():raise FileExistsError('Already submitted; inspect existing jobs')
    (root/'logs').mkdir(parents=True,exist_ok=True)
    gpu=['--gres=gpu:1','--constraint=a100|v100','--exclude=gpu203-02-r']
    phases=[('build',[],['--array=0-1','--time=02:00:00']),
            ('smoke',[],['--time=00:20:00']),
            ('preflight',['build'],['--time=01:00:00']),
            ('Task1',['preflight','smoke'],['--array=0-29%6','--time=02:00:00']),
            ('Task2',['preflight','smoke'],['--array=0-29%6','--time=04:00:00']+gpu),
            ('Task35',['preflight','smoke'],['--array=0-29%6','--time=06:00:00']+gpu),
            ('audit',['Task1','Task2','Task35'],['--time=00:30:00'])]
    jobs={}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for phase,deps,extra in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G',
                 f'--job-name=FMTnTDO_{phase}',f'--output={root}/logs/{phase}.%A_%a.out',
                 f'--error={root}/logs/{phase}.%A_%a.err']+extra
        if deps:command+=['--dependency=afterok:'+':'.join(jobs[k] for k in deps)]
        command+=['ibex_bash/verify_task1235_ntdo_1p1.sh',phase]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        assert job.isdigit();jobs[phase]=job
        item={'experiment':spec['experiment'],'phase':phase,'job':job,
              'submitted':datetime.datetime.now().astimezone().isoformat(),'config':config,
              'config_sha256':sha(config),'git_commit':commit,'source_manifest_sha256':sha('SOURCE_MANIFEST.sha256'),
              'expected_device':'A100 or V100' if phase in ('Task2','Task35') else 'CPU','command':command}
        for path in [root/'submissions.jsonl',Path('docs/ibex_run_registry.md')]:
            with path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(item)+'\n' if path.suffix=='.jsonl' else '\n- **SUBMITTED nTDO 1.1** '+json.dumps(item)+'\n')
                f.flush();os.fsync(f.fileno())
        print(json.dumps(item),flush=True)
    write_json(root/'jobs.json',jobs)


if __name__=='__main__':main()
