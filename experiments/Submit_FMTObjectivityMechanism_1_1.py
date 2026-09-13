"""Submit the bounded dependency chain and immediately persist every Slurm ID."""
import datetime
import json
from pathlib import Path
import subprocess
from experiments.Run_Task135_GeometricControls import sha,write_json


def main():
    config='config/Verify_FMTObjectivityMechanism_1.3.json'
    spec=json.loads(Path(config).read_text());root=Path(spec['output_root'])
    if (root/'submissions.jsonl').exists():raise FileExistsError('Existing submission; inspect, never duplicate')
    (root/'logs').mkdir(parents=True,exist_ok=True)
    jobs={}
    phases=[('preflight',[],['--time=00:45:00']),('smoke',[],['--time=00:15:00']),
            ('Task1',['preflight','smoke'],['--array=0-29%6','--time=03:00:00']),
            ('Task2',['preflight','smoke'],['--array=0-29%6','--time=04:00:00','--gres=gpu:1','--constraint=a100|v100','--exclude=gpu203-02-r']),
            ('audit',['Task1','Task2'],['--time=00:30:00'])]
    for phase,deps,extras in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G',
                 f'--job-name=FMTmechanism_{phase}',f'--output={root}/logs/{phase}.%A_%a.out',
                 f'--error={root}/logs/{phase}.%A_%a.err']+extras
        if deps:command+=['--dependency=afterok:'+':'.join(jobs[k] for k in deps)]
        command+=['ibex_bash/verify_fmt_objectivity_mechanism_1p1.sh',phase]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        assert job.isdigit();jobs[phase]=job
        item=dict(experiment=spec['experiment'],phase=phase,job=job,
                  submitted=datetime.datetime.now().astimezone().isoformat(),
                  config=config,config_sha256=sha(config),base_commit=spec['base_commit'],
                  source_manifest_sha256=sha('SOURCE_MANIFEST.sha256'),
                  expected_device='A100 or V100' if phase=='Task2' else 'CPU',command=command)
        for path in [root/'submissions.jsonl',Path('docs/ibex_run_registry.md')]:
            with path.open('a',encoding='utf-8') as f:
                text=json.dumps(item)
                f.write(text+'\n' if path.suffix=='.jsonl' else '\n- **SUBMITTED '+spec['experiment']+'** '+text+'\n')
                f.flush()
                import os
                os.fsync(f.fileno())
        print(json.dumps(item),flush=True)
    write_json(root/'jobs.json',jobs)


if __name__=='__main__':main()
