"""Submit the audited Task1/2/3 dependency chain and durably register every ID."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from experiments.Run_Task135_GeometricControls import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-jobs',required=True)
    parser.add_argument('--config',default='config/Verify_LargeNeighbor_1.1.json')
    args=parser.parse_args();spec=json.loads(Path(args.config).read_text());root=Path(spec['output_root'])
    register=root/'training_submissions.jsonl'
    if register.exists():raise FileExistsError('Submission registry already exists; inspect it before resuming')
    script='ibex_bash/verify_large_neighbor_1p1.sh'
    common=['--cpus-per-task=4','--mem=48G']
    def submit(name,dependency,extras,batch=script):
        command=['sbatch','--parsable',f'--job-name=LN11_{name}',
                 '--dependency=afterok:'+dependency,
                 '--output='+str(root/'logs'/f'{name}.%A_%a.out'),
                 '--error='+str(root/'logs'/f'{name}.%A_%a.err'),*common,*extras,batch]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        item={'name':name,'job':job,'submitted':datetime.datetime.now().astimezone().isoformat(),
              'experiment':spec['experiment'],'base_commit':spec['base_commit'],
              'config_sha256':sha(args.config),'source_manifest_sha256':sha('SOURCE_MANIFEST.sha256'),
              'command':command}
        with register.open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True);return job
    integrity=submit('integrity',args.cache_jobs,['--time=01:00:00'],
                     'ibex_bash/audit_large_neighbor_cache_1p1.sh')
    preflight=submit('preflight',integrity,['--time=01:00:00','--export=ALL,MODE=preflight'])
    ids=[]
    for task in ('Task1','Task2','Task3'):
        extras=['--array=0-49%'+('10' if task=='Task1' else '15'),
                '--export=ALL,MODE=run,TASK='+task]
        if task=='Task1':extras+=['--time=01:00:00']
        else:extras+=['--time='+('02:00:00' if task=='Task2' else '04:00:00'),
                      '--gres=gpu:1','--constraint=a100|v100','--exclude=gpu203-02-r']
        ids.append(submit(task,preflight,extras))
    submit('final_audit',':'.join(ids),['--time=01:00:00','--export=ALL,MODE=audit'])


if __name__=='__main__':main()
