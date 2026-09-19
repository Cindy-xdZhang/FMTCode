"""Submit ten single-seed v3 fits, after the shared CPU neighborhood array succeeds."""
import argparse
from pathlib import Path
import json
import shutil
import subprocess
from datetime import datetime,timezone
from experiments import Task4C_V3Training_3_1 as run


def submit(neighbor_job):
    spec=run.fmt_spec();root=Path(spec['output']);(root/'logs').mkdir(parents=True,exist_ok=True)
    assert not (root/'submission.json').exists(),'Already submitted'
    assert shutil.disk_usage(root).free>100*2**30
    assert spec['final']['seeds']==[96611]
    jobs={}
    def add(name,phase,deps,array=None,gpu=False,wall='02:00:00',memory='96G'):
        cmd=['sbatch','--parsable','--propagate=NONE','--kill-on-invalid-dep=yes','--cpus-per-task=4',
             '--mem='+memory,'--time='+wall,'--job-name=V3BQ_'+name,
             '--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
        if deps:cmd+=['--dependency=afterok:'+':'.join(deps)]
        if array:cmd+=['--array='+array]
        if gpu:cmd+=['--gres=gpu:1','--constraint=v100','--exclude=gpu213-18']
        cmd+=['ibex_bash/task4c_v3_training_3p1.sh',phase]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit();jobs[name]=job
        run.base_old.frozen.append_line(root/'submissions.jsonl',json.dumps(dict(name=name,job=job,command=cmd,at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        run.write(root/'submission.json',dict(identity=run.identity(run.CONFIG),jobs=jobs,neighbor_job=neighbor_job,seed=96611,formal_fits=10,
            index_map={str(i):run.result_location(spec,i)[2] for i in range(10)}))
        print(name,job,flush=True)
        return job
    source=add('source','verify-source',[neighbor_job],memory='48G')
    checks=add('preflight','preflight',[source],array='0-9%4',gpu=True,wall='01:00:00')
    # Each model starts after its own preflight succeeds; no unrelated baseline dependency.
    for index in range(10):
        add('train'+str(index),'train',[checks+'_'+str(index)],array=str(index),gpu=True,wall='14-00:00:00')
    add('audit','audit-results',[jobs['train'+str(i)] for i in range(10)],memory='64G')
    print(json.dumps(jobs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--neighbor-job',required=True);a=p.parse_args();submit(a.neighbor_job)
