"""Recover an evaluator using an equivalent distance kernel; model code is frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from FMT_Utils.FlowMapData_3D import sha256,write_json
from FMT_Utils.FlowMapAuditMetrics_3D import measured_trajectories
from experiments.Build_Task678_FlowMap_1_1 import provenance
from experiments.Submit_Task678_DirectFMTFit_1_1 import event
from experiments import Run_Task678_HanSampling_1_1 as original
from experiments.Run_Task678_FlowMap_1_1 import truth_radius

FROZEN_COMMIT='45eefee828f5439f6aad6940b01aafe7803aa691'
FROZEN_FILES=['FMT_Utils/HanFlowMapFit_3D.py','FMT_Utils/FlowMapFit_3D.py',
    'FMT_Utils/FlowMapModels_3D.py','FMT_Utils/FlowMapData_3D.py','FMT_Utils/HanFlowMapData_3D.py',
    'FMT_Utils/DFT_FMT_3D.py','experiments/Run_Task678_HanSampling_1_1.py','experiments/Run_Task678_FlowMap_1_1.py']


def validate_source():
    evidence={}
    for name in FROZEN_FILES:
        frozen=subprocess.check_output(['git','show',FROZEN_COMMIT+':'+name])
        current=Path(name).read_bytes()
        assert current.replace(b'\r\n',b'\n')==frozen.replace(b'\r\n',b'\n'),name
        evidence[name]=hashlib.sha256(frozen).hexdigest()
    return evidence


def check_existing(spec,config,index,evidence):
    dataset,arm,seed=original.matrix(spec)[index]
    root=Path(spec['output_root']);directory=root/'runs'/dataset/arm/f'seed{seed}'
    data,sources=original.load_role(spec,dataset,'fit');records=[]
    for task in ('Task6','Task7','Task8'):
        path=directory/f'{task}_fit_normal_predictions.npz'
        with np.load(path,allow_pickle=False) as archive:prediction=archive['prediction']
        truth,radius=truth_radius(data,task)
        started=time.perf_counter();computed=measured_trajectories(prediction,truth,radius)
        seconds=time.perf_counter()-started
        saved=directory/f'{task}_fit_normal.json';differences=[]
        if saved.exists():
            record=json.loads(saved.read_text())
            assert record['prediction_sha256']==sha256(path) and record['source_records']==sources
            for key,value in computed.items():
                if isinstance(value,str):assert value==record['metrics'][key]
                else:
                    np.testing.assert_allclose(value,record['metrics'][key],rtol=1e-9,atol=1e-9,err_msg=key)
                    differences.append(float(np.max(np.abs(np.asarray(value)-np.asarray(record['metrics'][key])))))
        records.append(dict(task=task,shape=list(prediction.shape),fast_kernel_seconds=seconds,
            prediction_sha256=sha256(path),compared_with_original=saved.exists(),
            maximum_absolute_difference=max(differences) if differences else None))
        print(json.dumps(records[-1]),flush=True)
    report={**provenance(config),'status':'PASS','dataset':dataset,'arm':arm,'seed':seed,
        'frozen_training_commit':FROZEN_COMMIT,'identical_source_files':evidence,
        'metric_kernel_sha256':sha256('FMT_Utils/FlowMapAuditMetrics_3D.py'),'records':records,
        'scope':'Same saved predictions and truth; no training or parameter selection. All pairs retained.'}
    target=root/'runtime_recovery_checks'/f'{dataset}_{arm}_seed{seed}.json'
    target.parent.mkdir(exist_ok=True)
    write_json(target,report)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--check-only',action='store_true')
    p.add_argument('--event',choices=['RUNNING','COMPLETED','FAILED']);p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text());spec['output_root']=str(Path(a.root).resolve())
    spec['_config_sha256']=sha256(a.config)
    phase='runtime_recovery_check' if a.check_only else 'runtime_recovery_train'
    if a.event:
        event(spec,a.config,phase,a.event,a.exit_code);return
    evidence=validate_source()
    if a.check_only:check_existing(spec,a.config,a.index,evidence)
    else:
        dataset,arm,seed=original.matrix(spec)[a.index]
        check=Path(spec['output_root'])/'runtime_recovery_checks'/f'{dataset}_{arm}_seed{seed}.json'
        report=json.loads(check.read_text());assert report['status']=='PASS'
        assert report['config_sha256']==sha256(a.config) and report['identical_source_files']==evidence
        assert report['metric_kernel_sha256']==sha256('FMT_Utils/FlowMapAuditMetrics_3D.py')
        original.measured_trajectories=measured_trajectories
        original.run(spec,a.config,dataset,arm,seed)


if __name__=='__main__':main()
