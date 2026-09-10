"""Short real-data GPU fit/throughput check while full diagnostics wait in Slurm."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from FMT_Utils.FlowMapData_3D import sha256,write_json,trajectory_metrics
from FMT_Utils.FlowMapFit_3D import fit_direct,predict_direct
from experiments.Build_Task678_FlowMap_1_1 import provenance
from experiments.Build_Task678_DirectFMTFit_1_1 import DEFAULT_CONFIG
from experiments.Run_Task678_DirectFMTFit_1_1 import load_data,arrays_for
from experiments.Run_Task678_FlowMap_1_1 import truth_radius


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    root=Path(a.output_root)
    if root.exists():
        raise FileExistsError('Keep earlier throughput checks')
    root.mkdir(parents=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    spec=json.loads(Path(DEFAULT_CONFIG).read_text())
    spec.update(output_root=a.source_root,_config_sha256=sha256(DEFAULT_CONFIG))
    settings=dict(spec['conditions']['memorize16'],optimizer_steps=a.steps,probe_every=500)
    common={**provenance(DEFAULT_CONFIG),'experiment':'Verify_Task678_DirectFMTGPU_1.1',
        'script_sha256':sha256(__file__),'arguments':vars(a),'device':torch.cuda.get_device_name(0),
        'dataset':'cylinder3d','settings':settings,'scope':'Real training data only; short GPU throughput/fit check, not a held-out result'}
    write_json(root/'started.json',common)
    data,source=load_data(spec,'cylinder3d','train',settings)
    records=[]
    for task in ('Task6','Task7'):
        arrays=arrays_for(data,task)
        def progress(item):
            with (root/f'{task}_learning_curve.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(item)+'\n')
            print(task,json.dumps(item),flush=True)
        model,info=fit_direct(arrays,settings,9100+(6 if task=='Task6' else 7),'cuda',progress)
        prediction=np.asarray(predict_direct(model,arrays,'cuda'),np.float32)
        truth,radius=truth_radius(data,task)
        metrics=trajectory_metrics(prediction,truth,radius)
        np.savez_compressed(root/f'{task}_train_predictions.npz',prediction=prediction)
        value={**common,**info,'task':task,'metrics':metrics,'source_records':source,
            'mean_seconds_per_update':info['fit_seconds']/info['optimizer_steps'],'checkpoint_files':0}
        write_json(root/f'{task}_result.json',value);records.append(value)
        print('REAL TRAIN FIT',task,info['initial_train_probe_nrmse'],metrics['position_nrmse'],flush=True)
        del model
    write_json(root/'completed.json',{**common,'status':'PASS','completed':provenance(DEFAULT_CONFIG),
        'results':[{k:r[k] for k in ('task','initial_train_probe_nrmse','metrics','mean_seconds_per_update')} for r in records],
        'checkpoint_files':0})

if __name__=='__main__':
    main()
