"""Three-epoch same-GPU repeats with and without deterministic CUDA kernels."""
from pathlib import Path
import copy
import hashlib
import json
import os
import time
import numpy as np
import torch

from experiments import FMTv8_Search_2_1 as run


def main():
    spec=run.load_spec(run.DEFAULT_CONFIG);base,_=run.parents(spec)
    task,dataset='Task3','channel';seed=40;torch.set_num_threads(4)
    train=run.read_cache(spec,task,dataset,'train');val=run.read_cache(spec,task,dataset,'validation')
    data=[(r['raw'],np.zeros((len(r['labels']),433),np.float32),r['labels']) for r in (train,val)]
    tr,va,_,stats=run.old._normalize_train_only(*data,None)
    settings=dict(experiment='Verify_FMTv8_Repeatability_1.1',model=base['raw_model'],training=copy.deepcopy(base['training']),evaluation={'test_enabled':False})
    settings['training'].update(max_epochs=3,patience=3)
    root=Path(spec['output'])/'engineering_smoke/Verify_FMTv8_Repeatability_1.1';root.mkdir(parents=True,exist_ok=False)
    results=[];began=time.time()
    for deterministic in (False,True):
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=deterministic
        torch.use_deterministic_algorithms(deterministic)
        previous=None
        for repeat in (0,1):
            folder=root/f'deterministic{int(deterministic)}_repeat{repeat}'
            outcome=run.old.train_raw(settings,dataset,'raw',seed,(tr,va,None),stats,torch.device('cuda'),folder)
            state=torch.load(outcome['checkpoint'],map_location='cpu',weights_only=False)
            values={k:v.numpy() for k,v in state['state_dict'].items()}
            digest=hashlib.sha256(b''.join(k.encode()+values[k].tobytes() for k in sorted(values))).hexdigest()
            entry=dict(deterministic=deterministic,repeat=repeat,state_array_sha256=digest,outcome=outcome)
            if previous is not None:
                entry['state_exact']=all(np.array_equal(previous[k],values[k]) for k in values)
                entry['maximum_weight_difference']=max(float(np.max(np.abs(previous[k]-values[k]))) for k in values)
            previous=values;results.append(entry)
    run.cleanup_weights(spec,root)
    run.write(root/'diagnosis.json',dict(results=results,seconds=time.time()-began,
        torch=torch.__version__,cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(0),
        workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),script_sha256=run.old.sha(__file__),
        task=task,dataset=dataset,seed=seed,test_read=False))
    print(json.dumps([{k:v for k,v in r.items() if k!='outcome'} for r in results]),flush=True)


if __name__=='__main__':main()
