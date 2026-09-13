"""Implementation checks on training fixtures, never model selection."""
import json
from pathlib import Path
import time
import torch
import numpy as np
from DeepUtils.utils import EasyConfig
from experiments.Verify_FMTAllV2_3D import records, features, objectivity_check, write_json, sha
from experiments.Verify_HighReVAE import _train

def main():
    config=Path('config/Verify_FMTAllV2_1.1.json')
    spec=json.loads(config.read_text())
    out=Path(spec['output_root'])/'preflight.json'
    if out.exists(): raise FileExistsError(out)
    # Two optimizer updates test the existing VAE interface, without flow labels.
    part=spec['task2']; source=EasyConfig()
    source.update({'task2':{'batch_size':256,'weight_decay':1e-5,'learning_rate':3e-4,'target_optimizer_steps':2}})
    rng=np.random.default_rng(7097)
    x=rng.normal(size=(32,231)).astype(np.float32)
    settings={**part['architecture'],'optimizer_steps':2}
    _train(x,x[:8],settings,source,100,torch.device('cpu'),return_model=True)
    checks=[]
    for task in ('Task1','Task5'):
        local=json.loads(json.dumps(spec));local[task.lower()]['train']=[0]
        for dataset in spec['datasets']:
            rows=records(local,task,dataset,'train')
            audit=objectivity_check(rows)
            old=features(rows,spec[task.lower()]['old_feature'],'cpu')
            new=features(rows,'fmt_all_v2','cpu')
            assert old.shape[1]==(189 if task=='Task1' else 268)
            assert new.shape==(old.shape[0],231)
            checks.append({'task':task,'dataset':dataset,'old_shape':old.shape,'new_shape':new.shape,**audit})
            print(checks[-1],flush=True)
    write_json(out,{'status':'PASS','config_sha256':sha(config),'checks':checks,'end_time':time.time(),
                    'confirmation_opened':False,'training_fixture_only':True})

if __name__=='__main__':main()
