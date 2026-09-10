"""Train direct full-FMT networks, then separately report fit and held-out errors."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256,write_json,trajectory_metrics
from FMT_Utils.FlowMapModels_3D import training_arrays
from FMT_Utils.FlowMapFit_3D import fit_direct,predict_direct,compose_direct
from experiments.Build_Task678_FlowMap_1_1 import provenance
from experiments.Build_Task678_DirectFMTFit_1_1 import DEFAULT_CONFIG
from experiments.Run_Task678_FlowMap_1_1 import truth_radius


def load_data(spec,dataset,role,condition=None):
    base_only=role!='train' or condition['data']!='expanded'
    metadata=json.loads((Path(spec['output_root'])/'build'/f"{dataset}{'_base' if base_only else ''}.json").read_text())
    assert metadata['status']=='PASS' and metadata['config_sha256']==spec['_config_sha256']
    rows=[r for r in metadata['records'] if r['role']==role]
    if role=='train':
        if condition['data'] in ('base','memorize'):
            rows=[r for r in rows if r['replicate']==0]
        if condition['data']=='memorize':
            rows=rows[:1]
    data=[]
    source=[]
    for r in rows:
        path=Path(r['cache_file'])
        assert sha256(path)==r['cache_sha256'] and r['config_sha256']==spec['_config_sha256']
        with np.load(path,allow_pickle=False) as a:
            record={k:a[k] for k in a.files}
        if role=='train' and condition['data']=='memorize':
            count=condition['memorize_regions']
            assert len(record['origin0'])>=count
            record={k:v[:count] for k,v in record.items()}
        data.append(record)
        source.append({'ordinal':r['ordinal'],'replicate':r['replicate'],'sha256':r['cache_sha256'],
                       'regions_used':len(record['origin0'])})
    assert rows and (role=='train' or all(r['replicate']==0 for r in rows))
    combined={k:np.concatenate([d[k] for d in data]) for k in data[0]}
    return combined,source


def arrays_for(data,task):
    features={k:data[f'fmt__{k}'] for k in ('support0','support1','context')}
    result=training_arrays(data,features,task)
    assert result['tokens'].shape[-1]==161
    return result


def save_result(directory,task,role,prediction,data,source,common,info):
    prediction=np.asarray(prediction,np.float32)
    truth,radius=truth_radius(data,task)
    path=directory/f'{task}_{role}_predictions.npz'
    np.savez_compressed(path,prediction=prediction)
    result={**common,**info,'task':task,'role':role,'source_records':source,
        'metrics':trajectory_metrics(prediction,truth,radius),'prediction_file':path.name,
        'prediction_sha256':sha256(path),'checkpoint_files':0}
    write_json(directory/f'{task}_{role}.json',result)
    print(f"RESULT {common['dataset']}/{common['condition']}/{task}/{role}: {result['metrics']['position_nrmse']:.7g}",flush=True)
    return result


def run(spec,config,dataset,condition_name,seed,device='cuda'):
    spec['_config_sha256']=sha256(config)
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation is required')
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK','4')))
    torch.backends.cuda.matmul.allow_tf32=False
    condition=spec['conditions'][condition_name]
    directory=Path(spec['output_root'])/'runs'/dataset/condition_name/f'seed{seed}'
    if directory.exists():
        raise FileExistsError('Never overwrite an earlier fit attempt; version recovery explicitly')
    directory.mkdir(parents=True)
    common={**provenance(config),'experiment':spec['experiment'],'dataset':dataset,'condition':condition_name,
        'seed':seed,'device':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        'frozen_encoder_sha256':sha256('FMT_Utils/DFT_FMT_3D.py'),'settings':condition,
        'torch':torch.__version__,'numpy':np.__version__}
    assert common['frozen_encoder_sha256']==spec['frozen_encoder_sha256']
    write_json(directory/'started.json',common)
    try:
        train,train_source=load_data(spec,dataset,'train',condition)
        models={}
        for task in ('Task6','Task7'):
            arrays=arrays_for(train,task)
            def progress(item):
                with (directory/f'{task}_learning_curve.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(item)+'\n');f.flush()
                print(f"FIT {dataset}/{condition_name}/{task} step={item['step']} train_nrmse={item['train_position_nrmse']:.6g}",flush=True)
            model,info=fit_direct(arrays,condition,seed+(6 if task=='Task6' else 7),device,progress)
            models[task]=(model,info)
            write_json(directory/f'{task}_training.json',{**common,**info,'source_records':train_source})
        # Both networks finish before any held-out trajectories are opened.
        roles=('train',) if condition['data']=='memorize' else ('train','validation','test')
        count=0
        for role in roles:
            data,source=(train,train_source) if role=='train' else load_data(spec,dataset,role)
            for task in ('Task6','Task7'):
                model,info=models[task]
                arrays=arrays_for(data,task)
                pred=predict_direct(model,arrays,device)
                save_result(directory,task,role,pred,data,source,common,info);count+=1
                if task=='Task6':
                    pred,composition=compose_direct(model,arrays,device)
                    save_result(directory,'Task8',role,pred,data,source,common,{**info,**composition,
                        'training_source':'same in-memory Task6 network; no long-target optimization',
                        'train_role_meaning':'composition in observed training regions, not directly optimized long trajectories'});count+=1
        write_json(directory/'completed.json',{**common,'status':'PASS','metric_records':count,'checkpoint_files':0,
                                              'completed':provenance(config)})
    except Exception:
        write_json(directory/'failed.json',{**common,'status':'FAILED','traceback':traceback.format_exc()})
        raise


def matrix(spec):
    return [(d,c,s) for c in spec['conditions'] for d in spec['datasets'] for s in spec['seeds']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text());run(spec,a.config,*matrix(spec)[a.index])

if __name__=='__main__':
    main()
