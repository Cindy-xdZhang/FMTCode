"""Recompute all train/held-out metrics from durable predictions and true trajectories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from FMT_Utils.FlowMapData_3D import trajectory_metrics,sha256,write_json
from experiments.Build_Task678_FlowMap_1_1 import provenance
from experiments.Build_Task678_DirectFMTFit_1_1 import DEFAULT_CONFIG
from experiments.Run_Task678_DirectFMTFit_1_1 import load_data,matrix
from experiments.Run_Task678_FlowMap_1_1 import truth_radius


def audit(spec,config):
    spec['_config_sha256']=sha256(config)
    root=Path(spec['output_root']);rows=[]
    for dataset,condition,seed in matrix(spec):
        settings=spec['conditions'][condition]
        directory=root/'runs'/dataset/condition/f'seed{seed}'
        done=json.loads((directory/'completed.json').read_text())
        assert done['status']=='PASS' and done['config_sha256']==spec['_config_sha256']
        assert done['frozen_encoder_sha256']==spec['frozen_encoder_sha256']
        roles=('train',) if settings['data']=='memorize' else ('train','validation','test')
        for role in roles:
            data,source=load_data(spec,dataset,role,settings if role=='train' else None)
            for task in ('Task6','Task7','Task8'):
                r=json.loads((directory/f'{task}_{role}.json').read_text())
                assert r['source_records']==source and r['input_dimension']==161 and r['projection']=='none'
                assert r['config_sha256']==spec['_config_sha256'] and r['optimizer_steps']<=settings['optimizer_steps']
                if settings.get('stop_train_nrmse') is None:
                    assert r['optimizer_steps']==settings['optimizer_steps']
                else:
                    assert r['optimizer_steps']==settings['optimizer_steps'] or r['final_train_probe_nrmse']<=settings['stop_train_nrmse']
                if task=='Task8':
                    assert r['second_query_uses']=='predicted_first_stage_endpoint'
                path=directory/r['prediction_file'];assert sha256(path)==r['prediction_sha256']
                with np.load(path,allow_pickle=False) as a:
                    truth,radius=truth_radius(data,task)
                    measured=trajectory_metrics(a['prediction'],truth,radius)
                for k,v in measured.items():
                    if isinstance(v,(float,int,list)):
                        np.testing.assert_allclose(v,r['metrics'][k],rtol=1e-10,atol=1e-10)
                rows.append(dict(dataset=dataset,condition=condition,seed=seed,task=task,role=role,
                    **{k:v for k,v in measured.items() if isinstance(v,(float,int))},
                    parameters=r['parameter_count'],steps=r['optimizer_steps'],initial_train_probe=r['initial_train_probe_nrmse'],
                    final_train_probe=r['final_train_probe_nrmse'],prediction_sha256=r['prediction_sha256']))
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth')) and not any(root.rglob('*.ckpt'))
    out=root/'metrics.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups={}
    for row in rows:
        key=(row['condition'],row['task'],row['role'])
        groups.setdefault(key,[]).append(row)
    summaries=[]
    for (condition,task,role),values in groups.items():
        per_dataset={d:float(np.mean([r['position_nrmse'] for r in values if r['dataset']==d])) for d in spec['datasets']}
        summaries.append(dict(condition=condition,task=task,role=role,per_dataset=per_dataset,
            dataset_macro_position_nrmse=float(np.mean(list(per_dataset.values())))))
    write_json(root/'summary.json',{**provenance(config),'status':'PASS','summary':summaries})
    write_json(root/'independent_audit.json',{**provenance(config),'status':'PASS','metric_records':len(rows),
        'checkpoint_files':0,'metrics_sha256':sha256(out),'summary_sha256':sha256(root/'summary.json')})
    print(f'FIT AUDIT PASS {len(rows)} records',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=DEFAULT_CONFIG);a=p.parse_args()
    audit(json.loads(Path(a.config).read_text()),a.config)

if __name__=='__main__':
    main()
