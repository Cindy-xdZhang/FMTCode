"""Versioned development of five-line Fourier fusion for FTLE super-resolution."""
from __future__ import annotations

from FMT_Utils.FMTNoConvolution_1_1 import reject_retired_fmt, assert_no_fmt_convolution
import argparse
import itertools
import os
from pathlib import Path
import random
import socket
import subprocess
import time
import numpy as np
import torch

from FMT_Utils.FTLE_Data_2D import file_sha256,interpolation
from FMT_Utils.FTLE_Temporal_Audit_2D import certify
from FMT_Utils.FTLE_P35_2D_2_1 import encode,WIDTHS
from FMT_Utils.FTLE_Fusion_2D_2_1 import FusionSR
from experiments import FTLE_Upsampling_2D_1_2 as old

read,dump,stamp=old.read,old.dump,old.stamp
SOURCES=['experiments/FTLE_P35_Fusion_2_1.py','FMT_Utils/FTLE_P35_2D_2_1.py',
         'FMT_Utils/FTLE_Fusion_2D_2_1.py','experiments/FTLE_Upsampling_2D_1_2.py',
         'FMT_Utils/FTLE_Data_2D.py','FMT_Utils/FTLE_Temporal_Audit_2D.py','FMT_Utils/FTLE_Baselines_2D.py',
         'experiments/Audit_FTLE_P35_Fusion_2_1.py','experiments/Audit_FTLE_Upsampling_2D_1_2.py',
         'config/Other_FTLEUpsampling2D_1.2.json']


def provenance(config):
    return {'commit':old.git_commit(),'config_sha256':file_sha256(config),'timestamp':stamp(),
            'hostname':socket.gethostname(),'job_id':os.environ.get('SLURM_JOB_ID'),
            'array_index':os.environ.get('SLURM_ARRAY_TASK_ID'),'torch':torch.__version__,'numpy':np.__version__,
            'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'source_sha256':{p:file_sha256(p) for p in SOURCES}}


def candidates():
    return [{'id':f'{architecture}_{feature}','architecture':architecture,'feature':feature,
             'eligible':feature.startswith('p35')}
            for architecture,features in [('pyramid',list(WIDTHS)),('unet',['none','raw','p35_multitime'])]
            for feature in features]


def matrix(spec):return list(itertools.product(spec['flows'],spec['scales']))


def prepare(spec,config,index,device):
    flow=spec['flows'][index]
    source=Path(spec['source_output'])/'data'/flow
    original=read(source/'manifest.json')
    certificate=certify(original['source'],original['time_splits'],spec['tau'])
    assert certificate==original['temporal_certificate']
    root=Path(spec['output'])/'data'/flow
    if (root/'manifest.json').exists():raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in original['records']:
        if row['scale'] not in spec['scales'] or row['split']=='test':continue
        path=source/row['file']
        assert file_sha256(path)==row['sha256']
        with np.load(path) as ds:
            arrays={k:ds[k] for k in ('low','high','valid_high','valid_low','mask','times','xs','ys')}
            valid=arrays['valid_low'].ravel()
            paths=torch.from_numpy(ds['paths_low'][valid]).to(device)
            parts=[encode(x) for x in paths.split(1024)]
            for feature in WIDTHS:
                if feature=='none':continue
                values=torch.cat([p[feature] for p in parts]).cpu().numpy()
                full=np.zeros((len(valid),WIDTHS[feature]),np.float32)
                full[valid]=values
                arrays[feature]=full.reshape(*arrays['low'].shape,-1).transpose(2,0,1)
            arrays['bicubic']=interpolation(arrays['low'],row['scale'],3).astype(np.float32)
        dest=root/row['file']
        np.savez_compressed(dest,**arrays)
        records.append({**row,'source_sha256':row['sha256'],'sha256':file_sha256(dest)})
        print(flow,row['file'],flush=True)
    dump(root/'manifest.json',{'version':spec['version'],'provenance':provenance(config),
                              'source_manifest_sha256':file_sha256(source/'manifest.json'),
                              'temporal_certificate':certificate,'records':records})


def load_data(spec,flow,scale,split):
    if split=='test':raise ValueError('Development program cannot read test data')
    root=Path(spec['output'])/'data'/flow
    m=read(root/'manifest.json')
    result=[]
    for row in m['records']:
        if row['scale']==scale and row['split']==split:
            with np.load(root/row['file']) as ds:
                result.append({k:ds[k] for k in ds.files}|{'file':row['file']})
    assert len(result)==spec['split_counts'][split]
    return result


def scalar_stats(data):return old.normalization(data,'unet')


def transform(x):return np.sign(x)*np.log1p(abs(x))


def stats_for(data,feature):
    stats=scalar_stats(data)
    if feature!='none':
        values=np.concatenate([transform(d[feature][:,d['valid_low']]) for d in data],1).astype(np.float64)
        mean,std=values.mean(1),values.std(1)
        std[std<1e-8]=1.
        stats.update(feature_mean=mean.tolist(),feature_std=std.tolist())
    return stats


def normalize(data,feature,stats):
    out=[]
    for d in data:
        low=((d['low']-stats['low_mean'])/stats['low_std']).astype(np.float32)
        low[~d['valid_low']]=0
        geo=None
        if feature!='none':
            geo=np.clip((transform(d[feature])-np.asarray(stats['feature_mean'])[:,None,None])/
                         np.asarray(stats['feature_std'])[:,None,None],-8,8).astype(np.float32)
            geo[:,~d['valid_low']]=0
        out.append({'input':low[None],'geometry':geo,'valid':d['valid_low'][None].astype(np.float32),
                    'target':((d['high']-stats['target_mean'])/stats['target_std'])[None],
                    'bicubic':((d['bicubic']-stats['target_mean'])/stats['target_std'])[None],
                    'low_target':((d['low']-stats['target_mean'])/stats['target_std'])[None],
                    'mask':d['mask'][None].astype(np.float32),'file':d['file']})
    return out


def to_device(data,device):
    return [{k:torch.as_tensor(v,device=device,dtype=torch.float32) if isinstance(v,np.ndarray) else v
             for k,v in d.items()} for d in data]


def model_batch(model,data,ids):
    args=[]
    for key in ('input','geometry','valid','bicubic','low_target'):
        args.append(torch.stack([data[i][key] for i in ids]) if data[0][key] is not None else None)
    return model(*args)


@torch.no_grad()
def predict(model,data,stats):
    model.eval()
    return [(model_batch(model,data,[i])[0,0]*stats['target_std']+stats['target_mean']).cpu().numpy() for i in range(len(data))]


def initialize(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def fit_fusion(spec,definition,scale,seed,training,validation,device):
    initialize(seed)
    stats=stats_for(training,definition['feature'])
    tr=to_device(normalize(training,definition['feature'],stats),device)
    va=to_device(normalize(validation,definition['feature'],stats),device)
    model=FusionSR(scale,definition['architecture'],spec['width']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['training']['lr'],weight_decay=spec['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2,factor=.5)
    rng=np.random.default_rng(seed)
    start=time.monotonic()
    best=old.validation_mse(validation,predict(model,va,stats))
    selected=0
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    history=[{'step':0,'validation_mse':best,'train_mse':None,'lr':spec['training']['lr']}]
    for step in range(spec['training']['steps']):
        model.train()
        ids=rng.choice(len(tr),spec['training']['batch_size'],replace=False).tolist()
        optimizer.zero_grad(set_to_none=True)
        prediction=model_batch(model,tr,ids)
        target=torch.stack([tr[i]['target'] for i in ids]);mask=torch.stack([tr[i]['mask'] for i in ids])
        loss=(((prediction-target)**2)*mask).sum()/mask.sum()
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if (step+1)%spec['training']['validate_every']!=0:continue
        score=old.validation_mse(validation,predict(model,va,stats))
        history.append({'step':step+1,'validation_mse':score,'train_mse':float(loss.detach()),'lr':optimizer.param_groups[0]['lr']})
        if score<best:
            best,selected=score,step+1
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step(score)
        if step+1-selected>=spec['training']['patience_steps']:break
    model.load_state_dict(state)
    return model,stats,predict(model,va,stats),{'history':history,'best_step':selected,'best_validation_mse':best,
            'seconds':time.monotonic()-start,'parameters':sum(p.numel() for p in model.parameters())}


def fit_reference(spec,method,scale,seed,training,validation,device):
    frozen=read(spec['reference_config'])
    initialize(seed)
    stats=scalar_stats(training)
    tr,va=old.normalized(training,method,stats),old.normalized(validation,method,stats)
    indices,size=old.patch_indices(tr,frozen,scale)
    model=old.Model(frozen,method,scale,0,0.).to(device)
    cfg=frozen['training']
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=.5)
    rng=np.random.default_rng(seed)
    best,selected,state=float('inf'),-1,None
    history=[];start=time.monotonic()
    for epoch in range(cfg['epochs']):
        model.train();order=rng.permutation(len(indices));losses=[]
        for begin in range(0,len(order),cfg['batch_size']):
            subset=[indices[i] for i in order[begin:begin+cfg['batch_size']]]
            low,geo,target=old.batch(tr,subset,size,scale,device)
            optimizer.zero_grad(set_to_none=True)
            loss=(model(low,geo)-target).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite reference loss')
            loss.backward();optimizer.step();losses.append((float(loss.detach()),len(subset)))
        preds=old.predict(model,va,stats,device)
        score=old.validation_mse(va,preds)
        history.append({'epoch':epoch,'validation_mse':score,'train_mse':sum(x*n for x,n in losses)/sum(n for _,n in losses),
                        'lr':optimizer.param_groups[0]['lr']})
        if score<best:
            best,selected=score,epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step(score)
        if epoch-selected>=cfg['patience']:break
    model.load_state_dict(state)
    return model,stats,old.predict(model,va,stats,device),{'history':history,'best_epoch':selected,'best_validation_mse':best,
           'seconds':time.monotonic()-start,'parameters':sum(p.numel() for p in model.parameters()),'patch_count':len(indices)}


def write_result(root,spec,config,flow,scale,seed,definition,stats,predictions,data,info):
    root.mkdir(parents=True,exist_ok=True)
    dump(root/'history.json',info.pop('history'))
    rows=[]
    for d,p in zip(data,predictions):
        name=d['file']
        np.savez_compressed(root/name,prediction=p,truth=d['high'],mask=d['mask'])
        rows.append({'file':name,'split':'validation',**old.metrics(d['high'],p,d['mask'],stats['data_range'])})
    dump(root/'result.json',{'version':spec['version'],'definition':definition,'flow':flow,'scale':scale,'seed':seed,
                            'provenance':provenance(config),'normalization':stats,'metrics':rows,**info})


def search(spec,config,index,device):
    flow,scale=matrix(spec)[index];seed=spec['search_seed']
    tr,va=load_data(spec,flow,scale,'train'),load_data(spec,flow,scale,'validation')
    definitions=[{'id':m,'reference':True} for m in ('espcn','unet')]+candidates()
    for definition in definitions:
        root=Path(spec['output'])/'search'/f'{flow}_x{scale}'/definition['id']
        if (root/'result.json').exists():raise FileExistsError(root)
        if definition.get('reference'):
            model,stats,predictions,info=fit_reference(spec,definition['id'],scale,seed,tr,va,device)
        else:
            model,stats,predictions,info=fit_fusion(spec,definition,scale,seed,tr,va,device)
        write_result(root,spec,config,flow,scale,seed,definition,stats,predictions,va,info)
        print(flow,scale,definition['id'],info['best_validation_mse'],flush=True)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()


def select(spec,config):
    rows=[]
    for flow,scale in matrix(spec):
        for name in ['espcn','unet']+[d['id'] for d in candidates()]:
            result=read(Path(spec['output'])/'search'/f'{flow}_x{scale}'/name/'result.json')
            rows.append({'flow':flow,'scale':scale,'candidate':name,
                         'psnr':float(np.mean([r['psnr'] for r in result['metrics']])),
                         'mse':result['best_validation_mse'],'seconds':result['seconds'],'parameters':result['parameters']})
    summary=[]
    for definition in candidates():
        group=[r for r in rows if r['candidate']==definition['id']]
        scale_means={str(s):float(np.mean([r['psnr'] for r in group if r['scale']==s])) for s in spec['scales']}
        summary.append({**definition,'mean_psnr':float(np.mean([r['psnr'] for r in group])),'scale_psnr':scale_means})
    selected=max((r for r in summary if r['eligible']),key=lambda r:(r['mean_psnr'],r['id']))
    comparisons=[]
    for scale in spec['scales']:
        for comparator in ('espcn','unet',selected['architecture']+'_none'):
            mean=float(np.mean([r['psnr'] for r in rows if r['scale']==scale and r['candidate']==comparator]))
            comparisons.append({'scale':scale,'baseline':comparator,'baseline_psnr':mean,
                                'psnr_gain':selected['scale_psnr'][str(scale)]-mean})
    passed=all(r['psnr_gain']>0 for r in comparisons)
    dump(Path(spec['output'])/'selection.json',{'version':spec['version'],'provenance':provenance(config),
          'selection_split':'validation','selected':selected,'gate_passed':passed,'comparisons':comparisons,
          'candidates':summary,'rows':rows,'test_read':False})
    print({'selected':selected,'gate_passed':passed,'comparisons':comparisons},flush=True)


def runtime(spec,config,phase,state,code=None):
    root=Path(spec['output'])/'runtime'
    record={'phase':phase,'state':state,'exit_code':code,**provenance(config)}
    dump(root/f"{record['job_id']}_{record['array_index']}_{phase}_{state}.json",record)


def submit(spec,config):
    reject_retired_fmt("Retired P35 convolutional FTLE fusion")
    root=Path(spec['output']);(root/'slurm').mkdir(parents=True,exist_ok=True)
    jobs=[]
    def dispatch(phase,count,dependency=None,gpu=False):
        command=['sbatch','--parsable','--cpus-per-task=4','--mem=48G','--time=02:00:00',
                 '--job-name=ftleP35-'+phase,'--output='+str(root/'slurm'/(phase+'-%A_%a.log'))]
        if count>1:command+=['--array',f'0-{count-1}%8']
        if gpu:command+=['--gres=gpu:1','--constraint=v100']
        if dependency:command+=['--dependency=afterok:'+dependency]
        command+=['ibex_bash/ftle_p35_fusion_2p1.sh',phase,config]
        submitted=stamp();job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row={'job_id':job,'version':spec['version'],'phase':phase,'count':count,'submitted_utc':submitted,
             'config':config,'commit':old.git_commit(),'config_sha256':file_sha256(config),
             'expected_device':'V100' if gpu else 'CPU','command':command}
        jobs.append(row);dump(root/'submission.json',jobs)
        with open('docs/ibex_run_registry.md','a',encoding='utf-8') as f:f.write('\n\nFTLE P35 2.1 submission: `'+str(row)+'`\n')
        print(row,flush=True);return job
    pre=dispatch('preflight',1)
    prep=dispatch('prepare',len(spec['flows']),pre,True)
    check=dispatch('audit-data',1,prep)
    fit=dispatch('search',len(matrix(spec)),check,True)
    merged=dispatch('select',1,fit)
    dispatch('audit-results',1,merged)


def main():
    reject_retired_fmt("Retired P35 convolutional FTLE fusion")
    parser=argparse.ArgumentParser();parser.add_argument('phase')
    parser.add_argument('--config',default='config/Other_FTLEP35Fusion_2.1.json')
    parser.add_argument('--index',type=int,default=0);parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--runtime-phase');parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=read(args.config)
    torch.set_num_threads(4)
    if args.phase=='runtime':runtime(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    elif args.phase=='prepare':prepare(spec,args.config,args.index,args.device)
    elif args.phase=='search':search(spec,args.config,args.index,args.device)
    elif args.phase=='select':select(spec,args.config)
    elif args.phase=='submit':submit(spec,args.config)
    else:
        from experiments.Audit_FTLE_P35_Fusion_2_1 import preflight,audit_data,audit_results
        {'preflight':preflight,'audit-data':audit_data,'audit-results':audit_results}[args.phase](spec,args.config)


if __name__=='__main__':main()
