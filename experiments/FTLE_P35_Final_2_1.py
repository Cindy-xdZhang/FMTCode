"""Locked three-seed evaluation; all fitting finishes before opening test paths."""
from __future__ import annotations
import argparse
import csv
import itertools
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from experiments import FTLE_P35_Fusion_2_1 as dev
from FMT_Utils.FTLE_Data_2D import file_sha256,interpolation
from FMT_Utils.FTLE_P35_2D_2_1 import encode,WIDTHS

read,dump,stamp=dev.read,dev.dump,dev.stamp
EXTRA=['experiments/FTLE_P35_Final_2_1.py','experiments/Audit_FTLE_P35_Final_2_1.py',
       'config/Other_FTLEP35Fusion_2.1_final.json','ibex_bash/ftle_p35_final_2p1.sh']


def provenance(config):
    result=dev.provenance(config)
    result['source_sha256'].update({p:file_sha256(p) for p in EXTRA})
    return result


def definitions(selected):
    result=[{'id':m,'reference':True} for m in ('espcn','unet')]
    arch=selected['architecture']
    for feature in dict.fromkeys(['none','raw','p35','signed',selected['feature']]):
        result.append({'id':f'{arch}_{feature}','architecture':arch,'feature':feature})
    return result


def lock(spec,config):
    development=read(spec['development_config']);source=Path(development['output'])
    choice=read(source/'selection.json');audit=read(source/'result_audit.json')
    assert choice['gate_passed'] is True and choice['test_read'] is False
    assert audit['status']=='PASS' and audit['gate_passed'] is True and audit['test_read'] is False
    assert audit['selected']==choice['selected']
    assert choice['provenance']['commit']==spec['development_commit']
    assert choice['provenance']['config_sha256']==file_sha256(spec['development_config'])
    assert spec['seeds']==development['final_seeds']
    path=Path(spec['output'])/'lock.json'
    if path.exists():raise FileExistsError(path)
    record={'version':spec['version'],'selected':choice['selected'],'definitions':definitions(choice['selected']),
            'selection_sha256':file_sha256(source/'selection.json'),
            'development_audit_sha256':file_sha256(source/'result_audit.json'),
            'development_config_sha256':file_sha256(spec['development_config']),
            'seeds':spec['seeds'],'locked_at':stamp(),'provenance':provenance(config)}
    dump(path,record);print('Validation choice locked; test remains unopened',flush=True)


def locked(spec):
    record=read(Path(spec['output'])/'lock.json')
    development=read(spec['development_config']);root=Path(development['output'])
    assert file_sha256(root/'selection.json')==record['selection_sha256']
    assert file_sha256(root/'result_audit.json')==record['development_audit_sha256']
    choice=read(root/'selection.json');audit=read(root/'result_audit.json')
    assert choice['gate_passed'] is True and choice['test_read'] is False
    assert audit['status']=='PASS' and audit['gate_passed'] is True and audit['test_read'] is False
    assert record['selected']==choice['selected']==audit['selected']
    assert file_sha256(spec['development_config'])==record['development_config_sha256']
    assert definitions(record['selected'])==record['definitions']
    assert spec['seeds']==record['seeds']
    return development,record


def matrix(spec):
    development=read(spec['development_config'])
    return list(itertools.product(development['flows'],development['scales'],spec['seeds']))


def test_data(development,flow,scale,device):
    root=Path(development['source_output'])/'data'/flow
    manifest=read(root/'manifest.json');result=[];records=[]
    started=time.monotonic()
    for row in manifest['records']:
        if row['split']!='test' or row['scale']!=scale:continue
        path=root/row['file'];assert file_sha256(path)==row['sha256']
        with np.load(path) as ds:
            data={k:ds[k] for k in ('low','high','valid_high','valid_low','mask','times','xs','ys')}
            valid=data['valid_low'].ravel()
            parts=[encode(p) for p in torch.from_numpy(ds['paths_low'][valid]).to(device).split(1024)]
            for feature in WIDTHS:
                if feature=='none':continue
                full=np.zeros((len(valid),WIDTHS[feature]),np.float32)
                full[valid]=torch.cat([p[feature] for p in parts]).cpu().numpy()
                data[feature]=full.reshape(*data['low'].shape,-1).transpose(2,0,1)
            data['bicubic']=interpolation(data['low'],scale,3).astype(np.float32)
        result.append(data|{'file':row['file']});records.append(row)
    assert len(result)==development['split_counts']['test']
    return result,{'source_manifest_sha256':file_sha256(root/'manifest.json'),'records':records,
                   'temporal_certificate':manifest['temporal_certificate'],
                   'test_encoding_io_seconds':time.monotonic()-started}


def infer(model,definition,data,stats,device):
    model=model.to(device)
    if definition.get('reference'):
        return dev.old.predict(model,dev.old.normalized(data,definition['id'],stats),stats,device)
    normalized=dev.to_device(dev.normalize(data,definition['feature'],stats),device)
    return dev.predict(model,normalized,stats)


@torch.no_grad()
def inference_time(model,definition,data,stats,device):
    # Validation data only. Timing includes transfer and normalization identically for all arms.
    infer(model,definition,data,stats,device)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    start=time.monotonic()
    for _ in range(5):infer(model,definition,data,stats,device)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    return (time.monotonic()-start)/(5*len(data))


def train(spec,config,index,device):
    development,record=locked(spec);flow,scale,seed=matrix(spec)[index]
    root=Path(spec['output'])/'runs'/f'{flow}_x{scale}_s{seed}'
    if (root/'run.json').exists():raise FileExistsError(root)
    tr=dev.load_data(development,flow,scale,'train')
    va=dev.load_data(development,flow,scale,'validation')
    fitted=[]
    for definition in record['definitions']:
        started=stamp()
        if definition.get('reference'):
            model,stats,predictions,info=dev.fit_reference(development,definition['id'],scale,seed,tr,va,device)
        else:
            model,stats,predictions,info=dev.fit_fusion(development,definition,scale,seed,tr,va,device)
        ended=stamp()
        seconds=inference_time(model,definition,va,stats,device)
        fitted.append((definition,model.cpu(),stats,predictions,info,started,ended,seconds))
        print(flow,scale,seed,definition['id'],'validation selected',info['best_validation_mse'],flush=True)
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # No test NPZ has been opened above; every network and stopping point is now fixed.
    test_opened=stamp();te,source=test_data(development,flow,scale,device)
    for definition,model,stats,val_predictions,info,started,ended,seconds in fitted:
        predictions=infer(model,definition,te,stats,device)
        folder=root/definition['id'];folder.mkdir(parents=True,exist_ok=True)
        dump(folder/'history.json',info.pop('history'));metrics=[]
        for split,data,values in [('validation',va,val_predictions),('test',te,predictions)]:
            for d,p in zip(data,values):
                np.savez_compressed(folder/d['file'],prediction=p,truth=d['high'],mask=d['mask'])
                metrics.append({'split':split,'file':d['file'],**dev.old.metrics(d['high'],p,d['mask'],stats['data_range'])})
        dump(folder/'result.json',{'version':spec['version'],'flow':flow,'scale':scale,'seed':seed,
              'definition':definition,'normalization':stats,'metrics':metrics,'fit_started':started,'fit_ended':ended,
              'test_opened':test_opened,'validation_inference_seconds_per_slice':seconds,
              'lock_sha256':file_sha256(Path(spec['output'])/'lock.json'),'provenance':provenance(config),**info})
        model.cpu()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    dump(root/'run.json',{'flow':flow,'scale':scale,'seed':seed,'test_opened':test_opened,
                         'definitions':record['definitions'],'source':source,'provenance':provenance(config)})


def merge(spec,config):
    development,record=locked(spec);rows=[]
    for flow,scale,seed in matrix(spec):
        root=Path(spec['output'])/'runs'/f'{flow}_x{scale}_s{seed}'
        assert (root/'run.json').exists()
        for definition in record['definitions']:
            result=read(root/definition['id']/'result.json')
            for split in ('validation','test'):
                group=[m for m in result['metrics'] if m['split']==split]
                rows.append({'flow':flow,'scale':scale,'seed':seed,'method':definition['id'],'split':split,
                             **{key:float(np.mean([r[key] for r in group])) for key in ('mse','rmse','mae','psnr','ssim')},
                             'parameters':result['parameters'],'training_seconds':result['seconds'],
                             'inference_seconds_per_slice':result['validation_inference_seconds_per_slice']})
    root=Path(spec['output']);summaries=[]
    for split,scale,definition in itertools.product(('validation','test'),development['scales'],record['definitions']):
        for flow in development['flows']+['macro']:
            group=[r for r in rows if r['split']==split and r['scale']==scale and r['method']==definition['id']
                   and (flow=='macro' or r['flow']==flow)]
            summary={'flow':flow,'scale':scale,'method':definition['id'],'split':split}
            for metric in ('mse','rmse','mae','psnr','ssim','training_seconds','inference_seconds_per_slice'):
                means=[float(np.mean([r[metric] for r in group if r['seed']==s])) for s in spec['seeds']]
                summary.update({metric+'_mean':float(np.mean(means)),metric+'_std':float(np.std(means,ddof=1))})
            summary['parameters']=group[0]['parameters'];summaries.append(summary)
    for name,values in [('per_run.csv',rows),('summary.csv',summaries)]:
        with (root/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    selected=record['selected']['id'];comparisons=[]
    for scale in development['scales']:
        selected_rows=[r for r in rows if r['method']==selected and r['scale']==scale and r['split']=='test']
        for definition in record['definitions']:
            if definition['id']==selected:continue
            paired=[]
            for r in selected_rows:
                baseline=next(x for x in rows if x['flow']==r['flow'] and x['scale']==scale and x['seed']==r['seed']
                              and x['method']==definition['id'] and x['split']=='test')
                paired.append({'flow':r['flow'],'seed':r['seed'],'psnr_gain':r['psnr']-baseline['psnr'],
                               'mse_relative_change':r['mse']/baseline['mse']-1})
            comparisons.append({'scale':scale,'baseline':definition['id'],'mean_psnr_gain':float(np.mean([p['psnr_gain'] for p in paired])),
                                'paired':paired})
    dump(root/'summary.json',{'version':spec['version'],'selected':record['selected'],'rows':rows,'summary':summaries,
                            'test_comparisons':comparisons,'provenance':provenance(config)})
    print('Final metrics merged',len(rows),'rows',flush=True)


def runtime(spec,config,phase,state,code):
    value={'phase':phase,'state':state,'exit_code':code,**provenance(config)}
    dump(Path(spec['output'])/'runtime'/f"{value['job_id']}_{value['array_index']}_{phase}_{state}.json",value)


def submit(spec,config):
    _,record=locked(spec);root=Path(spec['output']);(root/'slurm').mkdir(parents=True,exist_ok=True)
    if (root/'submission.json').exists():raise FileExistsError(root/'submission.json')
    jobs=[]
    def dispatch(phase,count,gpu=False,dependency=None):
        command=['sbatch','--parsable','--cpus-per-task=4','--mem=48G','--time=02:00:00',
                 '--job-name=ftleP35-final-'+phase,'--output='+str(root/'slurm'/(phase+'-%A_%a.log'))]
        if count>1:command+=['--array',f'0-{count-1}%8']
        if gpu:command+=['--gres=gpu:1','--constraint=v100']
        if dependency:command+=['--dependency=afterok:'+dependency]
        command+=['ibex_bash/ftle_p35_final_2p1.sh',phase,config]
        submitted=stamp();job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row={'job_id':job,'phase':phase,'count':count,'version':spec['version'],'submitted_utc':submitted,
             'config':config,'config_sha256':file_sha256(config),'commit':dev.old.git_commit(),
             'expected_device':'V100' if gpu else 'CPU','command':command}
        jobs.append(row);dump(root/'submission.json',jobs)
        with open('docs/ibex_run_registry.md','a',encoding='utf-8') as f:
            f.write('\n\nFTLE P35 final submission: `'+str(row)+'`\n')
        print(row,flush=True);return job
    check=dispatch('preflight',1)
    fit=dispatch('train',len(matrix(spec)),True,check)
    merged=dispatch('merge',1,dependency=fit)
    dispatch('audit',1,dependency=merged)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase')
    parser.add_argument('--config',default='config/Other_FTLEP35Fusion_2.1_final.json')
    parser.add_argument('--index',type=int,default=0);parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--runtime-phase');parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=read(args.config);torch.set_num_threads(4)
    if args.phase=='runtime':runtime(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    elif args.phase=='lock':lock(spec,args.config)
    elif args.phase=='submit':submit(spec,args.config)
    elif args.phase=='train':train(spec,args.config,args.index,args.device)
    elif args.phase=='merge':merge(spec,args.config)
    else:
        from experiments.Audit_FTLE_P35_Final_2_1 import preflight,audit
        {'preflight':preflight,'audit':audit}[args.phase](spec,args.config)


if __name__=='__main__':main()
