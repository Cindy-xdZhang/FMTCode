"""Frozen-method x2 extension and common x2/x4/x8 interpolation comparison."""
from __future__ import annotations

from FMT_Utils.FMTNoConvolution_1_1 import reject_retired_fmt, assert_no_fmt_convolution
import argparse
import ast
import csv
import itertools
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from experiments import FTLE_P35_Final_2_1 as previous
from experiments import FTLE_P35_Fusion_2_1 as dev
from FMT_Utils.FTLE_Data_2D import file_sha256, interpolation
from FMT_Utils.FTLE_Temporal_Audit_2D import certify
from FMT_Utils.FTLE_Fusion_2D_2_1 import FusionSR

read,dump,stamp=dev.read,dev.dump,dev.stamp
test_data=previous.test_data
definitions=previous.definitions
EXTRA=['experiments/FTLE_P35_Scales_2_2.py','experiments/Audit_FTLE_P35_Scales_2_2.py',
       'ibex_bash/ftle_p35_scales_2p2.sh','config/Other_FTLEP35Fusion_2.2.json']

def provenance(config):
    result=previous.provenance(config)
    result['source_sha256'].update({p:file_sha256(p) for p in EXTRA})
    return result

def development_spec(spec):
    result=read(spec['development_config'])
    result.update(version=spec['version'],output=spec['output'],scales=spec['scales'])
    return result

def matrix(spec):
    development=development_spec(spec)
    return list(itertools.product(development['flows'],development['scales'],spec['seeds']))

def lock(spec,config):
    assert spec['scales']==[2] and spec['report_scales']==[2,4,8]
    root=Path(spec['output']);old=Path(spec['previous_final']);development=development_spec(spec)
    assert not (root/'lock.json').exists()
    old_lock=read(old/'lock.json');audit=read(old/'audit.json');summary=read(old/'summary.json')
    assert audit['status']=='PASS' and audit['trainings']==144
    assert old_lock['selected']==audit['selected']==summary['selected']
    assert old_lock['selected']['id']=='pyramid_p35'
    assert spec['seeds']==old_lock['seeds']==development['final_seeds']
    assert old_lock['definitions']==definitions(old_lock['selected'])
    # All scientific inputs and original functions must still be byte-identical.
    for name,expected in old_lock['provenance']['source_sha256'].items():
        assert file_sha256(name)==expected,('Frozen source changed',name)
    files={str(old/name):file_sha256(old/name) for name in ('lock.json','audit.json','summary.json','summary.csv','per_run.csv')}
    manifests={}
    for flow in development['flows']:
        path=Path(development['source_output'])/'data'/flow/'manifest.json'
        manifest=read(path)
        assert certify(manifest['source'],manifest['time_splits'],development['tau'])==manifest['temporal_certificate']
        assert {r['scale'] for r in manifest['records']}=={2,4,8}
        manifests[str(path)]=file_sha256(path)
    dump(root/'lock.json',{'version':spec['version'],'selected':old_lock['selected'],
         'definitions':old_lock['definitions'],'seeds':spec['seeds'],'locked_at':stamp(),
         'previous_files':files,'source_manifests':manifests,'provenance':provenance(config),
         'policy':spec['policy'],'test_opened_by_lock':False})
    print('Frozen six-method x2 extension locked',flush=True)

def locked(spec):
    record=read(Path(spec['output'])/'lock.json')
    for path,expected in (record['previous_files']|record['source_manifests']).items():
        assert file_sha256(path)==expected,path
    assert spec['seeds']==record['seeds']
    assert record['definitions']==definitions(record['selected'])
    for path,expected in record['provenance']['source_sha256'].items():assert file_sha256(path)==expected,path
    return development_spec(spec),record

def preflight(spec,config):
    development,record=locked(spec)
    from experiments.Audit_FTLE_P35_Fusion_2_1 import preflight as scientific_preflight
    scientific_preflight(development,config)
    # Confirm copied orchestration has not changed the frozen fitting/evaluation path.
    trees=[ast.parse(Path(p).read_text()) for p in ('experiments/FTLE_P35_Final_2_1.py',__file__)]
    for name in ('infer','inference_time','train','merge'):
        nodes=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
        assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False),name
    device='cuda'
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)==spec['expected_device']
    dev.initialize(102)
    low=torch.randn(2,1,17,33,device=device);valid=torch.ones_like(low)
    geometry=torch.randn(2,321,17,33,device=device)
    bicubic=torch.randn(2,1,33,65,device=device)
    counts={}
    for definition in record['definitions']:
        if definition.get('reference'):
            model=dev.old.Model(read(development['reference_config']),definition['id'],2,0,0.).to(device)
            out=model(low)
        else:
            model=FusionSR(2,'pyramid',development['width']).to(device)
            with torch.no_grad():model.head.weight.normal_(0,.01)
            out=model(low,geometry,valid,bicubic,low)
            torch.testing.assert_close(out[...,::2,::2],low,atol=0,rtol=0)
        assert out.shape==(2,1,33,65)
        out.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        counts[definition['id']]=sum(p.numel() for p in model.parameters())
    dump(Path(spec['output'])/'gpu_preflight.json',{'status':'PASS','parameters':counts,
         'frozen_function_AST':'PASS','validation_or_test_read':False,'provenance':provenance(config)})
    print('x2 frozen-function and GPU checks PASS',flush=True)

def prepare(spec,config,index,device):
    development,_=locked(spec)
    dev.prepare(development,config,index,device)

def audit_data(spec,config):
    from experiments.Audit_FTLE_P35_Fusion_2_1 import audit_data as check
    development,_=locked(spec);check(development,config)

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


def interpolate_all(spec,config):
    development,_=locked(spec);root=Path(spec['output']);rows=[]
    assert read(root/'audit.json')['status']=='PASS'
    for flow in development['flows']:
        source=Path(development['source_output'])/'data'/flow;manifest=read(source/'manifest.json')
        stats=dev.scalar_stats(dev.load_data(development,flow,2,'train'))
        for split in ('validation','test'):
            for scale in spec['report_scales']:
                entries=[r for r in manifest['records'] if r['split']==split and r['scale']==scale]
                for row in entries:
                    path=source/row['file'];assert file_sha256(path)==row['sha256']
                    with np.load(path) as ds:
                        for method,order in [('bilinear',1),('bicubic',3)]:
                            prediction=interpolation(ds['low'],scale,order)
                            folder=root/'interpolation'/flow/method;folder.mkdir(parents=True,exist_ok=True)
                            output=folder/row['file']
                            np.savez_compressed(output,prediction=prediction,truth=ds['high'],mask=ds['mask'])
                            rows.append({'flow':flow,'scale':scale,'split':split,'method':method,'file':row['file'],
                                'source_sha256':row['sha256'],'prediction_sha256':file_sha256(output),
                                'data_range':stats['data_range'],**dev.old.metrics(ds['high'],prediction,ds['mask'],stats['data_range'])})
    assert len(rows)==144
    dump(root/'interpolation.json',{'rows':rows,'provenance':provenance(config)})
    print('Interpolation predictions complete',len(rows),flush=True)

def combined(spec,config):
    _,record=locked(spec);root=Path(spec['output']);old=Path(spec['previous_final'])
    summaries=[];rows=[]
    for folder in (root,old):
        assert read(folder/'audit.json')['status']=='PASS'
        data=read(folder/'summary.json')
        summaries.extend(r|{'source_version':data['version']} for r in data['summary'] if r['split']=='test')
        rows.extend(r|{'source_version':data['version']} for r in data['rows'] if r['split']=='test')
    interpolated=read(root/'interpolation.json')['rows'];development=development_spec(spec)
    for method,scale in itertools.product(('bilinear','bicubic'),spec['report_scales']):
        for flow in development['flows']+['macro']:
            group=[r for r in interpolated if r['method']==method and r['scale']==scale and r['split']=='test'
                   and (flow=='macro' or r['flow']==flow)]
            value={'flow':flow,'scale':scale,'method':method,'split':'test','source_version':spec['version'],'parameters':0}
            for key in ('mse','rmse','mae','psnr','ssim'):
                per_flow=[np.mean([r[key] for r in group if r['flow']==f]) for f in sorted({r['flow'] for r in group})]
                value[key+'_mean']=float(np.mean(per_flow));value[key+'_std']=None
            summaries.append(value)
    assert len(summaries)==120
    dump(root/'combined.json',{'version':spec['version'],'summary':summaries,'learned_per_run':rows,
         'interpolation_rows':interpolated,'provenance':provenance(config),
         'sources':{str(p):file_sha256(p) for p in [root/'summary.json',root/'audit.json',old/'summary.json',old/'audit.json',root/'interpolation.json']}})
    keys=['flow','scale','method','source_version','parameters']+[m+s for m in ('psnr','rmse','ssim','mse','mae') for s in ('_mean','_std')]
    with (root/'combined.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    print('Combined table complete',len(summaries),'rows',flush=True)

def report(spec,config):
    root=Path(spec['output']);assert read(root/'combined_audit.json')['status']=='PASS'
    data=read(root/'combined.json');summaries=data['summary']
    names={'bilinear':'双线性插值（bilinear）','bicubic':'双三次插值（bicubic）','espcn':'ESPCN',
           'unet':'U-Net','pyramid_none':'融合网络：仅低 FTLE','pyramid_raw':'融合网络：Raw 路径线',
           'pyramid_p35':'融合网络：p35','pyramid_signed':'融合网络：保留线身份的傅里叶系数'}
    lines=['# FTLE 2× / 4× / 8× 完整比较（Other_FTLEP35Fusion_2.2）','',
           '2×：本轮72次训练；4×/8×：复用2.1已核验144次训练。六方法、四流场、同三个种子98211–98213。',
           '全部使用未参与拟合和选轮的原测试时间块；这是已使用的benchmark，不是新的独立确认集。三倍率真值和评价区域相同。',
           '学习方法为种子均值±样本标准差；每种子先平均三切片，再平均四流场。插值没有优化随机性，只报确定性均值。',
           'PSNR（峰值信噪比，dB）和SSIM（结构相似性）越高越好；RMSE（均方根误差，FTLE物理单位）越低越好。',
           'ESPCN为在低网格卷积后进行像素重排的超分辨网络；U-Net为带跳跃连接的编码器—解码器。',
           '融合网络四组共享结构；仅低FTLE组的几何输入为零，Raw使用归一化的五线坐标，p35使用傅里叶后逐特征均值/最大值，最后一组保留各线复系数身份。',
           '本轮按2.1保持dropout=0；原1.2的dropout对照及旧FMT编码方法仍保留在历史表。',
           '插值沿用原SciPy节点重合三次样条/双线性定义，nearest边界，无输出裁剪。','']
    def table(flow,metric):
        lines.extend([f'## {flow} · {metric.upper()}','', '| 方法 | 2× | 4× | 8× |','|---|---:|---:|---:|'])
        for method,label in names.items():
            cells=[]
            for scale in spec['report_scales']:
                row=next(r for r in summaries if r['flow']==flow and r['method']==method and r['scale']==scale)
                mean,std=row[metric+'_mean'],row[metric+'_std'];digits=3 if metric=='psnr' else 5
                cells.append(f'{mean:.{digits}f}'+('' if std is None else f' ± {std:.{digits}f}'))
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines.append('')
    for metric in ('psnr','rmse','ssim'):table('macro',metric)
    for flow in development_spec(spec)['flows']:
        for metric in ('psnr','rmse','ssim'):table(flow,metric)
    lines.extend(['## 来源与完整记录','',
                  f"本轮科学commit：`{data['provenance']['commit']}`；冻结2.1科学commit：`f0993a58fe744de94991341ee118594e832ba2ed`。",
                  '机器可读总表：combined.csv / combined.json；2×逐种子：per_run.csv；插值逐切片：interpolation.json。',
                  'audit.json复算72次训练432份预测；combined_audit.json复算144份插值并核验复用来源及总表。',
                  '时间/源帧隔离证据保留于data/*/manifest.json；模型选择与test打开时间见runs/*；无模型文件。',
                  '方法结论见docs/experiment_log.md本实验条目。',''])
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def submit(spec,config):
    reject_retired_fmt("Retired P35 convolutional FTLE fusion")
    locked(spec);root=Path(spec['output']);(root/'slurm').mkdir(parents=True,exist_ok=True)
    assert not (root/'submission.json').exists();jobs=[]
    def dispatch(phase,count=1,gpu=False,dependency=None,minutes=60):
        command=['sbatch','--parsable','--cpus-per-task=4','--mem=32G','--time='+str(minutes),
                 '--job-name=ftleP35-2p2-'+phase,'--output='+str(root/'slurm'/(phase+'-%A_%a.log'))]
        if count>1:command+=['--array',f'0-{count-1}%8']
        if gpu:command+=['--gres=gpu:1','--constraint=gtx1080ti']
        if dependency:command+=['--dependency=afterok:'+dependency]
        command+=['ibex_bash/ftle_p35_scales_2p2.sh',phase,config]
        submitted=stamp();job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row={'job_id':job,'phase':phase,'count':count,'version':spec['version'],'submitted_utc':submitted,
             'config':config,'config_sha256':file_sha256(config),'commit':dev.old.git_commit(),
             'expected_device':spec['expected_device'] if gpu else 'CPU','command':command}
        jobs.append(row);dump(root/'submission.json',jobs)
        with open('docs/ibex_run_registry.md','a',encoding='utf-8') as f:f.write('\n\nFTLE 2.2 submission: `'+str(row)+'`\n')
        print(row,flush=True);return job
    pre=dispatch('preflight',gpu=True)
    prep=dispatch('prepare',4,dependency=pre)
    checked=dispatch('audit-data',dependency=prep)
    fit=dispatch('train',12,gpu=True,dependency=checked)
    merged=dispatch('merge',dependency=fit)
    audited=dispatch('audit',dependency=merged)
    interp=dispatch('interpolate',dependency=audited)
    combined_job=dispatch('combine',dependency=interp)
    combined_check=dispatch('audit-combined',dependency=combined_job)
    dispatch('report',dependency=combined_check)

def main():
    reject_retired_fmt("Retired P35 convolutional FTLE fusion")
    parser=argparse.ArgumentParser();parser.add_argument('phase')
    parser.add_argument('--config',default='config/Other_FTLEP35Fusion_2.2.json')
    parser.add_argument('--index',type=int,default=0);parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--runtime-phase');parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=read(args.config);torch.set_num_threads(4)
    if args.phase=='runtime':runtime(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    elif args.phase=='prepare':prepare(spec,args.config,args.index,args.device)
    elif args.phase=='train':
        assert torch.cuda.get_device_name(0)==spec['expected_device']
        train(spec,args.config,args.index,args.device)
    else:
        from experiments.Audit_FTLE_P35_Scales_2_2 import audit,audit_combined
        {'lock':lock,'submit':submit,'preflight':preflight,'audit-data':audit_data,'merge':merge,
         'audit':audit,'interpolate':interpolate_all,'combine':combined,'audit-combined':audit_combined,'report':report}[args.phase](spec,args.config)

if __name__=='__main__':main()
