"""Recompute every held-out metric and verify model-selection/test boundaries."""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from experiments import FTLE_P35_Scales_2_2 as final
from experiments import FTLE_P35_Fusion_2_1 as dev
from experiments.Audit_FTLE_Upsampling_2D_1_2 import independent_metrics
from FMT_Utils.FTLE_Data_2D import file_sha256
from FMT_Utils.FTLE_Statistics_Audit_2_1 import audit_training_statistics


def audit(spec,config):
    development,record=final.locked(spec);root=Path(spec['output']);rows=[];prediction_count=0;statistics_checks=[]
    for flow,scale in dev.matrix(development):
        tr=dev.load_data(development,flow,scale,'train');va=dev.load_data(development,flow,scale,'validation')
        # Final audit is permitted only after the choice was locked; no fitting occurs here.
        te,source=final.test_data(development,flow,scale,'cpu')
        for seed in spec['seeds']:
            run=root/'runs'/f'{flow}_x{scale}_s{seed}'
            run_record=final.read(run/'run.json')
            assert run_record['source']['records']==source['records']
            assert run_record['source']['source_manifest_sha256']==source['source_manifest_sha256']
            assert run_record['source']['temporal_certificate']==source['temporal_certificate']
            for definition in record['definitions']:
                folder=run/definition['id'];result=final.read(folder/'result.json')
                assert result['definition']==definition and result['seed']==seed
                assert result['flow']==flow and result['scale']==scale
                assert result['provenance']['config_sha256']==file_sha256(config)
                for name,expected in result['provenance']['source_sha256'].items():assert file_sha256(name)==expected
                assert result['lock_sha256']==file_sha256(root/'lock.json')
                assert datetime.fromisoformat(record['locked_at'])<=datetime.fromisoformat(result['fit_started'])
                assert datetime.fromisoformat(result['fit_ended'])<=datetime.fromisoformat(result['test_opened'])
                assert result['test_opened']==run_record['test_opened']
                history=final.read(folder/'history.json');best=min(history,key=lambda r:r['validation_mse'])
                assert best['validation_mse']==result['best_validation_mse']
                step='epoch' if definition.get('reference') else 'step'
                assert best[step]==result['best_'+step]
                stats=dev.stats_for(tr,definition.get('feature','none'))
                statistics_checks.append({'flow':flow,'scale':scale,'seed':seed,'method':definition['id'],
                    **audit_training_statistics(tr,definition.get('feature','none'),result['normalization'],dev.scalar_stats(tr))})
                assert len(result['metrics'])==len(va)+len(te)
                for split,data in [('validation',va),('test',te)]:
                    recomputed=[]
                    for d in data:
                        reported=next(r for r in result['metrics'] if r['split']==split and r['file']==d['file'])
                        with np.load(folder/d['file']) as ds:
                            np.testing.assert_array_equal(ds['truth'],d['high']);np.testing.assert_array_equal(ds['mask'],d['mask'])
                            values=independent_metrics(ds['truth'],ds['prediction'],ds['mask'],stats['data_range'])
                            for key,value in values.items():np.testing.assert_allclose(reported[key],value,atol=1e-8,rtol=2e-6)
                            if not definition.get('reference'):
                                np.testing.assert_allclose(ds['prediction'][::scale,::scale],d['low'],atol=2e-6,rtol=2e-6)
                        recomputed.append(values);prediction_count+=1
                    means={k:float(np.mean([r[k] for r in recomputed])) for k in ('mse','rmse','mae','psnr','ssim')}
                    if split=='validation':np.testing.assert_allclose(means['mse'],result['best_validation_mse'],atol=1e-8,rtol=2e-5)
                    rows.append({'flow':flow,'scale':scale,'seed':seed,'method':definition['id'],'split':split,**means,
                                 'training_seconds':result['seconds'],
                                 'inference_seconds_per_slice':result['validation_inference_seconds_per_slice']})
    merged=final.read(root/'summary.json')
    assert len(rows)==len(merged['rows'])
    with (root/'per_run.csv').open() as f:csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==len(rows)
    for row in rows:
        match=lambda x:all(str(x[k])==str(row[k]) for k in ('flow','scale','seed','method','split'))
        for reported in (next(r for r in merged['rows'] if match(r)),next(r for r in csv_rows if match(r))):
            for key in ('mse','rmse','mae','psnr','ssim'):np.testing.assert_allclose(float(reported[key]),row[key],atol=1e-7)
    for summary in merged['summary']:
        group=[r for r in rows if r['method']==summary['method'] and r['split']==summary['split'] and r['scale']==summary['scale']
               and (summary['flow']=='macro' or r['flow']==summary['flow'])]
        for metric in ('mse','rmse','mae','psnr','ssim','training_seconds','inference_seconds_per_slice'):
            means=[np.mean([r[metric] for r in group if r['seed']==s]) for s in spec['seeds']]
            np.testing.assert_allclose(summary[metric+'_mean'],np.mean(means),atol=1e-7)
            np.testing.assert_allclose(summary[metric+'_std'],np.std(means,ddof=1),atol=1e-7)
    for comparison in merged['test_comparisons']:
        gains=[]
        for pair in comparison['paired']:
            group=[r for r in rows if r['scale']==comparison['scale'] and r['flow']==pair['flow'] and r['seed']==pair['seed'] and r['split']=='test']
            method=next(r for r in group if r['method']==record['selected']['id'])
            baseline=next(r for r in group if r['method']==comparison['baseline'])
            gain=method['psnr']-baseline['psnr'];gains.append(gain)
            np.testing.assert_allclose(pair['psnr_gain'],gain,atol=1e-7)
            np.testing.assert_allclose(pair['mse_relative_change'],method['mse']/baseline['mse']-1,atol=1e-7)
        np.testing.assert_allclose(comparison['mean_psnr_gain'],np.mean(gains),atol=1e-7)
    assert not [p for p in root.rglob('*') if p.suffix in ('.pt','.pth','.ckpt')]
    final.dump(root/'audit.json',{'status':'PASS','trainings':len(rows)//2,'predictions_recomputed':prediction_count,
                               'selected':record['selected'],'statistics_checks':statistics_checks,'provenance':final.provenance(config)})
    print('Final audit PASS',len(rows)//2,'trainings',prediction_count,'predictions',flush=True)


def audit_combined(spec,config):
    development,record=final.locked(spec);root=Path(spec['output'])
    merged=final.read(root/'combined.json');interpolated=final.read(root/'interpolation.json')['rows']
    for path,expected in merged['sources'].items():assert file_sha256(path)==expected
    assert len(interpolated)==144 and len(merged['summary'])==120
    assert merged['interpolation_rows']==interpolated
    count=0;support={}
    for flow in development['flows']:
        source=Path(development['source_output'])/'data'/flow
        manifest=final.read(source/'manifest.json')
        stats=dev.scalar_stats(dev.load_data(development,flow,2,'train'))
        for row in [r for r in interpolated if r['flow']==flow]:
            original=next(r for r in manifest['records'] if r['file']==row['file'])
            assert original['sha256']==row['source_sha256']==file_sha256(source/row['file'])
            folder=root/'interpolation'/flow/row['method'];assert file_sha256(folder/row['file'])==row['prediction_sha256']
            with np.load(source/row['file']) as ds,np.load(folder/row['file']) as result:
                np.testing.assert_array_equal(result['truth'],ds['high']);np.testing.assert_array_equal(result['mask'],ds['mask'])
                order=1 if row['method']=='bilinear' else 3
                expected=final.interpolation(ds['low'],row['scale'],order)
                np.testing.assert_array_equal(result['prediction'],expected)
                np.testing.assert_allclose(result['prediction'][::row['scale'],::row['scale']],ds['low'],atol=1e-6,rtol=1e-6)
                if order==1:
                    # Independent explicit four-corner bilinear interpolation.
                    y=np.arange(ds['high'].shape[0])/row['scale'];x=np.arange(ds['high'].shape[1])/row['scale']
                    y0=np.floor(y).astype(int);x0=np.floor(x).astype(int)
                    y1=np.minimum(y0+1,len(ds['low'])-1);x1=np.minimum(x0+1,ds['low'].shape[1]-1)
                    a=(y-y0)[:,None];b=(x-x0)[None,:];low=ds['low']
                    direct=(1-a)*((1-b)*low[y0[:,None],x0]+b*low[y0[:,None],x1])+a*((1-b)*low[y1[:,None],x0]+b*low[y1[:,None],x1])
                    np.testing.assert_allclose(expected,direct,atol=1e-6,rtol=1e-6)
                metrics=independent_metrics(ds['high'],result['prediction'],ds['mask'],stats['data_range'])
                for key,value in metrics.items():np.testing.assert_allclose(value,row[key],atol=1e-8,rtol=2e-6)
                # Same physical times, truth and masks must be used for every scale.
                key=(flow,row['split'],tuple(ds['times'].tolist()))
                current=(ds['high'].copy(),ds['mask'].copy())
                if key in support:
                    for a,b in zip(support[key],current):np.testing.assert_array_equal(a,b)
                else:support[key]=current
            count+=1
    assert len(support)==24
    learned=[]
    for folder in (root,Path(spec['previous_final'])):
        source=final.read(folder/'summary.json')
        learned.extend(r|{'source_version':source['version']} for r in source['rows'] if r['split']=='test')
        for row in [r for r in source['summary'] if r['split']=='test']:
            actual=next(r for r in merged['summary'] if all(r[k]==row[k] for k in ('flow','scale','method')))
            assert actual==row|{'source_version':source['version']}
    assert merged['learned_per_run']==learned and len(learned)==216
    expected_keys={(f,s,m) for f in development['flows']+['macro'] for s in spec['report_scales']
                   for m in [d['id'] for d in record['definitions']]+['bilinear','bicubic']}
    assert {(r['flow'],r['scale'],r['method']) for r in merged['summary']}==expected_keys
    for row in merged['summary']:
        source=interpolated if row['method'] in ('bilinear','bicubic') else learned
        group=[r for r in source if r['scale']==row['scale'] and r['method']==row['method'] and r['split']=='test'
               and (row['flow']=='macro' or r['flow']==row['flow'])]
        for metric in ('mse','rmse','mae','psnr','ssim'):
            if source is learned:
                values=[np.mean([r[metric] for r in group if r['seed']==seed]) for seed in spec['seeds']]
                np.testing.assert_allclose(row[metric+'_std'],np.std(values,ddof=1),atol=1e-10)
            else:
                values=[np.mean([r[metric] for r in group if r['flow']==flow]) for flow in sorted({r['flow'] for r in group})]
                assert row[metric+'_std'] is None
            np.testing.assert_allclose(row[metric+'_mean'],np.mean(values),atol=1e-10)
    with (root/'combined.csv').open() as f:csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==120
    for row in csv_rows:
        expected=next(r for r in merged['summary'] if all(str(r[k])==row[k] for k in ('flow','scale','method')))
        for metric in ('psnr','rmse','ssim','mse','mae'):
            np.testing.assert_allclose(float(row[metric+'_mean']),expected[metric+'_mean'],atol=1e-12)
            if expected[metric+'_std'] is None:assert row[metric+'_std']==''
            else:np.testing.assert_allclose(float(row[metric+'_std']),expected[metric+'_std'],atol=1e-12)
    assert not [p for p in root.rglob('*') if p.suffix in ('.pt','.pth','.ckpt')]
    final.dump(root/'combined_audit.json',{'status':'PASS','new_trainings':72,'new_learned_predictions':432,
        'reused_trainings':144,'interpolation_predictions_recomputed':count,'summary_rows':120,
        'identical_multiscale_truth_and_masks':True,'provenance':final.provenance(config)})
    print('Combined independent audit PASS',count,'interpolation predictions',flush=True)
