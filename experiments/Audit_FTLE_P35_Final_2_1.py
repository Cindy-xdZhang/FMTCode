"""Recompute every held-out metric and verify model-selection/test boundaries."""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from experiments import FTLE_P35_Final_2_1 as final
from experiments import FTLE_P35_Fusion_2_1 as dev
from experiments.Audit_FTLE_Upsampling_2D_1_2 import independent_metrics
from FMT_Utils.FTLE_Data_2D import file_sha256


def preflight(spec,config):
    development,record=final.locked(spec)
    assert len(set(record['seeds']))==3
    assert not (set(record['seeds']) & {development['search_seed']})
    for name,expected in final.read(Path(development['output'])/'selection.json')['provenance']['source_sha256'].items():
        assert file_sha256(name)==expected,('Frozen development implementation changed',name)
    assert len(record['definitions']) in (6,7)
    final.dump(Path(spec['output'])/'preflight.json',{'status':'PASS','definitions':record['definitions'],
           'lock_sha256':file_sha256(Path(spec['output'])/'lock.json'),'test_read':False,'provenance':final.provenance(config)})
    print('Final preflight PASS',flush=True)


def audit(spec,config):
    development,record=final.locked(spec);root=Path(spec['output']);rows=[];prediction_count=0
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
                for key,value in stats.items():np.testing.assert_allclose(result['normalization'][key],value,atol=1e-9,rtol=1e-8)
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
                               'selected':record['selected'],'provenance':final.provenance(config)})
    print('Final audit PASS',len(rows)//2,'trainings',prediction_count,'predictions',flush=True)
