"""Revision of the development auditor only; frozen fits and metrics are unchanged."""
from experiments.Audit_FTLE_P35_Fusion_2_1 import *
from FMT_Utils.FTLE_Statistics_Audit_2_1 import audit_training_statistics

def numerical_provenance(config):
    value=provenance(config)
    for path in ('experiments/Audit_FTLE_P35_Numerics_2_1.py','FMT_Utils/FTLE_Statistics_Audit_2_1.py'):
        value['source_sha256'][path]=file_sha256(path)
    return value

def audit_results(spec,config):
    rows=[];count=0;statistics_checks=[]
    for flow,scale in matrix(spec):
        tr=load_data(spec,flow,scale,'train');va=load_data(spec,flow,scale,'validation')
        for name in ['espcn','unet']+[r['id'] for r in candidates()]:
            root=Path(spec['output'])/'search'/f'{flow}_x{scale}'/name
            result=read(root/'result.json');history=read(root/'history.json')
            best=min(history,key=lambda r:r['validation_mse'])
            assert best['validation_mse']==result['best_validation_mse']
            if 'best_step' in result:assert best['step']==result['best_step']
            else:assert best['epoch']==result['best_epoch']
            feature=result['definition'].get('feature','none')
            expected=stats_for(tr,feature)
            statistics_checks.append({'flow':flow,'scale':scale,'method':name,**audit_training_statistics(tr,feature,result['normalization'],stats_for(tr,'none'))})
            metrics=[]
            assert len(result['metrics'])==len(va)
            for d,row in zip(va,result['metrics']):
                assert row['split']=='validation' and row['file']==d['file']
                with np.load(root/row['file']) as ds:
                    np.testing.assert_array_equal(ds['truth'],d['high']);np.testing.assert_array_equal(ds['mask'],d['mask'])
                    values=independent_metrics(ds['truth'],ds['prediction'],ds['mask'],expected['data_range'])
                    for metric,value in values.items():np.testing.assert_allclose(row[metric],value,atol=1e-8,rtol=2e-6)
                    if not result['definition'].get('reference'):
                        np.testing.assert_allclose(ds['prediction'][::scale,::scale],d['low'],atol=2e-6,rtol=2e-6)
                count+=1;metrics.append(values)
            np.testing.assert_allclose(np.mean([r['mse'] for r in metrics]),result['best_validation_mse'],atol=1e-8,rtol=2e-5)
            assert result['provenance']['config_sha256']==file_sha256(config)
            rows.append({'flow':flow,'scale':scale,'candidate':name,'psnr':np.mean([r['psnr'] for r in metrics])})
    choice=read(Path(spec['output'])/'selection.json')
    for row in choice['rows']:
        actual=next(r for r in rows if all(r[k]==row[k] for k in ('flow','scale','candidate')))
        np.testing.assert_allclose(actual['psnr'],row['psnr'],atol=1e-7)
    for candidate in choice['candidates']:
        group=[r for r in rows if r['candidate']==candidate['id']]
        np.testing.assert_allclose(candidate['mean_psnr'],np.mean([r['psnr'] for r in group]),atol=1e-7)
        for scale in spec['scales']:
            np.testing.assert_allclose(candidate['scale_psnr'][str(scale)],np.mean([r['psnr'] for r in group if r['scale']==scale]),atol=1e-7)
    selected=max((r for r in choice['candidates'] if r['eligible']),key=lambda r:(r['mean_psnr'],r['id']))
    assert selected==choice['selected'] and choice['test_read'] is False
    for comparison in choice['comparisons']:
        actual=np.mean([r['psnr'] for r in rows if r['candidate']==comparison['baseline'] and r['scale']==comparison['scale']])
        np.testing.assert_allclose(actual,comparison['baseline_psnr'],atol=1e-7)
        np.testing.assert_allclose(comparison['psnr_gain'],selected['scale_psnr'][str(comparison['scale'])]-actual,atol=1e-7)
    assert choice['gate_passed']==all(r['psnr_gain']>0 for r in choice['comparisons'])
    assert not [p for p in Path(spec['output']).rglob('*') if p.suffix in ('.pt','.pth','.ckpt')]
    dump(Path(spec['output'])/'result_audit.json',{'status':'PASS','predictions_recomputed':count,
          'trainings':len(rows),'selected':selected,'gate_passed':choice['gate_passed'],'test_read':False,'statistics_checks':statistics_checks,'provenance':numerical_provenance(config)})
    print('Result audit PASS',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config/Other_FTLEP35Fusion_2.1.json')
    args=parser.parse_args();audit_results(read(args.config),args.config)
