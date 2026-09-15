"""Independent feature, source-data and validation-prediction checks."""
import ast
import hashlib
from pathlib import Path
import numpy as np
import torch

from FMT_Utils.FTLE_Data_2D import file_sha256,ftle
from FMT_Utils.FTLE_P35_2D_2_1 import encode,descriptor,WIDTHS
from FMT_Utils.FTLE_Fusion_2D_2_1 import FusionSR
from experiments.FTLE_P35_Fusion_2_1 import read,dump,provenance,candidates,matrix,load_data,stats_for
from experiments.Audit_FTLE_Upsampling_2D_1_2 import independent_metrics


def preflight(spec,config):
    checks={}
    t=torch.linspace(0,1,32,dtype=torch.float64)
    initial=torch.tensor([[0.,0.],[.005,0.],[-.005,0.],[0.,.005],[0.,-.005]],dtype=torch.float64)
    x=initial[None,:,None,:].repeat(3,1,32,1)
    x[...,0]*=torch.exp(.4*t);x[...,1]*=torch.exp(-.4*t)
    x[1,...,0]+=.1*t;x[2,...,1]+=.2*t
    features=encode(x)
    assert {k:v.shape[1] for k,v in features.items()}=={k:v for k,v in WIDTHS.items() if k!='none'}
    for name in ('p35','p35_multitime'):
        assert torch.isfinite(features[name]).all()
    permuted=encode(x[:,[0,3,1,4,2]])
    torch.testing.assert_close(features['p35'],permuted['p35'],atol=1e-6,rtol=1e-6)
    translated=encode(x+17.)
    for name in features:torch.testing.assert_close(features[name],translated[name],atol=2e-5,rtol=2e-5)
    for bad in (torch.zeros(3,7,32,2),torch.zeros(3,5,32,3)):
        try:encode(bad);raise AssertionError('Invalid shape accepted')
        except ValueError:pass
    checks['five_lines_2d_shapes_and_semantic_pooling']='PASS'
    # Direct trigonometric DFT independently checks every retained complex coefficient.
    delta=(x[:,1:]-x[:,:1]).diff(dim=-2)/.005
    phase=-2*np.pi*np.arange(6)[:,None]*np.arange(31)[None,:]/31
    reference=np.einsum('nltd,kt->nlkd',delta.numpy(),np.exp(1j*phase))
    actual=torch.fft.rfft(delta,dim=-2)[:,:,:6].numpy()
    np.testing.assert_allclose(actual,reference,atol=1e-12)
    # DC coefficients alone recover the opposite-pair deformation at coarse sites.
    end=initial[None,1:].numpy()+reference[:,:,0].real*.005
    dx=(end[:,0]-end[:,1])/.01;dy=(end[:,2]-end[:,3])/.01
    j=np.stack((dx,dy),-1)
    np.testing.assert_allclose(np.log(np.linalg.svd(j,compute_uv=False)[:,0]),ftle(x,1).numpy(),atol=1e-12)
    checks['DFT_coefficients_and_physical_deformation_information']='PASS'
    baseline=Path('FMT_Utils/FTLE_Baselines_2D.py').read_text(encoding='utf-8')
    manifest=read('config/ftle_2d_baseline_provenance.json')
    for node in ast.parse(baseline).body:
        if isinstance(node,ast.ClassDef):
            assert hashlib.sha256(ast.get_source_segment(baseline,node).encode()).hexdigest()==manifest['classes'][node.name]
    checks['frozen_baseline_bodies']='PASS'
    shapes=[]
    for architecture in ('pyramid','unet'):
        for scale in spec['scales']:
            torch.manual_seed(102)
            model=FusionSR(scale,architecture,spec['width'])
            low=torch.randn(2,1,9,13);valid=torch.ones_like(low)
            bicubic=torch.nn.functional.interpolate(low,size=(8*scale+1,12*scale+1),mode='bicubic',align_corners=True)
            geo=torch.randn(2,667,9,13)
            out=model(low,geo,valid,bicubic,low)
            torch.testing.assert_close(out[...,::scale,::scale],low,atol=0,rtol=0)
            mask=torch.ones_like(out,dtype=torch.bool);mask[...,::scale,::scale]=False
            torch.testing.assert_close(out[mask],bicubic[mask],atol=0,rtol=0)
            with torch.no_grad():model.head.weight.normal_(0,.01)
            a=model(low,geo,valid,bicubic,low)
            b=model(low,torch.zeros_like(geo),valid,bicubic,low)
            assert not torch.equal(a,b)
            loss=(a-bicubic).square().mean();loss.backward()
            assert model.geometry.weight.grad.abs().sum()>0
            shapes.append({'architecture':architecture,'scale':scale,'parameters':sum(p.numel() for p in model.parameters())})
    checks['residual_initialization_observation_constraint_and_geometry_gradient']=shapes
    assert len(candidates())==9
    dump(Path(spec['output'])/'preflight.json',{'status':'PASS','checks':checks,'provenance':provenance(config)})
    print('Preflight PASS',flush=True)


def audit_data(spec,config):
    checks=[]
    for flow in spec['flows']:
        root=Path(spec['output'])/'data'/flow
        m=read(root/'manifest.json');source=Path(spec['source_output'])/'data'/flow
        assert m['source_manifest_sha256']==file_sha256(source/'manifest.json')
        assert len(m['records'])==sum(spec['split_counts'][s] for s in ('train','validation'))*len(spec['scales'])
        assert all(r['split'] in ('train','validation') for r in m['records'])
        for row in m['records']:
            assert file_sha256(root/row['file'])==row['sha256']
            assert file_sha256(source/row['file'])==row['source_sha256']
            with np.load(root/row['file']) as ds,np.load(source/row['file']) as original:
                for key in ('low','high','valid_high','valid_low','mask','times','xs','ys'):
                    np.testing.assert_array_equal(ds[key],original[key])
                ids=np.flatnonzero(ds['valid_low'].ravel())
                ids=ids[np.linspace(0,len(ids)-1,min(len(ids),17)).astype(int)]
                expected=encode(original['paths_low'][ids])
                for feature,value in expected.items():
                    actual=ds[feature].reshape(WIDTHS[feature],-1).T[ids]
                    np.testing.assert_allclose(actual,value.numpy(),atol=1e-5,rtol=1e-5)
                assert not any(k.startswith('paths') for k in ds.files)
                assert ds['mask'].sum()>100
        checks.append({'flow':flow,'records':len(m['records']),'test_prepared':False,'temporal_certificate':m['temporal_certificate']['status']})
    dump(Path(spec['output'])/'data_audit.json',{'status':'PASS','checks':checks,'provenance':provenance(config)})
    print('Data audit PASS',flush=True)


def audit_results(spec,config):
    rows=[];count=0
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
            for key,value in expected.items():np.testing.assert_allclose(result['normalization'][key],value,rtol=1e-8,atol=1e-9)
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
          'trainings':len(rows),'selected':selected,'gate_passed':choice['gate_passed'],'test_read':False,'provenance':provenance(config)})
    print('Result audit PASS',flush=True)
