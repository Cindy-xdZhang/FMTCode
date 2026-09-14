"""Independent physical/data/result checks retained with the experiment."""
import ast
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.signal import convolve2d

from FMT_Utils.FTLE_Data_2D import (Velocity, file_sha256, ftle, integrate,
                                   interpolation, metadata, time_splits, evaluation_mask)
from FMT_Utils.FTLE_Encoders_2D import coordinates, encode, scalar_inputs
from experiments.FTLE_Upsampling_2D_1_2 import (GEOMETRY, Model, dump, load_data, matrix,
                                              normalization, patch_indices, provenance, read, run_root)
from FMT_Utils.FTLE_Temporal_Audit_2D import certify
from experiments.FTLE_Upsampling_2D_1_1 import Model as FrozenModel


def independent_ftle(paths, tau):
    initial, final = paths[:,:,0,:2], paths[:,:,-1,:2]
    dx = (final[:,1]-final[:,2])/(initial[:,1,0]-initial[:,2,0])[:,None]
    dy = (final[:,3]-final[:,4])/(initial[:,3,1]-initial[:,4,1])[:,None]
    a, b, d = (dx*dx).sum(1), (dx*dy).sum(1), (dy*dy).sum(1)
    largest = .5*(a+d+np.sqrt((a-d)**2+4*b*b))
    return .5*np.log(largest)/abs(tau)


def preflight(spec, config):
    torch.set_num_threads(4)
    checks = {}
    path=Path('FMT_Utils/FTLE_Baselines_2D.py')
    manifest=read('config/ftle_2d_baseline_provenance.json')
    text=path.read_text(encoding='utf-8')
    for node in ast.parse(text).body:
        if isinstance(node,ast.ClassDef):
            assert hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()==manifest['classes'][node.name]
    checks['frozen_baseline_classes']='PASS'
    t=torch.linspace(0,1,32,dtype=torch.float64)
    initial=torch.tensor([[0,0],[.005,0],[-.005,0],[0,.005],[0,-.005]],dtype=torch.float64)
    paths=initial[None,:,None,:].repeat(3,1,32,1)
    paths[...,0]*=torch.exp(.4*t)
    paths[...,1]*=torch.exp(-.4*t)
    paths[1,...,0]+=.1*t
    paths[2,...,1]+=.1*t
    angles=scalar_inputs(paths)
    assert angles['distances'].shape==(3,10,32)
    assert angles['angles'].shape==(3,6,32)
    assert scalar_inputs(paths)['angle_pairs'].tolist()==[[1,1,1,2,2,3],[2,3,4,3,4,4]]
    widths={m:encode(paths,m).shape[1] for m in GEOMETRY}
    assert widths=={'raw':320,'fmt':65,'fmt_objective_ntod_v2':176,'fmt_v5':131}
    assert torch.equal(encode(paths,'fmt'),encode(paths,'fmt_v5')[:,:65])
    xyt=torch.cat((paths,t[None,None,:,None].expand(3,5,-1,-1)),dim=-1)
    for m in GEOMETRY:
        assert torch.equal(encode(paths,m),encode(xyt,m))
    try:
        coordinates(torch.zeros(1,7,32,3))
        raise AssertionError('Seven lines were accepted')
    except ValueError:
        pass
    angle=2*t+.3
    q=torch.stack((torch.cos(angle),-torch.sin(angle),torch.sin(angle),torch.cos(angle)),-1).reshape(32,2,2)
    transformed=torch.einsum('nlti,tji->nltj',paths,q)+torch.stack((t*t,.4*t),-1)[None,None]
    torch.testing.assert_close(encode(paths,'fmt_objective_ntod_v2'),encode(transformed,'fmt_objective_ntod_v2'),atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(ftle(paths,1),torch.full((3,),.4,dtype=torch.float64),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(independent_ftle(paths.numpy(),1),ftle(paths,1).numpy(),atol=1e-12)
    checks['five_lines_time_axis_dimensions_prefix_and_objectivity']='PASS'
    checks['encoder_widths']=widths
    class Strain:
        device='cpu'
        def __call__(self,xy,t): return xy*xy.new_tensor([.4,-.4])
        def valid(self,xy): return torch.isfinite(xy).all(-1)
    integrated,valid,info=integrate(Strain(),np.array([[.1,.2],[.2,.1]]),.2,1.)
    assert valid.all() and info['steps']==217 and info['t1']==1.2
    np.testing.assert_allclose(independent_ftle(integrated.numpy(),1),.4,atol=1e-10)
    class Exiting(Strain):
        def valid(self,xy): return (xy[:,0]<.11) & torch.isfinite(xy).all(-1)
    _,valid,_=integrate(Exiting(),np.array([[.1,.2]]),.2,1.)
    assert not valid.any()
    checks['rk4_physical_endpoint_and_invalid_path_rejection']='PASS'
    yy,xx=np.meshgrid(np.arange(8),np.arange(11),indexing='ij')
    low=2*xx+3*yy+.7
    for scale in spec['scales']:
        output=interpolation(low,scale)
        hy,hx=np.meshgrid(np.arange(output.shape[0])/scale,np.arange(output.shape[1])/scale,indexing='ij')
        np.testing.assert_allclose(output,2*hx+3*hy+.7,atol=1e-12)
        np.testing.assert_allclose(output[::scale,::scale],low,atol=1e-12)
    checks['nested_grid_interpolation']='PASS'
    optimizations={}
    for method in spec['methods']:
        torch.manual_seed(217)
        model=Model(spec,method,2,widths.get(method,0))
        low=torch.rand(2,1,9,9)
        geometry=torch.rand(2,widths[method],9,9) if method in GEOMETRY else None
        target=torch.nn.functional.interpolate(low,size=(17,17),mode='bilinear',align_corners=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        values=[]
        for _ in range(35):
            model.train(); optimizer.zero_grad()
            output=model(low,geometry)
            assert output.shape==target.shape
            loss=(output-target).square().mean(); loss.backward(); optimizer.step()
            values.append(float(loss.detach()))
        assert np.isfinite(values).all() and min(values[-5:])<.25*values[0], (method,values[0],values[-1])
        optimizations[method]={'initial_mse':values[0],'final_mse':values[-1],
                               'parameters':sum(p.numel() for p in model.parameters())}
    assert len({optimizations[m]['parameters'] for m in GEOMETRY})==1
    checks['six_model_optimization']=optimizations
    model_checks=[]
    for method,scale in itertools.product(spec['methods'],spec['scales']):
        torch.manual_seed(153)
        frozen=FrozenModel(spec,method,scale,widths.get(method,0)).eval()
        low=torch.rand(2,1,9,13)
        geo=torch.rand(2,widths[method],9,13) if method in GEOMETRY else None
        expected=frozen(low,geo).detach()
        for dropout in spec['dropout_rates']:
            torch.manual_seed(153)
            model=Model(spec,method,scale,widths.get(method,0),dropout).eval()
            assert set(model.state_dict())==set(frozen.state_dict())
            for key in model.state_dict():
                torch.testing.assert_close(model.state_dict()[key],frozen.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(model(low,geo),expected,rtol=0,atol=0)
            assert expected.shape==(2,1,8*scale+1,12*scale+1)
            model.train()
            first,second=model(low,geo),model(low,geo)
            assert bool(torch.equal(first,second)) == (dropout==0)
            (first.square().mean()).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            model_checks.append({'method':method,'scale':scale,'dropout':dropout,'status':'PASS'})
    checks['dropout_train_eval_gradient_and_frozen_identity']=model_checks
    artificial={'name':'doublegyre2d','tmin':0.,'tmax':10.}
    try:
        certify(artificial,{'train':[1.], 'validation':[1.5], 'test':[5.]},1.)
        raise AssertionError('Overlapping window was not rejected')
    except ValueError:
        pass
    checks['deliberate_time_overlap_rejected']='PASS'

    # Source-only metadata and a tiny first-training-window integration; no test data.
    probes=[]
    for flow in spec['flows']:
        meta=metadata(spec['data_root'],flow['name'])
        splits=time_splits(meta,spec['split_counts'],spec['integration']['tau'])
        assert max(splits['train'])+1 < min(splits['validation'])
        assert max(splits['validation'])+1 < min(splits['test'])
        certificate=certify(meta,splits,spec['integration']['tau'])
        if meta['source_dt'] > 0:
            start=splits['train'][0]+.01*meta['source_dt']
            try:
                certify(meta,{'train':[start], 'validation':[start+.02*meta['source_dt']],
                              'test':[start+.04*meta['source_dt']]},.005*meta['source_dt'])
                raise AssertionError('Shared source frames were accepted')
            except ValueError as error:
                assert 'Shared interpolation source frames' in str(error)
        xmin,xmax,ymin,ymax=meta['bounds']
        yy,xx=np.meshgrid(np.linspace(ymin+.2*(ymax-ymin),ymax-.2*(ymax-ymin),4),
                          np.linspace(xmin+.2*(xmax-xmin),xmax-.2*(xmax-xmin),4),indexing='ij')
        velocity=Velocity(meta,splits['train'][0],1.,'cpu')
        p,valid,info=integrate(velocity,np.stack((xx,yy),-1).reshape(-1,2),splits['train'][0],**spec['integration'])
        assert valid.any(),flow['name']
        error=float(np.max(abs(independent_ftle(p[valid].numpy(),1)-ftle(p[valid],1).numpy())))
        assert error<1e-8
        probes.append({'flow':flow['name'],'valid_primitives':int(valid.sum()),'max_ftle_recompute_error':error,
                       'time_splits':splits,'temporal_certificate':certificate,'source':meta,'integration':info})
        del velocity
    checks['real_training_window_probes']=probes
    dump(Path(spec['output'])/'preflight.json',{'status':'PASS','provenance':provenance(config),'checks':checks})
    print('Preflight PASS',flush=True)


def audit_data(spec,config):
    report=[]
    for flow in spec['flows']:
        root=Path(spec['output'])/'data'/flow['name']
        manifest=read(root/'manifest.json')
        assert len(manifest['records'])==sum(spec['split_counts'].values())*len(spec['scales'])
        splits=manifest['time_splits']
        tau=spec['integration']['tau']
        assert max(splits['train'])+tau < min(splits['validation'])
        assert max(splits['validation'])+tau < min(splits['test'])
        assert certify(manifest['source'],splits,tau)==manifest['temporal_certificate']
        np.testing.assert_equal(splits,time_splits(manifest['source'],spec['split_counts'],tau))
        for row in manifest['records']:
            assert file_sha256(root/row['file'])==row['sha256']
            with np.load(root/row['file']) as ds:
                scale=row['scale']
                t0=splits[row['split']][row['slice_index']]
                assert row['integration']['t0']==t0 and row['integration']['t1']==t0+tau
                np.testing.assert_allclose(ds['times'],np.linspace(t0,t0+tau,spec['integration']['samples']),atol=1e-12,rtol=0)
                masks=[evaluation_mask(ds['valid_high'],ds['valid_high'][::s,::s],s) for s in spec['scales']]
                np.testing.assert_array_equal(ds['mask'],np.logical_and.reduce(masks))
                assert binary_erosion(ds['mask'],structure=np.ones((7,7),bool)).any()
                yy,xx=np.meshgrid(ds['ys'][::scale],ds['xs'][::scale],indexing='ij')
                expected_seeds=np.stack((xx,yy),-1).reshape(-1,2)
                np.testing.assert_allclose(ds['paths_low'][:,0,0,:],expected_seeds,rtol=0,atol=1e-12)
                np.testing.assert_array_equal(ds['low'],ds['high'][::scale,::scale])
                np.testing.assert_array_equal(ds['valid_low'],ds['valid_high'][::scale,::scale])
                paths=ds['paths_low'][ds['valid_low'].ravel()]
                expected=independent_ftle(paths,tau)
                np.testing.assert_allclose(expected,ds['low'][ds['valid_low']],atol=2e-6,rtol=2e-6)
                initial=paths[:,:,0]
                target=np.array([[0,0],[.005,0],[-.005,0],[0,.005],[0,-.005]])
                np.testing.assert_allclose(initial-initial[:,:1],np.broadcast_to(target,initial.shape),atol=2e-12)
                ids=np.linspace(0,len(paths)-1,min(13,len(paths))).astype(int)
                for method in GEOMETRY:
                    expected=encode(paths[ids],method,spec['fourier_bins']).numpy()
                    stored=ds[method][:,ds['valid_low']].T[ids]
                    np.testing.assert_allclose(expected,stored,atol=3e-5,rtol=3e-5)
                assert ds['mask'].sum()>100 and np.isfinite(ds['high']).all()
                assert np.all(ds['mask']<=ds['valid_high'])
                assert np.allclose(np.diff(ds['times']),tau/(spec['integration']['samples']-1))
                assert 'paths_high' not in ds.files
        for scale in spec['scales']:
            data=load_data(spec,flow['name'],scale,'train')
            patches,_=patch_indices(data,spec,scale)
            report.append({'flow':flow['name'],'scale':scale,'training_patches':len(patches)})
    dump(Path(spec['output'])/'data_audit.json',{'status':'PASS','provenance':provenance(config),'checks':report})
    print('Data audit PASS',flush=True)


def independent_metrics(gt,pred,mask,data_range):
    gt,pred=gt.astype(np.float64),pred.astype(np.float64)
    difference=gt[mask]-pred[mask]
    mse=np.dot(difference,difference)/len(difference)
    kernel=np.ones((7,7))/49
    avg=lambda x:convolve2d(x,kernel,mode='same',boundary='symm')
    x,y=avg(gt),avg(pred)
    var_x=(avg(gt**2)-x**2)*49/48
    var_y=(avg(pred**2)-y**2)*49/48
    cov=(avg(gt*pred)-x*y)*49/48
    ssim=((2*x*y+(.01*data_range)**2)*(2*cov+(.03*data_range)**2))/((x*x+y*y+(.01*data_range)**2)*(var_x+var_y+(.03*data_range)**2))
    safe=binary_erosion(mask,structure=np.ones((7,7),bool),border_value=0)
    return {'mse':float(mse),'rmse':float(math.sqrt(mse)),'mae':float(abs(difference).mean()),
            'psnr':float(10*math.log10(data_range**2/max(mse,1e-30))),
            'ssim':float(ssim[safe].mean()) if safe.any() else None}


def audit_results(spec,config):
    count=0
    audited_rows={}
    interpolation_stats={}
    for flow,scale,seed in matrix(spec):
        for dropout,method in itertools.product(spec['dropout_rates'],spec['methods']):
            root=run_root(spec,flow,scale,seed,method,dropout)
            result=read(root/'result.json'); history=read(root/'history.json')
            selection=read(root/'selection.json')
            manifest_path=Path(spec['output'])/'data'/flow/'manifest.json'
            manifest=read(manifest_path)
            assert selection['locked_at'] <= result['test_access']['first_read_at']
            assert result['test_access']['selection_sha256']==file_sha256(root/'selection.json')
            assert selection['history_sha256']==file_sha256(root/'history.json')
            assert selection['data_manifest_sha256']==file_sha256(manifest_path)==result['test_access']['data_manifest_sha256']
            assert selection['config_sha256']==file_sha256(config)
            for key in ('flow','scale','seed','method','dropout','normalization','best_epoch','best_validation_mse'):
                assert selection[key]==result[key]
            for split in ('train','validation'):
                assert selection[split+'_files']==[r['file'] for r in manifest['records'] if r['scale']==scale and r['split']==split]
            assert len(result['metrics'])==spec['split_counts']['test']+spec['split_counts']['validation']
            for split in ('test','validation'):
                assert [r['source_file'] for r in result['metrics'] if r['split']==split]==[r['file'] for r in manifest['records'] if r['scale']==scale and r['split']==split]
            best=min(history,key=lambda r:r['validation_mse'])
            assert best['epoch']==result['best_epoch']
            assert best['validation_mse']==result['best_validation_mse']
            val=[]
            for row in result['metrics']:
                with np.load(root/row['file']) as ds:
                    values=independent_metrics(ds['truth'],ds['prediction'],ds['mask'],result['normalization']['data_range'])
                    source=Path(spec['output'])/'data'/flow/row['source_file']
                    with np.load(source) as original:
                        np.testing.assert_array_equal(ds['truth'],original['high'])
                        np.testing.assert_array_equal(ds['mask'],original['mask'])
                    for metric,value in values.items():
                        if value is None: assert row[metric] is None
                        else: np.testing.assert_allclose(value,row[metric],rtol=2e-6,atol=1e-8)
                    if row['split']=='validation': val.append(values['mse'])
                    count+=1
                    audited_rows[(flow,scale,str(seed),method,dropout,row['split'],row['source_file'])]=values
            np.testing.assert_allclose(np.mean(val),result['best_validation_mse'],atol=1e-8,rtol=2e-5)
            assert result['provenance']['config_sha256']==file_sha256(config)
    summary=read(Path(spec['output'])/'summary.json')
    with (Path(spec['output'])/'metrics.csv').open(newline='',encoding='utf-8') as stream:
        table=list(csv.DictReader(stream))
    expected=len(matrix(spec))*len(spec['methods'])*len(spec['dropout_rates'])*(spec['split_counts']['test']+spec['split_counts']['validation'])
    expected+=len(spec['flows'])*len(spec['scales'])*2*(spec['split_counts']['test']+spec['split_counts']['validation'])
    assert len(table)==expected
    for row in table:
        key=(row['flow'],int(row['scale']),row['seed'],row['method'],float(row['dropout']),row['split'],row['source_file'])
        if key in audited_rows:
            for metric,value in audited_rows[key].items():
                if value is not None: np.testing.assert_allclose(float(row[metric]),value,atol=1e-8,rtol=2e-6)
        else:
            assert row['method'] in ('bilinear','bicubic') and row['seed']=='-1'
            data=load_data(spec,row['flow'],int(row['scale']),row['split'])
            d=next(d for d in data if d['file']==row['source_file'])
            stats_key=(row['flow'],int(row['scale']))
            if stats_key not in interpolation_stats:
                interpolation_stats[stats_key]=normalization(load_data(spec,row['flow'],int(row['scale']),'train'),'unet')
            stats=interpolation_stats[stats_key]
            values=independent_metrics(d['high'],interpolation(d['low'],int(row['scale']),1 if row['method']=='bilinear' else 3),d['mask'],stats['data_range'])
            for metric,value in values.items():
                if value is not None: np.testing.assert_allclose(float(row[metric]),value,atol=1e-8,rtol=2e-6)
    assert len(summary['results'])==len(spec['flows'])*len(spec['scales'])*(len(spec['methods'])*len(spec['dropout_rates'])+2)
    for row in summary['results']:
        group=[r for r in table if r['flow']==row['flow'] and int(r['scale'])==row['scale']
               and r['method']==row['method'] and float(r['dropout'])==row['dropout'] and r['split']=='test']
        for metric in ('rmse','mae','psnr','ssim'):
            means=[np.mean([float(r[metric]) for r in group if r['seed']==seed])
                   for seed in sorted(set(r['seed'] for r in group))]
            np.testing.assert_allclose(np.mean(means),row[metric+'_mean'],atol=1e-12)
            np.testing.assert_allclose(np.std(means),row[metric+'_std'],atol=1e-12)
    leftovers=[str(p) for p in Path(spec['output']).rglob('*') if p.suffix in ('.pt','.pth','.ckpt')]
    assert not leftovers,leftovers
    dump(Path(spec['output'])/'result_audit.json',{'status':'PASS','provenance':provenance(config),
                                                'prediction_files_recomputed':count,'summary_rows':len(summary['results']),
                                                'metric_rows':len(table),'checkpoint_files':leftovers})
    print('Result audit PASS',flush=True)
