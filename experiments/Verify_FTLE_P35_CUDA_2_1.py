"""GPU compatibility of fixed Fourier features and deterministic neural training."""
import os
from pathlib import Path
import numpy as np
import torch
from experiments import FTLE_P35_Fusion_2_1 as dev
from FMT_Utils.FTLE_P35_2D_2_1 import encode,WIDTHS
from FMT_Utils.FTLE_Fusion_2D_2_1 import FusionSR
from FMT_Utils.FTLE_Data_2D import file_sha256


def main():
    config='config/Other_FTLEP35Fusion_2.1.json';spec=dev.read(config)
    root=Path(spec['output']);phase='cuda-verification'
    def event(state,code=None):
        value={'phase':phase,'state':state,'exit_code':code,**dev.provenance(config)}
        value['source_sha256'][__file__]=file_sha256(__file__)
        dev.dump(root/'runtime'/f"{value['job_id']}_None_{phase}_{state}.json",value)
    event('STARTED')
    try:
        assert torch.cuda.is_available();torch.set_num_threads(4);dev.initialize(102)
        features=[]
        for flow in spec['flows']:
            source=Path(spec['source_output'])/'data'/flow
            row=next(r for r in dev.read(source/'manifest.json')['records'] if r['split']=='train' and r['scale']==4)
            with np.load(source/row['file']) as ds:
                ids=np.flatnonzero(ds['valid_low'].ravel())
                ids=ids[np.linspace(0,len(ids)-1,min(64,len(ids))).astype(int)]
                paths=ds['paths_low'][ids]
            cpu=encode(paths);gpu=encode(torch.from_numpy(paths).cuda())
            tr=dev.load_data(spec,flow,4,'train')
            for name in cpu:
                torch.testing.assert_close(cpu[name],gpu[name].cpu(),atol=1e-6,rtol=1e-6)
                stats=dev.stats_for(tr,name);mean=np.asarray(stats['feature_mean']);std=np.asarray(stats['feature_std'])
                a=np.clip((dev.transform(cpu[name].numpy())-mean)/std,-8,8)
                b=np.clip((dev.transform(gpu[name].cpu().numpy())-mean)/std,-8,8)
                np.testing.assert_allclose(a,b,atol=1e-4,rtol=1e-4)
                features.append({'flow':flow,'feature':name,'sampled_train_primitives':len(ids),
                                 'normalized_max_abs_difference':float(abs(a-b).max())})
        networks=[];unsupported=[]
        for scale in (4,8):
            low=torch.randn(2,1,17,33,device='cuda');geometry=torch.randn(2,667,17,33,device='cuda')
            valid=torch.ones_like(low);bicubic=torch.randn(2,1,16*scale+1,32*scale+1,device='cuda')
            for arch in ('pyramid','unet'):
                model=FusionSR(scale,arch,spec['width']).cuda()
                with torch.no_grad():model.head.weight.normal_(0,.01)
                try:
                    out=model(low,geometry,valid,bicubic,low)
                    torch.testing.assert_close(out[...,::scale,::scale],low,atol=0,rtol=0)
                    out.square().mean().backward()
                    assert torch.isfinite(model.geometry.weight.grad).all() and model.geometry.weight.grad.abs().sum()>0
                    networks.append({'architecture':arch,'scale':scale,'status':'PASS'})
                except RuntimeError as error:
                    if arch!='unet' or 'deterministic' not in str(error).lower():raise
                    unsupported.append({'architecture':arch,'scale':scale,'error':str(error),
                                        'policy':'Keep frozen deterministic algorithm; this candidate can run on CPU.'})
                del model
            reference=dev.read(spec['reference_config'])
            for name in ('espcn','unet'):
                model=dev.old.Model(reference,name,scale,0,0.).cuda()
                out=model(low);out.square().mean().backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                networks.append({'reference':name,'scale':scale,'status':'PASS'});del model
        assert all(any(r.get('architecture')=='pyramid' and r['scale']==s for r in networks) for s in (4,8))
        dev.dump(root/'cuda_verification.json',{'version':'Verify_FTLEP35CUDA_2.1','status':'PASS',
                 'supported_final_architectures':[a for a in ('pyramid','unet') if not any(r['architecture']==a for r in unsupported)],
                 'training_features':features,'networks':networks,'unsupported_deterministic_paths':unsupported,
                 'validation_or_test_access':False,'provenance':dev.provenance(config),'verification_source_sha256':file_sha256(__file__)})
        print('CUDA compatibility verification PASS; validation/test unopened',flush=True)
    except BaseException:
        event('ENDED',1);raise
    event('ENDED',0)


if __name__=='__main__':main()
