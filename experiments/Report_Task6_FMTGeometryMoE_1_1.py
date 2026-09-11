"""Read-only progress and provenance checks; never read held-out geometries."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from FMT_Utils.FlowMapData_3D import sha256
from experiments.Task6_DirectNeural_5_1 import read


def report(root,config):
    root=Path(root)
    spec=read(config)
    assert sha256(config)==sha256(root/'config.frozen.json')
    events=[read(p) for p in sorted((root/'events').glob('*.json'))]
    states=Counter((r['phase'],r['status']) for r in events)
    counts=Counter()
    by_arm=Counter()
    parameters={}
    fits=[]
    for phase in ('fit_check','screen','refine','final'):
        for path in sorted((root/phase).rglob('fit.json')):
            fit=read(path)
            assert np.isfinite([fit['train_rmse_r'],fit['validation_rmse_r']]).all()
            assert fit['selected_step']>0 and fit['selected_step']<=fit['updates']
            if phase!='fit_check':
                assert fit['train_samples']==spec['primary_train_size']
            if fit.get('arm') in spec['arms']:
                structure=fit['structure']
                assert structure['latent_dimension']==spec['latent_dimension']
                assert not structure['analytic_inverse'] and not structure['raw_geometry_skip']
                assert structure['fixed_pointnn_parameters']==0
                assert fit['statistics_train_samples']==fit['train_samples']
                key=(str(path.parent.parent),fit['candidate']['id'])
                parameters.setdefault(key,{})[fit['arm']]=structure['trainable_parameters']
            counts[phase]+=1
            by_arm[(phase,fit['arm'])]+=1
            fits.append(dict(path=str(path.relative_to(root)),arm=fit['arm'],
                train_rmse_r=fit['train_rmse_r'],validation_rmse_r=fit['validation_rmse_r'],
                seconds=fit['seconds']))
    for group in parameters.values():
        if 'fmt_moe' in group and 'geometry_moe' in group:
            assert group['fmt_moe']==group['geometry_moe']
    progress=[]
    for path in sorted(root.rglob('progress.json')):
        if not (path.parent/'fit.json').exists():
            progress.append(dict(path=str(path.relative_to(root)),**read(path)))
    out=dict(config_sha256=sha256(config),events={f'{p}:{s}':n for (p,s),n in states.items()},
             completed_models=dict(counts),models_by_arm={f'{p}:{a}':n for (p,a),n in by_arm.items()},
             incomplete_models=progress,completed_fits=fits)
    selection=root/'selection.json'
    if selection.exists():
        assert sha256(selection)==sha256(root/'selection.before_test.json')
        data=read(selection)
        assert not data['test_read'] and data['provenance']['config_sha256']==sha256(config)
        out['selection']={f:dict(candidate=r['candidate']['id'],qualified=r['qualified'],means=r['means'])
                          for f,r in data['families'].items()}
        out['gate_passed']=data['gate_passed']
    audit=root/'final_audit.json'
    if audit.exists():
        data=read(audit)
        assert data['audit_passed'] and data['provenance']['config_sha256']==sha256(config)
        out['final_summary']=data['summary']
        out['final_failures']=data['failures']
    for suffix in ('*.pt','*.pth','*.ckpt'):
        assert not list(root.rglob(suffix))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/Verify_Task6_FMTGeometryMoE_1.1.json')
    parser.add_argument('--root')
    args=parser.parse_args()
    spec=read(args.config)
    print(json.dumps(report(args.root or spec['output_root'],args.config),indent=2))
