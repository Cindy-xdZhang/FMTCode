"""CPU architecture inspection and synthetic center tracing; no model fitting."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from FMT_Utils import FMT_P35_NormFrequency_3_1 as p35
from FMT_Utils import Task4C_FPSAugmentSearch_1_1 as multi
from FMT_Utils import ASAPFMT_Task35_1_1 as asap
from FMT_Utils.Task4C_NeighborSelection_1_1 import neighbor_indices
from FMT_Utils.FMT_V8_Search_2_1 import pooling_candidates, PROFILES
from FMT_Utils.Task4B_Classifier_3D import PathlineMulticlassClassifier3D
from FMT_Utils.FMT_SingleCenter_1_1 import SingleCenterMLP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/Verify_FMTArchitectureAudit_1.1'


def inventory(name, model, history=None):
    convolution = [(n, type(m).__name__, list(m.kernel_size)) for n, m in model.named_modules() if isinstance(m, nn.modules.conv._ConvNd)]
    return dict(name=name, parameters=sum(p.numel() for p in model.parameters()), convolution=convolution,
                linear_layers=sum(isinstance(m, nn.Linear) for m in model.modules()),
                historical_source=history, network=str(model))


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = subprocess.check_output(['git','show','52f6284c:FMT_Utils/PathlineClassifier_3D.py'],cwd=ROOT)
    previous = types.ModuleType('audit_historical_classifier')
    exec(compile(frozen, 'historical_PathlineClassifier_3D', 'exec'), previous.__dict__)
    frozen_multi = types.ModuleType('audit_historical_task4_search')
    multi_source = subprocess.check_output(['git','show','5ab735c0:FMT_Utils/Task4C_FPSAugmentSearch_1_1.py'],cwd=ROOT)
    exec(compile(multi_source,'historical_Task4C_search','exec'),frozen_multi.__dict__)
    models = [inventory('historical_Task35_raw_fmt', previous.PathlineBinaryClassifier3D('raw_fmt',fmt_dim=141), '52f6284c')]
    for kind in ('geometry_fmt','fmt_only','dual'):
        model = previous.PathlineFMTResidualClassifier3D(previous.PathlineBinaryClassifier3D('raw'), fmt_dim=433, residual_input=kind)
        models.append(inventory('historical_Task35_residual_'+kind, model, '52f6284c'))
    for architecture in multi.ARCHITECTURES:
        models.append(inventory('Task4c_p35_'+architecture, frozen_multi.FourierClassifier('p35',architecture,'h0'),'5ab735c0'))
    for arm in asap.ARMS:
        models.append(inventory('direct_Task35_'+arm,asap.classifier(arm)))
    models.append(inventory('Task4b_fmt_only',PathlineMulticlassClassifier3D(variant='fmt_only',fmt_dim=161)))
    models.append(inventory('new_single_center_not_trained',SingleCenterMLP()))
    assert all(len(r['convolution']) == 3 for r in models[:4])
    assert all(not r['convolution'] for r in models[4:])
    rng = np.random.default_rng(317)
    x = torch.tensor(rng.normal(size=(2,10,32,3)).cumsum(2),dtype=torch.float32)
    seeds = x[:,:,0]; counts = torch.tensor([10,8])
    definition = next(d for d in p35.candidates() if d['id']=='n0_k06')
    calls = []

    def trace(module, callback):
        function = module.pathline_dft_features_3d
        called = []
        def spy(geometry, *args, **kwargs):
            called.extend(geometry[:,0].detach().cpu().numpy())
            return function(geometry,*args,**kwargs)
        with patch.object(module,'pathline_dft_features_3d',spy):
            result = callback()
        return dict(center_evaluations=len(called), result_shape=list(result.shape)), np.array(called)

    row, centers = trace(p35, lambda:p35.encode(x[:,:7],definition))
    assert row['center_evaluations']==2 and row['result_shape']==[2,141]
    normalized,_ = p35.normalize_geometry(x[:,:7], torch.tensor([7,7]),'max_radius')
    np.testing.assert_array_equal(centers,normalized[:,0])
    calls.append(dict(name='Task35_n0_fixed_input_adapter',**row))
    row, centers = trace(p35,lambda:p35.encode(x,definition,seeds,counts))
    assert row['center_evaluations']==18 and row['result_shape']==[2,10,141]
    normalized,mask=p35.normalize_geometry(x,counts,'max_radius')
    np.testing.assert_array_equal(centers,normalized[mask])
    calls.append(dict(name='Task4c_n0_each_valid_line_adapter',**row))
    neighbors = neighbor_indices(seeds,counts,'fps6')
    row, centers = trace(multi,lambda:multi.fourier_tokens(x,counts,neighbors,'p35'))
    assert row['center_evaluations']==18 and row['result_shape']==[2,10,142]
    calls.append(dict(name='Task4c_c156_FPS_each_valid_line',**row))
    row, centers = trace(multi,lambda:asap.encode(x[:,:7].numpy(),None,'fmt_c156'))
    assert row['center_evaluations']==14 and row['result_shape']==[2,7,142]
    calls.append(dict(name='Task35_fmt_c156_all7_centers',**row))
    row, centers = trace(multi,lambda:asap.encode(x[:,:7].numpy(),np.linspace(0,1,32),'asap_fmt'))
    assert row['center_evaluations']==14 and row['result_shape']==[2,7,142]
    calls.append(dict(name='Task35_asap_fmt_camera_then_all7_centers',**row))
    # Independently demonstrate the linear operator inside TemporalDFT.
    signal=rng.normal(size=32); weights=rng.normal(size=17)+1j*rng.normal(size=17)
    weights[[0,-1]]=weights[[0,-1]].real
    spectral=np.fft.irfft(np.fft.rfft(signal,norm='ortho')*weights,n=32,norm='ortho')
    kernel=np.fft.irfft(weights,n=32)
    circular=sum(kernel[k]*np.roll(signal,k) for k in range(32))
    error=float(np.abs(spectral-circular).max()); assert error<1e-12
    checks=dict(status='PASS', no_training=True, no_weights_written=True, models=models, center_traces=calls,
                temporal_dft_circular_convolution_max_error=error,
                historical_classifier_sha256=hashlib.sha256(frozen).hexdigest(),
                scope='Module constructors and synthetic feature calls. These checks do not certify any historical run completed or used a particular branch.')
    (OUT/'architecture_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    with (OUT/'pooling_recipes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','feature_dimensions','kind','definition','center_policy','convolution'])
        w.writeheader()
        for c in pooling_candidates():
            w.writerow(dict(id=c['id'],feature_dimensions=c['feature_dimensions'],kind=c['kind'],definition=json.dumps(c),
                            center_policy='not_specified_by_pooling_recipe; see task adapter',convolution='none_in_pooling'))
    (OUT/'normalization_recipes.json').write_text(json.dumps(p35.candidates(),indent=2))
    (OUT/'training_profiles.json').write_text(json.dumps(PROFILES,indent=2))
    from experiments import Task4C_FPSAugmentSearch_1_1 as candidate_driver
    spec=json.loads((ROOT/'config/Ablation_Task4C_FPSAugmentSearch_1.1.json').read_text())
    # Historical census, not a way to launch the retired graph candidate.
    with patch.object(candidate_driver.method,'parameter_count',frozen_multi.parameter_count):
        expanded=candidate_driver.candidates(spec)
    for c in expanded:
        c.update(center_policy='every_valid_line',neighbors='fps6',encoder_dimensions=multi.POOLS[c['pool']]['feature_dimensions'],
                 standard_convolution=False,graph_message_passing=c['architecture']=='fps_graph',
                 current_status='retired_graph_convolution' if c['architecture']=='fps_graph' else 'historical_multicenter_not_singlecenter')
    assert expanded[156]['architecture']=='wide_residual' and expanded[156]['parameters']==992386
    with (OUT/'task4c_all_264_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(expanded[0]));w.writeheader();w.writerows(expanded)
    print(json.dumps(dict(status=checks['status'],model_variants=len(models),center_traces=calls,spectral_error=error,task4_candidates=len(expanded)),indent=2))


if __name__=='__main__':
    main()
