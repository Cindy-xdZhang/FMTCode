"""Reproduce one frozen FMT model in memory and export real curve attributions."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
from types import FunctionType

import numpy as np
import torch

from experiments import Task4C_InstanceCoverage_9_15_v2 as engine
from experiments.Visualize_Task4C_Bundles_3D import diverse_centers
from FMT_Utils import Task4C_Saliency_1_1 as saliency
from FMT_Utils.FMT_P35_NormFrequency_3_1 import encode, candidates, apply_normalizer

CONFIG = 'config/Other_Task4C_Saliency_1.1.json'
sha, write = engine.sha, engine.write


def load_spec(config):
    spec = json.loads(Path(config).read_text())
    assert sha(spec['base_config']) == spec['base_config_sha256']
    assert spec['seed'] == 96611 and spec['reference_selected_epoch'] == 56
    assert spec['patch_intervals'] == [list(w) for w in saliency.WINDOWS]
    return spec


def identity(config):
    files = [__file__, 'FMT_Utils/Task4C_Saliency_1_1.py',
             'experiments/Task4C_InstanceCoverage_9_15_v2.py',
             'FMT_Utils/FMT_P35_NormFrequency_3_1.py', 'FMT_Utils/DFT_FMT_3D.py',
             'FMT_Utils/FMT_V8_Search_2_1.py', 'FMT_Utils/Task4C_LinePooling_4_3.py',
             'FMT_Utils/Task4C_Encoders_4_15.py', 'ibex_bash/task4c_saliency_1p1.sh']
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                config_sha256=sha(config), sources={str(p):sha(p) for p in files},
                host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'))


def preflight(spec, config, device):
    engine.deterministic(device)
    torch.manual_seed(91641)
    t = torch.linspace(-1, 1, 32, device=device)
    g = torch.randn((2, 27, 32, 3), device=device) * .02
    g[..., 0] += t
    g[..., 1] += .3 * torch.sin(3*t)
    counts = torch.tensor([10, 19], device=device)
    g[0, 10:] = 0; g[1, 19:] = 0
    seeds = torch.randn((2, 27, 3), device=device)
    ids = saliency.frozen_neighbors(seeds, counts)
    definition = next(c for c in candidates() if c['id'] == 'n0_k06')
    original = encode(g, definition, seeds, counts)
    norm = np.zeros(141), np.ones(141)
    actual = saliency.differentiable_tokens(g, counts, ids, norm)
    assert torch.equal(actual[..., :141], original)
    model = engine.task4_model(141, 'h0').to(device).eval()
    gradient, _, _ = saliency.point_gradients(model, g, counts, ids, norm)
    assert torch.count_nonzero(gradient[0, 10:]) == 0
    assert torch.count_nonzero(gradient[1, 19:]) == 0
    direction = torch.randn_like(g)
    direction[0, 10:] = 0; direction[1, 19:] = 0
    direction /= direction.norm()
    automatic = float((gradient * direction).sum())
    errors = []
    with torch.no_grad():
        for step in (1e-3, 3e-4, 1e-4):
            plus = saliency.evaluate(model, g + step*direction, counts, ids, norm)[0].sum()
            minus = saliency.evaluate(model, g - step*direction, counts, ids, norm)[0].sum()
            finite = float((plus-minus)/(2*step))
            errors.append(dict(step=step, finite=finite, absolute_error=abs(finite-automatic)))
    assert min(e['absolute_error'] for e in errors) < max(.002, abs(automatic)*.05), errors
    modified = saliency.straighten(g[0], 2, 4, 12)
    assert torch.equal(modified[:2], g[0, :2]) and torch.equal(modified[3:], g[0, 3:])
    assert torch.equal(modified[2, [4, 12]], g[0, 2, [4, 12]])
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
    zero = saliency.attribute_bundle(model, g[0], 10, ids[0], norm, 123, smooth_samples=2)
    for key in ('gradient', 'smooth_gradient', 'patch_margin_delta', 'patch_probability_delta'):
        assert np.count_nonzero(zero[key]) == 0, key
    result = dict(complete=True, identity=identity(config), device=device,
                  gpu=torch.cuda.get_device_name() if device=='cuda' else None,
                  original_forward_exact=True, padding_gradient_zero=True,
                  gradient_directional_derivative=automatic, finite_difference_checks=errors,
                  zero_model_zero_maps=True, straightening_endpoints_and_other_lines_unchanged=True)
    write(Path(spec['output'])/f'preflight_{device}.json', result)
    print(json.dumps(result), flush=True)


def replay(spec, config):
    engine.deterministic('cuda')
    root = Path(spec['output']); source = Path(spec['source_output'])
    ref_folder = source/'runs/p35'/f"seed{spec['seed']}"
    reference = json.loads((ref_folder/'result.json').read_text())
    assert reference['identity']['git_commit'] == spec['source_scientific_commit']
    assert reference['identity']['config_sha256'] == spec['base_config_sha256']
    assert reference['selected_epoch'] == spec['reference_selected_epoch']
    for file, digest in reference['identity']['sources'].items():
        local = Path(file)
        if local.is_absolute():
            local = Path('experiments') / local.name
        assert sha(local) == digest, f'Frozen source changed: {local}'
    base = json.loads(Path(spec['base_config']).read_text())
    # Read-only reuse: no symlinks, re-encoding, new seeds, or modified source files.
    source_spec = dict(base, output=str(source))
    physical_hashes = {}
    for flow in base['flows']:
        folder = source/'physical'/flow['name']
        report = json.loads((folder/'preparation.json').read_text())
        for role, item in report['splits'].items():
            for filename, digest in item['files'].items():
                assert sha(folder/role/filename) == digest
                physical_hashes[f"{flow['name']}/{role}/{filename}"] = digest
    replay_spec = copy.deepcopy(base)
    replay_spec.update(output=str(root), version=spec['version'], methods=['p35'])
    replay_spec['training']['seeds'] = [spec['seed']]
    replay_spec['training']['epochs'] = spec['reference_selected_epoch']
    write(root/'replay.lock.json', dict(identity=identity(config), reference_result_sha256=sha(ref_folder/'result.json'),
          reference_selected_epoch=56, physical_hashes=physical_hashes, training_function_unchanged=True,
          stop_rule='stop_at_previously_selected_epoch_no_new_selection', replay_spec=replay_spec))
    captured = {}
    def factory(*args, **kwargs):
        model = engine.task4_model(*args, **kwargs)
        captured['model'] = model
        return model
    def parts(unused_spec, role, method):
        return engine.load_parts(source_spec, role, method)
    fn = engine.train
    FunctionType(fn.__code__, dict(engine.__dict__, task4_model=factory, load_parts=parts,
                 identity=identity), argdefs=fn.__defaults__)(replay_spec, config, 0)
    folder = root/'runs/p35'/f"seed{spec['seed']}"
    result = json.loads((folder/'result.json').read_text())
    assert result['selected_epoch'] == reference['selected_epoch']
    comparisons = {}
    for role in engine.SPLITS:
        old_file = ref_folder/f'{role}_predictions.npz'
        assert sha(old_file) == reference['predictions'][role]
        with np.load(old_file) as old, np.load(folder/f'{role}_predictions.npz') as new:
            for key in ('labels', 'flow_index', 'instance', 'row_in_split'):
                assert np.array_equal(old[key], new[key]), (role, key)
            error = float(np.abs(old['probability']-new['probability']).max())
            assert error <= 1e-6, (role, error)
            assert np.array_equal(old['probability']>=.5, new['probability']>=.5)
            comparisons[role] = dict(max_probability_error=error, all_decisions_equal=True,
                                    source_sha256=sha(old_file), replay_sha256=sha(folder/f'{role}_predictions.npz'))
    write(root/'reproduction.json', dict(complete=True, identity=identity(config), comparisons=comparisons,
          selected_epoch=result['selected_epoch'], test_f1=result['metrics']['test']['combined']['f1']))
    model = captured['model'].eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    norm = np.asarray(result['normalization']['mean']), np.asarray(result['normalization']['std'])
    export(spec, config, model, norm, reference)


def export(spec, config, model, norm, reference):
    root = Path(spec['output']); source = Path(spec['source_output'])
    destination = root/'package'; destination.mkdir(parents=True, exist_ok=False)
    records = []; diagnostics = []
    with np.load(source/'runs/p35'/f"seed{spec['seed']}"/'test_predictions.npz') as z:
        predictions = {k:z[k] for k in z.files}
    randomized = copy.deepcopy(model)
    torch.manual_seed(91699)
    for module in randomized.modules():
        if hasattr(module, 'reset_parameters'): module.reset_parameters()
    randomized.eval()
    for fi, flow in enumerate(('channel', 'tbl')):
        folder = source/'physical'/flow/'test'
        with np.load(folder/'metadata.npz') as z: meta = {k:z[k] for k in z.files}
        geometry = np.load(folder/'geometry.npy', mmap_mode='r')
        seeds = np.load(folder/'seeds.npy', mmap_mode='r')
        cached = np.load(source/'encoded'/flow/'test/p35.npy', mmap_mode='r')
        chosen_flow = predictions['flow_index'] == fi
        assert np.array_equal(predictions['row_in_split'][chosen_flow], np.arange(len(geometry)))
        probability = predictions['probability'][chosen_flow]
        eligible = np.flatnonzero(probability >= spec['threshold'])
        selected = eligible[diverse_centers(meta['center'][eligible], spec['examples_per_flow'], spec['selection_seed']+fi)]
        neighbor_cache = {}
        for number, row in enumerate(selected):
            first = int(row)//32*32
            if first not in neighbor_cache:
                ss = torch.as_tensor(np.array(seeds[first:first+32]), device='cuda')
                cc = torch.as_tensor(meta['counts'][first:first+32].astype(np.int64), device='cuda')
                neighbor_cache[first] = saliency.frozen_neighbors(ss, cc)
            neighbors = neighbor_cache[first][int(row)-first]
            count = int(meta['counts'][row])
            g = torch.as_tensor(np.array(geometry[row]), device='cuda')
            counts = torch.tensor([count], device='cuda')
            with torch.no_grad():
                actual = saliency.differentiable_tokens(g[None], counts, neighbors[None], norm)
                target = apply_normalizer(cached[row:row+1,...,:141],
                      dict(kind='zscore',mean=norm[0],std=norm[1]), cached[row:row+1,...,-1:])
                token_error = float(np.abs(actual[0,...,:141].cpu().numpy()-target[0]).max())
                p = float(model(actual).softmax(-1)[0,1])
            assert token_error <= 1e-4, (flow, int(row), token_error)
            assert abs(p-float(probability[row])) <= 1e-4, (flow, int(row), p, probability[row])
            values = saliency.attribute_bundle(model, g, count, neighbors, norm,
                spec['selection_seed']+fi*100000+int(row), spec['smooth_samples'],
                spec['noise_sigma_normalized_radius'])
            if number < 8:
                grad, _, _ = saliency.point_gradients(randomized, g[None], counts, neighbors[None], norm)
                from scipy.stats import spearmanr
                rho = float(spearmanr(values['gradient'][:count].ravel(), grad[0,:count].norm(dim=-1).cpu().numpy().ravel()).statistic)
                diagnostics.append(dict(flow=flow,row=int(row),randomized_weight_gradient_rank_correlation=rho))
            name = f'{flow}_{int(row):05d}'
            physical = np.asarray(geometry[row], np.float64)*float(meta['radius'][row])+meta['centroid'][row]
            physical[count:] = 0
            np.savez_compressed(destination/f'{name}.npz', geometry=physical, normalized_geometry=np.array(geometry[row]),
                neighbors=neighbors.cpu().numpy(), count=count, **values)
            entry = dict(id=name, flow=flow, row=int(row), split='test', count=count,
                label=int(meta['labels'][row]), instance=int(meta['instance'][row]),
                head_component=int(meta['head_component'][row]), scale_id=int(meta['scale_id'][row]),
                center=meta['center'][row].tolist(), radius=float(meta['radius'][row]),
                probability=float(probability[row]), reproduced_probability=p, margin=values['margin'],
                feature_max_error=token_error, file=f'{name}.npz', sha256=sha(destination/f'{name}.npz'))
            records.append(entry)
            if number%20==0: print('attribution',flow,number+1,len(selected),name,flush=True)
    manifest = dict(complete=True, version=spec['version'], identity=identity(config), config=spec,
        model=dict(name='FMT p35 / n0_k06',seed=spec['seed'],parameters=76738,selected_epoch=56,
                   source_version='Ablation_Task4C_BottomDensity_1.2',source_commit=spec['source_scientific_commit'],
                   seed_test_metrics=reference['metrics']['test'],three_seed_test_f1_mean=.8884042765671557),
        records=records, randomization_diagnostics=diagnostics,
        reproduction_sha256=sha(root/'reproduction.json'), weights_saved=False)
    write(destination/'manifest.json',manifest)
    print(json.dumps(dict(complete=True,bundles=len(records),output=str(destination))),flush=True)


def runtime(spec, config, phase, state, code):
    row = dict(version=spec['version'],time=datetime.now(timezone.utc).isoformat(),phase=phase,state=state,
               exit_code=code,**identity(config))
    if torch.cuda.is_available(): row['gpu']=torch.cuda.get_device_name()
    engine.append_locked(Path(spec['output'])/'runtime.jsonl',json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def submit(spec, config):
    root=Path(spec['output']).resolve();root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    dependency=None
    for phase,limit in [('preflight','00:20:00'),('replay','04:00:00')]:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=48G',
            '--gres=gpu:1','--constraint=v100','--time='+limit,'--job-name=t4c-saliency-'+phase,
            f'--output={root}/logs/{phase}_%j.out',f'--error={root}/logs/{phase}_%j.err']
        if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        command+=['ibex_bash/task4c_saliency_1p1.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,version=spec['version'],phase=phase,submitted_at_utc=datetime.now(timezone.utc).isoformat(),
                 expected_device='V100',command=command,dependency=dependency,identity=identity(config))
        engine.append_locked(root/'submissions.jsonl',json.dumps(row)+'\n')
        engine.append_locked('docs/ibex_run_registry.md','\n- '+json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);dependency=job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','replay','submit','runtime'))
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--device',default='cuda')
    parser.add_argument('--runtime-phase');parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=load_spec(args.config)
    if args.phase=='preflight':preflight(spec,args.config,args.device)
    elif args.phase=='runtime':runtime(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    else:globals()[args.phase](spec,args.config)


if __name__=='__main__':main()
