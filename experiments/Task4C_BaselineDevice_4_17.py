"""Verify the single mismatched-device baseline; retain every original result."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments import Task4C_RetainDirection_4_16 as run
from experiments import Audit_Task4C_Encoders_4_15 as metrics

CONFIG = 'config/Verify_Task4C_BaselineDevice_4.17.json'


def specs():
    config = json.loads(Path(CONFIG).read_text())
    assert run.pipeline.sha(config['base_config']) == config['base_config_sha256']
    base = run.load_spec(config['base_config'])
    spec = copy.deepcopy(base)
    spec.update(version=config['version'], output=config['output'], encoders=[config['encoder']])
    spec['training']['seeds'] = [config['seed']]
    return config, spec, base


def verify(config, spec, base):
    assert torch.cuda.get_device_name(0) == config['required_device']
    local = run.variant_spec(spec, config['encoder']); destination = Path(local['output'])
    destination.mkdir(parents=True, exist_ok=False)
    parent = Path(base['parent_output']); original_manifest = json.loads((parent/'encoding.json').read_text())
    manifest = dict(identity=run.identity(CONFIG), splits={})
    for flow in spec['flows']:
        for split in run.SPLITS:
            key = flow['name']+'/'+split; dst = destination/key; dst.mkdir(parents=True)
            files = {}
            for name in ('fmt.npy', 'metadata.npz'):
                source = parent/key/name
                assert run.pipeline.sha(source) == original_manifest['splits'][key]['files'][name]
                (dst/name).symlink_to(source.resolve()); files[name] = run.pipeline.sha(source)
            manifest['splits'][key] = dict(files=files)
    run.pipeline.write(destination/'encoding.json', manifest)
    run.train(spec, CONFIG, 0)
    fresh_path = run.folder(spec, config['encoder'], config['seed'])/'result.json'
    fresh = json.loads(fresh_path.read_text())
    original_path = parent/'runs'/f"regularized_fmt_mlp_seed{config['seed']}"/'result.json'
    original = json.loads(original_path.read_text())
    assert fresh['normalization'] == original['normalization']
    assert fresh['selected_epoch'] == original['selected_epoch']
    with np.load(fresh_path.parent/'predictions.npz') as z:
        for split, field in (('train', 'training'), ('validation', 'validation'), ('test', 'test')):
            assert fresh[field]['pooled']['f1'] == original[field]['pooled']['f1']
            assert metrics.metric(z[split+'_labels'], z[split+'_probability'])['f1'] == original[field]['pooled']['f1']
    assert fresh['weights_written'] == 0
    finalize(config, spec, base, fresh)


def finalize(config, spec, base, reproduced):
    source = Path(config['source_output']); root = Path(spec['output']); parent = Path(base['parent_output'])
    prior = json.loads((source/'prediction_metrics_audit.json').read_text())
    assert prior['status'] == 'PASS' and len(prior['runs']) == 9
    summary = json.loads((source/'summary.json').read_text())
    assert summary['identity']['commit'] == config['source_commit']
    assert prior['summary_sha256'] == run.pipeline.sha(source/'summary.json')
    original = json.loads((parent/'summary.json').read_text())
    assert run.pipeline.sha(parent/'summary.json') == base['parent_summary_sha256']
    all_base = copy.deepcopy(base); all_base['output'] = str(source)
    retained = []
    for flow in base['flows']:
        for split in run.SPLITS:
            key = flow['name']+'/'+split
            old = np.load(parent/key/'fmt.npy', mmap_mode='r')
            v5 = np.load(source/'variants/fmt_v5_4_14'/key/'fmt.npy', mmap_mode='r')
            obj = np.load(source/'variants/fmt_objective_ntod_v2_4_14'/key/'fmt.npy', mmap_mode='r')
            for start in range(0,len(old),256):
                sl=slice(start,start+256)
                assert np.array_equal(old[sl,:,:233],v5[sl,:,:233])
                assert np.array_equal(old[sl,:,161:233],obj[sl,:,231:303])
                assert np.array_equal(v5[sl,:,233:398],obj[sl,:,303:468])
                assert np.array_equal(old[sl,:,-1],v5[sl,:,-1])
                assert np.array_equal(old[sl,:,-1],obj[sl,:,-1])
            retained.append(dict(split=key,rows=len(old),retained_blocks_exact=True))
    references=[]
    for seed in base['training']['seeds']:
        old=json.loads((parent/'runs'/f'regularized_fmt_mlp_seed{seed}'/'result.json').read_text())
        ref=reproduced if seed==config['seed'] else json.loads((run.folder(all_base,'fmt_4_14',seed)/'result.json').read_text())
        assert ref['normalization']==old['normalization'] and ref['device']==config['required_device']
        assert all(ref[s]['pooled']['f1']==old[s]['pooled']['f1'] for s in ('training','validation','test'))
        for encoder in ('fmt_v5_4_14','fmt_objective_ntod_v2_4_14'):
            value=json.loads((run.folder(all_base,encoder,seed)/'result.json').read_text())
            assert value['device']==config['required_device']
            for stat in ('mean','std'):
                actual=value['normalization'][stat]
                if encoder=='fmt_v5_4_14':
                    assert np.array_equal(actual[:233],old['normalization'][stat])
                else:
                    assert np.array_equal(actual[231:303],old['normalization'][stat][161:233])
        conv_path=parent/'runs'/f'regularized_conv3d_mlp_seed{seed}'/'result.json'
        conv=json.loads(conv_path.read_text())
        assert conv['predictions_sha256']==run.pipeline.sha(conv_path.parent/'predictions.npz')
        with np.load(conv_path.parent/'predictions.npz') as z:
            for split,field in (('train','training'),('validation','validation'),('test','test')):
                assert metrics.metric(z[split+'_labels'],z[split+'_probability'])['f1']==conv[field]['pooled']['f1']
        references.append(dict(seed=seed,baseline_normalization_and_f1_exact=True,
            conv3d_prediction_sha256=conv['predictions_sha256']))
    rows=[]
    for r in original['rows']:
        if r['method']=='fmt_mlp': rows.append(dict(r,encoder='fmt_4_14',reference_reused=True))
    rows += [r for r in summary['rows'] if r['encoder']!='fmt_4_14']
    for r in original['rows']:
        if r['method']=='conv3d_mlp': rows.append(dict(r,encoder='conv3d',reference_reused=True))
    report=dict(status='PASS',version=config['version'],identity=run.identity(CONFIG),
        source_summary_sha256=run.pipeline.sha(source/'summary.json'),
        source_prediction_audit_sha256=run.pipeline.sha(source/'prediction_metrics_audit.json'),
        rows=rows,retained_feature_checks=retained,reference_checks=references,
        original_fmt_baseline_reproduced_on_original_device=True,
        prior_a100_run_preserved_not_used_to_replace_frozen_reference=True,
        all_new_method_runs_are_original_prespecified_v100_runs=True,
        fixed_threshold=.5,no_hyperparameter_changes=True)
    run.pipeline.write(root/'validated_comparison.json',report)
    print(json.dumps(rows),flush=True)


def submit(config, spec):
    from experiments.Record_Task1235_AngleFeatures_1_1 import append_record
    root=Path(spec['output']);(root/'logs').mkdir(parents=True,exist_ok=True)
    command=['sbatch','--parsable','--cpus-per-task=4','--mem=64G','--time=01:00:00',
        '--gres=gpu:1','--constraint=v100','--job-name=t4c417-baseline',
        '--dependency=afterany:'+config['after_job'],
        '--output='+str(root/'logs/verify.%j.out'),'--error='+str(root/'logs/verify.%j.err'),
        'ibex_bash/task4c_baseline_device_4p17.sh']
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    r=dict(version=config['version'],phase='baseline_device_verify',job_id=job,command=command,
        submitted_at_utc=datetime.now(timezone.utc).isoformat(),config=CONFIG,
        expected_device=config['required_device'],**run.identity(CONFIG))
    append_record(root/'submissions.jsonl',r,'Task4C 4.17 submitted')
    append_record('docs/ibex_run_registry.md',r,'Task4C 4.17 submitted')
    r.pop('source_sha256');print(json.dumps(r),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('submit','runtime','verify'))
    p.add_argument('--state');p.add_argument('--exit-code',type=int);a=p.parse_args()
    config,spec,base=specs()
    if a.phase=='submit':submit(config,spec)
    elif a.phase=='runtime':run.runtime(spec,CONFIG,'train',a.state,a.exit_code)
    else:verify(config,spec,base)


if __name__=='__main__':main()
