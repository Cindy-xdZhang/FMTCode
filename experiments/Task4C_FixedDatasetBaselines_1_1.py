"""Non-FMT baselines on the Task4-c fixed bundle dataset v1 (mainExp_Task4C_FixedDatasetBaselines_1.1).

Families (each in its own output root with a read-only ``physical`` symlink to the frozen data):
  conv       : frozen engine p35 (rotating-center reference), Conv3D 16^3 and 24^3 (72,192 parameters)
  bilstm     : BiLSTM + MLP (76,786 parameters) from Ablation_Task4C_GeometricBaselines_1.1
  pointnet   : reduced-width PointNet (76,749) from Ablation_Task4C_PointNet_1.1
  pointnetpp : reduced-width PointNet++ SSG (76,723) from Ablation_Task4C_PointNetPlusPlus_1.1
The frozen model, encoding, pilot and training functions of those versions are executed unchanged
(only ``identity`` is rebound); the data root, counts and seeds come from this configuration.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
from types import FunctionType
import numpy as np
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine
from experiments import Task4C_GeometricBaselines_1_1 as geometry
from experiments import Task4C_PointNet_1_1 as pointnet
from experiments import Task4C_PointNetPlusPlus_1_1 as pointnetpp
from FMT_Utils import Task4C_FixedDataset_1_1 as data

sha, write = engine.sha, engine.write
CONFIG = 'config/mainExp_Task4C_FixedDatasetBaselines_1.1.json'
SPLITS = ('train', 'validation', 'test')
MODULES = dict(conv=engine, bilstm=geometry, pointnet=pointnet, pointnetpp=pointnetpp)
FILES = ('experiments/Task4C_FixedDatasetBaselines_1_1.py', 'experiments/Task4C_InstanceCoverage_9_15_v2.py', 'FMT_Utils/Task4C_InstanceCoverage_9_15_v2.py',
         'FMT_Utils/Task4C_InstanceCoverage_9_15.py', 'experiments/Task4C_GeometricBaselines_1_1.py', 'FMT_Utils/Task4C_GeometricBaselines_1_1.py',
         'experiments/Task4C_PointNet_1_1.py', 'FMT_Utils/Task4C_PointNet_1_1.py', 'experiments/Task4C_PointNetPlusPlus_1_1.py', 'FMT_Utils/Task4C_PointNetPlusPlus_1_1.py',
         'FMT_Utils/Task4C_PaperBundles_3_1.py', 'FMT_Utils/FMT_P35_NormFrequency_3_1.py', 'FMT_Utils/Task4C_FixedDataset_1_1.py',
         'config/Ablation_Task4C_BottomDensity_1.2.json', 'config/Ablation_Task4C_GeometricBaselines_1.1.json', 'config/Ablation_Task4C_PointNet_1.1.json',
         'config/Ablation_Task4C_PointNetPlusPlus_1.1.json', 'ibex_bash/task4c_fixed_dataset_baselines_1p1.sh')


def identity(config):
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), config_sha256=sha(config), sources={p: sha(p) for p in FILES})


def load_spec(config, family):
    """Frozen training/encoding settings of BottomDensity 1.2 plus the family's own model block; data keys from this config."""
    definition = json.loads(Path(config).read_text()); fam = definition['families'][family]
    spec = json.loads(Path(definition['base_config']).read_text())
    for key in ('version', 'execution_revision', 'source_output', 'source_audit_sha256', 'source_scientific_commit', 'dataset_policy', 'flows'): spec[key] = definition[key]
    spec['expected_counts'] = dict(definition['expected_counts'], total=sum(definition['expected_counts'].values()))
    spec['training'] = dict(spec['training'], seeds=definition['seeds']); spec['encoding'] = dict(spec['encoding'], voxel_resolutions=definition['voxel_resolutions'])
    spec['output'] = str(Path(definition['output'])/family); spec['methods'] = fam['methods']; spec['family'] = family; spec['_root_output'] = definition['output']
    if fam.get('model_config'):
        extra = json.loads(Path(fam['model_config']).read_text())
        for key in fam['model_keys']: spec[key] = extra[key]
    return spec


def run(module, name, *args):
    fn = getattr(module, name)
    return FunctionType(fn.__code__, dict(module.__dict__, identity=identity), argdefs=fn.__defaults__)(*args)


def reuse(spec, config):
    """Verify the frozen data set against its audit hashes and expose it read-only under this family root."""
    root = Path(spec['output']); src = Path(spec['source_output']); audit = json.loads((src/'data_audit.json').read_text())
    assert audit['complete'] and sha(src/'data_audit.json') == spec['source_audit_sha256'] and audit['identity']['git_commit'] == spec['source_scientific_commit']
    counts = {s: 0 for s in SPLITS}; hashes = {}
    for flow in spec['flows']:
        name = flow['name']; folder = src/'physical'/name; report = json.loads((folder/'preparation.json').read_text())
        assert report['complete'] and not report['pilot'] and report['identity']['git_commit'] == spec['source_scientific_commit']
        for split in SPLITS:
            entry = report['splits'][split]; assert entry['files'] == audit['frozen_files'][f'{name}/{split}']
            for filename, digest in entry['files'].items(): assert sha(folder/split/filename) == digest; hashes[f'{name}/{split}/{filename}'] = digest
            with np.load(folder/split/'metadata.npz') as z: assert len(z['labels']) == entry['samples'] and np.all(z['counts'] >= 17)
            counts[split] += entry['samples']
    assert counts == {s: spec['expected_counts'][s] for s in SPLITS}
    root.mkdir(parents=True, exist_ok=True); (root/'physical').symlink_to((src/'physical').resolve(), target_is_directory=True)
    write(root/'data_audit.json', dict(complete=True, identity=identity(config), counts=counts, source_output=str(src), source_commit=spec['source_scientific_commit'],
                                       source_data_audit_sha256=sha(src/'data_audit.json'), physical_files=hashes, read_only_symlink=True))


def summarize(spec, config):
    """Recompute test scores of every run; add held-out / covered-instance breakdowns from the frozen index."""
    from sklearn.metrics import f1_score
    root = Path(spec['output']); src = Path(spec['source_output']); seeds = spec['training']['seeds']; records = []
    heldout = np.concatenate([np.load(src/'physical'/f['name']/'test'/data.INDEX_FILE)['heldout_instance'] for f in spec['flows']])
    offset = np.concatenate([np.load(src/'physical'/f['name']/'test'/data.INDEX_FILE)['offset_test'] for f in spec['flows']])
    labels = np.concatenate([np.load(src/'physical'/f['name']/'test'/'metadata.npz')['labels'] for f in spec['flows']]).astype(np.int64)
    for method in spec['methods']:
        for seed in seeds:
            folder = root/'runs'/method/f'seed{seed}'; r = json.loads((folder/'result.json').read_text())
            assert r['complete'] and r['identity'] == identity(config) and r['method'] == method and r['seed'] == seed
            lock = json.loads((folder/'selection.lock.json').read_text()); assert lock['selected_epoch'] == r['selected_epoch'] and lock['test_loaded'] is False
            scores = {}
            for role in SPLITS:
                file = folder/f'{role}_predictions.npz'; assert sha(file) == r['predictions'][role]
                with np.load(file) as z: p = {k: z[k] for k in z.files}
                assert len(p['labels']) == spec['expected_counts'][role] and float(p['threshold']) == .5
                order = np.lexsort((p['row_in_split'], p['flow_index'])); y = p['labels'][order]; prob = p['probability'][order]
                assert abs(float(f1_score(y, prob >= .5))-r['metrics'][role]['combined']['f1']) < 1e-12
                if role == 'test':
                    assert np.array_equal(y, labels)
                    scores[role] = dict(combined=engine.metrics(y, prob), heldout_instances=engine.metrics(y[heldout], prob[heldout]),
                                        covered_instances=engine.metrics(y[~heldout], prob[~heldout]),
                                        covered_offset_positive_recall=float(np.mean(prob[offset] >= .5)) if offset.any() else None,
                                        per_flow={f['name']: engine.metrics(p['labels'][p['flow_index'] == i], p['probability'][p['flow_index'] == i]) for i, f in enumerate(spec['flows'])})
                else: scores[role] = dict(combined=r['metrics'][role]['combined'])
            records.append(dict(method=method, seed=seed, parameters=r['parameters'], epochs=r['epochs'], selected_epoch=r['selected_epoch'],
                                training_seconds=r['training_seconds'], scores=scores, result_sha256=sha(folder/'result.json')))
    summary = {}
    for method in spec['methods']:
        rows = [x for x in records if x['method'] == method]
        def agg(f):
            v = np.array([f(x) for x in rows], float); return dict(mean=float(v.mean()), sample_std=float(v.std(ddof=1)) if len(v) > 1 else None, values=v.tolist())
        summary[method] = dict(parameters=rows[0]['parameters'], seeds=[x['seed'] for x in rows],
                               test_f1=agg(lambda x: x['scores']['test']['combined']['f1']), test_ap=agg(lambda x: x['scores']['test']['combined']['average_precision']),
                               heldout_f1=agg(lambda x: x['scores']['test']['heldout_instances']['f1']), covered_f1=agg(lambda x: x['scores']['test']['covered_instances']['f1']),
                               training_minutes=agg(lambda x: x['training_seconds']/60))
    assert not list(root.rglob('*.pt')) and not list(root.rglob('*.pth'))
    write(root/'summary.json', dict(complete=True, identity=identity(config), family=spec['family'], methods=records, summary=summary, no_weight_files=True))


def runtime(spec, config, phase, state, code):
    engine.append_locked(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(config), family=spec['family'], phase=phase, state=state, exit_code=code,
        job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        hostname=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


PLANS = dict(
    conv=[('preflight', None, True, '00:20:00'), ('reuse', None, False, '00:30:00'), ('encode', '0-1', True, '03:00:00'), ('train', None, True, '08:00:00'), ('summarize', None, False, '00:30:00')],
    bilstm=[('preflight', None, True, '00:20:00'), ('reuse', None, False, '00:30:00'), ('pilot', None, True, '00:30:00'), ('train', None, True, '24:00:00'), ('summarize', None, False, '00:30:00')],
    pointnet=[('preflight', None, True, '00:20:00'), ('reuse', None, False, '00:30:00'), ('pilot', None, True, '00:30:00'), ('train', None, True, '24:00:00'), ('summarize', None, False, '00:30:00')],
    pointnetpp=[('preflight', None, True, '00:20:00'), ('reuse', None, False, '00:30:00'), ('pilot', None, True, '00:30:00'), ('encode', '0-1', True, '04:00:00'), ('train', None, True, '2-00:00:00'), ('summarize', None, False, '00:30:00')])


def submit(definition_path):
    definition = json.loads(Path(definition_path).read_text()); root = Path(definition['output']); (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(); jobs = {}
    for family in definition['families']:
        spec = load_spec(definition_path, family); runs = len(spec['methods'])*len(spec['training']['seeds']); previous = None; jobs[family] = {}
        for phase, array, gpu, wall in PLANS[family]:
            if phase == 'train': array = f"0-{runs-1}%{definition['families'][family]['concurrency']}"
            command = ['sbatch', '--parsable', '--propagate=NONE', f'--job-name=FixedBL11_{family}_{phase}', '--cpus-per-task=4', '--mem=48G', '--time='+wall,
                       '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
            if previous: command += ['--dependency=afterok:'+previous, '--kill-on-invalid-dep=yes']
            if array: command += ['--array='+array]
            if gpu: command += ['--gres=gpu:1', '--constraint=v100', '--exclude=gpu213-18']
            command += ['ibex_bash/task4c_fixed_dataset_baselines_1p1.sh', phase, family, definition_path]
            job = subprocess.check_output(command, text=True).strip().split(';')[0]; jobs[family][phase] = job; previous = job
            engine.append_locked(root/'submissions.jsonl', json.dumps(dict(family=family, phase=phase, job=job, command=command, submitted_at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        write(root/'submission.json', dict(identity=identity(definition_path), jobs=jobs))
    print(json.dumps(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['preflight', 'reuse', 'pilot', 'encode', 'train', 'summarize', 'submit', 'runtime'])
    p.add_argument('family', choices=list(MODULES)+['all']); p.add_argument('--config', default=CONFIG); p.add_argument('--index', type=int, default=0)
    p.add_argument('--device', default='cuda'); p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    a = p.parse_args()
    if a.phase == 'submit': submit(a.config); return
    spec = load_spec(a.config, a.family); module = MODULES[a.family]
    if a.phase == 'reuse': reuse(spec, a.config)
    elif a.phase == 'summarize': summarize(spec, a.config)
    elif a.phase == 'runtime': runtime(spec, a.config, a.runtime_phase, a.state, a.exit_code)
    elif a.phase == 'preflight': run(module, 'preflight', spec, a.config, a.device)
    elif a.phase == 'pilot': run(module, 'pilot', spec, a.config)
    elif a.phase == 'encode': run(module, 'encode', spec, a.config, a.index)
    elif a.phase == 'train': run(module, 'train', spec, a.config, a.index)


if __name__ == '__main__': main()
