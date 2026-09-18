"""Non-FMT baselines on the Task4-c fixed dataset v2 (mainExp_Task4C_FixedDatasetBaselines_2.1).

Families (frozen model / encoding / pilot / training code of the 1.1 versions, executed unchanged
with only ``identity`` rebound):
  conv       : Conv3D 16^3 and 24^3 (72,192 parameters)
  bilstm     : BiLSTM + MLP (76,786) from Ablation_Task4C_GeometricBaselines_1.1
  pointnet   : reduced-width PointNet (76,749) from Ablation_Task4C_PointNet_1.1
  pointnetpp : reduced-width PointNet++ SSG (76,723) from Ablation_Task4C_PointNetPlusPlus_1.1
Those engines read v1-style bundle files (``physical/<flow>/<split>/{geometry,seeds}.npy``,
``metadata.npz`` and a per-flow ``preparation.json``).  The ``export`` phase materialises them
from dataset v2 with exactly the clusters of mainExp_Task4C_FixedDatasetFMT_2.1: one row per
(sample, half length), slot 0 = the sample's own line, slots 1..16 = the FPS16 neighbours' lines
at the same half length, frozen whole-bundle normalisation, 27 slots zero padded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from types import FunctionType
import numpy as np
import torch
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine
from experiments import Task4C_GeometricBaselines_1_1 as geometry
from experiments import Task4C_PointNet_1_1 as pointnet
from experiments import Task4C_PointNetPlusPlus_1_1 as pointnetpp
from FMT_Utils import Task4C_FixedDataset_2_1 as data
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as v2
from FMT_Utils import Task4C_FixedDataset_1_1 as v1

sha, write = engine.sha, engine.write
CONFIG = 'config/mainExp_Task4C_FixedDatasetBaselines_2.1.json'
SPLITS = ('train', 'validation', 'test')
SLOTS, LINES = 27, 17
MODULES = dict(conv=engine, bilstm=geometry, pointnet=pointnet, pointnetpp=pointnetpp)
FILES = ('experiments/Task4C_FixedDatasetBaselines_2_1.py', 'FMT_Utils/Task4C_FixedDatasetFMT_2_1.py', 'FMT_Utils/Task4C_FixedDataset_2_1.py',
         'experiments/Task4C_InstanceCoverage_9_15_v2.py', 'FMT_Utils/Task4C_InstanceCoverage_9_15_v2.py', 'FMT_Utils/Task4C_InstanceCoverage_9_15.py',
         'experiments/Task4C_GeometricBaselines_1_1.py', 'FMT_Utils/Task4C_GeometricBaselines_1_1.py', 'experiments/Task4C_PointNet_1_1.py', 'FMT_Utils/Task4C_PointNet_1_1.py',
         'experiments/Task4C_PointNetPlusPlus_1_1.py', 'FMT_Utils/Task4C_PointNetPlusPlus_1_1.py', 'FMT_Utils/Task4C_PaperBundles_3_1.py', 'FMT_Utils/FMT_P35_NormFrequency_3_1.py',
         'config/Ablation_Task4C_BottomDensity_1.2.json', 'config/Ablation_Task4C_GeometricBaselines_1.1.json', 'config/Ablation_Task4C_PointNet_1.1.json',
         'config/Ablation_Task4C_PointNetPlusPlus_1.1.json', 'config/mainExp_Task4C_FixedDataset_2.1.json', 'config/mainExp_Task4C_FixedDatasetFMT_2.1.json',
         'ibex_bash/task4c_fixed_dataset_baselines_2p1.sh')


def append_line(path, text):
    try: engine.append_locked(path, text)
    except ModuleNotFoundError:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf8') as stream: stream.write(text)


def identity(config):
    # No hostname here: export, reuse and training run on different Slurm nodes and compare identities.
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), config_sha256=sha(config), sources={p: sha(p) for p in FILES})


def load_spec(config, family):
    """Frozen training/encoding settings of BottomDensity 1.2 plus the family's own model block; data keys from this config."""
    definition = json.loads(Path(config).read_text()); fam = definition['families'][family]
    spec = json.loads(Path(definition['base_config']).read_text())
    for key in ('version', 'execution_revision', 'source_output', 'source_audit_sha256', 'dataset_policy', 'flows', 'fmt_config', 'folds', 'validation', 'neighbors'): spec[key] = definition[key]
    spec['expected_counts'] = dict(definition['expected_counts'], total=sum(definition['expected_counts'].values()))
    spec['training'] = dict(spec['training'], seeds=definition['seeds']); spec['encoding'] = dict(spec['encoding'], voxel_resolutions=definition['voxel_resolutions'])
    spec['output'] = str(Path(definition['output'])/family); spec['methods'] = fam['methods']; spec['family'] = family; spec['_root_output'] = definition['output']
    spec['bundles'] = str(Path(definition['output'])/'bundles')
    if fam.get('model_config'):
        extra = json.loads(Path(fam['model_config']).read_text())
        for key in fam['model_keys']: spec[key] = extra[key]
    return spec


def run(module, name, *args):
    fn = getattr(module, name)
    return FunctionType(fn.__code__, dict(module.__dict__, identity=identity, append_locked=append_line), argdefs=fn.__defaults__)(*args)


# ---------------------------------------------------------------- export: v2 samples -> v1-style FPS16 bundles
def bundle_rows(curves, seeds, meta, neighbors, sample_ids, length_ids, steps, ds):
    """Normalised 27-slot bundles (17 valid lines) and the v1 metadata keys for the given rows."""
    B = len(sample_ids); k = LINES-1
    ids = np.concatenate((sample_ids[:, None], neighbors[sample_ids, :k]), 1)                 # [B, 17] sample ids, slot 0 = center
    lines = torch.tensor(curves[ids, length_ids[:, None]], dtype=torch.float32)               # [B, 17, 32, 3] physical
    points = torch.tensor(seeds[ids], dtype=torch.float32)
    normalized, normalized_seeds, counts = v2.normalize_bundle(lines, points)
    g = np.zeros((B, SLOTS, 32, 3), np.float32); g[:, :LINES] = normalized.numpy()
    s = np.zeros((B, SLOTS, 3), np.float32); s[:, :LINES] = normalized_seeds.numpy()
    physical = lines.double().numpy(); centroid = physical.mean((1, 2)); radius = np.sqrt(((physical-centroid[:, None, None])**2).sum(-1).max((1, 2)))
    arcs = np.full((B, SLOTS, 2), np.nan); arcs[:, :LINES] = meta['half_arc_lengths'][ids, length_ids[:, None]]
    stepc = np.zeros((B, SLOTS, 2), np.int32); stepc[:, :LINES] = meta['half_step_counts'][ids, length_ids[:, None]]
    distance = np.linalg.norm(seeds[ids[:, 1:]]-seeds[ids[:, :1]], axis=-1)
    m = dict(labels=meta['label'][sample_ids].astype(np.int64), counts=np.full(B, LINES, np.int64), head_component=meta['component'][sample_ids].astype(np.int64),
             instance=meta['instance'][sample_ids].astype(np.int64), scale_id=length_ids.astype(np.int64), center_number=sample_ids.astype(np.int64),
             source_cell=np.zeros(B, np.int64), center=seeds[sample_ids].astype(np.float64), centroid=centroid, radius=radius,
             bounds=np.stack((physical.min((1, 2)), physical.max((1, 2))), 1), neighbor_distance=distance.mean(1), seed_rms_distance=np.sqrt((distance**2).mean(1)),
             measured_mean_arc_length=np.linalg.norm(np.diff(physical, axis=2), axis=-1).sum(2).mean(1), half_arc_lengths=arcs, half_step_counts=stepc,
             ds=np.full(B, ds), maxiteration=np.asarray(steps)[length_ids].astype(np.int64), requested_half_length=np.asarray(steps)[length_ids]*ds,
             local_grid_scale=meta['local_grid_scale'][sample_ids], nearest_train_center_distance=np.full(B, np.nan), nearest_same_head_train_center_distance=np.full(B, np.nan))
    assert set(m) == set(v1.METADATA_KEYS)
    return g, s, m


def export(definition_path, index):
    """Write the FPS16 bundle files of one flow (all three splits) from dataset v2; the neighbour tables are the FMT 2.1 ones."""
    definition = json.loads(Path(definition_path).read_text()); spec = load_spec(definition_path, 'conv'); root = Path(spec['bundles'])
    source = Path(spec['source_output']); audit = json.loads((source/'data_audit.json').read_text())
    assert audit['complete'] and sha(source/'data_audit.json') == spec['source_audit_sha256']
    flow = spec['flows'][index]; name = flow['name']; folder = source/'physical'/name
    for filename, digest in audit['frozen_files'][name].items(): assert sha(folder/filename) == digest, (name, filename)
    fmt_spec = dict(output=str(root), source_output=str(source), flows=spec['flows'], neighbors=spec['neighbors'])
    tables = v2.build_neighbor_tables(fmt_spec, sha); neighbors = v2.load_neighbors(fmt_spec, name)
    seeds, curves, meta = data.load(folder); masks = v2.split_masks(meta, index, spec); K = len(flow['half_lengths'])
    steps = data.half_steps(flow['half_lengths'], flow['ds']); ds = flow['ds']
    report = dict(complete=False, pilot=False, version=spec['version'], identity=identity(definition_path), host=socket.gethostname(), flow=flow, source_flow_files=audit['frozen_files'][name],
                  neighbor_table=tables[name], cluster='slot 0 = sample line; slots 1..16 = FPS16 neighbours (64 nearest samples of the same flow), same half length; frozen normalisation; 27 slots zero padded',
                  splits={})
    for split in SPLITS:
        ids = np.flatnonzero(masks[split]); sample_ids = np.repeat(ids, K); length_ids = np.tile(np.arange(K), len(ids)); n = len(sample_ids)
        dest = root/'physical'/name/split; dest.mkdir(parents=True, exist_ok=False)
        g_out = np.lib.format.open_memmap(dest/'geometry.npy', mode='w+', dtype=np.float32, shape=(n, SLOTS, 32, 3))
        s_out = np.lib.format.open_memmap(dest/'seeds.npy', mode='w+', dtype=np.float32, shape=(n, SLOTS, 3)); parts = []
        for first in range(0, n, 4096):
            sl = slice(first, first+4096); g, s, m = bundle_rows(curves, seeds, meta, neighbors, sample_ids[sl], length_ids[sl], steps, ds)
            g_out[sl] = g; s_out[sl] = s; parts.append(m)
        g_out.flush(); s_out.flush(); del g_out, s_out
        metadata = {key: np.concatenate([p[key] for p in parts]) for key in v1.METADATA_KEYS}
        np.savez_compressed(dest/'metadata.npz', **metadata)
        files = {f: sha(dest/f) for f in ('geometry.npy', 'seeds.npy', 'metadata.npz')}
        report['splits'][split] = dict(samples=n, source_samples=int(len(ids)), files=files, classes=dict(non_hairpin=int(np.sum(metadata['labels'] == 0)), hairpin=int(np.sum(metadata['labels'] == 1))),
                                       sample_id_sha256=hashlib.sha256(sample_ids.astype('<i8').tobytes()).hexdigest())
        print(json.dumps(dict(flow=name, split=split, rows=n)), flush=True)
    report['complete'] = True; write(root/'physical'/name/'preparation.json', report)
    write(root/f'export_{name}.json', dict(complete=True, identity=identity(definition_path), splits={s: r['files'] for s, r in report['splits'].items()}, rows={s: r['samples'] for s, r in report['splits'].items()}))


def reuse(spec, config):
    """Verify the exported bundle files against the export records and expose them read-only under this family root."""
    root = Path(spec['output']); bundles = Path(spec['bundles']); counts = {s: 0 for s in SPLITS}; hashes = {}
    for flow in spec['flows']:
        name = flow['name']; report = json.loads((bundles/'physical'/name/'preparation.json').read_text()); record = json.loads((bundles/f'export_{name}.json').read_text())
        assert report['complete'] and record['complete'] and report['identity'] == identity(config) == record['identity']
        for split in SPLITS:
            entry = report['splits'][split]; assert entry['files'] == record['splits'][split]
            for filename, digest in entry['files'].items(): assert sha(bundles/'physical'/name/split/filename) == digest; hashes[f'{name}/{split}/{filename}'] = digest
            with np.load(bundles/'physical'/name/split/'metadata.npz') as z: assert len(z['labels']) == entry['samples'] and np.all(z['counts'] == LINES)
            counts[split] += entry['samples']
    assert counts == {s: spec['expected_counts'][s] for s in SPLITS}, counts
    root.mkdir(parents=True, exist_ok=True)
    if not (root/'physical').exists(): (root/'physical').symlink_to((bundles/'physical').resolve(), target_is_directory=True)
    write(root/'data_audit.json', dict(complete=True, identity=identity(config), counts=counts, source_output=spec['source_output'], source_data_audit_sha256=spec['source_audit_sha256'],
                                       bundle_files=hashes, read_only_symlink=True))


def summarize(spec, config):
    """Recompute test scores of every run; per flow, per half length and per GT instance."""
    from sklearn.metrics import f1_score
    root = Path(spec['output']); bundles = Path(spec['bundles']); seeds = spec['training']['seeds']; records = []
    meta = {f['name']: dict(np.load(bundles/'physical'/f['name']/'test'/'metadata.npz')) for f in spec['flows']}
    labels = np.concatenate([meta[f['name']]['labels'] for f in spec['flows']]).astype(np.int64)
    lengths = np.concatenate([meta[f['name']]['scale_id'] for f in spec['flows']]); instances = np.concatenate([meta[f['name']]['instance'] for f in spec['flows']])
    flows = np.concatenate([np.full(len(meta[f['name']]['labels']), i) for i, f in enumerate(spec['flows'])])
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
                    per_instance = {f"{spec['flows'][fi]['name']}/{int(i)}": dict(positive_rows=int(t.sum()), recall=float(np.mean(prob[t] >= .5)))
                                    for fi in range(len(spec['flows'])) for i in np.unique(instances[(flows == fi) & (y == 1)]) for t in [(flows == fi) & (y == 1) & (instances == i)]}
                    scores[role] = dict(combined=engine.metrics(y, prob), per_flow={f['name']: engine.metrics(y[flows == i], prob[flows == i]) for i, f in enumerate(spec['flows'])},
                                        per_length={str(k): engine.metrics(y[lengths == k], prob[lengths == k]) for k in np.unique(lengths)}, per_instance_recall=per_instance)
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
                               test_f1_channel=agg(lambda x: x['scores']['test']['per_flow']['channel']['f1']), test_f1_tbl=agg(lambda x: x['scores']['test']['per_flow']['tbl']['f1']),
                               test_f1_per_length={k: agg(lambda x, k=k: x['scores']['test']['per_length'][k]['f1']) for k in ('0', '1', '2')},
                               training_minutes=agg(lambda x: x['training_seconds']/60))
    assert not list(root.rglob('*.pt')) and not list(root.rglob('*.pth'))
    write(root/'summary.json', dict(complete=True, identity=identity(config), family=spec['family'], methods=records, summary=summary, no_weight_files=True))


def runtime(spec, config, phase, state, code):
    Path(spec['output']).mkdir(parents=True, exist_ok=True)
    append_line(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(config), family=spec['family'], phase=phase, state=state, exit_code=code,
        job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        hostname=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


PLANS = dict(
    conv=[('preflight', None, True, '00:20:00'), ('reuse', None, False, '01:00:00'), ('encode', '0-1', True, '12:00:00'), ('train', None, True, '2-00:00:00'), ('summarize', None, False, '01:00:00')],
    bilstm=[('preflight', None, True, '00:20:00'), ('reuse', None, False, '01:00:00'), ('pilot', None, True, '01:00:00'), ('train', None, True, '3-00:00:00'), ('summarize', None, False, '01:00:00')],
    pointnet=[('preflight', None, True, '00:20:00'), ('reuse', None, False, '01:00:00'), ('pilot', None, True, '01:00:00'), ('train', None, True, '3-00:00:00'), ('summarize', None, False, '01:00:00')],
    pointnetpp=[('preflight', None, True, '00:20:00'), ('reuse', None, False, '01:00:00'), ('pilot', None, True, '01:00:00'), ('encode', '0-1', True, '12:00:00'), ('train', None, True, '3-00:00:00'), ('summarize', None, False, '01:00:00')])


def submit(definition_path):
    definition = json.loads(Path(definition_path).read_text()); root = Path(definition['output']); (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(); jobs = {}
    def sbatch(name, deps, array, gpu, wall, memory, args):
        command = ['sbatch', '--parsable', '--propagate=NONE', f'--job-name={name}', '--cpus-per-task=4', '--mem='+memory, '--time='+wall,
                   '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps: command += ['--dependency=afterok:'+':'.join(deps), '--kill-on-invalid-dep=yes']
        if array: command += ['--array='+array]
        if gpu: command += ['--gres=gpu:1', '--constraint=v100', '--exclude=gpu213-18']
        command += ['ibex_bash/task4c_fixed_dataset_baselines_2p1.sh', *args, definition_path]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        append_line(root/'submissions.jsonl', json.dumps(dict(job=job, command=command, submitted_at_utc=datetime.now(timezone.utc).isoformat()))+'\n'); return job
    jobs['export'] = sbatch('FixedBL21_export', [], '0-1', False, '06:00:00', '96G', ['export', 'all'])
    for family in definition['families']:
        spec = load_spec(definition_path, family); runs = len(spec['methods'])*len(spec['training']['seeds']); previous = jobs['export']; jobs[family] = {}
        for phase, array, gpu, wall in PLANS[family]:
            if phase == 'train': array = f"0-{runs-1}%{definition['families'][family]['concurrency']}"
            job = sbatch(f'FixedBL21_{family}_{phase}', [previous], array, gpu, wall, '64G', [phase, family]); jobs[family][phase] = job; previous = job
        write(root/'submission.json', dict(identity=identity(definition_path), jobs=jobs))
    print(json.dumps(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['export', 'preflight', 'reuse', 'pilot', 'encode', 'train', 'summarize', 'submit', 'runtime'])
    p.add_argument('family', choices=list(MODULES)+['all']); p.add_argument('--config', default=CONFIG); p.add_argument('--index', type=int, default=0)
    p.add_argument('--device', default='cuda'); p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    a = p.parse_args()
    if a.phase == 'submit': submit(a.config); return
    if a.phase == 'export': export(a.config, a.index); return
    spec = load_spec(a.config, a.family); module = MODULES[a.family]
    if a.phase == 'reuse': reuse(spec, a.config)
    elif a.phase == 'summarize': summarize(spec, a.config)
    elif a.phase == 'runtime': runtime(spec, a.config, a.runtime_phase, a.state, a.exit_code)
    elif a.phase == 'preflight': run(module, 'preflight', spec, a.config, a.device)
    elif a.phase == 'pilot': run(module, 'pilot', spec, a.config)
    elif a.phase == 'encode': run(module, 'encode', spec, a.config, a.index)
    elif a.phase == 'train': run(module, 'train', spec, a.config, a.index)


if __name__ == '__main__': main()
