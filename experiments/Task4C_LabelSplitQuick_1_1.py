"""Ten-epoch factorial diagnosis, reusing verified p35 ball6 raw features."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from FMT_Utils import Task4C_LabelSplitQuick_1_1 as core

CONFIG = 'config/Verify_Task4C_LabelSplitQuick_1.1.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def spec():
    return json.loads(Path(CONFIG).read_text())


def verify_sources():
    frozen = json.loads(Path('deployment_source_sha256.json').read_text())
    for p, digest in frozen.items():
        assert sha(p) == digest, p
    return frozen


def old_labels(s, fi, seeds):
    from scipy import ndimage
    from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow, cell_centers
    from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset, sample_gt
    from FMT_Utils.Task4C_GTHeadCoverage_1_1 import spacing_at
    from experiments.Task4C_PhysicalLength_4_14 import build_catalog
    physical = json.loads(Path(s['physical_config']).read_text()); flow = physical['flows'][fi]
    root = Path(physical['input_root'])
    for key in ('flow', 'gt'):
        assert sha(root/flow[key]) == flow[key+'_sha256']
    axes, _, heads, _, _, _ = load_flow(root/flow['flow'], flow['lambda2_threshold'])
    gt = read_dataset(root/flow['gt'])
    components, catalog, report = build_catalog(axes, heads, gt, physical)
    lookup = np.full(components.max()+1, -1, np.int8)
    owner = np.full(len(lookup), -1, np.int64)
    for region in catalog:
        lookup[region['head_component']] = region['label']
        owner[region['head_component']] = region['instance']
    # Independent counting implementation, on every candidate cell center.
    cells = np.flatnonzero(heads); labels, _ = sample_gt(gt, cell_centers(cells, heads.shape, axes))
    ids = components.ravel()[cells]
    counts = np.bincount(ids, minlength=len(lookup))
    pairs, numbers = np.unique(np.column_stack((ids[labels >= 0], labels[labels >= 0])), axis=0, return_counts=True)
    largest = np.zeros(len(lookup), np.int64)
    if len(pairs): np.maximum.at(largest, pairs[:, 0], numbers)
    independent = np.full(len(lookup), -1, np.int8)
    independent[(counts >= 10) & (largest == 0)] = 0
    independent[(counts >= 10) & (2*largest >= counts)] = 1
    np.testing.assert_array_equal(lookup, independent)
    positions = [np.searchsorted(axis, seeds[:, d], side='right')-1 for d, axis in enumerate(axes)]
    assert all(np.all((x >= 0) & (x < len(a)-1)) for x, a in zip(positions, axes))
    region = components[positions[2], positions[1], positions[0]]
    scale = np.exp(np.log(spacing_at(seeds, axes)).mean(1))
    return lookup[region], region, owner[region], scale, dict(
        independent_all_cell_labels_passed=True, candidate_components=int(components.max()),
        eligible_components=len(catalog), excluded_components=len(report['excluded_heads']),
        seed_eligible=int((lookup[region] >= 0).sum()), total_seeds=len(seeds))


def prepare():
    s = spec(); sources = verify_sources(); out = Path(s['output'])
    assert not (out/'dataset.json').exists(), 'Frozen dataset already exists'
    root = Path(s['source_output']); audit = json.loads((root/'data_audit.json').read_text())
    assert sha(root/'data_audit.json') == s['source_audit_sha256']
    arrays = {role: {key: [] for key in ('features', 'old_label', 'new_label', 'flow', 'sample', 'length', 'instance', 'component', 'fold')}
              for role in ('heldout_train', 'shared_train', 'test')}
    evidence = {}; cache_hashes = {}
    cache = Path(s['feature_cache'])
    for role in ('train', 'test'):
        cache_hashes[role] = sha(cache/f'{role}_features.npy')
    for fi, name in enumerate(('channel', 'tbl')):
        src = root/'physical'/name
        for file in ('seeds.npy', 'curves.npy', 'metadata.npz'):
            assert sha(src/file) == audit['frozen_files'][name][file], (name, file)
        seeds = np.load(src/'seeds.npy', mmap_mode='r')
        with np.load(src/'metadata.npz') as z:
            meta = {k:z[k] for k in ('label', 'instance', 'fold', 'short_curve_gt_count')}
        np.testing.assert_array_equal(meta['label'], meta['short_curve_gt_count'] >= 17)
        old, component, old_owner, h, info = old_labels(s, fi, seeds)
        eligible = old >= 0
        rng = np.random.default_rng([s['seed'], fi, 1])
        test_pool = np.flatnonzero(eligible & (meta['fold'] == 0))
        test = np.sort(rng.choice(test_pool, s['test_points_per_flow'], replace=False))
        allowed = core.distance_allowed(seeds, test, h) & eligible
        heldout = core.choose_training(np.flatnonzero(allowed & (meta['fold'] != 0)), meta['instance'],
            meta['instance'][test], s['train_points_per_flow'], np.random.default_rng([s['seed'], fi, 2]), False)
        shared = core.choose_training(np.flatnonzero(allowed), meta['instance'], meta['instance'][test],
            s['train_points_per_flow'], np.random.default_rng([s['seed'], fi, 3]), True)
        roles = dict(heldout_train=heldout, shared_train=shared, test=test)
        info['all_eligible_label_confusion_old_rows_new_columns'] = np.bincount(
            2*old[eligible]+meta['label'][eligible], minlength=4).reshape(2, 2).tolist()
        info['roles'] = {}
        # Cache layout is flow -> original split sample IDs ascending -> three lengths.
        feature_by_sample = np.empty((len(seeds), 3, 1, 142), np.float32)
        for original in ('train', 'test'):
            indices = np.flatnonzero((meta['fold'] == 0) == (original == 'test'))
            offset = 0
            if fi:
                with np.load(root/'physical/channel/metadata.npz') as z:
                    offset = int(((z['fold'] == 0) == (original == 'test')).sum())*3
            raw = np.load(cache/f'{original}_features.npy', mmap_mode='r')
            feature_by_sample[indices] = raw[offset:offset+3*len(indices)].reshape(-1, 3, 1, 142)
        # Verify actual cached mapping against fresh frozen encoder, both original roles/classes.
        import torch
        from experiments import Task4C_V2NewLabelTraining_1_1 as original_run
        from FMT_Utils import Task4C_V2NewLabelTraining_1_1 as original_data
        original_spec = json.loads(Path(s['training_config']).read_text())
        original_spec.update(source_output=s['source_output'], neighbor_output=s['neighbor_output'])
        candidate = original_spec['candidates'][0]
        source = original_data.Source(original_spec, fi, 6)
        probe = np.unique(np.r_[heldout[:2], shared[:2], test[:2],
            *[np.flatnonzero((meta['label'] == label) & ((meta['fold'] == 0) == is_test))[:2]
              for label in (0, 1) for is_test in (False, True)]])
        samples = np.repeat(probe, 3); lengths = np.tile(np.arange(3), len(probe))
        lines, seed_points, _ = source.gather(samples, lengths)
        api = original_data.old
        g, ns, count = api.v2.normalize_bundle(torch.tensor(lines), torch.tensor(seed_points))
        encoded, used = api.fmt.encode(g, ns, count, torch.zeros(len(g), dtype=torch.long), candidate)
        encoded = encoded.numpy(); encoded[..., :141] = api.transform_fmt(encoded[..., :141], True)
        np.testing.assert_allclose(encoded, feature_by_sample[probe].reshape(-1, 1, 142), atol=2e-4, rtol=2e-4)
        info['cache_recompute_samples'] = len(probe)
        for role, take in roles.items():
            values = dict(features=feature_by_sample[take].reshape(-1, 1, 142), old_label=np.repeat(old[take], 3),
                new_label=np.repeat(meta['label'][take], 3), flow=np.full(3*len(take), fi, np.int8),
                sample=np.repeat(take, 3), length=np.tile(np.arange(3), len(take)),
                instance=np.repeat(meta['instance'][take], 3), component=np.repeat(component[take], 3), fold=np.repeat(meta['fold'][take], 3))
            for key, value in values.items(): arrays[role][key].append(value)
            info['roles'][role] = dict(points=len(take), old_positive=int(old[take].sum()), new_positive=int(meta['label'][take].sum()),
                instances=np.unique(meta['instance'][take]).tolist(),
                test_instances_shared=sorted(map(int, set(meta['instance'][take]) & set(meta['instance'][test]))),
                original_fold_zero_points=int((meta['fold'][take] == 0).sum()))
            if role != 'test':
                from scipy.spatial import cKDTree
                distances = cKDTree(seeds[take]).query(seeds[test])[0]/h[test]
                assert distances.min() >= 1-1e-10
                info['roles'][role]['test_to_train_distance_over_local_h'] = dict(min=float(distances.min()),
                    median=float(np.median(distances)), max=float(distances.max()), fraction_le_4=float((distances <= 4).mean()))
        np.savez_compressed(out/f'{name}_label_map.npz', old_label=old, new_label=meta['label'], component=component,
            region_owner=old_owner, instance=meta['instance'], fold=meta['fold'], local_h=h)
        evidence[name] = info
        print(json.dumps(dict(stage='flow_prepared', flow=name, **info)), flush=True)
    files = {}
    for role, fields in arrays.items():
        dest = out/'dataset'/role; dest.mkdir(parents=True, exist_ok=False)
        for key, parts in fields.items():
            values = np.concatenate(parts); np.save(dest/f'{key}.npy', values)
            files[str((dest/f'{key}.npy').relative_to(out))] = sha(dest/f'{key}.npy')
        for key in ('old_label', 'new_label'):
            assert len(np.unique(np.concatenate(fields[key]))) == 2
    write(out/'dataset.json', dict(complete=True, config=s, sources=sources, per_flow=evidence, files=files,
        source_audit_sha256=s['source_audit_sha256'], cache_sha256=cache_hashes, validation_points=0))


def load_dataset(role):
    out = Path(spec()['output']); folder = out/'dataset'/role
    return {p.stem: np.load(p) for p in folder.glob('*.npy')}


def train(index):
    import torch
    from experiments import Task4C_V2NewLabelTraining_1_1 as original
    s = spec(); verify_sources(); out = Path(s['output']); record = json.loads((out/'dataset.json').read_text())
    assert record['complete']
    arm = s['arms'][index]; dest = out/'runs'/arm['id']; dest.mkdir(parents=True, exist_ok=False)
    for role in (arm['split']+'_train', 'test'):
        for p, digest in record['files'].items():
            if p.startswith('dataset/'+role+'/'): assert sha(out/p) == digest
    tr = load_dataset(arm['split']+'_train'); te = load_dataset('test'); label_key = arm['label']+'_label'
    original.old.fmt_old.baseline.deterministic('cuda')
    torch.manual_seed(s['seed']); np.random.seed(s['seed']); rng = np.random.default_rng(s['seed'])
    candidate = json.loads(Path(s['training_config']).read_text())['candidates'][0]
    values = tr['features'][:, 0, :141]
    mean = values.mean(0, dtype=np.float64); std = values.std(0, dtype=np.float64); std[std < 1e-8] = 1
    norm = (torch.tensor(mean, device='cuda'), torch.tensor(std, device='cuda'), len(values))
    tensors = {name:original.standardize(candidate, torch.tensor(data['features'], device='cuda'), norm)
               for name, data in (('train', tr), ('test', te))}
    labels = torch.tensor(tr[label_key].astype(np.int64), device='cuda')
    weight = core.loss_weights(tr[label_key]); weights = torch.tensor(weight, device='cuda', dtype=torch.float32)
    model = original.make_model(candidate)
    # Engineering-only finite-gradient and batch consistency gate; reset before fitting.
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(tensors['train'][:32]), torch.cat([model(tensors['train'][i:i+8]) for i in range(0, 32, 8)]), atol=2e-5, rtol=2e-5)
    model.train(); probe_loss = torch.nn.functional.cross_entropy(model(tensors['train'][:128]), labels[:128])
    probe_loss.backward(); assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.manual_seed(s['seed']); model = original.make_model(candidate)
    init_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=s['learning_rate'], weight_decay=.0001)
    np.savez(dest/'normalizer.npz', mean=mean, std=std)
    def predict(name):
        model.eval(); probabilities = []
        with torch.no_grad():
            for first in range(0, len(tensors[name]), 1024):
                probabilities.append(model(tensors[name][first:first+1024]).softmax(-1)[:, 1].cpu().numpy())
        return np.concatenate(probabilities)
    history = []; started = time.perf_counter()
    for epoch in range(1, s['epochs']+1):
        order = rng.permutation(len(labels)); model.train(); total = 0
        for first in range(0, len(order), 128):
            ids = order[first:first+128]; optimizer.zero_grad(set_to_none=True)
            losses = torch.nn.functional.cross_entropy(model(tensors['train'][ids]), labels[ids], reduction='none')
            loss = (losses*weights[labels[ids]]).mean(); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True); optimizer.step()
            total += float(loss.detach())*len(ids)
        probability = predict('test'); training_probability = predict('train')
        row = dict(epoch=epoch, loss=total/len(labels), train=core.score(tr[label_key], training_probability),
            test=core.score(te[label_key], probability), per_flow={name:core.score(te[label_key][te['flow']==fi], probability[te['flow']==fi])
            for fi, name in enumerate(('channel', 'tbl'))},
            permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(), seconds=time.perf_counter()-started)
        history.append(row); write(dest/'progress.json', dict(arm=arm, latest=row, complete=False))
        with (dest/'history.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        np.savez_compressed(dest/f'predictions_epoch{epoch:02}.npz', probability=probability, label=te[label_key],
            sample=te['sample'], flow=te['flow'], length=te['length'], instance=te['instance'])
        print(json.dumps(dict(arm=arm['id'], **row)), flush=True)
    torch.save(model.state_dict(), dest/'final_model.pt')
    write(dest/'result.json', dict(complete=True, arm=arm, seed=s['seed'], history=history, primary=history[-1],
        initial_parameters_sha256=init_hash, class_weights=weight.tolist(),
        parameters=sum(p.numel() for p in model.parameters()), dataset_sha256=sha(out/'dataset.json'),
        selection='fixed final epoch; no test-guided scheduler or early stopping'))


def audit():
    s = spec(); verify_sources(); out = Path(s['output']); results = {}
    for arm in s['arms']:
        dest = out/'runs'/arm['id']; r = json.loads((dest/'result.json').read_text()); assert r['complete']
        te = load_dataset('test')
        for row in r['history']:
            with np.load(dest/f"predictions_epoch{row['epoch']:02}.npz") as z:
                for key, target in [('label', arm['label']+'_label'), ('sample','sample'), ('flow','flow'), ('length','length'), ('instance','instance')]:
                    np.testing.assert_array_equal(z[key], te[target])
                actual = core.score(z['label'], z['probability'])
                for key, value in actual.items():
                    if value is not None: assert np.isclose(value, row['test'][key], rtol=0, atol=1e-12)
                # Independent F1 expression from Boolean confusion masks.
                p = z['probability'] >= .5; y = z['label'].astype(bool)
                f1 = 2*np.sum(p & y)/max(np.sum(p)+np.sum(y), 1)
                assert np.isclose(f1, row['test']['f1'], rtol=0, atol=1e-12)
        results[arm['id']] = r
    assert len({r['initial_parameters_sha256'] for r in results.values()}) == 1
    assert len({tuple(x['permutation_sha256'] for x in r['history']) for r in results.values()}) == 1
    effects = {}
    for metric in ('f1', 'f1_prior_one_third', 'roc_auc', 'average_precision'):
        v = {key:r['primary']['test'][metric] for key, r in results.items()}
        split_old = v['old_shared']-v['old_heldout']; split_new = v['new_shared']-v['new_heldout']
        label_shared = v['old_shared']-v['new_shared']; label_heldout = v['old_heldout']-v['new_heldout']
        effects[metric] = dict(values=v, shared_minus_heldout_old=split_old, shared_minus_heldout_new=split_new,
            old_minus_new_shared=label_shared, old_minus_new_heldout=label_heldout,
            split_average=(split_old+split_new)/2, label_average=(label_shared+label_heldout)/2,
            interaction=split_old-split_new)
    write(out/'result_audit.json', dict(complete=True, fixed_epoch=s['epochs'], effects=effects,
        results={k:r['primary'] for k,r in results.items()}, single_seed=True, diagnosis_only=True,
        all_epoch_predictions_verified=True, same_initial_parameters=True, same_shuffle=True))
    print(json.dumps(effects, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare','train','audit']); parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.phase == 'train': train(args.index)
    else: {'prepare':prepare, 'audit':audit}[args.phase]()
