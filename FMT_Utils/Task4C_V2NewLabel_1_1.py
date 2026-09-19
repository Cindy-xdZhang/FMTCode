"""Task4-c labels from strict GT majority on the shortest 32-point curve."""
import json
import time
from pathlib import Path
from types import FunctionType
import numpy as np

from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt
from FMT_Utils import Task4C_FixedDataset_2_1 as v2
from FMT_Utils import Task4C_FixedDataset_3_1 as dense
from FMT_Utils import Task4C_FixedDataset_3_2 as quota

EXTRA_KEYS = ('seed_label', 'short_curve_gt_owner', 'short_curve_gt_count', 'short_curve_gt_fraction')
METADATA_KEYS = (*v2.METADATA_KEYS, *EXTRA_KEYS)


def majority_labels(owners):
    """Any nonnegative GT instance ID counts as hairpin; exactly half is negative."""
    owners = np.asarray(owners)
    if owners.ndim != 2 or owners.shape[1] != 32:
        raise ValueError('Expected exactly 32 GT point owners per shortest curve')
    if not np.issubdtype(owners.dtype, np.integer) or np.any(owners < -1):
        raise ValueError('GT owners must be integer instance IDs, or -1 outside GT')
    hits = (owners >= 0).sum(axis=1).astype(np.int16)
    return (hits > 16).astype(np.int64), hits


def shortest_curve_labels(gt, curves, locator=None, batch=4096):
    """Classify saved physical float32 geometry, using only length index zero."""
    curves = np.asarray(curves)
    if curves.ndim != 4 or curves.shape[1:] != (3, 32, 3):
        raise ValueError('Expected three shortest-to-longest curves, each with 32 points')
    if curves.dtype != np.float32 or not np.isfinite(curves).all():
        raise ValueError('Use the finite float32 physical geometry saved in the dataset')
    owners = np.empty((len(curves), 32), np.int64)
    for first in range(0, len(curves), batch):
        points = curves[first:first+batch, 0].reshape(-1, 3)
        found, locator = sample_gt(gt, points, locator)
        owners[first:first+len(points)//32] = found.reshape(-1, 32)
    label, hits = majority_labels(owners)
    return dict(label=label, short_curve_gt_owner=owners,
                short_curve_gt_count=hits, short_curve_gt_fraction=hits.astype(np.float32)/32)


def balanced_fraction(selected, kept, ratio=.5):
    selected = np.asarray(selected, np.float64)
    kept = np.asarray(kept, np.float64)
    if np.any(selected <= 0) or np.any(kept <= 0) or np.any(kept > selected):
        raise ValueError('Both classes must survive the pilot')
    survival = kept/selected
    fraction = ratio*survival[0]/(survival[1]+ratio*survival[0])
    return float(fraction), survival.tolist()


def proposal_labels(scene, pool, spec, output):
    """Short-curve prepass for class quotas; final three-curve cleaning is unchanged.

    An invalid shortest curve cannot become a published sample. Its proposal is
    counted in the nonpositive quota only, then rejected by the original final
    cleaning. Pilot survival correction includes these rejected proposals.
    """
    labels = np.zeros(len(pool), np.int8)
    valid = np.zeros(len(pool), bool)
    hits = np.full(len(pool), -1, np.int16)
    batch = spec['integration']['trace_batch_seeds']
    started = time.perf_counter()
    for first in range(0, len(pool), batch):
        sl = slice(first, first+batch)
        traced = v2.trace_curves(scene['grid'], pool[sl], scene['flow_spec']['ds'],
                                 [scene['steps'][0]], spec['integration'])
        good = traced['valid'][:, 0]
        short = traced['curves'][good]
        values = shortest_curve_labels(scene['gt'], np.repeat(short, 3, axis=1), scene['locator'])
        ids = np.arange(first, min(first+batch, len(pool)))[good]
        labels[ids] = values['label']; hits[ids] = values['short_curve_gt_count']; valid[ids] = True
        if first == 0 or (first+batch)//65536 != first//65536 or first+batch >= len(pool):
            print(json.dumps(dict(stage='short_curve_proposals', flow=scene['flow_spec']['name'],
                  done=min(first+batch, len(pool)), total=len(pool), seconds=time.perf_counter()-started)), flush=True)
    for name, values in [('proposal_labels.npy', labels), ('proposal_short_valid.npy', valid), ('proposal_gt_count.npy', hits)]:
        np.save(output/name, values)
    return labels, valid, hits


def fraction(spec, index, pilot):
    if pilot: return 1/3
    report = json.loads((Path(spec['output'])/'calibration.json').read_text())
    assert report['complete']
    return report['flows'][spec['flows'][index]['name']]['positive_fraction']


def prepare(spec, index, write, sha, identity, pilot=False):
    started = time.perf_counter()
    scene = v2.load_scene(spec, index); flow = scene['flow_spec']; name = flow['name']
    sampling = dict(spec['sampling'])
    if pilot: sampling.update(spec['pilot'])
    output = Path(spec['output'])/('pilot' if pilot else 'physical')/name
    output.mkdir(parents=True, exist_ok=False)
    mask = v2.candidate_mask(scene['lambda2'], scene['oyf'], scene['threshold'])
    components, sizes, count = v2.candidate_components(mask)
    cells = v2.candidate_cells(mask)
    pool, stats = v2.initial_pool(scene['axes'], scene['lambda2'], scene['oyf'], scene['threshold'], cells,
                                  sampling['pool_per_flow'], sampling['pool_seed']+index, sampling['pool_chunk'])
    proposal, short_valid, proposal_hits = proposal_labels(scene, pool, spec, output)
    target = sampling['samples_per_flow']; p = fraction(spec, index, pilot)
    limits = quota.quotas(target, p)
    available = np.bincount(proposal, minlength=2)
    if np.any(available < limits):
        raise ValueError(f'Not enough curve-labelled proposals: available={available.tolist()}, required={limits.tolist()}; no duplicate points or extra pool')
    order = np.random.default_rng([sampling['poisson_seed'], index]).permutation(len(pool))
    guess = (stats['candidate_cell_volume']*stats['acceptance']/target)**(1/3)
    selected, radius, log = quota.poisson_disk(pool[order], proposal[order], limits, guess,
                                               sampling['radius_relative_tolerance'], sampling['grid_capacity_per_cell'])
    pool_ids = order[selected]; seeds = pool[pool_ids]
    np.save(output/'selected_pool_indices.npy', pool_ids)
    assert np.array_equal(np.bincount(proposal[pool_ids], minlength=2), limits)
    print(json.dumps(dict(stage='poisson_done', flow=name, selected=limits.tolist(), radius=radius)), flush=True)
    parts = []; batch = spec['integration']['trace_batch_seeds']
    for first in range(0, len(seeds), batch):
        parts.append(v2.trace_curves(scene['grid'], seeds[first:first+batch], flow['ds'], scene['steps'], spec['integration']))
        print(json.dumps(dict(stage='three_curves', flow=name, done=min(first+batch, len(seeds)), total=len(seeds))), flush=True)
    traced = {key: np.concatenate([part[key] for part in parts])
              for key in ('curves', 'valid', 'relaxed', 'arcs', 'counts', 'bounds', 'termination')}
    keep = traced['valid'].all(1)
    # Test the independent short prepass against the actual saved shortest curves.
    values = shortest_curve_labels(scene['gt'], traced['curves'][keep], scene['locator'])
    assert short_valid[pool_ids[keep]].all()
    np.testing.assert_array_equal(values['short_curve_gt_count'], proposal_hits[pool_ids[keep]])
    np.testing.assert_array_equal(values['label'], proposal[pool_ids[keep]])
    reference = spec['reference_folds'][name]
    def folds(instances, low, high, counts, number, seed):
        return dense.fixed_instance_folds(instances, low, high, counts, number, seed, reference)
    rows_fn = FunctionType(v2.build_rows.__code__, dict(v2.__dict__, instance_folds=folds))
    rows = rows_fn(scene, index, seeds[keep], {key: val[keep] for key, val in traced.items()},
                   pool_ids[keep], mask, components, sizes, radius, spec)
    meta = rows['metadata']; meta['seed_label'] = meta['label'].copy(); meta.update(values)
    assert set(meta) == set(METADATA_KEYS)
    rows['assignment']['seed_positive_per_instance'] = rows['assignment'].pop('positive_per_instance')
    rows['assignment']['positive_per_instance'] = {str(i): int(np.sum((meta['instance'] == i) & (meta['label'] == 1))) for i in scene['instances']}
    kept = np.bincount(meta['label'], minlength=2); ratio = float(kept[1]/kept[0])
    saved = v2.save(rows, output, sha)
    for file in ('proposal_labels.npy', 'proposal_short_valid.npy', 'proposal_gt_count.npy', 'selected_pool_indices.npy'):
        saved['files'][file] = sha(output/file)
    stats_clean = dict(termination_reasons={}, curve_rejections={})
    for part in parts:
        for category in stats_clean:
            for key, val in part['stats'][category].items():
                stats_clean[category][key] = stats_clean[category].get(key, 0)+val
    report = dict(complete=True, version=spec['version'], dataset_name=spec['dataset_name'], flow=flow,
        pilot=pilot, identity=identity(), **saved,
        candidate=dict(threshold=scene['threshold'], grid_points=int(mask.size), candidate_points=int(mask.sum()), components=count),
        pool=dict(stats, seed=sampling['pool_seed']+index, poisson_seed=[sampling['poisson_seed'], index],
                  proposal_classes=available.tolist(), invalid_short_curves=int((~short_valid).sum())),
        poisson=dict(radius=radius, radius_guess=guess, target=target, bisection=log),
        integration=dict(ds=flow['ds'], half_lengths=flow['half_lengths'], half_steps=scene['steps'], **stats_clean),
        cleaning=dict(integrated=len(seeds), kept=int(keep.sum()), removed=int((~keep).sum()),
                      removed_rule='all three curves must be valid; no top-up'),
        labeling=spec['labeling'], balance=dict(positive_fraction=p, selected_by_class=limits.tolist(),
            kept_by_class=kept.tolist(), positive_to_negative_ratio=ratio, target_ratio=.5,
            within_relative_tolerance=bool(abs(ratio/.5-1) <= spec['balance']['ratio_relative_tolerance'])),
        label_changes=int(np.sum(meta['label'] != meta['seed_label'])), seconds=time.perf_counter()-started)
    write(output/'preparation.json', report)
    print(json.dumps(dict(stage='complete', flow=name, kept=kept.tolist(), ratio=ratio)), flush=True)
    return report
