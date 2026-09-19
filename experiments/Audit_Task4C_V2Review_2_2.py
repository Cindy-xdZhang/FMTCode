"""Independent file/label/prediction checks and a candidate-cell counterexample; no data changes."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from FMT_Utils import Task4C_FixedDataset_2_1 as data
from experiments.Analyze_Task4C_BallQuery_2_2 import sha


def main():
    root = Path('outputs/mainExp_Task4C_FixedDataset_2.1')
    audit = json.loads((root/'data_audit.json').read_text())
    report = dict(dataset_audit_sha256=sha(root/'data_audit.json'), source_hash_matches={}, flows={}, baseline_predictions={})
    for p, digest in audit['identity']['sources'].items():
        report['source_hash_matches'][p] = sha(p) == digest
    for name in ('channel', 'tbl'):
        folder = root/'physical'/name
        for f, digest in audit['frozen_files'][name].items(): assert sha(folder/f) == digest
        seeds, curves, meta = data.load(folder)
        assert np.array_equal(meta['label'], (meta['gt_owner'] >= 0).astype(np.int64))
        assert np.all((meta['fold'] == 0) == (meta['split'] == 1))
        assert all(len(np.unique(meta['fold'][meta['instance'] == i])) == 1 for i in np.unique(meta['instance']))
        assert curves.shape == (len(seeds), 3, 32, 3) and np.isfinite(curves).all()
        prep = json.loads((folder/'preparation.json').read_text())
        report['flows'][name] = dict(samples=len(seeds), shape=list(curves.shape), fold_counts=np.bincount(meta['fold'], minlength=5).tolist(),
            hairpin=int(meta['label'].sum()), classes_exactly_match_saved_gt_owner=True, instances_do_not_cross_folds=True,
            samples_per_flow_target=prep['poisson']['target'], initial_pool=prep['pool']['requested'])
    # True trilinear intersection inside a cell with no simultaneously qualifying vertex.
    axes = [np.array([0., 1.])]*3
    lam = np.broadcast_to(np.array([-1., .2])[None, None], (2, 2, 2)).copy()
    oyf = np.broadcast_to(np.array([-.2, 1.])[None, None], (2, 2, 2)).copy()
    mask = data.candidate_mask(lam, oyf, 0.)
    point = np.array([[.5, .5, .5]])
    l, _, _ = data.interpolate_scalar(point, axes, lam); o, _, _ = data.interpolate_scalar(point, axes, oyf)
    assert not mask.any() and l[0] < 0 and o[0] > 0 and len(data.candidate_cells(mask)) == 0
    report['candidate_cell_counterexample'] = dict(center_lambda2=float(l[0]), center_oyf=float(o[0]), threshold=0,
        qualifying_vertices=int(mask.sum()), eligible_cells=len(data.candidate_cells(mask)),
        implication='candidate cell prefilter can omit part of the continuous trilinear intersection; actual omitted dataset volume not quantified')
    base = Path('outputs/mainExp_Task4C_FixedDatasetFMT_2.1/arms')
    for candidate in ('p35_h0_fps6', 'p35_h0_fps16', 'c156_fps6', 'c156_fps16'):
        folder = base/candidate/'seed96611'/'final'/candidate/'seed96611'
        if not (folder/'result.json').exists(): continue
        result = json.loads((folder/'result.json').read_text())
        scores = {}
        for role in ('validation', 'test'):
            path = folder/f'{role}_predictions.npz'; assert sha(path) == result['predictions'][role]
            with np.load(path) as z: y, p = z['labels'], z['probability']
            f1 = float(f1_score(y, p >= .5)); expected = result['validation'] if role == 'validation' else result['test']['combined']
            assert abs(f1-expected['f1']) < 1e-12
            scores[role] = dict(f1=f1, precision=float(precision_score(y, p >= .5)), recall=float(recall_score(y, p >= .5)))
        report['baseline_predictions'][candidate] = dict(seed=96611, result=str(folder/'result.json'), result_sha256=sha(folder/'result.json'), scores=scores)
    out = Path('outputs/Verify_Task4C_V2Review_2.2'); out.mkdir(parents=True, exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
