"""Audit saved scalar records without loading geometry, models or test data."""
import argparse
from collections import Counter
import csv
import math
from pathlib import Path

from FMT_Utils.FlowMapData_3D import sha256, write_json
from experiments.Task6_DirectNeural_5_1 import read


def report(root, config):
    root = Path(root)
    spec = read(config)
    digest = sha256(config)
    assert sha256(root/'config.frozen.json') == digest
    events = [read(p) for p in (root/'events').glob('*.json')]
    summary = dict(experiment=spec['experiment'],config_sha256=digest,
        stages=dict(Counter(e['phase']+':'+e['status'] for e in events)),
        event_failures=[e for e in events if e['status']=='FAILED'],test_used_for_selection=False)
    rows = []
    for phase, training, pattern in [('baseline',spec['baseline_training'],'*/result.json'),
                                     ('screen',spec['screening'],'*/*/result.json'),
                                     ('refine',spec['training'],'*/*/result.json')]:
        count = 0
        for path in (root/phase).glob(pattern):
            result = read(path)
            assert not result['test_read'] and result['provenance']['config_sha256'] == digest
            items = result['rows'] if phase == 'refine' else [result]
            for row in items:
                fit = read(path.parent/row['arm']/'fit.json')
                candidate = spec['baseline_candidate'] if phase == 'baseline' else next(
                    c for c in spec['candidates'] if c['id'] == row['candidate'])
                assert fit['candidate'] == candidate
                assert fit['train_samples'] == spec['primary_train_size']
                if phase != 'baseline':
                    assert fit['statistics_train_samples'] == spec['primary_train_size']
                assert fit['updates'] == training['updates']
                assert fit['train_examples_exposed'] == training['updates']*training['batch_size']
                best = min((r for r in fit['curve'] if r['step']>0), key=lambda r:r['validation_rmse_r'])
                assert fit['selected_step'] == row['selected_step'] == best['step']
                for key in ('train_rmse_r','validation_rmse_r'):
                    assert math.isclose(fit[key],row[key],rel_tol=1e-10,abs_tol=1e-12)
                    assert math.isclose(fit[key],best[key],rel_tol=1e-10,abs_tol=1e-12)
                if phase != 'baseline':
                    assert fit['schedule']['completed_updates'] == training['updates']
                    assert fit['schedule']['spec'] == candidate['scheduler']
                    assert fit['training'] == training
                    assert all(math.isfinite(r['learning_rate_used']) and 0<r['learning_rate_used']<=candidate['learning_rate']
                               for r in fit['curve'] if r['step']>0)
                rows.append(dict(phase=phase,dataset=result['dataset'],candidate=candidate['id'],arm=row['arm'],
                    train_samples=fit['train_samples'],updates=fit['updates'],selected_step=row['selected_step'],
                    train_rmse_r=fit['train_rmse_r'],validation_rmse_r=fit['validation_rmse_r']))
                count += 1
        summary[phase+'_models'] = count
    if (root/'promotion.json').exists():
        assert sha256(root/'promotion.json') == sha256(root/'promotion.frozen.json')
        promotion = read(root/'promotion.json')
        assert not promotion['test_read'] and promotion['provenance']['config_sha256'] == digest
        summary['promoted'] = [c['id'] for c in promotion['candidates']]
        assert spec['reference_candidate'] in summary['promoted']
    if (root/'selection.json').exists():
        assert sha256(root/'selection.json') == sha256(root/'selection.before_test.json')
        selection = read(root/'selection.json')
        assert not selection['test_read'] and selection['provenance']['config_sha256'] == digest
        assert summary['baseline_models'] == len(spec['datasets'])
        assert summary['screen_models'] == len(spec['screen_datasets'])*len(spec['candidates'])
        assert summary['refine_models'] == len(spec['datasets'])*spec['promote_count']*2
        scores = {}
        for cid in summary['promoted']:
            values = [r['validation_rmse_r'] for r in rows if r['phase']=='refine' and r['candidate']==cid and r['arm']=='pnn_trans']
            assert len(values) == len(spec['datasets'])
            scores[cid] = sum(values)/len(values)
        selected = min(scores,key=lambda cid:(scores[cid],cid))
        assert selected == selection['candidate']['id']
        chosen = [r for r in rows if r['phase']=='baseline' or (r['phase']=='refine' and r['candidate']==selected)]
        means = {}
        for arm in spec['arms']:
            values = [r['validation_rmse_r'] for r in chosen if r['arm']==arm]
            assert len(values) == len(spec['datasets'])
            means[arm] = sum(values)/len(values)
            assert math.isclose(means[arm],selection['means'][arm],rel_tol=1e-12)
        numeric = all(r['validation_rmse_r'] < spec['validation_gate_rmse_r'] for r in chosen)
        beats = means['pnn_trans'] < min(means['raw_trans'],means['raw_frozen'])
        assert selection['gate_passed'] == (numeric and beats)
        summary.update(selected_candidate=selected,selected_rows=chosen,means=means,
                       gate_passed=numeric and beats,selection_sha256=sha256(root/'selection.json'))
    if rows:
        with (root/'checked_training_results.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary['record_consistency_passed'] = True
    write_json(root/'status_summary.json',summary)
    return summary


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser()
    p.add_argument('--root',default='outputs/Verify_Task6_PNNTrans_1.3')
    p.add_argument('--config',default='config/Verify_Task6_PNNTrans_1.3.json')
    args = p.parse_args()
    print(json.dumps(report(args.root,args.config),indent=2))
