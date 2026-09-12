"""Check recorded development results and summarize all preregistered variants.

This checks saved scalar evidence, not an independent prediction re-evaluation.
Validation-only diagnostics must not be presented as final test results.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(root):
    config_path = root / 'config.frozen.json'
    config = read(config_path)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    selection = read(root / 'selection.json')
    assert (root / 'selection.before_test.json').read_bytes() == (root / 'selection.json').read_bytes()
    assert selection['provenance']['config_sha256'] == config_hash
    assert not selection['test_read']
    rows = []
    for dataset in config['datasets']:
        for variant in config['variants']:
            folder = root / 'search' / dataset / variant
            result, fit = read(folder / 'result.json'), read(folder / 'fit.json')
            assert result['provenance']['config_sha256'] == config_hash and not result['test_read']
            assert (result['dataset'], result['variant']) == (dataset, variant)
            assert fit['seed'] == config['search_seed'] and fit['train_samples'] == config['primary_train_size']
            assert fit['training'] == config['training'] and fit['candidate'] == config['candidate']
            assert result['validation'] == fit['validation']
            assert [r for r in selection['rows'] if (r['dataset'], r['variant']) == (dataset, variant)] == [result]
            best = min((p for p in fit['curve'] if p['step'] > 0), key=lambda p: p['validation']['selection_score'])
            assert fit['selected_step'] == best['step'] == result['selected_step']
            assert fit['curve'][-1]['step'] == config['training']['updates']
            val = fit['validation']
            for key in ('task6_rmse_r', 'selection_score'):
                if val[key] is not None:
                    assert math.isclose(val[key], best['validation'][key], rel_tol=1e-6, abs_tol=1e-7)
            use3, use6 = variant != 'single_task6', variant != 'single_task3'
            cls = val['normalized_classification_loss'] if use3 else 0.
            rec = val['selection_score'] - cls if use6 else None
            rows.append(dict(dataset=dataset, variant=variant,
                average_precision=val['task3']['average_precision'] if use3 else None,
                f1=val['task3']['f1'] if use3 else None,
                reconstruction_rmse_r=val['task6_rmse_r'],
                branch_a_gate_classification=val['gate3_mean'] if use3 else None,
                branch_a_gate_reconstruction=val['gate6_mean'] if use6 else None,
                selected_step=fit['selected_step'], classification_loss=cls if use3 else None,
                normalized_reconstruction_loss=rec,
                implied_train_variance=val['task6_rmse_r']**2 / rec if rec is not None and rec > 0 else None,
                minimum_reconstruction_along_curve=min(p['validation']['task6_rmse_r'] for p in fit['curve'][1:]) if use6 else None,
                parameters=fit['parameters'], seconds=fit['seconds']))
    assert len(rows) == len(config['datasets']) * len(config['variants'])
    failures = [r for r in rows if r['reconstruction_rmse_r'] is not None
                and r['reconstruction_rmse_r'] >= config['validation_gate_rmse_r']]
    assert {(r['dataset'], r['variant']) for r in failures} == {(r['dataset'], r['variant']) for r in selection['failures']}
    assert selection['gate_passed'] == (not failures)
    summaries = []
    for variant in config['variants']:
        chosen = [r for r in rows if r['variant'] == variant]
        keys = ('average_precision', 'f1', 'reconstruction_rmse_r', 'branch_a_gate_classification', 'branch_a_gate_reconstruction')
        summaries.append(dict(variant=variant, flows=len(chosen), **{
            key: statistics.mean(r[key] for r in chosen) if all(r[key] is not None for r in chosen) else None
            for key in keys}))
    comparisons = []
    main = {r['dataset']: r for r in rows if r['variant'] == 'joint_task_gates'}
    for variant in config['variants'][1:]:
        other = {r['dataset']: r for r in rows if r['variant'] == variant}
        out = dict(control=variant)
        for key, direction in [('average_precision', 1), ('reconstruction_rmse_r', -1)]:
            if all(other[d][key] is not None for d in main):
                deltas = [main[d][key] - other[d][key] for d in main]
                mean_control = statistics.mean(other[d][key] for d in main)
                out[key] = dict(main_minus_control=statistics.mean(deltas),
                    relative_difference_percent=100 * statistics.mean(deltas) / mean_control,
                    main_better_flows=sum(direction * v > 0 for v in deltas), compared_flows=len(deltas))
        comparisons.append(out)
    runtime = [json.loads(line) for line in (root / 'runtime_events.jsonl').read_text().splitlines()]
    starts = {r['job_id']: r for r in runtime if r['state'] == 'STARTED'}
    ends = {r['job_id']: r for r in runtime if r['state'] == 'ENDED'}
    assert starts.keys() == ends.keys()
    timeline = sorted((r['time'], 1 if r['state'] == 'STARTED' else -1)
                      for r in runtime if r['phase'] == 'search')
    active, peak = 0, 0
    for _, change in timeline:
        active += change
        peak = max(peak, active)
    output = dict(experiment=config['experiment'], config_sha256=config_hash,
        recorded_scalar_checks_passed=True, scope='Nine-flow validation only, development seed 94110; no final test results',
        rows=rows, summary=summaries, comparisons=comparisons, failures=failures,
        processes=len(ends), successful_processes=sum(r['exit_code'] == 0 for r in ends.values()),
        failed_processes=[r for r in ends.values() if r['exit_code'] != 0], maximum_simultaneous_gpus=peak,
        first_training_start=min(r['time'] for r in starts.values() if r['phase'] == 'search'),
        last_training_end=max(r['time'] for r in ends.values() if r['phase'] == 'search'),
        final_test_submitted=any(r['phase'] == 'final' for r in runtime))
    (root / 'development_summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    with (root / 'development_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/Verify_Task36_MultiGate_1.1/evidence_development'))
    args = parser.parse_args()
    output = summarize(args.root)
    print(json.dumps({k: v for k, v in output.items() if k != 'rows'}, indent=2))
