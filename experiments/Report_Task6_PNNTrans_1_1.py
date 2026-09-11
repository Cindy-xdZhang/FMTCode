"""Validate and summarize saved validation records without loading test data."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(root, config):
    root = Path(root)
    spec, selection = read(config), read(root/'selection.json')
    config_hash = digest(config)
    assert digest(root/'config.frozen.json') == config_hash
    assert digest(root/'selection.json') == digest(root/'selection.before_test.json')
    assert selection['provenance']['config_sha256'] == config_hash and not selection['test_read']
    expected_commit = selection['provenance']['git_commit']
    rows, memorization = [], []
    for dataset in spec['datasets']:
        check = read(root/'fit_check'/dataset/'result.json')
        assert check['passed'] and not check['test_read']
        memorization.extend(dict(dataset=dataset, **r) for r in check['rows'])
        for candidate in spec['candidates']:
            folder = root/'search'/dataset/candidate['id']
            result = read(folder/'result.json')
            assert result['provenance']['config_sha256'] == config_hash
            assert result['provenance']['git_commit'] == expected_commit
            assert not result['test_read'] and sorted(r['arm'] for r in result['rows']) == sorted(spec['arms'])
            for arm in spec['arms']:
                fit = read(folder/arm/'fit.json')
                row = next(r for r in result['rows'] if r['arm'] == arm)
                selected = next(r for r in selection['rows'] if r['dataset'] == dataset
                                and r['candidate'] == candidate['id'] and r['arm'] == arm)
                assert fit['train_samples'] == spec['primary_train_size']
                assert fit['validation_samples'] == 12000
                assert fit['updates'] == spec['training']['updates'] == 12000
                assert fit['train_examples_exposed'] == 1536000
                best = min((r for r in fit['curve'] if r['step'] > 0), key=lambda r:r['validation_rmse_r'])
                assert fit['selected_step'] == row['selected_step'] == selected['selected_step'] == best['step']
                for key in ('train_rmse_r','validation_rmse_r'):
                    assert math.isclose(fit[key], row[key], rel_tol=1e-12)
                    assert math.isclose(fit[key], selected[key], rel_tol=1e-12)
                    assert math.isclose(fit[key], best[key], rel_tol=1e-12)
                rows.append(dict(dataset=dataset, **row))
    assert len(rows) == len(selection['rows']) == 36 and len(memorization) == 18
    selected_id = selection['candidate']['id']
    chosen = [r for r in rows if r['candidate'] == selected_id]
    means = {arm:{key:statistics.mean(r[key] for r in chosen if r['arm'] == arm)
                  for key in ('train_rmse_r','validation_rmse_r')} for arm in spec['arms']}
    wins = sum(next(r for r in chosen if r['dataset'] == d and r['arm'] == 'pnn_trans')['validation_rmse_r']
               < next(r for r in chosen if r['dataset'] == d and r['arm'] == 'raw_trans')['validation_rmse_r']
               for d in spec['datasets'])
    assert means['pnn_trans']['validation_rmse_r'] == selection['scores'][selected_id]
    failures = [r for r in chosen if r['validation_rmse_r'] >= spec['validation_gate_rmse_r']]
    assert bool(selection['gate_passed']) == (len(failures) == 0)
    assert not (root/'final_submission.lock').exists() and not (root/'final').exists()
    submissions = [json.loads(line) for line in (root/'submissions.jsonl').read_text().splitlines()]
    assert len(submissions) == 6 and not any(r['phase'] in ('final','audit','merge') for r in submissions)
    with (root/'validation_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(experiment=spec['experiment'], scientific_commit=expected_commit,
        config_sha256=config_hash, selection_sha256=digest(root/'selection.json'),
        checked_search_models=len(rows), fit_checks=len(memorization), record_consistency_passed=True,
        gate_passed=selection['gate_passed'], selected_candidate=selection['candidate'], means=means,
        pnn_validation_wins=wins, datasets=len(spec['datasets']), failures=failures,
        final_submitted=False, test_evaluated=False, rows=chosen,
        limits='Saved record consistency only; no checkpoint replay, no new training, no test metrics.')
    (root/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='outputs/Verify_Task6_PNNTrans_1.1')
    parser.add_argument('--config',default='config/Verify_Task6_PNNTrans_1.1.json')
    args = parser.parse_args()
    report(args.root,args.config)
