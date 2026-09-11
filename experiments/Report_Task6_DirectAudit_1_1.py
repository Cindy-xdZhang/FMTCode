"""Audit frozen scalar records without hiding failed FMT runs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = Path('outputs/Verify_Task6_DirectNeural_5.1')
    output = Path('outputs/Verify_Task6_DirectAudit_1.1')
    spec, selection, audit = [read(source/name) for name in ('config.frozen.json', 'selection.json', 'final_audit.json')]
    expected_commit = '1496fd7c31a6dae3d71fd25228dfba6bafa3b289'
    expected_hash = '6b3c62a72994b7d870259895ee3b5d74c65e80f0ce7b5fd2044df19fe3ec9227'
    assert digest(source/'config.frozen.json') == expected_hash
    assert digest(source/'selection.json') == digest(source/'selection.before_test.json') == audit['selection_sha256']
    fit_rows, metrics = [], []
    for d in spec['datasets']:
        for c in spec['candidates']:
            search = read(source/'search'/d/c['id']/'result.json')
            assert search['test_read'] is False
            for row in search['rows']:
                fit = read(source/'search'/d/c['id']/row['arm']/'fit.json')
                minimum = min(r['validation_rmse_r'] for r in fit['curve'] if r['step'] > 0)
                assert row['validation_rmse_r'] == fit['validation_rmse_r'] == minimum
    for arm in spec['arms']:
        for c in spec['candidates']:
            matching = [r['validation_rmse_r'] for r in selection['rows'] if r['arm'] == arm and r['candidate'] == c['id']]
            assert len(matching) == 9
            np.testing.assert_allclose(np.mean(matching), selection['scores'][arm+'_'+c['id']], rtol=1e-14)
        chosen = min(spec['candidates'], key=lambda c: (selection['scores'][arm+'_'+c['id']], c['id']))
        assert chosen == selection['fmt' if arm.startswith('signed') else 'raw_selected']
    assert selection['fmt'] == selection['raw_matched']
    for d in spec['datasets']:
        for n in spec['train_sizes']:
            for seed in spec['seeds']:
                folder = source/'final'/d/str(n)/str(seed)
                result = read(folder/'result.json')
                assert result['provenance']['git_commit'] == expected_commit
                assert result['provenance']['config_sha256'] == expected_hash
                assert result['selection_sha256'] == audit['selection_sha256']
                for method in ('signed_fmt16_neural_c2', 'raw_neural_c2', 'raw_neural_c1'):
                    fit = read(folder/method/'fit.json')
                    assert fit['train_samples'] == fit['statistics_train_samples'] == n
                    assert fit['updates'] == 20000 and fit['validation_samples'] == 12000
                    assert fit['train_examples_exposed'] == 5120000
                    best = min((r for r in fit['curve'] if r['step'] > 0), key=lambda r: r['validation_rmse_r'])
                    assert fit['selected_step'] == best['step'] and fit['validation_rmse_r'] == best['validation_rmse_r']
                    assert fit['structure']['analytic_inverse_path'] is False and fit['structure']['pca_in_neural_model'] is False
                    assert fit['structure']['trainable_parameters'] == 4267990
                    fit_rows.append(dict(dataset=d, train_size=n, seed=seed, method=method,
                        train_rmse_r=fit['train_rmse_r'], validation_rmse_r=fit['validation_rmse_r'],
                        failed_validation_gate=fit['validation_rmse_r'] >= 3, selected_step=fit['selected_step']))
                for fit in result['fits']:
                    for metric in fit['metrics']:
                        for comparison in fit['roles']:
                            metrics.append(dict(dataset=d, train_size=n, seed=seed, comparison=comparison,
                                **{k: v for k, v in metric.items() if k != 'time_rmse_r'}))
    keys = lambda r: (r['dataset'], int(r['train_size']), int(r['seed']), r['comparison'], r['role'], int(r['scale_id']))
    actual = {keys(r): r for r in metrics}
    assert len(actual) == len(metrics) == audit['rows']
    with (source/'metrics.csv').open(newline='') as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(actual)
    for row in csv_rows:
        other = actual[keys(row)]
        for k in ('position_rmse_r', 'all_time_rmse_r', 'initial_rmse_r', 'center_rmse_r', 'neighbor_rmse_r', 'endpoint_rmse_r', 'pair_distance_rmse_r', 'samples'):
            assert float(row[k]) == float(other[k])
    summaries = []
    for n in spec['train_sizes']:
        item = dict(train_size=n)
        for name, method in [('fmt', 'signed_fmt16_neural_c2'), ('raw_matched', 'raw_neural_c2'), ('raw_selected', 'raw_neural_c1')]:
            group = [r for r in fit_rows if r['train_size'] == n and r['method'] == method]
            assert len(group) == 27
            item[name+'_validation'] = float(np.mean([r['validation_rmse_r'] for r in group]))
            item[name+'_train'] = float(np.mean([r['train_rmse_r'] for r in group]))
            item[name+'_failed'] = sum(r['failed_validation_gate'] for r in group)
        for role in ('test', 'unseen_scale'):
            fmt_rows = [r for r in metrics if r['train_size'] == n and r['comparison'] == 'fmt' and r['role'] == role and r['scale_id'] == -1]
            item['fmt_'+role+'_available_models'] = len(fmt_rows)
            item['fmt_'+role+'_macro'] = None  # Never average an incomplete primary population.
            item['fmt_'+role+'_paired_wins'] = sum(r['position_rmse_r'] < actual[(r['dataset'], n, r['seed'], 'raw_matched', role, -1)]['position_rmse_r'] for r in fmt_rows)
            item['fmt_'+role+'_pair_distance_wins'] = sum(r['pair_distance_rmse_r'] < actual[(r['dataset'], n, r['seed'], 'raw_matched', role, -1)]['pair_distance_rmse_r'] for r in fmt_rows)
            for name in ('raw_matched', 'raw_selected', 'pca'):
                group = [r for r in metrics if r['train_size'] == n and r['comparison'] == name and r['role'] == role and r['scale_id'] == -1]
                assert len(group) == 27
                item[name+'_'+role+'_macro'] = float(np.mean([r['position_rmse_r'] for r in group]))
        summaries.append(item)
    report = dict(scalar_consistency_passed=True, experiment_passed=audit['passed'],
        scientific_commit=expected_commit, config_sha256=expected_hash, selection_sha256=audit['selection_sha256'],
        final_neural_models=len(fit_rows), independent_pca_models=81, evaluated_models=audit['models'],
        compared_rows=len(metrics), failed_fmt_models=len(audit['failures']), summaries=summaries,
        interpretation='Validation macro includes all 27 models per size. FMT test macro withheld because failed models are missing. Paired wins describe only actually evaluated models.')
    independent = [read(output/(d+'.json')) for d in spec['datasets']]
    assert all(r['independent_audit_passed'] for r in independent)
    assert sum(r['models_with_predictions'] for r in independent) == audit['models']
    assert sum(r['compared_rows'] for r in independent) == audit['rows']
    assert sum(len(r['failures']) for r in independent) == len(audit['failures'])
    report['independent_audit'] = dict(flows=9, data_and_saved_predictions_passed=True,
        max_metric_absolute_difference=max(r['max_metric_absolute_difference'] for r in independent),
        max_validation_roundtrip_rmse_r=max(r['diagnostic']['signed_fmt16_neural']['validation_analytic_roundtrip']['rmse_r'] for r in independent),
        relative_roundtrip_rmse_range=[min(r['diagnostic']['signed_fmt16_neural']['validation_analytic_roundtrip']['relative_rmse'] for r in independent), max(r['diagnostic']['signed_fmt16_neural']['validation_analytic_roundtrip']['relative_rmse'] for r in independent)])
    pairs = {(r['dataset'], r['train_size'], r['seed'], r['method']): r for r in fit_rows}
    report['fmt_validation_wins_vs_matched_raw'] = sum(r['validation_rmse_r'] < pairs[(r['dataset'], r['train_size'], r['seed'], 'raw_neural_c2')]['validation_rmse_r'] for r in fit_rows if r['method'] == 'signed_fmt16_neural_c2')
    output.mkdir(exist_ok=True)
    (output/'scalar_audit.json').write_text(json.dumps(report, indent=2))
    for name, rows in [('per_model_validation.csv', fit_rows), ('complete_validation_and_controls.csv', summaries)]:
        with (output/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
