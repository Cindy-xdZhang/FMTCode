"""Reconcile validation/test tables directly from archived predictions.

Read-only with respect to scientific evidence; no training imports or metric
library calls. F1 is calculated from integer confusion counts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import numpy as np


VERSION = 'Verify_FMTv8_TableProvenance_1.1'
TASKS = ('Task3', 'Task5', 'Task4C')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_source_metadata(root):
    """Emit a fresh digest of the original 4.14 arrays without loading features."""
    manifest = read(root/'encoding.json')
    result = dict(encoding_sha256=sha((root/'encoding.json').read_bytes()), roles={})
    for split in ('validation', 'test'):
        arrays = {k:[] for k in ('labels', 'scale_id', 'head_component', 'instance', 'flow_index')}
        files = []
        for index, flow in enumerate(('channel', 'tbl')):
            path = root/flow/split/'metadata.npz'
            digest = sha(path.read_bytes())
            assert digest == manifest['splits'][flow+'/'+split]['files']['metadata.npz']
            with np.load(path, allow_pickle=False) as z:
                for key in arrays:
                    arrays[key].append(np.full(len(z['labels']), index, dtype=np.int64)
                                       if key == 'flow_index' else z[key])
            files.append(dict(path=str(path), sha256=digest))
        arrays = {k:np.concatenate(v) for k,v in arrays.items()}
        result['roles'][split] = dict(samples=len(arrays['labels']),
            positive_count=int(arrays['labels'].sum()), files=files,
            array_hashes={k:sha(np.asarray(v, dtype='<i8').tobytes()) for k,v in arrays.items()})
    print(json.dumps(result))


def audit(root):
    spec = read(root/'run_config.json')
    lock = read(root/'selection.lock.json')
    summary = read(root/'summary.json')
    identity = summary['identity']
    source = read(root/'table_provenance_source_check.json')
    assert source['encoding_sha256'] == spec['task4_encoding_sha256']
    needed = {'run_config.json', 'selection.lock.json', 'summary.json'}
    records = []
    max_error = 0.
    normalizations = []

    def prediction(path, split):
        nonlocal max_error
        row = read(path)
        assert all(row['identity'][k] == identity[k] for k in
                   ('commit', 'config_sha256', 'source_manifest_sha256'))
        filename = 'test_predictions.npz' if split == 'test' else 'validation_predictions.npz'
        p = path.parent/filename
        expected_sha = row['test_predictions_sha256' if split == 'test' else 'predictions_sha256']
        assert sha(p.read_bytes()) == expected_sha
        needed.update((path.relative_to(root).as_posix(), p.relative_to(root).as_posix()))
        with np.load(p, allow_pickle=False) as z:
            labels, probability, threshold = z['labels'], z['probability'], float(z['threshold'])
            assert labels.ndim == probability.ndim == 1 and labels.shape == probability.shape
            assert np.isin(labels, [0, 1]).all() and np.isfinite(probability).all()
            assert ((probability >= 0) & (probability <= 1)).all()
            assert threshold == row['threshold']
            pred, positive = probability >= threshold, labels == 1
            tp = int(np.count_nonzero(pred & positive))
            fp = int(np.count_nonzero(pred & ~positive))
            fn = int(np.count_nonzero(~pred & positive))
            tn = int(np.count_nonzero(~pred & ~positive))
            f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
            error = abs(f1-row[split]['f1'])
            max_error = max(max_error, error)
            assert error < 1e-12, path
            if row['task'] == 'Task4C':
                assert threshold == .5
                original = source['roles'][split]
                assert len(labels) == original['samples']
                for key, digest in original['array_hashes'].items():
                    assert sha(np.asarray(z[key], dtype='<i8').tobytes()) == digest, (path, key)
                if split == 'validation':
                    assert row['validation']['confusion_matrix'] == [[tn, fp], [fn, tp]]
                    epoch = next(h for h in row['history'] if h['epoch'] == row['selected_epoch'])
                    assert abs(epoch['validation_f1']-f1) < 1e-12
                    assert abs(max(h['validation_f1'] for h in row['history'])-f1) < 1e-12
                if row['pool'] == 'p35' and row['profile'] == 'h0':
                    assert row['feature_dimensions'] == 141 and row['parameters'] == 76738
                    normalizations.append(row['normalization'])
        records.append(dict(phase=path.relative_to(root).parts[0], split=split,
            task=row['task'], dataset=row['dataset'], pool=row['pool'], profile=row['profile'],
            seed=row['seed'], samples=len(labels), positive_count=tp+fn,
            tp=tp, fp=fp, fn=fn, tn=tn, f1=f1, threshold=threshold,
            selected_epoch=row.get('selected_epoch'), job=row['identity']['job'],
            result_path=path.relative_to(root).as_posix(), predictions_sha256=expected_sha))

    for phase in ('search', 'refine'):
        for path in sorted((root/phase).glob('*/*/*/seed*/result.json')):
            assert read(path)['test_read'] is False
            prediction(path, 'validation')
    assert len(records) == 1239
    print('Recomputed 1239 search/refinement validation predictions.', flush=True)
    search_records = list(records)
    for path in sorted((root/'final').glob('*/*/*/seed*/final_result.json')):
        row = read(path)
        assert row['selection_lock_sha256'] == sha((root/'selection.lock.json').read_bytes())
        assert row['models_frozen_before_test'] > datetime.fromisoformat(lock['selected_at_utc']).timestamp()
        prediction(path, 'validation')
        prediction(path, 'test')
    assert len(records) == 1239+2*126
    assert all(n == normalizations[0] for n in normalizations)

    groups = defaultdict(list)
    for row in search_records:
        groups[row['pool'], row['profile'], row['task']].append(row['f1'])
    for candidate in lock['ranking']:
        task_means = []
        for task in TASKS:
            values = groups[candidate['pool'], candidate['profile'], task]
            assert len(values) == (1 if task == 'Task4C' else 10)
            value = statistics.mean(values)
            assert abs(value-candidate['tasks'][task]) < 1e-12
            task_means.append(value)
        assert abs(statistics.mean(task_means)-candidate['score']) < 1e-12
    assert max(lock['ranking'], key=lambda x:x['score']) == lock['selected']
    for aggregate in summary['rows']:
        part = [r for r in records if r['split'] == 'test' and
                (r['pool'], r['profile'], r['task']) ==
                (aggregate['pool'], aggregate['profile'], aggregate['task'])]
        per_seed = defaultdict(list)
        for row in part:
            per_seed[row['seed']].append(row['f1'])
        assert len(per_seed) == 3
        values = [statistics.mean(v) for v in per_seed.values()]
        assert abs(statistics.mean(values)-aggregate['f1_mean']) < 1e-12
        assert abs(statistics.stdev(values)-aggregate['f1_std']) < 1e-12

    manifest = read(root/'evidence_manifest.json')
    archive = root/'metrics_evidence.tar.gz'
    assert sha(archive.read_bytes()) == manifest['archive_sha256']
    checked = set()
    with tarfile.open(archive, 'r:gz') as z:
        for member in z:
            if member.name in needed:
                assert member.isfile()
                assert sha(z.extractfile(member).read()) == sha((root/member.name).read_bytes())
                checked.add(member.name)
    assert checked == needed
    focus = [r for r in records if r['task'] == 'Task4C' and
             (r['pool'], r['profile']) == ('p35', 'h0')]
    outcome = dict(version=VERSION, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        status='PASS', max_f1_difference=max_error, validation_search_arrays=1239,
        final_validation_arrays=126, final_test_arrays=126, candidate_aggregates=59,
        final_aggregates=6, archive_files_matched=len(checked),
        original_archive_sha256=manifest['archive_sha256'], scientific_commit=identity['commit'],
        source_check_sha256=sha((root/'table_provenance_source_check.json').read_bytes()),
        verifier_sha256=sha(Path(__file__).read_bytes()), task4_source_labels_and_groups_exact=True,
        p35_task4_training_normalization_equal=True, focus=focus)
    (root/'table_provenance_audit.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# p35/h0两张表的数值来源核验', '',
        '实验：Verify_FMTv8_TableProvenance_1.1。只读复算既有预测，不重训、不更改指标。', '',
        '0.7270560190703218是搜索种子96611的验证F1；0.7086847701194756是最终种子96621/96622/96623的测试F1算术平均。配置相同，训练运行、数据集合及统计方式不同。', '',
        '| 阶段 | 集合 | 种子 | 样本数 | 真正例 TP | 假正例 FP | 假负例 FN | F1 | 验证选定轮次 | 作业ID |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in focus:
        lines.append(f"| {row['phase']} | {row['split']} | {row['seed']} | {row['samples']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['f1']:.12f} | {row['selected_epoch']} | {row['job']} |")
    values = [r['f1'] for r in focus if r['split'] == 'test']
    lines += ['', f"最终测试均值 ± 样本标准差：{statistics.mean(values):.12f} ± {statistics.stdev(values):.12f}。", '',
        'F1直接由整数计数计算：2×TP / (2×TP＋FP＋FN)，全部Task4-c预测阈值均为0.5。三个最终模型分别由各自验证集最佳轮次选定；不是把搜索种子96611的同一个模型评估三遍。', '',
        f"复算1,239份搜索验证预测、126份最终训练的验证预测、126份最终测试预测；59组验证汇总与6组最终均值/标准差全部吻合，F1最大浮点差{max_error:.3g}。", '',
        f"核对{len(checked)}份本地原始证据与原归档逐文件SHA256一致。Task4-c所有预测内标签及流场、尺度、头区、实例数组与Ibex原4.14验证/测试metadata逐元素一致。验证3,000例含449正例；测试10,000例含1,767正例。", '',
        '本核验确认表中F1的来源与计算，不把验证到测试的差值归因于某个单一原因，也不把测试集改称独立涡实例确认集。', '',
        '## 对应原始记录', '']
    for row in focus:
        p = (root/row['result_path']).resolve().as_posix()
        lines.append(f"- {row['split']} / seed {row['seed']}：[原始记录]({p})，预测SHA256 `{row['predictions_sha256']}`。")
    (root/'table_provenance_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in outcome.items() if k != 'focus'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/Ablation_FMTv8_Search_2.2'))
    parser.add_argument('--source-root', type=Path, help='Read original Task4-c metadata and emit JSON only')
    args = parser.parse_args()
    check_source_metadata(args.source_root) if args.source_root else audit(args.root)
