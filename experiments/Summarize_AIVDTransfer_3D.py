"""Report every predeclared arm, with paired effects and no result selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from experiments.Audit_AIVDTransfer_3D import write_csv, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Verify_AIVDTransfer_1.1.json')
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    root = Path(spec['output_root'])
    audit = json.loads((root/'independent_audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['per_run_sha256'] == digest(root/'per_run.csv')
    with (root/'per_run.csv').open() as f:
        rows = list(csv.DictReader(f))
    metrics = ['f1', 'iou', 'precision', 'recall', 'balanced_accuracy', 'ari', 'nmi']
    dataset_rows, macro, comparisons = [], [], []
    for task in spec['tasks']:
        part = spec[task.lower()]
        for dataset in spec['datasets']:
            selected = [r for r in rows if r['task'] == task and r['dataset'] == dataset]
            for arm in part['arms']:
                record = {'task': task, 'dataset': dataset, 'arm': arm, 'seeds': len(part['seeds'])}
                current = [r for r in selected if r['arm'] == arm]
                for key in metrics:
                    values = [float(r[key]) for r in current]
                    record[key+'_mean'] = float(np.mean(values))
                    record[key+'_std'] = float(np.std(values, ddof=1))
                record['sample_count'] = int(current[0]['sample_count'])
                dataset_rows.append(record)
            for arm in ('aivd', 'aivd_kin4'):
                record = {'task': task, 'dataset': dataset, 'new_arm': arm, 'baseline': 'old_fmt'}
                for key in metrics:
                    delta = [next(float(r[key]) for r in selected if r['arm'] == arm and int(r['seed']) == s)
                             - next(float(r[key]) for r in selected if r['arm'] == 'old_fmt' and int(r['seed']) == s)
                             for s in part['seeds']]
                    record['delta_'+key+'_mean'] = float(np.mean(delta))
                    record['delta_'+key+'_std'] = float(np.std(delta, ddof=1))
                comparisons.append(record)
        for arm in part['arms']:
            record = {'task': task, 'arm': arm, 'datasets': len(spec['datasets']), 'seeds': len(part['seeds'])}
            for key in metrics:
                per_seed = [np.mean([float(r[key]) for r in rows if r['task'] == task and r['arm'] == arm and int(r['seed']) == s])
                            for s in part['seeds']]
                record[key+'_mean'] = float(np.mean(per_seed))
                record[key+'_macro_seed_std'] = float(np.std(per_seed, ddof=1))
                baseline = [np.mean([float(r[key]) for r in rows if r['task'] == task and r['arm'] == 'old_fmt' and int(r['seed']) == s])
                            for s in part['seeds']]
                record['delta_'+key+'_mean'] = float(np.mean(np.asarray(per_seed)-baseline))
                record['delta_'+key+'_macro_seed_std'] = float(np.std(np.asarray(per_seed)-baseline, ddof=1))
            if arm in ('aivd', 'aivd_kin4'):
                effects = [r['delta_f1_mean'] for r in comparisons if r['task'] == task and r['new_arm'] == arm]
                record['datasets_improved_f1'] = sum(d > 0 for d in effects)
                record['datasets_decreased_f1'] = sum(d < 0 for d in effects)
            macro.append(record)
    write_csv(root/'dataset_metrics.csv', dataset_rows)
    write_csv(root/'paired_comparisons.csv', comparisons)
    write_csv(root/'task_macro.csv', macro)
    summary = {'experiment': spec['experiment'], 'status': 'COMPLETE', 'audit': 'PASS',
               'shards': 100, 'metric_rows': 350, 'macro': macro,
               'cylinder_time_minimum': 7.5, 'old_baselines_retrained': True,
               'benchmark': spec['protocol']['benchmark']}
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    labels = {'channel': 'Channel', 'cylinder3d': 'Half-cylinder Re160',
              'halfcylinderRe640': 'Half-cylinder Re640', 'halfcylinderRe6400': 'Half-cylinder Re6400',
              'tangaroa': 'Tangaroa', 'deltaWing_resampled': 'Delta wing resampled',
              'deltaWing_LBM': 'Delta wing LBM', 'f22raptor': 'F22',
              'boeing747': 'Boeing747', 'smokeBuoyancy': 'Smoke buoyancy'}
    lines = ['# aivd1w3_dft 移植到 Task1、Task2：性能报告', '',
             '实验：Verify_AIVDTransfer_1.1。全部10个3D数据条目、每项5个种子；100个分片、350条指标记录均通过独立审计。所有Cylinder仅保留t≥7.5，旧FMT及Task2 Raw均重新训练。', '',
             '旧FMT为fmt_all+kin4（189维）；主转移臂仅使用aivd1w3_dft（1维）；附加对照为aivd1w3_dft+kin4（29维），只替换fmt_all核心。以下F1为0–1尺度，变化量为新减旧，pp表示百分点。每个流场给出五种子均值±样本标准差。', '',
             '所有方法使用相同任务内数据和有效primitive，维持原训练/验证/测试归属。Re160/Re6400 Task2仅余t=7.7和8.8两片训练数据，优化更新仍固定7000次。此次数据是已用过benchmark的后期子集，不是新独立确认集；不能把新结果直接与旧全时段结果解释为纯方法变化。', '']
    for task in spec['tasks']:
        lines += [f'## {task}', '', '| 数据 | 旧FMT | AIVD单独 | Δ (pp) | AIVD+kin4 | Δ (pp) |'+(' Raw+VAE |' if task == 'Task2' else ''),
                  '|---|---:|---:|---:|---:|---:|'+('---:|' if task == 'Task2' else '')]
        for dataset in spec['datasets']:
            d = {r['arm']: r for r in dataset_rows if r['task'] == task and r['dataset'] == dataset}
            def fmt(arm):
                return f"{d[arm]['f1_mean']:.4f}±{d[arm]['f1_std']:.4f}"
            old = d['old_fmt']['f1_mean']
            lines.append(f"| {labels[dataset]} | {fmt('old_fmt')} | {fmt('aivd')} | {(d['aivd']['f1_mean']-old)*100:+.2f} | {fmt('aivd_kin4')} | {(d['aivd_kin4']['f1_mean']-old)*100:+.2f} |"+ (f" {fmt('raw')} |" if task == 'Task2' else ''))
        m = {r['arm']: r for r in macro if r['task'] == task}
        lines.append(f"| **10项等权平均** | {m['old_fmt']['f1_mean']:.4f} | {m['aivd']['f1_mean']:.4f} | {m['aivd']['delta_f1_mean']*100:+.2f} | {m['aivd_kin4']['f1_mean']:.4f} | {m['aivd_kin4']['delta_f1_mean']*100:+.2f} |"+(f" {m['raw']['f1_mean']:.4f} |" if task == 'Task2' else ''))
        lines.append('')
    lines += ['## 解释边界', '',
              '- Task1：原方法和29维对照保留PCA8；1维输入无需降维，跳过PCA。编码器无可训练参数，标准化/PCA/KMeans仍需拟合。验证标签只确定簇名称，未搜索额外分类阈值。',
              '- Task2：四臂保持相同隐藏层[512,256]、潜变量64、KL权重1e−6、学习率3e−4和7000更新。输入/输出宽度不同，参数总量不同；详见training_diagnostics.csv。每个数据与种子内四臂使用同一GPU。',
              '- AIVD由几何估计而来，未读取真实curl、IVD或测试标签作为特征。原始配方按采样序号求导，按整个源时间片的有效primitive求平均涡量。',
              '- aivd1w3_dft只有前三个标量的零频和，没有非零频率信息。这次性能比较不能证明傅里叶频率分析的贡献，也不构成严格客观性的验证。',
              '- 所有预注册臂和种子均报告；没有依结果选择参数、epoch或流场。正负差为描述性结果，未做显著性或等效性检验。', '',
              '完整逐次数据：per_run.csv；逐流场指标：dataset_metrics.csv；配对差值：paired_comparisons.csv；任务平均：task_macro.csv；独立审计：independent_audit.json；硬件与执行时间：execution_records.csv。未写入或下载模型checkpoint。', '']
    (root/'report_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
