"""Read-only statistical report from the completed, independently audited search.

Run after Audit_FMTv8_Search_2_2, using only archived JSON evidence. This report
cannot train, select a model, or modify the frozen experiment configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


TASKS = ('Task3', 'Task5', 'Task4C')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def moments(values):
    assert len(values) == 3
    return dict(mean=statistics.mean(values), std=statistics.stdev(values))


def elapsed_seconds(value):
    days, sep, clock = value.partition('-')
    if not sep:
        clock, days = days, '0'
    parts = [int(v) for v in clock.split(':')]
    assert len(parts) == 3
    return int(days)*86400 + parts[0]*3600 + parts[1]*60 + parts[2]


def gpu_hours(rows):
    total = 0
    for row in rows:
        if '.' in row['JobID']:
            continue
        resources = dict(v.split('=', 1) for v in row['AllocTRES'].split(',') if '=' in v)
        total += elapsed_seconds(row['Elapsed'])*int(resources.get('gres/gpu', 0))
    return total/3600


def pool_description(pool):
    kind = pool['kind']
    if kind == 'absolute_top':
        return f"全局按绝对值保留{pool['k']}项、保留符号并按幅值排列，再加有符号均值"
    if kind == 'signed_extremes':
        k = pool['k']
        return f"全局保留数值最大{(k+1)//2}项和最小{k//2}项，再加有符号均值"
    if kind == 'indexed_top':
        return f"全局按绝对值选{pool['k']}项，保留符号和原138列位置，其余置零"
    if kind == 'semantic_statistics':
        names = dict(mean='均值', max='最大值', min='最小值', std='总体标准差', rms='均方根')
        return '每个描述量分别对六个排序值计算'+'＋'.join(names[v] for v in pool['statistics'])
    if kind == 'semantic_ranks':
        return '每个描述量保留数值降序排序位置'+str(pool['ranks'])+'（从0开始）'
    if kind == 'semantic_absolute_top':
        return f"每个描述量按绝对值保留{pool['k']}项，保留符号并按幅值排列"
    raise ValueError(kind)


def format_moments(row):
    return f"{row['mean']:.6f} ± {row['std']:.6f}"


def report(root, aborted):
    summary, lock = read(root/'summary.json'), read(root/'selection.lock.json')
    audit, completion = read(root/'independent_audit.json'), read(root/'completion_audit.json')
    spec = read(root/'run_config.json')
    assert audit['status'] == completion['status'] == 'PASS'
    assert audit['validation_fits'] == 1239 and audit['exact_control_repeats'] == 21
    assert completion['processes'] == 277 and completion['events'] == 554
    selected = (lock['selected']['pool'], lock['selected']['profile'])
    methods = list(dict.fromkeys([('p00', 'h0'), selected]))
    finals = [read(p) for p in sorted((root/'final').glob('*/*/*/seed*/final_result.json'))]
    assert len(finals) == audit['final_runs'] == 63*len(methods)
    seeds = {t:spec['final_task4_seeds'] if t == 'Task4C' else spec['final_seeds'] for t in TASKS}
    macro, flow, subgroup = {}, [], []
    for method in methods:
        for task in TASKS:
            rows = [r for r in finals if (r['pool'], r['profile'], r['task']) == (*method, task)]
            assert len({(r['parameters'], r['trainable_parameters'], r['feature_dimensions']) for r in rows}) == 1
            values = [statistics.mean(r['test']['f1'] for r in rows if r['seed'] == s) for s in seeds[task]]
            macro[(*method, task)] = dict(**moments(values), values=values)
            saved = next(r for r in summary['rows'] if (r['pool'], r['profile'], r['task']) == (*method, task))
            assert abs(saved['f1_mean']-statistics.mean(values)) < 1e-12
            assert abs(saved['f1_std']-statistics.stdev(values)) < 1e-12
            for dataset in sorted({r['dataset'] for r in rows}):
                part = [r for r in rows if r['dataset'] == dataset]
                flow.append(dict(pool=method[0], profile=method[1], task=task, dataset=dataset,
                                 **moments([r['test']['f1'] for r in part])))
            if task == 'Task4C':
                for index, name in enumerate(('Channel', 'TBL')):
                    subgroup.append(dict(pool=method[0], profile=method[1], flow=name,
                        **moments([r['per_group']['flow_index'][str(index)]['f1'] for r in rows])))
    paired = []
    for task in TASKS:
        baseline, chosen = macro[('p00', 'h0', task)], macro[(*selected, task)]
        differences = [b-a for a, b in zip(baseline['values'], chosen['values'])]
        paired.append(dict(task=task, baseline=baseline, selected=chosen,
                           absolute_change=chosen['mean']-baseline['mean'],
                           relative_change=chosen['mean']/baseline['mean']-1,
                           paired_absolute_change=moments(differences)))
    # Seed positions couple independent task repetitions for a descriptive SD.
    # This is not uncertainty over independent physical flow realizations.
    composite = {}
    for method in methods:
        values = [statistics.mean(macro[(*method, t)]['values'][i] for t in TASKS) for i in range(3)]
        composite['_'.join(method)] = dict(**moments(values), values=values)
    original = composite['p00_h0']['mean']; current = composite['_'.join(selected)]['mean']
    assert abs(current/original-1-summary['relative_test_gain']) < 1e-12
    old = read(aborted/'aborted_run_audit.json')
    cost = dict(current_allocated_gpu_hours=gpu_hours(completion['scheduler_rows']),
                aborted_allocated_gpu_hours=gpu_hours(old['scheduler_rows']),
                validation_classifier_seconds=summary['search_classifier_seconds'],
                validation_backbone_seconds=summary['search_backbone_seconds'],
                extra_repeatability_classifier_seconds=summary['repeatability_classifier_seconds'])
    cost['total_allocated_gpu_hours'] = cost['current_allocated_gpu_hours']+cost['aborted_allocated_gpu_hours']
    model_cost_rows = []
    for row in summary['rows']:
        part = [r for r in finals if (r['pool'],r['profile'],r['task']) == (row['pool'],row['profile'],row['task'])]
        extra = statistics.mean(r['raw_wide_selection_seconds'] for r in part)
        model_cost_rows.append(dict(**row, mean_raw_wide_selection_seconds=extra,
            mean_full_fit_seconds=statistics.mean(r['seconds']+r['raw_training_seconds']+r['raw_wide_selection_seconds'] for r in part)))
    pools = {p['id']:p for p in spec['pools']}
    baseline_validation = next(r['score'] for r in lock['ranking']
                               if (r['pool'], r['profile']) == ('p00', 'h0'))
    ranking = []
    for index, row in enumerate(lock['ranking'], 1):
        p = pools[row['pool']]
        ranking.append(dict(rank=index, pool=row['pool'], profile=row['profile'],
                            total_dimensions=p['feature_dimensions'], neighbor_dimensions=p['pooled_dimensions'],
                            operation=pool_description(p), composite_validation_f1=row['score'],
                            relative_validation_gain=row['score']/baseline_validation-1,
                            **{t+'_validation_f1':row['tasks'][t] for t in TASKS},
                            mean_parameters=row['mean_parameters'], fit_seconds=row['total_fit_seconds']))
    with (root/'candidate_ranking.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(ranking[0]));writer.writeheader();writer.writerows(ranking)
    ranking_lines = ['# 全部59组配置的验证结果', '',
        '实验：Ablation_FMTv8_Search_2.2。50个池化执行配置（48种不同表示），再对前三种不同表示各追加三种训练配置，共59组。下表按三任务等权验证F1排列，全部是验证结果；仅原V8与验证选定方案进行了最终测试。', '',
        'Task3/5各先取十流场F1平均，再与Task4-c 4.14的Channel＋TBL合并F1等权平均。验证搜索使用单种子：Task3/5为40，Task4-c为96611。相对变化以原V8 p00/h0为参照。', '',
        '所有配置保留中心频域23维和坐标／单位切向方向频谱72维。表中操作只作用于邻居频域138维（6个排序值×23种描述量）；特征维数不含Task4-c的有效线标记。均值保留符号，最大／最小值按数值比较；只有明确写“绝对值”的操作才按幅值选择。', '',
        '| 训练配置 | 改动 |', '|---|---|',
        '| h0 | 原版训练与投影配置 |',
        '| h1 | 加强正则化：Task3/5辅助分支dropout=0.15，Task4-c dropout=0.30；weight decay均为0.001 |',
        '| h2 | 使用SiLU激活函数（输入乘以其sigmoid值）；学习率0.0005；Task3/5辅助分支dropout=0.05 |',
        '| h3 | 对中心、方向、邻居三个特征块分别作可学习的线性投影、LayerNorm归一化和GELU激活，再拼接；固定几何池化仍无参数 |', '',
        '原V8为p00/h0。p31与p40、p29与p45分别为等价映射，已验证对应结果完全一致；这些重复配置没有被计作不同表示进入前三名细化。', '',
        '| 排名 | 配置 | 特征维数 | 邻居池化操作 | Task3 | Task5 | Task4-c | 等权均值 | 相对原V8 |',
        '|---:|---|---:|---|---:|---:|---:|---:|---:|']
    for r in ranking:
        ranking_lines.append(f"| {r['rank']} | {r['pool']}/{r['profile']} | {r['total_dimensions']} | {r['operation']} | {r['Task3_validation_f1']:.6f} | {r['Task5_validation_f1']:.6f} | {r['Task4C_validation_f1']:.6f} | {r['composite_validation_f1']:.6f} | {r['relative_validation_gain']:+.2%} |")
    ranking_lines += ['', '完整精度、参数统计与各配置21次分类／残差拟合秒数保存在candidate_ranking.csv；共享Raw骨干成本另见report_tables.md。', '',
                      f"科学源码：{summary['identity']['commit']}；选择证据：selection.lock.json；指标与逐次预测核验：independent_audit.json。", '']
    (root/'validation_rankings.md').write_text('\n'.join(ranking_lines), encoding='utf-8')
    result = dict(selected_pool=pools[selected[0]], selected_profile=selected[1],
                  selected_profile_settings=spec['profiles'][selected[1]],
                  selected_operation=pool_description(pools[selected[0]]), paired=paired,
                  composite=composite, relative_gain=current/original-1,
                  costs=cost, model_cost_rows=model_cost_rows, per_dataset=flow, task4_per_flow=subgroup,
                  scientific_commit=summary['identity']['commit'],
                  report_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'report_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines=['# FMT v8.2池化与训练搜索结果', '',
           f"选定`{selected[0]}/{selected[1]}`：{result['selected_operation']}。输入为中心频域23＋方向频谱72＋邻居池化{pools[selected[0]]['pooled_dimensions']}，共{pools[selected[0]]['feature_dimensions']}维。", '',
           '| 任务 | 本批原V8 | 选定方法 | 相对变化 | 配对差值均值 ± 标准差 |',
           '|---|---:|---:|---:|---:|']
    for row in paired:
        lines.append(f"| {row['task']} | {format_moments(row['baseline'])} | {format_moments(row['selected'])} | {row['relative_change']:+.2%} | {format_moments(row['paired_absolute_change'])} |")
    lines += ['',f"三任务等权测试F1：{original:.6f} → {current:.6f}，相对变化{current/original-1:+.2%}。", '',
              '标准差来自三个优化随机种子，不能解释为独立物理流场泛化的不确定性。选择只使用验证集；测试是此前已使用的benchmark。', '',
              '| 任务/流场 | 本批原V8 | 选定方法 |', '|---|---:|---:|']
    for task in ('Task3','Task5'):
        datasets=sorted({r['dataset'] for r in flow if r['task']==task})
        for dataset in datasets:
            values=[next(r for r in flow if (r['pool'],r['profile'],r['task'],r['dataset'])==(*m,task,dataset)) for m in [('p00','h0'),selected]]
            lines.append(f"| {task}/{dataset} | {format_moments(values[0])} | {format_moments(values[1])} |")
    for name in ('Channel','TBL'):
        values=[next(r for r in subgroup if (r['pool'],r['profile'],r['flow'])==(*m,name)) for m in [('p00','h0'),selected]]
        lines.append(f"| Task4C/{name} | {format_moments(values[0])} | {format_moments(values[1])} |")
    lines += ['', '| 方法 | 任务 | 特征维数 | 总参数 | 可训练参数 | 含参照网络的完整拟合秒 | Raw＋本模型秒 | 分类/残差阶段秒 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in model_cost_rows:
        lines.append(f"| {r['pool']}/{r['profile']} | {r['task']} | {r['input_dimensions']} | {r['parameters']:,} | {r['trainable_parameters']:,} | {r['mean_full_fit_seconds']:.3f} | {r['mean_training_seconds']:.3f} | {r['mean_residual_or_classifier_seconds']:.3f} |")
    lines += ['', '特征维数仅指有效几何特征：Task4-c另加一个有效线标记；Task3/5辅助输入补零到433槽并保留原Raw骨干。', '',
              'Raw指直接读取轨迹的骨干网络；Raw-wide是加宽的Raw参照网络，沿用原流程用于验证阶段的比较，不属于最终分类推理网络。完整拟合列计入一次Raw、一次Raw-wide及本方法残差，表示单独重训该方法的参考开销；本次共享骨干在总成本中只计算一次。不同表示和分支投影可能有不同参数量，不能都称同参数比较。', '',
              '计时范围：Task4-c拟合计时含池化、标准化和训练中的验证；Task3/5辅助特征准备在残差训练计时之外。完整轨迹积分与几何编码未计入该列；总GPU占用另按Slurm核算。时间来自共享集群运行，会受节点负载影响。', '',
              f"本批总分配GPU时间{cost['current_allocated_gpu_hours']:.3f}小时；停止的2.1及其诊断{cost['aborted_allocated_gpu_hours']:.3f}小时；合计{cost['total_allocated_gpu_hours']:.3f}小时。此数包含预检、缓存处理和评估期间的GPU占用，不等于纯训练时间。", '',
              f"1,239次主验证拟合的分类/残差训练合计{cost['validation_classifier_seconds']/3600:.3f}小时，共同Raw/Raw-wide骨干{cost['validation_backbone_seconds']/3600:.3f}小时，额外21次重复性拟合{cost['extra_repeatability_classifier_seconds']/3600:.3f}小时。", '',
              '完整59候选验证排名见`validation_rankings.md`和`candidate_ranking.csv`；原50执行配置含两对等价映射，即48种不同表示。', '',
              f"科学源码`{result['scientific_commit']}`；独立复算1,239份验证及{len(finals)}份最终测试预测、21对重复和42对等价结果，核对277个调度进程/554次起止事件，全部通过。", '']
    (root/'report_tables.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(selected=selected, paired=paired, relative_gain=current/original-1, costs=cost), ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/Ablation_FMTv8_Search_2.2'))
    parser.add_argument('--aborted', type=Path, default=Path('outputs/Ablation_FMTv8_Search_2.1'))
    args=parser.parse_args();report(args.root, args.aborted)
