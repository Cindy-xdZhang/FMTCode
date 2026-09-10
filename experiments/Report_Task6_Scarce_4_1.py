"""Summarize the frozen small-data comparison without selecting from test results."""
import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np

from FMT_Utils.FlowMapData_3D import sha256, write_json


def report(root, record=False):
    spec=json.loads((root/'config.frozen.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    assert sha256(root/'selection.json')==sha256(root/'selection.before_test.json')
    audit=json.loads((root/'final_audit.json').read_text())
    assert audit['passed'] and audit['rows']==9396
    assert audit['provenance']['config_sha256']==sha256(root/'config.frozen.json')
    assert audit['selection_sha256']==sha256(root/'selection.json')
    with (root/'metrics.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==audit['rows']
    index={(r['dataset'],int(r['train_size']),int(r['seed']),r['comparison'],r['role'],int(r['scale_id'])):r for r in rows}
    assert len(index)==len(rows)
    initialization_rows=[]
    for dataset in spec['datasets']:
        for n in spec['train_sizes']:
            folder=root/'final'/dataset/str(n)
            result=json.loads((folder/'result.json').read_text())
            assert result['selection_sha256']==audit['selection_sha256']
            assert result['provenance']['config_sha256']==sha256(root/'config.frozen.json')
            for fit in result['fits']:
                fitting=json.loads((folder/str(fit['seed'])/(fit['arm']+'_'+fit['candidate']['id'])/'fit.json').read_text())
                assert fitting['updates']==spec['training']['updates']
                assert fitting['initialization']['train_samples']==n
                assert fitting['initialization']['latent_dim']==spec['training']['latent_dim']
                assert fitting['selected_step']>0
                for comparison in fit['roles']:
                    initialization_rows.append(dict(dataset=dataset,train_size=n,seed=fit['seed'],comparison=comparison,
                        initial_validation_rmse_r=fitting['curve'][0]['validation_rmse_r'],
                        trained_validation_rmse_r=fitting['validation_rmse_r'],train_rmse_r=fitting['train_rmse_r']))
            for metric in result['metrics']:
                for comparison in metric['comparisons']:
                    row=index[(dataset,n,metric['seed'],comparison,metric['role'],metric['scale_id'])]
                    for key,value in metric.items():
                        if key in ('position_rmse_r','center_rmse_r','neighbor_rmse_r','endpoint_rmse_r','pair_distance_rmse_r','samples'):
                            np.testing.assert_allclose(float(row[key]),value,rtol=1e-12,atol=1e-12)
    summaries=[]
    aggregate=[]
    roles=['test','unseen_scale']
    comparisons=['fmt','raw_frozen','raw_matched','raw_selected']
    for n in spec['train_sizes']:
        for dataset in spec['datasets']:
            row=dict(dataset=dataset,train_size=n)
            for role in roles:
                for comparison in comparisons:
                    values=np.array([float(index[(dataset,n,s,comparison,role,-1)]['position_rmse_r']) for s in spec['seeds']])
                    row[f'{comparison}_{role}_mean']=float(values.mean())
                    row[f'{comparison}_{role}_std']=float(values.std(ddof=1))
                    row[f'{comparison}_{role}_max_seed']=float(values.max())
            summaries.append(row)
        for role in roles:
            sub=[r for r in summaries if r['train_size']==n]
            summary=dict(train_size=n,role=role)
            for comparison in comparisons:
                summary[comparison]=float(np.mean([r[f'{comparison}_{role}_mean'] for r in sub]))
            for comparison in comparisons[1:]:
                summary['fmt_reduction_vs_'+comparison]=1-summary['fmt']/summary[comparison]
                summary['fmt_winning_flows_vs_'+comparison]=sum(r[f'fmt_{role}_mean']<r[f'{comparison}_{role}_mean'] for r in sub)
                paired=[]
                for seed in spec['seeds']:
                    paired.append(float(np.mean([float(index[(d,n,seed,'fmt',role,-1)]['position_rmse_r'])-
                        float(index[(d,n,seed,comparison,role,-1)]['position_rmse_r']) for d in spec['datasets']])))
                summary['paired_macro_differences_vs_'+comparison]=paired
            aggregate.append(summary)
    summary=dict(experiment=spec['experiment'],audit=audit,selection_sha256=sha256(root/'selection.json'),
        selected_candidates={k:selection[k] for k in ('fmt','raw_frozen','raw_matched','raw_selected')},
        per_flow=summaries,aggregate=aggregate,initialization=initialization_rows,primary_train_size=spec['primary_train_size'],
        test_usage='Already used 3.1 benchmark; selection and model fitting use validation only',
        metrics_sha256=sha256(root/'metrics.csv'))
    write_json(root/'summary.json',summary)
    with (root/'per_flow_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(aggregate,indent=2))
    if record:
        log=Path('docs/experiment_log.md')
        marker='Verify_Task6_ScarceGeneralization_4.1 最终测试结果'
        assert marker not in log.read_text(encoding='utf-8')
        lines=[f'\n\n### {datetime.date.today().isoformat()} — {marker}\n',
            f"证据：训练代码`{audit['provenance']['git_commit']}`、配置SHA256 `{audit['provenance']['config_sha256']}`、选择SHA256 `{audit['selection_sha256']}`；独立复算9396条比较/尺度指标通过，实际唯一模型数{audit['models']}。\n",
            '此前3.1在24万训练primitive下Raw九流场均优于signed_fmt10；本版改变为有限训练数据及预注册正则化，结论需按本版条件报告，旧结果不改写。小训练集包含全部标准化/初始化/网络训练输入。测试仍为3.1已用benchmark，不是新的确认集。\n',
            '选择只使用1024样本的九流场平均validation；同一候选用于全部流场与三个样本量。实际选择：\n']
        for name in ('fmt','raw_frozen','raw_matched','raw_selected'):
            lines.append('- '+name+': `'+json.dumps(selection[name],ensure_ascii=False)+'`')
        lines += ['\n每流场三个新训练子集/优化种子的均值，再等权平均九流场。RMSE/r定义保持不变。\n',
            '| 训练数量 | 测试集合 | FMT | Raw原配方 | Raw相同正则化 | Raw验证最优 | FMT优于原Raw流场数 |',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in aggregate:
            lines.append(f"| {row['train_size']} | {row['role']} | {row['fmt']:.9g} | {row['raw_frozen']:.9g} | {row['raw_matched']:.9g} | {row['raw_selected']:.9g} | {row['fmt_winning_flows_vs_raw_frozen']}/9 |")
        primary=next(r for r in aggregate if r['train_size']==spec['primary_train_size'] and r['role']=='test')
        for control in comparisons[1:]:
            lines.append(f"\n主比较n={spec['primary_train_size']}：相对{control}的误差降低率{100*primary['fmt_reduction_vs_'+control]:.6g}%；负值表示FMT更差。三个配对子集种子的宏平均误差差值(FMT−Raw)为{primary['paired_macro_differences_vs_'+control]}。")
        lines += ['\n主设置逐流场（均值±样本标准差）：\n',
            '| 流场 | FMT | Raw原配方 | Raw相同正则化 | Raw验证最优 |',
            '|---|---:|---:|---:|---:|']
        for row in summaries:
            if row['train_size']==spec['primary_train_size']:
                lines.append('| '+row['dataset']+' | '+' | '.join(f"{row[c+'_test_mean']:.8g} ± {row[c+'_test_std']:.3g}" for c in comparisons)+' |')
        lines += ['\n主设置的初始化与训练后validation（九流场/三子集种子等权均值；不用于追加测试选择）：\n',
            '| 方法角色 | 初始化validation | 训练后validation | 训练集误差 |',
            '|---|---:|---:|---:|']
        for comparison in comparisons:
            values=[r for r in initialization_rows if r['train_size']==spec['primary_train_size'] and r['comparison']==comparison]
            lines.append('| '+comparison+' | '+' | '.join(f"{np.mean([r[key] for r in values]):.9g}" for key in
                ('initial_validation_rmse_r','trained_validation_rmse_r','train_rmse_r'))+' |')
        if selection['fmt']['frequencies']==16:
            lines.append('\n所选16频编码保留全部651个独立系数，相对31间隔信号不做频谱压缩；最终192维VAE潜变量才是压缩。结果不能归因于丢弃高频，也不属于旧161维fmt_all。')
        lines.append('\n不同Raw角色若候选相同，使用同一模型预测，不能当成独立证据。完整训练曲线、选择记录与逐次指标保存在`outputs/Verify_Task6_ScarceGeneralization_4.1/`；不保留checkpoint。')
        log.open('a',encoding='utf-8').write('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/Verify_Task6_ScarceGeneralization_4.1'))
    p.add_argument('--record-log',action='store_true');args=p.parse_args();report(args.root,args.record_log)
