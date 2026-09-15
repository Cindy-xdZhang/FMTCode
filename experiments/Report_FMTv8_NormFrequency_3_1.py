"""Generate user tables only after independent P35 3.1 audit passes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def render(root):
    data=json.loads((root/'summary.json').read_text())
    audit=json.loads((root/'independent_audit.json').read_text())
    assert audit['status']=='PASS' and data['selected']['id']==audit['selected']['id']
    ranking=data['ranking'];best=ranking[0];baseline=data['baseline']
    text=['# P35统一归一化与频率：20组结果','',
          '版本：Ablation_FMTv8_NormFrequency_3.1；本表所有F1均来自原验证集合，参与模型和配置选择。',
          'Task4-c沿用4.14的27,000训练束与3,000验证束，不混入后续采样版本。',
          'Task3/5固定种子40，分别对10流场取平均；Task4-c固定种子96611，Channel与TBL合并。综合分为三任务F1等权平均。单种子结果，不写成三种子均值。','',
          '|归一化|几何尺度|DFT与池化后处理|','|---|---|---|',
          '|n0|最大半径|训练集逐特征标准化|',
          '|n1|最大半径|保留符号log1p→训练集逐特征标准化→截断[-8,8]|',
          '|n2|均方根半径|训练集逐特征标准化|',
          '|n3|均方根半径|保留符号log1p→训练集逐特征标准化→截断[-8,8]|','',
          '四组均先减有效点质心，统一邻居差分倍率和特征权重为1。归一化统计只取训练数据；Task4-c补零线不参与。K包含零频率，总维数24K−3。','',
          '|排名|配置|频率数K|维数|Task3 F1|Task5 F1|Task4-c F1|三任务均值|相对原p35|',
          '|---:|---|---:|---:|---:|---:|---:|---:|---:|',
          f"|参考|原p35/h0|6|141|{baseline['Task3']:.6f}|{baseline['Task5']:.6f}|{baseline['Task4C']:.6f}|{baseline['score']:.6f}|—|"]
    for i,row in enumerate(ranking,1):
        t=row['tasks'];gain=100*(row['score']/baseline['score']-1)
        text.append(f"|{i}|{row['id']}|{row['frequencies']}|{row['feature_dimensions']}|{t['Task3']:.6f}|{t['Task5']:.6f}|{t['Task4C']:.6f}|{row['score']:.6f}|{gain:+.2f}%|")
    text.extend(['','## 五档频率下的归一化平均表现','',
        '下表对每种归一化的五档K取等权平均，仅描述本次网格中的整体表现，不替代按单个共同配置排名。','',
        '|归一化|Task3 F1|Task5 F1|Task4-c F1|三任务均值|','|---|---:|---:|---:|---:|'])
    for recipe in ('n0','n1','n2','n3'):
        rows=[r for r in ranking if r['id'].startswith(recipe+'_')]
        assert len(rows)==5
        scores=[sum(r['tasks'][task] for r in rows)/5 for task in ('Task3','Task5','Task4C')]
        text.append('|'+recipe+'|'+ '|'.join(f'{value:.6f}' for value in scores+[sum(scores)/3])+'|')
    text.extend(['','## 共同最佳方案与原p35的逐流场结果','',
        '|任务|流场|原p35|共同最佳|差值|','|---|---|---:|---:|---:|'])
    controls={(r['task'],r['dataset']):r['f1'] for r in audit['counts'] if r['phase']=='controls'}
    selected_rows=[r for r in audit['counts'] if r['phase']=='search' and r['candidate']==best['id']]
    for row in sorted(selected_rows,key=lambda r:(r['task'],r['dataset'])):
        previous=controls[row['task'],row['dataset']]
        text.append(f"|{row['task']}|{row['dataset']}|{previous:.6f}|{row['f1']:.6f}|{row['f1']-previous:+.6f}|")
    text.extend(['','## 参数与训练代价','',
        'Task3/5全部142,914参数：冻结Raw 89,921，训练残差52,993；新特征补零到433维。',
        'Task4-c随输入宽度改变第一层参数；其余网络和h0训练保持。以下时间为每个任务单元平均耗时，Task3/5另有共享骨干的一次性训练成本。','',
        '|配置|任务|参数|特征准备秒|拟合秒|特征准备＋拟合秒|','|---|---|---:|---:|---:|---:|'])
    for row in audit['costs']:
        if row['candidate']=='p35_h0':continue
        text.append(f"|{row['candidate']}|{row['task']}|{row['mean_parameters']:.0f}|{row['mean_feature_seconds']:.2f}|{row['mean_fitting_seconds']:.2f}|{row['mean_encoding_and_fitting_seconds']:.2f}|")
    raw=wide=0.
    for path in (root/'cache').glob('*/*/temporary_backbones/backbones.json'):
        r=json.loads(path.read_text())['results'];raw+=r['raw']['seconds'];wide+=r['raw_wide']['seconds']
    text.extend(['',f'共享Raw骨干训练合计{raw/60:.3f}分钟，原选择规则所需Raw-wide合计{wide/60:.3f}分钟；每套只训练一次，20组共同复用。',
        f"本次登记{audit['scheduler_processes']}项Slurm作业，实际执行{audit['executed_processes']}项，4项Task4-c作业在启动前替换为20项短作业；累计分配GPU {audit['gpu_allocated_hours']:.3f}卡小时，包含两次工程预检、共享准备、原p35复现和新候选拟合。",'',
        '## 核验依据','',
        f"441份逐样本预测、原源文件与准备缓存、21项原p35逐概率复现、149项调度记录（含4项启动前取消）及290条运行事件核对通过。独立核验代码SHA256：`{audit['auditor_sha256']}`。",'',
        '初始准备/原p35复现commit：0d77e3e5；新候选训练commit：c61434ac。搜索开始前已恢复Task4-c原V100/32束最近邻计算；64束检查中原GPU缓存逐数值复现，CPU会为236条有效线选出不同邻居集合。没有把此设备差异纳入候选结果。',
        '原p35保持原分任务归一化，单列参考；新K=6配方也改变了公共几何尺度和邻居倍率，因此不应称为原p35的逐值相同实现。所有结果都沿用同一评价集合角色，不混入历史test成绩。',''])
    (root/'performance_tables.md').write_text('\n'.join(text),encoding='utf-8')
    summary=dict(selected=best,baseline=baseline,relative_gain=best['score']/baseline['score']-1,
                 raw_training_seconds=raw,raw_wide_training_seconds=wide,gpu_hours=audit['gpu_allocated_hours'])
    (root/'report_metrics.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();render(a.root)
