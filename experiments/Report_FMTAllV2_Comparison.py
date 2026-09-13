"""Combine both audited comparisons; never select a model or omit a flow."""
import csv
import json
from pathlib import Path


def main():
    base=Path(__file__).resolve().parents[1]
    roots=[base/'outputs'/f'Verify_FMTAllV2_{v}' for v in ('1.1','1.2')]
    summaries=[]
    for root in roots:
        assert json.loads((root/'independent_audit.json').read_text())['status']=='PASS'
        summaries.append(json.loads((root/'summary.json').read_text()))
    output=base/'outputs/Verify_FMTAllV2_report'
    output.mkdir(exist_ok=True)
    lines=['# fmt_all_v2：实现与两组性能对照','',
           '日期：2026-09-07。覆盖全部10个3D数据条目，每任务5个随机种子；所有指标均来自已审计的预测。F1为精度和召回率的调和平均。表中总体均值对10个数据条目等权。标签沿用瞬时涡量偏差（Instantaneous Vorticity Deviation，IVD）的全场第95百分位阈值。Raw表示不使用FMT的轨线坐标输入。','',
           '## 两组比较的区别','',
           '- **1.1：仅用v2作为FMT输入。** Task1/2用231维v2替代旧fmt_all+kin4；Task5保留Raw输入，将FMT输入替换为v2，补零到268维。',
           '- **1.2：只替换fmt_all核心。** Task1/2保留kin4；Task5保留gram2+kin6，两臂统一338维、136834个网络参数。Task1旧版及Task2旧版/Raw结果从1.1复用，经逐文件、标签和指标校验。',
           '- 1.2是在1.1已产生部分结果后登记的配方对照，不是新的未见测试集。没有搜索超参数，也没有根据测试结果选择模型或epoch。Task2复用结果的种子配对可使用不同GPU，完整设备记录已保留。','',
           'kin4/kin6是旧邻居运动特征块：从邻居位移估计涡量偏差、应变大小、散度及旋转–应变指标，再保留4/6个非负傅里叶频率项（含直流项）。gram2是同时间邻居内积的2个非负频率项。','',
           '## 平均F1','',
           '| 任务 | 1.1：旧配方 → 仅v2 | 变化（百分点） | 1.2：旧配方 → 替换核心、保留附加块 | 变化（百分点） |',
           '|---|---:|---:|---:|---:|']
    joint=[]
    for task in ('Task1','Task2','Task5'):
        a,b=[next(r for r in s['macro'] if r['task']==task) for s in summaries]
        lines.append(f"| {task} | {a['old_fmt_f1_mean']:.4f} → {a['fmt_all_v2_f1_mean']:.4f} | {100*a['delta_f1_mean']:+.2f} | {b['old_fmt_f1_mean']:.4f} → {b['fmt_all_v2_f1_mean']:.4f} | {100*b['delta_f1_mean']:+.2f} |")
        joint.append({'task':task,'standalone_old_f1':a['old_fmt_f1_mean'],'standalone_v2_f1':a['fmt_all_v2_f1_mean'],
                      'replacement_old_f1':b['old_fmt_f1_mean'],'replacement_v2_f1':b['fmt_all_v2_f1_mean']})
    lines+=['',
            'Task5平均精度（Average Precision，AP）在1.1中为0.7195→0.6410，在1.2中为0.7195→0.7240。Task2的Raw F1为0.4885，Task5的Raw F1为0.5762。Task5两组旧版网络的输入宽度不同，因此必须在各自组内比较，不能混用两组旧版数值。',
            '', '1.2的Task5平均F1差为−0.12个百分点；配对差值在5个种子间的样本标准差为0.99个百分点。这里描述均值，没有进行显著性或等效性检验。','',
            '## 各流场结果','']
    for index,summary in enumerate(summaries):
        lines += [f"### {summary['experiment']}",'',
                  '| 流场 | Task1旧 → 新（百分点变化） | Task2旧 → 新（百分点变化） | Task5旧 → 新（百分点变化） |',
                  '|---|---:|---:|---:|']
        datasets=list(dict.fromkeys(r['dataset'] for r in summary['dataset_comparison']))
        for dataset in datasets:
            cells=[]
            for task in ('Task1','Task2','Task5'):
                r=next(r for r in summary['dataset_comparison'] if r['dataset']==dataset and r['task']==task)
                cells.append(f"{r['old_fmt_f1_mean']:.4f} → {r['fmt_all_v2_f1_mean']:.4f} ({100*r['delta_f1_mean']:+.2f})")
            lines.append('| '+dataset+' | '+' | '.join(cells)+' |')
        lines += ['',f"![{summary['experiment']}](../{summary['experiment']}/figures/fmt_all_v2_comparison.png)",'']
    lines += ['## 编码器与证据边界','',
              '代码：`FMT_Utils/FMTAllV2_3D.py`，公共入口`fmt_all_v2(pathlines)`。每个primitive包含同一批7条物质轨线；中心点只用于定义同时间邻居位移，没有单独的中心轨线特征。',
              '',
              '`d_i(t)=x_i(t)-x_0(t)`，`G_ij(t)=d_i(t)·d_j(t)`。对21个同时间内积序列按初始邻居尺度归一化、减去初始内积，再取6个实傅里叶系数和5个非零频率的虚部，得到231维。没有跨时间直接相减点坐标。',
              '',
              '对于任意时变刚体变换，差向量会随坐标轴旋转，但同时间内积不变，随后傅里叶系数也不变。合成测试与训练样本检查通过；最大绝对输出差1.86e−9，最大相对L2差1.75e−10。该证书仅针对v2核心，不覆盖保留的kinematic附加块或Raw输入的完整网络。频率单位是每个采样窗口的周期数，不直接解释为物理Hz。',
              '',
              '**时间边界：**本轮复用旧缓存及冻结划分，cylinder仍包含早期样本；没有重新生成满足t≥7.5的后半程基准。本表不能标注为后半程专用结果。',
              '',
              '**审计与保留：**两组共300个有效实验分片、650条新产生的总体指标；两份汇总各400行，其中1.2明确复用了150行。两组各1350条Task5逐尺度指标。一次训练前GPU初始化失败已记录并以同配置、同种子重试成功。所有本次临时模型在结果写完后删除，模型未下载。',
              '',
              '方法解释和结论统一记录于`docs/experiment_log.md`。配置、源代码快照、源文件哈希、设备、逐种子指标、逐尺度指标和独立审计位于两个实验输出目录。旧论文主表未被覆盖。','',
              '## 文件索引','',
              '- [1.1全部逐次指标](../Verify_FMTAllV2_1.1/per_run.csv)',
              '- [1.2全部逐次指标及复用来源](../Verify_FMTAllV2_1.2/per_run.csv)',
              '- [1.1逐流场均值和标准差](../Verify_FMTAllV2_1.1/dataset_comparison.csv)',
              '- [1.2逐流场均值和标准差](../Verify_FMTAllV2_1.2/dataset_comparison.csv)',
              '- [1.1矢量图PDF](../Verify_FMTAllV2_1.1/figures/fmt_all_v2_comparison.pdf)',
              '- [1.2矢量图PDF](../Verify_FMTAllV2_1.2/figures/fmt_all_v2_comparison.pdf)','']
    (output/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    with (output/'macro_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(joint[0]));writer.writeheader();writer.writerows(joint)
    print(output/'report_zh.md')


if __name__=='__main__':main()
