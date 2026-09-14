"""Render audited FTLE tables; no training, selection, or test reevaluation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='outputs/Other_FTLEUpsampling2D_1.1')
    args=parser.parse_args()
    root=Path(args.root)
    audit=json.loads((root/'result_audit.json').read_text())
    assert audit['status']=='PASS'
    summary=json.loads((root/'summary.json').read_text())
    with (root/'metrics.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    methods=['bilinear','bicubic','espcn','unet','raw','fmt','fmt_objective_ntod_v2','fmt_v5']
    names={'bilinear':'双线性插值','bicubic':'双三次插值','espcn':'ESPCN','unet':'U-Net','raw':'Raw + U-Net',
           'fmt':'FMT + U-Net','fmt_objective_ntod_v2':'距离＋夹角 2.2 + U-Net','fmt_v5':'FMT v5 + U-Net'}
    flows=['cylinder2d','boussinesq','pipedcylinder2d','doublegyre2d']
    records=summary['results']
    lookup={(r['flow'],r['scale'],r['method']):r for r in records}
    macro=[]
    for method in methods:
        for scale in (2,4):
            group=[r for r in rows if r['method']==method and int(r['scale'])==scale and r['split']=='test']
            seeds=sorted({r['seed'] for r in group})
            values=[]
            for seed in seeds:
                flow_means=[np.mean([float(r['psnr']) for r in group if r['seed']==seed and r['flow']==flow]) for flow in flows]
                assert all(np.isfinite(flow_means))
                values.append(float(np.mean(flow_means)))
            macro.append({'method':method,'scale':scale,'psnr_mean':float(np.mean(values)),
                          'psnr_std':float(np.std(values)),'seed_macro_psnr':dict(zip(seeds,values))})
    comparisons=[]
    for method in ('fmt','fmt_objective_ntod_v2','fmt_v5'):
        for baseline in ('raw','unet','espcn','bicubic'):
            differences=[]
            for flow in flows:
                for scale in (2,4):
                    differences.append({'flow':flow,'scale':scale,
                                        'rmse_change':lookup[flow,scale,method]['rmse_mean']-lookup[flow,scale,baseline]['rmse_mean']})
            comparisons.append({'method':method,'baseline':baseline,
                                'lower_rmse_cases':sum(r['rmse_change']<0 for r in differences),'cases':differences})
    lines=['# 非定常二维流场的 FTLE 超分辨结果','',
           '实验：`Other_FTLEUpsampling2D_1.1`。科学提交：`9943fcfe10dccb9914cc981ab21112aa0c1a84cb`。',
           '四流场、两倍率、六种学习方法、三种子，共144次训练。全部模型权重仅保存在运行进程内存中。',
           '测试完整留出的积分时间窗，空间区域共享；本实验不证明未见流场泛化。各方法共享真值网格与有效掩码。',
           '每种子先等权平均三个测试切片，再报告种子均值 ± 总体标准差。',
           '每倍率有效训练patch数：cylinder 38、boussinesq 207、piped cylinder 193、doublegyre 252。Cylinder的训练样本尤其有限。',
           '“距离＋夹角2.2”对应`fmt_objective_ntod_v2`二维适配；三种编码器和Raw使用相同U-Net及32通道几何投影。', '',
           '## 四流场等权平均峰值信噪比（PSNR，dB）','',
           '峰值信噪比衡量相对于固定数据范围的重建误差，越高越好。使用各流场/倍率训练真值max−min作为固定范围，先在每个种子内平均四流场。',
           '', '| 方法 | 2× | 4× |','|---|---:|---:|']
    for method in methods:
        values=[]
        for scale in (2,4):
            r=next(r for r in macro if r['method']==method and r['scale']==scale)
            values.append(f"{r['psnr_mean']:.3f}"+(f" ± {r['psnr_std']:.3f}" if method not in ('bilinear','bicubic') else ''))
        lines.append('| '+names[method]+' | '+' | '.join(values)+' |')
    for metric,title in [('rmse','均方根误差（RMSE，FTLE物理单位，越低越好）'),('ssim','结构相似性（SSIM，比较局部结构，越高越好）')]:
        for scale in (2,4):
            lines.extend(['',f'## {scale}× {title}','','| 方法 | Cylinder | Boussinesq | Piped cylinder | Double gyre |',
                          '|---|---:|---:|---:|---:|'])
            for method in methods:
                values=[]
                for flow in flows:
                    r=lookup[flow,scale,method]
                    values.append(f"{r[metric+'_mean']:.5f}"+(f" ± {r[metric+'_std']:.5f}" if method not in ('bilinear','bicubic') else ''))
                lines.append('| '+names[method]+' | '+' | '.join(values)+' |')
    lines.extend(['','## 复算证据','',
                  f"独立审计：{audit['prediction_files_recomputed']}份预测、{audit['metric_rows']}行逐次指标、{audit['summary_rows']}行汇总，全部PASS。",
                  '逐次指标：`metrics.csv`；每次模型选择与完整训练曲线：`runs/`；源数据哈希和时刻：`data/*/manifest.json`。',
                  '两种插值是确定性方法，不报告虚构的种子标准差。',''])
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (root/'macro_and_paired_comparisons.json').write_text(json.dumps({'macro_psnr':macro,'comparisons':comparisons},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'macro_psnr':macro,'case_counts':[{k:r[k] for k in ('method','baseline','lower_rmse_cases')} for r in comparisons]},indent=2))


if __name__=='__main__':
    main()
