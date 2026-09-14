"""Report audited held-out time-slice results and paired dropout differences."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


NAMES={'bilinear':'双线性插值','bicubic':'双三次插值','espcn':'ESPCN','unet':'U-Net',
       'raw':'Raw + U-Net','fmt':'FMT + U-Net','fmt_objective_ntod_v2':'距离＋夹角 2.2 + U-Net',
       'fmt_v5':'FMT v5 + U-Net'}


def report(root):
    read=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
    audit=read('result_audit.json')
    assert audit['status']=='PASS'
    summary=read('summary.json')
    records=summary['results']
    with (root/'metrics.csv').open(newline='',encoding='utf-8') as stream:
        rows=[r for r in csv.DictReader(stream) if r['split']=='test']
    flows=sorted({r['flow'] for r in records})
    scales=sorted({r['scale'] for r in records})
    methods=[m for m in NAMES if m not in ('bilinear','bicubic')]
    lookup={(r['flow'],r['scale'],r['method'],r['dropout']):r for r in records}
    variants=[('bilinear',-1.),('bicubic',-1.)]+list(itertools.product(methods,[0.,.15]))
    macro=[]
    for method,dropout in variants:
        for scale in scales:
            group=[r for r in rows if r['method']==method and float(r['dropout'])==dropout and int(r['scale'])==scale]
            values={}
            for seed in sorted({r['seed'] for r in group}):
                by_flow=[np.mean([float(r['psnr']) for r in group if r['seed']==seed and r['flow']==flow]) for flow in flows]
                assert np.isfinite(by_flow).all()
                values[seed]=float(np.mean(by_flow))
            macro.append({'method':method,'dropout':dropout,'scale':scale,'seed_macro_psnr':values,
                          'psnr_mean':float(np.mean(list(values.values()))),'psnr_std':float(np.std(list(values.values())))})
    macro_lookup={(r['method'],r['scale'],r['dropout']):r for r in macro}
    paired=[]
    for method,scale in itertools.product(methods,scales):
        p0=macro_lookup[method,scale,0.]['seed_macro_psnr']
        p15=macro_lookup[method,scale,.15]['seed_macro_psnr']
        deltas=[p15[s]-p0[s] for s in p0]
        cases=[{'flow':flow,'rmse_p0':lookup[flow,scale,method,0.]['rmse_mean'],
                'rmse_p15':lookup[flow,scale,method,.15]['rmse_mean'],
                'rmse_change':lookup[flow,scale,method,.15]['rmse_mean']-lookup[flow,scale,method,0.]['rmse_mean']} for flow in flows]
        paired.append({'method':method,'scale':scale,'macro_psnr_change':float(np.mean(deltas)),
                       'macro_psnr_change_std':float(np.std(deltas)),
                       'lower_rmse_flows':sum(r['rmse_change']<0 for r in cases),'cases':cases})
    comparisons=[]
    for method,baseline,dropout in itertools.product(('fmt','fmt_objective_ntod_v2','fmt_v5'),('raw','unet','espcn','bicubic'),(0.,.15)):
        base_dropout=-1. if baseline=='bicubic' else dropout
        cases=[{'flow':flow,'scale':scale,
                'rmse_change':lookup[flow,scale,method,dropout]['rmse_mean']-lookup[flow,scale,baseline,base_dropout]['rmse_mean']}
               for flow,scale in itertools.product(flows,scales)]
        comparisons.append({'method':method,'baseline':baseline,'dropout':dropout,
                            'lower_rmse_cases':sum(r['rmse_change']<0 for r in cases),'cases':cases})
    training_count=sum(r['seeds'] for r in records if r['dropout']>=0)
    test_count=len({r['source_file'] for r in rows if r['flow']==flows[0] and int(r['scale'])==scales[0]})
    seed_count=len({r['seed'] for r in rows if r['seed']!='-1'})
    benchmark_text=('测试集为1.1已用benchmark，未参与本版本训练、归一化或模型选择，不能称为全新确认集。'
                    if summary['version']=='Other_FTLEUpsampling2D_1.2' else '工程开发时段，仅验证软件流程，不作正式性能结论。')
    lines=['# FTLE二维超分辨：dropout与8×','',
           f"实验：`{summary['version']}`；科学commit：`{summary['provenance']['commit']}`。",
           'FTLE（Finite-Time Lyapunov Exponent，有限时间李雅普诺夫指数）衡量有限时间内的粒子分离速率。',
           f'{len(flows)}流场×2/4/8倍×六学习方法×dropout 0/0.15×{seed_count}种子，共{training_count}次训练。',
           '全部主表指标来自完整留出的测试时间窗；训练、验证、测试不共享积分时间或源插值帧。',
           benchmark_text,
           '每个切片的所有方法、两个dropout概率和三个倍率使用相同有效高网格像素。',
           '掩码比1.1严格，故本表p0为重新训练的配对对照；不能直接与1.1表相减归因于dropout。',
           f'每种子等权平均{test_count}个测试切片，再报告{seed_count}个种子均值±总体标准差；宏平均在每种子内等权平均{len(flows)}流场。','',
           '## 流场宏平均峰值信噪比','',
           'PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）衡量相对于固定训练数据范围的重建误差，单位dB，越高越好。',
           'ESPCN（Efficient Sub-Pixel Convolutional Neural Network）使用卷积和像素重排上采样；U-Net是带跳跃连接的编码、解码卷积网络。',
           'dropout 0.15在训练时随机丢弃15%特征通道，测试关闭；没有对概率作测试集搜索。','',
           '| 方法 | dropout | 2× | 4× | 8× |','|---|---:|---:|---:|---:|']
    for method,dropout in variants:
        values=[]
        for scale in scales:
            r=macro_lookup[method,scale,dropout]
            values.append(f"{r['psnr_mean']:.3f}"+(f" ± {r['psnr_std']:.3f}" if dropout>=0 else ''))
        lines.append('| '+NAMES[method]+' | '+('—' if dropout<0 else str(dropout))+' | '+' | '.join(values)+' |')
    lines.extend(['','## dropout配对变化','','正值表示0.15相对于0的PSNR改善；每对使用相同数据、种子和GPU。',
                  '', '| 方法 | 2× ΔPSNR | 4× ΔPSNR | 8× ΔPSNR | RMSE降低的流场×倍率数 |',
                  '|---|---:|---:|---:|---:|'])
    for method in methods:
        group=[r for r in paired if r['method']==method]
        vals=[f"{r['macro_psnr_change']:+.3f} ± {r['macro_psnr_change_std']:.3f}" for r in group]
        lines.append('| '+NAMES[method]+' | '+' | '.join(vals)+f" | {sum(r['lower_rmse_flows'] for r in group)}/{len(flows)*len(scales)} |")
    for metric,title in [('rmse','RMSE（Root Mean Squared Error，均方根误差；FTLE物理单位，越低越好）'),
                         ('ssim','SSIM（Structural Similarity Index，结构相似性；越高越好）')]:
        for scale in scales:
            lines.extend(['',f'## {scale}× {title}','','| 方法 | dropout | '+' | '.join(flows)+' |',
                          '|---|---:|'+'---:|'*len(flows)])
            for method,dropout in variants:
                vals=[]
                for flow in flows:
                    r=lookup[flow,scale,method,dropout]
                    vals.append(f"{r[metric+'_mean']:.5f}"+(f" ± {r[metric+'_std']:.5f}" if dropout>=0 else ''))
                lines.append('| '+NAMES[method]+' | '+('—' if dropout<0 else str(dropout))+' | '+' | '.join(vals)+' |')
    lines.extend(['','## 时间隔离证据','','完整逐窗时刻及源帧索引见各流场`data/*/manifest.json`。',
                  '下表为各split完整积分窗覆盖的最小—最大物理时间；同一split内部窗口允许重叠。','',
                  '| 流场 | 训练时间覆盖 | 验证时间覆盖 | 测试时间覆盖 |', '|---|---|---|---|'])
    for flow in flows:
        m=read(f'data/{flow}/manifest.json')
        vals=[]
        for split in ('train','validation','test'):
            windows=m['temporal_certificate']['windows'][split]
            vals.append(f"{min(r['t0'] for r in windows):.6f}–{max(r['t1'] for r in windows):.6f}")
        lines.append('| '+flow+' | '+' | '.join(vals)+' |')
    lines.extend(['','## 评价区域覆盖','',
                  '三个倍率取固定几何有效区域的交集，会排除边界和障碍物附近的一部分有效高网格点。',
                  '指标描述以下共同区域；它们不代表被排除位置的精度。各列按测试切片顺序列出像素数。','',
                  '| 流场 | 共同评价像素 | 全部物理有效高网格像素 |', '|---|---|---|'])
    for flow in flows:
        m=read(f'data/{flow}/manifest.json')
        selected=[r for r in m['records'] if r['split']=='test' and r['scale']==max(scales)]
        lines.append('| '+flow+' | '+', '.join(str(r['evaluated_pixels']) for r in selected)+' | '+
                     ', '.join(str(r['valid_high']) for r in selected)+' |')
    lines.extend(['','## 审计证据','',
                  f"独立重算{audit['prediction_files_recomputed']}份预测、{audit['metric_rows']}行指标、{audit['summary_rows']}行汇总：PASS。",
                  '每次`selection.json`记录测试读取前的模型选择；`history.json`保存完整训练和验证曲线。',
                  '所有预测对应原始真值和统一掩码；没有checkpoint文件。',''])
    if (root/'scheduler_and_evidence_audit.json').exists():
        evidence=read('scheduler_and_evidence_audit.json')
        lines.extend(['训练设备及次数：'+json.dumps(evidence['training_devices'])+'。',
                      '同一流场/倍率/种子的全部方法和dropout对照共用一个GPU进程；不同组允许A100/V100，种子间波动也可能包含设备差异。',''])
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    values={'macro_psnr':macro,'dropout_paired_changes':paired,'encoder_comparisons':comparisons}
    (root/'macro_and_paired_comparisons.json').write_text(json.dumps(values,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'macro_psnr':macro,'dropout_paired_changes':paired},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('outputs/Other_FTLEUpsampling2D_1.2'))
    report(parser.parse_args().root)
