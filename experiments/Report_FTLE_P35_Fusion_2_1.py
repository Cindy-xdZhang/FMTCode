"""Generate traceable development and final tables without selecting on test."""
import argparse
import json
from pathlib import Path
import numpy as np

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def development(root):
    choice=read(root/'selection.json');audit=read(root/'result_audit.json')
    assert audit['status']=='PASS'
    names=['espcn','unet']+[r['id'] for r in choice['candidates']]
    text=['# FTLE P35 2.1：开发集结果','',
          '仅验证集，用于统一选取一个跨流场和倍率的方法。PSNR为峰值信噪比，数值越大表示平方误差越小。',
          '', '| 候选 | 4×平均PSNR (dB) | 8×平均PSNR (dB) | 可成为主方法 |', '|---|---:|---:|---|']
    for name in names:
        values=[np.mean([r['psnr'] for r in choice['rows'] if r['candidate']==name and r['scale']==s]) for s in (4,8)]
        eligible=any(r['id']==name and r['eligible'] for r in choice['candidates'])
        text.append(f'| {name} | {values[0]:.4f} | {values[1]:.4f} | {"是" if eligible else "否（对照）"} |')
    text+=['',f'固定选择：`{choice["selected"]["id"]}`。预注册开发门槛：{"通过" if choice["gate_passed"] else "未通过"}。',
           '',f'全部{audit["trainings"]}次拟合、{audit["predictions_recomputed"]}份预测独立复算通过；本阶段未打开test文件。',
           '', '| 流场 | 倍率 | ESPCN | U-Net | 同结构无几何 | 同结构Raw | 选定P35 |', '|---|---:|---:|---:|---:|---:|---:|']
    selected=choice['selected'];arch=selected['architecture']
    for flow in list(dict.fromkeys(r['flow'] for r in choice['rows'])):
        for scale in (4,8):
            values=[next(r['psnr'] for r in choice['rows'] if r['flow']==flow and r['scale']==scale and r['candidate']==name)
                    for name in ('espcn','unet',arch+'_none',arch+'_raw',selected['id'])]
            text.append(f'| {flow} | {scale}× | '+' | '.join(f'{x:.4f}' for x in values)+' |')
    text+=['','科学代码：`'+choice['provenance']['commit']+'`。',
           'CPU开发比较；每个流场/倍率内所有方法在同一进程、同一设备上顺序运行。仅报告实际完成结果，不与历史GPU耗时直接比较。']
    (root/'development_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def final_report(root):
    final=root/'final';audit=read(final/'audit.json');summary=read(final/'summary.json');lock=read(final/'lock.json')
    assert audit['status']=='PASS'
    selected=summary['selected']['id'];arch=summary['selected']['architecture']
    names=[r['id'] for r in lock['definitions']]
    display={'espcn':'ESPCN','unet':'U-Net',arch+'_none':'同结构无几何',arch+'_raw':'同结构Raw',
             arch+'_signed':'同结构保线身份傅里叶',arch+'_p35':'P35',selected:'选定P35融合'}
    text=['# FTLE P35 2.1：固定方案的测试结果','',
          f'统一方法为`{selected}`，由开发集选择并锁定。三种子为`{lock["seeds"]}`。',
          'P35先对五线簇实际作傅里叶变换，再连接中心特征与四邻居逐特征均值、最大值。低FTLE只保留末端最大拉伸率，线簇表示则提供更多变形信息；傅里叶变换本身不创造新的观测。',
          '选定连接方式：低FTLE与103维p35分别映射到48通道 → 局部残差卷积与有效区域全局均值/最大值 → 不同膨胀率卷积 → 像素重排 → 加双三次插值并保持已知格点。大部分卷积在低网格执行。',
          '评价使用训练和本轮候选选择未读过的时间块，完整积分窗与源帧支撑跨集合隔离；该test是1.1/1.2已使用的历史benchmark，不称全新的开发确认集。',
          'ESPCN为Efficient Sub-Pixel Convolutional Neural Network（高效亚像素卷积神经网络）；U-Net为带跳接的编码器—解码器网络。PSNR为峰值信噪比，越大表示相对于固定训练值域的平方误差越小。',
          '', '| 方法 | 4× PSNR (dB) | 8× PSNR (dB) | 参数量 |', '|---|---:|---:|---:|']
    def get(flow,scale,name):
        return next(r for r in summary['summary'] if r['flow']==flow and r['scale']==scale and r['method']==name and r['split']=='test')
    for name in names:
        a,b=get('macro',4,name),get('macro',8,name)
        count=str(a['parameters'])+' / '+str(b['parameters'])
        text.append(f'| {display[name]} | {a["psnr_mean"]:.4f} ± {a["psnr_std"]:.4f} | {b["psnr_mean"]:.4f} ± {b["psnr_std"]:.4f} | {count} |')
    text+=['','每个种子先对各流场三张测试切片取均值，再对四流场等权平均；±为三个优化种子的样本标准差。参数列依次为4×/8×。',
           '', '| 倍率 | 对照 | 配对PSNR变化 (dB) | 配对MSE相对变化 |', '|---|---|---:|---:|']
    for comparison in summary['test_comparisons']:
        relative=np.mean([r['mse_relative_change'] for r in comparison['paired']])
        text.append(f'| {comparison["scale"]}× | {display[comparison["baseline"]]} | {comparison["mean_psnr_gain"]:+.4f} | {relative:+.2%} |')
    text+=['','MSE为均方误差；相对变化先在相同流场/倍率/种子内计算，再等权平均，负值表示误差下降。',
           '', '| 流场 | 倍率 | ESPCN | U-Net | 同结构无几何 | 同结构Raw | 选定P35 |', '|---|---:|---:|---:|---:|---:|---:|']
    flows=list(dict.fromkeys(r['flow'] for r in summary['rows']))
    for flow in flows:
        for scale in (4,8):
            values=[get(flow,scale,name) for name in ('espcn','unet',arch+'_none',arch+'_raw',selected)]
            text.append(f'| {flow} | {scale}× | '+' | '.join(f'{r["psnr_mean"]:.3f} ± {r["psnr_std"]:.3f}' for r in values)+' |')
    text+=['','## 输入和归因边界','',
           '每格点严格使用中心、±x、±y五条路径线，不含z邻居。原低分辨率FTLE已经由同一四邻居终点计算，因此几何输入的差异在于保留方向、相位与演化信息，不能称相对当前基线额外积分四线。',
           '新网络与冻结ESPCN/U-Net的结构及训练方式不同；同结构无几何和Raw对照用于分开检查这些变化。新输入臂补零到共同宽度以固定参数形状，实际使用的特征列数不同。',
           '全部新网络使用双三次残差预测并保持已知低网格观测；step0可被验证集选中。在这种情况下，该运行没有显示几何编码的额外收益。',
           '', '## 复现与成本','',
           f'最终科学commit：`{audit["provenance"]["commit"]}`。审计复算{audit["trainings"]}次训练、{audit["predictions_recomputed"]}份预测，全部PASS。没有模型文件。',
           '精确分割、预测、完整指标、每次最佳轮次/步骤、设备和配置哈希保留在运行记录中。实际设备上的训练成本及包含标准化/传输的验证推断时间见per_run.csv；特征编码与文件读取时间按进程记录，不计为网络纯推断时间。开发在CPU进行，最终评测在可用GPU进行；每个流场、倍率和种子内所有方法共用同一设备。']
    text+=['', '| 方法 | 4×训练/推断 (s / ms每帧) | 8×训练/推断 (s / ms每帧) |', '|---|---:|---:|']
    for name in names:
        values=[get('macro',scale,name) for scale in (4,8)]
        text.append('| '+display[name]+' | '+' | '.join(f'{r["training_seconds_mean"]:.2f} / {1000*r["inference_seconds_per_slice_mean"]:.2f}' for r in values)+' |')
    devices={}
    for file in (final/'runs').glob('*/*/result.json'):
        device=read(file)['provenance']['device'];devices[device]=devices.get(device,0)+1
    text+=['','实际GPU与训练次数：'+', '.join(f'{name}: {count}' for name,count in sorted(devices.items()))+'。',
           '本表描述本次实际设备上的执行成本，不能与历史不同设备的耗时直接比较。几何预计算不包含在上述网络推断计时中。最终执行为保证对照一致，一次编码全部几何表示；该编码总耗时不能当作部署时仅计算p35的必要成本。']
    text+=['','## 图与原始指标','',
           '[配对增益图](figures/paired_gains.pdf)；[逐次指标](per_run.csv)；[均值与标准差](summary.csv)。',
           '', '| 流场 | 4×首个测试切片 | 8×首个测试切片 |', '|---|---|---|']
    for flow in flows:
        text.append(f'| {flow} | [图](figures/{flow}_x4.pdf) | [图](figures/{flow}_x8.pdf) |')
    text+=['','每幅图统一使用seed98211和首个测试切片，同时展示参考、评价区域、五方法预测及绝对误差。所有色标在同图方法间共享，未裁去误差异常值。']
    (final/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/Other_FTLEP35Fusion_2.1'))
    parser.add_argument('--final',action='store_true');args=parser.parse_args()
    development(args.root)
    if args.final:final_report(args.root)
