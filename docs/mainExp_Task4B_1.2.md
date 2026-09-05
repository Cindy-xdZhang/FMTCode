# mainExp_Task4B_1.2：Task4-b 四分类正式监督实验

## 状态与正式结论

已在 Ibex 完成正式三随机种子实验，并通过独立逐样本审计。作业
`51152830` 于 `2026-09-01T17:51:00+03:00` 在
`gpu101-16-l` 的 NVIDIA A100-SXM4-80GB 上启动，于
`17:52:56+03:00` 以 `COMPLETED, ExitCode=0:0` 结束；12/12 个正式
run、12 个 prediction 和 12 个 history 均完整，stderr 为 0 byte，持久化
checkpoint 数为 0。

正式主比较只支持以下有限结论：Raw+FMT 相对 Raw 的 test macro-F1
平均提高 `+0.02107 ± 0.03091`，但三个同 seed 差值为
`[-0.00841, +0.01838, +0.05323]`，只有 2/3 为正，故该平均增益对训练
随机种子不稳定。Raw+FMT 的 balanced accuracy 平均提高
`+0.03443 ± 0.04288`，同样只有 2/3 seed 为正；one-vs-rest macro
Average Precision 平均提高 `+0.03359 ± 0.02164`，三个 seed 均为正。

这个总体改善没有转化为 hairpin anatomy 改善。相对 Raw，Raw+FMT 的
ordinary-streamwise 与 ordinary-spanwise F1 分别提高
`+0.07016 ± 0.02064` 和 `+0.06977 ± 0.02766`，两个比较均为 3/3 seed
正增益；但 hairpin-head 与 hairpin-limb F1 分别变化
`-0.01117 ± 0.03263` 和 `-0.04449 ± 0.06272`，都只有 1/3 seed 为正。
按六个完整测试 `VortexId` 等权的 head/limb macro-F1 差为
`-0.00131 ± 0.05838`，三个 seed 差值为
`[-0.04571, -0.02303, +0.06482]`。因此，当前结果不支持“FMT 改善
hairpin-head/limb 四分类”的结论；平均总体增益主要来自两个 ordinary
方向类别。

容量控制也不支持更强的表示结论。Raw-wide 参数量为 123,652，高于
Raw+FMT 的 108,996；其 macro-F1 为 `0.41892 ± 0.01805`，高于
Raw+FMT 的 `0.40944 ± 0.02275`。Raw+FMT−Raw-wide 的同 seed 平均差为
`-0.00948 ± 0.03820`，三次为
`[-0.05340, +0.00895, +0.01602]`。所以不能排除主比较中的部分改善来自
模型容量或优化随机性。

`±` 均表示三个 optimizer seed 的样本标准差（`ddof=1`），不是新流场或
新 hairpin 的置信区间。`n=3` 时即便三次同号，双侧 exact sign test 的
最小 p 值仍为 0.25；本实验不声称统计显著。

## 冻结任务与方法

四类为：

1. `ordinary_streamwise`
2. `ordinary_spanwise`
3. `hairpin_head`
4. `hairpin_limb`

non-vortex 在四分类训练前排除。训练使用每个 batch 的 exact 四类等额
抽样、只由 train split 拟合的 Raw/FMT normalization，以及 validation
macro-F1 的内存 best-state 选择；test 在 best state 恢复后只评估一次。
正式 variants 为 Raw、参数更多的 Raw-wide、FMT-only、Raw+FMT，seeds 为
7068、7069、7070，最多 50 epochs，patience 为 8。模型不写 `.pt`、
`.pth` 或 `.ckpt`。

冻结 cache 共有 20,641 个样本，Raw/FMT feature shape 分别为
`(20641, 693)` 和 `(20641, 161)`。train/validation/test 的逐类 support
分别为 `[3000,3000,2136,5250]`、`[1000,1000,409,1516]` 和
`[1000,1000,284,1046]`。最小周期 x gap `0.21991149` 大于两倍 primitive
支持半径 `0.21444159`，空间拆分证书为 `certified=true`。

## 正式 test 结果

| Variant | Parameters | Macro-F1 | Balanced accuracy | Macro Average Precision |
|---|---:|---:|---:|---:|
| Raw | 90,308 | 0.38837 ± 0.02276 | 0.40049 ± 0.02040 | 0.40410 ± 0.00587 |
| Raw-wide | 123,652 | 0.41892 ± 0.01805 | 0.42640 ± 0.02523 | 0.40464 ± 0.01442 |
| FMT-only | 19,588 | 0.40214 ± 0.00239 | 0.41747 ± 0.00351 | 0.42102 ± 0.00219 |
| Raw+FMT | 108,996 | 0.40944 ± 0.02275 | 0.43492 ± 0.02560 | 0.43769 ± 0.02230 |

逐类 F1 均值：

| Variant | Ordinary streamwise | Ordinary spanwise | Hairpin head | Hairpin limb |
|---|---:|---:|---:|---:|
| Raw | 0.39318 | 0.43157 | 0.40329 | 0.32545 |
| Raw-wide | 0.39012 | 0.43180 | 0.42330 | 0.43046 |
| FMT-only | 0.50924 | 0.48820 | 0.23476 | 0.37636 |
| Raw+FMT | 0.46334 | 0.50134 | 0.39212 | 0.28096 |

## 六个完整测试 hairpin 的结构诊断

测试 `VortexId=[38,39,51,53,62,76]` 均作为完整实例保留。proxy 自身的
严格“2 streamwise + 1 spanwise dominant components”只有 3/6 成功，
所以该量只能作结构诊断，不能作监督模型的人工真值通过门槛。

Raw+FMT 三个 seed 的严格 2+1 成功数均为 2/6；Raw 为 `[2,1,1]/6`，
Raw-wide 为 `[3,1,1]/6`，FMT-only 为 `[0,0,0]/6`。Raw+FMT 的 mean soft
topology score 为 `0.26299`，高于 Raw 的 `0.18517`，但明显低于 proxy
自身的 `0.67226`。这不改变 hairpin 等权 F1 没有改善的主结论。

## 独立审计与可追溯证据

`experiments/Audit_Task4B_Supervised_1_2.py` 不导入训练 driver，从 12 个 prediction
NPZ 独立重算 confusion matrix、precision、recall、F1、balanced accuracy
与 Average Precision；同时重新执行 validation 的顺序 best-state / patience
规则，核验参数量、split indices、targets、probability simplex、六实例
拓扑、cache SHA、设备和无 checkpoint 约束。正式状态为 `PASS`，没有硬失败
或 warning。

- deployment base commit：`dcdbbba91b24ef99e0ac2fe25eb3f499241fd72a6`
- training bundle SHA-256：`78afca974a50b9393eeac685b5047403d90aeba23f2e81a8df415ab2c4c3aaf1`
- cache SHA-256：`4be1dcedf69c3c5b4e2a47fd26fc46b2567db5dda9f547448e0cf6407af0a190`
- proxy-label SHA-256：`f545c46427991842c05814d8df71bfa09b3ea11e33d3f240e18fccee0d8383e8`
- model/trainer/config SHA-256：`63a59e6f…e5ff` / `6041e462…3d7f` / `5dc2ded0…5d17`
- audit script SHA-256：`491e48a4…ffb0c`
- formal summary SHA-256：`27ab0035…37fb`
- per-run CSV SHA-256：`69049ceb…b07`
- independent audit SHA-256：`f13acda2…0ac7`
- stdout/stderr SHA-256：`cfbe38e6…22b1` / `e3b0c442…b855`
- evidence archive：`mainExp_Task4B_1.2_evidence.tar.gz`，33,207,834 bytes，
  127 entries，checkpoint 0，SHA-256 `86a031f6e6d8ef610436fb522da1f4b65ae5a320812c085f0da47de10f47d3fe`

主要证据路径：

- `outputs/mainExp_Task4B_1.2/channel_GTs/summary.json`
- `outputs/mainExp_Task4B_1.2/channel_GTs/per_run_metrics.csv`
- `outputs/mainExp_Task4B_1.2/channel_GTs/independent_audit.json`
- `outputs/mainExp_Task4B_1.2/channel_GTs/artifact_manifest.json`
- `outputs/mainExp_Task4B_1.2/ibex_job_51152830/scheduler_record.json`
- `outputs/mainExp_Task4B_1.2/mainExp_Task4B_1.2_evidence.tar.gz`

## 正式可视化

`experiments/Visualize_Task4B_Supervised_1_2.py` 只读取已通过审计的
`independent_audit.json`、冻结 cache 和 12 个 prediction，并在绘图前再次
核验其 SHA。量化主图展示四种 variant 的 macro-F1 mean ± 三 seed 样本
标准差、Raw+FMT−Raw / Raw+FMT−Raw-wide 的同 seed 差值及四类 F1。

三维 cube 图固定展示全部六个 test `VortexId`，不按性能选实例。主图使用
三 seed mean probability 的 argmax 共识；另分别导出 seeds 7068、7069、
7070 的完整六实例图，并为六个实例各导出 isometric 与沿 `+y` 的双视角
附图。proxy、Raw 和 Raw+FMT 使用相同 cube 中心、物理边界、颜色和相机，
没有平滑、形态学处理或结果筛选。

全部图由 Python/Matplotlib Agg 生成 SVG、PDF、600 dpi TIFF 和 PNG。
严格 source preflight 为 21 PASS / 0 WARN / 0 FAIL；11/11 panel alignment、
PDF 5 pt 字号审计和 collision audit 均 PASS，最小实际字号 5.5 pt，最终
collision 为 0 FAIL / 0 WARN。绘图脚本 SHA-256 为
`4f343354c9d5387a2511e1d7dccfb9f39552fe5f40344d7a08dbf90aa664d66e`。

- 量化主图：`outputs/mainExp_Task4B_1.2/channel_GTs/figures/task4b_supervised_1.2/task4b_supervised_quantitative.png`
- 六实例共识图：`outputs/mainExp_Task4B_1.2/channel_GTs/figures/task4b_supervised_1.2/task4b_cube_segmentation_all_test_ids_consensus.png`
- QA：`outputs/mainExp_Task4B_1.2/channel_GTs/figures/task4b_supervised_1.2/figure_qa_notes.md`
- 源数据：`task4b_quantitative_source_data.csv`（66 rows）与 `task4b_cube_source_data.csv`（1,330 rows）

## 证据边界

标签是 velocity–curl 与 channel-profile vorticity-deviation 构造的确定性
proxy，不是人工 head/limb anatomy ground truth。test 是固定分层抽样并使用
空间缓冲 slab，不是 whole-field prevalence estimate。本实验只有一个 steady
channel volume、一个固定 spatial holdout 和三个 optimizer seeds，不能证明
跨时间、跨 Reynolds number、跨流场或人工解剖标签的泛化。
