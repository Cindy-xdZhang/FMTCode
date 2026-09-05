# mainExp_Task4B_PooledInstanceSplit_3.1：channel+TBL 合并、按 hairpin 实例留出的四分类

## 状态

预注册于 2026-09-03，训练前冻结；同日在 Ibex 完成（jobs 51277193 / 51277194），独立审计 `PASS`。
拆分、缓冲与训练配方的全部决策均在训练前写入本文与 config，结果见下文。

## 研究问题

把 channel 与 TBL 两个 volume 的 vortex cube 合并，按 **完整 hairpin 实例** 留出约 20%
作为测试，用其余实例训练同一个四分类网络（ordinary streamwise、ordinary spanwise、
hairpin head、hairpin limb）。核心比较仍是 Raw+FMT 相对 Raw 与参数更多的 Raw-wide。
与 2.3（channel 训练→TBL 测试）不同，这里测试实例与训练实例来自同样的两个 volume，
回答的是"同分布下未见 hairpin 实例"的问题，不是跨流场迁移。

## 冻结协议

### 数据来源

- channel：`Verify_Task4B_ProxyLabels_1.2` 的全部 388,750 个候选 cube（proxy 标签、
  73 个 `VortexId`、16,711 个 hairpin cube），流线用 1.2 的冻结配方重新积分
  （7 线、双向 16 步 unit-velocity RK4、`spatial_step=0.5×mean voxel=0.006639`、
  `offset=0.25×min voxel=0.000997`、x/y 周期）。
- TBL：`mainExp_Task4B_ChannelToTBL_2.3` 的冻结 target cache 全部 385,231 个有效行
  （57 个有有效流线的 `VortexId`、32,152 个 hairpin cube；`spatial_step=0.11940`、
  `offset=0.017928`），逐 chunk 核对 SHA-256。
- 表示：与 2.3 相同的每 volume 无量纲坐标 `(primitive − 首点)/spatial_step`，FMT
  （6 频、Gram、chirality、sorted neighbor，161 维）在无量纲坐标上重算；
  normalization 只在训练行上拟合。

### 拆分规则（训练前冻结）

1. **实例拆分**：每个 volume 内，按 cube 数降序排列实例，每 10 个为一块，块内随机
   置换固定模式 `2 test / 1 validation / 7 train`（seed 7068，TBL 用 7069）。
   validation 从 80% 的拟合份额中划出，只用于选 epoch，不进入测试。
2. **ordinary cube 归属**：继承同 volume 内最近 hairpin cube 所属实例的拆分
   （按实例的 Voronoi 划分；channel 的 x/y 按周期处理）。
3. **缓冲**：删除所有与任意测试 cube 种子距离小于 **一个 primitive 支持半径**
   `R = 16×spatial_step + offset` 的 train/validation cube；测试 cube 一律保留。
   这与否决 Task4-b 1.1 时使用的泄漏判据相同（训练 primitive 不得触及测试种子）。
   预注册前在真实标签上量化过：2R 会删掉 channel 77%、TBL 64% 的训练 hairpin cube，
   因此不采用；1R 删掉约 39%/32%，并会整体删除若干靠近测试实例的训练实例。
4. **采样**：hairpin cube 全部保留；ordinary 两类每 volume 每 split 上限
   train 8000 / validation 2000 / test 4000（固定 RNG）。

### 训练与评估

训练配方与 `mainExp_Task4B_1.2` 完全相同：`PathlineMulticlassClassifier3D`
（temporal 64 / embedding 128 / auxiliary 64、dropout .1）、exact 四类等额 batch 256、
AdamW lr 3e-4、weight decay 1e-4、梯度裁剪 1.0、最多 50 epoch、patience 8、以
validation macro-F1 选内存 best state、test 只评估一次；seeds 7068–7070；不写 checkpoint。

报告：pooled 与逐 volume 的 macro-F1、balanced accuracy、one-vs-rest macro AP、逐类 F1、
confusion matrix；hairpin→ordinary 错误率、hairpin membership F1、已知 hairpin mask 下的
head/limb macro-F1、按 `VortexId` 等权的 head/limb macro-F1；同 seed 配对差
Raw+FMT−Raw 与 Raw+FMT−Raw-wide。

### 独立审计

`experiments/Audit_Task4B_PooledInstanceSplit_3_1.py` 不导入训练器与拆分模块：用独立
KD-tree 重算两 volume 的缓冲证书、核对实例纯度与声明的实例分配、从 prediction NPZ 用
sklearn 重算全部指标并与 summary 比对（容差 1e-9）、检查参数量顺序
raw_wide > raw_fmt > raw、检查无 checkpoint。

## 代码与配置

- `config/mainExp_Task4B_PooledInstanceSplit_3.1.yaml`
- `FMT_Utils/Task4B_PooledSplit_3D.py`（拆分、Voronoi 归属、缓冲）
- `experiments/Build_Task4B_PooledInstanceSplit_3_1.py`
- `experiments/Train_Task4B_PooledInstanceSplit_3_1.py`（复用 1.2 训练循环）
- `experiments/Audit_Task4B_PooledInstanceSplit_3_1.py`
- `tests/test_task4b_pooled_split_3d.py`
- `ibex_bash/mainexp_task4b_pooled_3p1_{build_cpu,train_gpu}.sh`

## 结果（Ibex jobs 51277193 / 51277194，2026-09-03，独立审计 PASS）

cache：91,711 行；hairpin head/limb 行 train `4,400/18,427`、validation `721/2,559`、
test `1,806/7,798`；ordinary 每类 train 16,000、validation 4,000、test 8,000。缓存实例数
（缓冲后仍有 cube 的实例）channel train/validation/test `45/6/14`、TBL `37/6/12`；缓冲证书
channel 最小非测试→测试距离 `.107233 ≥ .107221`，TBL `1.929063 ≥ 1.928395`。设备
Tesla V100-SXM2-32GB，12 个 run 共 11 分钟；训练配方与 1.2 相同。

| Variant | 参数量 | test macro-F1 | balanced accuracy | macro AP | F1 ord-streamwise / ord-spanwise / head / limb |
|---|---:|---:|---:|---:|---|
| Raw | 90,308 | .3841 ± .0064 | .3961 | .3797 | .3625 / .4306 / .3094 / .4338 |
| Raw-wide | 123,652 | .3687 ± .0094 | .3854 | .3656 | .4006 / .4202 / .3020 / .3519 |
| FMT-only | 19,588 | .3979 ± .0044 | .4211 | .4071 | .4644 / .4277 / .2397 / .4600 |
| Raw+FMT | 108,996 | **.4196 ± .0039** | **.4422** | **.4178** | .4590 / .4606 / .3420 / .4170 |

同 seed 配对差（正值表示 Raw+FMT 更好）：

| 指标 | Raw+FMT − Raw | 正 seed | Raw+FMT − Raw-wide | 正 seed |
|---|---:|---:|---:|---:|
| pooled macro-F1 | +.0356 ± .0094（+.0252/+.0433/+.0382） | 3/3 | +.0510 ± .0055 | 3/3 |
| balanced accuracy | +.0461 ± .0106 | 3/3 | +.0568 ± .0071 | 3/3 |
| macro AP | +.0381 ± .0051 | 3/3 | +.0522 ± .0088 | 3/3 |
| channel macro-F1 | +.0250 ± .0224 | 2/3 | +.0501 ± .0158 | 3/3 |
| TBL macro-F1 | +.0523 ± .0089 | 3/3 | +.0629 ± .0098 | 3/3 |
| known-mask head/limb macro-F1 | +.0177 ± .0123 | 3/3 | +.0294 ± .0223 | 3/3 |
| VortexId 等权 head/limb macro-F1 | −.0024 ± .0185 | 2/3 | +.0761 ± .0401 | 3/3 |
| hairpin→ordinary 错误率（负为好） | +.0224 ± .0485 | — | −.1014 ± .0370 | — |

hairpin→ordinary 错误率：Raw `.399`、Raw-wide `.523`、FMT-only `.322`、Raw+FMT `.422`。

**结论**：在合并 channel+TBL、留出完整 hairpin 实例的同分布测试上，Raw+FMT 的 macro-F1
稳定高于 Raw（3/3 seed）和参数更多的 Raw-wide（3/3 seed），TBL 部分增益大于 channel。
但 absolute macro-F1 只有 `.42`，约 42% 的 hairpin cube 仍被判为 ordinary，且这一项没有因
加入 FMT 而改善（只有 FMT-only 降到 .32，代价是 head F1 最低）。因此本实验支持“FMT 是有用的
辅助表示”，不支持“已解决 hairpin 四分类”。`±` 为 3 个 optimizer seed 的样本标准差，
`n=3` 不构成统计显著性声明。

证据：`outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/{summary.json,independent_audit.json,
per_run_metrics.csv,predictions/,cache/}`；cache SHA `40f21e35…d8a3`，summary SHA `64547021…9aeb`，
audit SHA `d1113b88…65c5`；本地用同一审计器对下载副本重跑亦 `PASS`。

## 证据边界

- 标签仍是确定性物理 proxy，不是人工 head/limb 解剖标注。
- 只有一个 channel volume 与一个 TBL volume；实例留出发生在这两个 volume 内部。
- 缓冲会删除靠近测试实例的训练 cube，训练实例的有效数量少于名义 80%。
