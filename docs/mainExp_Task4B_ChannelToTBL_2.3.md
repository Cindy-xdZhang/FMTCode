# mainExp_Task4B_ChannelToTBL_2.3：channel 训练到 TBL 测试的四分类迁移

## 状态与结论

`mainExp_Task4B_ChannelToTBL_2.3` 是本地完成并通过独立审计的正式协议版本。
本文的 Fourier Motion Transformer（FMT）是无可训练参数的 pathline 频域几何
encoder；“proxy 标签”指由冻结物理规则生成、用来替代缺失人工四类标注的监督标签。
四种方法、三个配对随机种子共 12/12 个 run 完整；每个 run 都先在 channel 的
全部 20,641 行上达到连续三个 epoch 零分类错误且最小真实类 logit margin 为正，
然后才对同一批冻结 TBL 测试行执行一次推理。独立审计状态为 `PASS`，没有持久化
模型 checkpoint。

主指标上，Raw+FMT 的 TBL macro-F1 为 `0.286830 ± 0.001761`。相对 Raw，
同 seed 差为 `[+0.015387, +0.016481, +0.014083]`，均值
`+0.015317 ± 0.001201`；相对参数更多的 Raw-wide，同 seed 差为
`[+0.016151, +0.016507, +0.016987]`，均值 `+0.016548 ± 0.000419`。
两个主比较均为 3/3 seed 正增益。因此，在这一对固定 volume 和当前 proxy 标签下，
结果支持 Raw+FMT 比 Raw 及 Raw-wide 有稳定但数值不大的总体优势；Raw-wide 有
5,089,156 个参数，高于 Raw+FMT 的 3,153,796 个参数，故这个比较不支持“增益只由
更大参数量造成”的解释。

但绝对性能仍然很差：Raw+FMT 的 hairpin-head / hairpin-limb F1 只有
`0.113340 / 0.144431`，coverage-adjusted hairpin-to-ordinary error 仍为
`0.490971 ± 0.014740`。这说明 channel 已被完全记忆后，跨到 TBL 时仍有约一半
人工 hairpin 体素被预测为 ordinary 类；当前结果不能称为已经解决 hairpin anatomy
四分类。

本文所有 `均值 ± 标准差` 都是 seeds `7068, 7069, 7070` 的样本标准差
（`ddof=1`），不是跨 flow、跨 volume 或跨 hairpin 的置信区间。三个 seed 即使全部
同号，也不足以单独支持统计显著性声明。

## 版本有效性

- `2.1` 是独立审计前的部分试跑，只产生 FMT-only 与 Raw+FMT 的 6/12 个 run，
  没有完整 Raw / Raw-wide 对照，也没有 `independent_audit.json`。这些性能数字无效，
  不得用于结论。
- `2.2` 增加了输入身份、coverage 和相对路径审计，但配置中的
  `source_offset_over_spatial_step=0.1501503378764808` 是近似值；channel cache
  记录的精确值为 `0.1501458034894056`，绝对差 `4.5343870752e-6`、相对差
  `3.0199892170e-5`。训练器在第一个 epoch 前拒绝该配置，2.2 没有训练结果。
- `2.3` 使用 channel cache 的精确 offset/step 比例，并完成 12/12 个正式 run；
  独立审计为 `PASS`。本文只报告 2.3。

## 冻结协议

### 训练集与测试集

训练只使用 channel 冻结 cache 的全部 20,641 行，包括旧 channel 空间拆分中的
全部行；逐类 support 为 `[5000, 5000, 2829, 7812]`，依次对应：

1. `ordinary_streamwise`
2. `ordinary_spanwise`
3. `hairpin_head`
4. `hairpin_limb`

这一步刻意不再拆 channel train/validation/test，因为当前问题首先要求排除“网络连
channel 都无法拟合”的实现错误。TBL 是唯一跨 volume 测试集，不参与参数训练、
normalization 拟合、epoch 选择或超参数选择。

TBL 原速度场为 `276 × 251 × 224` structured grid。人工标注文件提供 cell-level
`VortexIds`；voxelization 使用 `171 × 139 × 83` cube grid，共保留 58 个非空人工
VortexId、32,488 个人工 hairpin cubes。x/y/z 分别解释为 streamwise、spanwise、
wall-normal，三个方向都按 nonperiodic 边界处理。

### TBL proxy 标签与测试总体

TBL 四分类标签不是四类人工标签。人工 `VortexIds` 只定义 hairpin support 和
hairpin identity；head/limb 及 ordinary orientation 仍由确定性 physics proxy 构造：

- 在每个 volume 上计算 wall-normal plane vorticity deviation，percentile 固定为
  channel 训练期选择的 `80.5`；TBL 对应阈值为 `0.43332682788372046`。
- percentile 产生 384,702 个候选 cubes；随后把所有人工 hairpin support 纳入，得到
  391,953 个候选。人工覆盖前的 hairpin recall 为 micro `0.776810`、VortexId 等权
  `0.787322`、最差 VortexId `0.504188`。
- ordinary 区域按 velocity-curl 夹角是否不超过 45°分为 streamwise/spanwise。
  hairpin support 内，只有同时满足 velocity-curl 夹角大于 45°、`oyf>0`，且
  `|oyf|` 大于 `|omega_x|` 和 `|omega_z|` 的 cubes 被标为 head；其余标为 limb。
- 对全部候选作 nonperiodic streamline 有效性检查后保留 385,231 行，不做类别平衡
  抽样。逐类 support 为 `[107441, 245638, 4533, 27619]`，保存为 48 个 chunk。

因此，这个实验评估的是“已知 TBL 人工 hairpin support/identity 条件下，channel
四类 proxy 分类器能否迁移”，不是不借助人工 support 的 whole-field hairpin 检测。

### 流线与 scale-normalized 表示

每个 cube 以 unit-velocity、双向四阶 Runge-Kutta 积分，各方向 16 步，共 33 个
采样点。为消除 channel 与 TBL 物理坐标单位和网格尺度不一致，Raw primitive 在每个
volume 内定义为：

```text
(primitive - primitive_first_center_point) / spatial_step
```

channel 的 `spatial_step=0.006638998689603197`、邻线
`offset=0.0009968177926155829`；TBL 的
`spatial_step=0.11940419241924695`、`offset=0.017928038410791425`。两者严格共用
`offset/spatial_step=0.1501458034894056`。FMT 特征在这套无量纲坐标上重新计算；
encoder 固定为 6 frequencies、Gram mode、chirality、sorted neighbor pooling，输出
161 维，无可训练参数。

Raw 和 FMT 的标准化统计都只由 20,641 个 channel 行拟合，独立审计确认
`target_rows_used_for_fit=0` 且重算统计逐 bit 相同。应用 channel normalization 后，
TBL Raw/FMT 标量特征的标准差分别为 `1.262252 / 0.834083`，`|z|>5` 比例分别为
`0.006007 / 0.002763`；无量纲化减小了单位差异，但没有消除分布差异。

### 模型、训练门禁与 TBL 一次评估

四个 variant 为 Raw、Raw-wide、FMT-only 和 Raw+FMT；模型 family 为
`PathlineMulticlassClassifier3D`。训练使用 unweighted cross-entropy、Adam
（learning rate `0.001`）、batch size 512、每个 epoch 对所有 channel 行恰好一次
without-replacement permutation，最多 1,000 epochs；不使用 dropout、mixed
precision 或 gradient clipping。模型只保存在内存中。

| Variant | Parameters | seeds 7068/7069/7070 达到门禁的 epoch | 对应最小真实类 logit margin |
|---|---:|---:|---:|
| Raw | 1,937,284 | 510 / 486 / 530 | 1.469219 / 1.107507 / 1.894707 |
| Raw-wide | 5,089,156 | 546 / 523 / 489 | 1.829144 / 0.941559 / 0.411019 |
| FMT-only | 1,223,684 | 25 / 25 / 24 | 3.170871 / 2.987366 / 3.141751 |
| Raw+FMT | 3,153,796 | 30 / 29 / 29 | 3.601820 / 2.981755 / 1.256043 |

12 个 run 在选定 epoch 的 channel accuracy、macro-F1 和 balanced accuracy 都为
`1.0`，error count 都为 `0`，并且 terminal zero-error streak 都为 `3`。因此 TBL
性能差不能归因于 source training underfit。每个 run 的
`target_evaluation_count=1`；审计还确认 12 次推理使用完全相同且顺序一致的
385,231 行，ordered-row SHA-256 为
`4550c328b54d0d383f77c71586cd0bd5f66c82c853326e525e7c6644a667d656`。
这里的“一次”特指每个通过 source 门禁的网络只对冻结 TBL cache 计算一次预测；
此前的确定性 voxelization、proxy-label 构建和 curl 数值审计不是模型评估，也没有
用于选择网络或 epoch。

## TBL 四分类结果

### 总体指标

One-vs-rest macro Average Precision 是把每一类依次作为正类计算 Average
Precision 后取四类平均。Balanced accuracy 是四类 recall 的算术平均；
hairpin-membership F1 则把 head/limb 合为正类、两个 ordinary 类合为负类。

| Variant | Macro-F1 | Balanced accuracy | Macro Average Precision | Hairpin-membership F1 |
|---|---:|---:|---:|---:|
| Raw | 0.271513 ± 0.002193 | 0.314264 ± 0.002181 | 0.287299 ± 0.001745 | 0.156423 ± 0.001167 |
| Raw-wide | 0.270282 ± 0.001434 | 0.314516 ± 0.003573 | 0.288086 ± 0.001683 | 0.157541 ± 0.002115 |
| FMT-only | 0.253944 ± 0.001351 | 0.328276 ± 0.001046 | 0.303805 ± 0.001034 | 0.167883 ± 0.000519 |
| Raw+FMT | **0.286830 ± 0.001761** | **0.373191 ± 0.004729** | **0.320573 ± 0.001444** | **0.172379 ± 0.001328** |

### 每类 F1

| Variant | Ordinary streamwise | Ordinary spanwise | Hairpin head | Hairpin limb |
|---|---:|---:|---:|---:|
| Raw | 0.347827 ± 0.009310 | **0.526906 ± 0.012763** | 0.072401 ± 0.003299 | 0.138918 ± 0.001794 |
| Raw-wide | 0.354099 ± 0.008868 | 0.518483 ± 0.009160 | 0.067176 ± 0.003129 | 0.141371 ± 0.003962 |
| FMT-only | 0.338779 ± 0.003504 | 0.472822 ± 0.004455 | 0.058902 ± 0.001194 | **0.145272 ± 0.001118** |
| Raw+FMT | **0.372935 ± 0.004947** | 0.516615 ± 0.005146 | **0.113340 ± 0.002248** | 0.144431 ± 0.001365 |

Raw+FMT 相对 Raw 的逐类配对差依次为
`+0.025108 ± 0.008710`（3/3 seed 正）、
`-0.010291 ± 0.016994`（1/3 正）、
`+0.040939 ± 0.005420`（3/3 正）和
`+0.005513 ± 0.002882`（3/3 正）。相对 Raw-wide，对应为
`+0.018836 ± 0.007922`（3/3）、
`-0.001868 ± 0.014257`（1/3）、
`+0.046164 ± 0.005142`（3/3）和
`+0.003060 ± 0.004685`（2/3）。所以总体 macro-F1 增益主要来自 ordinary
streamwise 和 hairpin-head；ordinary spanwise 没有改善。

### 核心配对比较

下表中 F1、balanced accuracy 和 Average Precision 的正值定义为
`Raw+FMT - baseline`；error rate 的正值定义为 `baseline - Raw+FMT`，因此所有列
都是“正值表示 Raw+FMT 较好”。

| 指标 | Raw+FMT vs Raw | 正 seed | Raw+FMT vs Raw-wide | 正 seed |
|---|---:|---:|---:|---:|
| Macro-F1 | +0.015317 ± 0.001201 | 3/3 | +0.016548 ± 0.000419 | 3/3 |
| Balanced accuracy | +0.058926 ± 0.006845 | 3/3 | +0.058675 ± 0.008194 | 3/3 |
| Macro Average Precision | +0.033275 ± 0.003177 | 3/3 | +0.032487 ± 0.003101 | 3/3 |
| Hairpin-membership F1 | +0.015956 ± 0.002258 | 3/3 | +0.014838 ± 0.003428 | 3/3 |
| Valid hairpin→ordinary error reduction | +0.161048 ± 0.013275 | 3/3 | +0.151313 ± 0.014127 | 3/3 |
| Coverage-adjusted hairpin→ordinary/invalid error reduction | +0.159382 ± 0.013138 | 3/3 | +0.149748 ± 0.013981 | 3/3 |
| VortexId-equal coverage-adjusted error reduction | +0.169175 ± 0.014427 | 3/3 | +0.158317 ± 0.014165 | 3/3 |
| Known-mask head/limb macro-F1 gain | +0.043139 ± 0.010054 | 3/3 | +0.053243 ± 0.012640 | 3/3 |
| Valid VortexId-equal four-class head/limb macro-F1 gain | +0.110002 ± 0.006188 | 3/3 | +0.106392 ± 0.010166 | 3/3 |

## Streamline coverage 与 hairpin 诊断

全部 391,953 个候选中 385,231 个具有有效 nonperiodic streamlines，整体 coverage
为 `0.982850`。人工 hairpin 的 32,488 个 cubes 中保留 32,152 个，hairpin
coverage 为 `0.989658`。58 个人工 VortexId 中 57 个至少有一条有效 streamline；
VortexId 12 的 133/133 行全部无效，VortexId 62 只保留 153/354 行。

为避免静默忽略这些失败，coverage-adjusted 指标把无效 hairpin 行直接计为错误，并在
VortexId-equal 指标中把完全无有效行的 VortexId 12 记为 error rate 1.0。

| Variant | Valid row hairpin→ordinary error | Coverage-adjusted row error | Valid VortexId-equal error | Coverage-adjusted VortexId-equal error |
|---|---:|---:|---:|---:|
| Raw | 0.646699 ± 0.009704 | 0.650353 ± 0.009603 | 0.643221 ± 0.012729 | 0.654040 ± 0.012612 |
| Raw-wide | 0.636964 ± 0.008509 | 0.640719 ± 0.008421 | 0.632959 ± 0.010103 | 0.643182 ± 0.009850 |
| FMT-only | **0.435432 ± 0.006872** | **0.441271 ± 0.006801** | **0.421657 ± 0.007369** | **0.437662 ± 0.006950** |
| Raw+FMT | 0.485651 ± 0.014894 | 0.490971 ± 0.014740 | 0.470414 ± 0.014867 | 0.484865 ± 0.014354 |

FMT-only 在“是否仍预测为 hairpin”这一项最好，但它的总体 macro-F1 最低。这是一个
真实 trade-off，不应只选择有利指标。若人工 hairpin mask 已知，并强制只在
head/limb 两个概率间选择，Raw+FMT 的 head/limb macro-F1 为
`0.654735 ± 0.006242`；但保留完整四类预测并按 57 个有效 VortexId 等权后只有
`0.414163 ± 0.009091`。两者的差距再次表明主要瓶颈仍是 hairpin 被压到 ordinary
类别，而不只是 head 与 limb 之间互相混淆。

| Variant | Known hairpin mask 后的 head/limb macro-F1 | Valid VortexId-equal 四类 head/limb macro-F1 |
|---|---:|---:|
| Raw | 0.611596 ± 0.004881 | 0.304160 ± 0.004899 |
| Raw-wide | 0.601492 ± 0.008510 | 0.307770 ± 0.006625 |
| FMT-only | 0.586914 ± 0.006747 | 0.390570 ± 0.001337 |
| Raw+FMT | **0.654735 ± 0.006242** | **0.414163 ± 0.009091** |

## Curl 数值审计

`tbl.vtk` 没有完整三分量 vorticity，因此 proxy orientation 所需的三分量 curl 由
速度场作二阶有限差分得到；网络输入仍是速度积分得到的 pathline 及其 FMT 特征。
`tbl_GTs.vtk` 中通过 `PointIds` 对应的 stored vorticity 只用于数值审计，绝不进入
标签或网络特征。审计覆盖 224,677 个 point records、217,574 个 unique PointIds；
7,103 个重复记录也保留核验。

| 审计量 | x | y | z | 门槛/结果 |
|---|---:|---:|---:|---:|
| finite-difference vs stored vorticity correlation | 0.995649 | 0.997400 | 0.982725 | 每分量 ≥ 0.97，PASS |
| vorticity sign agreement | 0.977991 | 0.984159 | 0.964887 | 每分量 ≥ 0.95，PASS |
| vector relative root mean squared error | — | — | — | 0.114691 ≤ 0.15，PASS |
| derived vs stored `oyf` correlation | — | — | — | 0.971980 ≥ 0.90，PASS |
| PointIds coordinate max absolute error | — | — | — | 0.0 |
| stored `oyf` max absolute error | — | — | — | 0.0 |

## 输入身份与独立审计

| 输入/冻结产物 | SHA-256 |
|---|---|
| `outputs/Other_Task4B_FMTFourClassClustering_1.2/channel_GTs/cache/task4b_four_class_cache.npz` | `4be1dcedf69c3c5b4e2a47fd26fc46b2567db5dda9f547448e0cf6407af0a190` |
| `C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D/tbl_flow/tbl.vtk` | `cc82db1d3ca4e1964ec19dcd30df5a82c3c5635c16b8e0182f01fa57bf0d65bb` |
| `C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D/tbl_flow/tbl_GTs.vtk` | `4fcd5efcfe13e10765095afec377b104d72c7092a646d39531f9bfbfc05c419e` |
| `config/mainExp_Task4B_ChannelToTBL_2.3.yaml` | `955c238ebdd265290fe462434ae7f31aca3c880495fcdbfa3e7fe494536b0a3f` |
| `target_cache/target_cache_manifest.json` | `d2551336564220bbb22b46f646ee282efdb71b12685e774a62fa6778e14287a7` |
| `normalization_source_channel_only.npz` | `7b14e01f9941e718d313ded256112a962d2da689782344f8d90d178b394e09a7` |
| ordered TBL test rows | `4550c328b54d0d383f77c71586cd0bd5f66c82c853326e525e7c6644a667d656` |

独立审计器不依赖 summary 中的汇总数字，重新读取 12 个 prediction，重算 source / TBL
confusion matrix、per-class F1、macro-F1、balanced accuracy、Average Precision、
hairpin diagnostics、VortexId-equal 与 coverage-adjusted 指标，并重建四方法均值、样本
标准差和同 seed 差。它还核验：

- 12 run / 12 prediction / 12 history 完整且没有意外文件；
- 每个 source epoch 都是完整且不重复的 20,641 行 permutation；
- source error 均为 0、连续三 epoch 门禁成立；
- 概率 simplex、source/target targets 和 12 次 target row 顺序一致；
- normalization 只由 channel 重算，TBL fit row 为 0；
- curl gate 通过，持久化 checkpoint 数为 0。

正式 audit 为 `PASS`，生成时间 `2026-09-02T17:01:15.077899+03:00`。

| 证据产物 | SHA-256 |
|---|---|
| `summary.json` | `fe5eb248ba99197373e6887364af20e1a451f0650011946afac9200b7308670f` |
| `independent_audit.json` | `40ea4105c0d71a6281f6cb969bcd0c195d54d053ff91e1df2c46f1bd482b8a67` |
| `target_preparation_summary.json` | `0461acd6b49f982c1ed6c4a90960492c8143c58642856ec38efb75ebd5585165` |
| `per_run_metrics.csv` | `fb2086ac4a1dcf43f801bcbe6fbb73fc7949bc89b14fa45a3e755d8ad3f8e1be` |
| `target_distribution_shift.json` | `c6e08446c401ae4728d0a151956c5b89e2b8eb407f1ef3b51e6e0fd069ac26a4` |
| audit script | `429ab55671023b6e6efa90e236efe0b16f8d5918cf19cde337eb0e1e1f71b9ea` |

主要证据路径：

- `outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs/summary.json`
- `outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs/independent_audit.json`
- `outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs/target_preparation_summary.json`
- `outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs/per_run_metrics.csv`
- `outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs/target_cache/target_cache_manifest.json`
- `config/mainExp_Task4B_ChannelToTBL_2.3.yaml`

## 证据边界

1. TBL 的人工标签只标注 hairpin support 和 VortexId identity，没有人工
   ordinary-streamwise、ordinary-spanwise、hairpin-head 或 hairpin-limb 四类真值；
   因而四分类 F1 衡量的是网络与 velocity-curl / vorticity-deviation proxy 的一致程度，
   不是与人工 anatomy 标签的一致程度。
2. 人工 hairpin support 被用于补入 TBL candidate 并定义 hairpin/ordinary 身份，所以
   该实验不能证明不依赖人工 support 的自动 whole-field hairpin detection。
3. 证据只有一个 channel 训练 volume、一个 TBL 测试 volume 和三个 optimizer seeds。
   它不能外推到其他 Reynolds number、其他 TBL volume、其他 flow family 或一般 3D
   流场。
4. 当前运行设备是本地 NVIDIA GeForce RTX 3090，PyTorch `2.6.0+cu126`；它不是
   Ibex 正式复现。只有在 Ibex 上按相同输入 SHA、config、12-run 门禁和独立审计完整
   重跑后，才能形成 Ibex 正式监督结论。
5. 2.3 使用 per-volume spatial-step 无量纲坐标和 per-volume p80.5 rank preprocessing，
   与早先 channel 内 1.2 的物理范围、测试总体及预处理均不同，不能直接横向比较其
   绝对 F1。
