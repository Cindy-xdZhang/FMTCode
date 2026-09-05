# mainExp_Task2_3D_2.3

## 论文问题

Task2 的主比较固定为：

- `Raw pathline → VAE → latent → KMeans`
- `FMT(pathline) → 同一个 VAE → latent → KMeans`

本实验只改变 VAE 的输入表示。它不要求 Raw 使用自己单独调出的最强 VAE，也不检验
“VAE 是否提高 FMT direct feature”。

## same-VAE 控制协议

- 每个 physical-family group 冻结一个 VAE；组内 Raw/FMT 两臂使用完全相同的 hidden
  layers、latent dimension、KL 权重、学习率、optimizer steps 和三个训练 seed。输入/输出
  层宽度随 Raw/FMT representation dimension 调整，这是两种输入不可避免的维度差异。
- 固定 VAE 取自 `mainExp_Task2_3D_2.2` 中 FMT 在开发 validation 上的 winner，随后同时
  用于 Raw 与 FMT。这个选择有意回答“在适合 FMT 的同一个 VAE 中，换成 Raw 输入会怎样”，
  不声称它是 Raw 的最强架构。
- 所有 VAE 使用 AdamW、学习率 `1e-3`、exact 4600 optimizer steps。架构分别为：
  channel `linear latent-8, β=1e-5`；Re160 `MLP[128,64], latent-8, β=1e-4`；
  Re640/Re6400 `linear latent-2, β=1e-5`；Tangaroa、两个 delta-wing、F-22
  `MLP[256,128], latent-16, β=1e-3`。
- 开发片 1–8 训练，9–10 只冻结 KMeans 匿名簇到 vortex/non-vortex 的映射；最终评测为
  4 个新时间片。确认片不参与 VAE、FMT block 或 cluster-ID 选择。
- Raw 使用 center-relative pathline geometry；FMT block 可按 physical family 冻结为不同配置。
  标签统一为固定的全场 IVD-p95，没有为了结果修改数据或标签。

代码与配置：`experiments/Run_Task2_3D_Main.py`、`FMT_Utils/VAE_3D.py`、
`config/mainExp_Task2_3D_2.3.yaml`。

## Ibex A100 新时间片结果（论文主表）

| Flow | 同一 VAE | Raw+VAE F1 | FMT+VAE F1 | 配对 F1 增益 | FMT ARI | FMT NMI |
|---|---|---:|---:|---:|---:|---:|
| channel | linear-8 | .0532±.0003 | .2224±.0570 | **+.1693±.0568** | .1773 | .0687 |
| half-cylinder Re160 | MLP-8 | .5548±.0398 | .5535±.0165 | **−.0013±.0549** | .4232 | .2828 |
| half-cylinder Re640 | linear-2 | .4725±.0057 | .6709±.0075 | **+.1984±.0034** | .6048 | .4273 |
| half-cylinder Re6400 | linear-2 | .5636±.0064 | .6897±.0127 | **+.1261±.0178** | .6391 | .4327 |
| Tangaroa | MLP-16 | .7044±.0135 | .7290±.0095 | **+.0246±.0042** | .6713 | .4996 |
| delta-wing resampled | MLP-16 | .3133±.0156 | .8207±.0142 | **+.5074±.0283** | .7993 | .6795 |
| delta-wing original LBM | MLP-16 | .3030±.1098 | .8306±.0101 | **+.5276±.1139** | .8074 | .6801 |
| F-22 | MLP-16 | .6297±.0124 | .4772±.0434 | **−.1525±.0453** | .4062 | .2196 |

8 个数据条目中 6 个平均正增益；条目平均增益 `+.1750`，中位数 `+.1477`。先合并
重复数据条目后，5 个 physical family 中 channel、half-cylinder、Tangaroa、delta-wing
为正，F-22 为负，即 4/5 family 正增益。Re160 的均值为 `−.0013`，在当前误差范围内
应解释为近似持平，而不是 FMT 提升或明显退化。F-22 是稳定反例。

### 与本地 RTX 3090 结果的修订

- **此前本地结果：**7/8 条目正增益；Re160 `+.0140`，Tangaroa `+.0439`，条目平均
  `+.1753`。
- **现在 Ibex A100 复跑：**6/8 条目正增益；Re160 `−.0013`，Tangaroa `+.0246`，条目
  平均 `+.1750`。
- **结论变化：**Re160 从“弱支持”修订为“近似持平”；其余条目的增益方向不变，整体
  4/5 family 正增益的结论不变。
- **原因：**训练代码固定随机 seed，但没有启用 `torch.use_deterministic_algorithms` 或
  CUDA deterministic 设置；因此不同 GPU 上的 VAE 优化不是逐位确定的。由于 Re160
  效应本来接近零，其符号发生翻转。两套机器可读结果均保留，不用一套覆盖另一套。

## 与 2.2 的并列解释

- **2.2 回答的问题：**Raw 和 FMT 各自选择最强 validation VAE 后谁更好。结果为 5/8
  数据条目正增益，3/5 family 正增益。
- **2.3 回答的问题：**同一个按 FMT 开发冻结的 VAE 中，只把输入从 Raw 换成 FMT 后
  latent 聚类是否提高。Ibex 结果为 6/8 数据条目、4/5 family 正增益。
- **变更原因：**Task2 的预定主命题是同一 VAE 的输入表示比较；2.2 擅自允许 Raw 独立
  调优，改变了核心比较。2.2 不删除，降为 strongest-Raw stress test；2.3 是论文主表。

## 可写结论

在当前 5 个独立 3D flow family、8 个数据条目的 same-VAE Ibex 实验中，FMT 输入在 6/8
条目、4/5 family 上提高了 VAE latent 的涡/非涡 KMeans F1，平均配对增益为 `+.175`。
这支持“FMT 通常是比 Raw pathline 更有效的 VAE 输入”，但不能写成“所有 3D flow 都
提高”：F-22 是明确反例，Re160 近似持平。

本地预跑设备为 NVIDIA GeForce RTX 3090，结果位于 `outputs/mainExp_Task2_3D_2.3/`。
Ibex A100-SXM4-80GB 复跑 array job 为 `50790521[0-5]`，汇总 job 为 `50790532`；
论文主结果副本位于 `outputs/mainExp_Task2_3D_2.3_ibex_a100/`。
