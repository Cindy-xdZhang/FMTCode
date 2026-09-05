# mainExp_Task2_3D_2.2

> **状态修订（2026-08-23）：**该实验保留为 strongest-Raw stress test，不再作为
> Task2 论文主表。Task2 的 same-VAE 主实验已经由 `mainExp_Task2_3D_2.3` 完成；原因是
> 2.2 允许 Raw/FMT 独立选择 VAE，改变了“只比较输入表示”的核心问题。

## 论文问题

Task2 的唯一主比较是：**Raw pathline → VAE → latent → KMeans** 对比
**FMT(pathline) → VAE → latent → KMeans**。它检验 FMT 是否为 VAE 提供更好的输入，
不检验“VAE 是否提高 FMT direct feature”。

## 冻结与公平性

- 开发片 1–6 训练 VAE，7–8 为每个配置组分别选择 Raw+VAE 和 FMT+VAE 的最佳
  架构，9–10 只确定 KMeans 匿名簇编号；最终另用 4 个新时间片。
- Raw 与 FMT baseline **独立调优**，避免故意使用较弱 Raw。候选均为 exact 4600
  optimizer steps：linear latent 2/8、MLP latent 8/16。每个方法只按自身验证 F1 选择。
- 高 Reynolds 数 half-cylinder 使用已验证的 max-dim160、20³ seeds、短时间窗和
  36D neighbor real-spectrum；其余物理族使用各自开发验证阶段冻结的 FMT block。
- 最终结果是 3 个配对训练 seed；标签是固定的全场 IVD-p95，最终片不参与超参数或
  cluster-ID 选择。

代码与配置：`experiments/Run_Task2_3D_Main.py`、`FMT_Utils/VAE_3D.py`、
`config/mainExp_Task2_3D_2.2.yaml`。

## 新时间片结果

| Flow | Raw+VAE F1 | FMT+VAE F1 | 配对 F1 增益 | FMT ARI | FMT NMI |
|---|---:|---:|---:|---:|---:|
| channel | .0533±.0004 | .2213±.0562 | **+.1681±.0559** | .1761 | .0680 |
| half-cylinder Re160 | .6036±.0060 | .5621±.0165 | **−.0415±.0180** | .4352 | .2898 |
| half-cylinder Re640 | .4726±.0058 | .6709±.0075 | **+.1983±.0034** | .6048 | .4273 |
| half-cylinder Re6400 | .5633±.0061 | .6896±.0127 | **+.1263±.0172** | .6389 | .4326 |
| Tangaroa | .7842±.0037 | .7438±.0095 | **−.0404±.0129** | .6924 | .5087 |
| delta-wing resampled | .3766±.0278 | .8135±.0120 | **+.4370±.0364** | .7916 | .6746 |
| delta-wing original LBM | .4908±.0377 | .8292±.0045 | **+.3383±.0405** | .8058 | .6792 |
| F-22 | .7117±.0076 | .4741±.0245 | **−.2375±.0246** | .4038 | .2183 |

八个数据条目中 5 个正增益、3 个负增益；条目平均增益 `+.1186`、中位数
`+.1472`。按独立物理族先合并重复数据，channel、half-cylinder、delta-wing 为正，
Tangaroa、F-22 为负，即 3/5 family 为正。这个结果不能支持“所有 3D 流场普适提高”。

## 对旧结论的明确修订

- **之前：**旧统一 VAE 的 F-22 报 `+.119`，但该实验在最终 test label 上选择匿名簇
  方向，Raw+VAE 也只有 `.340`。
- **现在：**簇方向由独立 calibration slice 冻结，而且 Raw 与 FMT 分别选择最佳 VAE；
  F-22 Raw+VAE 达 `.712`，FMT+VAE 只有 `.474`，故旧正增益不再成立。
- **原因：**旧结果主要来自较弱的 Raw baseline 和乐观的 test-time cluster mapping，
  不是 FMT 在 F-22 上的稳定优势。

## 可写结论

FMT 显著提高了 VAE 在 high-Re half-cylinder 和两套 delta-wing 上的 latent 聚类质量，
也提高了 channel；但 Re160、Tangaroa 和 F-22 不支持提升。论文可以写“FMT 在多个差异
很大的 3D 物理族上提高 VAE，并在 8 个条目上取得正的平均效应”，不能写“该提升对
所有 3D flow 普适”。

本机运行设备为 NVIDIA GeForce RTX 3090。Ibex array job `50788608[0-5]` 在占用
GPU 前取消，没有产生 Ibex 结果。完整本机机器可读结果：
`outputs/mainExp_Task2_3D_2.2/`。
