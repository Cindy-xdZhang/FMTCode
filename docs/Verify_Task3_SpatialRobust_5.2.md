# Verify_Task3_SpatialRobust_5.2：空间增强开发搜索预注册

## 状态

本实验已预注册代码、数据边界、候选空间和最终空间相位，尚未提交 Ibex，亦未生成
`mainExp_Task3_3D_5.2` 的任何最终 cache。`mainExp_Task3_3D_5.1` 的最终结论永久
保持为 F1 增益 `+.13591`、未达到 `+.15`；5.2 不修改该记录。

## 目的与主比较

5.1 的 FMT residual 相对同宽、同结构 Raw-PCA residual 在 10 个数据条目上全部
正增益，但新空间 primitive population 的 dataset-macro F1 增益为 `+.13591`，
距离预注册目标还差 `+.01409`。5.2 只研究已经暴露的 development 数据，目标是
降低不同 seed-grid phase 引起的空间采样敏感性。

主比较仍固定为：

`FMT residual − same-width, same-structure, train-only Raw-PCA residual`。

Raw 与 Raw-wide 仅作诊断，不能替换主对照。whole-field IVD p95 标签保持不变。

## 数据边界

- 基础训练：原 development ordinals 0--5。
- 公开训练增强：`mainExp_Task3_3D_4.1` 的空间相位
  `[0.31,-0.23,0.17]`，4 个切片。该结果早已公开，5.2 明确标为
  `exposed_development`，并同时加入 FMT 与 Raw-PCA residual 的训练集。
- 基础验证：原 development ordinals 6--9。
- 公开空间验证：`mainExp_Task3_3D_5.1` 的空间相位
  `[-0.37,0.29,-0.11]`，4 个切片。它只用于 5.2 development 选择。
- 最终确认：在 Stage 2 selection 冻结前禁止生成。最终相位和物理时间见下文。

冻结 Raw backbone 的坐标均值和标准差始终从原 Raw checkpoint 读取，不因训练增强
而重算。新增 primitive 只扩展两条共同 residual 训练臂；FMT normalization 与
Raw-PCA 仅在相同增强训练样本上拟合。运行时逐 seed 强制核对 checkpoint 的 Raw
normalization，并强制 FMT/Raw-PCA 两臂的 trainable parameter count 相同。

## 搜索空间

Stage 1 对 10 个数据条目搜索 30 个无可训练参数、只读取 pathline primitive 的
feature block，paired seeds 为 40--41。它保留 5.1 的全部 18 个候选，并增加：

- 二阶单边端点差分 `d2`：以
  `(-3D0 + 4D1 - D2)/2` 更准确地估计 seed-time flow-map differential 导数；
- `aivd1w3d2`、`aivd2w8d2` 及短/长时间尺度组合；
- anchored vorticity-deviation block 与 kinematic/FMT block 的组合。

Stage 2 每个 physical family 保留四个 feature，再与 18 个 residual 设置组合：
`geometry+FMT`、`FMT-only`、`dual`，auxiliary width 64/96/128，学习率
`3e-4/1e-3/2e-3`，固定 alpha 1，以及 alpha `[0,6]`、步长 `.05` 的搜索。
paired seeds 为 40--42。family-specific 配方按 validation 上相对匹配 Raw-PCA 的
F1 增益排序，Average Precision、最差 seed 和绝对 FMT F1依次作 tie-break。

开发目标为 dataset-macro F1 增益至少 `+.15`。开发值只用于冻结配方，不进入论文
最终主表。

## 一次性最终确认

最终 phase 不是人工选择。固定字符串
`mainExp_Task3_3D_5.2|final-phase-v1` 的 SHA-256 为：

`45f7218a508f675d58750bd33b41c0718c398fbeaa11fc0b225ddf914b0df655`。

由此前注册的确定性规则取 centered Halton index 395、bases `(2,3,5)`，得到：

`[0.318359375, 0.4561042524005485, -0.3352]`。

该相位与 4.1、5.1 均不同。最终使用与二者完全相同的四个 physical time indices，
只改变空间 primitive population。Stage 2 selection JSON 及 SHA-256 写入不可变
manifest 后，builder 才允许生成 40 个最终切片。最终使用 paired seeds 40--44；
confirmation 不选择 feature、网络、epoch、threshold、alpha 或标签。无论是否达到
`+.15`，不得更换 phase 或重新选择配方。

## 入口

- Feature：`FMT_Utils/DFT_FMT_3D.py`, `FMT_Utils/Task12Data_3D.py`
- 搜索：`Search_Task3_FMTResidual_3D.py`,
  `Search_Task3_FMTResidual_Stage2_3D.py`
- 搜索 config：`config/Verify_Task3_SpatialRobust_5.2.yaml`
- 最终 builder：`Build_Task3_SpatialRobust_Confirmation_5_2.py`
- 最终 evaluator：`Run_Task3_FMTResidual_Frozen_5_2.py`
