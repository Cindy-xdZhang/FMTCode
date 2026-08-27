# Verify_Task3_AnchoredRobust_5.1

## 目的

`mainExp_Task3_3D_4.1` 在新空间 primitive population 上确认 FMT residual
相对同宽、同结构 Raw-PCA residual 的 dataset-macro F1/AP 增益为
`+.10000/+.11896`；方向在 10/10 个数据条目上为正，但 F1 没有达到预期的
`+.15`。本实验只改进 FMT feature 与 residual 超参数，不改变 whole-field IVD
p95 标签，也不以 IVD percentile 选择方法。

## 数据边界

- 旧 development ordinal 0--5：训练。
- 旧 ordinal 6--9：已被 4.1 搜索和 outer check 打开，全部归为 5.1
  development validation。
- 4.1 的空间相位 `[.31,-.23,.17]`：结果已公开，归为 5.1 development
  validation；绝不再称为 confirmation。
- 5.1 当前没有生成或打开最终 confirmation。Stage 2 配方冻结并写出 SHA-256
  后，才允许生成预注册空间相位 `[-.37,.29,-.11]` 的 primitive。物理时间与
  4.1 相同，从而只检验新的空间 primitive population。

训练 population 与冻结 Raw backbone 保持不变；公开空间相位只加入 validation，
不加入训练。每个 FMT 候选均配一个仅在训练集拟合、与 FMT 输入等宽的 Raw-PCA
residual；两臂使用相同 residual route、auxiliary width、学习率、训练轮数和 seed。

## 新增 FMT block

`aivd{k}w{L}` 是无可训练参数的 anchored vorticity-deviation Fourier block：

1. 由 7-line primitive 的三组正负邻居差分估计局部 flow-map differential；
2. 由其时间差分和 pseudoinverse 估计局部 velocity gradient；
3. 计算 pathline 推导的 `|vorticity - spatial mean vorticity|` sequence；
4. 对前 `L` 个采样点保留 `k` 个离散 Fourier 频率，并附加首值、早期均值、
   全窗均值、标准差、最大值、最小值和末值。

它只读取 pathline primitive，不读取 IVD volume、p95 threshold 或二值 label。
`aivdq` 另加入 signed Q-like channel；`akin` 使用四个 kinematic channels。

## Stage 0：固定线性预筛

设备为本地 NVIDIA GeForce RTX 3090 24 GB。每个 feature block 使用相同的
balanced logistic classifier；同宽 Raw-PCA 只在 ordinal 0--5 拟合。阈值只在旧
ordinal 6--9 选择，然后原样评估 4.1 空间相位。该步骤只删除明显弱候选，不能作为
Task3 最终性能。

| Family | 首名 feature | 维数 | 四项 family-macro 最小增益 | 四项平均增益 |
|---|---|---:|---:|---:|
| Channel observer | `aivd2w4` | 10 | +.6932 | +.7931 |
| Half-cylinder | `aivd4w16` | 14 | +.4696 | +.5204 |
| Tangaroa | `aivd6w16` | 18 | +.1862 | +.2574 |
| Delta-wing | `aivd6w32` | 18 | +.4031 | +.5675 |
| F-22 | `aivd1w3` | 8 | +.2379 | +.3528 |
| Boeing 747 | `fmt_all+kin4` | 189 | +.2180 | +.3256 |
| Smoke buoyancy | `aivdq2w8` | 20 | +.2683 | +.3129 |

“四项”是旧 validation 与公开空间相位各自的 F1、Average Precision 相对同宽
Raw-PCA 的增益。完整表含 10 datasets × 18 candidates = 180 行。

本地没有下载 Boeing/Smoke 的独立 Task3 label mirror，因此 Stage 0 对这两个场显式
使用 source cache 中保存的 `reference`；4.1 Task3 主链曾要求该数组与独立 label
逐位一致。Ibex Stage 1/2 不允许此 fallback，缺独立 label cache 会直接失败。

Stage 0 SHA-256：

- `per_dataset_candidate.csv`：
  `c4ce2c8c0a5ab33c68fd79bf75cdc40337af2751edb0cc5ff5f36172faebd465`
- `family_leaderboard.csv`：
  `6ec17a83c110a09449544ccb33dfd4233b6e660dd1c0011310e030305c4a1f19`

## Ibex 预注册流程

1. Stage 1：18 个 feature candidates × 10 datasets，paired seeds 40--41；
   family-specific 保留前三名。
2. Stage 2：每个 family 的前三个 feature 与 10 个 residual network/fusion
   设置组合，paired seeds 40--42；以同结构 Raw-PCA 的 validation F1 增益排序，
   AP、最差 seed 和绝对 FMT F1 依次作 tie-break。
3. 写出冻结 Stage 2 selection 及 SHA-256；到此为止不生成 5.1 confirmation。
4. 使用预注册相位 `[-.37,.29,-.11]` 生成全新 primitive population，以五个
   paired seeds 最终评估。生成器在 Stage 2 selection 不存在时会直接拒绝运行，
   首次生成时把 selection SHA-256 原子写入冻结 manifest。

主目标仍是十个数据条目的 dataset-macro `FMT residual - Raw-PCA residual` F1
增益至少 `+.15`。Raw 与 Raw-wide 同时报告，但不能替代主对照。

## 入口

- Feature：`FMT_Utils/DFT_FMT_3D.py`, `FMT_Utils/Task12Data_3D.py`
- Stage 0：`Screen_Task3_AnchoredRobust_3D.py`
- Stage 1/2：`Search_Task3_FMTResidual_3D.py`,
  `Search_Task3_FMTResidual_Stage2_3D.py`
- Config：`config/Verify_Task3_AnchoredRobust_5.1.yaml`
- 小型结果：`docs/results/Verify_Task3_AnchoredRobust_5.1/`

## Ibex 搜索与最终状态

Stage 1 的 180/180 jobs、Stage 2 的 300/300 jobs 均完成。Stage 2 development
dataset-macro F1/Average Precision 增益为 `+.16025/+.17163`，selection SHA-256
为 `8341272e5984008cb0d39059f6fb84dbeea4251989b0a37040e931481968a2ab`。

冻结后生成空间相位 `[-.37,.29,-.11]` 的 40 个新切片，共 155,157 个有效
primitive。最终 10 条目×4 方法×5 seeds 的 200/200 条记录完整，所有 stderr
为空。相对同宽同结构 Raw-PCA residual，最终 dataset-macro F1/Average Precision
增益为 `+.13591/+.14971`，两项均 10/10 条目、5/5 paired seeds 为正。

因此 anchored FMT 明显改进了 4.1 的 `+.10000/+.11896`，但预注册的 F1
`+.15` 目标仍未达到，不能按成功记载。完整逐流场表、机器、哈希和结论边界见
`docs/mainExp_Task3_3D_5.1.md`。
