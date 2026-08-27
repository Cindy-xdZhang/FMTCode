# Verify_Task3_F22AnchoredFeatures_1.2

## 目的

`Verify_Task3_F22Hyperparams_1.1` 已证明 FMT residual 明显优于 Raw/Raw-wide，
但在 8 个 fresh F-22 时间片上相对同结构 Raw-PCA residual 仅
`F1 +.00003 / AP -.00082`，只能判为持平。本实验继续在 development
时间片上搜索 F-22 family-specific FMT，目标是在新的、冻结的时间片上确认
FMT 是否也优于这个同结构 Raw-only 压力测试。

本实验没有降低 Raw-PCA 的训练预算。每个 FMT 候选均与相同 auxiliary
维度、相同 residual 网络、相同学习率、相同训练轮数和相同随机种子的
train-only Raw-PCA 配对。

## 方法

旧 `kin6` 对完整 32 个 pathline 采样点作低频 Fourier 汇总，可能稀释
Task3 seed-time IVD 标签所需的早期局部旋转信息。新增的 anchored
kinematic Fourier block 仍然无可训练参数：

1. 用 7-line primitive 的三组正负邻居差分估计局部 flow-map differential；
2. 由时间差分和 pseudoinverse 估计局部速度梯度；
3. 计算涡量相对同一时间片空间均值的模长；
4. 对早期窗口作离散 Fourier 变换，并附加首值、前四分之一均值、全窗
   均值、标准差、最大值、最小值和末值七个时间锚点。

最终 `aivd2w8` 只使用上述 IVD-like scalar 的前 8 个采样点、2 个 Fourier
频率，输出 10 维 feature。它不读取 IVD volume、percentile threshold 或
二分类 label；监督信息只进入后续 residual classifier 的训练。

## Development 搜索协议

- 12 个 source starts：`[33,42,51,67,73,83,97,103,110,125,132,139]`。
- Train starts：`[33,51,67,83,97,110,125,139]`；validation starts：
  `[42,73,103,132]`。
- Stage 0：对 19 个固定 FMT 候选运行同一 Logistic Regression 快速筛选，
  按 `min(F1 gain, AP gain)` 保留前三名。
- Stage 1：前三名使用完整 frozen-Raw residual 网络，seeds 50–54；按五个
  seeds 中所有 F1/AP 配对增益的最小值选择候选。
- `aivd4w8` 与 `aivd4w16` 都是 14 维，因此其 Raw-PCA 输入和训练过程完全
  相同；逐 epoch 验证一致后复用同一组 Raw-PCA 结果，不重复计算。

| 候选 | 维数 | FMT F1 | Raw-PCA F1 | F1 gain | FMT AP | Raw-PCA AP | AP gain | 五种子最差 F1/AP gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `aivd2w8` | 10 | .95370 | .94270 | +.01100 | .98776 | .97918 | +.00858 | +.00667 |
| `aivd4w16` | 14 | .95487 | .94475 | +.01013 | .98799 | .97984 | +.00815 | +.00553 |
| `aivd4w8` | 14 | .95321 | .94475 | +.00846 | .98788 | .97984 | +.00804 | +.00447 |

因此在打开 confirmation cache 前冻结 `aivd2w8`。Selection SHA-256 为
`5a0ed082935487d93e765e2ee30d05a0f097b6e6895c4ac4f3f74ce61bd2daa9`。

## Fresh confirmation

确认时间索引 `[36,55,76,91,106,120,129,140]` 在本轮任何候选选择前冻结，
与 development 和冻结时可见的全部 F-22 cache manifest 不重叠。Schedule
SHA-256 为
`24bcc8b3ef1984f456165d7263798bdfe04881199b01c36bafa2322701f57511`。
标签仍是 whole-field IVD p95，确认集不参与训练、checkpoint、threshold、
residual alpha 或候选选择。

| 方法 | Mean F1 | Mean AP |
|---|---:|---:|
| Raw | .93303 | .97598 |
| Raw-wide | .93179 | .97379 |
| 10D Raw-PCA residual | .94569 | .98179 |
| 10D anchored FMT residual | **.95458** | **.98954** |

| Seed | F1 gain vs Raw-PCA | AP gain vs Raw-PCA |
|---:|---:|---:|
| 50 | +.00605 | +.00820 |
| 51 | +.01780 | +.01011 |
| 52 | +.00839 | +.00716 |
| 53 | +.00795 | +.00680 |
| 54 | +.00423 | +.00649 |
| Mean | **+.00889** | **+.00775** |

五个随机种子的 F1 与 AP 配对增益全部为正。Confirmation audit SHA-256
为 `8472bbcd19e278ce5c9f1f1766d5b446af05191289bb258c86576775ce4a7e2f`；
逐 seed CSV SHA-256 为
`4db3a9eaf67ed6f57b7168a163693f2156eccee328d96f1ecf35bc2190d93b94`。

## 结论修订

- **之前：**`kin6` 在 fresh F-22 上相对 Raw-PCA residual 持平，不能宣称
  FMT 优于所有 Raw-only 特征扩展。
- **现在：**针对 seed-time IVD 设计并仅用 development 选择的
  `aivd2w8`，在另一组 8 个 fresh 时间片上相对同宽 Raw-PCA residual 的
  F1/AP 提高 `+.00889/+.00775`，且五个 seeds 两项均为正。
- **原因：**旧 block 主要汇总完整积分窗；新 block 明确保留 seed 和早期
  IVD-like 几何信息。变化来自 FMT 表示，而非削弱 Raw-PCA 网络。
- **边界：**这是 F-22 family-specific extension，不应静默回填或覆盖
  `mainExp_Task3_3D_3.2_global_ivd` 的冻结主表。它支持“F-22 可以通过合适
  FMT 获得稳定增益”，不证明一个统一 FMT 配方对任意 3D flow 都最优。

## 可追溯入口

- Feature：`FMT_Utils/DFT_FMT_3D.py`、`FMT_Utils/Task12Data_3D.py`
- 搜索：`Screen_Task3_F22AnchoredFeatures.py`、
  `Search_Task3_FMTResidual_3D.py`、`Select_Task3_F22AnchoredFeatures.py`
- Confirmation：`Evaluate_Task3_F22_AnchoredConfirmation.py`
- 配置：`config/Verify_Task3_F22AnchoredFeatures_1.2_*.yaml`、
  `config/Confirm_Task3_F22AnchoredFeatures_1.2_*.yaml`
- 已提交实现 commits：`07ee64b`、`7568ac6`
- 小型结果归档：`docs/results/Verify_Task3_F22AnchoredFeatures_1.2/`
- 完整本地结果：`outputs/Verify_Task3_F22AnchoredFeatures_1.2/`、
  `outputs/Confirm_Task3_F22AnchoredFeatures_1.2/`
- 设备：本地 NVIDIA GeForce RTX 3090 24 GB；本实验未提交 Ibex job。
