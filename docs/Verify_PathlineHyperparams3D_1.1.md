# Verify_PathlineHyperparams3D_1.1：3D Task2 路径线超参数

## 研究问题

本实验分别测量三个路径线参数对 3D Task2 的影响：

1. 积分步长：在固定物理观察时间窗下改变 Runge–Kutta 4 阶积分器的步长。
2. 总积分步数：固定积分步长，因此同时改变路径线的物理观察时间窗。
3. 积分后采样点数：积分轨迹不变，只改变送入 Fourier Motion Token（FMT）的时间采样数。

评价同时覆盖直接 FMT + KMeans 和 FMT → 变分自编码器（Variational Autoencoder, VAE）+ KMeans。这样可以区分“路径线/FMT 本身改变”与“VAE 是否保留该变化”。

## 冻结项

- 数据和 Task2 训练协议来自 `Verify_Task2Universality_1.1`。
- 8 个数据条目：halfcylinder Re160/Re640/Re6400、tangaroa、两个 deltaWing 条目、F22，以及 Killing observer 生成的 synthetic channel control。
- 每个条目固定 10 个 source time；前 8 个无标签训练，后 2 个 held-out 测试。
- 每片 16³ seeds、7 条 pathline primitive；FMT 使用 6 个 Fourier 频率、Gram invariant、chirality、sorted-neighbor pooling 和 post-StandardScaler neighbor weight 0.5。
- IVD 使用标准全域定义，正类阈值固定为完整 IVD volume 的 p95；标签不参与 VAE 训练。
- VAE 为 hidden `[256,128]`、latent dimension 16、β=0.001、约 4600 optimizer steps；训练随机种子 7068、7069、7070。

## 三个控制变量轴

| 轴 | variant | dt（source-frame interval） | 总步数 | 物理时间窗 | 积分后采样数 |
|---|---|---:|---:|---:|---:|
| 积分步长 | `dt_small` | 0.125 | 96 | 12 | 32 |
| 积分步长 | `baseline` | 0.25 | 48 | 12 | 32 |
| 积分步长 | `dt_large` | 0.375 | 32 | 12 | 32 |
| 总步数/时间窗 | `steps_short` | 0.25 | 32 | 8 | 32 |
| 总步数/时间窗 | `baseline` | 0.25 | 48 | 12 | 32 |
| 总步数/时间窗 | `steps_long` | 0.25 | 96 | 24 | 32 |
| 积分后采样 | `samples_16` | 0.25 | 48 | 12 | 16 |
| 积分后采样 | `baseline` | 0.25 | 48 | 12 | 32 |
| 积分后采样 | `samples_48` | 0.25 | 48 | 12 | 48 |

这里“总步数”的实验不能解释为纯计算步数效应：dt 固定时，总步数和物理观察时间窗绑定。积分步长轴才是在相同 12-frame 时间窗下测量数值离散与轨迹采样密度。

## 可比性修正

第一次单种子 pilot 直接使用每个 variant 各自有效的 primitive。长时间窗会让更多路径线离开空间域，因此不同 variant 的评测样本不一致；这些 pilot 结果位于 `outputs/Verify_PathlineHyperparams3D_1.1/results`，不进入最终结论。

最终协议使用 `results_common`：

- 7 个 variant 对每个数据条目使用完全相同的 10 个 source indices。
- 每个 timeslice 对 7 个 `valid_mask` 求交集，只保留所有 variant 都完整积分的 seed。
- 代码逐片检查 source index 和共同 seed 上的 IVD 标签逐位相同，不一致立即报错。
- 另行报告每个 variant 原生的有效 primitive 比例，因此较长时间窗造成的数据覆盖损失不会被隐藏。

固定 source indices：

| 数据条目 | 10 个 source indices |
|---|---|
| cylinder3d / Re160 | 31, 41, 52, 62, 73, 83, 94, 104, 115, 125 |
| halfcylinderRe640 | 16, 20, 24, 27, 31, 35, 39, 42, 46, 50 |
| halfcylinderRe6400 | 31, 41, 52, 62, 73, 83, 94, 104, 115, 125 |
| tangaroa | 41, 56, 71, 86, 101, 115, 130, 145, 160, 175 |
| deltaWing_resampled | 35, 47, 59, 72, 84, 96, 108, 121, 133, 145 |
| deltaWing_LBM | 47, 65, 83, 101, 119, 136, 154, 172, 190, 208 |
| f22raptor | 32, 43, 54, 66, 77, 88, 99, 111, 122, 133 |
| channel | 32, 43, 54, 66, 77, 88, 99, 111, 122, 133 |

## 汇总规则

三个 halfcylinder Reynolds number 先求 family mean，两个 deltaWing 条目先求 family mean；再与 tangaroa、F22 等权平均，得到 physical-family macro F1。Synthetic channel 是客观性 control，单独报告，不混入物理 family 总结。参数效应使用同一数据条目、同一 VAE seed 相对 baseline 的配对 F1 差值。三个 seed 用于估计训练随机性，不把 8 个数据条目错误当作 8 个独立重复实验。

## 结果

主汇总使用 physical-family macro F1；VAE 的 `±` 是三个训练随机种子上的标准差。`ΔF1` 是与 baseline 同一流场、同一训练随机种子的配对差值。构建时间是本机实测总时间的近似比值，不作为跨硬件 benchmark。

| variant | FMT direct F1 | FMT+VAE F1 | direct ΔF1 | VAE ΔF1 | 构建时间 / baseline | 原生有效 primitive |
|---|---:|---:|---:|---:|---:|---:|
| `dt_small` | 0.6071 | 0.5126 ± 0.0409 | −0.0005 | −0.0219 ± 0.0338 | 1.752 | 90.47% |
| `baseline` | 0.6076 | **0.5345 ± 0.0075** | — | — | 1.000 | 90.47% |
| `dt_large` | 0.5985 | 0.5213 ± 0.0508 | −0.0091 | −0.0132 ± 0.0490 | 0.745 | 90.47% |
| `steps_short` | **0.6179** | 0.5277 ± 0.0221 | +0.0103 | −0.0069 ± 0.0251 | **0.689** | **94.52%** |
| `steps_long` | 0.5746 | 0.4196 ± 0.0448 | −0.0330 | **−0.1149 ± 0.0524** | 1.827 | **77.53%** |
| `samples_16` | 0.6174 | 0.5091 ± 0.0127 | +0.0098 | −0.0255 ± 0.0052 | 1.050 | 90.47% |
| `samples_48` | 0.6024 | 0.5258 ± 0.0044 | −0.0051 | −0.0088 ± 0.0053 | 1.007 | 90.47% |

### 结论 1：Task2 的最佳已测试配置仍是 baseline

在本搜索范围内，`dt_scale=0.25, integration_steps=48, sampled_steps=32` 的 FMT+VAE physical-family macro F1 最高（0.5345），而且三个 seed 的标准差最小（0.0075）。因此不修改 `Verify_Task2Universality_1.1` 的路径线配置。

固定 12-frame 物理时间窗时，dt=0.125 没有比 dt=0.25 更准，构建时间却增至 1.75 倍；dt=0.375 节省约 25% 构建时间，但 direct FMT 下降 0.0091，VAE 的变化在三个 seed 间不稳定。dt=0.25 是本实验中较稳妥的数值/成本折中。

### 结论 2：96 步、24-frame 的长时间窗明确有害

`steps_long` 是唯一在 aggregate 上同时明显破坏 direct FMT 和 FMT+VAE 的设置。它相对 baseline 的 VAE 配对差值在三个 seed 上全部为负，范围为 −0.1773 到 −0.0492；构建时间达到 1.83 倍，原生有效 primitive 从 90.47% 降到 77.53%。逐流场看，它降低 6/7 个物理数据条目的 direct F1；VAE 只在 Re6400 和 F22 上增加，另外 5 项降低。因此当前 FMT 不应默认追求更长 pathline。

`steps_short` 的 direct FMT 反而提高 0.0103，构建时间降至 0.689 倍，有效率升至 94.52%；但 VAE aggregate 只与 baseline 近似持平（−0.0069 ± 0.0251），且流场差异很大：tangaroa 增加 0.1461，未降采样 deltaWing 降低 0.3724。它可作为强调速度/覆盖率的备选配置，不能称为跨流场更优。

### 结论 3：32 个积分后采样点最适合当前 VAE

16 点使 direct FMT 提高 0.0098，但经 VAE 后三个 seed 的 aggregate 差值全部为负（−0.0323 到 −0.0197）。这说明 VAE 没有保留 16 点配置产生的判别优势。48 点同样没有提升：direct 降低 0.0051，VAE 三个 seed 全部小幅降低（−0.0163 到 −0.0048），构建时间也几乎不变。当前 Task2 应保留 32 点。

### 结论 4：重构误差仍不能代表聚类质量

dt=0.375 和 48 个采样点的重构均方误差都低于 baseline，但 F1 更低；`steps_short` 的重构误差最高（0.1080 vs baseline 0.0589），VAE F1 却接近 baseline。7 个 variant 的重构误差与 F1 的描述性 Pearson correlation 只有 0.075。样本数只有 7 且三个轴不是独立实验，因此不作显著性解释；它只再次证明不能用 reconstruction loss 代替 vortex clustering 指标。

### 数据覆盖与确定性限制

全 variant 共同有效集合的平均覆盖率为 77.53%。逐数据条目为：Re160 70.39%、Re640 70.24%、Re6400 72.01%、tangaroa 80.74%、两个 deltaWing 均为 100%、channel 99.82%、F22 27.07%。所以 F22 的 F1 仅描述能在 24-frame 长窗口内完整存活的粒子子集；不能外推到其全部 seed。

早期并行筛选重复计算 direct KMeans 时出现最大 0.00223 的浮点漂移。最终 direct 表全部由单线程 KMeans 重新计算，构建/训练脚本也已固定该线程设置。小于约 0.002 的旧 direct 差别不作方法优劣解释。

## 可追溯文件

- 配置：`config/Verify_PathlineHyperparams3D_1.1.yaml`
- 构建/训练：`Run_Pathline_Hyperparams_3D.py`
- 汇总：`Summarize_Pathline_Hyperparams_3D.py`
- 最终逐次结果：`outputs/Verify_PathlineHyperparams3D_1.1/all_runs_common.csv`
- variant 汇总：`outputs/Verify_PathlineHyperparams3D_1.1/variant_summary.csv`
- 相对 baseline 的配对效应：`outputs/Verify_PathlineHyperparams3D_1.1/effects_vs_baseline.csv`
- 主三轴图：`outputs/Verify_PathlineHyperparams3D_1.1/pathline_hyperparameter_effects.png`
- 逐流场差值图：`outputs/Verify_PathlineHyperparams3D_1.1/pathline_hyperparameter_per_flow.png`
- 成本/有效率图：`outputs/Verify_PathlineHyperparams3D_1.1/pathline_hyperparameter_cost.png`
- 重构误差图：`outputs/Verify_PathlineHyperparams3D_1.1/pathline_hyperparameter_reconstruction.png`
