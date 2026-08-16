# 表1：PyflowVis 中过去实现的全面梳理（截至 2026-08-16，main @ 3040ace）

> 本文档回答"PyflowVis 里过去到底做了什么"。每条结论均标注代码路径 / git commit / 文档出处。
> 相关度指与本 FMT 仓库三个任务的相关度（Task1 = FMT+KMeans 聚类区分涡区域；Task2 = FMT 插入 VAE；Task3 = FMT 用于 3D 涡分类）。

## 1. 研究线总览

PyflowVis 实际上承载了 **6 条研究线**（这正是"混乱感"的来源）：

| # | 研究线 | 时间段 (git) | 状态 | 与 FMT 三任务相关度 |
|---|---|---|---|---|
| L1 | **VortexTransformer**（2D 非定常流客观涡检测） | 2024-06 ~ 2024-11 | 已发表 CGF 44(2) 2025 | 中（baseline + 数据结构来源） |
| L2 | **3D 6D Observer 交互探索** | （主要在闭源 C++ 引擎） | 已发表 TVCG 2026 | 低 |
| L3 | **FMT：pointwise FTLE 回归 → FTLE/flowmap 超分辨率** | 2025-07 ~ 2026-06 | **失败告终**（见 §3） | 中（encoder 演化史 + 教训） |
| L4 | **FMT：pathline 聚类（= Task1 前身）** | 2025-09 ~ 2026-06 | 定性成功，无定量评测 | **高（直接前身）** |
| L5 | **INR 流场压缩（reference-frame 引导）** | 2025-12 ~ 2026-07（活跃） | 2D 已闭合、3D 平手 | 低（但实验方法论极有价值） |
| L6 | **Hairpin 涡分割（3D）** | 2026-07-15 起，"start another branch" | 移到 `optimal-connection` 仓库（C++ voxel3d） | 中（= Task3 的语义来源） |

## 2. L1：VortexTransformer（已发表，冻结 baseline）

**核心代码**：`DeepUtils/models/segmentation/pathline_transformer.py`（`PathlineTransformerV0`）、`train.py`、`test.py`、`CppProjects/src/VectorFieldCompute.cpp:682-789`（数据生成）、`config/segmentation/pathline_transformer.yaml`。

**数据流**（我们今天说的 primitive 概念即源于此）：
- C++ 生成器按 grid cross seeding：每个种子 + 4 个偏移点（center, x+, x−, y+, y−）积分 5 条 pathline = 1 个 cross primitive；`KpathlinePerGroup: 5`。
- 输入张量 `[B, L=16, K, C=7]`，7 通道 = `(px, py, t, ivd, distance/label槽, vx, vy)`。注意：**输入不只坐标，还带 IVD 和速度**（commit 59f74f0 2024-09-28 曾把 IVD 通道置 0 做消融）。
- **label**：不是 IVD 阈值。是"种子点是否落在拟合出的 Vatistas 涡核（逆仿射回标准坐标后 `rc − ‖x‖ > 0`）"的 0/1，藏在 `pathline[:,0,4]` 槽位内（in-band），Python 端 `getSegmentationofPathlines` 取出后抹 0。
- **训练数据 = Vatistas 合成场**：真实场（cylinder / boussinesq / RFC64）切 32×32 patch → 模拟退火+梯度下降拟合 Vatistas 参数（`FittingVatistasParam.py`，最佳拟合 MSE 0.0094 ≈ 34.7 dB）→ 3000 个稳态场（1500 拟合 + 1400 分布采样 + 100 Killing 硬负样本）× 20 个刚体观察者 = 60,000 个非定常场（约 39 GB）。观察者变换在 t0 是空间恒等 → 20 个变体共享同一客观标签。详见 `docs/from_pyflowvis/vatistas_profile.md`。
- 模型：Linear 嵌入 (pos 3→72, feat 4→72) → 3 层 KNN 向量注意力（Point-Transformer 风格）→ 时间 mean+max 聚合 → kNN 特征传播回全部线 → sigmoid 逐线二分类。
- "客观性"的实现途径 = **数据侧**（Killing 观察者增强 + Vatistas 剖面对半径非线性故旋转观察者消不掉涡），**不是**架构侧不变性。

**遗留问题（复用其数据/评测前必须知道）**：
1. `UnsteadyVastisDataset.py:98` pathline 条数 `*4`，而生成器/test.py/yaml 都是 `*5`（commit 626fc26 引入）——分组边界可能整体错位。
2. `test.py:62` 的 `segmentationCriteria` 只取 `pred[...,0]`——**逐线模型只评估每个样本第 0 条线**，precision/recall/F1 的计算子集极小。
3. pathline 坐标**从未归一化**（yaml 里的 MinMaxNormalization/WhiteNoise 只作用于 dummy field 分支，对 pathline 是空转）；截断线用 −1000 填充且**无 mask** 直接进网络。
4. label 靠文件名含 "saddle"/"zero" 判定 si 类别（`data_utils.py:255`）。

## 3. L3：FMT 之"FTLE/flowmap 超分辨率"线（失败，教训最多）

这条线是 2025-08 ~ 2026-06 的主战场，**FMT encoder 是为它发明的**，但最终结论是负面的。

**演化时间线**（git 为证）：

| 时间 | commit | 事件 |
|---|---|---|
| 2025-08-27 | d89ee4b | 首个 demo：pointwise FTLE 估计 = FMT + NN |
| 2025-09-01 | 9742aa8 | **"fmt pointwise outperform mlp"，FTLE 回归 psnr=41.12 dB**（唯一记录在案的 FMT 正面定量结果；对照 MLP 只吃首尾点） |
| 2025-09-16~18 | 4930b26/712dec4 | UNet 上采样 FTLE、FTLEupsamplingFMT_UnetV2 |
| 2025-09-20~21 | b5621c1/3ff80b0/b5c9100 | tokenizer fmt_vit、ConditionalFMTNet v2/v3（后被清理） |
| 2025-09-24~25 | f8d27fb/eb1f05c/c745d00 | reorganize；hierarchy FMT clustering + 可视化 |
| 2026-05-24 | 6183a5c | **"clean code for restart"**（第一次重启） |
| 2026-06-23 | 16e04e7 | Add DCT（DCT_FMT_encoder + compare 脚本） |
| 2026-06-24 | 7b01b9e | **"Flowmap upsampling model is not good for now, usually ESPCN/Unet is around 27db while bilinear is around 39db"** —— 学习模型比 bilinear 插值差 12 dB，**这条线实质死亡** |

**关键实现**（都已复制到本仓库）：
- `FMT_Utils/model_zoo.py`：全部是超分回归模型。`FTLEUpsamplingFMT_Unet`（逐格点 FMT，强制 stages=0 = 真无参）、`FTLEupsamplingFMT_UnetV2/V3`（8×8 滑窗 FMT token + concat / AttentionFusion 注入 UNet，默认 stages=1 → **带 BatchNorm 可学习参数**）、`FTLEupsamplingDCT_FMT_UnetV2`、`ESPCN`/`UNet` 基线、`PointWiseFMT_Regressor` vs `PointWiseMLP_Regressor`。
- `FMT_Utils/FTLE_fitting_utils.py`：primitive 生成（`generate_Flowmap_SLICE`）、cross-primitive FTLE（`computeFTLEFromPathlineCrossPrimitive`：中心差分 Jacobian → Cauchy-Green 最大特征值）、`flowmap_to_relative`（把 O(offset) 的邻差信号抬到 O(1)——很好的设计思想）。
- `FMT_Utils/flowmap_sr.py`：Jakob et al. 2020 flow-map SR 忠实复现（其"低分 = 高分 `hi[::k,::k]` 子集"的做法是对的，FTLE 线的 linspace 反而错位）。
- `compare_DCT_FMT_vs_FMT.py`（未复制）：设计良好的 tokenizer A/B 框架，但 **outputs/figures 与全部日志中查无 FMT/DCT 两臂的产物，且 `assert mode=="upsamplingFTLE"` 与 yaml 的 `upsamplingFLowMap` 冲突、import 的 `FTLEUpsamplingTrainDataset` 已被删除——即"DCT vs FMT 谁好"从未得到过答案**。

**为什么失败（结合代码的事后诊断，非当时记录）**：滑窗 8×8 使 FMT 特征图分辨率只有 FTLE 场的 1/8，几乎必然被 UNet 的低分 FTLE 通道主导；低/高分辨率网格因 linspace 不构成子集关系，上采样标签本身错位；出界粒子零填充/冻结产生虚假 FTLE 结构。

## 4. L4：FMT pathline 聚类线（= Task1 直接前身）

**核心代码**：`FMT_Clustering.py` + `config/PathlineFMTclustering.yaml` + `FMT_Utils/FMT_encoder.py` + `FMT_Utils/FlowlinePostProcessing.py` + `pnn/libs/flows.py`（可视化）。

**实际做法**（2025-09-25 c745d00 可视化成型，2026-05-24 清理）：
1. 数据：4 个真实/解析 2D 非定常场（cylinder2d, doublegyre2d, beads2d, pipedcylinder2d），时间窗 [0.6, 0.8]×T，grid_sampling 0.25，RK4，dt=0.005，max_steps=300，offset_dist=0.02 → cross primitive `(ny·nx, 5, 300, 3)`，(x,y,t) 物理坐标。
2. 只保留 5 条线全部积分满 300 步的 primitive；`AngleAwareSampling` 把 300 步降到 L=30（按全局转角显著性选同一组时间索引）。
3. 编码器 `FMT`（`temporal_head=None`）：PosE 正弦嵌入（alpha=1000, beta=19）→ `GeoLinePicker`（邻居 = 同一时间步的另外 4 条线上的点，取代 KNN）→ LGA 归一化 + PosE_Geo 加权 → max+mean 池化 + **BatchNorm+GELU** × 2 stages（embed_dim 24 → 输出 96 维）。
4. 对 4 种输入视图（原始 / domain 归一化 / 局部化 / 归一化+局部化）分别 KMeans(k=2)，`multi_points_vis_fast` 并排画图，**人眼评估**。
5. `HierachyFMT_encoder`（多感受野窗口 + 双线性上采样拼特征图）已实现但 config 里 `receptive_fields:[1]`，**实际从未用过 >1 的窗口**（原因见问题分析 P9：内存二次爆炸）。

**结论状态**：用户判定"2D 傅里叶 FMT 聚类成功"——依据是可视化定性结果。**仓库中没有任何定量指标**（无 ARI/NMI/F1，无与 IVD/Q 判据标签的对照）。`FMT_Clustering.py` 在 main 分支当前状态因函数改名而 **ImportError**（本仓库已修复）。

**encoder 家族谱系**（谁是谁的改版）：

| 版本 | 位置 | 邻域 | 时序处理 | 可学习参数 |
|---|---|---|---|---|
| Point-NN 原版 `EncNP` | `pnn/models/point_nn.py`（已整段注释） | FPS+KNN | 无（当无序点云） | BN |
| `EncNPNew` | `pnn/models/point_nn.py:152` | KNN（FPS 关闭；LGA std 修正为逐组逐通道） | 无 | stages≥1 时 BN |
| `FMT` | `FMT_Utils/FMT_encoder.py:283` | `GeoLinePicker`：同一时间步跨线 | 可选 `TemporalDFT`（**可学习复数滤波**；聚类实验里=None） | BN（+DFT 时更多） |
| `HierachyFMT_encoder` | 同文件 :362 | 多感受野窗口 | 同上 | 同上 |
| `DCT_FMT` | `FMT_Utils/DCT_FMT_encoder.py` | 显式 center/neighbor 结构 | \|FFT(x+iy)\| 频谱 | **严格 0** |

## 5. L5：INR 压缩线（无直接关系，但方法论是全仓库最成熟的）

代码在 `experiments/referenceframe_inr_v2|_3d/`，文档 `docs/referenceframe_inr_*.md`（97KB 主文档 + 交接文档 + 实验编号体系）。与 FMT 无关，**但它沉淀的实验纪律应当全盘继承**：
- 任何指标对比前，先验证训练对输入微扰的敏感性 ≪ 待测方法差异（SIREN 曾对 1e-9 扰动敏感 0.7~18 dB；换 Fourier features + ReLU MLP 后降到 0.13 dB）。
- CUDA 确定性配方可做到同配置两次训练 ΔPSNR=0.000000 dB。
- 大网络必报多种子分布（曾出现种子间极差 22.4 dB）。
- 结论可追溯（实验编号 + job id + config），修订必须新旧并列，禁止静默翻转。

## 6. L6：Hairpin 涡分割（Task3 的语义上下文）

2026-07-15 commit 1940fdb "start another branch for hairpin vortex segmentation project" 之后，相关实现在 **`C:\Users\xingdi\sources\optimal-connection`**（C++，`src/voxel3d/ObjectHairpinVortex.*`、`SegmentationField.*`），论文笔记在 `C:\Users\xingdi\sources\PaperReading\guoningChenHairpin/`（Guoning Chen 组的 template-fitting corelines VIS22、extract & characterize TVCG2024 等 4 篇）。PyflowVis 内唯一相关文档是 `docs/small-label-morphology-naming.md`（已复制到 `docs/from_pyflowvis/`）：提出把 Task3 做成 GCD（Generalized Category Discovery，广义类别发现）+ 少量标签（10~50/类）的 5 阶段管线，并有两条重要裁定——**冻结特征+原型头优于端到端微调**；**2D 用于打通管线，3D 才是多形态命名主战场**。

## 7. Task2（VAE）现状

**全仓库（PyflowVis）不存在任何 VAE / autoencoder 实现**（grep VAE/reparam/logvar/KLD 零命中）。`small-label-morphology-naming.md` 把 "FMT/DCT/VAE" 并列为已有 encoder 属于笔误/规划口径。Task2 需要从零写；可直接借鉴的只有 model_zoo 里两种 token 注入范式（concat vs AttentionFusion）与 `flowmap_to_relative`/`flowmap_unit_normalize` 的输入标准化思想。

## 8. 本仓库复制清单与修改记录

从 PyflowVis (main @ 3040ace) 复制，**仅做了两处刻意修改**：

| 修改 | 原因 |
|---|---|
| `FMT_Clustering.py`：`generate_FLowMap_SLICE` → `generate_Flowmap_SLICE`（import + 调用点，共 2 行） | 上游改名导致 ImportError，Task1 基线必须可运行 |
| `requirements_fmt.txt`：UTF-16LE → UTF-8 重新编码 | 原文件是 PowerShell 重定向产物，`pip install -r` 无法读取 |

复制内容：`FLowUtils/`（2D/3D 场结构、积分、判据、采样、Killing observer、可微 ODE、NetCDF/Amira 加载）、`FMT_Utils/`（全部 encoder + 数据工具 + debug_checks + model_zoo + flowmap_sr + vortexExtraction_utiles〔已知不可运行，见问题分析〕）、`DeepUtils/utils/`（EasyConfig）、`pnn/`（models 纯 torch 闭包 + libs 可视化/积分 + configs）、`assets/cuda_kernal/PathlineIntegration2D.cu`、`FittingVatistasParam.py` + `VatistasFlowDatasetGenerator.py` + 对应 config（Task1 定量评测的合成 GT 来源）、`config/PathlineFMTclustering.yaml`、两篇相关文档。

**刻意不复制**：VortexTransformer 模型与训练管线（作为已发表 baseline 冻结在 PyflowVis，遵循"冻结基线"原则）、FTLE_experiment.py 与 compare 脚本（失败的 SR 线入口）、`FLowUtils/vortexCriteria.py`（循环边界 bug，被 `ScalarField2d.py` 向量化版本取代）、`FLowUtils/FTLE.py`（模块顶层 `import pycuda.autoinit`，无 CUDA 即崩）、`flowDatasetUtils/JHTDB_Lodader.py`（**第 9 行明文硬编码 API key，不应进入新仓库**；需要 JHTDB 时把 key 移到环境变量后再引入）、INR 压缩全线、GUI 引擎。
