# 表1：PyflowVis 里过去都做了什么

写给"半年后忘光了的自己"读。所有结论都能追溯：括号里给出代码路径、git commit 或文档出处。
（考证过程的原始版本更密集，本版为可读性重写；2026-08-16。）

---

## 0. 名词表（先读这个，后文不再解释）

| 名词 | 含义 |
|---|---|
| **primitive（cross primitive）** | 本研究的最小几何单元：一个种子点 + 它上下左右 4 个偏移点，共 5 个点各积分出一条 pathline。张量形状 `[5, L, 3]`，L 是时间采样数，3 是 (x, y, t) 物理坐标。5 条线的固定顺序：中心、x+、x−、y+、y−。 |
| **FTLE** | Finite-Time Lyapunov Exponent，有限时间李雅普诺夫指数。衡量相邻粒子在有限时间内被拉开的速率，常用于找流动结构的"骨架"。可以直接从 primitive 算出来（4 条邻线对中心线做有限差分）。 |
| **IVD** | Instantaneous Vorticity Deviation，瞬时涡量偏差 = \|涡量 − 全场平均涡量\|。是一个客观（与参考系无关）的涡判据，可拿来当涡区域的参考标签。 |
| **Vatistas 涡模型** | 一个带参数的解析涡速度剖面。用它合成的流场自带"涡核在哪里"的精确答案，所以能当训练标签。 |
| **超分辨率（SR, super-resolution）** | 从低分辨率场恢复高分辨率场的任务。 |
| **INR** | Implicit Neural Representation，隐式神经表示：用一个小神经网络过拟合整个场，网络权重即压缩后的数据。 |
| **Point-NN / PosE** | Point-NN 是一个"无训练点云网络"（CVPR 2023）：只用 sin/cos 位置编码（PosE）+ 池化，不学任何权重。FMT encoder 是在它基础上改出来的。 |
| **tokenizer / token** | 把一段几何（这里是一个 primitive 或一窗 primitive）压缩成一个固定长度向量（token）的模块。 |

---

## 1. 一段话总结

PyflowVis 是一个流场可视化基础设施，六年间在上面先后长出了 **6 个研究项目**，这是它现在混乱的根本原因。和本 FMT 仓库真正有关的只有 3 条线：**VortexTransformer**（已发表；贡献了 primitive 这个数据结构和 Vatistas 合成数据管线）、**FMT 超分线**（失败收场；但 FMT encoder 是在这条线上发明和迭代的）、**FMT 聚类线**（Task1 的直接前身；定性成功，但没有留下任何数字）。其余 3 条线（3D 观察者交互、INR 压缩、hairpin 涡分割）与本仓库无直接代码关系。

## 2. 六条线总表

| # | 项目 | 时间 | 一句话说明 | 结局 | 对本仓库的用处 |
|---|---|---|---|---|---|
| L1 | VortexTransformer | 2024-06 ~ 11 | 用 transformer 对 2D 非定常流做逐 pathline 的涡/非涡二分类 | **已发表**（CGF 2025） | primitive 数据结构、Vatistas 合成标签管线、可对照的有参 baseline |
| L2 | 3D 6D-Observer 交互 | （闭源引擎为主） | 3D 非定常流的观察者空间交互探索 | **已发表**（TVCG 2026） | 无直接用处 |
| L3 | FMT 超分线 | 2025-07 ~ 2026-06 | 用 FMT token 辅助 FTLE / flow map 超分辨率 | **失败**：学习模型比 bilinear 插值差 12 dB | FMT encoder 全家、失败教训 |
| L4 | FMT 聚类线 | 2025-09 ~ 2026-06 | FMT 特征 + KMeans 区分涡/非涡区域 | 定性成功，**零定量指标** | = Task1 的起点 |
| L5 | INR 压缩 | 2025-12 ~ 2026-07 | 参考系变换能否帮 INR 压得更好 | 2D 结论闭合，3D 平手 | 只借它的实验纪律 |
| L6 | Hairpin 涡分割 | 2026-07 起 | 3D 发卡涡的提取与分割 | 移去 `optimal-connection` 仓库（C++） | = Task3 的语义背景 |

---

## 3. 逐条线细说

### L1. VortexTransformer（已发表，baseline 冻结在 PyflowVis）

**做了什么**：把每个种子点的 5 条 pathline（即 primitive）喂给一个 Point-Transformer 风格的网络，输出"这条线的种子是否在涡里"的 0/1。输入不只坐标：每个点带 7 个通道 (x, y, t, IVD, 距离/标签槽, vx, vy)。训练数据完全是合成的：先用真实流场的 32×32 小块拟合 Vatistas 参数（模拟退火+梯度下降），再用拟合分布生成 3000 个稳态场，每个套 20 个刚体运动观察者，共 60,000 个非定常场。标签 = 种子点是否落在 Vatistas 涡核内（解析可判），藏在输入张量的第 4 通道里。

**结果**：论文发表（CGF 44(2), 2025）。"客观性"靠**数据增强**实现（同一场的 20 个观察者变体共享同一标签），不是靠架构不变性。

**留给我们什么**：primitive 的定义和 5 线顺序约定；Vatistas 合成数据管线（`FittingVatistasParam.py` + `VatistasFlowDatasetGenerator.py`，已复制，是 Task1 定量评测的标签来源）；一个可对照的"有参数"编码器。

**复用它的评测/数据前要知道的坑**（细节在问题分析 P7）：数据集类里 pathline 条数写成 `*4` 而生成器是 `*5`，分组可能整体错位；旧评测函数只算了每个样本第 0 条线的指标；pathline 坐标从未归一化，截断线用 −1000 填充且无掩码。

关键文件（PyflowVis 内，未复制）：`DeepUtils/models/segmentation/pathline_transformer.py`、`train.py`、`test.py`、`CppProjects/src/VectorFieldCompute.cpp`。
已复制到本仓库：`FittingVatistasParam.py`、`VatistasFlowDatasetGenerator.py`、`docs/from_pyflowvis/vatistas_profile.md`。

### L3. FMT 超分线（失败，但 encoder 在这里诞生）

**做了什么**：想法是"低分辨率 FTLE/flow map + pathline 几何 token → 恢复高分辨率场"。为此把 Point-NN 改造成 pathline 编码器，走了三步：
1. `EncNPNew`（`pnn/models/point_nn.py`）：关掉 Point-NN 的下采样，修正归一化，把 primitive 当点云编码；
2. `FMT`（`FMT_Utils/FMT_encoder.py`）：邻居不再用 KNN，改成"同一时刻的另外 4 条线"（时空有别，这是对的方向）；
3. 各种注入方式：token 拼进 UNet（V2）、注意力门控融合（V3）、8×8 滑窗成特征图。

**结果**：三个数字说明一切。
- 2025-09-01（commit 9742aa8）：逐点 FTLE 回归，FMT+MLP 达到 41.12 dB，好于纯 MLP——**唯一的正面定量结果**；
- 2026-06-24（commit 7b01b9e）：flow map 超分，学习模型约 27 dB，而 **bilinear 插值 39 dB**——整条线宣告失败；
- DCT_FMT vs FMT 的 A/B 对比脚本（`compare_DCT_FMT_vs_FMT.py`）设计良好但**从未跑通**（输出目录里没有这两个模型的任何图和日志），"哪个 tokenizer 好"至今无答案。

**事后诊断**（代码证据，非当时记录）：8×8 滑窗让 FMT 特征图分辨率只有目标场的 1/8，被 UNet 的低分辨率通道主导；低/高分辨率网格用 linspace 生成，互相不是子集，标签本身错位；出界粒子零填充造成假结构。

**留给我们什么**：FMT encoder 全家（已复制）；`flowmap_to_relative`"把小信号抬到 O(1)"的思想；`DCT_FMT`——严格零参数的频域描述子，是 Task1 最值得先试的编码器（其自旋盲区 bug 已在本仓库修复，见问题分析 P2）。

关键文件（已复制）：`FMT_Utils/FMT_encoder.py`、`DCT_FMT_encoder.py`、`DCT_utils.py`、`model_zoo.py`、`FTLE_fitting_utils.py`、`flowmap_sr.py`。

### L4. FMT 聚类线（Task1 的直接前身）

**做了什么**：在 4 个真实/解析 2D 非定常场（cylinder2d、doublegyre2d、beads2d、pipedcylinder2d）上，按 1/4 分辨率撒种子，每个种子积分 primitive（300 步，dt=0.005），按"转角显著性"降采样到 30 个时刻，然后：FMT 编码（2 个 stage，输出 96 维）→ KMeans 分 2 类 → 画图。对 4 种输入（原始 / 域归一化 / 减种子点 / 两者）各做一遍，肉眼对比。

**结果**：可视化上能把涡区域和非涡区域分开（用户判定"成功"）。但仓库里**没有任何数字**：没有和 IVD 标签比过 F1，没有聚类指标，没有固定随机种子的复跑记录。另有两个隐患留给复跑验证：encoder 从未切 eval 模式（BatchNorm 用了批统计量，特征依赖同批样本，可能恰好起了"标准化"的作用）；时间降采样在过滤无效线之前做（零填充段污染显著性排序）。

**留给我们什么**：`FMT_Clustering.py`（Task1 入口，已修复导入并清理）+ `config/PathlineFMTclustering.yaml`。这条管线冻结后复跑、补上数字，就是 mainExp_1.1。

关键文件（已复制）：`FMT_Clustering.py`、`config/PathlineFMTclustering.yaml`、`FMT_Utils/FlowlinePostProcessing.py`（降采样与局部化）、`pnn/libs/flows.py`（画图）。

### L2 / L5 / L6. 与本仓库无直接代码关系的三条线（各一段）

**L2 · 3D 观察者交互**：TVCG 2026 论文，实现主要在闭源 C++ 引擎，PyflowVis 里只有节选。对本仓库无用处。

**L5 · INR 压缩**：研究"先做参考系变换再用 INR 压缩会不会更好"。结论本身与 FMT 无关，但这条线的**实验纪律**是全仓库最好的，直接搬来用：任何指标对比前先验证训练噪声远小于待测差异；锁定 CUDA 确定性（可做到两次训练差 0.000000 dB）；大网络必须报多随机种子分布；每个实验有编号、spec 和结果文档，结论修订必须新旧并列。本仓库的 `docs/experiment_log.md` 就是按这个标准建的。

**L6 · Hairpin 涡分割**：2026-07-15 起在另一个仓库 `optimal-connection`（C++）做 3D 发卡涡。PyflowVis 里留下一份规划文档 `small-label-morphology-naming.md`（已复制到 `docs/from_pyflowvis/`），对 Task3 有两条重要裁定：小标签预算下**冻结特征+简单分类头优于端到端微调**；**2D 用来打通管线，3D 才是多涡型分类的主战场**。

---

## 4. 三个任务现在各自站在哪

| 任务 | 现状 | 下一步（见问题分析 P9） |
|---|---|---|
| **Task1** 聚类分涡 | 管线能跑（本仓库已修复导入），但"成功"只有肉眼证据 | 冻结复跑 + 补定量协议 = mainExp_1.1 |
| **Task2** FMT 插入 VAE | **代码不存在**（PyflowVis 里从未有过 VAE，规划文档把它列为"已有"是口径错误） | 从零写；可借鉴 model_zoo 的两种 token 注入方式 |
| **Task3** 3D 涡分类 | 只有语义背景（L6）和 2D 工具；3D encoder 未设计（按用户指示暂不动） | 等 Task1/2 钉死后再动 |

---

## 5. 本仓库从 PyflowVis 拿了什么、改了什么

**来源**：PyflowVis main @ 3040ace（2026-07-23）。

**复制了**（约 55 个文件）：FMT encoder 全家（`FMT_Utils/`）、2D/3D 流场结构与 pathline 积分（`FLowUtils/`，含 CUDA kernel）、配置工具（`DeepUtils/utils/`）、Point-NN 改造版与画图（`pnn/`）、Vatistas 合成标签管线、Task1 配置、两篇相关文档。

**修改了**（每一处都有 commit 可查）：

| commit | 修改 | 原因 |
|---|---|---|
| a3fd97d | `FMT_Clustering.py` 导入名 `generate_FLowMap_SLICE` → `generate_Flowmap_SLICE`（2 行） | 上游改名导致 Task1 入口 ImportError |
| a3fd97d | `requirements_fmt.txt` 转 UTF-8 | 原文件是 UTF-16，pip 读不了 |
| bd55d45 | **修复 DCT_FMT 自旋盲区 bug**：DFT 幅值同时保留正、负频率对 | 旧代码只取正频率，顺时针涡的特征范数只有逆时针的 8%（9.64 vs 116.93）；修复后两者相等（117.09）。测试：`tests/test_dct_fmt.py` |
| ca653fd | **移除 `HierachyFMT_encoder` 与 `GeoLinePicker`**，FMT 内联等价的同时刻跨线分组 `group_same_timestep` | 层级设计从未真正用过（内存对窗口大小二次爆炸）；重构与旧实现**逐位一致**（eval/train × 有/无 DFT head 四种模式 max\|diff\|=0）。测试：`tests/test_fmt_encoder.py` |

**刻意没拿**：VortexTransformer 模型与训练管线（已发表 baseline，冻结在原仓库）；失败超分线的实验入口；有循环边界 bug 的 `vortexCriteria.py`（用 `ScalarField2d.py` 的向量化判据代替）；模块顶层就要求 CUDA 的 `FTLE.py`；**明文硬编码 API key** 的 JHTDB loader（安全问题）；INR 压缩全线；GUI 引擎。
