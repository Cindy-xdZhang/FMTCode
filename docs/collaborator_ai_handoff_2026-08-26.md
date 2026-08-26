# FMT 研究交接文档：给合作者及其 AI session

> 状态日期：2026-08-26 16:55（Asia/Riyadh）
> 本地仓库：`C:\Users\xingdi\sources\FMT`
> Ibex 工作目录：`/home/zhanx0o/FMT_Task12_3D_20260823`
> 写本文档时 Git HEAD：`7b31944`
> 当前工作重点：3D Task1、Task2、Task3；Task5 正在运行；Task4 尚未开始。

本文档应作为新 AI session 的入口。它总结研究动机、固定协议、当前可信结果、代码与
数据位置、已知反例及未完成事项。详细结论仍以
`docs/experiment_log.md` 和机器结果 CSV 为准。

---

## 1. 给接手 AI 的首要指令

1. 先读 `AGENTS.md`、`docs/research_tasks_and_protocol.md`、
   `docs/experiment_log.md` 和 `docs/ibex_run_registry.md`。
2. 不要根据旧 README 或旧实验目录自行判断“当前主结果”。本文第8节列出了 canonical
   experiment version。
3. 不得使用 confirmation/test 标签选择 feature、VAE、checkpoint、分类阈值、
   residual 权重或 cluster-to-class 映射。
4. 任何新实验必须使用版本号并保留失败结果。任何 Ibex 进程提交后立即登记到
   `docs/ibex_run_registry.md`。
5. 仓库工作树长期存在其他研究者的 modified/untracked 文件。不得清理、reset、覆盖或
   一次性全部提交；每次只 stage 自己明确修改的文件。
6. 用户接受不同 physical family 使用不同超参数，只要选择过程不读取 test、代码最终
   开源、负结果不被删除。学术问题不是“所有任务必须共享同一超参数”，而是不得为制造
   胜利而篡改数据、标签或隐藏反例。
7. 用户对 Task1 的目标是实际聚类性能，不要求证明完整参考系客观性。不要擅自把
   “严格 objective encoder”设成 Task1 的验收条件。

---

## 2. 研究背景与核心命题

研究对象是非定常流场中的粒子轨线（pathline）。传统神经网络常把轨线当作无序点云，
在时间和空间上不加区分地做 KNN，这不符合 pathline 的结构。我们改用一个固定的局部
几何单元——pathline cross primitive——并研究无训练参数的 Fourier Map Tokenizer
（FMT）能否提取对涡识别有用的几何特征。

3D primitive 由一个中心种子和 `x±、y±、z±` 六个邻居种子组成，每个种子积分一条
pathline，共7条线。固定顺序为：

```text
center, x+, x-, y+, y-, z+, z-
```

典型网络输入形状为：

```text
[N, 7, 32, 3]
```

其中 `N` 是 primitive 数量，32 是积分后统一重采样点数，3 是三维空间坐标。时间由
采样顺序隐式表示，原始缓存 flatten 后是 `7×32×3 = 672` 维。

FMT encoder 本身不通过标签或重构损失学习参数。当前 3D 实现主要位于
`FMT_Utils/DFT_FMT_3D.py`，由 DFT、Gram 几何量、chirality、邻居聚合及若干运动学
序列组成。注意：KMeans、VAE 和监督分类器仍然需要拟合；“training-free”只描述
FMT encoder。

研究的总体问题不是“FMT 是否在数学上对所有变换、所有流场都必胜”，而是：

- FMT 是否能直接产生可聚类的涡/非涡表示；
- FMT 是否是比 Raw pathline 更好的 VAE 输入；
- FMT 是否能提高 IVD 监督分类器；
- 这些提升在多少种不同 3D flow family 上成立，反例是什么。

---

## 3. 标签：统一 whole-field IVD p95

当前 Task1–Task3 的主参考标签统一为 Instantaneous Vorticity Deviation（IVD，瞬时
涡量偏差）：

```text
IVD(x,t) = || omega(x,t) - spatial_mean_x(omega(x,t)) ||
```

对每个时间片，先在整个三维体数据上计算 IVD，再取全体体素的第95百分位：

```text
vortex(seed) = IVD(seed) >= percentile_95(IVD_volume)
```

重要边界：

- 这里没有旧版的局部 `11×11×11` 平均区域参数 `a`。
- Task3 3.2 的标签直接逐位复制 Task1/Task2 source cache 中冻结的 `reference`，避免
  两套 IVD 实现漂移。
- 旧 Task3 3.1 使用 `.9 × mean_11^3(IVD)` 的局部标签，只保留作历史对照，不再是
  当前论文主标签。

---

## 4. 固定任务定义

### Task1：training-free FMT + KMeans

```text
primitive -> FMT -> feature -> KMeans(k=2)
```

核心比较：Raw geometry + KMeans 与 FMT + KMeans。主要指标为 ARI（Adjusted Rand
Index，调整兰德指数）、NMI（Normalized Mutual Information，归一化互信息）、F1 和
IoU。KMeans 只在 development/train feature 上拟合，匿名 cluster 到涡/非涡的映射
只能在 calibration/validation 上冻结。

Task1 只回答 FMT feature 是否足以做无监督二类涡区域聚类，以及是否优于 Raw。完整
参考系不变性不是当前验收条件。

### Task2：FMT 是否改善同一 VAE 的输入

固定主比较：

```text
Raw pathline -> VAE -> latent -> KMeans
FMT(pathline) -> same VAE -> latent -> KMeans
```

同一 physical family 内，两臂必须使用同一个已冻结的 VAE hidden layers、latent
dimension、KL 权重、学习率、训练步数和 checkpoint 规则。VAE 先围绕 FMT 在
development 上开发，随后完全相同的配置用于 Raw；不得在主表中用 confirmation 为 Raw
另搜一套更强 VAE。独立优化的 strongest-Raw 只能进入附录压力测试。

Task2 的正确结论是“FMT 是否是更好的 VAE 输入”，不是“VAE 是否提高 FMT feature”。
`FMT+VAE < FMT direct` 不会否定 Task2，但说明 VAE 可能损失了 FMT 的判别信息。

### Task3：有监督 IVD 涡/非涡二分类

核心比较包含：

```text
Raw
Raw-wide
frozen Raw backbone + 268D Raw-PCA residual
frozen Raw backbone + 268D FMT residual
```

当前最重要的强对照是 Raw-PCA residual：它与 FMT residual 使用相同两阶段结构、相同
辅助维度和相同可训练参数量，只把 FMT 换成 train-only PCA 的 Raw feature。这样可排除
“提升只是 residual 训练结构造成”的解释。

每种方法只按自己的 development-validation Average Precision（AP，平均精确率）选
checkpoint；F1 阈值也只在 development 冻结；residual alpha 固定为1.0。不得使用
“必须比 Raw 高”作为模型选择条件。

### Task4：3D 涡类型多分类

目标是 streamwise、spanwise、hairpin 等类别。它与 Task3 的 IVD 二分类严格分开。
尚未建立最终标签协议，也没有正式实验结果。

### Task5：variable-scale primitive（正在进行）

Task5 是 Task3 的尺度扩展：每个 primitive 的邻居距离、RK4 积分步长和积分步数变化，
但最终仍重采样为固定 `7×32×3`。标签、分类器、268D FMT、268D Raw-PCA 对照和选择
规则沿用 Task3 3.2。详细协议见 `docs/mainExp_Task5_3D_1.1.md`。

---

## 5. 统一实验纪律

- 按时间片拆 train/development/confirmation，不随机拆空间 seed。
- 计算时间泄漏时必须包含 pathline 的未来积分窗口，而不只看 seed time。
- normalization、PCA 和其他统计量只在 train 上拟合。
- 神经网络至少3个训练 seed；当前主实验多为5个 seed。
- KMeans 固定 `random_state` 和 `n_init`。
- 报告总参数量与可训练参数量，并保留容量更大的 Raw-wide 或同构 Raw residual。
- 失败、取消、超时和负结果均保留，不覆盖旧输出。
- 结论必须指向 experiment ID、config、commit、逐次 CSV、汇总文件和设备记录。
- confirmation 一旦促使修改模型或协议，就不能继续声称是同一轮独立 confirmation；
  必须换新时间片。

当前 Task1/Task2/Task3 使用的8个 held-out 时间片没有参与对应轮次的训练和选择，但它们
此前被其他任务看过，因此是本轮 held-out confirmation，不是全项目从未查看的 sealed set。

---

## 6. 数据集与 physical family

共有10个数据条目、7个 physical family：

| Physical family | 数据条目 | 说明 |
|---|---|---|
| Channel | `channel` | steady channel VTK 经 time-dependent Killing observer pushforward 变为 unsteady field |
| Half-cylinder | `cylinder3d`, `halfcylinderRe640`, `halfcylinderRe6400` | 分别对应 Re160、Re640、Re6400 |
| Tangaroa | `tangaroa` | 单独 family |
| Delta-wing | `deltaWing_resampled`, `deltaWing_LBM` | 降采样版本与未降采样 LBM 原场 |
| F-22 | `f22raptor` | 单独 family，多个任务中的稳定困难场 |
| Boeing 747 | `boeing747` | LBM 飞机绕流 |
| Smoke buoyancy | `smokeBuoyancy` | 浮力烟流 |

主要外部数据位于：

```text
C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData3D
```

重要文件包括：

```text
channel_flow/channel.vtk
halfcylinderRe160Resampled.nc
halfcylinderRe640resampled.nc
halfcylinderRe6400.nc
tangaroa.nc
deltaWing_mag0_3reesampled.nc
LBM_3D/deltaWing_mag0_3reesampled.nc
LBM_3D/f22raptor_re400000.nc
LBM_3D/boeing747_808_2392.nc
SmokeBuoyancy80_239.nc
```

采样时间通常避开数据最开始20%和最后10%。Channel 的观察者生成代码在
`FMT_Utils/KillingObserver3D.py` 及相关 cache builder 中。

### Dataset macro 与 family macro

- Dataset macro：10个数据条目等权平均，因此3个 Half-cylinder 和2个 Delta-wing 会
  获得更大总权重。
- Family macro：先在同一 physical family 内平均，再对7个 family 等权平均。用于回答
  “跨不同物理类型是否广泛有效”时，family macro 更重要。

两者都应报告，不能只挑更高的一个。

---

## 7. FMT 表示与固定 pathline 配置

固定尺度主缓存通常使用：

```text
RK4
seed grid = 16^3
dt_scale = 0.25 source-frame interval
integration_steps = 48
sampled_steps = 32
cross offset = 0.5 × minimum spatial grid interval
maximum resampled spatial dimension = 96
```

Task1 的 family-specific FMT subset/PCA 是 development-only 选择的：

| Family/flow | FMT feature | PCA |
|---|---|---:|
| Channel | `fmt_chirality_all` | 2 |
| Half-cylinder | `fmt_all+kin4` | 8 |
| Tangaroa | `fmt_all+kin4` | none |
| Delta-wing | `fmt_real_neighbor` | 2 |
| F-22 | `fmt_all+kin4` | 2 |
| Boeing 747 | `kin2` | 2 |
| Smoke buoyancy | `fmt_all` | 2 |

Task3 使用 `all_plus_gram_kinematic`：旧 base FMT 加 time-local Gram DFT 和 pathline
kinematic DFT，总辅助维数为268。Raw-PCA residual 也严格使用268维。

关键实现：

- `FMT_Utils/DFT_FMT_3D.py`
- `FMT_Utils/FMT_3D_pipeline.py`
- `FMT_Utils/Task12Data_3D.py`
- `FMT_Utils/RawPathline_3D.py`
- `FMT_Utils/PathlineClassifier_3D.py`

---

## 8. 当前 canonical 结果

### 8.1 Task1：`mainExp_Task1_3D_3.3_reference_*`

这是在与 Task2 3.3 相同的8个 fresh held-out 时间片上重算的直接 FMT+KMeans 结果。

| Flow | FMT F1 | Raw F1 | FMT−Raw |
|---|---:|---:|---:|
| Channel | .2526 | .0638 | +.1888 |
| Re160 | .5516 | .3765 | +.1750 |
| Re640 | .5352 | .3281 | +.2071 |
| Re6400 | .5256 | .3921 | +.1335 |
| Tangaroa | .7425 | .5900 | +.1525 |
| Delta-wing resampled | .7586 | .5673 | +.1914 |
| Delta-wing LBM | .7713 | .5308 | +.2405 |
| F-22 | .3141 | .6545 | **−.3404** |
| Boeing 747 | .8403 | .4815 | +.3588 |
| Smoke buoyancy | .7718 | .5550 | +.2169 |

FMT 平均 F1 为 `.606360`，9/10 数据条目高于 Raw；F-22 是明确反例。证据：

- `outputs/mainExp_Task1_3D_3.3_reference_old8/paper_table.csv`
- `outputs/mainExp_Task1_3D_3.3_reference_new2/paper_table.csv`

允许的结论：FMT+KMeans 在当前多数3D流场中可以达到有意义的无监督涡/非涡聚类
准确率，并显著优于 Raw direct；不能写成所有流场都提高。

### 8.2 Task2：`mainExp_Task2_3D_3.3`

每个 family 的 Raw/FMT 两臂共享完全相同的 VAE。development 使用7个训练 seed 做稳健
选择，confirmation 使用新训练 seed `9068–9072` 和8个 held-out 时间片。

| Flow | Raw+VAE F1 | FMT+VAE F1 | 配对增益 |
|---|---:|---:|---:|
| Channel | .2039 | .2502 | +.0464 |
| Re160 | .5806 | .5474 | **−.0332** |
| Re640 | .4373 | .5348 | +.0976 |
| Re6400 | .5230 | .5257 | +.0027 |
| Tangaroa | .7527 | .7500 | **−.0027** |
| Delta-wing resampled | .7259 | .7793 | +.0535 |
| Delta-wing LBM | .7566 | .7892 | +.0326 |
| F-22 | .6632 | .5470 | **−.1161** |
| Boeing 747 | .8521 | .8469 | **−.0052** |
| Smoke buoyancy | .8244 | .7712 | **−.0532** |

10条目平均层级：

```text
Task1 FMT+KMeans .606360
< Raw+VAE          .631958
< FMT+VAE          .634191
```

因此 `FMT+VAE − Raw+VAE = +.002233`，但只有5/10数据条目、3/5训练 seed 为正。

允许的结论：当前冻结协议在10条目 macro-average 上支持 FMT 是略好的 VAE 输入；优势
很小，不具有逐流场或逐 seed 普适性。不得使用旧 2.3/2.4 的 `+.1562` 作为当前主结果。

证据：

- `outputs/mainExp_Task2_3D_3.3/paper_table.csv`
- `outputs/mainExp_Task2_3D_3.3/hierarchy.json`
- `config/mainExp_Task2_3D_3.3.yaml`

### 8.3 Task3：`mainExp_Task3_3D_3.2_global_ivd`

主比较为 Raw+FMT residual 与同结构、同维度的 Raw-PCA residual。10个数据条目、7个
family、5个训练 seed、每条目8个 held-out 时间片均完成。

| Flow | Raw-PCA F1 | Raw+FMT F1 | F1 gain | Raw-PCA AP | Raw+FMT AP | AP gain |
|---|---:|---:|---:|---:|---:|---:|
| Boeing 747 | .8744 | .9038 | +.0294 | .9397 | .9682 | +.0285 |
| Channel | .5618 | .7974 | +.2356 | .6727 | .8746 | +.2019 |
| Re160 | .6877 | .7670 | +.0793 | .7574 | .8595 | +.1020 |
| Delta-wing LBM | .8722 | .9203 | +.0481 | .9487 | .9754 | +.0267 |
| Delta-wing resampled | .9020 | .9335 | +.0315 | .9635 | .9816 | +.0181 |
| F-22 | .9172 | .8903 | **−.0269** | .9533 | .9404 | **−.0129** |
| Re640 | .7549 | .7577 | +.0028 | .8470 | .8421 | **−.0049** |
| Re6400 | .6714 | .7079 | +.0364 | .7553 | .7770 | +.0218 |
| Smoke buoyancy | .7915 | .8239 | +.0324 | .8825 | .9124 | +.0300 |
| Tangaroa | .7850 | .8187 | +.0337 | .8545 | .8884 | +.0339 |

汇总：

| Aggregate | Raw-PCA F1 | FMT F1 | F1 gain | Raw-PCA AP | FMT AP | AP gain |
|---|---:|---:|---:|---:|---:|---:|
| Dataset macro | .7818 | .8320 | **+.0502** | .8575 | .9020 | **+.0445** |
| Family macro | .7888 | .8436 | **+.0548** | .8636 | .9127 | **+.0490** |

F1 在9/10条目、6/7 family 为正；AP 在8/10条目、6/7 family 为正。F-22 是稳定
反例；Re640 F1 置信区间跨0且AP略降。

允许的结论：FMT 在当前多数3D flow及 macro-average 上提高了强 Raw-PCA residual 的
IVD-p95监督识别性能；不支持“每个flow都提高”。

证据：

- `outputs/mainExp_Task3_3D_3.2_global_ivd/final_confirmation/paper_table.csv`
- `outputs/mainExp_Task3_3D_3.2_global_ivd/final_confirmation/per_run.csv`
- `outputs/mainExp_Task3_3D_3.2_global_ivd/final_confirmation/per_slice.csv`
- `docs/mainExp_Task3_3D_3.2_global_ivd.md`
- 结果归档 SHA-256：`583ec77ca1e3c355b86de50987cc4548d9a557c2a4aad0291358b10fe0a4c040`

### 8.4 Task4

尚未开始。不要把 Task3 的结果写成 streamwise/spanwise/hairpin 分类结果。

### 8.5 Task5 实时状态

截至本文档时间：

- `50892842[0-1]` baseline：new2 child 已结束，old8 child 仍在 V100 上运行；
- `50892843[0-19]` FMT/Raw-PCA residual shards：等待 baseline dependency；
- `50892844` merge：等待；
- `50892845` confirmation：等待。

未得到最终 confirmation 前不得写 Task5 方法结论。最新状态必须查询
`docs/ibex_run_registry.md` 和 Ibex `squeue/sacct`，不要只依赖本段快照。

---

## 9. 论文可视化现状

Task1 图包括 flow/pathline、FMT cluster、预测与 IVD 对照。Task2/Task3 使用无标题、无
图例的固定三联图，保留物理 bounding box、坐标名称和刻度。三栏顺序：

```text
Task2: IVD-p95 ground truth | Raw+VAE | FMT+same VAE
Task3: IVD-p95 ground truth | Raw-PCA residual | Raw+FMT residual
```

所有图使用固定 held-out ordinal 4；Task2 seed 9068，Task3 seed 40。没有为了图像好看
更换时间片。单片可能与多 seed、多时间片主表方向不同，这是正常现象，不能用单片图
替代性能表。

代码与输出：

- `Visualize_Task1_3D_PaperCandidates.py`
- `Visualize_Task1_3D_Horizontal.py`
- `Visualize_Task23_3D_Horizontal.py`
- `outputs/Task1_3D_horizontal_clean_1.1/`
- `outputs/Task2_3D_horizontal_main_3.3/`
- `outputs/Task3_3D_horizontal_main_3.2/`

Task2/Task3 共20张360 DPI PNG，20/20 通过 PIL 完整性检查。预测 archive SHA-256：
`da6171fba55ebfbe795635ece1de4e9447f1a1100abd5d670f4423eb08e5b0a9`。

---

## 10. 代码入口与复现命令

### Task1 3.3 reference

```bash
python Run_Task1_3D_Main.py --config config/mainExp_Task1_3D_3.3_reference_old8.yaml
python Run_Task1_3D_Main.py --config config/mainExp_Task1_3D_3.3_reference_new2.yaml
```

### Task2 3.3

```bash
python Run_Task2_3D_Main.py \
  --config config/mainExp_Task2_3D_3.3.yaml \
  --group halfcylinder
```

合法 group：`channel, halfcylinder, tangaroa, deltaWing, f22raptor, boeing747,
smokeBuoyancy`。

### Task3 3.2 evaluation-only

在标签、缓存与 checkpoint 已存在时：

```bash
python Evaluate_Task3_MainTable.py \
  --config config/mainExp_Task3_3D_3.2_global_ivd_evaluate.yaml
```

### Task2/Task3 三联图

Ibex 只生成 prediction artifacts，本地读取原始数据坐标后渲染：

```bash
python Visualize_Task23_3D_Horizontal.py \
  --tasks task2 task3 --predictions-only

python Visualize_Task23_3D_Horizontal.py \
  --tasks task2 task3 --dpi 360 --render-only
```

### Task5

```bash
python Build_Task5_Multiscale_Cache.py \
  --config config/mainExp_Task5_3D_1.1.yaml

python Evaluate_Task5_Multiscale.py \
  --config config/mainExp_Task5_3D_1.1_evaluate.yaml
```

Ibex 环境通常为：

```bash
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
```

CUDA module 在不同节点可能不存在，因此现有脚本通常使用：

```bash
module load cuda/11.8 2>/dev/null || true
```

---

## 11. 仓库与结果文件导航

| 路径 | 用途 |
|---|---|
| `docs/research_tasks_and_protocol.md` | Task1–Task5 唯一正式定义 |
| `docs/experiment_log.md` | 方法级结论的唯一总表 |
| `docs/ibex_run_registry.md` | 所有 Ibex job、设备、时间、结果与失败记录 |
| `docs/Table1_pyflowvis_review.md` | PyflowVis 历史项目与复制边界 |
| `docs/first_principles_analysis.md` | 旧代码第一性原理审计和已知风险 |
| `FMT_Utils/DFT_FMT_3D.py` | 3D FMT 主实现 |
| `FMT_Utils/FMT_3D_pipeline.py` | 3D seed、primitive、IVD/FMT pipeline |
| `Run_Task1_3D_Main.py` | Task1 主入口 |
| `Run_Task2_3D_Main.py` | Task2 主入口 |
| `Verify_Task3_FMTClassifier.py` | Task3 Raw/Raw-wide 训练 |
| `Verify_Task3_FMTResidual.py` | Task3 FMT/Raw-PCA residual 训练 |
| `Evaluate_Task3_MainTable.py` | Task3 冻结主表评估 |
| `Build_Task5_Multiscale_Cache.py` | Task5 variable-scale cache |
| `Evaluate_Task5_Multiscale.py` | Task5 confirmation |

---

## 12. 已知反例、旧结论和易踩坑

1. **F-22 是稳定困难场。**Task1 中 FMT 显著低于 Raw；Task2 3.3 中
   `FMT+VAE−Raw+VAE = −.1161`；Task3 3.2 中 FMT 对 Raw-PCA 的 F1/AP 也为负。
   不得隐藏。可能原因包括尺度、积分窗口或 FMT subset 不适合，但尚无冻结诊断结论。
2. **Task2 当前证据很弱。**3.3 的 macro gain 只有 `+.002233`，仅5/10 flow 为正。
   不能继续引用旧 2.3/2.4 的大增益作为主结论。
3. **Task3 强烈依赖标签定义。**局部 IVD 的3.1下，FMT 对 Raw-PCA F1 平均为
   `−.0009`；改为统一 whole-field IVD p95 并全部重训后，3.2变为 `+.0502`。必须
   新旧并列说明，不能静默改写历史。
4. **README 有滞后。**当前 `README.md` 仍把 Task3 3.1 写成论文主表，并没有完整反映
   Task5。以 `docs/research_tasks_and_protocol.md` 和本文档为准。
5. **旧论文汇总表的 Task2 有滞后。**`docs/paper_tables_task123_3d.md` 的 Task2 部分
   仍是2.3+2.4；当前 canonical Task2 是3.3的 `outputs/.../paper_table.csv`。
6. **输出不等于 Git 跟踪内容。**大量 cache、checkpoint、PNG 和历史脚本未被 Git
   跟踪。新机器仅 clone 仓库不会得到全部结果，需从本地输出或 Ibex 工作目录复制。
7. **绝对路径跨平台。**cache metadata 中常含 Windows OneDrive 路径；Ibex 的精简包
   不含原始 flow 文件。模型预测可在 Ibex 做，依赖原始坐标/几何的最终渲染通常在本地做。
8. **不要把单片可视化当汇总证据。**固定 ordinal 4 的某张图可能是负增益片，但主表
   是5个训练 seed和8个时间片的汇总。
9. **不要重新引入旧 DCT/FMT bug。**历史上已修复只保留单侧频率导致旋向盲区的问题；
   相关测试与修复见 commit `bd55d45`。
10. **不要删除 dirty worktree 文件。**它们可能属于用户或并行 AI session。

---

## 13. 建议的下一步，但必须先由用户确认

1. 继续监控并完成 Task5 1.1；在 final confirmation 结束前不解释中间训练结果。
2. 更新 `README.md` 和 `docs/paper_tables_task123_3d.md`，使其指向 Task1 3.3、
   Task2 3.3、Task3 3.2；更新时保留旧版本修订记录。
3. 如果论文要强化 Task2，应优先做预先注册的新 held-out 时间片确认或解释 F-22、
   Smoke、Re160 的失败；不能在现有 confirmation 上继续调 VAE 后仍称独立确认。
4. 若诊断 F-22，应只在 development 上搜索空间尺度、积分步长/步数、FMT block；之后
   必须换新的 confirmation 时间片。
5. Task4 开始前先冻结类别来源、background 处理及跨数据集类别映射，不能直接复用
   Task3 标签。

---

## 14. 一句话交接结论

当前最稳固的论文证据是 Task3：统一 whole-field IVD p95 后，FMT residual 相对同构
Raw-PCA residual 在10个数据条目的 dataset-macro F1/AP 上提高 `+.0502/+.0445`，
大多数 flow 为正但 F-22 是反例。Task1 也在9/10条目优于 Raw。Task2 只得到很小的
macro-average优势 `+.002233`，逐场并不普适。任何后续工作都必须保留这三个强弱不同的
证据边界，不能把它们写成同等强度的“普适提升”。
