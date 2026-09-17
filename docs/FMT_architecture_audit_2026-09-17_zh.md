# FMT 中心选择、傅里叶操作与完整网络审计

审计版本：`Verify_FMTArchitectureAudit_1.1`。审计起点：`5ab735c0412ad19fb7539e6ccada416a3262ac3c`。日期：2026-09-17。本次只审查结构和证据，不训练、不重新排序方法、不改写历史指标。

**必须纠正的结论：p35 是一个局部特征池化配方，不能单凭这个名字判断是否轮换中心或是否使用卷积。旧 Task3/5 的 p35 固定中心，但完整预测模型有三层 Conv1d；Task4-c 的 p35 对每条有效线轮换中心，但 FMT 分支没有 ConvNd。c156 和目前的 asap_fmt 都轮换中心，没有 ConvNd。**

此前把“编码函数无卷积”说成“整个方案无卷积”、把“局部函数固定第零条线”说成“整个 primitive 只算一次中心”，都是错误。下面逐层分开说明。

## 1. 审查范围和证据强度

| 范围 | 已完成内容 | 不能据此声称的内容 |
|---|---|---|
| 当前工作树、全部本地可达 Git 历史、两个研究归档 ZIP | 4,851 条源文件版本记录，3,379 份不同字节内容；Python 语法解析错误 0 | 不是说逐行执行了所有历史程序 |
| 配置 | 993 条配置版本，573 个不同配置路径；每条保存来源、提交、SHA-256 和完整配置 | 配置存在不等于实验执行或完成 |
| 运行登记 | 原样提取 2,980 条调度表记录；包括失败、取消、预检和汇总 | 不是 2,980 次成功训练 |
| 本地结果目录 | 登记 510 个目录 | 空目录、预检目录不算成功实验 |
| Ibex 部署副本 | 198 个 `FMT*` 部署目录，指定源码/配置子目录中 71,744 条文件记录；哈希逐份比较 | 不覆盖已删除且没有备份的远端文件，也不保证任意其他目录没有私人副本 |
| Ibex 与本地差异 | 初始 510 份不同内容；503 份仅换行不同；7 份早期客观性实验修改另存并人工检查 | 不把原始字节哈希不同直接称为算法改变 |
| 实际构造与合成输入核验 | 15 种代表网络的完整模块树；固定/轮换中心实际调用计数；频域乘法与循环卷积的数值比较 | 没有重新执行每个历史随机种子的训练，不据此复核旧 F1 |

完整清单：

- [完整审计证据包（ZIP）](../outputs/Verify_FMTArchitectureAudit_1.1_evidence.zip)，包括以下原始清单、哈希、核验记录和远端独有源码快照。
- [573 项配置逐行索引](FMT_experiment_architecture_catalog_2026-09-17.md)。结构栏按审查过的共享调用族归类；同一实验里的不同分支分别说明。
- [全部配置版本及完整内容](../outputs/Verify_FMTArchitectureAudit_1.1/configuration_inventory.csv)。同一路径被覆盖的历史设置也保留。
- [源码版本清单](../outputs/Verify_FMTArchitectureAudit_1.1/source_inventory.csv)、[卷积调用位置](../outputs/Verify_FMTArchitectureAudit_1.1/convolution_calls.csv)、[中心选择候选位置](../outputs/Verify_FMTArchitectureAudit_1.1/center_candidates.csv)。后者是静态定位结果，不把正则命中当作执行证明。
- [全部 p00–p49 配方](../outputs/Verify_FMTArchitectureAudit_1.1/pooling_recipes.csv)、[全部 c000–c263 配方](../outputs/Verify_FMTArchitectureAudit_1.1/task4c_all_264_candidates.csv)。由原候选生成函数直接导出。
- [完整模块树和中心调用计数](../outputs/Verify_FMTArchitectureAudit_1.1/architecture_checks.json)、[Ibex 比较摘要](../outputs/Verify_FMTArchitectureAudit_1.1/ibex_reconciliation_summary.json)、[Ibex 全部文件记录](../outputs/Verify_FMTArchitectureAudit_1.1/ibex_source_inventory.json)。
- [全部原始调度表记录](../outputs/Verify_FMTArchitectureAudit_1.1/scheduler_record_inventory.csv)。历史已删除的卷积代码从 `52f6284c` 等 Git 版本审查，不能只搜索当前删除后的文件。

## 2. 从几何定义区分六种操作

设一个 primitive 有曲线 `x_i(t)`，原中心线是 `i=0`，其他为邻居。下列操作不能混称为“换中心”。

| 操作 | 数学/代码含义 | 是否在同一个 primitive 内轮换中心线 |
|---|---|---|
| 整体归一化 | 所有点减同一个质心，再除同一个最大/RMS 半径 | 否 |
| 固定中心相对位置 | `r_j(t)=x_j(t)-x_0(t)`；一次编码得到一个向量 | 否 |
| 邻居排序或池化 | 固定中心不变，对六个邻居的同一种描述量求均值/最大值等 | 否 |
| 两两距离/夹角 | 一次构造无序点对距离；或在固定中心测邻居夹角 | 不自动意味着轮换 |
| 相机旋转 | 固定原中心的平移，估计随时间变化的刚体旋转 | 否 |
| 外层逐线重新编码 | 对每个 `i` 构造 `[x_i, x_j1,…,x_j6]`，让 `i` 进入局部编码器的 slot 0；输出 `N×D`，再跨 `N` 条线池化 | **是** |

外层即使向量化成 `arange→gather→reshape`，没有显式 `for i`，也仍是轮换。对有效线数 `N` 的一束线，其 FMT 中心编码次数是 `N`，不是 1。不同训练样本各自选择播种中心，则是产生不同 primitive，另当别论。

共享逐线多层感知机（Multilayer Perceptron，MLP：由全连接层与非线性组成的网络）不自动构造中心—邻居关系；必须查看输入是原线坐标，还是重新以该线为中心计算的 FMT。`raw_c156` 属于前者，`fmt_c156` 属于后者。

实际合成输入计数：

| 被测真实入口 | 输入 | 底层中心编码数 | 输出形状 |
|---|---|---:|---|
| `FMT_P35_NormFrequency_3_1.encode(seeds=None)` | 2 个七线 primitive | 2 | `[2,141]` |
| 同函数，提供 Task4-c seeds/counts | 2 束，10/8 条有效线 | 18 | `[2,10,141]`，含填充位置 |
| `fourier_tokens` | 同上 | 18 | `[2,10,142]`，最后一维是有效性标记 |
| `ASAPFMT_Task35.encode(arm='fmt_c156')` | 2 个七线 primitive | 14 | `[2,7,142]` |
| 同函数，`arm='asap_fmt'`，先真实执行相机 | 2 个七线 primitive | 14 | `[2,7,142]` |

141 是**每个中心**的特征宽度，不保证每个原始 primitive 只有 141 个输入数。七中心 c156 在池化前实际有 `7×141` 个特征数。

## 3. 局部 Fourier 描述到底计算什么

离散傅里叶变换（Discrete Fourier Transform，DFT）把沿采样序列的信号转换成复数频率系数；代码多数使用 `rfft`。本节只描述局部编码器，完整网络见下一节。

令保留频率数 `K=6`。旧三维向量频谱描述 `V(a)` 对每个复数三维系数的实部/虚部计算长度和夹角，再加相邻频率的有符号三重积项，宽度 `3K+(K−1)=23`。标量频谱 `S(s)` 保留 `K` 个实部与除直流外 `K−1` 个虚部，宽度 11。其确切归一化和符号规则以函数实现为准。

| 名称/版本 | 局部输入与计算 | 宽度（K=6） | 局部中心策略 | 主要实现 |
|---|---|---:|---|---|
| 原 `pathline_dft_features_3d`，常简称 FMT/old_fmt | `V(Δx_0)`；六条 `V(Δ(x_j−x_0))`；邻居描述逐列排序并拼接 | 23+138=161 | 一次固定 slot 0 | `DFT_FMT_3D.py` |
| magnitude / 去 chirality 子配置 | 分别只取频谱模，或删除相邻频率三重积项；其他几何输入同上 | 42 / 126 | 固定 | 同文件，配置 mode/include_chirality |
| FMT 子块选择 | 只保留中心/邻居/部分描述列；不是重新选择中心线 | 依配置 | 固定 | `fmt_feature_indices` 等调用 |
| `fmt_all_v2` / time-local Gram | 六条相对位置的同时间 Gram 矩阵，取含对角的 21 项；按初始尺度归一化、减初始矩阵后做标量 DFT | 21×11=231 | 固定原中心 | `FMTAllV2_3D.py`、`DFT_FMT_3D.py` |
| LargeNeighbor | 原中心周围三层、24 邻居；Gram 上三角 300 项的标量频谱 | 3300 | 固定；25 条线不代表25次轮换 | `LargeNeighbor_3D.py` |
| `objective_fmt_nTDO` 1.1 | 六条相对位置**不做时间差分**的向量频谱 138；全部21对同时间点距离频谱231 | **369** | 固定原中心；距离块覆盖所有点对 | `objective_fmt_nTDO.py` |
| `objective_fmt_nTDO_v2` 2.1 | 只保留21对同时间距离频谱 | 231 | 无逐中心重复编码 | `objective_fmt_nTDO_v2.py` |
| angle block | 以固定中心为顶点，六邻居共15种夹角；每条角度序列做标量频谱 | 165 | 固定 | `FMT_Angles_3D.py` |
| `fmt_v5` 5.1 | 原161 + 角度165 | 326 | 固定 | `fmt_v5.py` |
| `fmt_objective_ntod_v2` 2.2 | 距离231 + 角度165；注意与前一大小写相似的名字不同 | 396 | 固定 | `fmt_objective_ntod_v2.py` |
| V6 | 原161 + 中心坐标/单位切向量的实虚频谱72 | 233 | 每次局部描述固定；Task4-c包装器逐线调用 | `Task4C_FMTv6v7_4_20.py`、跨任务版本调用 |
| V7 | 中心23 + 坐标/切向72；删除邻居138 | 95 | 同上 | `Task4C_FMTv6v7_4_20.py`、跨任务版本调用 |
| V8.1 | 中心23 + 方向72 + 从整个138维邻居块选绝对值最大4个数（保留符号）及整体均值 | 100 | 固定 | `fmt_v8.py` |
| **p35** | 中心23 + 方向72 + 六邻居每一种描述的均值23 + **数值最大值**23 | **141** | **局部固定；外层是否重复由调用者决定** | `FMT_V8_Search_2_1.py` |
| p35/n0_k06 | 同上；统一整样本最大半径归一化，6频率；训练集逐特征标准化 | 141 | 同上 | `FMT_P35_NormFrequency_3_1.py` |
| signed geometry FFT | 中心差分与相对位置差分的3个坐标系数，保留实部和虚部，不做23维压缩/邻居排序 | 7×3×11=231 | 固定 | `Task6Recovery_3D.py` 等 |
| kin4 / aivd 派生 | 对立邻居构造局部矩阵，用时间导数和伪逆估计速度梯度，再算涡量偏差、应变、散度、Q类量；可拼频谱/时间窗口统计 | 随配方 | 固定几何中心 | `Task12Data_3D.py`、`PathlineClassifier_3D.py` 等 |

注意：旧函数默认 `neighbor_scale=100, neighbor_weight=.5`；Task3/5 的一些原缓存显式用 `1,1`。Task4-c 4.3 的 `line_fmt` 显式用 `100,.5`，后来的 n0/c156 显式用 `1,1`。因此“同样141维”仍不能证明数值预处理相同。p35/h0 原搜索复用历史161缓存，而 n0 搜索重新统一几何归一化；此差别已经逐项保存在配置中。

方向72是中心坐标及单位切向的六个分量，各6个频率的实虚部，即 `6×6×2`。这不是两两距离特征。输入同时间相对位置是客观几何向量；对其跨时间坐标分量作普通 DFT 的输出是否客观，必须另行判断。nTDO1.1 的距离块在时变刚体变换下不变，不代表其向量频谱块也不变。

## 4. p00–p49、h0–h3、n0–n3、c156 各指什么

这些名字分属不同层，不能互换：`pXX` 定义邻居描述怎么池化，`hX` 定义训练/投影设置，`nX_kYY` 定义归一化/频率，`cXXX` 是一次整套候选搜索的编号。

| 配方编号 | 对原138维邻居块的操作 | 与中心轮换的关系 |
|---|---|---|
| p00–p09 | 全138个数按绝对值选前k，保留符号，再加全体均值；k顺序为4/1/8/16/24/32/48/64/96/138 | 都不决定外层中心数 |
| p10–p19 | 取最大/最小有符号极值组合，再加均值；k顺序同上 | 同上 |
| p20–p29 | 按绝对值选前k，在原138维位置保留，其他置零 | 同上 |
| p30/p31/p32/p33/p34 | 六邻居逐特征 mean/max/min/std/RMS | 同上 |
| p35/p36/p37 | mean+max / mean+std / min+max | 同上 |
| p38/p39 | mean+max+std / mean+min+max+std | 同上 |
| p40–p45 | 保留逐特征排序后的秩：[0]、[0,5]、[0,2,5]、[0,1,4,5]、[0,1,2,4,5]、全部 | 同上 |
| p46–p49 | 每一种描述独立选绝对值前1/2/3/4名，保留符号 | 同上 |

每个 p 配方最终加中心23和方向72，完整宽度是95加该池化宽度。逐特征排序不保持一个完整邻居的所有23个描述为一组；p35的均值/最大值对这种排序不敏感。

| 设置 | 实际含义 |
|---|---|
| h0 | 继承冻结的训练、投影和网络设置；不是“无隐藏层”，不是“无卷积”开关 |
| h1 | 更强 dropout/weight decay |
| h2 | SiLU 激活、较低学习率等指定设置 |
| h3 | 中心/方向/池化邻居分别学习投影；仍须检查完整外层网络 |
| n0 | 最大半径归一化 + 训练集逐特征 z-score |
| n1 | 最大半径 + signed-log、z-score、截断到±8 |
| n2/n3 | 将 n0/n1 的最大半径换为均方根半径 |
| k02/k04/k06/k10/k16 | 保留频率数；p35宽度公式 `24K−3` |

### c156 的完整定义

来源：`Ablation_Task4C_FPSAugmentSearch_1.2` 的选定记录和 `independent_final_audit.json`；其科学执行代码包括 `9d05abc5`，最终1.2记录 `21cf1fb7`。具体操作：

1. 每条线按弧长均匀采48点，不加训练增强。
2. 全束平移到质心、按最大半径归一化。
3. **每条有效线轮流成为中心**；在播种点上用最远点采样（Farthest Point Sampling，FPS：逐次选距已选点集最远的候选）取六个邻居。它不是“最近六邻居”。
4. 每个中心算 p35 的141维，频率6，邻居 scale/weight 为1/1；训练集特征标准化。
5. 共享逐中心 `141→256` 全连接层和三个残差块；每块内部 `256→512→256`。
6. 跨所有中心 mean/max，得到512维；分类头 `512→256→128→2`。
7. 总参数 **992,386**。无 Conv1d/2d/3d，但明确使用多中心表示。

原 Task4-c p35/h0 小网络是32点、每中心**最近六邻居**、141维、**76,738参数**。它和 c156 都轮换中心；差别包括邻居选择、采样点数和下游网络，绝不能用“p35不轮换，c156才轮换”概括。

| 三套容易混淆的Task4-c输入配方 | 邻居差分倍率/特征权重 | 特征标准化 | 每束中心数 |
|---|---|---|---|
| 原p35/h0，包括最新P35GTHead复跑 | 100 / 0.5 | signed-log → 训练集z-score → 截断±8 | 有效线数N |
| p35/n0_k06 | 1 / 1 | 训练集z-score | 有效线数N |
| c156的p35/h0 | 1 / 1 | 训练集z-score | 有效线数N |

因此c156中的`h0`也不意味着恢复原p35/h0缓存的倍率和signed-log。证据分别为`Verify_Task4C_P35GTHead_1.1.json`的encoding字段、`FMT_P35_NormFrequency_3_1.py`和`fourier_tokens`/`fit_normalizer`。

全部264个候选都是多中心/FPS6；六类网络为 original、wide_residual、attention_pool、set_attention、fps_graph、block_branches。`fps_graph` 在邻居边上计算共享 MLP(`[h_i,h_j−h_i]`) 并做 max 聚合，两轮更新；这是图消息传递/EdgeConv式操作。**即使没有 `nn.ConvNd`，也不能把它称为“只有独立MLP和最终池化”，或直接认定符合禁止空间卷积的限制。** 所有候选在CSV中单独标出此项。c156使用wide_residual，不使用fps_graph。

## 5. 按完整实验管线判定

### Task3 / Task5

| 方案/实验族 | Fourier几何中心 | 完整预测路径 | 卷积判定 |
|---|---|---|---|
| 早期 raw_fmt 分类，161或kin增强特征 | 原中心一次 | 原轨迹逐线时序编码 + FMT全连接投影 → 分类头 | **有三层Conv1d** |
| Raw+FMT residual，各搜索/优化/确认版本 | 原中心一次 | 冻结Raw网络的logit + α×FMT残差；残差可另拼Raw几何表示 | **有三层Conv1d；冻结不等于删除** |
| nTDO1.1 369、距离版2.1、角度版2.2、fmt_v5 | 原中心一次 | 接上述Raw+FMT残差体系 | **有三层Conv1d** |
| V6/V7/V8.1、p00–p49、p35/h0、n0–n3各频率 | 原中心一次 | 接上述Raw+FMT残差体系；142,914是常用总参数，不是纯MLP的证明 | **有三层Conv1d** |
| residual_input=`fmt_only` | 原中心一次 | 只限制残差头输入，最终仍加Raw logit | **仍有三层Conv1d** |
| 近期 `fmt_c156` | **七条线分别作中心** | 每中心p35 → wide_residual → 跨中心mean/max → MLP | 无ConvNd；有中心轮换 |
| 近期 `asap_fmt` | 相机固定原中心；**后续编码七中心** | ASAP相机 → 每线弧长48点 → 与fmt_c156相同的编码和分类 | 无ConvNd；有中心轮换 |
| `raw_c156` 对照 | 不做重新中心化的FMT | 每线48×3原坐标 → 共享网络 → 跨线mean/max → MLP | 无ConvNd；也不含FMT |
| 新 single-center 1.1 候选 | 一个固定中心 | 单个p35向量 → MLP | 无卷积；**尚无正式核验成绩** |

证据主链：`PathlineClassifier_3D.TemporalLineEncoder3D` → `PathlineGeometryEncoder3D` → `PathlineBinaryClassifier3D` / 历史 `PathlineFMTResidualClassifier3D`。卷积核分别5、5、3，默认通道3→32→64→64，沿每条曲线采样序列作用；不应冒称Conv2d/3d，也不应因它沿序列作用就隐瞒其卷积性质。

`ASAPFrame_1_1` 的相邻时刻刚体最小二乘相机：固定第零条中心的平移，用对应邻居作SVD，得到相邻时刻的 proper rotation 后累积。之后 `ASAPFMT_Task35_1_1.encode` 明确构造七行邻居索引并调用 `fourier_tokens`。相机去旋转、弧长重采样、轮换中心是三个独立步骤。最初连续 Killing 场 demo 也固定中心，但其角速度最小二乘目标与有限步SVD不是逐样本完全同一算法。

### Task4-c

以下“无卷积”只指表中FMT分支。同场比较的体素Conv3D是独立非FMT方法，必须另列，不能拼成“FMT内含Conv3D”。

| 实验版本/族 | 中心/邻居 | 特征和后续操作 | FMT完整分支卷积 |
|---|---|---|---|
| 早期 BundleQuality / PaperBaseline 1.1 | 每条有效线作中心，最近6邻居 | 每线161投影，与原坐标/切向的逐线双向LSTM相加；再束级双向LSTM和分类器 | 无ConvNd；**含Raw和BiLSTM，不是纯FFT+MLP** |
| HairpinBinary **2.1** | 七线primitive，固定原中心 | 单个161 → MLP | **无；不轮换** |
| PaperBundles **3.1** | 每条有效线作中心，最近6 | 每中心161 → 跨中心mean/max得322 → MLP | 无；**轮换** |
| Multiscale **4.1** | 同3.1 | 322和多尺度/训练设置变化 | 无；轮换 |
| Regularized **4.2** | 同3.1 | 322 + 逐线坐标/切向72的mean/std/max/min共288，合计610 → MLP | 无；轮换 |
| LinePooling **4.3** | 同3.1 | 每中心161+72=233 → 共享MLP → 跨中心mean/max → MLP | 无；轮换 |
| **4.4–4.14**：mixup、尺度、正式测试、分辨率、SAM、更多采样中心、对比损失、广义交叉熵、训练拟合、邻近采样、物理长度 | 延续逐中心编码；4.7频率改变 | 网络/优化/数据按各版本变化，未取消外层逐线中心 | 无；轮换 |
| Encoders **4.15** | 每中心最近6 | 326的fmt_v5 / 231距离 / 396距离+角度 → 逐中心MLP及mean/max；未保留旧72方向 | 无；轮换 |
| RetainDirection/Device **4.16/4.17** | 同上 | 原233；326+72=398；396+72=468 | 无；轮换 |
| Capacity **4.18** | FMT复用4.14 | 新训练的是独立Conv3D容量对照 | FMT无；对照有Conv3D |
| Blocks **4.19** | 同上 | 398内各块置零/移除信息，保持约定输入/网络 | 无；轮换 |
| V6/V7 **4.20** | 每有效线一个描述 | 233 / 95；V7不再用邻居138，但仍逐线取中心23+方向72后汇总 | 无；多个中心描述 |
| V8 **4.21**、Search **2.1/2.2** | 每中心最近6 | 100或p00–p49；p35为141，逐中心MLP和mean/max | 无；轮换 |
| NormFrequency **3.1** | 每中心最近6 | n0–n3 × 2/4/6/10/16频率；共同选定n0_k06 | 无；轮换 |
| InstanceCoverage **9.15 / v2** | 每中心最近6 | p35/n0_k06；采样/训练范围变化不改变多中心编码 | 无；轮换 |
| BottomDensity **1.1/1.2/1.3** | 每中心最近6 | 同141维小网络；Conv8/12/16/24/32等为另一个方法 | FMT无；轮换 |
| NeighborSelection **1.1** | 每中心分别用nearest6/fps6/nearest3fps3 | 改邻居规则，未改为一个中心 | 无；轮换 |
| FPSAugmentSearch **1.1/1.2** 全264配置 | 每中心FPS6 | 32/48点、均匀/曲率采样、增强、网络比较；详见264行CSV | 无ConvNd；fps_graph另有图消息传递；全部轮换 |
| 选定 **c156** / GTHeadCoverage **1.1** | 每中心FPS6 | 48点、p35/h0、992386参数wide_residual | 无ConvNd；轮换 |
| 扩充集上的原 **P35GTHead 1.1** | 每中心最近6 | 32点、141维、76738参数小网络 | 无ConvNd；**仍轮换** |
| 新 single-center **1.1** | 整束只选一次中心；其余有效线作邻居池 | 统一几何、141维、单向量MLP | 无卷积；与旧p35不同；正式训练未开始 |

`4.9 CenterDiversity` 的“更多中心”指训练播种位置增加；旧模型在此之前已对一束的全部有效线轮换。不能把4.9当作轮换开始的版本。

### 其他任务与容易混淆的旧名称

| 方案/任务族 | 实际操作 | 中心情况 | 卷积/其他空间运算 |
|---|---|---|---|
| Task1 三维固定特征聚类 | 局部DFT描述 → 标准化/PCA等 → KMeans | 固定原中心 | 无卷积 |
| 旧Task2 FMT-VAE | 固定特征 → 全连接变分自编码器（VAE）→ 潜在空间聚类 | 固定 | 无卷积；已因用户另一个决定弃用，不能混称为卷积撤回 |
| 新Task2直接可视分析 | p35等直接特征/Raw/PCA，另比较坐标重构VAE | 各特征按明确入口 | 无新增预测CNN；坐标VAE与FMT分开 |
| Task4-a及Task4-b聚类 | 固定七线特征，可能跨多个primitive统计，再KMeans | 不在同一primitive重新选中心 | 无卷积 |
| Task4-b旧raw_fmt分类 | 原轨迹卷积分支 + FMT辅助特征 | 固定 | **Conv1d×3** |
| Task4-b fmt_only；PatchSegmentation5.1、GeometrySearch5.2 | 固定特征 → 普通/残差/分块MLP | 固定 | 无卷积；这里的fmt_only与Task3残差参数不是同一含义 |
| Task6 PrimitiveVAE2.1 | 固定161 → 全连接VAE → 七线672坐标 | 固定 | 无卷积 |
| Task6 Recovery/Reconstruction/DirectNeural | 有符号频谱/Raw，线性初始化或MLP残差、几何解码器 | 固定 | 无卷积 |
| Task6/7/8 FlowMap | 固定原中心的旧FMT/kin/Gram/大邻居特征 → VAE/查询MLP | 固定 | 无卷积 |
| Task6 PNNTrans | 每个同时间点当局部中心，点对相对位置/特征差、正弦余弦嵌入、max/mean；再时间Transformer和MLP解码 | **多个点局部中心**，不是固定中心DFT | 无ConvNd；有Point-NN邻域聚合和注意力 |
| Task6 FMTGeometryMoE / Task36 MultiGate | 固定FMT专家 + Raw或Point-NN几何专家 + 学习路由/重建或分类 | FMT支路固定；Point-NN支路多个点中心 | 无ConvNd；不是“只有Fourier和池化” |
| 最初二维 `DCT_FMT` | 名字叫DCT，实际把xy写成复数，做DFT，取直流及正负频率幅值；固定五线中心与相对差分 | 固定五线中心；窗口内多个primitive另作池化 | 编码器无卷积 |
| 最初二维 `FMT_encoder.FMT`，Point-NN式 | 同时间跨线点邻域、位置正余弦、max+mean、BatchNorm/GELU | **每个同时间点作局部中心** | 本体无ConvNd，但BatchNorm含可训练参数；不是后来161维DFT |
| 上述类可选 `TemporalDFT` | rFFT → **可学习复数频率乘子** → irFFT → BN/GELU/残差 | 不新增中心 | **乘子部分等价于时间循环卷积**；此前只查ConvNd漏掉此类 |
| `PointWiseFMT_Regressor` | `pnn.EncNPNew` 点云FPS/KNN/几何嵌入 → LinearDecoder | 多个点邻域中心 | 无ConvNd；不是固定中心轨迹DFT |
| 历史四个FMT U-Net类 | FMT/Point-NN编码接空间U-Net；部分有AttentionFusion | DCT分支固定；Point-NN分支多个局部点中心 | **Conv2d和转置Conv2d** |
| FTLE Upsampling2D1.1/1.2 | fmt65 / fmt_v5为131 / 距离+角度176，特征网格接1×1Conv2d和U-Net | 五线固定中心 | **Conv2d/转置卷积** |
| FTLE P35 Fusion2.1、Scales2.2 | 2D p35为103（含尺度）、signed121、合并223、多时间窗667，再空间融合/超分网络 | 固定；早/晚/全时间窗不换中心 | **Conv2d、残差/空洞或U-Net** |
| 第三方 `pnn/models/point_pn.py` | 库中的1×1Conv1d/2d构造 | 点云多邻域 | 源码有卷积；**未在所查正式FMT调用入口找到实际导入它的证据** |
| Task4-c TangentCurvature/PointNN/BiLSTM/PointNet/PointNet++ 基线 | 独立非FMT方法；当前PointNet实现用Linear，PointNet++有FPS/球邻域 | PointNN/PointNet++多个点邻域；不称FMT轮换 | 核查实现无ConvNd；含各自点运算/LSTM，不因模型名称推断卷积 |
| Task4-c独立体素、二维Raw U-Net/ESPCN | 独立非FMT卷积对照 | 不适用 | 有明确Conv3d或Conv2d；与FMT结果分列 |
| Vatistas解析拟合/流场生成、GT体素化、查看器 | 解析参数优化、标签/数据操作、展示 | 不适用 | 不据patch/grid等词判断存在预测卷积 |

旧二维FMT U-Net具体类名：`FTLEUpsamplingFMT_Unet`、`FTLEupsamplingFMT_UnetV2`、`FTLEupsamplingFMT_UnetV3`、`FTLEupsamplingDCT_FMT_UnetV2`；`AttentionFusion`还有1×1 Conv2d。这些历史实现已在此前删除/拒绝入口的提交中处理，Git审计保留其原结构。

`TemporalDFT` 的正式运行证据必须谨慎：现有二维聚类入口明确 `temporal_head=None`，核查的调用历史也未提供“正式训练启用dft头”的证据；旧单元测试会构造这个头。**确认源码提供了卷积等价模块，不等于确认某个已发表/已报告成绩使用了它。** 本次合成数值检查仅证明数学等价，最大误差约 `8.9×10⁻¹⁶`。

本次依照已有禁令补漏：已移除 `TemporalDFT` 的可执行滤波实现及 `fps_graph` 的图邻域实现，旧构造入口明确报错；不自动换成另一种网络。原始结构以 `5ab735c0` 留存，264候选清单描述历史定义，图分支标为撤销。普通p35小网络和c156的网络权重初始化/输出不因这一处理改变。其中心轮换仍照实记录，不因此变成单中心。

## 6. 卷积检查为什么必须超过搜索 Conv2d/Conv3d

完整预测路径包括预处理后的可学习空间运算、Raw分支、冻结网络、残差logit来源和分类头。`requires_grad=False`、只训练MLP、当前分支名叫fmt_only，都不足以证明无卷积。

本次同时检查了构造模块树、历史源码调用、邻域MLP/注意力和FFT逆变换操作。必须分别记录：

- 标准卷积：Conv1d/2d/3d、转置卷积、functional conv。
- 卷积等价频域滤波：`IFFT(W·FFT(x))` 的可学习平移不变滤波。
- 图消息传递/EdgeConv式邻域网络：不能仅凭类名为Linear就纳入“纯MLP”范围。
- 固定DFT特征、普通逐点MLP、有限差分几何导数、可视化平滑：分别报告其实际用途，不把一切线性运算都泛称违规CNN。

`assert_no_fmt_convolution` 递归模块检查能发现所有登记的ConvNd，包括冻结分支；它不是上述全部操作的自动数学证明。此前将它视为充分检查漏掉了 `TemporalDFT`，本报告明确更正。

## 7. 历史说法与此次纠正

| 之前错误/不完整说法 | 核查后的明确说法 | 原因与证据 |
|---|---|---|
| “p35不轮换，c156轮换” | **Task3/5旧p35不轮换；Task4-c旧p35和c156都轮换** | 漏查了`line_fmt`/`encode(seeds=...)`外层 |
| “p35有卷积” | **旧Task3/5完整p35模型有Conv1d；Task4-c p35的FMT分支无ConvNd** | 将任务中的完整网络与局部特征名字混为一谈 |
| “nTDO/141维描述没卷积，所以结果是Fourier+MLP” | 描述本身无卷积；**旧Task3/5成绩来自Raw卷积+描述的完整模型** | 历史模型模块树含3层Conv1d |
| “asap_fmt固定相机中心，所以没有中心轮换” | 相机固定；**相机输出之后c156再作七中心编码** | `ASAPFMT_Task35.encode`中的邻居索引及真实调用计数 |
| “没有ConvNd就一定满足禁止空间卷积” | 还须审查频域滤波、图消息传递等实际操作 | TemporalDFT与fps_graph是具体反例/边界 |
| “p35/h0就是同一套完整模型” | p35是141维配方；h0继承所在实验的网络/训练。必须附任务、中心规则、邻居规则、网络版本 | 同名在Task3/5、Task4-c小网络、c156的完整路径不同 |

这些修正不改变原始预测数值，但改变其可以支持的科学结论。**旧Task3/5含卷积成绩不能证明纯FMT+MLP的性能；旧Task4-c多中心成绩不能证明单中心FMT的性能。现有证据不足以回答“单中心、无卷积、统一FMT在Task3/5/4-c哪个最好”。** 不能把原冠军名字换个标签就当作已验证。

## 8. Ibex 独有修改及当前停止状态

7份字节内容确有不同的远端源码来自两套早期 `FMT_ObjectivityMechanism_20260909` 部署：

| 文件 | 数量 | 与当前源码比较的实质差异 | 本次结构判定 |
|---|---:|---|---|
| `Task12Data_3D.py` | 1 | 没有后来加入的longtime窗口配方 | 未引入中心轮换/卷积 |
| `ObjectivityMechanism_3D.py` | 2 | 1.1沿用100/.5邻居尺度；1.2改1/1；没有1.3显式区分缓存/重算的两个对照分支 | 固定原中心；不改下游网络结构，继承相应任务原模型 |
| `Verify_FMTObjectivityMechanism_1_1.py` | 2 | 默认版本、缓存与重算一致性检查、事件文本不同 | 未引入新的中心或卷积 |
| `Submit_FMTObjectivityMechanism_1_1.py` | 2 | 提交配置1.1/1.2和登记文本不同 | 调度脚本，不是特征/网络 |

原文、SHA-256、远端路径和逐文件差异保存在审计输出，不执行这些历史副本。

之前刚准备的单中心候选只有预检作业 **52008644** 被提交。它在导入时因部署缺少 `FLowUtils` 而 `FAILED 1:0`，没有进入正式训练；计划中的21次拟合未提交。此次按用户要求转为全面审查，已检查队列为空，**不自动修复后继续训练**。因此没有可以报告的新单中心F1。

今后每个FMT结果必须同时写明：几何输入、中心编码次数/策略、邻居选择、变换前信号、频率与归一化、邻居池化、跨中心池化、完整网络及所有旁路、是否有图邻域/频域滤波、代码和配置提交。禁止再用一个p35/c156简称代替完整方法说明。

## 9. 本次收尾核验

24项相关测试通过，包括禁止入口、冻结旁路卷积检查、p35/c156与历史版本相同初始化下参数及输出逐值相同、显式无频域头的早期FMT、单中心构造；15种历史/当前代表网络已构造检查，五组中心调用计数通过。测试出现一条既有NumPy扩展二进制尺寸警告，没有失败。记录为`policy_tests.xml`及`architecture_checks.json`。

这次封禁不改写历史训练结果，不恢复暂停实验。573配置索引按配置路径列举，详细定义保留993条版本；静态命中与正式运行证据有不同强度，报告没有把它们混为“所有历史任务都重新跑过”。
