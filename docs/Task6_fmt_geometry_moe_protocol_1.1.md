# Task6：原 FMT 频域特征与几何特征的混合专家，1.1

2026-09-12，用户授权实现并在 Ibex 的每流场 3072 条训练 primitive 上比较。实验版本：`Verify_Task6_FMTGeometryMoE_1.1`。本次 Task6 是单流场、七条路径线 primitive 的自监督重建；输入整个 primitive，输出同一个 primitive 的全部轨迹，不涉及未输入查询粒子、IVD 分类标签或跨流场共享训练。

## 1. 问题与证据边界

目标是检验：在保留几何分支的情况下，原 FMT 对形状变化的概括是否能改善重建，而不是要求仅由有损 FMT 恢复所有坐标。

完整、保留实部和虚部的离散傅里叶变换（DFT）可逆；不能把“频率域”本身等同于信息丢失。本实验的原 `fmt_all` 包括频率截断、几何量提取和邻居排序，确实不是完整可逆 DFT。也不能把此前不同任务的所有配方都说成这个相同的 161 维配方：这里严格沿用 Task6 既有 `fmt_tokens` 对原 `fmt_all` 的具体调用。

混合专家（Mixture of Experts, MoE）是由输入决定各专家权重的网络。参考 [Jacobs 等，1991，Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) 的输入相关门控思想；本实验在隐向量上做连续加权，再接共享解码器，不复现该论文的概率似然，也不使用离散 top-k 路由。

截至本版本制定，`Verify_Task6_PNNTrans_1.3` 的九流场开发集平均位置误差为 Point-NN 0.706686、同规模 Raw Transformer 0.566087、冻结 Raw Transformer 0.460250。上述是验证误差，不是测试误差。原来最强的 Raw 对照包含 Transformer，并不是纯全连接网络；本次纯全连接几何分支是新候选，不能预先称为“已证明更好”。

## 2. 冻结数据和指标

- 九流场：cylinder3d、halfcylinderRe640、halfcylinderRe6400、tangaroa、deltaWing_resampled、deltaWing_LBM、f22raptor、boeing747、smokeBuoyancy。
- 每个网络只训练一个流场。开发种子 94110，最终独立训练种子 94111、94112、94113。每次取原冻结随机索引序列的前 3072 个训练样本；必须与 1.3 基线的索引 SHA256 一致。标准化统计只由该次训练样本拟合。
- 继续只读使用 3.1 源数据和 4.1 子集。源配置 SHA256 为 `eaf8c0cf6e902589cd5c68b8811601b593fe040e51d660c354e320b705cca6cb`；子集配置为 `2b6006ac4f01441dfd44b7f70655f115b82061fdac5b891bb9fdd33eb6b069c8`。
- 每条 primitive 为 `7 × 32 × 3`，按原中心起点平移、按初始邻居半径 r 归一化。原始完整时间窗、起点、积分时长和邻居尺度均保留；不在缓存内重新分配训练/验证/测试。Cylinder 起点遵循原模拟 `[0,15]` 的 `[7.5,12]` 范围，其他流场为原时间范围的 10%–80%。
- 七个初始点是固定十字播种位置；网络预测后 31 步的 651 个坐标，初始点按定义补回。评价不计固定初始点。
- 主指标为位置均方根误差除以 r：`sqrt(sum((prediction-truth)^2)/(N*7*31))`，三维坐标平方先求和，不能再除以 3。汇报普通测试集和未见尺度测试集、逐流场和九流场等权平均，均越低越好。
- 测试集是已使用过的 benchmark，不宣称为全新确认集。所有选型和 epoch 选择只能看训练/验证。

## 3. 网络与方法表

流程：`primitive → [冻结原 FMT → 可训练专家 A；几何分支 B] → 输入相关门控 → 192 维 token → 共享全连接解码器 → 7 × 32 × 3`。

| 模块或方法 | 具体实现 | 代码 |
|---|---|---|
| A：原 FMT | 中心路径线跨时间位移；六条邻居相对中心的位移乘 100；实数 DFT 保留 6 个频率；每频率的实部范数、虚部范数及夹角余弦，加 5 个手性量；每线 23 维，邻居按分量排序并乘 0.5，最终 161 维 | `FMT_Utils/DFT_FMT_3D.py:pathline_dft_features_3d`；本版本 `original_fmt` |
| A 的学习器 | 161 维补零到 651 维，仅用于严格匹配 Raw 专家的参数数量；训练样本逐通道标准化；线性编码加 0.1 倍残差全连接编码，宽 512、3 个残差块，输出 192 维 | `FMT_Utils/Task6FMTGeometryMoE_3D.py:DenseEncoder` |
| B：raw_mlp | 输入 651 个原始几何坐标；训练样本逐通道标准化；宽 512、3 个残差块的全连接编码器，含可训练线性通路；输出 192 维 | `DenseEncoder` |
| B：pointnn | 沿用冻结 `SnapshotPointNN`，每个时间步由三角函数、邻域聚合提取 1008 维；训练样本标准化后投影到 256 维，6 层、8 头 Transformer 融合 32 步，再压到 192 维 | `FMT_Utils/Task6PNNTrans_3D.py:SnapshotPointNN`；本版本 `TemporalEncoder` |
| scalar 门控 | 拼接两个专家的 192 维向量，经 LayerNorm、128 宽全连接、GELU，生成一对 softmax 权重；整个 primitive 共用一对权重 | `FMTGeometryMoE.encode_features` |
| channel 门控 | 同一结构产生 192 对 softmax 权重；每个隐向量分量各有一对权重 | `FMTGeometryMoE.encode_features` |
| 共享解码器 | 只读取混合后的 192 维 token；可训练线性输出加 0.1 倍残差全连接输出，宽 768、4 个残差块，输出 651 个坐标 | `FMTGeometryMoE.decode` |
| fmt_moe | A 为原 FMT，B 为本行指定几何分支 | `FMTGeometryMoE` |
| geometry_moe | 将 A 换成原几何的全连接专家，B 不变；总参数量与 fmt_moe 完全相同，用来区分 FMT 信息与增加专家容量的作用 | 同上 |
| geometry_only | 只有 B 和相同解码器；B 和解码器的初始参数与两个 MoE 对照相同 | 同上 |
| raw_frozen | 原 1.1 c1：128 宽、3 层 Transformer，192 维 token，512 宽、3 块解码器；学习率 0.0003、dropout 0.1、weight decay 0.001、12000 次更新 | 冻结 `FMT_Utils/Task6PNNTrans_3D.py:train_model` |

Point-NN 的无参数前端保持原实现；本次时间编码器重新组合，未沿用 1.3 中额外的 MLP 投影。纯全连接分支不做时间傅里叶变换。本实验不声称客观性。

两个专家的隐向量是共同训练得到的。融合公式为 `z = g_A*z_A + (1-g_A)*z_B`。门控初始为 A=0.1、B=0.9，之后完全由训练决定；不强制平均分配。辅助损失初值 0.05，对两个专家各自通过共享解码器的重建误差取平均，在前四分之一训练内线性衰减到零，帮助专家进入可融合的隐空间。后四分之三仅优化混合输出。

所有分支均须经过 192 维隐向量。无原坐标到输出的旁路、无逆 DFT、无主成分分析（PCA）初始化或解析重建。线性通路也全部可训练，并且经过 192 维瓶颈。解码器末层置零，初始输出为训练集平均几何，其余层随机初始化。补零只增加名义匹配参数，不增加 FMT 信息；Raw 对照实际可用的输入自由度更多。

## 4. 搜索、训练与对照

预先固定八候选：两个几何分支 × 两种门控 × 两个学习率（0.0001、0.0003）。AdamW，dropout 0.1、weight decay 0.001、梯度裁剪 1；余弦学习率调度，批大小 128。新版本不加 1.3 的几何混合或噪声增强。相同候选的三个新方法使用相同训练步数、随机种子和批次顺序。

1. 九流场各取前 32 个训练样本，对两个 MoE 家族各做 3000 次训练过拟合检查。关闭 dropout、权重衰减和辅助损失，学习率 0.001。要求训练误差小于初始误差的 15%，且清零 token 后误差至少为训练误差的两倍。检查只读训练集。
2. 固定 cylinder3d、f22raptor、smokeBuoyancy 三个开发流场，每个跑八候选，各 12000 次更新、前 1200 次预热、每 2000 次记录一次。按验证误差与相同 3072 数据的冻结 Raw 误差比值，在每个 B 家族保留前两名，共四候选。
3. 四候选在九流场各跑 fmt_moe、geometry_moe、geometry_only，均 72000 次更新、前 3000 次预热、每 3000 次记录。按 fmt_moe 九流场平均验证误差，每个 B 家族选一套全局配置；对照使用该同一配置。两种 B 都保留，不按测试集选择 B。
4. 只要一个家族的九流场、三个方法的验证误差全部小于 3，该家族进入最终三种子重训与测试。数值不合格家族保留失败记录。此新版本与 1.3 的差别明确：**超过 Raw 是待检验的实验结果，不再作为允许读取测试集的条件**。旧版结果不改写。
5. 最终冻结 Raw 仍按原 12000 次更新重训；新方法各 72000 次。与冻结 Raw 比较反映最终可用性能，不能单独归因为 FMT，因为训练预算不同；相同预算的 geometry_moe 和 geometry_only 才是本版本内部对照。
6. 单个最终模型若验证误差非有限或大于等于 3，登记失败、不读取其测试；不得只汇总成功子集冒充完整平均值。

计划训练量：训练拟合检查 18 模型；初筛 24 模型；完整开发训练 108 模型；若两个家族均数值合格，最终 189 模型。最大 18 个 GPU 作业并行，一个作业内依次完成对应方法，不把三个方法同时挤在同一 GPU。

## 5. 如何判断频域分支是否有用

至少同时给出：相同预算的 geometry_only、相同参数量的 geometry_moe、原冻结 Raw；报告逐流场和全部种子，不预设 MoE 会胜出。

记录验证集平均 A 权重、权重标准差、趋近 0/1 的比例；还对训练完的 MoE 强制仅走 B、仅走 A，以及将 A 在样本间错配后重建。这些是依赖性诊断，不替代独立重训对照。只有门控不为零不能证明 A 提供有效几何信息；若 FMT MoE 不优于两个新几何对照，则不能归结为原 FMT 带来的收益。

## 6. 实现、验证和保存

- 主网络：`FMT_Utils/Task6FMTGeometryMoE_3D.py`。
- 实验流程：`experiments/Task6_FMTGeometryMoE_1_1.py`；只读状态核验：`experiments/Report_Task6_FMTGeometryMoE_1_1.py`。
- 冻结配置：`config/Verify_Task6_FMTGeometryMoE_1.1.json`。
- 提交：`Submit_Task6_FMTGeometryMoE_1_1.py` 和 `ibex_bash/task6_fmt_geometry_moe_1p1.sh`。
- 测试：`tests/test_task6_fmt_geometry_moe_1_1.py`、`tests/test_task6_fmt_geometry_moe_pipeline_1_1.py`。包括原配方特征逐项相等、批次独立性、参数量、门控端点、训练统计隔离、两类网络实际拟合，以及真实训练贯穿准备/选型/最终测试/独立误差复算的微型流程。故意破坏预测文件必须导致审计失败。
- Ibex 先运行相同测试，通过后验证源数据及基线哈希，再开始 GPU 训练。每阶段作业登记 `docs/ibex_run_registry.md`，事件和指标保存为 JSON/CSV。
- 最好参数仅在 RAM 保存。不得生成或下载 `.pt/.pth/.ckpt`；测试预测只在 Ibex 保存用于独立复算，不下载模型。最终审计检查各方法、种子、流场和尺度完整性、预测形状/有限性、固定初始点及位置误差；不完整结果不产生完整宏平均值。

配置、代码 commit、种子、索引 SHA256、节点、GPU、起止时间、学习率轨迹和逐次误差共同定义可复现记录。运行结果与结论只追加到 `docs/experiment_log.md`，不覆盖历史记录。
