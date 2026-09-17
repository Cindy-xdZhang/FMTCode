# 验证实验总表

<!-- fmt-convolution-withdrawal-notice-20260917 -->
> **2026-09-17撤销通知：** 本文旧Raw卷积＋FMT/残差及二维FMT卷积方案已撤销，不能重跑或用作当前FMT主表、附录、“最佳方法”证据。历史数值仅供核对。无卷积的固定特征和全连接分支另行保留；以 [FMT禁止空间卷积协议](FMT_no_spatial_convolution_protocol_1.1.md) 为准。
<!-- fmt-convolution-withdrawal-notice-20260917-end -->

<a id="status-20260913"></a>

## 截至2026-09-13的进展索引

当前停止现有客观化方向，暂停几何tokenizer与混合专家调参。下面按问题合并检索入口；
方法结论、数值与修订原因集中在[项目进展记录](experiment_log.md#progress-2026-09-13)。
后面的09-05总表和预注册原文保留其日期，不把当时的“待完成”误当作最新状态。

| 研究问题 | 合并的验证系列 | 状态与查阅位置 |
|---|---|---|
| 分类收益、强对照、消融与噪声 | Task1/2/3/5统一确认、StrongBaselines、FMTComponents、Noise系列 | 主结果冻结；本页原总表与论文表保留不同对照的边界 |
| 输入客观性与后续运算 | ObservedPathline、ObjectivityMechanism 1.1–1.3、DistanceInputIdentity、NoCenter、RelativeFourier | 核验记录已写入实验流水；本页末尾合并原协议，冻结算法不改 |
| 客观标量编码效果 | FMTAllV2 1.1/1.2、LargeNeighbor 1.1、ObjectiveFMTnTDO 1.1/2.1 | 已有结果完整保留；停止继续追求此路线 |
| 代理标签涡型分类 | Task4A系列、Task4B 1.x–5.2 | 早期泄漏结果无效；5.1/5.2保留各自空间划分和标签边界，不能混为Task3二分类 |
| Task4-c实现检查 | Verify_Task4C_Pipeline_1.1、Verify_Task4C_BinaryPipeline_2.1 | 1.1四类BiLSTM仅保留历史；2.1检查GT二分类、局部曲线、FMT/Conv3D训练与空间隔离。临时检查源码按用户要求删除，报告与[记录](experiment_log.md#task4c-binary-2026-09-13)保留；数值检查不是性能结论 |
| 几何查询、重建与直接神经核验 | Task678 1.1系列、Task6 2.1/3.1/4.1/5.1及DirectAudit | 区分任务定义、线性初始化和纯神经结果；路线暂停 |
| 网络容量、样本量与学习率 | Task6 PNNTrans 1.1/1.2/1.3 | 验证搜索完成，未达到测试启动门槛；没有最终test成绩 |
| 混合专家与联合任务 | Task6 FMTGeometryMoE 1.1、Task36 MultiGate 1.1与BalancedScale 1.2 | 最后一版已完成最终测试；优势取决于指标，暂停后续调参 |
| 图形、数据与实现检查 | AIVD图形校验、数据加载检查、单位测试、管线预检 | 审计通过不等于科学主张成立；集群必需预检与核心测试保留，一次性检查已归档 |

十二份分散的`Verify_*.md`原文已并入本页末尾；本轮20份测试、22份检查/报告脚本与46份过时提交脚本移出工作目录。
逐文件清单、校验和恢复方式见[维护说明](repository_maintenance.md#cleanup-20260913)。

按研究问题合并历史验证，不再逐个陈列参数扫描版本。以下为截至 2026-09-05 的结论。
完整配置、运行状态、失败记录和指标仍在 [实验流水](experiment_log.md)，
作业证据在 [Ibex 登记表](ibex_run_registry.md)，主结果在 [论文总表](paper_tables_tasks_3d.md)。
“开发集”用于选择配置；“独立确认”指配置冻结后在未参与选择的数据上评估。

术语：FMT 是无可训练参数的轨线几何编码器；VAE（变分自编码器）学习低维表示；
PCA（主成分分析）用于线性维度压缩；DFT（离散傅里叶变换）提取频率信息；
IVD（瞬时涡量偏差）用于定义涡区标签。F1 综合精确率与召回率，越高越好；
Average Precision 是跨分类阈值的精确率–召回率指标，越高越好。
数值一般为数据集等权平均，表内明确标注的局部实验除外。

| 任务 | 验证了什么 | 结果与适用边界 | 实验依据 |
|---|---|---|---|
| 跨任务 | 二分类标签阈值 | p80–p95 扫描中，p95 的 Task2 F1 增益及 Task3 F1、Average Precision 增益最大；当前固定 p95，其他阈值只作敏感性分析 | `Ablation_Task23IVDPercentile_1.2` |
| Task1 | 早期单流场标签与 FMT 参数搜索 | Re160 单时刻 420 组搜索，最佳 F1 .85275；只是局部开发结果，标签区域事后搜索也不是独立真值，不能替代跨流场确认 | `Verify_IVDSearch_1.1`、`Verify_3DFMTHyperparam_1.1` |
| Task2 | 高雷诺数下增加 VAE 容量是否解决失败 | Re6400 重构均方误差从 .1100 降至 .0367，但最佳聚类 F1 仅 .210，低于直接 FMT 的 .556；重构更好不代表涡区表示更好 | `Verify_VAEFailureHighRe_1.1` |
| Task2 | 平衡各几何块的重构损失、匹配网络容量 | Re640 有改善；Re6400 仍不如 Raw。纠正重构偏置不足以解决高雷诺数失败 | `Verify_VAEObjective3D_1.1` |
| Task2 | 高雷诺数采样、输入表示及 VAE 开发扫描 | `HighReVAE_1.1–1.9` 等是开发配置，不能逐个当独立确认。后续受控确认更改空间采样、积分窗口、邻居频率表示和 VAE 后，Re640/Re6400 的 FMT 相对 Raw 增益为 +.1692/+.1276，6/6 配对种子为正；无法把收益归因于其中一个改动 | `Verify_HighReTask2_Controlled_1.1`；对应 HighRe 开发文件见归档清单 |
| Task1–3 | 固定积分步长、轨迹长度和重采样点数 | 早期扫描支持 .25/48/32。后续七组联合验证选择 48 点，独立确认的三任务平均相对增益增加 .002221，但 FMT 绝对 F1 平均下降 .002749；不足以替换主表的 32 点 | `Verify_PathlineHyperparams3D_1.1`、`Verify_Task123_FixedPathline_1.1` |
| Task2 | VAE 瓶颈与统一 FMT 配方 | 先前逐流场族开发只作历史补充；统一开发选择 189 维 FMT、隐层 [512,256]、潜变量 64、7000 步。独立确认 Raw/FMT F1 .4884/.5840，8/10 数据条目为正，Boeing 与 F-22 为反例 | `Verify_Task2_LatentBottleneck_5.1`、`Verify_Task2_UniformFMT_6.1`、`mainExp_Task2_3D_6.2_uniform_confirmation` |
| Task3 | 直接拼接 FMT、残差结构与额外空间采样 | F-22 上直接拼接会失败；冻结 Raw 主干、增加 FMT 残差后改善。必须以同结构 Raw-PCA 残差为主对照。旧 7.2/8.1 在额外空间样本上的增益约 .175/.180，但使用逐流场族配方，只作补充 | `Verify_Task3_F22AnchoredFeatures_1.2`、`Verify_Task3_SpatialRobust_5.2`、Task3 7.2/8.1 |
| Task3 | 网络、损失和优化器的大量扫描 | 6.1–47.1 主要是开发集内的小幅、流场相关变化；48.1–54.1 的组合选择将开发增益提高到约 .2055，不能直接作为独立确认成绩 | Task3 搜索链 6.1–54.1，详细记录见 `mainExp_Task3_3D.md` |
| Task3 | 后续辅助分支微调是否值得继续 | 57–64 累计开发增益仅 +.001597，65–76 未通过原有保留条件；没有形成新的独立确认优势 | Task3 辅助分支搜索链 55.1–76.1 |
| Task3 | 终止的搜索链 | 77.1 不完整；后续性能阶段未运行，111.1/112.1 只有本地预注册。没有有效性能结论，不读取部分结果挑选赢家 | Task3 搜索链 77.1–112.1；取消状态见 Ibex 登记表 |
| Task3 | 单一配方能否跨十个数据条目成立 | 独立确认 Raw-PCA/FMT F1 .6398/.8583，Average Precision .6891/.9236；10/10 条目增益为正，是当前主结果 | `Verify_Task3_UniformFMT_9.1`、`mainExp_Task3_3D_9.2_uniform_confirmation` |
| Task1 | 更强的传统几何与普通 DFT 对照 | FMT F1 .6014，普通逐坐标 DFT .5843，曲率/扭率/速度序列 .4471；FMT 仅比最强对照高约 .017，不能描述为大幅领先所有几何表示 | `Verify_Task123_StrongBaselines_1.1/1.2`，两版独立审计通过 |
| Task2 | 维度匹配 Raw-PCA 与专门调优 Raw VAE | 新种子中 Raw-PCA189 + 同一 VAE 为 .5770，配对 FMT .5785，近乎持平；Raw 专用 VAE .5407，普通 DFT + VAE .5526。FMT 相对直接 Raw 的优势不等于领先所有强对照 | `Verify_Task123_StrongBaselines_1.1/1.2`、`Ablation_Task123_FMTComponents_1.2` |
| Task3 | 更大 Raw 网络能否追上 FMT | Raw-wide F1 .6340，FMT 约 .8583；单纯扩大 Raw 网络没有缩小差距。新种子版强对照是冻结网络推理复放，不是重新训练 | `Verify_Task123_StrongBaselines_1.1/1.2` |
| Task1/2 | 删除邻居频率信息，或只保留运动学信息 | 新种子中删除六个邻居频率块使 F1 下降 .3307/.1067；仅保留运动学信息下降 .3563/.2784。支持邻居频率信息重要，但这不是“只去掉邻居聚合操作”的消融 | `Ablation_Task123_FMTComponents_1.1/1.2`，输入宽度、下游模型和预算固定 |
| Task1/2 | 手性、实虚部余弦、中心线与局部运动学信息 | 两版中删除手性和余弦均降低 F1，说明有正贡献；Task2 中删除中心频率块从 −.0107 变为 +.0004，其作用不稳定。不能把约 .02 的波动直接当显著性阈值 | `Ablation_Task123_FMTComponents_1.1/1.2` |
| Task3 | 时间傅里叶变换是否解释主要收益 | 只保留初始时刻相对完整频率表示的 F1 差，两版仅 −.0003/−.0023；不支持将主要收益归因于时间傅里叶变换 | `Ablation_Task123_FMTComponents_1.1/1.2` |
| Task1–3 | 随机丢帧和轨迹截断 | 冻结模型测试：丢帧 40% 时 FMT F1 .606/.572/.856，下降小；但截断至 50% 时，Task1 普通 DFT .512 高于 FMT .282，Task2 调优 Raw .596 高于 FMT .338。不能说三任务对截断都最鲁棒 | `Verify_Task123_NoiseRobustness_1.1/1.2`，独立审计通过 |
| Task1–3 | 坐标高斯噪声及更细强度 | 噪声标准差为邻居间距的 .005 时，Task1/2 FMT 已降至 .336/.297；.05 时 Task3 FMT .597 也低于 Raw-PCA .635。逐点噪声是明确弱点 | `Verify_Task123_NoiseRobustness_1.1/1.2`、`Verify_Task123_NoiseTypes_1.1`、`Verify_Task123_NoiseTypesStrong_1.1` |
| Task1–3 | 刚体旋转与整条邻居轨线平移 | 旋转 90° 时，FMT F1 .601/.585/.680；Task1 最强对照是几何特征 .447，不是普通 DFT 的 .209。邻居轨线整体平移下 FMT 也保持较高性能，但没有重新积分，不能称为真正的 seed jitter 测试 | `Verify_Task123_NoiseTypes_1.1`、`Verify_Task123_NoiseTypesStrong_1.1`；合并 `robustness_table.csv` |
| Task1–3 | 共模噪声、时间弯曲、平滑噪声与脉冲离群点 | Task3 在共模噪声和时间弯曲下仍领先；Task2 在较强等级下落后。平滑噪声和脉冲下三任务普遍较弱。全部条件都应报告，不能概括为普遍抗噪 | 同上，两组独立审计通过 |
| Task4 | 传统涡型规则与无监督几何聚类 | 73 个 hairpin 实例中，速度–旋度规则通过严格代理判定的有 13 个，全局纯 FMT 聚类为 4 个；未证明无监督 FMT 已解决涡型分解 | `Verify_Task4A_VelocityCurlBaseline_1.1`、`Verify_Task4A_PerVortexFMTProxySweep_2.1`、`Verify_Task4A_GeometricCurlKMeans_2.2` |
| Task4 | 代理标签监督、空间泄漏与训练可记忆性 | 1.1 因跨集合空间重叠无效；1.2 改为带缓冲拆分。后续 channel+TBL 未见实例确认中 Raw/Raw+FMT 宏平均 F1 .3841/.4196，提升小且绝对性能低；同数据训练和评估的完全记忆测试 9/9 为 1.0，只说明管线可学习，不证明泛化 | Task4-b 1.1/1.2；`mainExp_Task4B_PooledInstanceSplit_3.1`、`Verify_Task4B_FullVolumeMemorization_1.1`、`Verify_Task4B_PooledMemorization_3.1` |
| Task5 | 尺度参数选择、新时间片与未见尺度组合 | cylinder 参数开发及 Re160 新时间片复核后，主确认相对同结构 Raw-PCA 残差的 F1/Average Precision 增益为 +.0891/+.1116；F1 10/10 条目为正，Re640 的 Average Precision 是反例 | `Verify_Task5_CylinderHyperparams_1.1`、`Verify_Task5_Re160FreshTimes_1.1`、`mainExp_Task5_3D_1.1` |
| 尚未完成 | 聚合操作独立消融、真实种子扰动、噪声增强训练 | 尚无完成证据；现有噪声实验只覆盖 Task1–3，不覆盖 Task4/5。不得把整条轨线平移或冻结模型测试替代这些实验 | 当前消融与鲁棒性运行器及配置；后续需新版本 |

整理时纠正了旧摘要的三处表述：删除手性/余弦后下降，意味着组件本身有正贡献；
旋转下传统几何对照没有全部崩溃；截断下 FMT 并非三任务均最优。
这是摘要解释修正，不是重新计算或改写指标；原文快照保存在本地归档，完整流水保留。

历史脚本的保留和恢复方式见 [仓库维护说明](repository_maintenance.md)。以后新增验证先写入
完整实验流水，仅在结论变化时更新本表；不再为每轮扫描新增独立摘要文件。

## Verify_AIVDLongtime_1.1 — 2026-09-07 预注册

用户明确授权：保留aivd1w3_dft，新增aivd1w3_dft_longtime，部署Ibex评估Task1/2/3/5。仅把标量汇总窗口从3点改为完整32点；原差分函数、伪逆阈值、涡量、向量均值扣除、单个未归一化零频系数都不改。不是延长积分轨迹，也没有增加非零频率。原配方保持原输出。

| 方法/版本 | 技术细节 | 主要代码 | 指标状态 |
|---|---|---|---|
| aivd1w3_dft（冻结） | 前3个估计IVD标量的和，1维 | FMT_Utils/Task12Data_3D.py::_anchored_recipe；DFT_FMT_3D.py原函数 | 既有实验保留；本轮配对重训 |
| aivd1w3_dft_longtime / Verify_AIVDLongtime_1.1 | 同一函数window=None，全部输入32个估计IVD标量的和，1维 | 同一配方解析函数新增独立名称；experiments/Run_Task1235_AIVDLongtime_1_1.py | 200分片完成，1000总体/3150尺度指标独立审计PASS；完整数值与结论见experiment_log.md对应完成记录及outputs/Verify_AIVDLongtime_1.1/report_zh.md |

10个3D数据条目×5种子×4任务=200分片。Task1为Raw direct、旧fmt_all+kin4、短/长标量；Task2同四臂、相同VAE隐藏层512/256、latent64、KL1e-6、lr3e-4、恰7000更新。Task3为Raw、Raw-wide、结构匹配Raw-PCA residual、Raw+短/长标量；沿用统一Task3原结构及validation alpha0..3/61点/minimum_gain规则，两种Raw本轮重训。Task5同监督对照并加旧268维FMT和fixed-scale本轮Task3 Raw transfer；所有residual输入宽度268，标量无损零填充，训练结构一致。参数数目必须逐次记录。

全部Cylinder角色t>=7.5。Task1/2沿Verify_AIVDTransfer_1.1晚期划分；Task3沿其Task2晚期划分，并重训Raw/Raw-wide，不能加载早期训练的共享模型。Task5三个Cylinder重新生成缓存，train初始时间7.5/8.0/8.5/9.0，validation10.6/11.3，confirmation12.9/13.6；原训练/验证/未见测试尺度tuple保持不变，含插值guard的跨角色source窗口严格分离，固定seed15068。非Cylinder沿既有Task5划分，作为已用benchmark配对研究。Re640按文件起始7.5换算索引，不再次截半。Re6400仅导出原场所需后期75..149帧，空间抽样与既有ceil(size/96)一致，再于Ibex积分。

Task5短/长标量均按同一个源时间片及完全相同的物理采样时间序列分组，等权扣除组内同时间涡量均值；不把不同尺度下的同一采样序号误当同一物理时间。保留原样按序号差分，不同时更改物理导数或搜索时间窗口。Task1/2/3原同时间片上下文不变。标签仍为初始时刻whole-field IVD p95；长窗口可能改变与初始标签的关联，正负结果均报告。

配置config/Verify_AIVDLongtime_1.1.json、config/Verify_AIVDLongtime_1.1_task5_cache.yaml。基底commit aee4bd562d340158118f2e41f40129a9918e06a1；复用上一轮冻结源码快照，加独立入口和窗口别名。测试8项通过，含原配方不变、独立数值和、后期几何只影响长窗口。预检仅对训练片计算特征，不计算test特征/指标。所有模型选择用validation，200分片全部落盘预测后独立审计，再删除本实验所有临时模型；Task3 Raw临时保留到Task5迁移评估完成，绝不下载checkpoint。

## Verify_AIVDTranslationObservers_1.2 — 相机标注修订

用户要求原Re160七联图每格右上加极简黑白相机与平移速度。仅修改注释层：两笔矢量相机轮廓/镜头，速度精确标为0、1/6、1/3、1/2、2/3、5/6、1倍全局平均速度u_mean(t)。均值速度有轻微时变，因此不伪装为恒定标量，也不新增未给出的物理单位。保持原7220条中心路径线、原预测、色彩、透明度、取景、无边界盒及七行布局。输入文件和标签数组哈希必须不变；不重训、不重分类、不重做客观性实验。

代码experiments/Plot_AIVDTranslationObservers_CameraLabels_3D.py与Render_AIVDTranslationCameraLabels_3D.py，配置config/Verify_AIVDTranslationObservers_1.2.json；源码预检21PASS/0WARN/0FAIL。Python原生绘制相机，纸张和幻灯片两版均重新做面板尺寸、PDF字体、碰撞及逐格视觉检查。原1.1图保留；输出到独立1.2目录。

## Verify_AIVDTranslationObservers_1.3 — 相机速度矢量箭头

按用户要求在每个相机旁加入黑色速度箭头，公式也使用矢量箭头符号。所有箭头表示初始时刻t0=10.5的相机平移速度：从原始observer均值速度时间序列插值得到三维向量，经原图正交投影矩阵映射到纸面方向。七档共用长度比例尺，paper最大60pt、slides最大90pt，其余乘以0、1/6、1/3、1/2、2/3、5/6；零速度以圆点表示，不画非零箭头。时间序列严格递增、数值有限性检查后才插值。完整时变速度定义仍保留在公式中，图注声明箭头对应t0。

仅修改注释，原1.1/1.2文件保留；路径线、分类、取景和布局沿用1.2。代码experiments/Plot_AIVDTranslationObservers_VelocityArrows_3D.py与Render_AIVDTranslationVelocityArrows_3D.py，配置config/Verify_AIVDTranslationObservers_1.3.json。输入哈希、原标签逐字节一致性、注释区域外像素不变，以及最终PDF字体/碰撞/面板尺寸和逐格视觉检查为交付条件。

## Verify_LargeNeighbor_1.1 — 三层球面邻居与全轨线 Fourier 特征

2026-09-07 用户授权方案一largeNeighbor，评估3D Task1/2/3。保留原FMT、aivd1w3_dft及被停止采用的longtime历史版本，不覆盖其结果。本轮明确是新方案验证，不改变既有论文统一主表。当前只实施用户具体定义的方案一，不自行编造另外两种方案。

| 方法/版本 | 技术细节 | 主要代码 | 指标状态 |
|---|---|---|---|
| largeNeighbor / Verify_LargeNeighbor_1.1 | 球半径r、1.5r、2r；每层8个立方体顶点方向，中心加24邻居；同时间Gram内积序列，全32时刻，零频+5个非零频，共3300维 | FMT_Utils/LargeNeighbor_3D.py；experiments/Build_LargeNeighbor_3D.py；experiments/Run_LargeNeighbor_3D.py；config/Verify_LargeNeighbor_1.1.json | 远程审计通过；[最终结果](experiment_log.md#large-neighbor-1-1-final)已于2026-09-08记入，原始CSV仍保留在Ibex |

r沿原primitive的0.5×最小网格间距。每层8点均在球面，使用(±1,±1,±1)/sqrt(3)八个确定方向，所有时刻保持同一物质邻居身份。只在t0播种，之后各自独立积分；不是每步重新在中心旁生成点。沿原RK4的48步积分、0.25×源时间间隔、32个原有采样索引，使用整条轨线。原32采样沿48步取整，Fourier频率仍按采样序号/轨线窗口解释，不冒充物理Hz。

每时刻d_i=x_i−x_center（24个客观几何向量）；300个上三角内积G_ij=d_i·d_j为坐标不变标量。按初始平均平方邻距归一化、减去初始G后，对每个标量的完整32点序列作实数离散傅里叶变换，保留频率0..5的6个实部、1..5的5个虚部。没有中心点的时间差分，也不跨时刻直接相减向量分量；没有IVD/curl或原kin模块混入largeNeighbor。初始尺度归一化与Gram形式延续fmt_all_v2，新增的是24邻居和三层空间范围。该描述符为训练无参数编码器；Task1的PCA/KMeans及Task3的PCA/网络仍需拟合。初始时刻全体点间几何固定，其初始Gram减法不删除随时间演化信息。

所有10个3D条目、每任务原5个种子；训练/验证/评估时间片和标签沿Verify_AIVDLongtime_1.1的Task1/2/3晚期划分。所有Cylinder初始t≥7.5；Re6400已上传late96文件保留原始时间索引，不能减75或再次截半。原中心种子、原7线轨迹和原whole-field IVD p95标签保持；新增25线完整有效且原7线有效的共同样本作为所有臂唯一数据集。删除仅依据轨线完整性和边界，不依据标签或分数；报告各片新增无效数量。复算旧7线的固定样本并验证新旧中心轨线，以确认源场/时间/空间步幅没有漂移。原标签只按共同索引取子集，不重设分位阈值。短IVD用同一保留集合计算同时间涡量均值，算法不改。

Task1：Raw7、Raw25、旧fmt_all+kin4、短IVD、六邻居fmt_all_v2、largeNeighbor；同训练集StandardScaler/PCA8/KMeans（标量不PCA），validation只确定cluster含义。Task2：同六臂、同冻结VAE隐藏层512/256、latent64、KL1e-6、lr3e-4、batch256、7000次更新及原种子；输入/输出宽度随表示变化，逐次记录参数量，不能宣称参数数目完全匹配。Raw25明确提供同样25条原路径线；不为Raw额外调VAE。

Task3：所有臂共享本轮重训的25线Raw骨干；Raw、Raw-wide、同结构Raw25-PCA残差、旧FMT残差、短IVD残差、六邻居Gram残差、largeNeighbor残差。时间卷积按线共享并聚合，25线不改变Raw参数量。所有残差辅助输入统一268维；largeNeighbor只在train拟合StandardScaler+PCA268，旧189/1/231维表示零填充，Raw-PCA从同25线拟合268维。使用既有结构、训练预算、validation的epoch/alpha/threshold规则，所有残差参数量完全一致。此Task3验证25线primitive下FMT的额外贡献；不能把其历史7线Raw骨干结果直接作为成对成绩，也不能把Raw神经分支称为客观网络。

实现检查包括三层几何、时变旋转/平移、共同中心输运不变、已知非零频率恢复、后半轨线和外层几何敏感性。真实数据客观性检查只用训练片。缓存构建是确定的离线几何处理，允许预先生成各角色缓存；学习、标准化/PCA拟合、模型与阈值选择严格只用train/validation，全部模型冻结后才加载test到评估器。每个分片所有臂数据一致；全150分片预测落盘后独立复算指标并汇总每流场/等权macro及配对差。不得用测试结果决定配置修订。

该实验仅评估初始时刻IVD二分类。用户关于短IVD适合单时刻标签、可能不适合路径几何token化的解释记为待验证假设，不能用本轮二分类成绩证明或否定几何token化能力。所有临时模型只保留到最终独立审计完成，不下载；最终保存代码/config/随机种子/数据与设备证据、逐次指标和预测，删除本实验checkpoint。


## 历史验证协议与逐版记录（2026-09-13 合并）

以下十二份原文集中保存，预注册中的“待运行”描述属于当时状态；完成情况以实验流水及本页进展索引为准。原文件的逐字节版本保存在本轮归档中。

<a id="archived-verify-aivdtransfer-1-1"></a>

### 原文件：`Verify_AIVDTransfer_1.1.md`

### Verify_AIVDTransfer_1.1 — unchanged Task3 AIVD transfer to Task1/Task2

Registered 2026-09-07 before this experiment's performance evaluation. The user explicitly requests Task3's `aivd1w3_dft` in Task1/Task2, with all cylinder data restricted to physical start time t>=7.5 and the old baselines retrained. This is a diagnostic comparison on previously used benchmark data, not a fresh confirmation and not a replacement of the frozen paper main tables.

#### Fixed comparisons

| Arm | Exact input | Dimensions | Role |
|---|---|---:|---|
| old_fmt | fmt_all+kin4 | 189 | Original task-level unified FMT recipe, retrained |
| aivd | aivd1w3_dft | 1 | Primary standalone transfer of Task3's scalar feature |
| aivd_kin4 | aivd1w3_dft+kin4 | 29 | Replace only fmt_all, retain the original kin4 addon |
| raw | centre-relative raw pathlines | 672 | Task2 frozen Raw VAE protocol, retrained |

The primary comparison replaces the full original FMT input with the one-dimensional Task3 representation; it also removes kin4. The separate predeclared control keeps kin4, so these two questions are not conflated. All outcomes are reported; neither arm is chosen using test scores.

Task1: training features -> training-only StandardScaler -> PCA8 for multidimensional arms -> KMeans, two clusters, n_init20. The scalar arm has no PCA because its feature dimension is one; this is a deterministic dimensional requirement, not a searched hyperparameter. Seeds7080–7084. The encoder has no learned parameters, while normalization, PCA and KMeans are fitted. Validation labels only assign the two anonymous clusters to vortex/non-vortex; no supervised decision-threshold search is added.

Task2: same frozen FeatureVAE3D hidden layers [512,256], GELU, latent64, KL weight1e-6, AdamW learning rate3e-4, weight decay1e-5, batch256, exactly7000 optimizer updates, seeds100–104. No validation early stopping or new VAE search. Latent means are clustered with KMeans seed7068/n_init20. Raw uses the existing pre_group_rms normalization; FMT inputs use the existing training-only StandardScaler. Input/output widths change with representation, as in the frozen Raw/FMT protocol; total parameter counts must be reported, not claimed equal. Within each dataset/seed, all four arms train on the same allocated GPU.

#### Time and split policy

All ten 3D data entries from the previous replay are included. Only cylinder/half-cylinder are time-filtered. The original simulation interval is [0,15], so the latter half starts at7.5. Select cached slice ordinals solely from recorded source-time metadata, preserving their original roles. Re640 already meets the cutoff; do not halve its [7.5,15] source file again.

| Dataset | Task1 train / validation / test ordinals | Task2 train / validation / test ordinals |
|---|---|---|
| Re160, Re6400 | 4–7 / 8–9 / 2–3 of separate confirmation cache | 4–5 / 6–7 / 8–9 |
| Re640 | 0–7 / 8–9 / 0–3 of separate confirmation cache | 0–5 / 6–7 / 8–9 |
| Other seven entries | Original splits, unchanged | Original splits, unchanged |

Consequently Re160/Re6400 Task2 train on two eligible cached source slices. Keep7000 updates fixed, record the resulting epochs, and do not alter the split based on the outcome. This is not a newly generated uniformly sampled late-time dataset. All arms use identical valid primitives and frozen whole-field IVD p95 references within each run. Old full-time metrics are not the primary comparator.

#### Exact implementation boundary

Use the original `FMT_Utils/Task12Data_3D.py` dispatcher and `FMT_Utils/DFT_FMT_3D.py` implementation. No changes to the old encoder, pseudoinverse tolerance, float32 arithmetic, index-time differences, derivative endpoint scheme or Fourier normalization. Derivatives are computed before selecting the first three scalar samples. Each source slice is processed as one complete mean-vorticity context before VAE minibatching. Do not compute features independently per minibatch.

Translation and constant-rotation invariance apply to the scalar in exact arithmetic with a fixed sample context. Strict finite-step invariance under time-dependent rotations is not guaranteed. The retained kin4 block and Raw input are not covered by a scalar objectivity claim. Prediction quality on p95 references is not an objectivity test.

#### Validation and retention

Five local synthetic tests verify independent NumPy equivalence, scalar/DFT interpretation, centre independence, translation/fixed-rotation invariance, a time-dependent-rotation counterexample with temporal refinement, and mean-context/fourth-coordinate-frame dependence. Preflight reads only the first eligible training slice of each dataset and checks exact Task3-call equivalence, widths1/29/189, and unchanged kin4; two-update synthetic VAE checks validate widths1 and29 without scientific model selection.

All models and validation cluster mappings in a shard are persisted as metadata before any test cache is opened. No model checkpoint is written. Retain source snapshot/manifest, complete config, per-slice input hashes/time/validity certificates, software/device/timing records, independent prediction-based audit, all350 per-run rows, per-dataset mean/sample-standard-deviation and paired deltas. Macro weights all10 entries equally; no significance or equivalence claim without a separate justified test. Report all100 scientific shards and all failures/retries in the Ibex registry.

Method-level conclusions go only in `docs/experiment_log.md`; the result report can reproduce its recorded evidence with the same scope.


<a id="archived-verify-aivdtranslationobservers-1-1"></a>

### 原文件：`Verify_AIVDTranslationObservers_1.1.md`

### Verify_AIVDTranslationObservers_1.1

2026-09-07 用户要求：将原样 `aivd1w3_dft` 用于 Re160 3D 七档平移 observer，按真实 changed labels 检查平移不变性。

#### 冻结协议

- 复用 `Verify_FMTObservedPathline_2.1` 的 7,220 个物质 primitive、七组独立 observed-field 积分轨迹；t0=10.5，96 步显示、前48步重采样32点用于分类。重新验证每组轨迹等于 lab 轨迹减去积分相机位移。没有重新积分或修改这些原始轨迹，也不读取旧预测用于着色。
- observer 速度是原场全局空间平均速度的 0、1/6、2/6、3/6、4/6、5/6、1 倍；积分速度得到相机位移。禁止逐幅重新居中、原域 bounding box 和按标签筛选轨线。
- 分类器采用 `Verify_AIVDTransfer_1.1` Task1/cylinder3d/seed7080 标量分支：只使用晚期 train ordinals4–7、validation8–9，一次 StandardScaler+KMeans(n_init20)，无 PCA；验证原运行的 calibration 指标和类别对应完全复现后冻结。
- 特征函数和 float32 运算保持原样；与旧七联图相同，先把绝对坐标转float32，再减各primitive初始中心。所有7,220个primitive共同计算每时刻涡量均值，不按batch重算。
- 对照报告精确坐标变换、float64运算、同一时刻邻居差、分类距离差。对照只诊断误差，不能替换主图算法、拷贝标签或调整阈值来制造不变性。
- 图只验证平移；不用于证明任意时变旋转下的严格客观性。方法结论只写入 experiment_log.md。

#### 图形约定

问题：固定分类器下，平移 observer 是否改变同一物质 primitive 的特征或标签？图形为纵向七幅轨迹及计数，按用户要求保留全部速度档。a 为原参考系，b–f 为平移速度过渡，g 为全部平均速度抵消环境平移；不将它们当独立统计重复，无误差棒。

结构复用已有七联图 renderer；输入是 [7,N,7,97,3] 轨迹、[7,N] 独立分类和每档计数，显示所有中心轨迹的整条标签，统一物理范围和相机。蓝=非涡、红=涡，不作裁剪。输出 paper 7.2×17.085英寸/7pt、slides10.8×25.6275英寸/13pt、PDF/SVG/PNG和paper600dpiTIFF；这是用户指定的长条比较图，不声明符合单页期刊高度。Python backend。最终输出需字体、面板尺寸、碰撞及逐幅和全图检查。

配置：`config/Verify_AIVDTranslationObservers_1.1.json`。计算：`experiments/Verify_AIVDTranslationObservers_3D.py`。绘图：`experiments/Plot_AIVDTranslationObservers_3D.py`。不写模型checkpoint。


<a id="archived-verify-distanceinputidentity-1-1"></a>

### 原文件：`Verify_DistanceInputIdentity_1.1.md`

### Verify_DistanceInputIdentity_1.1 验证协议

验证目标：直接检查同时间点间距离经过标量时间差分、Fourier、固定网络和固定聚类流程是否保持观察者不变性，并捕获旧编码器实际读取的数组。方法级结论与修订仅记入 `docs/experiment_log.md`。

输入固定为 `C:/Users/xingdi/sources/FLowlineClusteringVisualAnalysis/outputs/Other_FeatureSilhouette_1.1/*/geometry/geometry_bundle.npz` 的全部8份 `paths`，每个primitive含中心及六邻居，每线32点。载入后转换为float64；结果保存每个源文件SHA256。不读IVD标签、不选择样本。另设64个整数坐标primitive构造以消除旋转三角函数舍入误差。

所有材料点共享同一个随时间变化的旋转和平移，不重新播种。计算七点全部21对同时间欧氏距离及其标量时间差分。距离序列取前六个实Fourier频率，拼接实部和非零频率虚部；标准化、8维主成分分析及KMeans只在原观察者下拟合，变换后复用参数。固定随机小网络采用231→32→8、tanh、双精度、eval模式，种子12091；它只检验函数恒等性，不用于分类性能比较。

临时在内存中包装 `DFT_FMT_3D.dft_rotation_invariants_3d`，记录旧编码器送入的数组；finally恢复原函数。断言实际邻居输入等于三个坐标分量的时间差分。冻结磁盘文件不修改。输出逐数据绝对误差、相对误差、逐位相等标志、簇标签变化数、实际输入形状，以及原算法/Gram对照误差。

执行入口：

```powershell
./tmp/jhtdb-venv/Scripts/python.exe -m experiments.Verify_DistanceInputIdentity_1_1
```

结果：`outputs/Verify_DistanceInputIdentity_1.1/results.json`。测试只验证预定义变换下的数值行为；一般距离不变性来自欧氏范数的正交不变性。此实验没有训练任务或Ibex作业，没有checkpoint。


<a id="archived-verify-fmtallv2-1-1"></a>

### 原文件：`Verify_FMTAllV2_1.1.md`

### Verify_FMTAllV2_1.1 — protocol fixed before performance evaluation

2026-09-07. User-authorized replacement of the nonobjective centre/temporal-coordinate construction, followed by paired reruns of 3D Task1, Task2 and Task5 on all ten existing entries. Implementation conclusions and performance conclusions belong in `docs/experiment_log.md`; this file specifies the experiment.

#### Encoder

For the same seven material particles, define at every sampled time
`d_i(t) = x_i(t) - x_0(t)`, i=1,...,6. A time-dependent rigid observer gives
`d_i*(t) = Q(t)d_i(t)`: the vector is covariant, not componentwise invariant.
Before temporal analysis form 21 scalar sequences `G_ij(t)=d_i(t)^T d_j(t)`, i<=j.
Then `G_ij*(t)=G_ij(t)` for every orthogonal Q(t); translation cancels separately at each time.

Normalize by the mean of the six initial squared offsets, subtract the normalized initial Gram matrix, and apply a real-input discrete Fourier transform (DFT) along the sample index. Retain six real coefficients and five nonzero-frequency imaginary coefficients per scalar: 231 features. There is no centre-trajectory feature, temporal point-coordinate difference, or old feature appendage. Normalization/subtraction and bin count use existing Gram-feature conventions, without a search.

The centre is only the origin of same-time relative offsets, not a standalone descriptor. The representation retains relative lengths and angles, loses handedness, and is not invariant to arbitrary changes of neighbour identity/order. Rotated observers must follow the same material particles, not reseed a new axis cross. Temporal Fourier frequencies are cycles per sampled window, not physical Hz; the frozen 32-point caches can have rounded nonuniform physical sample intervals. This does not affect frame objectivity, but limits frequency interpretation.

This is a redesigned objective neighbour encoder, not simply deletion of 23 centre dimensions from the old 161-dimensional feature. No claim of equal information or performance is made in advance.

#### Frozen comparison

| Task | Old task recipe | New recipe | Unchanged downstream procedure |
|---|---|---|---|
| Task1 | fmt_all + kin4, 189 dimensions | fmt_all_v2, 231 | train-only StandardScaler, PCA to 8 components, two-cluster KMeans; 5 original seeds; held-out calibration of cluster identity |
| Task2 | fmt_all + kin4, 189 | fmt_all_v2, 231; also rerun Raw | same VAE hidden widths 512/256, latent width 64, KL weight 1e-6, AdamW learning rate 3e-4, 7000 updates, batch 256, decay 1e-5; 5 original seeds; KMeans and calibration unchanged |
| Task5 | fmt_all + gram2 + kin6, 268 | fmt_all_v2 zero padded to 268 | identical residual network and shared frozen Raw backbone, same training/validation scales, 100-epoch maximum with original validation stopping, 5 original seeds; held-out scales evaluated after model/threshold freeze |

Task2 input/output layers necessarily follow the representation width; hidden architecture, objective and budget are identical. Record actual parameter counts. Task5 padding equalizes the entire network parameter count; its Raw branch still uses coordinates, so encoder objectivity does **not** imply full Task5-network objectivity.

Use all ten entries, existing masks, sample identities, splits, IVD p95 labels and seeds. This is a predeclared paired replay on previously used benchmarks, not a new unseen confirmation. Preserve the frozen cylinder times in both arms to isolate the encoder; do not regenerate early-time cylinder data or apply a late-only filter to just one arm. New sampling continues to obey the original-time cutoff t>=7.5. No model, threshold, frequency bin count or feature selection uses test data. Each shard freezes all trained arms and calibrated decisions before loading confirmation data.

Report pooled-per-dataset F1, IoU, precision, recall, and cluster ARI/NMI for Tasks1/2; also Average Precision (AP) for Task5, plus per-unseen-scale results. Macro averages give each dataset equal weight. Primary performance changes are paired new minus old under the same seed and dataset. Do not substitute development-selection values for test performance.

#### Evidence and retention

Config: `config/Verify_FMTAllV2_1.1.json`; encoder: `FMT_Utils/FMTAllV2_3D.py`; runner: `experiments/Verify_FMTAllV2_3D.py`; tests: `tests/test_fmt_all_v2_3d.py`. Record code/config SHA256, base git commit, input file hashes, material-row certificates, device and actual scheduler timestamps. Synthetic and training-fixture objectivity checks precede fitting. Fail on mismatched rows, nonfinite features or incomplete primitives; do not filter differently per arm.

Task2 models remain in memory. Task5 writes temporary residual checkpoints only within the shard until final predictions and CSV/JSON metrics are durable, then deletes them. Existing shared Raw backbones are read-only dependencies. No checkpoint download. Preserve all failures and register all submitted jobs.

#### Relation to prior code

The same-time scalar construction follows the existing `time_local_gram_dft_features_3d` in `FMT_Utils/DFT_FMT_3D.py`. The new independent NumPy implementation computes geometry and Fourier coefficients in float64 before returning float32, validates finite/positive initial scale, and exposes an explicit `fmt_all_v2` entry without any centre-spectrum or old add-on block. For initial scales above the old epsilon clamp, its mathematical descriptor is the six-bin version of that existing time-local Gram construction. Old Task5 already included a two-bin Gram block alongside nonobjective components; the new encoder uses this objective construction alone. This is an implementation/protocol revision, not a claim that the same-time Gram identity was newly discovered.


<a id="archived-verify-fmtallv2-1-2"></a>

### 原文件：`Verify_FMTAllV2_1.2.md`

### Verify_FMTAllV2_1.2 — core-encoder replacement control

Registered 2026-09-07 after Task1 and part of Task2/Task5 in1.1 were evaluated. This is a methodological control on an already used benchmark, not an unseen confirmation or a model-selection experiment. No hyperparameter search is performed.

The1.1 comparison used the entire old task recipe versus the231-dimensional objective encoder alone. That valid standalone comparison also removed the old auxiliary neighbour blocks. It cannot isolate replacing `fmt_all` within the established task pipelines. Preserve1.1 and add this separate comparison:

| Task | Old arm | New arm | Capacity and reuse |
|---|---|---|---|
| Task1 | fmt_all+kin4 (189) | fmt_all_v2+kin4 (259) | same PCA8 and KMeans; reuse exactly the1.1 old-arm predictions |
| Task2 | fmt_all+kin4 (189) | fmt_all_v2+kin4 (259) | same frozen VAE settings; reuse1.1 Raw/old predictions, retrain new arm only |
| Task5 | fmt_all+gram2+kin6 (268) | fmt_all_v2+gram2+kin6 (338) | pad old to338, retrain both residual arms with identical338-wide architecture and shared frozen Raw backbone |

All datasets, seeds, data, labels and splits remain exactly those in1.1. Retain all1.1 outcomes; no result-based candidate or epoch changes. Task5's larger matched width is necessary to retain the old auxiliary blocks and equalize capacity; do not compare its new arm directly against the smaller1.1 old network as the primary contrast. Original source dependencies are taken from the frozen1.1 snapshot. Only the runner's declarative new-feature dispatch and this control config change.

The `fmt_all_v2` core remains231-dimensional and objective as tested. Retained `kin4/kin6` and the Raw branch are not covered by that exact numerical invariance certificate. Thus this control isolates pipeline performance under core replacement; it does not establish strict objectivity of the complete augmented feature/network. The two comparisons answer different questions and must be reported separately.

Reuse is explicit: aggregated Task1/Task2 old/Raw rows name their1.1 source; verify input file identities and labels. Task5 native outputs alone supply its matched-width comparison. No checkpoint download or long-term storage; delete new residual checkpoints after predictions and metrics are durable.

Task1/Task2 old-arm reuse preserves exact data and seeds but Task2 retraining may be allocated a different GPU model from its1.1 source run. Device records are retained for both; this is a seed/data paired comparison, not a claim of identical floating-point hardware for reused Task2 pairs. Task5's two retrained arms share the same allocated GPU and exactly matched parameter count.


<a id="archived-verify-fmtnocenterobjectivity-1-1"></a>

### 原文件：`Verify_FMTNoCenterObjectivity_1.1.md`

### Verify_FMTNoCenterObjectivity_1.1

目标：直接验证用户所指第三种方案，即原冻结FMT只将中心23维置零，邻居138维原样保留。额外审计原性能实验的28维kin4辅助块；核心与辅助块必须分开判断。方法级结论只写入 `docs/experiment_log.md`。

代码：`experiments/Verify_FMTNoCenterObjectivity_1_1.py`。显式复用冻结编码器的6频率、邻居缩放1、权重1、排序池化、频域范数/内积及手性配方，唯一干预是中心23维置零；逐元素断言其他维度未变，执行前后核对冻结编码器文件哈希。

输入：轮廓系数项目全部8个真实几何文件的所有七线primitive，32采样点，文件路径及SHA256逐项保存。另设中心及六个轴向单位邻居全部静止的解析构造，满足同一材料点连续轨迹定义。无标签、无训练、无新播种。

预定义变换：恒等、时变平移、恒定90°旋转、时变旋转、时变旋转加平移。旋转为z轴0至1.3弧度随采样时间线性变化；平移为(t,t²,sin(t))，t为0到1的窗口相位。每个时间所有材料点共享同一Q(t)与c(t)，断言正交性与行列式为1。几何对照检查全部21对同时间距离、同时间内积矩阵以及相对向量的正确变换关系。

所有输入分别以float64和float32执行；float64用于区分结构性变化与float32舍入。恒等流程必须逐位相同。报告原始单位最大绝对误差、L2范数误差、对称相对L2误差及超过rtol=atol=1e-5的primitive数量，不跨表示用绝对误差作性能排名。单个合法变换下稳定的反例即可否定任意观察者不变性；有限测试通过本身不证明普遍不变。

```powershell
./tmp/jhtdb-venv/Scripts/python.exe -m experiments.Verify_FMTNoCenterObjectivity_1_1
```

输出目录 `outputs/Verify_FMTNoCenterObjectivity_1.1` 拒绝覆盖，包含结果JSON、逐块误差CSV和静止七点反例的完整输入/输出NPZ。`execution_status`指验证程序是否完成，`encoder_observer_invariance`单列编码器判定，避免将测试执行成功误读为算法客观。只在本地运行，不提交Task3/Task5训练，不修改冻结算法或历史指标。


<a id="archived-verify-fmtobjectivitymechanism-1-1"></a>

### 原文件：`Verify_FMTObjectivityMechanism_1.1.md`

### Verify_FMTObjectivityMechanism_1.1

2026-09-09 用户要求部署 Task1/2 客观化性能下降的原因诊断。此实验不改冻结算法、不替代主表，也不以旧确认集选择新方法。所有对照在运行前固定，属于已使用 benchmark 上的机制探索。

#### 待检验问题

1. 删除中心块是否足以解释下降？严格零遮罩保留旧邻居时间差分、傅里叶不变量、排序池化及运动学块。
2. 去除邻居分量差分是否改变性能？旧邻居算子只取消时间差分，其余缩放、频率数、排序及权重不变。此臂仍不保证时变旋转客观性。
3. 同时刻 Gram 编码的损失来自哪些因素？分别改变初始尺度归一化、标量时间差分、频率截断和时间域/完整频谱表达。完整频谱与时间序列保存相同归一化 Gram 信息；标准化后距离仍可能不同。
4. 主成分分析（Principal Component Analysis，PCA）按方差保留低维方向，是否主导 Task1 的退化？所有表示固定运行 PCA8 和无 PCA 两种后处理；不选较好的一项作为主结果。
5. 高 F1 是否仅符合随机对应？固定预测后，在每个测试时间片内打乱标签100次，保留正类比例；报告零假设分布。通过此控制只反对随机对应，不能排除标签相关捷径或数据混杂。

#### 数据、容量和训练

复用 Verify_AIVDTransfer_1.1 的10数据条目、p95标签及已冻结晚期时间片划分。Cylinder 起点≥7.5；不将晚期新结果与全时间旧指标直接作因果比较。训练/校准/测试文件身份和 SHA256逐项保存。Task1/2各3种子，10×3×2=60分片。

12种表示：Raw、旧全配方、旧去中心、旧邻居单独、无邻居差分+kin4、客观Gram6、Gram6+kin4、不作初始尺度归一化的Gram6+kin4、标量差分Gram6+kin4、完整Gram频谱+kin4、完整Gram时间序列+kin4、aivd1w3_dft+kin4。kin4指速度梯度估计所得四条标量序列的四个傅里叶频率，总28维；其有限差分实现不在严格客观认证范围内。

全部无损补零到700维。Task2各臂同一700→512→256→64均值/方差编码器、镜像解码器，7000更新、学习率3e-4、KL权重1e-6、batch256、weight decay1e-5；复用冻结训练函数，未调优。Raw采用中心相对初始位置+同时间邻居偏移、原pre_group_rms训练侧归一化，其他臂逐列训练侧标准化。所有模型和簇含义在读取测试性能前冻结。任务比较使用同一模型结构、同一宽度；这是新容量控制，不宣称逐位复现189维旧主表。补零不等于各臂携带相同信息，保留原重构损失在700维上取均值，输入统计及有效维度必须随结果报告。

Task1每臂两种后处理共24行/分片；Task2每臂一行共12行/分片，合计1080行。所有拟合仅用训练特征，验证标签仅选择匿名簇对应的涡类别；不能把标签校准流程称为完全不使用标签的最终判读。

#### 数值机制与几何诊断

解析反例：三个空间分离的局部刚性邻域角速度为0、1、3，全部同时间距离/内积恒定，但相对于三者平均值的涡量偏差不同。可在不相交邻域构造相应局部流；此为信息不足反例，不作为真实流场性能。全局刚体观察者对全场使用同一Q(t)，各primitive分别转动不是观察者变换。逐primitive Gram编码对后者也不变，因此其不变范围超过物理客观性要求。数值检查同步验证有限差分估计与解析量一致；这不证明真实数据退化全部由该原因造成。

追加768个固定测试样本的轮廓系数：同一分组分别在其实际聚类空间、共享的归一化完整Gram时间序列距离中评分；不使用二维投影。共享距离只描述同时刻形变几何，不代表完整跨时刻运动。样本固定、不按得分选取，单簇记空值。分类指标使用全部样本。

#### 来源与执行

新增 `FMT_Utils/ObjectivityMechanism_3D.py`、`experiments/Verify_FMTObjectivityMechanism_1_1.py` 和独立单元测试。基线代码使用已冻结fmt_all_v2_1p2源快照，保持原函数不变；新增文件覆盖在独立实验目录，全体源码清单哈希固定后才能提交。preflight验证全部输入身份、旧FMT缓存重算及解析反例；smoke用3更新验证两条任务路径，不作性能证据；全部批处理依赖两者成功。

每个Slurm提交、开始、设备、完成/失败都写入作业登记与独立JSON。最后审计独立从预测复算分类指标，检查全部分片、VAE参数量和7000步；不保存模型checkpoint。

#### 示意图契约

单面板、Python制作、分析性示意坐标，不是流场测量。七条灰线、每时刻七点；蓝色实线表示同时间相对位置及不变距离，橙色虚线箭头表示中心跨时刻坐标差。图中几何是二维投影，不可测量其投影长度替代三维真实距离。PDF/SVG/PNG与坐标、脚本和字号/碰撞检查一起保留。客观性讨论采用同一组材料点，不能更换播种邻居。


<a id="archived-verify-fmtobjectivitymechanism-1-2"></a>

### 原文件：`Verify_FMTObjectivityMechanism_1.2.md`

### Verify_FMTObjectivityMechanism_1.2

2026-09-09，1.1集群预检失败后建立修订版本；仍遵循1.1的12臂、10数据条目、3种子、晚期Cylinder划分、700维容量控制和全部评估协议。

原因：新诊断误用了`pathline_dft_features_3d`函数默认`neighbor_scale=100, neighbor_weight=.5`，但冻结缓存生成器`Build_Task2_Universality_Cache.py:135`明确使用二者均1.0。由于手性三重积有分母下限，缩放不一定能被最终StandardScaler完全抵消。1.1预检在第一个Channel输入处以原定rtol/atol2e-4拒绝缓存复放；未产生正式Task1/2测试指标。1.1的3更新smoke已完成，不能作为有效科学对照。

修订：显式以缓存生成参数1.0/1.0重算旧特征；“只去掉邻居时间差分”臂也使用相同1.0/1.0，使之真正只改变差分操作。原冻结FMT函数、缓存和已有主表均不修改。数值容差保持不变；增加缓存参数单元测试，总7项。运行入口代码路径复用，1.1原始7文件上传包及远端独立目录完整保留，以源码SHA区分版本。

1.2在新的`/home/zhanx0o/FMT_ObjectivityMechanism_20260909_1p2`运行，输出独立`Verify_FMTObjectivityMechanism_1.2`目录。原1.1失败预检和已完成smoke保留；未启动的依赖数组及其审计取消并登记。全部1.2任务重新依赖新预检和新smoke，测试前仍冻结所有候选，未读取任何正式测试性能用于本修订。


<a id="archived-verify-fmtobjectivitymechanism-1-3"></a>

### 原文件：`Verify_FMTObjectivityMechanism_1.3.md`

### Verify_FMTObjectivityMechanism_1.3

2026-09-09，延续1.1/1.2全部诊断设置，但明确拆开旧缓存和从保存坐标重算的旧算法。当前实际执行版本为1.3，原两版代码包、输出和失败记录保留。

1.2修正邻居缩放参数后，Channel缓存重算检查通过；Re160仍有0.378%分量超出rtol/atol2e-4，最大绝对差.00777543，因此预检51633095失败，正式数组没有启动。不能假装这批重算值与缓存逐位一致，也不能把差异直接归因于客观化。

已确认缓存生成代码先用原坐标提取FMT，而保存Raw时作float32中心起点平移；从Raw重算与原提取的数值运算次序不同。差异的具体成因仍可能包括局部坐标舍入与CPU/GPU计算差异，本版本不声称已逐项定位。原7项单元测试及1.2可执行smoke仍按其证据边界保留。

1.3新增两个预注册臂：`old_recomputed_full`、`old_recomputed_no_center`；总14臂，60个科学分片，Task1每分片28行、Task2每分片14行，共1260行。所有其他数据、种子、网络、训练步数及评价规则不变，没有读取正式测试性能来修订。

对照解释：

- 旧缓存full↔旧缓存no_center：只删除中心块。
- 旧重算full↔旧重算no_center：相同保存坐标下再复现中心删除。
- 旧缓存↔旧重算：隔离缓存数值来源差别。
- 旧重算no_center↔neighbor_undifferenced_kin：同一保存坐标、缩放1、权重1、排序池化及kin4，只改变邻居时间差分。
- Gram频率、归一化、时间域与后处理对照沿用1.1。

预检不再将两个本来不同的来源断言为数值相等，而是逐流场记录原容差之外的比例、最大差与相对误差；分别严格逐元素检查cached臂确实保留缓存、recomputed臂确实调用冻结算法、零遮罩不改其他分量。增加第8个单元测试，防止将两类来源静默混合。原容差仍用于报告差异，不宣称缓存重算通过旧等价检查。

1.3独立目录`/home/zhanx0o/FMT_ObjectivityMechanism_20260909_1p3`，新config、完整源清单SHA256及新作业ID登记到ibex_run_registry。当前请求到此只部署该诊断，不改变冻结主表。


<a id="archived-verify-fmtobservedpathline-2-1"></a>

### 原文件：`Verify_FMTObservedPathline_2.1.md`

### Observer worldline 与 observed pathline 独立核对

依用户要求，按 `optimal-connection/src/flow3d/ReferenceFrame3d.cpp` 的三步顺序新建实现：observer field中数值积分相机worldline；用observed→lab变换构造observed场；在observed场中积分粒子轨线。当前只验证纯平移，旋转矩阵为单位阵。

源代码：`FMT_Utils/ObservedFieldWorldline_3D.py`；实验入口：`experiments/Verify_FMT_Observed_Pathline_2_1.py`；配置：`config/Verify_FMTObservedPathline_2.1.json`。

每个流场保留1.4同一批物质种子、后半程初始时间、96步展示和48步分类窗口；不改冻结Task1-4.1分类器。七档observer速度比例为0至1、间隔1/6。相机参考时刻等于粒子播种时刻，所以初始变换为恒等；此后逐时刻累积相机位移。去掉全部bounding box，保留所有档位共同的取景尺度，不逐幅重置中心。

验证内容：

- 独立积分的observed pathline与原轨线精确坐标变换比较。
- 同一时刻邻居相对位置、双精度邻居FMT特征，以及中心轨线FMT特征分别比较。
- 对精确变换轨线重新分类，排查积分器数值误差是否导致标签变化。
- 固定其他特征只替换中心块，以及固定中心块只替换其他块，分别计算标签变化数量。相同数量不自动代表逐条变化集合相同。
- 错误对照直接积分`v(y,t)-u(t)`，使用按原顺序选定的前128条中心轨线；只在该对照完整者上比较误差，报告其数量；不改变主图队列。

11项本地数值测试通过，新增4项覆盖参考时刻/相机起始点、逆查询错误对照、越出原盒、同刻和跨时刻几何的区别。实际数据结论和全部诊断数值只写入 `docs/experiment_log.md`；不推断benchmark准确率，不用复制标签或修改冻结配方制造不变性。

理论文档已更新至实际正式路径 `optimal-connection/docs/referenceframe/referenceFrame_overview_zh.md` 的§2.3–2.6；用户指定的 `referenceFrame/_overview/_zh.md` 建为通向正式文档的入口。保留原文其余内容；没有修改C++源码，因此不需要C++编译。

最终21份图片已下载并检查，ZIP为outputs/Verify_FMTObservedPathline_2.1/Cylinder_observed_worldline_figures.zip。全部七栏大小检查通过，六份PDF均0FAIL；最小字体论文7pt/PPT13pt，42栏及六张整图完成目视复核。各图7个统计文字跨不可见白色axes背景边缘WARN已接受；无可见边框或轨线裁剪。运行后清除了继承的旧四档审计字段，保留原JSON备份，不改变图像和数值结果。


<a id="archived-verify-task1235-objectivefmtntdo-2-1-protocol-zh"></a>

### 原文件：`Verify_Task1235_ObjectiveFMTnTDO_2.1_protocol_zh.md`

### objective_fmt_nTDO_v2：Task1/2/3/5 性能实验 2.1

2026-09-10 预注册。用户要求将 nTDO 的相对向量输入替换成同一时刻的点间距离标量，并部署 Ibex 性能实验。原版和历史结果冻结。

#### 方法

| 方法版本 | 技术定义 | 代码 | 结果 |
| --- | --- | --- | --- |
| nTDO 1.1（对照） | 同时间相对向量与距离分别做傅里叶编码，138+231=369 维 | `FMT_Utils/objective_fmt_nTDO.py` | 历史完整结果见 experiment_log；本批复跑 |
| objective_fmt_nTDO_v2 2.1 | 全部 21 个同时间点对的欧氏距离直接做时间轴实数快速傅里叶变换；保留 6 个频率的实部与非直流频率的虚部，共 231 维 | `FMT_Utils/objective_fmt_nTDO_v2.py` | 待运行，不预断结果 |

输入为 `[N,7,L,3或4]`，第四坐标如存在则不进入编码。直接计算 `norm(x_j(t)-x_i(t))`，`0<=i<j<=6`，共 6 个中心—邻居距离及 15 个邻居—邻居距离。与第一版重心化后求距离相比，直接点对相减可能存在浮点舍入差异，数学定义相同。

送入傅里叶变换的唯一数组为 `[N,21,L]` 距离标量；不保留向量分支，不计算时间差分、不减初始距离、不作尺度归一化，不加入 IVD 估计或运动学特征。保留直流分量。频率按均匀采样窗口定义，不解释为物理赫兹。同时间距离标量的客观性作为已知前提，依用户要求不做观察者变换或客观性验证。代码检查仅覆盖点对枚举、标量傅里叶输入、输出顺序、接口和训练流程。

#### 冻结的比较

实验 ID：`Verify_Task1235_ObjectiveFMTnTDO_2.1`。完整配置：`config/Verify_Task1235_ObjectiveFMTnTDO_2.1.json`。运行、提交、审计入口均为对应的 `_2_1.py` 文件。

沿用 1.1 的全部 10 个三维数据条目、各任务 3 个随机种子、IVD p95 标签、训练/验证/测试切片与训练预算。没有新增超参数搜索。每个数据/种子内先冻结所有比较臂的模型及阈值，再加载测试特征和标签。本实验复用已评估的 benchmark，不声称为新的独立确认；历史时间窗口可能重叠的边界沿用 1.1。

| 任务 | 比较臂 | 固定设置 |
| --- | --- | --- |
| Task1 | Raw、原始 FMT、nTDO 1.1、v2 | 训练集标准化和主成分分析至 8 维，KMeans 二类聚类；种子 7380/7381/7382 |
| Task2 | Raw、原始 FMT、nTDO 1.1、v2 | 全臂零填充至 700 维，完全相同的变分自编码器；7000 次优化更新；种子 310/311/312 |
| Task3 | Raw、加宽 Raw、Raw 主成分残差、原始 FMT、nTDO 1.1、v2 | 特征辅助输入统一填充至 369 维，所有残差网络总参数均为 138818；种子 40/41/42 |
| Task5 | Task3 Raw 固定尺度迁移及 Task3 中列出的全部可变尺度训练臂 | 同一网络参数量及原始尺度划分；种子 40/41/42 |

v2 的 231 个有效特征在 Task3/5 后接 138 个零；这保持网络结构和参数量一致，不改变编码器输出。Task2 同理填充至 700。原始 FMT 对照为 Task1/2/3 的 `fmt_all+kin4` 与 Task5 的 `fmt_all+gram2+kin6`；不加入 a1ivd 对照。`ntdo` 结果键指第一版，`ntdo_v2` 指本次新版。

预期 120 个任务/数据/种子分片，630 行主指标，1890 行 Task5 分尺度指标。汇总以 10 数据 × 3 种子的宏平均为主，同时保留逐数据、逐种子和 v2 相对各对照的配对差值。

#### 数据与执行

三个 cylinder 条目的有效起始时间均不早于 7.5。Task5 Re160/Re6400 直接只读复用前批已修正并通过审计的 `late_task5_cache_r2`，不重建数据；其他条目沿用同一冻结缓存。配置中保存缓存绝对路径。Task5 训练、验证、测试尺度组合分别为 18、6、9 种，沿用原划分。

独立 Ibex Git checkout：`/home/zhanx0o/FMT_nTDO_v2_20260910`。输出位于其中 `outputs/Verify_Task1235_ObjectiveFMTnTDO_2.1`。提交前固定代码 commit、配置及源文件散列；所有进程登记到 `docs/ibex_run_registry.md`。训练检查与数据预检通过后启动性能阵列，Task3 完成后在同一进程执行 Task5。末尾从保留的预测重新计算指标并审计。

只在 Task3→Task5 依赖链中临时保存 checkpoint，完成评分及审计后删除。长期保留逐次指标、预测、配置、数据证据及设备记录，不下载模型。


<a id="archived-verify-task6-directaudit-1-1"></a>

### 原文件：`Verify_Task6_DirectAudit_1.1.md`

### Task6 5.1 独立审计 1.1

2026-09-11用户要求“审计”。版本`Verify_Task6_DirectAudit_1.1`只读取已冻结5.1输出，绝不修改训练代码、模型选择、数据或历史指标；不训练新模型。科学代码commit `1496fd7c31a6dae3d71fd25228dfba6bafa3b289`。

核对九流场：科学文件/配置/选择哈希；全部缓存哈希、原始几何与四个训练子集逐值一致；primitive ID、完整积分源时间片隔离与播种时间；全部最终模型的样本量、参数数量、训练步数、最佳验证步；失败模型是否确实未生成test预测；全部保存预测的RMSE/r、中心、邻居、端点、同时间点间距离及逐时刻误差。复算代码不调用原geometry_metrics函数。不得删除失败模型后报告总体FMT测试均值。

只在搜索种子94110的1024训练样本与原validation上附加信息诊断：完整16频float32系数能否独立解析还原；逐通道标准差、标准化后协方差有效秩/99%方差所需维数、验证值超出训练通道范围的比例。诊断中的解析逆变换与训练/推理隔离；其用途仅是判断频率系数是否保留几何，不是新模型或新baseline。描述性统计不能单独证明训练失败的因果机制。

结论及新旧表述修订只写入`docs/experiment_log.md`。运行登记`docs/ibex_run_registry.md`；输出`outputs/Verify_Task6_DirectAudit_1.1/`。不保存或下载checkpoint。
