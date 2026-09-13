# 验证实验总表

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
