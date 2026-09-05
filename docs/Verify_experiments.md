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
