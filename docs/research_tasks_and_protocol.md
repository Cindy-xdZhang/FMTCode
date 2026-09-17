# FMT 研究任务与统一协议

<!-- fmt-architecture-audit-20260917 -->
2026-09-17全面架构审计取代此前继续部署：旧Task3/5的nTDO、p35固定中心但完整路径有3层Conv1d；Task4-c p35/h0/n0与c156都逐有效线轮换中心、FMT分支无ConvNd；ASAP相机固定中心但后接c156七中心。禁止用p35简称推断完整方法。TemporalDFT可学习循环卷积和fps_graph的EdgeConv式图邻域入口已依既有禁令撤销；历史数值/源码保留。每个方案必须登记中心次数/邻居/DFT信号/归一化/两级池化/完整网络及旁路。完整清单见`FMT_architecture_audit_2026-09-17_zh.md`及`FMT_experiment_architecture_catalog_2026-09-17.md`。单中心预检52008644因部署缺少FLowUtils而FAILED/1:0，21次正式训练未提交，无新F1；按用户当前审计要求暂停，不自动恢复。
<!-- fmt-architecture-audit-20260917-end -->

<!-- fmt-single-center-20260917 -->
2026-09-17 用户新增统一FMT比较条件：完整路径无卷积、一个primitive始终固定一个中心，不允许c156式轮换中心。输入点数和后续MLP可因任务不同；傅里叶几何描述和邻居池化配方须统一。历史Task3/5共同搜索含卷积、Task4-c p35/h0/n0逐线轮换中心，不能证明新条件下的统一最佳。当前仅优先核验p35/n0_k06：141维、六频率、邻居逐特征均值＋数值最大值；Task3/5固定原中心，Task4-c一次选离原播种中心最近的有效线，其余全部有效线池化，每束只输出一个向量。只用seed40，共21次正式拟合，MLP78,530参数；保留原Task3/5数据及最新Task4-c196,960/3,000/11,320数据。版本`Verify_FMT_SingleCenter_Task354C_1.1`，协议`docs/FMT_single_center_verification_protocol_1.1.md`。这是新条件下的单种子候选核验，不是旧成绩复现；未比较允许的多个候选前不能称最好。无回复时采用上述已告知默认，不扩展搜索；截至立项无新F1。
<!-- fmt-single-center-20260917-end -->

<!-- fmt-no-convolution-20260917 -->
## 2026-09-17 用户最高优先级：FMT禁止空间卷积

FMT任意探索的完整预测路径禁止空间卷积；包括原始坐标旁路、冻结骨干、融合和解码器。本次同时禁止沿轨迹采样轴的Conv1d，不得以“时间卷积”绕过；Conv1d/2d/3d、转置卷积、函数式及自定义等价卷积均禁止。旧Raw＋FMT/残差、Task4-b raw_fmt和二维FMT＋U-Net/P35卷积融合方案已撤销，不得重跑或进入当前主表、附录、“最佳FMT”结论。独立非FMT卷积对照必须明确标名并分开统计。每个新FMT完整模型在训练前必须递归检查所有分支并记录网络结构/参数，发现卷积直接失败；不得擅自豁免。

Task4-c p35/h0 76,738参数、c156 992,386参数，以及直接fmt_c156/ASAP全连接版本没有卷积，保持原定义。Task3/5旧p35/h0含Raw卷积骨干，不能与Task4-c同名特征配方混称。历史数值/预测/日志仅保留撤销证据，不改写成无卷积成绩。详见 [FMT无空间卷积硬性协议1.1](FMT_no_spatial_convolution_protocol_1.1.md)。本条优先于本文以下全部历史授权和“冻结/当前最佳”表述。
<!-- fmt-no-convolution-20260917-end -->

2026-09-14用户要求`Ablation_Task4C_ConvCapacity_4.18`：新Conv3D 83,918参数，为原4.14 FMT 88,514参数的94.8076%；卷积通道12/24/48、分类头384→96→64→2。原4.14数据、体素缓存、训练器及三种子固定，限定V100，原FMT和219,602参数Conv结果保留作为参考。五项Slurm任务和三份预测已独立核对完成，合并测试F1 0.7499±0.0126（用户独立核验修订），平均训练18.432分钟；协议`docs/Task4C_conv_capacity_protocol_4.18.md`，完整结果见实验日志和主表。该实验与4.16/4.17编码器核验分开。

2026-09-14完成4.16完整输入纠正和4.17单个原设备复核：原4.14 FMT三种子复现0.673624±0.009154；保留原233维并追加165维角度的fmt_v5为0.556003±0.005039；距离＋夹角2.2保留原72维方向分支为0.516540±0.019008；复用原Conv3D为0.811226±0.013250。完整输入、标准化及预测核验通过。4.16一次A100基线差异及由此触发的核验失败保留，4.17只在原V100复核该基线种子，不重跑新方法。科学commit分别dfdc87b1/e5de2877；14个进程、28条事件全部核对。协议为Task4C_retain_direction_protocol_4.16和Task4C_baseline_device_protocol_4.17；结论见experiment_log的task4c-retain-direction-4-16-2026-09-14。4.15错误接入结果不删除；当前纠正任务完成，不自动追加调参。

2026-09-14纠正启动记录：修复4.15移除原72维方向特征的错误，立项`Verify_Task4C_RetainDirection_4.16`；完整复用4.14数据/训练，复跑原233维FMT，fmt_v5保留原233维并追加165维角度，距离+角度2.2保留72维原方向分支。Conv3D复用已经复现的4.14结果。详见`Task4C_retain_direction_protocol_4.16.md`。4.15不能回答单独加入角度的效果，其成绩和源码保留。

2026-09-14完成`Verify_Task4C_Encoders_4.15`：复用Task4-c 4.14的27,000/3,000/10,000样本，纯fmt_v5 5.1（326维）、用户明确选定的距离+夹角fmt_objective_ntod_v2 2.2（396维）及原Conv3D各三种子。固定0.5合并测试F1分别0.320565±0.010570、0.202781±0.032300、0.811226±0.013250；Conv三种子与4.14完全一致。新编码输入无kin/IVD估计或旧72维方向分支，不能把与历史161+72维FMT的差距单独归因于加夹角。科学commit785f96ce，九次训练及独立预测/调度复算PASS；13个成功进程及1个失败预检均保留，无模型文件。协议`docs/Task4C_encoders_protocol_4.15.md`，结论见experiment_log的task4c-encoders-4-15-2026-09-14。当前不自动追加调参。Task1/2/3/5旧含kin基线的命名更正继续有效。

2026-09-14用户新增授权 `Verify_Task1235_AngleFeatures_1.1`：开发fmt_v5 5.1（原FMT+15条同时间中心夹角傅里叶）与fmt_objective_ntod_v2 2.2（冻结距离版+同一夹角分支），在原10个3D条目上配对测试Task1/2/3/5。三种子、旧数据/标签/训练规则保持；Task6/7/8仍暂停。协议`docs/Task1235_angle_features_protocol_1.1.md`，科学commit5fabbe4d；94个调度进程、120分片全部完成，780行主指标与2160行逐尺度指标、原始源标签及容量检查全部复算通过，450个临时模型已清理。完整结果见experiment_log的task1235-angles-2026-09-14及paper_tables_tasks_3d。当前没有自动追加实验。旧objective_fmt_nTDO_v2 2.1及主表不改。


本文件是 Task1–Task8 的唯一任务定义。旧文档若与本文件冲突，以本文件为准。协议自 2026-08-23 起生效；Task5 自 2026-08-26 起加入；Task6/7/8 自 2026-09-09 起加入；历史实验 ID 和输出目录不追溯改名。

2026-09-14最新用户要求推进`mainExp_Task4C_PhysicalLength_4.14`：重新构建27,000训练、3,000验证、10,000测试。Channel单向目标弧长0.08/0.10/0.12，TBL4/5/6，ds均在用户范围内；RK45物理步长、实际半线弧长与最大步数独立校验。每个候选头区内部5个空间块分配3/1/1，评价中心距任意训练中心至少1h且距同头区训练中心不超过4h；共享头区/实例/源场，不宣称独立涡实例泛化。两原模型固定dropout0.15、weight decay0.0001各三种子，验证选模型后测试一次。解析检查、真实流场小预检及六次工程pilot通过，科学commit5c9c9042；协议`docs/Task4C_physical_length_protocol_4.14.md`。Ibex preflight51877188、prepare51877189、encode51877190、train51877191[0-5]、merge51877192共12个任务及六次预测均已独立核对完成。固定0.5阈值合并测试FMT0.673624±0.009154、Conv3D0.811226±0.013250；分流场Channel/TBL为FMT0.722775/0.591833、Conv0.848018/0.751596。结果见experiment_log与paper_tables_tasks_3d；未按新测试调数据距离，旧版本不改写。

2026-09-14用户指定的`mainExp_Task4C_NearbySampling_4.13`已完成：原27,000训练束保持，新增3,000邻近验证和10,000邻近测试，偏移原邻居距离5%–15%后重新积分；共享训练头区和源网格支持。原FMT/Conv3D各无正则化与dropout0.15、weight decay0.0001三种子，邻近测试F1依次为FMT0.943793±0.001428 / 0.989491±0.004233，Conv0.984681±0.001704 / 0.990186±0.005765。18个调度任务和12份预测独立核对完成，科学commit52ae26a4；见`docs/Task4C_nearby_sampling_protocol_4.13.md`及experiment_log。只衡量局部泛化，不覆盖旧4.6空间测试；当前没有自动新增实验。


2026-09-14用户最新要求的完整训练集拟合已完成：`Verify_Task4C_TrainingMemorization_4.12`原27,000训练束固定阈值0.5，原FMT/Conv3D训练F1=0.997007/0.995148，错误26/42，均连续三轮≥0.99；全量预测与样本覆盖独立复算通过。原结构无需增容。额外增容FMT已完成并保留，增容Conv及四组merge在原模型达标后取消，不宣称四组比较完成。科学commit26f52805，见`docs/Task4C_training_memorization_protocol_4.12.md`。当前不继续测试调参，4.6历史测试结果不变。


**2026-09-13 当前推进状态（取代下面的历史推进状态）**：目标会议为 ICLR；按用户决定，停止现有客观化路线，
暂停 Task6/7/8 几何 tokenizer、Point-NN 与混合专家探索（包括 Task36）。Task1/2/3/5 保留冻结主结果。

2026-09-14补齐 **`Ablation_Task4C_GeneralizedCrossEntropy_4.11`**：固定原数据/网络/优化设置，比较加权平滑广义交叉熵q=0/0.3/0.7，14次训练和三种子选择全部独立复算通过。对照/选定正q验证F1为FMT0.392898/0.398741（q0.3）、Conv0.428428/0.432079（q0.7）。无test读取，科学commit97e3f7ea，方法判断见experiment_log。见[4.11协议](Task4C_generalized_cross_entropy_protocol_4.11.md)。

用户在当天后续指定 **Task4-c 3.1**：Channel+TBL各一帧，加入论文lambda2/正oyf头区、RK45双向涡线、清洗、弧长重采样及整束归一化，比较FMT＋多层感知机与Conv3D＋多层感知机。
这取代当天较早“仅整理、不启动训练、新任务未选定”的状态；Task4-a/b 算法与历史结果冻结。
3.1 的[论文预处理与二分类协议](Task4C_paper_bundles_protocol_3.1.md)及完整结果冻结。
用户要求增加dropout/正则化与学习率搜索，改变实际邻居距离、RK45参数/长度；[多尺度4.1](Task4C_multiscale_protocol_4.1.md)已完成28次开发训练，三种子验证F1为FMT0.301773、Conv3D0.420529，未进入测试。
此前 **`mainExp_Task4C_Regularized_4.2`** 完全复用4.1物理样本，原FMT拼接坐标/单位切向的有方向频谱，并对训练几何增强及余弦间隔分类作对照，见[4.2协议](Task4C_regularized_protocol_4.2.md)。每流场15,000训练开发（内部13,500拟合+1,500验证）、5,000固定测试primitive；额外训练增强不计作独立物理样本，原3.1测试每流场500不变。
4.2已完成20次开发训练，三种子验证FMT扩展版0.382105、Conv3D0.430561，未通过验证门槛；最终测试未执行。
此前 **`mainExp_Task4C_LinePooling_4.3`** 已完成：相同物理数据和固定测试，逐线233维固定FMT后共享MLP学习聚合，并比较Conv3D通道数扩大；保留4.2原结构对照、不作合成增强，详见[4.3协议](Task4C_line_pooling_protocol_4.3.md)。
4.3三种子验证FMT0.390572、Conv3D0.429361；4.4隐藏层插值版本已完成，验证0.372803/0.430801；4.5同中心物理尺度配对已完成，验证0.399091/0.436234。见[4.4协议](Task4C_manifold_mixup_protocol_4.4.md)与[4.5协议](Task4C_scale_consistency_protocol_4.5.md)，全部未通过各自原验证门槛。
最新完成的 **`mainExp_Task4C_FinalAssessment_4.6`** 已完成最终评估：关闭4.1–4.5开发，按完整三种子验证结果选择，两方法均选中4.5。明确修订此前助手自加的验证0.6门槛，改为冻结选择后一次测试；旧源码与结果保持。三个全新训练种子95811/95812/95813，仍要求两方法各自合并测试Hairpin F1均值≥0.6；测试不参与配置/阈值/epoch选择。见[4.6协议](Task4C_final_assessment_protocol_4.6.md)。实际合并测试F1均值±样本标准差为FMT0.335526±0.017885、Conv3D0.304615±0.031948，目标未达成；完整证据已登记。
已完成 **`Ablation_Task4C_RepresentationResolution_4.7`**，仅用原训练/验证数据比较FMT有符号频率6/17与Conv体素24³/48³，同方法参数数一致；两种分辨率分别选择学习率后均做三种子复核。验证F1分别FMT0.365484/0.349484、Conv0.427184/0.428600，16次训练及最终选择独立复算通过。见[4.7协议](Task4C_representation_resolution_protocol_4.7.md)。原10,000束测试及4.6结果不变，0.6目标仍未达到。
已完成 **`Ablation_Task4C_Sharpness_4.8`**，固定4.5已选数据、网络、dropout、学习率和尺度一致性，只比较SAM扰动半径0/0.05/0.1；保留普通AdamW对照，同种子三次验证，入口不读取test。见[4.8协议](Task4C_sharpness_protocol_4.8.md)。科学commit81533108，14次训练及独立复算已完成；选定验证F1为FMT0.395441、Conv3D0.429179，未达到目标，原测试不变。
2026-09-14完成 **`Ablation_Task4C_CenterDiversity_4.9`**：保持每流场13,500训练、1,500验证、5,000测试及原head/scale/标签/实例数量，训练中心从4,460增至13,875。真实预处理、完整数据和12次训练/选择独立复算均通过；原采样/更多中心三种子验证FMT0.415513/0.404945、Conv3D0.440833/0.407675。科学commit4ac7e6ab，见[4.9协议](Task4C_center_diversity_protocol_4.9.md)。原采样是4.5同配置的新种子复跑，旧结果和4.6测试不变。
最新完成 **`Ablation_Task4C_SupervisedContrastive_4.10`**：保持原4.1数据、4.5优化设置及4.3分类推断函数，在训练投影上增加监督式对比损失，权重0/0.1/0.5；保留零权重对照和选定正权重的三种子验证。见[4.10协议](Task4C_supervised_contrastive_protocol_4.10.md)。科学commit6bfc357b，14次训练、最终选择及独立复算完成；零对照/正权重验证FMT0.384054/0.394229、Conv3D0.422313/0.421731，不读取测试，目标仍未达到。
多个尺度可来自同一中心，但头区与完整GT实例及源数据支撑不得跨集合；样本数不代表独立物理涡数。
2.1已完成的Channel局部七线中心点实验及结果保持冻结。
现有GT提供hairpin单元支持；1.1四类方案因缺标签被用户改为二类，BiLSTM保留但不运行。用户已授权验证、删除临时验证代码、commit、push和Ibex部署运行。
各版本结论与方法限制见[进展记录](experiment_log.md#progress-2026-09-13)。

术语边界：vortex identification 输出涡核**区域**；vortex classification 区分三维涡**类型**；
vortex coreline identification 输出涡核**曲线**。当前 Task1/3/5 的区域结果不是 coreline 检测结果。
Channel/TBL 的 Task4-a/b 输入是三维定常快照，其 primitive 由速度场 streamline 构成。
Task4-c 3.1沿涡量积分每束10..256条有效线；4.1用27个邻居种子、保留10..27条有效线，实际多尺度追踪；均每线32点、以完整候选头区GT重叠作二分类。历史2.1为局部七线中心分类。vortex lines不是vortex corelines。
项目对象是具有局部信息的线簇几何，不限定pathline、streamline或vortex line；曲线类型由各版本明确选择。
用户说明 half-cylinder Re160/320/640/6400 属于同一仿真家族，原始网格为 `640×240×80×151`，
时空范围为 `[-0.5,7.5]×[-1.5,1.5]×[-0.5,0.5]×[0,15]`；后续采样避开原始前50%时间。
现有十条目主表只包含其中 Re160/640/6400，不能把 Re320 自动计入已完成实验。

**2026-09-10 当前推进状态**：当前只推进Task6，按用户新定义改为**单流场、多尺度primitive的VAE几何重建**：七线geometry→冻结FMT token→VAE→同一簇七线geometry，测试整个未见primitive。每个网络只在一个流场中训练；一般流场播种于原始时段10%–80%，Cylinder于50%–80%且t>=7，完整积分另行保证。原基线`mainExp_Task6_PrimitiveVAE_2.1`保持冻结，详见[Task6协议2.1](Task6_primitive_vae_protocol_2.1.md)；当前修复版本为`mainExp_Task6_Reconstruction_3.1`，使用新signed_fmt10与192维VAE、每流场24万训练primitive，详见[Task6修复协议3.1](Task6_reconstruction_protocol_3.1.md)。新编码与原fmt_all分别记录，不互换方法名称。Task7/8暂不推进；Task4继续暂缓。Task1/2/3/5及旧Task6/7/8冻结源码、配置与历史结果不改写。

**2026-09-09 历史登记**：当时增加的Task6局部流映射查询、Task7遮挡区域补全、Task8短流映射组合及实现、验证、Git推送和Ibex批量运行授权见[Task6/7/8旧协议1.1](Task678_flowmap_protocol_1.1.md)。旧Task8使用旧Task6的逐粒子查询器，不能因当前任务定义改变而自动改用VAE几何重建网络。方法证据和轮廓系数观察见[进展记录](experiment_log.md#progress-2026-09-09)。

## 0. 客观性（Objectivity）基础定义 — 2026-09-10

以下按用户提供的定义抄录，仅整理公式排版并显式保留时间参数；本节不改变既有算法、实验版本或性能指标。

> **Objectivity.** Given a time-dependent reference frame transformation (rotation \(Q\) and translation \(c\)):
>
> \[
> x^\ast=Q(t)x+c(t).
> \]
>
> Objectivity can be formally defined as follows: A scalar field \(s\) is objective if the scalar values remain unchanged under any rigid-body reference frame transformation, i.e.
>
> \[
> s^\ast(x^\ast,t)=s(x,t).
> \]
>
> A vector field \(v\) is objective if the vectors transform as:
>
> \[
> v^\ast(x^\ast,t)=Q(t)v(x,t).
> \]
>
> A second-order tensor field \(T\) is objective if the tensors transform as:
>
> \[
> T^\ast(x^\ast,t)=Q(t)T(x,t)Q(t)^{\mathsf T}.
> \]

这里 \(Q(t)^{\mathsf T}Q(t)=I\)，旋转满足 \(\det Q(t)=1\)，同一时刻全部空间点采用同一个 \(Q(t)\) 和 \(c(t)\)，时间不变。上述 \(v\) 是客观向量场定义中的一般符号，不意味着任意被称为“向量场”的具体物理量都已满足此变换规律。

**同一时刻、任意一对对应物质点之间的相对位置向量天然客观，点间欧氏距离天然是客观标量。** 设
\[
d_{ij}(t)=x_j(t)-x_i(t),\qquad \ell_{ij}(t)=\|d_{ij}(t)\|,
\]
则
\[
d_{ij}^\ast(t)=Q(t)d_{ij}(t),\qquad
\ell_{ij}^\ast(t)=\ell_{ij}(t).
\]
同一时刻相对向量之间的内积和夹角也保持不变。不得以相对位置向量的坐标分量随 \(Q(t)\) 改变为理由，否定其客观性；不得要求客观向量逐分量数值不变。

**后续计算必须独立核查，不能改变对输入的上述判断。** 每次分析先标明正在判断输入、中间量还是输出，并声明它是标量、几何向量还是二阶张量。标量检查数值不变；向量检查 \(v^\ast-Qv\)；张量检查 \(T^\ast-QTQ^{\mathsf T}\)。由若干标量拼成的“feature vector”不因名称含vector就成为按物理空间 \(Q(t)\) 变换的几何向量。

同时间客观相对位置作为编码输入时，必须明确写“输入是客观向量”。若后续沿时间对坐标分量做差分、傅里叶或其他混合运算，另行核查这些运算及输出的变换规律；输出检查失败不推翻输入客观性。若输入是数值不变的距离标量数组，相同固定确定性计算的输出也不变，包括该标量序列的时间差分、傅里叶以及固定网络；不得把向量坐标分量的运算与距离标量运算混为一谈。

## 1. 总体研究命题

Task1/2/3/5 的研究对象是 pathline cross primitive。2D primitive 通常为中心线和 `x±、y±` 共 5 条线；3D primitive 为中心线和 `x±、y±、z±` 共 7 条线。
Task4-c 3.1使用清洗后的10..256条涡线、每线32点、整束质心/最大半径归一化；2.1七线中心分类与1.1论文形态四分类保持历史实现。

FMT 是由 Fourier 变换、`sin/cos`、几何不变量和 aggregation 构成的 **training-free encoder**。这里“training-free”只描述 encoder 本身没有通过标签或重构损失更新的参数；KMeans、VAE 和监督分类器仍然需要训练拟合。
当前研究包含 **3D Task1、Task2、Task3、Task5**，以及仅限 3D 的 **Task4-b proxy-label 实验**。Task4-b 1.1 因跨 split primitive 空间重叠被审计否决；channel 内标签与空间拆分协议冻结为 1.2。`mainExp_Task4B_ChannelToTBL_2.3` 只研究一个 channel source volume 到一个 TBL target volume 的跨 volume 迁移，不把 Task4-b 扩展解释为一般跨流场结论。`mainExp_Task4B_PooledInstanceSplit_3.1` 把 channel 与 TBL 合并、按完整 hairpin 实例留出测试，回答同分布未见实例问题，同样不构成一般跨流场结论。Task1–Task3 和 Task5 在 2D、3D 都有定义，但 2D 扩展暂不进入当前实验计划；Task4 只在 3D 中成立。

## 2. 八项任务的固定定义

| 任务 | 维度 | 输入与方法 | 核心比较 | 主要输出 | 允许的核心结论 |
|---|---|---|---|---|---|
| **Task1：无监督涡区域聚类** | 2D、3D | `primitive -> FMT -> feature -> KMeans(k=2)` | Raw geometry + KMeans、FMT + KMeans，以及必要的无参数几何 baseline | ARI、NMI、固定映射后的 F1/IoU；逐流场与 family macro | FMT feature 是否足以区分涡区域和非涡区域；是否优于 Raw 聚类 |
| **Task2：FMT 作为 VAE 输入** | 2D、3D | 无监督 VAE 编码后，对 latent feature 做 KMeans 二类聚类 | **主比较：Raw+VAE vs FMT+VAE**；FMT direct 只作诊断 | held-out ARI、NMI、F1/IoU；多 VAE seed 分布 | FMT 是否是比 Raw pathline 更好的 VAE 输入 |
| **Task3：有监督 IVD 涡识别** | 2D、3D | IVD 标签监督的涡/非涡二分类网络 | Raw、参数量控制 Raw、Raw+FMT | F1、Average Precision、AUROC、precision、recall；多训练 seed | 加入 FMT 是否提高有监督涡区域识别 |
| **Task4：有监督涡类型分类** | **仅 3D** | 对已定义的 3D 涡型标签做多分类 | 不使用 FMT vs 加入 FMT | macro-F1、每类 F1、balanced accuracy、confusion matrix | 加入 FMT 是否提高 streamwise、spanwise、hairpin 等涡型分类 |
| **Task4-c：Hairpin区域二分类（2.1）** | **仅 Channel 3D 快照** | 局部七线簇几何→Hairpin / Non-hairpin，标签为中心是否在GT区域 | FMT＋MLP与同几何体素化的Conv3D＋MLP | 正类F1、AP、平衡准确率、混淆矩阵、积分有效率 | FMT是否改善隔离空间区间的hairpin区域识别；不作为论文曲面检测F1 |
| **Task4-c：论文预处理线簇二分类（冻结3.1）** | **Channel+TBL各一帧** | lambda2/正oyf头区→RK45→清洗及32点重采样→10..256线整束归一化；完整头区GT重叠二类标签 | 同束FMT＋MLP与Conv3D＋MLP | 合并、分流场及每头区一票的F1/AP等；独立头区和GT实例数、清洗统计 | 两帧空间留出上的表示比较；多线子集不等于独立涡实例，不是论文人工四类/曲面检测 |
| **Task4-c：多尺度线簇二分类（4.1数据、4.6最终评估）** | **Channel+TBL各一帧** | 相同头区/GT二类标签；中心周围27种子、10..27条清洗后涡线，实际改变距离和积分尺度 | 原表示＋有版本的正则化网络，两方法使用同一束几何 | 合并/分流场/分尺度/每头区F1等，三个新优化种子；扩大测试10,000束 | 仅以训练/验证冻结选择后一次测试；4.1–4.5旧验证门槛记录保留；不是新增独立快照 |
| **Task5：不同尺度几何学习** | 2D、3D | 每个 primitive 的邻居距离、积分步长和积分步数可变；积分后统一重采样为固定 `K×L×C`，再做 IVD 监督二分类 | 固定尺度 Task3 迁移、variable-scale Raw、结构匹配 Raw-PCA residual、variable-scale Raw+FMT | unseen-scale confirmation 的 F1、Average Precision；逐尺度、逐流场及 family macro | 模型能否学习跨尺度 primitive；FMT 是否提高 variable-scale IVD 涡识别 |
| **Task6：单流场primitive几何重建（2.1冻结；当前修复3.1）** | 当前 3D | 多时间、多位置、多尺度七线geometry→冻结FMT token→VAE→同一簇完整七线geometry；每流场单独训练，测试完整未见primitive | 2.1原FMT与3.1新signed-FMT分别与各自同数据、同VAE结构及同潜变量维数的Raw geometry→VAE→geometry配对；原实验不覆盖 | 七条对应路径线全时段几何重建误差、同时间粒子间距误差、逐尺度/时长及训练/测试分项 | FMT是否帮助VAE学习和重建该流场的primitive几何分布 |
| **Task7：遮挡区域补全** | 当前 3D | 外部可见 primitive tokens → 隐藏区域材料轨迹；原流场积分自监督 | 同上下文网络与相同可见材料点下的各特征方案 | 隐藏区域位置及相对几何误差、存储/计算开销 | tokens 是否能支持区域间上下文推断 |
| **Task8：短流映射组合** | 当前 3D | 已观测的短流映射 tokens 逐段查询，将预测到达位置输入下一段 | 同逐段解码器设置的特征方案与原轨线插值组合 | 组合轨迹位置和形变误差、分段误差增长 | token 是否支持流映射实际组合；不宣称预测未知未来 |

Task4 的背景类处理必须在首个实验前冻结。Task4-b 1.2 冻结为涡候选内部四类：ordinary streamwise、ordinary spanwise、hairpin head、hairpin limb；non-vortex 使用 `ignore=-1`，不进入四分类损失或指标。若加入 non-vortex，必须建立五分类新版本；两种协议不得混在同一结果表中。

## 3. 不得混用的表述

- Task2 的正确主命题是 `FMT+VAE > Raw+VAE`。`FMT+VAE < FMT direct` 不会否定这个主命题，但说明 VAE 没有进一步改善 FMT 本身。因此不能把 Task2 写成“VAE 提高 FMT feature”。
- Task3 是 IVD 监督的**二分类识别**，不是 streamwise/spanwise/hairpin 多分类。后者统一属于 Task4。

## 4. 所有 Tasks 实验共享的最低协议

1. **数据拆分**：同一数据集内实验按时间片拆 train/validation/test，不得随机拆空间 seed；pathline source window 必须一起计入时间泄漏检查。预先冻结的跨 volume 迁移可改用完全不同的 source volume 作训练、target volume 作一次测试，但模型参数、normalization、停止规则、feature、proxy-rule percentile 和超参数都必须在不读取 target 性能指标的情况下冻结；target 人工标签只能用于预先声明的测试标签、测试总体构建及最终评估。
2. **test 冻结**：test/confirmation 标签不得参与 feature 选择、cluster-to-class 映射、checkpoint、threshold、alpha 或超参数选择。
3. **标签冻结**：IVD 定义、空间平均区域、阈值和边界处理必须写入 config；
   - 当前论文中所有采用 whole-field IVD 二分类的 3D 实验统一固定为 **IVD p95**：Task1/Task2 将其用于评估，Task3/Task5 将其用于监督。依据 `Ablation_Task23IVDPercentile_1.2`，p95 在 p80、p85、p87.5、p90、p92.5、p95 的完整扫描中给出最大的 Task2 F1 增益以及 Task3 F1、Average Precision 增益。p80–p92.5 只作为标签敏感性分析；后续若改变阈值，必须建立新实验版本并与已有 p95 结果并列报告。
4. **论文主表使用任务级统一 FMT 配方**：Task1、Task2、Task3 的 3D 论文主表必须分别从既有候选中选择一套任务级配方，并在全部 10 个数据条目上保持相同的 FMT feature blocks、FMT 侧权重/缩放和 FMT 后处理维度（例如 PCA 维数）。选择只能使用 development 数据；允许用新随机种子复跑既有候选，但不得为了统一主表新增未搜索过的超参数。旧逐flow或逐physical-family最优配方、代码和结果完整保留，只能进入补充表和可视化，不能与统一配方混合计算论文主表 macro。Task5 的 FMT encoder 配方同样固定；其唯一例外是研究问题本身预注册的邻居距离、积分步长和积分步数会按尺度 tuple 变化。更严格地同时固定 VAE 或监督网络配方是允许的，但不能因此改变 Raw/FMT 两臂的公平比较。
5. **统一预处理**：normalization 只在 train 上拟合；Raw 与 FMT 的维度、缩放和邻居权重必须明确。
6. **容量控制**：训练网络必须报告总参数和可训练参数；至少包含参数量更多的 Raw 或结构匹配的 Raw residual 对照。
7. **重复实验**：神经网络至少 3 个训练 seed。KMeans 必须固定并报告 `random_state` 和 `n_init`。
8. **失败保留**：失败版本和反例必须留在总实验记录中；新方法不得覆盖旧输出。
9. **可复现证据**：每项结论必须指向 experiment version、config、git commit、逐次 CSV、汇总文件和设备记录。
10. **IBex部署**：一般任务倾向去ibex部署而不是本机。只要任务运行时间你估算超过5分钟，就必须只去ibex部署。
**Ibex（KAUST 集群）工作流:**
>1. 本地开发 → `git push`；登录ibex：ssh -x  zhanx0o@glogin.ibex.kaust.edu.sa；ibex 上找到对应项目文件夹 `git pull` 后**核对 HEAD 一致**再跑。
>2. 数据：`scp`/`rsync -P` 到 `/ibex/user/zhanx0o/FLowDataFolder/`（大文件）或
   `$HOME/DeepVortex/FLowDataFolder`；提交前核对字节数一致。
>3. sbatch 模板（参照 `PyflowVis/ibex_bash/refframe_3d_v1.sh`）：array 任务矩阵、
   `module load cuda/11.8`、`conda activate deepvortex`、`slurm_logs/` 输出、
   `export PYFLOWVIS_DATA2D/3D=...`、每任务独立输出目录 + metrics.json、脚本 LF 行尾、
   python 失败 `|| exit 1`。
>4. 回收：解析各任务 json 汇总成表，写入实验 log 文档并标注 job 号。


11. **模型文件不长期保留**：checkpoint 只允许作为同一实验依赖链中的临时文件存在于运行设备；最终评估记录写入逐次 CSV/JSON 和总实验表后必须删除。不得把 `.pt`、`.pth`、`.ckpt` 或包含模型的压缩包下载到本地、提交 Git 或作为实验归档。除非用户明确要求，复现必须依靠 commit、config、seed、数据划分和设备记录重新训练。
12. **输出规则**：
    - 研究 Markdown 只放入 `docs/`；仓库根目录的 `README.md`、`AGENTS.md` 属于项目控制文件，
      临时部署快照中的副本不视为独立研究文档。完整实验流水只追加到
      `docs/experiment_log.md`，Ibex
      作业只追加到 `docs/ibex_run_registry.md`；Task3 主实验、Verify 核心结论、Audit
      核心结论和论文表分别维护在 `docs/mainExp_Task3_3D.md`、
      `docs/Verify_experiments.md`、`docs/Audit_experiments.md` 和
      `docs/paper_tables_tasks_3d.md`。不再为每轮参数扫描新增一份 `Verify_*.md` 或
      `Audit_*.md`。
    - CSV、JSON、NPZ、JPG、PNG、PDF 等机器结果只放入
      `outputs/<experiment_id>/`。禁止新增 `docs/results/` 或根目录 `output/` 内容。
      2026-09-01 前的 `output/` 是只读历史镜像，不追溯改名；新代码和文档不得继续
      引用它作为默认输出路径。
    - 实验入口与编排代码统一放入 `experiments/`；文件名必须保留 Task 和实验 ID，
      例如 `experiments/Run_Task2_3D_Main.py`。通用库保留在 `FMT_Utils/`、
      `FLowUtils/`、`DeepUtils/`；一次性仓库维护工具放入 `tools/`。禁止继续在仓库根目录新增
      `Audit_*.py`、`Evaluate_*.py`、`Diagnose_*.py`、`Build_*.py`、`Select_*.py`
      等实验脚本。
    - 临时传输包和可删除中间文件只放 `tmp/`；Slurm 日志应进入对应
      `outputs/<experiment_id>/logs/`。不得把 checkpoint 当长期输出。
    - 开源前历史整理例外（2026-09-05）：不再被当前流程引用的旧脚本、配置及配套测试，
      可以在逐文件校验并完整备份后移出工作目录；原路径和校验值保存在
      `docs/research_archive_manifest.json`，恢复说明见 `docs/repository_maintenance.md`。
      不删实验流水、失败结论或结果文件，不改冻结算法。实验版本仍保留在配置和记录中，
      相同执行逻辑应复用入口，不因每次参数扫描复制一个脚本。



## 5. 各任务的额外协议

### Task1

- KMeans 只能在 train feature 上拟合。
- 二类 cluster 到涡/非涡的映射只能由 validation 冻结；test 主指标优先使用无需标签映射的 ARI/NMI。
- 必须同时报告 `Raw direct` 和 `FMT direct`；只展示 FMT 的 3D 图不能证明增益。
- Task1 的验收只依据聚类性能；不要求 FMT 对时变观察者变换严格不变。若另做客观性分析，只能作为独立附加实验，不能改变 Task1 成败。

### Task2

- 主实验的 VAE 先按 FMT 在 train/validation 上开发并冻结；同一 physical-family 内，Raw+VAE 与 FMT+VAE 使用完全相同的 VAE 架构、latent 维数、优化器、学习率、训练步数、KL 权重和 checkpoint 规则。不得为 Raw 单独搜索更强 VAE 后替换主 baseline。
- 输入维数不同导致参数量不同，必须明确报告。独立优化的 strongest-Raw 或 dimension-matched Raw（例如 train-only PCA）可以作为附录压力测试，但不得取代 same-VAE 主表。
- 主结果是 `FMT+VAE − Raw+VAE`；`FMT direct`、`Raw direct` 和 reconstruction loss 只作解释性诊断。重构误差不能替代 latent 聚类质量。

### Task3

- 主标签为冻结的 IVD 二分类协议。
- 每个方法按自身 validation 指标独立选 checkpoint；不得用“必须比 Raw 高多少”作为论文最终模型选择规则。
- 除 Raw-wide 外，建议增加结构匹配的 `frozen Raw + Raw residual`，排除两阶段 residual 训练本身带来的收益。
- Average Precision 是不依赖最终二分类阈值的主要排序指标；F1 是 validation 冻结阈值后的主要区域识别指标。

### Task4

- Task4-c 当前为Channel+TBL多尺度线簇二分类，见 `Task4C_multiscale_protocol_4.1.md`；3.1预处理与2.1局部七线定义和结果冻结。下列Task4-b部位标签和IVD阈值不应用于Task4-c。GT实例0有效；3.1/4.1完整头区被单一实例覆盖至少50%为正、完全无GT重叠为负，其他部分重叠排除。完整头区、GT实例和原生源节点支撑隔离。
- 只允许 3D 数据；必须报告每类样本数和 class-balanced 指标。
- 涡区域定位误差与涡型分类误差应分开统计。
- Task4 开始前必须单独建立标签来源、类别定义和跨数据集名称映射文档。
- Channel Task4-b 1.2 的标签定义、阈值选择、空间/实例拆分和物理质检冻结在 `docs/Task4B_channel_proxy_label_protocol_1.2.md`。其中 `norm(omega-mean_xy(omega)(z))` 必须称为 channel-profile vorticity deviation，不得称为标准 IVD，也不得据此改写 Task1/Task2/Task3/Task5 的 whole-field IVD p95 协议。
- `VortexIds>0` 只提供人工 hairpin support 与实例编号。ordinary streamwise/spanwise orientation 是 velocity–curl 45°规则产生的 proxy；hairpin head/limb 是 velocity–curl、`omega_y_prime` 及分量占优规则产生的 proxy。四类标签都不得表述为人工 anatomical ground truth；位置序和左右涡量符号只作逐实例物理质检。
- `mainExp_Task4B_ChannelToTBL_2.3` 的 TBL 测试总体是 **oracle-enriched population**：先按冻结的 per-volume p80.5 规则产生 vortex candidates，再显式加入全部人工 `VortexIds>0` hairpin support。该实验衡量已知人工 support/identity 条件下的四类 proxy 迁移，不衡量 blind whole-field hairpin detection。
- 合并 channel+TBL 的 3.1 按每 volume 内完整 `VortexId` 实例分块随机拆分（2 test / 1 validation / 7 train），ordinary cube 继承最近 hairpin 实例的拆分，并删除距任一测试种子小于一个 primitive 支持半径（`16×spatial_step+offset`）的 train/validation cube；测试 cube 不删除。缓冲半径必须在训练前用真实标签量化后冻结并写入 config；改变半径或拆分规则必须建立新版本。
- Channel→TBL 2.3 使用全部 channel source rows 训练，并以连续三 epoch source 零错误和正 minimum true-logit margin 作为进入 TBL 一次评估的门禁；它回答“完全拟合 source 后能否迁移”，不是常规 source holdout 泛化。必须报告 streamline coverage，并把无有效 streamline 的人工 hairpin 行及完整 VortexId 计入 coverage-adjusted 失败指标。即使 Ibex 独立审计 `PASS`，证据仍只来自一个 source volume 和一个 target volume。

### Task5

- Task5 是 Task3 的尺度扩展，监督目标仍是冻结的 IVD 涡/非涡二分类；不得同时改变标签定义后把差异归因于尺度学习。
- “尺度”至少完整记录三项：中心到邻居种子的距离、数值积分步长、积分步数；总积分时间由后两者的乘积给出。最终送入网络的线数 `K` 和每线采样数 `L` 必须固定。当前 3D 主协议使用 7 条线，2D 使用 5 条线。
- 空间尺度、时间步长和积分步数的组合必须在 train、validation、confirmation 间按 tuple 拆分。主 confirmation 至少包含训练中未出现的组合，并明确区分尺度插值与尺度外推。
- 尺度 tuple 的分配必须与空间 seed 和 IVD 标签独立，且每个时间片中各 tuple 数量近似均衡；不得把大尺度主要分给涡区、小尺度主要分给背景。
- 主表至少包含 variable-scale Raw、同结构同维度 Raw-PCA residual、variable-scale Raw+FMT；同时报告固定尺度 Task3 模型直接迁移到 variable-scale confirmation 的结果。
- 除总体 F1 和 Average Precision 外，必须输出逐尺度 tuple 的指标，避免总体平均掩盖某一尺度范围的系统失败。

## 2026-09-06 用户重新定义 Task4-b（4.1 起）

以用户本次明确要求为准：channel+TBL 训练同一个有监督四分类网络。四类为 ordinary-streamwise、ordinary-spanwise、hairpin-head、hairpin-leg；在涡区内只根据 hairpin 标注内/外与 velocity–curl 更接近平行/垂直决定，45°归平行，反平行同属平行，hairpin 内平行为 leg、垂直为 head；不使用额外 head 判据或角度排除带。每流场重算全域 IVD，通过原始正标注单元中心及目标体素中心的全覆盖约束确定 a，固定 IVD>0.9a 为涡区。该 Task4-b 标签协议不修改 Task1/2/3/5 的 p95 规则。

首先执行 `Verify_Task4B_VelocityCurlMemorization_4.1`：七条流线 primitive、固定 FMT、单个共享 fmt_only 四分类网络；fit=evaluation，只作有限样本记忆验证。具体离散支持、归一化、抽样、训练、通过判据见 [4.1 协议](Task4B_velocity_curl_protocol_4.1.md)。历史 1.2/2.x/3.x 的标签与结果保持不变，不混入本次结果；后续泛化实验另立版本。

### 当前 Task4-b 阈值更新（2026-09-06，4.4）

用户在软件检查并咨询CFD专家后指定 Channel IVD>5.785、TBL IVD>0.08；此固定物理阈值取代4.1–4.3的hairpin最小IVD阈值定义，旧配置与结果保留。当前 Other_Task4B_ProxyGTThreshold_4.4 只更新完整ground truth及三维图，不训练；四类方向规则、channel+TBL同一网络的后续目标不变。阈值以下hairpin标注不强制补回，覆盖率单独报告。该修改只适用于Task4-b，不改变Task1/2/3/5的p95。

### 当前 Task4-b 小区域分割训练（2026-09-06，5.1）

用户已授权新训练：channel与TBL各900个训练、100个测试小区域，含全部hairpin实例包围盒及同尺度随机滑窗，共享一个FMT四分类网络。mainExp_Task4B_PatchSegmentation_5.1冻结实例数Channel66/7、TBL52/6，跨集合patch至少1体素间隔；各点7条33点流线的积分与原生速度插值节点限制在其patch内部。保持4.4固定IVD阈值，非涡区不进四分类。训练200epoch后最终test一次、无test选模，输出完整test涡区分割与覆盖。细节见[5.1协议](Task4B_patch_segmentation_protocol_5.1.md)，旧版数值保留。

### Task4-b 网络、学习率和几何差分探索（2026-09-06，5.2）

用户授权调整网络结构、学习率，允许训练样本过拟合，并从几何点差构造curl代理量。Other_Task4B_GeometrySearch_5.2冻结8个候选，以5.1原训练patch内部、有1体素间隔的验证集选模型及epoch；最终恢复原900+900训练区域重训。七线primitive保留33点，比较等弧长与保留相对速度大小的等时间几何，附加邻线空间差分、梯度/curl序列的FMT和局部描述量。所有输入均来自patch内部速度积分几何，真实curl、IVD及hairpin成员关系只用于监督，不进入网络输入。原5.1的test只在冻结选择后评估一次，并声明为重复使用的benchmark，不作为新confirmation。完整预注册见[5.2协议](Task4B_geometry_search_protocol_5.2.md)，方法结论只记入experiment_log。


## 2026-09-15 后续双集合评价偏好

用户要求后续仅划分训练集和测试/评估集，不强制第三份数据。新实验按两集合组织；具体协议须说明评估集是否用于选择池化、学习率、训练轮次、融合系数或分类阈值。允许报告用于模型选择的训练外评估成绩，但必须同时说明其选择用途，不能称整个开发过程都未见的独立测试。若预先固定模型与训练/阈值设置，则两集合也可以保留不参与选择的测试评价。

本条仅更新后续数据组织偏好，不改变已完成实验的文件角色与指标，不将本批搜索验证F1改写成既有最终测试F1；旧3,000验证与10,000测试记录继续并列。p35/h0的141维来源及Task3/5与Task4-c实际网络见FMT_p35_h0_structure_2.2_zh.md。


### Ablation_FMTv8_NormFrequency_3.1（2026-09-15）

用户指定p35/h0，统一FMT几何归一化、邻居倍率和后置特征标准化，四配方×K=2/4/6/10/16共20组；中心4K−1、方向12K、邻居逐特征均值与最大值各4K−1，共24K−3维。Raw分支和h0训练规则冻结。直接在原validation作epoch/阈值/配置选择，不新增test拟合。协议`docs/FMT_v8_norm_frequency_protocol_3.1.md`，已完成的全部数据见`docs/paper_tables_tasks_3d.md#fmtv8-norm-frequency-results-3-1-2026-09-15`，方法判断见experiment_log同锚点。
