# FMT 项目研究协议

<!-- task4c-gt-head-coverage-final-20260917 -->
2026-09-17 `mainExp_Task4C_GTHeadCoverage_1.1` 已完成；取代此前“训练中/等待浏览器检查”的状态。科学commit `c3952a7b`，固定c156/seed96721、992,386参数，196,960训练/3,000验证/11,320测试；132个GT每个新增30训练＋10测试，旧数据前缀和验证保持。113轮训练、验证选择63轮、25.712分钟；验证F1=0.921547，新完整测试F1=0.912008，同一个新模型在原10,000束子集F1=0.922946。新增正类头部1,320束识别1,071束、漏检249束，召回81.14%；132个GT均至少识别3/10束。这是单种子、共享GT实例的局部评估，不改写历史c156三种子结果。

重试训练52001957与导出52001958均COMPLETED/0:0；前91轮训练排列及验证F1与失败52000896完全一致。原退出根因仍未确证：stderr无栈、CPU20:00.904、只有STARTED。新运行实际CPU限制为unlimited，累计CPU25:47.310；换节点并禁用资源限制继承后一遍完成，不能据此单独证明旧CPU限制就是原因。失败证据、缺失的结束事件及取消任务保留。独立审计通过：r2七个实际进程/13事件、全部预测与原数据哈希、选择轮次和GT配额。

工作台1.5已上线，旧1.4入口跳转1.5并保留旧HTML。第三页测试全部11,320束，训练展示新增3,960束；每个GT均可查看正确识别与漏检。最终交互检查发现全局GT体单元查询在重叠区域可能返回另一有效GT，已逐个GT独立验证全部5,280新增中心的实际包含关系，再按其播种GT计数；每束只计一次，不改标签/训练/预测。构建强制每GT10/30配额，控件脚本带内容哈希以避免旧缓存。第一、二页内容继承1.4，形状解释仍是原p35模型。

证据：`outputs/mainExp_Task4C_GTHeadCoverage_1.1/{independent_results_audit,browser_checks,local_delivery_status}.json`，查看器 `outputs/Other_Task4C_GTHeadCoverage_1.1/viewer/head_coverage_manifest.json`；协议 `docs/Task4C_GT_head_coverage_protocol_1.1.md`、`docs/FMT_analysis_workbench_protocol_1.5.md`。入口 `http://127.0.0.1:8767/Other_FMT_AnalysisWorkbench_1.5/index.html#results`。当前请求完成，不自动新增训练或调参。
<!-- task4c-gt-head-coverage-final-20260917-end -->

2026-09-16 PointNet基线1.1已部署：科学commit b4c8a80c，Ibex51932378–51932382共七进程，pointnet_small三个种子96611–96613。保留两个T-Net、BatchNorm/ReLU、max和官方正交正则，小宽度76,749参数（FMT76,738）。CPU/V100、原18份数据文件哈希及32真实训练束小拟合全部通过，小拟合F1=1.0、loss0.695694→0.014498、批128反向正常；仅为工程检查。正式三种子已进入GPU队列，无新测试F1。193,000/3,000/10,000数据和优化/选择规则固定，原三几何基线继续；协议docs/Task4C_pointnet_protocol_1.1.md，证据outputs/Ablation_Task4C_PointNet_1.1/startup_evidence.json。


2026-09-16追加用户要求的经典PointNet，版本 `Ablation_Task4C_PointNet_1.1`：沿用同193,000/3,000/10,000数据、三种子和优化规则。保留输入3×3及特征11×11两个T-Net、逐点BatchNorm/ReLU、全局max、分类MLP和官方0.001特征变换正交正则；缩小通道后的pointnet_small总76,749参数（FMT76,738），不称论文原始宽度版本。CPU顺序/填充/正则/梯度与容量核验通过；计划独立Ibex部署，不重跑或改变前三基线。协议docs/Task4C_pointnet_protocol_1.1.md。


2026-09-16最新Task4-c新增三基线 `Ablation_Task4C_GeometricBaselines_1.1` 已部署Ibex：冻结BottomDensity1.2的193,000/3,000/10,000数据，切线+曲率池化MLP / 整束Point-NN固定点云特征MLP / 逐线BiLSTM池化MLP，总参数76,782/76,704/76,786，参照FMT76,738，各三种子96611–96613，训练及数据规则不改。科学commit d20c2458，修复Point-NN高频函数放大float32跨批误差，改内部float64后V100预检通过；首链失败及取消完整保留。现链51930208–51930213（15进程）已提交，原18文件哈希与GPU检查通过，32真实训练束三方法小拟合F1均1.0、批128反向正常；这是工程检查，不是正式测试成绩。全量编码/九次训练等待依赖调度，无新测试F1。协议docs/Task4C_geometric_baselines_protocol_1.1.md，启动证据outputs/Ablation_Task4C_GeometricBaselines_1.1/startup_evidence.json；不得修改旧数据或自动搜索。


2026-09-16新增授权 `Ablation_Task4C_BottomDensity_1.3`：只追加8³/12³ Conv3D，各三种子96611–96613，共六次训练；完整复用1.2的193,000/3,000/10,000数据，全部文件哈希固定，不重新采样。相同72,192参数网络、V100和原训练器不变；FMT/Conv16/Conv24复用1.2已核验成绩。科学commit12a0ffa7已公开推送，Ibex51927009–51927013共11进程已提交（含六次训练），CPU/V100及全部18个原数据文件哈希核验通过；2026-09-16全部完成：六次训练、18预测、11进程22事件独立核验通过，原数据哈希相同。Conv8/Conv12测试F1为0.855112±0.002812/0.896707±0.003988，平均训练38.134/54.009分钟；无模型文件，结果已入日志和主表，当前请求完成，不自动追加调参；协议docs/Task4C_bottom_density_protocol_1.3.md。

2026-09-16用户要求 `Ablation_Task4C_BottomDensity_1.2`：沿用已完成1.1，仅将训练由38,600扩大五倍至193,000；保留原训练前缀，并在相同头区/类别/实例/尺度/完整或下半区采样池为每个来源样本新积分四束。Channel98,500、TBL94,500；验证3,000/测试10,000文件不变，原标签/负类规则/积分/空间块不变，下部Hairpin偏置保持。仅p35/n0_k06与同72,192参数Conv16/24，各三种子；训练器不改。科学commit a7643890已公开推送，Ibex51920910–51920917共19进程已提交，含九次训练。2026-09-16全部完成：193,000/3,000/10,000数据核验、九次训练、27预测、19进程38事件独立复算通过。FMT/Conv16/Conv24测试F1为0.888404±0.011497/0.913490±0.001233/0.933198±0.006292，平均训练28.565/72.521/141.377分钟。原评价文件相同，无模型文件，结果已入日志和主表；当前请求完成，不自动追加调参；协议docs/Task4C_bottom_density_protocol_1.2.md。旧1.1结果冻结。

2026-09-15最新用户要求Ablation_Task4C_BottomDensity_1.1：停止修正版v2剩余Conv48，回到原4.14缓存，仅增加底部Hairpin训练采样，不新增积分长度。已取消51914157_9–11及51914158；v2完成9次训练，FMT/Conv32/Conv36测试F1分别0.734754/0.838906/0.855863，原结果保留，48无最终测试。新实验保留原40,000束及原验证/测试全部文件，仅给原训练集Channel62/TBL54实例各追加100束，共51,600束；此数量为已说明的默认方案。原标签、负样本和单向长度Channel0.08/0.10/0.12、TBL4/5/6不变。最佳p35/n0_k06和同72,192参数Conv8/24/32各三种子；协议docs/Task4C_bottom_density_protocol_1.1.md。首次2ea8da44的模型预检/TBL通过，Channel实例11因额外GT内部硬下半区限制失败，18个依赖进程取消；已撤回新增限制，改原候选完整池与下半区交替采样，旧标签/积分/距离不变。科学commit99cb6438已推送，独立目录FMT_Task4C_BottomDensity_20260915_r2，51919126–51919132共21进程已提交；116实例348束真实预检全部通过，正式追加已完成，全量核验通过：38,600训练/3,000验证/10,000测试；116实例各100束，旧训练全部字段前缀相同，原验证/测试文件哈希相同，原负样本不变。2026-09-16已完成12次训练与汇总：FMT/Conv8/Conv24/Conv32测试F1为0.760842±0.011395/0.709658±0.004362/0.831460±0.012406/0.853504±0.008073，平均训练7.652/5.348/19.149/46.661分钟。36份预测、21进程42事件全部复算通过，无模型文件。结论见experiment_log，结果已入paper_tables_tasks_3d；当前请求完成，不自动调参。

2026-09-15最新纠正（优先于下文同名v2历史）：用户否决13实例留出、局部标签、近邻负样本配额和固定150轮。commit9769c68d运行标为废弃，剩余9个Conv及汇总已取消，原证据保留。v2名称不变，内部execution_revision=restored_4.14；恢复4.14整头区标签、旧负类候选/采样、同实例空间块及距离规则、验证选最佳轮次/平台降lr/早停，不专门留出完整实例（0个）。只保留长度变化和底部正类播种加密；132,000束沿旧27:3:10比例，89,100/9,900/33,000。指定p35/n0_k06与72,192参数Conv32/36/48各三种子。两真实流场1,104束小预检与CPU模型核验通过；科学commit1e088e49已公开推送；Ibex51914151–51914158共22进程于16:52:59–16:53:00 UTC提交，含12次训练，完整编号见运行登记。尚无新F1。协议docs/Task4C_instance_coverage_protocol_9.15_v2.md；旧规则排除的实例不改标签补入，评价实例必须在训练中出现。

2026-09-15最新用户确认`task4-c-v9.15_v2`长度均为单向：Channel[0.08,0.12]、TBL[3,5.5]。正式配置已解除待确认状态，保持132,000束及每实例3束head；使用指定p35/n0_k06（最大半径、训练集逐特征标准化、6频率），与32³/36³/48³小Conv比较。科学commit9769c68d已公开推送；Ibex51911059–51911066共50进程已提交，依次GPU预检、132实例pilot、实测空间检查、正式构建、全量数据核验、编码、12次训练、汇总。正式132,000束已构建并通过全量数据核验，396必选head齐全；每实例覆盖全部11/12档长度。编码/训练依赖链继续执行，暂无新F1，状态见docs/ibex_run_registry.md。用户已明确允许公开整个当前分支，此授权取代历史公开推送待确认条目。

2026-09-15用户指定`task4-c-v9.15_v2`：确认保持132,000束（每实例500正＋500负），增加Channel11档/TBL12档长度，每实例3束必选head保留。长度区间[0.08,0.12]/[3,5.5]的单向或双向含义仍待回复，正式配置明确待确认并阻止采样/训练；尚无v2数据或F1。原v9.15预检51907304失败、47依赖进程开始前取消；v2预检51908822继续定位最大池化问题，修复平均/最大池化后51908979在V100通过（科学commitaf392332），模型参数不变。查看器1.3新增GT Head＋Non-hairpin模式和TP/TN/FP/FN着色，目前展示旧4.14结果；原页面刷新即可进入。协议docs/Task4C_instance_coverage_protocol_9.15_v2.md，失败与预检全部登记。

2026-09-15新增`mainExp_Task4C_InstanceCoverage_9.15`（用户task4-c-v9.15）：每个Channel/TBL Hairpin实例目标500正＋500负，其中3束由GT内部velocity–curl夹角>45°的head播种。用户已确认有效播种点局部标签；重叠扩张采样盒同集合，67/7和52/6实例划分，119,000训练/13,000测试。132实例792束小预检、两实例各1,000束配额及CPU模型/体素检查通过。p35 141维76,738参数，对比32³/36³/48³小Conv72,192参数；150固定轮次后三种子测试一次。科学commit433c71b4，作业51907304–51907309（48进程）已提交，尚无新F1。协议docs/Task4C_instance_coverage_protocol_9.15.md。查看器新增预测两类独立数量控制；旧实验不改。

2026-09-15新增`Other_Task4C_BundleVisualization_1.1`：已生成按预测二分类着色的真实涡线簇＋半透明GT曲面查看器，可切换Channel/TBL、训练/测试及已完成模型。首版FMT4.14与Conv32³4.22，seed96611；960束/16,981条有效线，物理坐标、VTP标签与概率、GT边界核验PASS。导出科学commitd5b778f9、CPU作业51906466完成；本地HTML和六个VTP位于outputs/Other_Task4C_BundleVisualization_1.1/viewer。浏览器交互核验因本机origin访问的自动审批拒绝而待授权，不能称交互测试完成。协议docs/Task4C_bundle_visualization_protocol_1.1.md；旧指标不改。

2026-09-15完成Ablation_FMTv8_NormFrequency_3.1：20组公共几何/特征归一化×DFT频率，420次拟合及21项原p35复现全部完成。唯一共同最佳n0_k06（141维，最大半径、训练集逐特征标准化），原验证F1 Task3/5/Task4-c=0.789635/0.751417/0.740214，三任务均值0.760422，相对原p35+0.97%。直接使用原验证集合进行选择，不混入历史test。Task3/5总142,914参数，Task4-c最佳76,738参数；GPU 6.572卡小时。441预测、333源文件、149调度（4项启动前取消）、145执行/290事件独立核验PASS，无模型文件。科学commit0d77e3e5/c61434ac，调度适配器9b6b8451；协议docs/FMT_v8_norm_frequency_protocol_3.1.md，完整表和方法结论见paper_tables_tasks_3d/experiment_log同名结果段。当前请求完成，不自动追加实验。

2026-09-15用户更新后续数据组织偏好：新实验默认训练集＋测试/评估集，不强制额外划分第三份。若该评估集参与配置、轮次、融合系数或阈值选择，报告必须明确其选择用途，不能称整个开发过程未见的独立测试；若这些设置预先固定，则可进行不参与选择的两集合测试。历史实验的validation/test文件角色、分数和选择记录保留，不通过改名改变用途。本轮仅解释p35/h0并记录偏好，不新增训练。结构详解见docs/FMT_p35_h0_structure_2.2_zh.md。

2026-09-15完成Ablation_FMTv8_Search_2.2：50配置（48不同表示）及9项训练配置，1,239次主验证拟合；共同选定p35/h0（141维），对原V8三种子配对测试126次。Task3/Task5/Task4-c的选定F1为0.728163/0.658147/0.708685；三任务等权F1相对变化+3.01%，未达到15%–20%目标。21对重复、42对等价结果、全部预测及277进程/554事件核对通过；无模型文件。科学commitbc09a242，执行并发初始12后提高至24，预算不变。旧v8.1和停止的2.1均保留；详见docs/FMT_v8_search_protocol_2.2.md及experiment_log的fmtv8-search-final-2-2-2026-09-15。

2026-09-14完成FMT v8.1五组比较：161/233/326/398/100维均在Task1/3/5和Task4-c 4.14配对比较。Task4-c五组三种子完成，V8为0.728480±0.004107、71,490参数、平均训练2.206分钟；小Conv采用用户独立修订0.7499±0.0126。Task1/3/5共90分片、690行结果、所有源标签/尺度/预测与63进程126事件核验通过；V8三个任务的10流场平均F1分别0.399681/0.666199/0.638731。所有方法无kin/IVD估计；Task3/5五组同总参数142,914。科学commita25bcc0d/0ae17ea2，审计aa0437f1，无模型文件；协议docs/FMT_v8_protocol_1.1.md，方法结论见experiment_log。当前请求全部完成，不自动追加调参。

2026-09-15用户明确要求提交`Ablation_Task4C_ConvResolution_4.22`：固定4.18小Conv3D的83,918参数及原4.14数据/评估，32³/36³/48³各三种子；24³仅作为用户修订0.7499±0.0126的历史参考。科学commit223ea781，准备51905885[0–2]、训练51905886[0–8]、汇总51905887共13进程已提交Ibex。GPU预检/训练限定V100，batch128不变，64 GiB主机内存。协议`docs/Task4C_conv_resolution_protocol_4.22.md`；此次提交授权取代原等待确认状态，不表示已经复核24³修订原因。

2026-09-14完成`Ablation_Task4C_FMTv6v7_4.20`：用户指定V6=中心23＋邻居138＋方向72（233维），V7=中心23＋方向72（95维）；原4.14数据和训练、V100及三种子96611–96613不变。V6三次新训练逐样本预测完全复现原FMT，测试F1 0.673624±0.009154；V7为0.723036±0.012807，参数70,850。原Conv3D0.811226±0.013250保留。科学commit63a3cf4d，8进程/16事件及6份预测独立复算PASS，无模型文件；协议`docs/Task4C_fmt_v6_v7_protocol_4.20.md`，结论见experiment_log的task4c-fmtv6v7-4-20-2026-09-14。本次请求完成，不自动新增实验。

2026-09-14完成`Ablation_Task4C_FMTv5Blocks_4.19`：固定seed96611，完整398维验证F1=0.587302；分别移除中心23/邻居138/方向72/角度实部90/角度虚部75后为0.568345/0.588652/0.301172/0.646635/0.631214。保持398输入、109634参数及同初始化，五次重新训练，仅用27,000训练/3,000验证，不读test。科学commitc0d33df0；7进程和14条事件、六份预测独立复算通过。单种子整块诊断，方法判断见experiment_log的task4c-fmtv5-blocks-4-19-2026-09-14；协议`docs/Task4C_fmt_v5_blocks_protocol_4.19.md`。当前请求完成，不自动追加实验；其他任务独立继续。

2026-09-14用户要求`Ablation_Task4C_ConvCapacity_4.18`：新Conv3D 83,918参数，为原4.14 FMT 88,514参数的94.8076%；卷积通道12/24/48、分类头384→96→64→2。原4.14数据、体素缓存、训练器及三种子固定，限定V100，原FMT和219,602参数Conv结果保留作为参考。五项Slurm任务和三份预测已独立核对完成，合并测试F1 0.7499±0.0126（用户独立核验修订），平均训练18.432分钟；协议`docs/Task4C_conv_capacity_protocol_4.18.md`，完整结果见实验日志和主表。该实验与4.16/4.17编码器核验分开。

2026-09-14完成4.16完整输入纠正和4.17单个原设备复核：原4.14 FMT三种子复现0.673624±0.009154；保留原233维并追加165维角度的fmt_v5为0.556003±0.005039；距离＋夹角2.2保留原72维方向分支为0.516540±0.019008；复用原Conv3D为0.811226±0.013250。完整输入、标准化及预测核验通过。4.16一次A100基线差异及由此触发的核验失败保留，4.17只在原V100复核该基线种子，不重跑新方法。科学commit分别dfdc87b1/e5de2877；14个进程、28条事件全部核对。协议为Task4C_retain_direction_protocol_4.16和Task4C_baseline_device_protocol_4.17；结论见experiment_log的task4c-retain-direction-4-16-2026-09-14。4.15错误接入结果不删除；当前纠正任务完成，不自动追加调参。

2026-09-14纠正启动记录：用户指出4.15误删原4.14的72维几何方向频谱，立项为`Verify_Task4C_RetainDirection_4.16`。原233维FMT三种子复跑；fmt_v5在原233维后仅追加165维角度；距离+角度2.2保留原72维分支，整套模型不称完全客观。原数据/训练不变，Conv3D复用经复现的原结果。协议`docs/Task4C_retain_direction_protocol_4.16.md`；4.15成绩保留为偏离用户要求的比较，不能回答单独加入角度的效果。原始立项记录保留，最终状态以上文完成条目为准。

2026-09-14完成`Verify_Task4C_Encoders_4.15`：复用Task4-c 4.14的27,000/3,000/10,000样本，纯fmt_v5 5.1（326维）、用户明确选定的距离+夹角fmt_objective_ntod_v2 2.2（396维）及原Conv3D各三种子。固定0.5合并测试F1分别0.320565±0.010570、0.202781±0.032300、0.811226±0.013250；Conv三种子与4.14完全一致。新编码输入无kin/IVD估计或旧72维方向分支，不能把与历史161+72维FMT的差距单独归因于加夹角。科学commit785f96ce，九次训练及独立预测/调度复算PASS；13个成功进程及1个失败预检均保留，无模型文件。协议`docs/Task4C_encoders_protocol_4.15.md`，结论见experiment_log的task4c-encoders-4-15-2026-09-14。当前不自动追加调参。Task1/2/3/5旧含kin基线的命名更正继续有效。

2026-09-14用户新增授权 `Verify_Task1235_AngleFeatures_1.1`：开发fmt_v5 5.1（原FMT+15条同时间中心夹角傅里叶）与fmt_objective_ntod_v2 2.2（冻结距离版+同一夹角分支），在原10个3D条目上配对测试Task1/2/3/5。三种子、旧数据/标签/训练规则保持；Task6/7/8仍暂停。协议`docs/Task1235_angle_features_protocol_1.1.md`，科学commit5fabbe4d；94个调度进程、120分片全部完成，780行主指标与2160行逐尺度指标、原始源标签及容量检查全部复算通过，450个临时模型已清理。完整结果见experiment_log的task1235-angles-2026-09-14及paper_tables_tasks_3d。当前没有自动追加实验。旧objective_fmt_nTDO_v2 2.1及主表不改。


## 当前工作状态（2026-09-13 用户更新，优先于下文历史“当前”状态）

2026-09-14最新用户要求推进`mainExp_Task4C_PhysicalLength_4.14`：重新构建27,000训练、3,000验证、10,000测试。Channel单向目标弧长0.08/0.10/0.12，TBL4/5/6，ds均在用户范围内；RK45物理步长、实际半线弧长与最大步数独立校验。每个候选头区内部5个空间块分配3/1/1，评价中心距任意训练中心至少1h且距同头区训练中心不超过4h；共享头区/实例/源场，不宣称独立涡实例泛化。两原模型固定dropout0.15、weight decay0.0001各三种子，验证选模型后测试一次。解析检查、真实流场小预检及六次工程pilot通过，科学commit5c9c9042；协议`docs/Task4C_physical_length_protocol_4.14.md`。Ibex preflight51877188、prepare51877189、encode51877190、train51877191[0-5]、merge51877192共12个任务及六次预测均已独立核对完成。固定0.5阈值合并测试FMT0.673624±0.009154、Conv3D0.811226±0.013250；分流场Channel/TBL为FMT0.722775/0.591833、Conv0.848018/0.751596。结果见experiment_log与paper_tables_tasks_3d；未按新测试调数据距离，旧版本不改写。

2026-09-14用户指定的`mainExp_Task4C_NearbySampling_4.13`已完成：原27,000训练束保持，新增3,000邻近验证和10,000邻近测试，偏移原邻居距离5%–15%后重新积分；共享训练头区和源网格支持。原FMT/Conv3D各无正则化与dropout0.15、weight decay0.0001三种子，邻近测试F1依次为FMT0.943793±0.001428 / 0.989491±0.004233，Conv0.984681±0.001704 / 0.990186±0.005765。18个调度任务和12份预测独立核对完成，科学commit52ae26a4；见`docs/Task4C_nearby_sampling_protocol_4.13.md`及experiment_log。只衡量局部泛化，不覆盖旧4.6空间测试；当前没有自动新增实验。


2026-09-14用户最新要求的完整训练集拟合已完成：`Verify_Task4C_TrainingMemorization_4.12`原27,000训练束固定阈值0.5，原FMT/Conv3D训练F1=0.997007/0.995148，错误26/42，均连续三轮≥0.99；全量预测与样本覆盖独立复算通过。原结构无需增容。额外增容FMT已完成并保留，增容Conv及四组merge在原模型达标后取消，不宣称四组比较完成。科学commit26f52805，见`docs/Task4C_training_memorization_protocol_4.12.md`。当前不继续测试调参，4.6历史测试结果不变。


2026-09-14补齐`Ablation_Task4C_GeneralizedCrossEntropy_4.11`最终记录，协议`docs/Task4C_generalized_cross_entropy_protocol_4.11.md`。固定原4.1数据、4.3网络、4.5训练，仅比较广义交叉熵q=0/0.3/0.7；14次训练及三种子选择全部独立复算通过。对照/选定正q验证F1：FMT0.392898/0.398741（q0.3），Conv0.428428/0.432079（q0.7）。无test读取，科学commit97e3f7ea；方法结论见experiment_log，不改变当前4.13。


目标会议为 ICLR。停止继续推进现有客观化方向；暂停 Task6/7/8 几何 tokenizer、Point-NN 与混合专家调参，
包括 Task36 联合分类/重建。Task1/2/3/5 主表与 Task4-a/b 历史结果保持。
09-13 最新用户要求 Task4-c 加入论文预处理，扩展到Channel+TBL两个快照的Hairpin / Non-hairpin二分类；
当前仍以两个方法三种子合并测试F1≥0.6为目标。4.1–4.5开发与4.6正式测试均完成；4.6为FMT0.335526±0.017885、Conv3D0.304615±0.031948，目标未达成。旧门槛/源码/结果冻结。`Ablation_Task4C_RepresentationResolution_4.7`已完成（`docs/Task4C_representation_resolution_protocol_4.7.md`）：保持4.1数据，FMT6/17频率验证0.365484/0.349484、Conv24³/48³验证0.427184/0.428600，全部16次训练独立复算通过，不重开test。
2026-09-14最新完成`Ablation_Task4C_SupervisedContrastive_4.10`，见`docs/Task4C_supervised_contrastive_protocol_4.10.md`：保持原4.1数据与4.5配置/4.3分类推断函数，比较辅助监督式对比损失权重0/0.1/0.5，原测试不变。科学commit6bfc357b，14次训练与最终选择独立复算通过；零对照/正权重验证FMT0.384054/0.394229、Conv3D0.422313/0.421731，目标未达成。4.9增加中心消融全部结束：FMT原采样/更多中心验证0.415513/0.404945，Conv3D0.440833/0.407675；原采样是4.5同配置的新种子复跑，不能择优改写旧结果。4.8 SAM消融已完成，选定验证FMT0.395441、Conv3D0.429179；全部判断及原始证据位置见experiment_log。
此前 `mainExp_Task4C_LinePooling_4.3`，协议 `docs/Task4C_line_pooling_protocol_4.3.md`：保持4.1物理数据和测试，比较逐线固定FMT后学习聚合及Conv3D通道数扩大，保留4.2结构对照；无合成增强，训练/验证选择。
已完成 `mainExp_Task4C_ManifoldMixup_4.4`（`docs/Task4C_manifold_mixup_protocol_4.4.md`），只有4.3三种子选择完成且门槛未通过才允许提交；保持4.3网络/物理数据，训练时在64维隐藏表示作插值正则化，包含alpha=0对照。
此前 `mainExp_Task4C_Regularized_4.2`，协议 `docs/Task4C_regularized_protocol_4.2.md`：完全复用4.1物理样本与固定测试，原322维FMT拼接288维坐标/单位切向有方向频谱，设几何增强及余弦间隔分类对照；仅以train/validation选择，三种子验证门槛通过后才编码和评估test。
此前 `mainExp_Task4C_Multiscale_4.1` 保持冻结，协议 `docs/Task4C_multiscale_protocol_4.1.md`：用户要求多尺度、dropout/正则化及学习率调整，目标两模型F1≥0.6。
每流场13,500拟合+1,500验证+5,000测试；邻居距离与RK45参数/长度实际改变，同中心跨尺度仅在同一集合。候选头区播种后沿分区完整原始涡量追踪，至少10条有效线、最多27线。
十候选和三种子选择只用训练/验证，两模型三种子验证F1均值达标才用三个新优化种子评估封存测试；默认目标为三种子合并测试F1均值，并报告分流场。
以下3.1历史设置保持冻结，协议 `docs/Task4C_paper_bundles_protocol_3.1.md`。
每流场15,000训练开发primitive（内部13,500拟合+1,500验证）和500测试primitive，共30,000+1,000。
lambda2与正oyf筛选头区，RK45双向涡线，短线/退化线及不足10线整束剔除；每线32点、最多256线，整束质心/Rmax归一化。
FMT＋MLP与Conv3D＋MLP使用同一束几何；完整头区、GT实例和源数据节点支撑均隔离。多种子子集不是独立物理涡实例。
整束标签由完整头区GT单实例过半/零重叠得到，部分混合归属排除；不是论文人工四类标签。
2.1局部七线中心点实验与1.1历史实现冻结；BiLSTM保留但不运行。
用户已授权完成验证后删除临时验证代码、commit、push并在Ibex部署运行；验证记录保留，正式运行的数据检查保留。
公开推送注意：GitHub已核实为用户本人所有的公开`Cindy-xdZhang/FMTCode`。自动审批针对公开发布具体实验记录拒绝推送，本地提交`e1749eac`尚未push，明确公开授权问题正在等待用户回复。不得换路径公开发布这些记录；本地提交、读取及已授权个人Ibex实验继续。
3.1三种子×两方法及汇总均已完成，独立预测复算通过；科学commit `f1865b16`，结论见experiment_log的`task4c-paper-bundles-2026-09-13`。
2.1首轮三种子×两方法及汇总已全部完成，独立预测/GT复算通过；科学运行固定commit `63b44d8c`。
结论见`docs/experiment_log.md`的`task4c-binary-2026-09-13`，不得把已用test当作新确认集继续挑参数。
现有证据不能概括为“所有客观特征都无效”或“傅里叶编码必然无法回归”。方法进展和证据边界见
`docs/experiment_log.md` 的 `progress-2026-09-13`；归档路径与恢复方式见 `docs/repository_maintenance.md`。
下文的任务定义与历史版本继续有效，其中推进授权不覆盖本次暂停决定。

## 客观性基础定义（2026-09-10 用户明确固定）

在同一时刻全部点共享的时变刚体变换 `x* = Q(t)x + c(t)` 下，`Q(t)^T Q(t)=I`、`det Q(t)=1`，时间不变：

- **客观标量**：`s*(x*,t) = s(x,t)`，标量数值不变。
- **客观向量**：`v*(x*,t) = Q(t)v(x,t)`，遵守向量变换规律，不要求坐标分量不变。
- **客观二阶张量**：`T*(x*,t) = Q(t)T(x,t)Q(t)^T`。
- **同一时刻任意一对对应物质点的相对位置天然是客观向量，点间欧氏距离天然是客观标量**：`d_ij = x_j-x_i`，`d_ij* = Q(t)d_ij`，`||d_ij*|| = ||d_ij||`。同时间内积及夹角也不变。禁止以向量坐标分量改变为由否定其客观性。
- 分开判断输入、中间运算和输出，并先声明其类型。输入客观相对位置向量不得被说成“不客观输入”；后续跨时间坐标差分/傅里叶的输出性质须独立检查。输出失败不推翻输入客观性。若输入为数值不变的距离标量数组，相同固定确定性运算及固定网络输出仍不变。
- 拼接标量得到的feature vector不自动具有物理空间几何向量的变换规律。验证几何向量应比较`v*`与`Qv`，二阶张量应比较`T*`与`QTQ^T`，不能全部用逐分量不变作为标准。

用户提供的英文定义及完整数学排版抄录于 `docs/research_tasks_and_protocol.md` 第0节；本定义不修改历史算法和指标。

## 客观性表述（2026-09-07 用户强调）

- 同一时刻、同一对物质点的相对位置是客观几何向量；任意时变刚体观察者不会改变点间距离及同时间向量之间的夹角。
- 必须区分几何向量与坐标分量：在 x*=Q(t)x+c(t) 的分量约定下，d*=Q(t)d 是客观向量的正确变换规律，不能据此声称相对位置不客观或物理几何改变。客观标量要求数值不变，客观向量/张量要求遵守对应的基底变换规律；不得把两者混为一谈。
- 对后续计算另行逐步说明：客观向量的普通时间导数会包含运动基底项，不能直接从输入的客观性推断任意跨时间分量运算的结果。分析 aivd1w3_dft 时，明确连续涡量偏差公式、代码有限差分、同一时刻样本均值和最终标量之间的区别。
- 解释纠正不得静默改写冻结算法或历史指标。详细实现说明见 docs/aivd1w3_dft_explained_zh.md，方法结论与修订证据记入 docs/experiment_log.md。

## 唯一任务定义

- **Task1（2D/3D）**：training-free FMT encoder 作用于 pathline primitive，输出 feature vector；使用 KMeans 做涡区域/非涡区域二类聚类。FMT encoder 无可训练参数，KMeans 仍需拟合聚类中心。
- **Task2（2D/3D）**：无监督表示学习。核心比较固定为 `Raw pathline -> VAE -> latent -> KMeans` 与 `FMT(pathline) -> VAE -> latent -> KMeans`；同一 physical-family 内两臂必须使用为 FMT 开发并冻结的同一个 VAE（架构、latent dimension、KL 权重、学习率、训练步数相同），不得为 Raw 单独搜索更强 VAE 后替换主 baseline。主结论只回答 FMT 是否改善该同一 VAE 的输入表示；独立优化的 strongest-Raw 只能作附录压力测试。
- **Task3（2D/3D）**：有监督 IVD 涡识别。核心比较固定为不使用 FMT 与加入 FMT 的神经网络在涡/非涡二分类上的性能。
- **Task4（仅 3D）**：有监督涡类型多分类，例如 streamwise、spanwise、hairpin。Task4 与 Task3 严格分开。
- **Task4-c（当前 Channel+TBL，3.1）**：由lambda2/正oyf头部播种、RK45积分及论文规则清洗的涡线簇做区域标签二分类；FMT＋MLP与同束Conv3D＋MLP比较。完整头区和GT实例空间留出，多种子线子集只算多视图；2.1七线中心分类和1.1人工四类历史保持，不报告论文曲面检测F1。
- **Task5（2D/3D）**：Task3 的不同尺度扩展。primitive 的邻居距离、积分步长和积分步数可变，但输出保持固定线数和每线采样点数；使用同一 IVD 二分类监督。核心比较为 fixed-scale Task3 transfer、variable-scale Raw、同结构 Raw-PCA residual 与 variable-scale Raw+FMT，并在训练未见尺度组合上确认。
- **Task6（当前3D，2026-09-10重新定义）**：单流场、多尺度primitive的VAE几何重建。每个网络只在一个流场训练；大量不同起始时间、位置、积分时长与邻居距离的七线primitive，经冻结FMT变成token，再由VAE重建同一簇七条完整路径线。测试整个未见primitive，不是查询额外粒子；VAE解码器输出geometry而非token。原基线`mainExp_Task6_PrimitiveVAE_2.1`保持冻结；当前修复版为`mainExp_Task6_Reconstruction_3.1`，详见`docs/Task6_reconstruction_protocol_3.1.md`，原协议见`docs/Task6_primitive_vae_protocol_2.1.md`。
- **Task7（当前3D）**：遮挡区域补全。仅由隐藏区域外的可见 primitive tokens 恢复内部材料轨迹；必须防止重叠邻居泄漏。
- **Task8（当前3D）**：短流映射组合。使用各段已知 tokens，把第一段预测到达位置传入下一段查询，评估真实长轨迹；不是未知未来预测。

2026-09-10 当前只推进重新定义后的Task6；Task7/8暂不推进，旧Task8仍依赖旧版Task6逐粒子查询器，不能自动接入新VAE。Task6播种时刻按裁剪前原始时段的一般流场10%–80%、Cylinder 50%–80%选择，Cylinder另保留t>=7；当前[0,15]数据对应[7.5,12]。该范围只限制播种时刻，仍需保证完整积分；不能对窗口文件重新截百分比。

2026-09-09 用户选择Task6/7/8全部执行的历史定义及授权记录保留于`docs/Task678_flowmap_protocol_1.1.md`，已完成代码、实验和结果不改写。Task4暂缓推进，保留以下历史协议及结果。Task6/7/8不使用IVD p95或人工类别标签，不能把Task1–5的分类指标直接沿用为新任务目标。

当前研究范围包含 **3D Task1、Task2、Task3、Task5**，以及 **Task4-b 有监督四分类**。2026-09-06 用户重新定义 Task4-b：channel+TBL 合并训练同一个网络；涡区按 hairpin 标注内/外及 velocity–curl 更接近平行/垂直明确二分为 ordinary-streamwise、ordinary-spanwise、hairpin-head、hairpin-leg（平行含反平行，45°归平行；hairpin 内平行为 leg、垂直为 head），不加额外 head 条件。当前先执行 `Verify_Task4B_VelocityCurlMemorization_4.1` 过拟合验证，阈值使用标注全覆盖约束下的每流场 IVD>0.9a，详见 `docs/Task4B_velocity_curl_protocol_4.1.md`。历史 Task4-b 1.1 因跨 split primitive 空间重叠被否决，1.2/2.x/3.x 结果保留原协议与证据边界，不能替代新标签实验。Task4-b 与 Task3 的 whole-field IVD 二分类结果严格分开；旧实验 ID 不因本协议改名。

当前论文中所有采用 whole-field IVD 二分类的 3D 实验统一固定为 **IVD p95**：Task1/Task2 将其用于评估，Task3/Task5 将其用于监督。`Ablation_Task23IVDPercentile_1.2` 表明，在 p80、p85、p87.5、p90、p92.5、p95 的完整扫描中，p95 给出最大的 Task2 F1 增益以及 Task3 F1、Average Precision 增益；较低百分位只作为标签敏感性分析。任何后续阈值变更必须建立新实验版本，不得重写已有 p95 结果。

论文主表采用任务级统一 FMT 配方：Task1、Task2、Task3 必须分别从既有候选中选择一套跨全部10个3D数据条目不变的FMT feature blocks、FMT侧权重/缩放及后处理维度；允许用新随机种子复跑既有候选，不得新增未搜索过的超参数。旧逐flow/逐physical-family最优配置、代码和结果只作补充表与可视化，不得混入统一主表macro。Task5的FMT encoder配方同样固定；只有其研究问题预注册的邻居距离、积分步长和积分步数可按尺度tuple变化。

完整定义和评测边界见 `docs/research_tasks_and_protocol.md`。

## 实验记录

Task4-b 当前阈值以2026-09-06用户最新指定为准：**Channel IVD>5.785，TBL IVD>0.08**，取代上述4.1及后续4.2/4.3基于hairpin最小IVD的选择。当前 `Other_Task4B_ProxyGTThreshold_4.4` 仅生成ground truth和三维图，不训练；四类velocity-curl规则保持不变，不强制补回阈值以下hairpin标注。旧版数据和训练结果保留，后续训练需新版本。该更新不修改其他任务的p95。

- 方法级结论只写入 `docs/experiment_log.md`。
- Task4-b 的 `mainExp_Task4B_PatchSegmentation_5.1` 已冻结：channel/TBL各900训练+100测试小区域、实例66/7与52/6、共享FMT四分类网络；跨集合patch隔离、流线不出patch，固定200epoch后test一次。沿用4.4固定IVD阈值；详见 `docs/Task4B_patch_segmentation_protocol_5.1.md`。
- 当前用户授权的 `Other_Task4B_GeometrySearch_5.2` 尝试网络结构、学习率及几何点差估计的梯度/curl特征。只在5.1原训练区域内部划分有空间间隔的拟合/验证集来选候选和epoch，再恢复全部900+900训练patch重训；最后使用原test一次，明确其为已用过的benchmark，非新的confirmation。仍为共享四分类，不把真实curl、IVD、实例ID或标签作为输入。详见 `docs/Task4B_geometry_search_protocol_5.2.md`；原5.1结果不改写。
- 每个向 Ibex 提交的进程必须登记到 `docs/ibex_run_registry.md`；失败、取消、超时和无效实验也不得删除。
- 提交后立即登记 job ID、实验版本、任务、提交时间、config、git commit 和预期设备；开始后补实际开始时间、节点和 GPU；结束后补主要结果及其支持/反对的结论。
- 不得用 test/confirmation 数据选模型、阈值、特征、epoch 或超参数。任何修订必须新旧并列记录。

## Cylinder 默认时间范围（2026-09-06 用户更新）

- 所有 cylinder/half-cylinder 数据（含Re160、Re640、Re6400）后续可视化及新实验默认忽略原始模拟前50%的初始阶段，同时要求初始物理时间t>=7。当前三个3D数据原始模拟区间均为[0,15]，故有效起始下限为t=7.5。
- 按原始模拟时段计算50%，不能对已裁剪的文件再次截半；Re640现有文件[7.5,15]已经符合这一范围。
- 已有确认缓存用于新图时，按时间选择首个合规时间片，不依据图像、标签或指标择优：当前Task1 Re160/Re640/Re6400分别为t=10.5/9.5/10.5。
- 已冻结实验、训练/测试划分与旧结果不追溯修改；后续改变实验采样范围必须建新版本并记录。机器可读默认值见config/cylinder_time_policy.json。

## 实验文件保留规则

- 训练模型 checkpoint 不属于长期实验结果。默认不得在本地保存、下载或归档 `.pt`、`.pth`、`.ckpt` 及包含这些文件的压缩包。
- 如果 Ibex 上的后续依赖作业必须读取 checkpoint，可以只在该实验依赖链运行期间临时保存；最终评估的逐次 CSV/JSON 和总实验表写完后必须删除 checkpoint。
- 可复现依据是代码 commit、完整 config、随机种子、数据时间片/划分、设备记录、逐次指标和汇总表，不是训练好的模型文件。只有用户明确要求保留某个模型时才可例外。

2026-09-10 用户要求继续修复Task6高重建误差，当前修复版本为`mainExp_Task6_Reconstruction_3.1`：新signed_fmt10保留复系数方向/相位和七线身份，192维VAE重建完整几何，每流场24万训练primitive。旧161维fmt_all和2.1历史结果保持；两者不可混称。新配置仅由train/validation诊断选择，目标每流场test RMSE/r<1，保持原误差单位与完整轨迹；详见`docs/Task6_reconstruction_protocol_3.1.md`。

2026-09-10 用户要求小训练集与dropout等泛化验证，新增 `Verify_Task6_ScarceGeneralization_4.1`：每流场256/1024/4096个primitive，1024为主比较；初始化也只读取小训练集。8个FMT与4个Raw候选只用validation选择，最后以3个新子集/优化种子评估。保留无正则化Raw、相同正则化Raw及validation选择Raw对照；3.1代码与结果冻结。协议见 `docs/Task6_scarce_generalization_protocol_4.1.md`。

## Task6 直接神经编码核验（2026-09-11，5.1）

当前用户授权版本为`Verify_Task6_DirectNeural_5.1`，协议`docs/Task6_direct_neural_protocol_5.1.md`。冻结3.1/4.1数据与历史结果，删除神经推理中的固定解析逆变换/PCA矩阵。新网络全部输入相关映射由训练学习，192维潜变量；Raw/FMT均651维输入且网络参数完全相同，PCA仅作独立基线。先真实32样本拟合核验，再九流场validation选配置，最后三档样本量、三种子测试。测试集是已用benchmark；不能将完整16频FMT本身描述成压缩token，也不能预设必定胜出。结果仅记入experiment_log。


<!-- ftlep35-final-state-2.1-begin -->
2026-09-15完成`Other_FTLEP35Fusion_2.1`：从分类p35迁移二维五线的逐特征均值/最大值，保留真实Fourier变换。88次validation拟合后统一选择pyramid_p35；低网格局部残差卷积、全局均值/最大值及像素重排，三种子×四流场×4/8××六方法共144次最终拟合。四场平均PSNR4×/8×为P35 39.630193/33.157295，ESPCN36.715430/32.583144，U-Net31.215084/28.223757；相对同结构none为+0.708843/−0.039553dB，相对Raw+0.156547/+0.135416dB。Doublegyre8×仍低于ESPCN，不能宣称8×独立编码收益已证明或全部流场均获益。开发科学52ec9d8b、最终/数值审计f0993a58；264+864份预测独立复算、57进程/90事件核对PASS（44完成、1原审计失败、12未启动取消），无模型文件。原数值审计失败与修订保留。完整积分窗/源帧跨集合隔离，沿用已用benchmark；没有按test再选方案。九幅报告图通过字体、绘图区和重叠检查。结论在experiment_log的ftle-p35-fusion-final-2-1-2026-09-15，结果在outputs/Other_FTLEP35Fusion_2.1/final；旧实验和其他任务不改，本轮比较完成，不自动继续调参。
<!-- ftlep35-final-state-2.1-end -->

2026-09-16新增用户授权 `Ablation_Task4C_GeometricBaselines_1.1`：在冻结BottomDensity1.2的193,000/3,000/10,000线簇上追加三种基线，各种子96611–96613。baseline0切线/曲率后mean+max再MLP，76,782参数；baseline1整束点云固定Point-NN特征后MLP，76,704参数；baseline2逐线BiLSTM后线间mean+max与MLP，76,786参数（递归55,728）。与当前FMT76,738参数匹配，数据、标签、长度、空间块和训练器不改。CPU解析曲率、点云顺序/批次独立性、padding、BiLSTM序列性和梯度核验通过；正式提交前无新测试F1。协议docs/Task4C_geometric_baselines_protocol_1.1.md。此授权不恢复Task6等已暂停方向。

<!-- ftlep35-scales-2.2-submitted -->
2026-09-16用户要求补测各方案2×及bilinear/bicubic的2/4/8×，新增Other_FTLEP35Fusion_2.2。固定2.1六方案/四流场/三种子，共72次新增训练，4/8×复用已核验144次结果。科学commit64378a05，Ibex51929949–51929958共24进程已提交；相同GTX1080Ti，原数据/时间隔离/训练/选轮不变，无新方法搜索、无checkpoint。协议docs/FTLE_p35_scales_protocol_2.2.md，结果待核验。
<!-- ftlep35-scales-2.2-submitted-end -->

<!-- ftlep35-scales-final-2.2-begin -->
2026-09-16完成`Other_FTLEP35Fusion_2.2`：固定2.1六方案补2×四场三种子，共72训练/432预测，4/8×复用原144训练；新增bilinear/bicubic三倍率144预测。科学commit64378a05，相同GTX1080Ti、原五线/时间块/共同mask/训练规则，无重新选方案。p35测试PSNR2/4/8×为47.527190/39.630193/33.157295；2×相对同网络仅低FTLE+1.119368、Raw+0.295512dB；DoubleGyre2×全选step0双三次，8×仍无优于仅低FTLE的平均PSNR增益（物理RMSE排名不同，不混用指标）。120行总表、24进程48事件及全部预测独立核验PASS，无模型文件。完整表outputs/Other_FTLEP35Fusion_2.2/report.md和paper_tables_ftle_2d；结论见experiment_log同名条目。当前请求完成，不自动追加调参，其他任务独立继续。
<!-- ftlep35-scales-final-2.2-end -->


2026-09-16用户要求核对Task4-c播种至线簇及训练/测试选邻居规则，并测试FPS。新增`Ablation_Task4C_NeighborSelection_1.1`：冻结BottomDensity1.2的193000/3000/10000束；每条有效线作anchor，比较最近6条（复用三种子原p35）、FPS6、最近3条+FPS3。两新方法各三种子，141维/76738参数及原训练不变。物理生成保留全部10–27条有效线；最近邻只用于FMT内部。CPU预检及训练循环AST核对通过，正式GPU核验/训练待提交；协议`docs/Task4C_neighbor_selection_protocol_1.1.md`。其他基线继续，不自动重采样或修改标签/积分。


2026-09-16邻居选择1.1已提交Ibex：科学commit686ef144，51936278–51936283共12进程；V100预检及206000束全部种子布局/18文件哈希核验通过，pilot等待GPU，编码与六次训练依赖等待。正式FPS指标尚无，禁止预设改善。


2026-09-16用户要求追加PointNet++并部署Ibex。新增`Ablation_Task4C_PointNetPlusPlus_1.1`，冻结BottomDensity1.2全部193000/3000/10000束与训练设置；官方SSG拓扑512/128/global、半径0.2/0.4、32/64邻点，缩小通道，总76723参数（比FMT少15）。固定几何FPS/球形分组索引提前缓存，网络全部层学习，三个种子96611–96613。CPU独立FPS/ball-query、排列/补零、梯度及训练循环AST核验通过；正式V100预检/真实32束pilot后编码与训练。协议`docs/Task4C_pointnetplusplus_protocol_1.1.md`，其余实验保持。


2026-09-16 PointNet++1.1已部署并提交：科学commitf3d98dd3；51946834预检、51946835数据复用、51946836真实pilot、51946837[0–1]分组缓存、51946838[0–2]三种子训练、51946839独立汇总，共9进程。当前GPU预检Priority等待，后续Dependency等待；CPU验证通过，尚无正式F1。


2026-09-16 PointNet++1.1的V100预检51946834和全部18源文件核验51946835已通过；实际GPU为gpu214-02/Tesla V100-SXM2-32GB。真实数据pilot51946836等待Priority，编码及三种子训练依赖等待，尚无F1。


2026-09-16完成Task4-c三个扩展：几何基线d20c2458、PointNet b4c8a80c、FMT邻居686ef144，18次训练与34进程/68事件核验通过。三种子testF1：切线/曲率+MLP0.580051±0.001726，Point-NN式编码+MLP0.303573±0.020438，BiLSTM0.919499±0.010500，PointNet0.889407±0.006155，FMT FPS6 0.893015±0.004408、最近3+FPS3 0.886779±0.006196；原FMT0.888404±0.011497保留。仅局部共享实例测试；不按test继续调参。主表/log已更新，outputs各final_verified_evidence.json保存证据。PointNet++51946838[0–2]仍运行，17:15均20轮/每轮约3.1分钟，三卡已分配。


2026-09-16用户澄清BiLSTM+MLP重跑预算是缩小到约56,786总参数，原5.6M为笔误。新增`Ablation_Task4C_BiLSTMCapacity_1.1`：逐线单层双向hidden71、mean/max后MLP284→47→2，递归43,168+分类头13,585=56,753参数。冻结193000/3000/10000数据与原训练函数，三种子96611–96613；原76,786参数F1=0.919499±0.010500保留。CPU参数/补零/顺序/梯度及仅宽度变化的forward AST核验通过；准备提交Ibex。协议`docs/Task4C_bilstm_capacity_protocol_1.1.md`，其他实验不改。


2026-09-16小BiLSTM+MLP容量1.1已部署Ibex：科学commit2b92207b；51959150预检、51959151数据核验、51959152真实32束pilot、51959153[0–2]三种子训练、51959154独立汇总，共7进程。总56,753参数（43,168+13,585）；CPU预检通过，初次检查GPU预检Priority等待、训练依赖等待，尚无新F1。

<!-- task4c-saliency-completed-1.1 -->
2026-09-16完成用户要求的`Other_Task4C_Saliency_1.1`：固定BottomDensity1.2 FMT p35/n0_k06 seed96611，原训练函数回放至预先选定epoch56；所有56轮及206,000预测数组与原运行完全一致。Channel/TBL各200个预测Hairpin束，共7,244线/231,808点，逐点梯度、SmoothGrad及50,708次局部拉直干预已生成。三维查看器支持画廊/单束、贡献着色、片段叠加和标签不一致筛选，真实浏览器核验通过。科学df325fbf；四Slurm进程/八事件（含一次训练前字段错误）完整保留，无权重。协议docs/Task4C_saliency_protocol_1.1.md；独立审计outputs/Other_Task4C_Saliency_1.1/independent_audit.json；解释边界见experiment_log。当前请求完成，不启动其他消融。
<!-- task4c-saliency-completed-1.1-end -->


<!-- task4c-bilstm-small-final-20260916 -->
2026-09-16完成`Ablation_Task4C_BiLSTMCapacity_1.1`：用户修正预算约56,786后，仅把原BiLSTM单向hidden81→71、MLP hidden64→47；总参数56,753（递归43,168+MLP13,585）。冻结193000/3000/10000数据、训练函数与三种子，测试F1为0.907316±0.008391，平均训练69.755分钟；原76,786参数0.919499±0.010500保持。科学commit2b92207b，七进程/14事件及三份测试预测独立核验通过，结果/结论见experiment_log同标记。19:44沙特时间PointNet++三种子仍运行在68/68/69轮，尚无测试结果，不自动追加调参。

<!-- fmt-feature-contrast-complete-1.1 -->
2026-09-16完成用户要求`Other_FMT_FeatureContrast_1.1`：2160个现有Cylinder3D直接p35/n0_k06 primitive，t7.7–11.4，完整141维标准化后KMeans K2；提供逐维中心差异、完整向量、分布、热图、样本距离分解及真实轨线联动。统一HTML入口`outputs/Other_FMT_AnalysisWorkbench_1.1/index.html`另页复用400束Hairpin显著性。原编码/标准化精确复现、独立141维公式与全部浏览器数组/几何核验通过，页面交互通过。无IVD标签/质量指标、新训练或Ibex作业；描述性结论只在experiment_log。协议docs/FMT_feature_contrast_protocol_1.1.md；启动start_fmt_analysis.ps1。当前请求完成。
<!-- fmt-feature-contrast-complete-1.1-end -->


<!-- task4c-fps-augmentation-search-start-20260916 -->
2026-09-16用户新增授权`Ablation_Task4C_FPSAugmentSearch_1.1`：在冻结BottomDensity1.2的193000/3000/10000线簇上，FPS6下比较旧Task4-c验证前五p13/p10/p14/p17/p21，额外p35对照，以及五种保留真实傅里叶的新网络。32/48点×均匀/曲率采样×无增强/切向抖动/小角度旋转/全SO(3)旋转及两种组合，共264候选；前12各三种子完整验证，锁定一方案后与原FPS控制各三种子测试，共306拟合、312进程，最多24个V100并行。只对现有折线重采样；标签、中心、积分、划分不变。目标约0.93，须高于0.933198才超过冻结Conv24。CPU精确特征/几何/梯度核验通过，尚未提交。协议docs/Task4C_fps_augmentation_search_protocol_1.1.md。此授权只覆盖本轮预定搜索，旧结果不改。


<!-- task4c-fps-augmentation-search-submitted-20260916 -->
2026-09-16 FPS增强搜索1.1已部署Ibex，科学commit9d05abc5；51974951预检、51974952数据核验、51974953真实pilot、51974954[0–263]初筛、51974955初选、51974956[0–35]三种子细化、51974957选定、51974958[0–5]最终、51974959汇总，共312进程，已登记。CPU检查通过；初次检查GPU预检等待资源，后续依赖等待。最多24个V100并行，无新F1。

2026-09-16 22:15沙特时间补充：V100预检51974951（gpu214-14）与18个源文件哈希核验51974952均通过，数据计数193000/3000/10000不变。51974953真实pilot等待GPU；264初筛仍依赖等待。


<!-- task4c-fpsaug-progress-20260917-0128 -->
2026-09-17 01:28沙特时间FPS增强1.1更新：真实pilot通过；初筛134完成/15运行/115待启动的快照后，已独立核对135份验证预测。完成最高c046验证F1=0.909492；运行中c144宽残差p35在epoch49验证0.921053，非测试结果。暂无失败，尚未进入前三种子细化和最终测试；312进程依赖链继续。记录见experiment_log/ibex_registry同标记，冻结科学commit9d05abc5。


<!-- task4c-fpsaug-top20-extension-20260917 -->
2026-09-17用户要求把FPS增强搜索复核扩大到前20名，新增Ablation_Task4C_FPSAugmentSearch_1.2。原264初筛和前12名36次复核保持；只给固定初筛第13–20名追加24次同三种子全预算复核，共60次。数据/模型/增强/优化规则不改；新选择必须同时等待两组复核完成。旧待运行选择51974957已hold，旧六次final51974958和merge51974959在新依赖链成功提交后取消并留记录。按已说明默认解释：前20名验证复核后选定一个方案与原FPS对照做六次最终测试，不扩大测试集选参范围。总并行不超过24V100。协议docs/Task4C_fps_augmentation_search_protocol_1.2.md；旧1.1输出与身份保持，1.2仅作追加和最终选择替代。


<!-- task4c-fpsaug-top20-deployed-20260917 -->
2026-09-17前20名扩展1.2已部署：科学commit21cf1fb75b556741ae7a6da8bd0f08104c1a0db5，Ibex51984022准备、51984023[0–23]新增24次复核、51984024选择、51984025[0–5]最终训练、51984026汇总，共33进程。准备已成功，原264预测/排名、源代码及18份数据哈希核验通过；原51974956的36次复核继续，新增24次等待GPU。新选择同时依赖两组复核。旧51974957、51974958[0–5]、51974959共8个未启动进程于07:44:31Saudi确认取消，所有记录保留；没有取消已完成或正在进行的训练。旧数组未来上限12、新数组上限10（部署时旧有14在跑），总<=24V100。新版本尚无训练结果，源实验验证成绩不变。


<!-- task4c-fpsaug-final-20260917 -->
2026-09-17 FPS增强搜索1.1/前20名扩展1.2全部完成：264初筛+60复核+6最终训练，选择c156=p35＋加宽残差网络/48均匀点/无增强，992,386参数。复核验证0.933609±0.008155；最终三新种子测试0.920390±0.003032，原FPS配对控制0.882974±0.010458。未达到0.93，也未超过冻结Conv24=0.933198；与原BiLSTM=0.919499接近，不能宣称可靠胜出。336预测、330训练记录、337进程674事件独立核验PASS，8个旧待运行取消记录保留，无模型文件；11:47Saudi全部结束。科学commits9d05abc5/21cf1fb7；1.2实际Git/Ibex配置哈希14a849f5，旧e9086507是Windows CRLF字节哈希，内容完全相同，纠正说明见日志。完整20验证排名和最终测试见paper_tables_tasks_3d；方法结论见experiment_log。当前请求完成，不自动追加调参或测试。

<!-- fmt-analysis-workbench-1.2-2026-09-17 -->
2026-09-17完成`Other_FMT_AnalysisWorkbench_1.2`：特征差异页新增141维逐特征计算步骤图；Hairpin页默认显示精确最大正支持窗口、原/局部拉直概率与有符号热图，紫色人工替代线默认关闭。冻结400束50,708次原干预及特征/KMeans数组，逐项独立复核和五项回归检查通过；浏览器已验证主要交互。代码及核查见`docs/FMT_analysis_workbench_protocol_1.2.md`；旧工作台入口自动转至1.2，原HTML归档，不训练或换模型。显示问题与证据边界见experiment_log同名条目。
<!-- fmt-analysis-workbench-1.2-2026-09-17-end -->

<!-- fmt-color-explanation-1.3-2026-09-17 -->
2026-09-17分析工作台更新`Other_FMT_AnalysisWorkbench_1.3`：五种着色改为直接说明用途的名称，并在图上方显示计算、颜色与限制；非拉直模式隐藏片段控制/拉直前后概率/窗口表。原梯度内置色带的低红高黄改为明确低黄高红；数值不变。827份继承文件哈希一致，五模式浏览器交互核验通过；旧1.2入口自动跳转，原HTML归档。协议docs/FMT_analysis_workbench_protocol_1.3.md，当前不训练或更换c156模型。
<!-- fmt-color-explanation-1.3-2026-09-17-end -->

<!-- fmt-whole-curve-results-1.4-2026-09-17 -->
2026-09-17完成用户要求的 `Other_FMT_AnalysisWorkbench_1.4`：新增全部线段按拉直影响着色，默认每32点曲线取已实测0–8/8–16/16–24/24–31四段，完整覆盖且不平均重叠分数；保留单段等旧模式。新增第三页嵌入已有4.14分类查看器（FMT414/Conv32/Conv36、seed96611、37,000束），来源明确，与解释页p35/n0_k06及最新c156分开。400束7,244线28,976片段覆盖/数值核验、827继承文件和294结果依赖哈希及浏览器交互通过；无新训练。协议 `docs/FMT_analysis_workbench_protocol_1.4.md`，当前入口1.4，旧1.3自动跳转。
<!-- fmt-whole-curve-results-1.4-2026-09-17-end -->

<!-- task4c-gt-head-coverage-start-20260917 -->
2026-09-17用户要求分类查看器覆盖每个GT Hairpin头部；已确认使用固定c156，新增束按人工GT头部归属标正类，旧标签不变。诊断旧4.14测试头部覆盖Channel55/74、TBL46/58，部分实例被原整片区域规则排除。新增 `mainExp_Task4C_GTHeadCoverage_1.1`：在BottomDensity1.2后每GT追加10测试＋30邻近训练，目标196960/3000/11320；原前缀、验证、标签和积分长度保留。132实例528束小积分及GT/夹角/长度/距离核验通过，不放宽1h距离。固定c156 992386参数，单种子96721，仅用于覆盖可视化，不搜索。科学commit8fd6be94已推送；Ibex52000820[0–1]准备、52000821审计、52000822 V100预检、52000823训练、52000824导出共6进程已提交；首次查询两数据任务运行，其余依赖等待。协议docs/Task4C_GT_head_coverage_protocol_1.1.md，暂无新F1，新页面待预测完成。
<!-- task4c-gt-head-coverage-start-20260917-end -->

<!-- task4c-gt-head-coverage-r2-20260917 -->
2026-09-17 GT头部覆盖1.1首链已保留失败：52000820_0/1完成全部740/580组后，在旧数组末批拼接时失败；未改源数据，未训练。四个待运行依赖52000821–52000824取消。修复截断末批并通过独立513/708/804行拼接检查，科学commitc3952a7b；独立r2目录 `/ibex/user/zhanx0o/FMT_Task4C_GTHeadCoverage_20260917_r2`，新链52000893[0–1]、52000894、52000895、52000896、52000897共6进程已提交。采样、c156和配额不改；全量审计及训练尚待完成。
<!-- task4c-gt-head-coverage-r2-20260917-end -->

<!-- task4c-gt-head-training-retry-20260917 -->
2026-09-17 GT头部覆盖1.1已完成正式数据及V100工程检查：196960/3000/11320，132个GT各10新增测试＋30新增训练，原前缀/验证文件相同。科学commitc3952a7b；数据证据outputs/Verify_Task4C_GTHeadCoverage_1.1/formal_data_audit.json。首次正式训练52000896在第91轮无错误栈退出1，CPU20:00.904、墙钟20:26，缺少ENDED事件；根因未确证，不称训练完成。失败历史保留failed_runs/52000896，原导出52000897取消。同科学代码/数据/c156/seed96721重试52001957，后接导出52001958；V100、3h、16GiB，排除原gpu213-18，增加进程限制/故障日志，未改学习设置。工作台1.5代码和逐GT表已准备，旧真实4.14诊断交互通过，但新c156结果尚无，原1.4页面未替换。一次性本地收尾进程跟踪重试，成功后独立核验并构建，状态outputs/mainExp_Task4C_GTHeadCoverage_1.1/local_delivery_status.json；新页面交互仍需正式结果后核对。协议docs/Task4C_GT_head_coverage_protocol_1.1.md及docs/FMT_analysis_workbench_protocol_1.5.md。
<!-- task4c-gt-head-training-retry-20260917-end -->
