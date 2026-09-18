# Task4-c Hairpin 分类：最新交接

<!-- task4c-fixed-dataset-1p1-r3-submit-20260918 -->
2026-09-18 `mainExp_Task4C_FixedDataset_1.1` r3 部署并提交（科学commit `d9dc32bf`）。用户晚间第二条裁定：未见实例的测试点直接在该实例 GT 包围盒内均匀随机取，**不加任何筛选**（无 λ₂/oyf/夹角），标签按中心点 GT 归属（`sample_kind=2`）；排除区就是 GT 包围盒，不外扩。与此前 r3 改动（1h、100 次放松、覆盖下限 2/目标 10、配对中心夹角检查、GT 头部补样）一起进入 `docs/Task4C_fixed_dataset_rules_v1.md`（变更记录第 4 行）。核验新增：包围盒行只在测试、中心在盒内、标签＝GT 归属；未见实例覆盖按盒内测试点计；`preparation.json` 增 `gt_boxes`。本地 10 项构建测试通过；私有部署 `/ibex/user/zhanx0o/FMT_Task4C_FixedDataset_1p1_20260918_r3`（HEAD d9dc32bf，远端 10 项测试通过）。13:37 UTC 提交输入核验 52049385、pilot 52049386[0–1]、正式重建 52049387[0–1]、数据核验 52049388。训练版本与 baseline 版本仍待冻结哈希。
<!-- task4c-fixed-dataset-1p1-r3-submit-20260918-end -->

<!-- task4c-fixed-dataset-1p1-r3-rules-20260918 -->
2026-09-18 `mainExp_Task4C_FixedDataset_1.1` r2 pilot 52048031_0/1 FAILED（channel 43s、TBL 66s）：测试覆盖满足但训练覆盖不满足，channel 11 个、TBL 4 个覆盖实例只有 1 个头部中心；根因是`pair_sample`只要求配对中心在 GT 包围盒内，未要求覆盖计数所用的头部夹角条件，配对行落在 GT 内但不在头部区域。用户随即叫停并细化规则：评价点距训练 3h→**1h**、放松阈值 1000→**100** 次、覆盖下限 2（目标 10），并要求把构建规则单独写成规范文档。已取消 52048032[0-1]/52048033，删除 r2 pilot 输出（远端目录其余文件保留）。r3 改动：配对中心加头部夹角检查；每覆盖实例在头区填充前补非放松 GT 头部训练行到目标 10（`allow_short`，下限 2 由覆盖检查保证）；测试 GT 头部行同样目标 10/下限 2；配置 `execution_revision=r3_rules_2026-09-18_evening`。规范文本 `docs/Task4C_fixed_dataset_rules_v1.md`（新），实现记录 `docs/Task4C_fixed_dataset_protocol_1.1.md` 重写，总协议 2b 与交接 3b 改为指向规范文本。本地 8+3+2 项测试通过。**r3 尚未部署**，待用户确认规范文本；训练/baseline 版本仍待冻结哈希。
<!-- task4c-fixed-dataset-1p1-r3-rules-20260918-end -->

<!-- task4c-fixed-dataset-1p1-r2-20260918 -->
2026-09-18 `mainExp_Task4C_FixedDataset_1.1` r1（ae65e920）：输入核验52047083、pilot 52047084_0/1通过（channel 59/15、TBL 46/12实例分组，覆盖满足，放松仅channel测试2行），但正式重建52047085_0 channel在测试集GT头部实例10处FAILED：训练集先生成、9万行填满头区后，头区内没有任何点离所有训练中心≥3h（拒绝31,902次`too_close_to_train`、7,271次`too_far_from_same_instance_train`），放松只解除λ₂/oyf不解除距离；52047085_1与核验52047086取消。r2（科学commit `e4500686`）把生成顺序改为test→validation→train：验证距测试中心≥0.5h_test，训练距所有评价中心≥3h_eval；每个覆盖实例的正类测试点在同实例内3h–6h环带配对生成一个训练中心（保证≤6h规则），配不到的测试正类删除并复核覆盖；核验新增配对距离与调整后行数检查；重复中心断言改为(中心,尺度)唯一。本地7项测试通过；私有部署`/ibex/user/zhanx0o/FMT_Task4C_FixedDataset_1p1_20260918_r2`（远端测试通过）。13:00 UTC提交输入核验52048030、pilot 52048031[0–1]、重建52048032[0–1]、数据核验52048033。训练版本`mainExp_Task4C_FixedDatasetFMT_1.1`（d7a71ecf，四臂×3种子）与`mainExp_Task4C_FixedDatasetBaselines_1.1`（0240a366：p35轮换对照、Conv3D 16³/24³、BiLSTM、PointNet、PointNet++，各3种子，复用冻结驱动只重绑identity与数据根）已提交代码，待数据核验哈希填入后部署。
<!-- task4c-fixed-dataset-1p1-r2-20260918-end -->

<!-- task4c-fixed-dataset-1p1-submitted-20260918 -->
2026-09-18 立项并提交`mainExp_Task4C_FixedDataset_1.1`（科学commit `ae65e920`）：按总协议2b节从头生成Task4-c固定线簇数据集v1。每流场train 90,000/validation 1,500/test 6,000（合计180,000/3,000/12,000）；hairpin实例按流场固定种子分80%覆盖组/20%未见组（重叠包围盒同组），训练每个覆盖实例≥10个GT头部中心，测试每个实例≥10个（覆盖实例为距任意训练中心≥3h、距同实例训练中心≤6h的偏移点，未见实例为新点）；验证/测试距训练≥3h、测试距验证≥0.5h；训练/验证不进入未见实例GT包围盒外扩一格的排除区；负类头区跟随最近实例分组。中心只筛λ₂/oyf/同头区，26邻居仅域内，≥17线保留，1000次失败后放松并按中心GT归属定标签；逐线保存播种点与head_candidate/oyf_positive。r2替换式重建核验失败的“中心唯一”断言查明为断言写错（模板允许同一中心配三种积分长度，channel训练模板本身有3,678组重复中心），新版本以(中心,尺度)为唯一签名。本地6项新测试通过；私有部署`/ibex/user/zhanx0o/FMT_Task4C_FixedDataset_1p1_20260918`（HEAD与3份源码哈希一致，远端14项测试通过）。12:33 UTC提交输入核验52047083、两流场pilot 52047084[0–1]（每流场900/150/300行、每实例≥2覆盖）、正式重建52047085[0–1]、数据核验52047086，共6进程；尚无数据。协议`docs/Task4C_fixed_dataset_protocol_1.1.md`。训练与baseline比较待数据冻结后另立版本。
<!-- task4c-fixed-dataset-1p1-submitted-20260918-end -->

<!-- task4c-fps16-1p2-submitted-20260918 -->
2026-09-18 Claude接替codex，立项`Ablation_Task4C_FPS16_1.2`。查询确认1.1的r2 pilot 52044303_0/1 FAILED/1:0不是导入问题：冻结的`center_and_neighbors`要求27个模板点都oyf>0且属同一连通头区，codex诊断的失败层头区只有7–21个单元、1.0h模板最多12–15个合格点，channel pilot 4,096次试探中46,963/47,550次拒绝为`fewer_than_17_seed_points`，≥17播种点在旧规则下结构性不可能；r2后续52044304–52044307已取消，无F1。

用户2026-09-18决定：样本的根本是采样点，邻居播种点只要求在计算域内，只有中心保留λ₂/oyf/同头区筛选；样本须中心线有效且≥16邻居通过清理（≥17线），否则重采；一层连续1000次失败后放松中心筛选、标签按中心点GT归属决定并标记relaxed；只替换不满足条件的行，模板为完整GTHeadCoverage 196,960/3,000/11,320（需替换72,043/1,248/4,518行，另加因训练中心被替换而失去4h内同头区训练中心的评价行）；每条线保存播种点坐标及`head_candidate`（step1条件：λ₂<阈值∧oyf>0）与`oyf_positive`标签，播种点与线束成为显式键值对，此后数据集长期固定。四臂p35_h0_fps16/c156_fps16/p35_h0_six_control/c156_six_control、141维、76,738/992,386参数、训练规则与seed96721与1.1完全相同。替换行中心固定第0槽并保存全部域内存活邻居（16–26），索引记录`neighbor_filter`（0旧行/1新行/2放松）与`relaxed`；保留旧行与新行的邻居过滤规则不同，为用户接受的已知不一致。

r1（科学commit `83ec5139`，尚无逐线属性）：私有部署`/ibex/user/zhanx0o/FMT_Task4C_FPS16_1p2_20260918`，源核验52045843、GPU预检52045844（四臂32束F1=1.0）、两流场pilot 52045845_0/1均COMPLETED/0:0；pilot新行27线占多数（channel训练137/148）、最少17线、平均1.6–6.4次试探，但channel验证集1个槽第1001次才成功，拒绝几乎全是与已被替换的旧训练中心保持1h间距。用户随即追加逐线属性要求，重建52045846启动1分42秒后取消（未写数据），52045847–52045849未启动；证据下载至`outputs/Ablation_Task4C_FPS16_1.2/r1_cancelled/`。

r2（科学commit `e67b03f9`）：新增`line_seed_attributes.npz`（seed_points、stencil_slot、lambda2、oyf、inside、head_candidate、oyf_positive），距离规则改为只针对最终数据集（旧中心仅禁止重合），数据核验从原始场重新插值复核。本地18项测试（1.2新增10项＋1.1五项＋原中心三项）通过；私有部署`/ibex/user/zhanx0o/FMT_Task4C_FPS16_1p2_20260918_r2`（远端HEAD与4份源码哈希一致，远端13项测试通过）。11:59 UTC提交源核验52046103、GPU预检52046104、pilot 52046105[0–1]、重建52046106[0–1]、数据核验52046107、四臂训练52046108[0–3]%4、结果复算52046109，共12进程，尚无F1。存储：用户授权删除`/ibex/user/zhanx0o/project_pvit_codefiles`（732 GB）与`/ibex/user/zhanx0o/outputs`（131 GB，旧FTLE超分数据集），11:40–11:42 UTC完成，`/ibex/user`可用3.8 GB→871 GB；其他目录未动。协议`docs/Task4C_FPS16_protocol_1.2.md`，状态`outputs/Ablation_Task4C_FPS16_1.2/deployment_status.json`。未公开推送。
<!-- task4c-fps16-1p2-submitted-20260918-end -->

<!-- task4c-fps16-r2-20260918 -->
2026-09-18查询发现FPS16首链源核验52036137和GPU检查52036138已成功，但两个补样本pilot52036139_0/1均FAILED/1:0，分别55/30秒：GT新增样本分支错误地从experiments模块导入vector_at。首次正式重建和训练从未开始，无F1；取消后续52036140–52036143八个未启动进程，失败证据保留在outputs/Ablation_Task4C_FPS16_1.1/failed_attempt_bc41b003。此前“排队”是提交时状态，本条取代其当前状态。

修复科学commit8888eecb：从FMT_Utils中的原实现导入并提前至模块加载检查；增加真实插值/夹角/播种分支测试，五项本地测试通过；依赖源码加入哈希。数据规则、标签、固定原中心、FPS16、四模型、seed96721及训练预算均不变。已部署独立目录/ibex/user/zhanx0o/FMT_Task4C_FPS16_20260918_r2；新链源核验52044301、GPU52044302、pilot52044303[0–1]、重建52044304[0–1]、数据核验52044305、训练52044306[0–3]%4、结果复算52044307，共12进程。r2源核验52044301已COMPLETED/0:0/12秒，GPU和实际补样本pilot排队；尚无新F1，不把首链小拟合分数当正式性能。实际证据r2/submission.json，状态deployment_status.json。
<!-- task4c-fps16-r2-20260918-end -->


<!-- task4c-fps16-submitted-20260918 -->
2026-09-18用户明确“授权上传，立刻跑实验”，此前上传审批阻碍已解除。`Ablation_Task4C_FPS16_1.1`科学commit `bc41b0034955f6e1db82f68604e9825278ac0ae2`已私有部署到`/ibex/user/zhanx0o/FMT_Task4C_FPS16_20260918`，远端包SHA256和科学源码/配置哈希一致，完整依赖导入通过。

已提交：源数据核验52036137、V100四臂预检52036138、Channel/TBL补样本pilot52036139[0–1]、正式重建52036140[0–1]、全量数据核验52036141、四次固定中心训练52036142[0–3]%4、预测复算52036143，共12进程。源核验52036137已COMPLETED/0:0/29秒，18份源数据和六份共同子集索引及原中心核验通过；GPU/pilot排队，后续作业按成功依赖等待。尚未称GPU/pilot通过或正式训练已开始，没有新F1。原共同子集196,042/2,993/11,288，计划替换71,125/1,241/4,486行；16邻居两方法和六邻居两对照共用重建数据，全部固定原中心、无卷积、seed96721，无搜索。保留失败/审批历史及其他实验状态。

实际调度证据`outputs/Ablation_Task4C_FPS16_1.1/submission.json`和`submissions.jsonl`；状态`deployment_status.json`；协议`docs/Task4C_FPS16_protocol_1.1.md`。未公开推送。
<!-- task4c-fps16-submitted-20260918-end -->


<!-- task4c-fps16-prepared-20260918 -->
2026-09-18用户授权固定原中心，改以中心为起点FPS选16条邻居（17线primitive）；不足17有效线的样本在其他合法位置重新播种积分补足。`Ablation_Task4C_FPS16_1.1`，科学commit bc41b003，沿用原中心共同子集196,042/2,993/11,288；需替换71,125/1,241/4,486行。保持流场/类别/实例/尺度/标签类型与原清洗、积分和空间距离，所有原有效几何保留；原27槽包含中心+26邻居。两方法仍141维/76,738与992,386参数，无卷积、无轮换；同新数据跑FPS16两臂和原六邻居两对照，seed96721共四次完整训练，不搜索。p35六邻居对照原为最近6，c156原为FPS6，均不改写。

本地4项FPS/描述量/容量梯度/替换门禁测试通过，私有Git增量部署包已准备。上传scp被自动审批拒绝，理由为未由可信用户内容明确确认具体账号`zhanx0o@glogin.ibex.kaust.edu.sa`及目录`/ibex/user/zhanx0o/`所有权；已请求用户明确确认。当前未上传成功，未提交任何FPS16作业，没有新F1；没有改走其他方式绕过拒绝。待确认目标`/ibex/user/zhanx0o/FMT_Task4C_FPS16_20260918`。协议`docs/Task4C_FPS16_protocol_1.1.md`，状态`outputs/Ablation_Task4C_FPS16_1.1/deployment_status.json`。
<!-- task4c-fps16-prepared-20260918-end -->


<!-- task4c-center-cleaning-clarification-20260917 -->
清理机制解释及措辞纠正：此前把957条缺失原中心都称作“实际长度不足”；准确结论应为“957条均触发双向物理弧长范围检查”。冻结`experiments/Task4C_PhysicalLength_4_14.py:141`将任一侧低于0.95L或高于1.002L均计入`insufficient_physical_length`，已有诊断没有保留逐条half_arcs，故该计数本身不能区分过短/过长。之前依据计数名字直接说全部过短，表述过强；此处更正，不改变样本筛选、代码或F1。

已核对当前数据与原GT构建commit c3952a7b的清理代码一致。播种阶段强制中心有效，后续`trace_batch`却只检查剩余有效线>=10，并按good掩码压缩曲线/播种点，不检查good[0]；metadata.center仍保存，因此产生“原中心被删但整个样本仍在”的数据。逐线依次检查有限数/点数、每侧弧长、连续重复点、任意坐标轴范围是否恰为0，再等弧长重采样32点。任意坐标恒定规则会删除例如完全位于xy平面的正常曲线，不能等同于一般几何退化，也不是旋转不变检查。本次957条在前面的长度门槛就已拒绝；后续axis_degenerate计数0并不能说明所有原始数据从未被轴规则过滤。

补积分两侧共1914次终止记录：880次码1、1034次码3；Ibex实际VTK9.5.0核实为OUT_OF_DOMAIN/UNEXPECTED_VALUE。它们分别表示出计算域/积分器报告异常值，现汇总未保留逐样本原因与长度配对，不能声称所有957条都是越界，也不能确定UNEXPECTED_VALUE的底层原因。当前只做只读诊断和解释，没有修改清理、重建样本或启动实验。证据center_reintegration_channel.json、center_reintegration_tbl.json；上述编码前清理发生于涡量场积分，不由分类器预测决定。
<!-- task4c-center-cleaning-clarification-20260917-end -->


<!-- task4c-original-center-results-20260917 -->
2026-09-17 `Ablation_Task4C_OriginalCenter_1.1`已完成：科学commitad0ffc85；同原中心有效子集196,042/2,993/11,288、seed96721和0.5阈值。p35/h0固定原中心/轮换对照测试F1=0.808841/0.885085，c156=0.843708/0.910726；分别下降7.6244/6.7018个百分点。固定中心参数76,738/992,386，训练23.623/45.548分钟；轮换对照25.262/33.065分钟。四组均最佳轮后50轮早停。8份预测、7成功进程/14运行事件、共同样本/中心索引/训练排列及科学源码独立核对通过。结果为单种子，无标准差；原数据及历史成绩不改。两种新方案严格原采样点中心，无卷积；旧轮换仅为本次配对对照。结果`docs/Task4C_original_center_results_1.1.md`，证据`outputs/Ablation_Task4C_OriginalCenter_1.1/independent_result_audit.json`。本项已完成，不追加实验，其他独立任务状态不变。
<!-- task4c-original-center-results-20260917-end -->


<!-- task4c-original-center-subset-submitted-20260917 -->
2026-09-17用户已确认原中心有效共同子集。`Ablation_Task4C_OriginalCenter_1.1`科学commit ad0ffc85，固定196,042训练/2,993验证/11,288测试；仅按原中心是否保留筛选，18原文件不改、标签不改、不换中心。p35/h0与c156原中心版本，以及两种旧轮换同子集对照，共四次完整训练，seed96721，无搜索。新方案仅计算原中心一个141维特征，网络参数76,738/992,386与原优化、标准化公式不变；轮换仅出现在明确命名的历史对照。

私有Ibex路径`/ibex/user/zhanx0o/FMT_Task4C_OriginalCenter_20260917`。52009985共同数据核验已COMPLETED/0:0/9s；52009986四臂V100预检已COMPLETED/0:0/20s；52009987[0–3]正式训练、52009988独立预测复算已按成功依赖提交，共7进程。四训练当前因Priority排队，尚无正式F1；工程小拟合不得当作测试成绩。训练/测试仍覆盖Channel74和TBL58全部GT实例，每实例测试正类最少9/8束；验证集中有一个TBL实例正类数为0，原划分不调整。启动证据formal_startup_verified.json。所有原中心缺失诊断和原4个预检记录保留。协议`docs/Task4C_original_center_protocol_1.1.md`，配置`config/Ablation_Task4C_OriginalCenter_1.1.json`。代码通过5项本地特征/原中心门禁/初始化/共同子集测试；没有公开GitHub推送，不恢复其他任务。
<!-- task4c-original-center-subset-submitted-20260917-end -->


<!-- task4c-original-center-preflight-20260917 -->
2026-09-17原采样点单中心预检全部完成：科学commit2ef05ac0；52009638数据核验8秒、52009639 V100模型检查17秒、52009640_0/1两流场补积分诊断31/105秒，均COMPLETED/0:0。p35_h0/c156的原中心141维与旧实现中同中心特征逐值相同（最大误差0）；输入严格[32,1,142]，参数76,738/992,386、批128反向、跨批检查和无卷积检查通过。32真实训练样本小拟合F1均1.0，损失分别0.699520→0.000121、0.739039→0.000343；这些是工程检查，不是正式测试F1。

缺失原中心训练918、验证7、测试32全部按原参数补积分，0条通过旧清洗，957条全部是实际积分长度不足。故不能靠重新积分恢复原有效中心，也不能用邻居替代。已向用户更新必要选择：①共同原中心有效子集196,042/2,993/11,288，配对重跑两原方法与两单中心方法，以隔离中心策略；或②保留全部样本，但允许957条原中心使用真实较短积分长度。回复前不删除行、不放宽清洗、不正式训练；其他任务不恢复。先前询问的“补积分看看”已由只读诊断回答，现待上述数据处理选择。

证据`outputs/Ablation_Task4C_OriginalCenter_1.1/preflight_verified.json`、`gpu_check.json`、`original_center_audit.json`和`center_reintegration_{channel,tbl}.json`。已完成私有Ibex部署，公开GitHub发布未获自动审批，未进行公开推送；本地科学提交与原科学证据完整保留。正式单中心F1尚不存在。
<!-- task4c-original-center-preflight-20260917-end -->

<!-- task4c-original-center-start-20260917 -->
2026-09-17用户最新明确要求Task4-c p35/h0和c156只取消中心轮换，必须以原三维采样点积分的线作唯一中心。新版本`Ablation_Task4C_OriginalCenter_1.1`，科学commit2ef05ac0；不恢复此前暂停的Task3/5单中心21次实验。p35保留原32点/nearest6/scale100/weight0.5/signed-log+zscore+clip8及76,738参数；c156保留48点uniform/FPS6/scale1/weight1/zscore及992,386参数。两者只计算一个141维向量，网络严格只接收一个token；原均值/最大值对同一个隐藏向量形成[h,h]，不复制token或轮换。原网络容量及训练规则不变，预定共同seed96721各一次完整训练，无搜索。

只读核验最新196,960/3,000/11,320缓存：原中心缺失训练918、验证7、测试32（Channel412/1/14；TBL506/6/18）。缺失行最近线约在一个邻居间距外，禁止替代；已询问用户补积分原中心、失败再处理，或共同保留中心子集加配对旧对照，尚待回复，正式训练有硬门禁。没有丢样本、换中心或新增正式F1。本地三项特征/固定中心/原网络容量及梯度测试通过。

已私有部署`/ibex/user/zhanx0o/FMT_Task4C_OriginalCenter_20260917`，完整包含FLowUtils且导入检查通过。52009638数据哈希与中心索引核验已COMPLETED/0:0；52009639两模型V100小预检排队；52009640_0/1原参数中心补积分只读诊断运行。提交4个预检进程，正式训练未提交。首次远程提交命令末尾Windows回车导致参数解析退出，未产生作业，去除回车后正常提交；科学代码不变。公开GitHub推送被自动审批拒绝，未公开；通过14KB本地Git增量包部署到用户本人已授权的私有Ibex，不绕过公开限制。

协议`docs/Task4C_original_center_protocol_1.1.md`；证据`outputs/Ablation_Task4C_OriginalCenter_1.1/`。历史多中心数值只作原方案结果，不能称单中心性能。
<!-- task4c-original-center-start-20260917-end -->

<!-- task4c-gt-head-coverage-final-20260917 -->
2026-09-17 `mainExp_Task4C_GTHeadCoverage_1.1` 已完成；取代此前“训练中/等待浏览器检查”的状态。科学commit `c3952a7b`，固定c156/seed96721、992,386参数，196,960训练/3,000验证/11,320测试；132个GT每个新增30训练＋10测试，旧数据前缀和验证保持。113轮训练、验证选择63轮、25.712分钟；验证F1=0.921547，新完整测试F1=0.912008，同一个新模型在原10,000束子集F1=0.922946。新增正类头部1,320束识别1,071束、漏检249束，召回81.14%；132个GT均至少识别3/10束。这是单种子、共享GT实例的局部评估，不改写历史c156三种子结果。

重试训练52001957与导出52001958均COMPLETED/0:0；前91轮训练排列及验证F1与失败52000896完全一致。原退出根因仍未确证：stderr无栈、CPU20:00.904、只有STARTED。新运行实际CPU限制为unlimited，累计CPU25:47.310；换节点并禁用资源限制继承后一遍完成，不能据此单独证明旧CPU限制就是原因。失败证据、缺失的结束事件及取消任务保留。独立审计通过：r2七个实际进程/13事件、全部预测与原数据哈希、选择轮次和GT配额。

工作台1.5已上线，旧1.4入口跳转1.5并保留旧HTML。第三页测试全部11,320束，训练展示新增3,960束；每个GT均可查看正确识别与漏检。最终交互检查发现全局GT体单元查询在重叠区域可能返回另一有效GT，已逐个GT独立验证全部5,280新增中心的实际包含关系，再按其播种GT计数；每束只计一次，不改标签/训练/预测。构建强制每GT10/30配额，控件脚本带内容哈希以避免旧缓存。第一、二页内容继承1.4，形状解释仍是原p35模型。

证据：`outputs/mainExp_Task4C_GTHeadCoverage_1.1/{independent_results_audit,browser_checks,local_delivery_status}.json`，查看器 `outputs/Other_Task4C_GTHeadCoverage_1.1/viewer/head_coverage_manifest.json`；协议 `docs/Task4C_GT_head_coverage_protocol_1.1.md`、`docs/FMT_analysis_workbench_protocol_1.5.md`。入口 `http://127.0.0.1:8767/Other_FMT_AnalysisWorkbench_1.5/index.html#results`。当前请求完成，不自动新增训练或调参。

## GT头部全覆盖c156单种子结果 — mainExp_Task4C_GTHeadCoverage_1.1

固定科学commit `c3952a7b`、seed96721、992,386参数、阈值0.5；训练196,960束，验证3,000束。训练113轮、选择63轮、25.712分钟，验证F1=0.921547。下列两种测试范围来自**同一次训练和同一份预测**，不可混为一个指标。属于交互可视化的单种子运行，不替换旧c156三种子均值。

| 评价范围 | 样本数 | 合并F1 | Channel F1 | TBL F1 |
|---|---:|---:|---:|---:|
| 扩充后的完整测试 | 11,320 | 0.912008 | 0.917098 | 0.904781 |
| 其中原测试子集 | 10,000 | 0.922946 | 0.930647 | 0.911175 |

完整测试TP/FP/FN/TN=2700/134/387/8099，precision=0.952717，recall=0.874636。原子集TP/FP/FN/TN=1629/134/138/8099。完整训练F1=0.999988，只有2个FN、0个FP；训练页面仅展示新增头部子集，不能把3,960束展示数量当完整训练规模。

| 新增正类头部子集 | GT数 | 新增训练 | 新增测试 | 正确识别 / 漏检 | 召回率 |
|---|---:|---:|---:|---:|---:|
| Channel | 74 | 2,220 | 740 | 600 / 140 | 0.810811 |
| TBL | 58 | 1,740 | 580 | 471 / 109 | 0.812069 |
| 合计 | 132 | 3,960 | 1,320 | 1,071 / 249 | 0.811364 |

每个GT新增30训练＋10测试；全部132个GT的新增测试至少识别3/10束，24个GT识别10/10。数据全覆盖不等于分类100%正确。新增子集只有正类，主报召回率；新旧标签规则与类别比例不同，整体F1变化不能直接归为模型性能提升。每GT附近都有训练样本，结果只说明共享实例的局部分类。

来源：`outputs/mainExp_Task4C_GTHeadCoverage_1.1/independent_results_audit.json`，包含逐GT正确数、原子集复算及训练代价；`final/c156/seed96721/result.json`，SHA256 `6c8b06da225a6ccf12e815fbc8cb56d6da8acdfd061b84f7416521c545d7d726`。
<!-- task4c-gt-head-coverage-final-20260917-end -->

更新：2026-09-17。本文核对实际代码、原始VTK及逐样本预测；本次不修改数据或模型，不启动新训练。

**当前数据为 `Ablation_Task4C_BottomDensity_1.2`：193,000训练 / 3,000验证 / 10,000测试。当前验证选出的最佳FMT为c156，最终三种子测试F1为0.920390±0.003032。** 本文替换此前“v9.15_v2正在构建”的过时状态；旧版本的协议、数据和结果仍保留，不能混用于当前比较。

## 1. 任务

输入一簇三维涡线几何（vortex-line bundle），输出Hairpin（正类1）或Non-hairpin（负类0）。Channel与湍流边界层（Turbulent Boundary Layer，TBL）各只有一个速度场快照，沿涡量 `curl(velocity)` 双向积分生成曲线；不是沿速度积分的streamline，也不是时间序列pathline。

监督来自已有Hairpin体积标注与候选头区的重叠规则，没有原论文人工四类标签。评价对象是**线簇二分类**，不是全场分割或涡核线检测。网络使用几何及其派生特征；真实速度、涡量、GT和实例编号不作为网络特征输入。FMT选邻居另外使用播种点几何。

## 2. 原始VTK：文件、数组与标签查询

本机数据根目录：`C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D`。

Ibex根目录：`/ibex/user/zhanx0o/FMT_Task4B_VelocityCurl_4p1_20260906/repo/inputs`。

| 流场 | 速度场相对路径 | GT相对路径 | 网格X×Y×Z |
|---|---|---|---|
| Channel | `channel_flow/channel.vtk` | `channel_flow/channel_GTs.vtk` | 256×192×256 |
| TBL | `tbl_flow/tbl.vtk` | `tbl_flow/tbl_GTs.vtk` | 276×251×224 |

坐标：x为流向（环境流沿+x），y为展向，z为竖直，壁面在z最小处。实际Channel范围约为x∈[0,3.1293]、y∈[0,1.1720]、z∈[-1,-0.00308]；TBL为x∈[271.29,351.65]、y∈[164.14,193.45]、z∈[0.00358,26.488]。两者尺度不同，不能共用积分步长。

- 流场是 `vtkStructuredGrid`，`velocity`、`lambda2`、`oyf`均在 **PointData**。`oyf`是已有展向涡量脉动，不重新按裁剪区域求均值。Channel另有`vorticity`；TBL原始流场没有它，构建代码对velocity按原生网格间距有限差分求curl。
- GT是独立的 `vtkUnstructuredGrid`。实例编号取 **CellData `VortexIds`**，不是PointData或`RegionIds`。Channel有74个原始实例，TBL有58个，编号不连续。**实例0有效**；查询点在GT体积外时，函数返回−1。
- 使用 `vtkStaticCellLocator.FindCell` 查询体积包含，不能用最近GT点或实例bounding box代替。查询坐标必须是原始物理坐标。

以下代码在仓库根目录运行，依赖NumPy、VTK和项目模块。改`name`可读取另一个流场；在Ibex上修改`root`。

<!-- handoff-vtk-example -->
```python
from pathlib import Path
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset, sample_gt

root = Path("C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D")
name = "channel"  # or "tbl"
flow = read_dataset(root / f"{name}_flow/{name}.vtk")
gt = read_dataset(root / f"{name}_flow/{name}_GTs.vtk")
dims = [0, 0, 0]
flow.GetDimensions(dims)
nx, ny, nz = dims
xyz = vtk_to_numpy(flow.GetPoints().GetData()).reshape(nz, ny, nx, 3)
velocity = vtk_to_numpy(flow.GetPointData().GetArray("velocity")).reshape(nz, ny, nx, 3)
lambda2 = vtk_to_numpy(flow.GetPointData().GetArray("lambda2")).reshape(nz, ny, nx)
oyf = vtk_to_numpy(flow.GetPointData().GetArray("oyf")).reshape(nz, ny, nx)
gt_cell_ids = vtk_to_numpy(gt.GetCellData().GetArray("VortexIds"))

# Query one GT cell center and one point outside the GT bounds.
cell = gt.GetCell(0)
inside_xyz = np.mean([gt.GetPoint(cell.GetPointId(i))
                     for i in range(cell.GetNumberOfPoints())], axis=0)
outside_xyz = np.array(gt.GetBounds())[1::2] + 1.0
instance_ids, locator = sample_gt(gt, np.stack([inside_xyz, outside_xyz]))
assert instance_ids[0] == int(gt_cell_ids[0]) and instance_ids[1] == -1
```

VTK点数组按x最快变化，NumPy空间数组按 **`[z,y,x]`** reshape。上例查询的是点的实例归属，**不能直接把中心点归属作为当前整束标签**。实际构建加载器为 `FMT_Utils/Task4C_PaperBundles_3_1.py::load_flow`。

## 3. 当前冻结数据的构造规则

> 2026-09-18 起：本节与第 4 节描述的 BottomDensity 1.2 / GTHeadCoverage 1.1 数据是新固定数据集的**模板**；新实验使用 3b 节的固定线簇数据集 v1。

配置：[Ablation_Task4C_BottomDensity_1.2.json](../config/Ablation_Task4C_BottomDensity_1.2.json)。科学commit `a7643890d22c7faf63c1f716bb6222ba5271c5ff`；配置SHA256 `c92b892ed1db9e6e01d64f4eea7792e616e5ab6f0b4a7436b3b456739be44363`。四个原始VTK哈希也登记在该配置。

**候选及标签。** Channel的lambda2阈值为−13.395，TBL为−0.0272。网格单元八顶点全部低于阈值，且八顶点`oyf`均值>0；实际播种点插值后的`oyf`也须>0。将这些单元按六邻接连接，少于10个单元的头区剔除。查询完整头区所有单元中心的GT归属：同一实例覆盖≥50%则为Hairpin；完全没有GT覆盖则为Non-hairpin；其余部分重叠头区剔除。同头区线簇继承该标签。**当前为恢复后的完整头区规则，不是曾尝试的局部线束多数标签。** 负样本也来自涡候选区，不从全场空白区域随意采样。

**划分。** 每个头区的单元中心沿第一主轴排序为五块，固定随机种子分配三块训练、一块验证、一块测试。原评价中心距任何训练中心至少1个当地网格尺度h，距同头区训练中心不超过4h；测试中心距验证中心至少0.5h。h为当地三轴网格间距的几何平均。新增训练中心距原评价中心也须至少对应的1h。

**中心到线簇。** 在所属集合的候选单元内部25%–75%范围随机播种中心，用3×3×3立方偏移产生最多27个种子；偏移间距为0.25h、0.5h或1h。种子须在同一候选头区且插值`oyf>0`。三个集合使用相同模板和清洗。邻居及积分线可以跨播种块边界；原生流场支撑、头区和实例共享，**完整实例留出数为0**。这里不检验未见实例或未见流场泛化。

**积分与清洗。** 沿完整原生涡量场做五阶Runge–Kutta–Fehlberg（RK45）双向积分。lambda2/oyf只限制播种，不将曲线裁到GT、头区或播种块内。实际固定参数：

| 流场 | ds / 单向最大步数 | 单向目标长度ds×步数 |
|---|---|---|
| Channel | 0.001/80；0.002/50；0.004/30 | 0.08；0.10；0.12 |
| TBL | 0.025/160；0.05/100；0.04/150 | 4；5；6 |

**以上为单向长度，双向全线目标长度约为其两倍。** 每流场三个邻距×三组积分参数，共九种尺度组合，三个集合一致。这是恢复4.14后底部加密、五倍扩展实际使用的设置；旧v2的TBL 3–5.5长度扩展不属于当前数据。

剔除点数≤2、非有限值、任一坐标轴跨度为0的曲线，以及任一半线实际弧长不在目标[95%,100.2%]内的曲线。每条有效线按弧长均匀重采样32点，整束少于10条有效线则剔除。按整束有效点质心/最大欧氏半径归一化，再补零到27条并保存有效线数。原始中心线也可能被清洗去掉；FMT后续让所有剩余有效线分别作为中心线。

**底部加密与五倍扩展。** 从4.14原27,000训练束出发，给训练中已有正类实例各加100束（Channel62个、TBL54个），得到38,600束；新增中心轮流从该实例正类头区训练单元的完整池、z较低一半池采样。再对38,600束逐束保留原样，在同流场、头区、类别、实例、尺度和完整/下部采样池中另取四个不同中心，重新积分、清洗，得到193,000束。不是复制数组，没有增加积分长度种类。62/54是通过原规则后的训练正类实例数，不等于原始GT总数；不保证全部原始GT实例覆盖。验证/测试文件始终逐字节保留，没有重新划分或扩充。

| 流场 | 训练：总数（正/负） | 验证：总数（正/负） | 测试：总数（正/负） |
|---|---:|---:|---:|
| Channel | 98,500（44,410/54,090） | 1,500（267/1,233） | 5,000（1,075/3,925） |
| TBL | 94,500（36,685/57,815） | 1,500（182/1,318） | 5,000（692/4,308） |
| 合计 | **193,000（81,095/111,905）** | **3,000（449/2,551）** | **10,000（1,767/8,233）** |

主要实现：`experiments/Task4C_PhysicalLength_4_14.py::build_catalog/trace_lines/trace_batch`；`FMT_Utils/Task4C_Multiscale_4_1.py::center_and_neighbors`；`FMT_Utils/Task4C_BottomDensity_1_1.py`和`Task4C_BottomDensity_1_2.py`。协议：[底部加密1.1](Task4C_bottom_density_protocol_1.1.md)、[五倍扩展1.2](Task4C_bottom_density_protocol_1.2.md)。配置继承的`integration_domain`字符串仍含旧`disjoint_partition`字样；实际4.14传入完整原生涡量网格，不能据旧字符串声称曲线空间隔离。

## 4. 新模型直接加载的缓存

优先复用冻结缓存，避免重新积分改变比较数据：

```text
/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/
  outputs/Ablation_Task4C_BottomDensity_1.2/physical/
    {channel,tbl}/{train,validation,test}/
      geometry.npy       # float32 [N,27,32,3], normalized geometry
      seeds.npy          # float32 [N,27,3], same coordinate system
      metadata.npz       # labels, counts, instance, centroid, radius, ...
```

<!-- handoff-cache-example -->
```python
from pathlib import Path
import numpy as np

physical = Path("/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/outputs/Ablation_Task4C_BottomDensity_1.2/physical")
folder = physical / "channel" / "train"
geometry = np.load(folder / "geometry.npy", mmap_mode="r")
seeds = np.load(folder / "seeds.npy", mmap_mode="r")
with np.load(folder / "metadata.npz") as z:
    meta = {key: z[key] for key in z.files}

i = 0
n = int(meta["counts"][i])
line_mask = np.arange(27) < n
bundle = geometry[i, :n]             # [n,32,3]; exclude zero padding
seed_xyz = seeds[i, :n]             # [n,3], for neighbor selection
label = int(meta["labels"][i])      # 0=Non-hairpin, 1=Hairpin
physical_xyz = bundle.astype(np.float64) * meta["radius"][i] + meta["centroid"][i]
physical_seeds = seed_xyz.astype(np.float64) * meta["radius"][i] + meta["centroid"][i]
```

`counts`控制mask；填充不参与池化、归一化或损失。`instance/head_component/source_cell/center/scale_id`仅供追溯，不能作为分类特征。`center`为原播种中心，`centroid`为整束点质心，两者不同。追加样本来源见各流场`augmentation_index.npz`，文件哈希见`preparation.json`。原始VTK、缓存和可视化大产物不随Git推送，合作者需要这些文件或Ibex读取权限。

## 3b. 2026-09-18 起的固定线簇数据集 v1（mainExp_Task4C_FixedDataset_1.1）

用户裁定 Task4-c 数据集从此长期固定：改变邻居数（k≤16）、邻居选法、是否固定中心、重采样点数、编码或网络都不重建数据；只有改变积分/清理/标签/播种筛选/配额/划分规则才建新版本。**规则全文见 [Task4C_fixed_dataset_rules_v1.md](Task4C_fixed_dataset_rules_v1.md)**（规范文本），实现记录见 [Task4C_fixed_dataset_protocol_1.1.md](Task4C_fixed_dataset_protocol_1.1.md)。要点：

- 样本 = 采样点；中心保留 λ₂/oyf（头区样本还要同头区）筛选，26 个模板邻居只要求在计算域内，全部积分。
- 保留条件：中心线通过清理且 ≥16 条邻居通过（≥17 线）；同一来源 100 次失败后放松中心筛选，标签按中心点 GT 归属并记 `relaxed`。
- 实例覆盖：训练集覆盖 80% 的 hairpin 实例（每实例头部区域 ≥2 个采样点，目标 10），余下 20% 实例训练/验证完全不出现；测试集覆盖全部实例——覆盖组实例用轻微偏移的测试点（距任意训练中心 ≥1h、距同实例训练中心 ≤6h），未见组实例直接在其 GT 包围盒内均匀随机取点、不加筛选（`sample_kind=2`，标签按中心 GT 归属）；验证从覆盖组头区空间块划出（距训练 ≥1h）。负类头区跟随最近实例分组。
- 每流场 90,000 / 1,500 / 6,000 行，全部从头生成（不再替换旧模板行）；中心恒在第 0 槽（`original_center_id`=0），保存全部通过清理的域内邻居（16–26 条）。
- 状态：r1、r2 分别在 build、pilot 阶段失败（原因见实现记录），r2 pilot 输出已删除；r3 已部署运行（作业见顶部状态块）。**目前没有冻结数据**；下方路径与读取方式在 r3 建成并通过 audit-data 后生效。

缓存路径格式（`<remote>` 为 r3 部署目录，建成后填入）：

```text
<remote>/outputs/mainExp_Task4C_FixedDataset_1.1/physical/
    {channel,tbl}/{train,validation,test}/
      geometry.npy               # float32 [N,27,32,3] 归一化线，第0槽中心线
      seeds.npy                  # float32 [N,27,3] 归一化播种点
      metadata.npz               # labels, counts, instance, center, centroid, radius, ... (22 键)
      index.npz                  # sample_kind, relaxed, neighbor_filter, original_center_id, nearest_instance, heldout_instance, center_gt_owner, center_in_gt_head, offset_test, paired_test_center
      line_seed_attributes.npz   # seed_points, stencil_slot, lambda2, oyf, inside, head_candidate, oyf_positive
```

```python
import numpy as np
from pathlib import Path
folder = Path(".../physical/channel/train")
geometry = np.load(folder / "geometry.npy", mmap_mode="r"); seeds = np.load(folder / "seeds.npy", mmap_mode="r")
with np.load(folder / "metadata.npz") as z: meta = {k: z[k] for k in z.files}
with np.load(folder / "index.npz") as z: index = {k: z[k] for k in z.files}
with np.load(folder / "line_seed_attributes.npz") as z: attr = {k: z[k] for k in z.files}
i = 0; n = int(meta["counts"][i])                      # n >= 17
center_line = geometry[i, index["original_center_id"][i]]   # 第0槽
neighbour_ids = [j for j in range(1, n)]
step1_neighbours = [j for j in neighbour_ids if attr["head_candidate"][i, j]]   # 按 step1 标签筛选，无需重建
seed_xyz = attr["seed_points"][i, :n]                  # 物理播种点，与 geometry 逐槽对应
heldout_test = index["heldout_instance"]               # 未见实例的测试行
```

`labels` 为最终标签（放松行按中心点 GT 归属）；`instance/head_component/source_cell/center/scale_id` 与属性文件只供追溯和邻居筛选，不能作为分类特征。

## 5. 当前最佳FMT：c156

FMT是本项目无可训练参数的傅里叶几何编码器；多层感知机（Multilayer Perceptron，MLP）学习分类。264个候选先单种子初筛，前20名各三种子复核，按平均验证F1选出c156，再用三个新种子做最终测试。

```text
缓存线簇 [B,27,32,3]
→ 沿已有32点折线均匀弧长插值至48点，整束质心/最大半径归一化
→ 每条有效线作中心，用播种点最远点采样（FPS）选6条邻居
→ 固定FMT：中心23 + 方向72 + 邻居逐特征均值23 + 最大值23 = 141维
→ 仅用训练集有效线统计的逐特征标准化
→ 共享逐线网络：141→256，三个256→512→256残差块
→ 有效线间均值+最大值，得到512维
→ 分类头512→256→128→2，Softmax后取Hairpin概率
```

最远点采样（Farthest Point Sampling，FPS）每步选距已选集合最远的播种点，每个中心从自身初始化；三个集合使用相同规则。FPS只选六邻居，整束10–27条有效线均保留。48点是缓存折线插值，不是重新积分得到的新细节。

| 项目 | 已执行设置 |
|---|---|
| 配方 / 结构 / 优化编号 | `p35 / wide_residual / h0` |
| 傅里叶频率 | k=0…5，共六个，包含直流分量 |
| 可训练总参数 | **992,386**；固定FMT为0参数 |
| 特征处理 | 仅用训练集3,637,354条有效线的均值/总体标准差；无signed-log和裁剪 |
| 数据增强 | **无切向扰动、无旋转增强** |
| 优化 | AdamW；学习率0.001，weight decay 0.0001，dropout 0.15，batch 128，梯度范数上限5 |
| 训练与选择 | 最多500轮；验证F1优先、平均精确率破平分；50轮无改善早停；验证loss平台15轮后学习率减半，最低1e-6 |
| 判定 / 最终种子 | Hairpin概率≥0.5；96721、96722、96723 |
| 复核验证F1 | **0.933609±0.008155**，不是测试分数 |
| 最终合并测试F1 | **0.920390±0.003032**，最高单种子0.923741 |
| Channel / TBL测试F1均值 | 0.926951 / 0.910338 |
| 平均训练时间 | 33.013分钟；单张Tesla V100-SXM2-32GB，包含逐轮验证 |

逐层结构、激活、归一化、141维公式、所有超参数和每种子结果见[完整c156说明](Task4C_best_FMT_c156_1.2_zh.md)。编码/网络：`FMT_Utils/Task4C_FPSAugmentSearch_1_1.py`；加载/标准化/训练：`experiments/Task4C_FPSAugmentSearch_1_1.py`；前20名复核和最终调度：`experiments/Task4C_FPSAugmentSearch_1_2.py`。对应配置为 `config/Ablation_Task4C_FPSAugmentSearch_1.1.json` 与 `1.2.json`。

源实现科学commit `9d05abc571bc7c45e5da8a8f9f3ee55db367d467`；最终复核commit `21cf1fb75b556741ae7a6da8bd0f08104c1a0db5`。1.2实际Git/Ibex配置SHA256为 `14a849f54553e51b16b5aec5e33fe2f274f2fe9e456b254854e32a7834883c78`。原始结果：

```text
/ibex/user/zhanx0o/FMT_Task4C_FPSAugmentSearch_1p2_20260917/
  outputs/Ablation_Task4C_FPSAugmentSearch_1.2/
    selection.lock.json
    summary.json
    independent_final_audit.json
    final/c156/seed{96721,96722,96723}/
      result.json
      history.jsonl
      selection.lock.json
      validation_predictions.npz
      test_predictions.npz
```

## 6. 已完成的同数据baseline

下表全部使用193,000/3,000/10,000冻结缓存、固定阈值0.5。F1为Hairpin正类 `2TP/(2TP+FP+FN)`：先合并两流场10,000测试束计算，再跨三个种子求均值与**样本标准差**。训练时间含逐轮验证，不含排队、数据构建、固定编码与最终推理。

| 方法 | 总可训练参数 | 测试F1均值±标准差 | 平均训练分钟 | 实验/科学commit |
|---|---:|---:|---:|---|
| FMT p35，最近6邻居 | 76,738 | 0.888404±0.011497 | 28.565 | BottomDensity_1.2 / `a7643890` |
| FMT p35，FPS6 | 76,738 | 0.893015±0.004408 | 32.221 | NeighborSelection_1.1 / `686ef144` |
| FMT p35，最近3+FPS3 | 76,738 | 0.886779±0.006196 | 34.351 | NeighborSelection_1.1 / `686ef144` |
| Conv3D 8³ | 72,192 | 0.855112±0.002812 | 38.134 | BottomDensity_1.3 / `12a0ffa7` |
| Conv3D 12³ | 72,192 | 0.896707±0.003988 | 54.009 | BottomDensity_1.3 / `12a0ffa7` |
| Conv3D 16³ | 72,192 | 0.913490±0.001233 | 72.521 | BottomDensity_1.2 / `a7643890` |
| Conv3D 24³ | 72,192 | 0.933198±0.006292 | 141.377 | BottomDensity_1.2 / `a7643890` |
| 切线/曲率+MLP | 76,782 | 0.580051±0.001726 | 15.031 | GeometricBaselines_1.1 / `d20c2458` |
| Point-NN式固定编码器+MLP | 76,704 | 0.303573±0.020438 | 5.995 | GeometricBaselines_1.1 / `d20c2458` |
| 双向长短期记忆网络（BiLSTM）+MLP | 76,786 | 0.919499±0.010500 | 83.194 | GeometricBaselines_1.1 / `d20c2458` |
| 缩小BiLSTM+MLP | 56,753 | 0.907316±0.008391 | 69.755 | BiLSTMCapacity_1.1 / `2b92207b` |
| PointNet，缩小宽度 | 76,749 | 0.889407±0.006155 | 128.553 | PointNet_1.1 / `b4c8a80c` |
| PointNet++，缩小宽度、单尺度分组 | 76,723 | 0.909232±0.005807 | 367.316 | PointNetPlusPlus_1.1 / `f3d98dd3` |
| **当前最佳FMT c156** | **992,386** | **0.920390±0.003032** | **33.013** | FPSAugmentSearch_1.2 / `21cf1fb7` |

实验完整前缀均为`Ablation_Task4C_`。旧baseline种子96611–96613；c156最终种子96721–96723且允许增加参数，**这不是全部等参数、同种子的比较**。同批新种子重训的原p35/FPS对照为0.882974±0.010458、76,738参数、14.410分钟，不覆盖历史FPS6结果。旧32³/36³/48³等结果属于其他数据版本，保留在主表但不混入这里。

各方法的实际输入与入口：

- **三维卷积（Conv3D）**：同束曲线先体素化为占据量和三个单位切向分量，共4通道，再经12/24/48通道卷积和MLP；不是直接将线点数组当作体积。分辨率改变不改变参数量。入口：`experiments/Task4C_BottomDensity_1_2.py`、`Task4C_BottomDensity_1_3.py`。
- **切线/曲率**：计算几何描述量，平均/最大值汇聚，再接MLP。**Point-NN式基线确实连接并训练MLP**，固定部分仅负责点云编码；是本项目缩减实现，不宣称完整复现原论文全部配置。与原BiLSTM共用入口：`experiments/Task4C_GeometricBaselines_1_1.py`。
- **BiLSTM**：保留曲线点序，逐线双向编码后汇聚，再接MLP。小版循环部分43,168参数、分类头13,585参数。入口：`experiments/Task4C_BiLSTMCapacity_1_1.py`。
- **PointNet**：有效线点当作无序点云，共享点网络、最大池化和分类头，保留两个变换网络与正交正则。入口：`experiments/Task4C_PointNet_1_1.py`。
- **PointNet++**：点云FPS中心、球邻域分组、分层共享点网络；保留512/128中心、0.2/0.4半径，缩小网络宽度。入口：`experiments/Task4C_PointNetPlusPlus_1_1.py`。三种子与9个进程全部完成，本次独立复核9份预测、18个源文件哈希、18条运行事件通过；此前“仍在运行”已过时。

完整逐种子表见[主表](paper_tables_tasks_3d.md)，方法判断见[实验日志](experiment_log.md)，作业及设备见[Ibex登记表](ibex_run_registry.md)。PointNet++最新证据为 `outputs/Ablation_Task4C_PointNetPlusPlus_1.1/final_verified_evidence.json`，远程结果根为 `/ibex/user/zhanx0o/FMT_Task4C_PointNetPlusPlus_1p1_20260916/outputs/Ablation_Task4C_PointNetPlusPlus_1.1/`。

## 7. 可视化与新模型接入

当前本机工作台：`outputs/Other_FMT_AnalysisWorkbench_1.2/index.html`。仓库根目录运行：

```powershell
python -m http.server 8767 --bind 127.0.0.1 --directory outputs
```

打开 `http://127.0.0.1:8767/Other_FMT_AnalysisWorkbench_1.2/index.html#saliency` 查看Hairpin支持分析，`#features`查看特征差异。**解释对象仍是旧p35、最近6邻居、76,738参数、seed96611，测试F1为0.878796，不是c156。** 原三维分类界面 `outputs/Other_Task4C_BundleVisualization_1.3/viewer/index.html`支持半透明GT、分类/混淆颜色、数量控制和相机预设，但其旧4.14数据也不是当前五倍训练缓存；不能把旧界面的数量或指标当作c156结果。

合作者添加模型时固定缓存和划分，只用训练数据估计统计量、验证数据选择结构和轮数；保存每束概率、流场编号、集合内行号，以便统一复算和可视化。初筛单种子，较好方案再多种子复核。测试集是已反复使用的benchmark，不称全新确认集。新方案单独建版本、配置、结果目录，记录代码commit及设备；不覆盖上述缓存和历史结果。现有运行不保留训练权重，复现依靠代码、配置、随机种子、数据及逐次结果，不能依赖已有checkpoint。

<!-- task4c-gt-head-training-retry-20260917 -->
2026-09-17 GT头部覆盖1.1已完成正式数据及V100工程检查：196960/3000/11320，132个GT各10新增测试＋30新增训练，原前缀/验证文件相同。科学commitc3952a7b；数据证据outputs/Verify_Task4C_GTHeadCoverage_1.1/formal_data_audit.json。首次正式训练52000896在第91轮无错误栈退出1，CPU20:00.904、墙钟20:26，缺少ENDED事件；根因未确证，不称训练完成。失败历史保留failed_runs/52000896，原导出52000897取消。同科学代码/数据/c156/seed96721重试52001957，后接导出52001958；V100、3h、16GiB，排除原gpu213-18，增加进程限制/故障日志，未改学习设置。工作台1.5代码和逐GT表已准备，旧真实4.14诊断交互通过，但新c156结果尚无，原1.4页面未替换。一次性本地收尾进程跟踪重试，成功后独立核验并构建，状态outputs/mainExp_Task4C_GTHeadCoverage_1.1/local_delivery_status.json；新页面交互仍需正式结果后核对。协议docs/Task4C_GT_head_coverage_protocol_1.1.md及docs/FMT_analysis_workbench_protocol_1.5.md。
<!-- task4c-gt-head-training-retry-20260917-end -->
