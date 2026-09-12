# FMT 完整实验记录

规则（与全局研究规范一致）：
1. 版本命名：主实验 `mainExp_[xxx]_x.y`；组件验证 `Verify_[xxx]_x.y`；非主线探索 `Other_[xxx]_x.y`；消融 `Ablation_[xxx]_x.y`。`[xxx]` 是简短实验名，x = 大迭代，y = 小迭代。
2. 每行必须可追溯：技术细节 + 主要代码路径 + config + commit + 指标。没有数字的结论不进表。
3. 修订旧结论必须新旧并列写明变更原因，禁止静默翻转。
4. Baseline 冻结：VortexTransformer（已发表）冻结在 PyflowVis 仓库，不复制不修改；本仓库的 legacy 聚类管线（mainExp_1.1 待跑）一经记录即冻结。
5. Task1–Task5 的含义以 `docs/research_tasks_and_protocol.md` 为准：Task3 是固定尺度 IVD 监督二分类；streamwise/spanwise/hairpin 多分类属于 Task4；Task5 是可变尺度 IVD 监督二分类。
6. 所有 Ibex 进程必须逐 job 登记到 `docs/ibex_run_registry.md`，字段至少包含实验 ID、任务/版本、开始时间、运行结果、支持/反对结论和实际 GPU；失败 job 不得删除。
7. 表内“当前主结果”等状态词保留的是该行写入时的历史状态；截至2026-09-02，现行3D统一配置主结果为Task1-4.1、Task2-6.2、Task3-9.2与Task5-1.1，完整历史对照见`docs/paper_tables_tasks_3d.md`。
8. 本表不得因文档合并而删减失败、取消、预注册或已被取代的实验。跨实验核心结论与论文表不在本表重复总结，分别维护在`docs/mainExp_Task3_3D.md`、`docs/Verify_experiments.md`。

## 2026-09-05 摘要勘误（不重算指标、不删除历史行）

开源前整理将验证结论合并到 [验证实验总表](Verify_experiments.md)，旧文件的恢复方式见
[维护说明](repository_maintenance.md)。历史版本表中的代码/config 路径保留原值，部分现位于本地源文件归档。

- 旧摘要将“删除 chirality/余弦后 F1 为负变化”写成组件有负贡献；现改为组件有正贡献。
  依据是 `Ablation_Task123_FMTComponents_1.1/1.2` 的删除实验，错误来自把删除操作的影响与组件作用混淆。
- 旧摘要称旋转下非 FMT 对照全部崩溃，Task1 表漏列传统几何臂；现明确旋转 90° 时
  几何特征 F1 .447481、普通 DFT .208832、FMT .601347。
  依据为 `Verify_Task123_NoiseTypesStrong_1.1/robustness_table.csv`，这是遗漏对照的解释错误。
- 旧摘要称截断下三任务 FMT 都最鲁棒；现改为 Task1/2 有明确反例：保留前 50% 轨迹时
  普通 DFT/FMT 为 .512/.282，调优 Raw/FMT 为 .596/.338。
  依据为 `Verify_Task123_NoiseRobustness_1.2`；此前概括超出了结果支持范围。
- 现有 `line_offset` 只整体平移邻居轨线，没有重新积分，不能当真正的种子扰动；
  邻居频率块整体删除也不能当只删除聚合操作。以上修正不改变原配置、代码或数值。

## 2026-09-06 FMT 范围说明（保留原实验分支名称与数值）

用户明确 FMT 指傅里叶特征提取模块，包含普通 DFT/幅度特征，不限于主表选定的某个几何配方。
以下旧行中的 `fmt` 是机器结果中的具体分支名，不代表全部 FMT 实现；`plain_dft` 与
`plain_magnitude` 也属于用户所定义的 FMT。此前把普通傅里叶排除在 FMT 外、进而笼统称
“本次结果不支持 FMT 鲁棒性”超出了证据范围。现修正为：部分固定几何配方退化明显，
但普通傅里叶幅度实现有坐标噪声优势，例如 GeometryControls 1.1 中邻距0.5%高斯噪声下
Task5 普通幅度 F1 `.6769→.6696`、同结构几何网络 `.7902→.6273`；这支持该傅里叶实现的
局部抗噪价值，不证明所有实现/扰动均优越。Task1/Task5干净数据几何对照更高的数值仍保留。
不得按测试强度逐项选择最佳傅里叶分支后拼成一个统一方法的结果。

## 版本表

| 版本 | 日期 | 任务 | 技术要点 | 主要代码路径 | config | 指标（协议见下） | 结论 |
|---|---|---|---|---|---|---|---|
| Verify_Task123_GeometryParameterStress_1.1（已部署，故意偏离正常配置的诊断） | 2026-09-06 | Task1/2/3 几何基线退化维度 | 用户明确要求故意调差几何超参数。10数据条目×3种子×3任务共90分片。保留正常seed-time七线几何IVD参考；12组几何参数含正常配置、差分跨度4/16/31、平滑窗9/31、伪逆截断.99/1.01、均值权重0/2、特征缩放.01/100，仅在冻结正常模型推理阶段替换；6档决策阈值偏移为正常阈值±.5/2/8倍训练分数标准差，分数保持不变。训练侧每次只改一项：Task1 KMeans迭代1次/初始化1次/随机初始化；Task2 VAE仅10步/KL权重100/学习率1e-7；Task3仅1epoch/weight decay100/学习率1e-7。每个训练变体单独用验证集定阈值或簇映射，所有变体冻结后才评分确认集 | `FMT_Utils/GeometricControls_3D.py::stressed_seed_ivd`；通用 `experiments/Run_GeometryParameterStress.py`、`experiments/Audit_GeometryParameterStress.py`；不修改原始提取器和训练器 | `config/Verify_Task123_GeometryParameterStress_1.1.yaml`；源码`aee4bd56`；preflight `51392255` PASS；Task1/2/3 arrays `51392311/51392312/51392313`，summary `51392342`→audit `51392361`；正常Task2为新增几何标量补零189维→原冻结512/256隐藏层、64维latent VAE，7000步、beta1e-6、lr3e-4；不替代历史Raw/FMT主表 | Ibex全部15项测试及30个任务×数据集、340切片证书预检PASS。Task1已完成30/30子任务、1170指标行；Task2首个A100训练中、Task3排队中；最终预计3510指标行，含F1/AP/IoU/precision/recall/balanced accuracy、四格混淆计数、预测正类率、特征分布和相对正常配置下降。暂无性能结论 | 只回答故意错设参数如何影响该几何基线；不能把退化配置作为最强基线，也不按确认集挑最差配置。几何参数改变仅发生在推理阶段，不能混称该参数下重新训练的模型表现；训练参数变体则重新拟合。阈值错设应只改分类指标、不改排序指标。伪逆>1为显式零特征负对照。保留无变化和改善结果，不能预设每一项都会降分。FMT按用户定义包含普通DFT/傅里叶，本次不修改任何傅里叶实现或将其重新排除在FMT之外 |
| Verify_Task135_GeometricControls_1.1（完成，独立指标审计 PASS） | 2026-09-05 | Task1/3/5 简单几何与 FMT 公平比较 | 不改既有确认集、IVD p95 标签和冻结 Raw 主干；从七线对向间距矩阵 D 重建 Ddot·pinv(D)，真实非均匀物理时间求导，减去未使用标签的采样点平均涡量。Task1 比较几何 IVD 直接阈值、1D KMeans、普通 DFT 复数/幅度、无 Fourier 四通道几何时间序列、原 fmt_all+kin4；Task3/5 比较直接阈值及五种表示进入同一残差分支，另复放 Raw。各辅助输入无损补零至268维，64维辅助嵌入，所有学习臂参数量完全相同；保留各任务原 FMT 配方，Task3 为 aivd1w3_dft（前三时刻的零频总和），Task5 为 all+gram2+index_kin6。100 epoch、patience20、学习率.001，5个配对种子40–44。全部模型和验证集阈值冻结后才评价确认集 | `FMT_Utils/GeometricControls_3D.py`；`experiments/Run_Task135_GeometricControls.py`；独立 `experiments/Audit_Task135_GeometricControls.py`；复用冻结 residual 训练器，不修改旧算法 | `config/Verify_Task135_GeometricControls_1.1.yaml`；源码提交及作业 ID 见 Ibex 登记表 | 源码 `39471787`；arrays `51360408/51360409` 全150子任务完成；summary `51360430`、audit `51360437` 完成。10数据条目等权、5种子平均 clean F1（Task1/3/5）：FMT `.6015/.8474/.6735`；几何直接阈值 `.8593/.8857/.8290`；几何 IVD+KMeans/同结构网络 `.6337/.8406/.7902`；无 Fourier 几何 `.6372/.8033/.7573`；普通幅度 DFT `.5847/.7425/.6769`（Task5普通复数DFT `.6785`）。表与逐次数据：`outputs/Verify_Task135_GeometricControls_1.1/`，独立审计PASS、34000指标行及107100分尺度指标行，表SHA `fd1f1a9d…a6844a4e` | 旧结论→新增证据：FMT优于此前Raw/普通DFT的结果仍按旧协议保留，但不能据此推断超越简单物理量；原对照缺少从七线直接重建标签相关涡量偏差。现在：Task1 FMT不及几何KMeans；Task3相对同结构几何网络仅+.0068，未做显著性检验；Task5低于几何网络−.1167；三个任务均不及几何直接阈值。反对“FMT普遍提供简单物理量之外的预测收益”；也不能仅凭本比较断言收益完全来自网络。Task1阈值用验证标签校准，是诊断对照，不冒充纯无监督KMeans。几何IVD是采样均值偏差而非直接读取标签；Task3 FMT为前三时刻零频和，不能当非零频抗噪证据。新同结构容量匹配比较不覆盖历史主表；无Fourier几何是时间域控制，不是严格单算子消融 |
| Verify_Task135_PairedRobustness_1.1（完成，独立指标审计 PASS） | 2026-09-05 | Task1/3/5 坐标噪声、丢帧、截断配对测试 | 使用上行相同轨线、标签和冻结模型。坐标高斯σ为每条primitive自身原始邻距的.001/.005/.01/.02/.05/.10；丢帧10/25/40%，每个primitive独立取缺失时刻、七线共享、保留端点并按物理时间插值；保留前75/50%轨线，只使用保留点重采样并更新物理时间，不读取已截断未来点。3个扰动重复17068/27068/37068，独立于训练种子，所有方法共享逐点相同实现；干净条件只评估一次 | 同一 `experiments/Run_Task135_GeometricControls.py` 和独立审计，避免复制第二套训练器；`ibex_bash/task135_geometric_controls.sh` | 同一 `config/Verify_Task135_GeometricControls_1.1.yaml`；输出 `outputs/Verify_Task135_GeometricControls_1.1/` 下 clean_comparison、robustness_table、逐次预测与 Task5 逐尺度结果 | 150分片、34000指标行完整，全部11扰动强度×3重复，审计PASS；表见同目录 `robustness_table.csv`。Gaussian σ=.005（自身邻距0.5%）FMT clean→noisy F1：Task1 `.6015→.3365`（降.2650），Task3 `.8474→.7726`（降.0748），Task5 `.6735→.5112`（降.1623）；普通幅度DFT分别 `.5847→.5836`（降.0011）、`.7425→.7333`（降.0092）、`.6769→.6696`（降.0074）。Task3同结构几何网络 `.8406→.7332`（降.1074）。40%内点dropout FMT下降 `.0043/.0008/.0005`，多数对照同样接近不变。只保留前50%轨线 FMT `.2441/.4903/.5533`（降.3575/.3571/.1202），几何阈值约 `.8593/.8857/.8290`（接近不变） | 不支持跨Task1/3/5的稳定鲁棒性优势。局部正结果：Task3在σ=.005/.01/.02时FMT的绝对F1高于全部本次对照，且相对几何网络下降较小；但普通Fourier下降更小，σ=.05/.10时普通Fourier绝对性能反超。Task1/5逐点高斯与截断不支持FMT更可靠；丢帧后各方法都较稳定，不能归为FMT独有优势。截断保留最初时刻，seed-time几何量几乎不变属于该扰动设定的预期性质，不是普遍缺失鲁棒性。不得挑选有利强度或用领先Raw代替下降报告。噪声单位、标签、配对样本与干净冻结模型均固定，未做噪声增强 |
| （待跑）mainExp_1.1 | — | Task1 | 旧法冻结复跑：FMT(PosE+同时刻跨线分组+LGA+Pool, stages=2, embed24, alpha1000, beta19, temporal_head=None) + KMeans(k=2)，4 输入视图；补定量协议与确定性设置。注：分组已从 GeoLinePicker 重构为 `group_same_timestep`，经测试与旧实现逐位一致（commit ca653fd），故仍视为"旧法冻结"。复跑前已修复两处数据侧 bug（commit 944d206）：时间重采样改为先过滤无效 primitive；积分后端 CPU/CUDA 语义统一——它们改变的是"数据正确性"而非方法本身。eval/train 模式两臂均测（协议要求） | `experiments/FMT_Clustering.py`, `FMT_Utils/FMT_encoder.py` | `config/PathlineFMTclustering.yaml` | ARI / NMI / F1(vs IVD阈值) / F1(vs Vatistas解析标签) | — |
| （待跑）Verify_objectivity_1.1 | — | Task1 | Killing 观察者不变性测试：随机时变刚体观察者变换场 → 重积分 → 重编码 → 特征漂移量化 | `FLowUtils/KillingObserver2D.py`（符号先校准，见问题分析 P7） | — | 特征相对漂移 / 聚类标签翻转率 | — |
| mainExp_3DFMT_1.1 | 2026-08-17 | Task1-3D | 每个种子生成 7 条 pathline（center、x±、y±、z±）；中心位移与邻居相对位移先做时间差分；三维实序列逐坐标做 Fourier 变换，每个频率取 Gram 不变量并附加旋向三重积；邻居特征排序池化；StandardScaler + KMeans(k=2)；输出真实物理比例的 3D/正交投影/Z截面/pathline 图，并计算同一时刻 3D IVD 体数据及 p90/p95/p97.5 等值面 | `experiments/FMT_Clustering_3D.py`, `FMT_Utils/DFT_FMT_3D.py`, `FMT_Utils/FMT_3D_pipeline.py` | `config/PathlineFMTclustering3D.yaml` | 合成 smoke：180/180 valid，feature `[180,77]`，cluster `[60,120]`。真实 `halfcylinderRe160Resampled.nc`：两次均为 7200/8000 valid、feature `[7200,161]`。默认 `t=3`：cluster `[7000,200]`，少数 cluster 1 对 IVD 的 p90/p95/p97.5 F1=.347/.558/.833。指定 `--seed-time 7`：cluster `[828,6372]`，少数 cluster 0 对 IVD：p90 F1=.674/IoU=.509/P=.774/R=.597；p95 F1=.736/IoU=.582/P=.615/R=.917；p97.5 F1=.468/IoU=.306/P=.306/R=1.000 | `t=3` 少数簇只对应近圆柱最强 IVD 核；`t=7` 尾迹已发展为明显的波动结构，少数簇沿近尾迹展开，和 p95 IVD 区域最接近。它覆盖全部 p97.5 强 IVD 点但相对该窄阈值过分割，因此不能只凭最高分位判定成败。KMeans 簇编号每次运行可交换；这里的“少数簇”按样本数识别。IVD 百分位只是可追溯参考阈值，不是唯一 ground truth。只保证常数旋转/常数平移不变，不保证时变观察者下的完整客观性。 |
| Verify_IVDSearch_1.1 | 2026-08-17 | Task1-3D 诊断 | 定义局部扩展 `IVD_a=&#124;&#124;ω-mean_a(ω)&#124;&#124;`；搜索 `a=[3,5,7,9,11,15,global]`，有限 `a` 是 `a³` 体素 box mean，`global` 是标准全域 IVD；对每个 `a`、两个簇和种子处全部可区分阈值精确搜索 F1。为防止“几乎全场为涡”的多数类退化解，涡候选要求簇和 IVD 正类均不超过 50%；无约束最优仍落盘供审计 | `FMT_Utils/IVD_parameter_search_3D.py`, `FLowUtils/ScalarField3d.py`, `experiments/FMT_Clustering_3D.py` | `visualization.ivd_search_averaging_sizes` | `halfcylinderRe160Resampled.nc`, `t=7`：涡候选最优 `a=11`, level=`0.913665`, cluster 0，F1=`.853`, IoU=`.743`, P=`.835`, R=`.871`。无约束最优 `a=3`, level=`3.77958e-7`, cluster 1，F1=`.939`，但 IVD 正类覆盖 100% 种子、所选多数簇占 88.5% | 局部 11³ 体素平均比标准全域平均更接近当前少数簇；全域基线对 cluster 0 的最佳 F1=`.757`。该搜索使用聚类标签选择 IVD 参数，属于事后相似度诊断，不能作为独立 ground truth 或无偏性能指标。 |
| Verify_3DFMTHyperparam_1.1 | 2026-08-17 | Task1-3D 超参数 | 冻结 `halfcylinderRe160Resampled, t=7, local-IVD a=11, level=.913665` 和同一批 7200 个 primitive；搜索 7 个频率数 × 2 种 Fourier 不变量 × chirality 开/关 × 3 种邻居池化 × 5 个邻居权重，共 420 组。粗筛 KMeans `n_init=1`；前 10 名用 5 个 random state、每次 `n_init=10` 复核。`neighbor_scale` 因会被逐列 StandardScaler 严格抵消而不列入搜索 | `experiments/Verify_3DFMTHyperparam.py` | `config/Verify_3DFMTHyperparam_1.1.yaml` | 最佳：`num_freq=6, mode=gram, chirality=true, pool=sort, neighbor_weight=.5`，5 次 mean/min/max F1 均为 `.852750`，std≈0。第二名关闭 chirality：F1=`.851697`；第三名 `freq=8` 且其余同最佳：F1=`.850450`。420 组加稳定性复核耗时 100.76 s（CUDA） | 当前 `mainExp_3DFMT_1.1` 配置恰为本搜索空间内稳定最优，不修改冻结基线。chirality 收益仅 `.00105`，很小但在 5 个 KMeans seed 上稳定。该结论只来自一个场的一个时刻，并且参考 IVD 参数曾在同一数据上选择；它是训练集内超参数结果，必须在其他时间和其他 3D 场上做 held-out 验证后才能称为可泛化最佳。 |
| mainExp_3DFMTVAE_1.1 | 2026-08-17 | Task2-3D | 固定 cylinder3d `t=7` 的 7200 个 primitive 和 local-IVD `a=11, level=.913665`；无标签随机 80/20 train/test。四臂：raw local primitive direct、raw→VAE、FMT direct、FMT→VAE。raw 先减中心线起点；两种输入的 StandardScaler 只 fit train。两种 VAE 共用 hidden `[256,128]`、latent 16、β=.001、200 epoch；5 个训练 seed。KMeans 只 fit train latent，IVD 仅用于 test F1 | `experiments/Train_3DFMT_VAE.py`, `FMT_Utils/VAE_3D.py` | `config/mainExp_3DFMTVAE_1.1.yaml` | held-out F1：raw direct `.5935`；raw+VAE mean `.5644±.0241`（`.5407–.5961`）；FMT direct `.7961`；FMT+VAE mean `.7815±.0634`（`.6563–.8254`）。FMT+VAE − raw+VAE=`+.2171`；FMT+VAE − FMT direct=`−.0145` | FMT tokenizer 对 VAE 有明显价值：相同 VAE 主体下比 raw 输入高 `.217`。审计修订：该版加载 FMT feature 后漏施 Task1 最佳配置的 post-StandardScaler neighbor weight `.5`，实际权重为 `1.0`；不否定“FMT vs raw”的方向，但不是严格最佳 FMT 配置，不能与后续 `.5` 实验混作同一协议。本版也不能证明“VAE 提高 FMT feature”：平均值低于 FMT direct。 |
| Verify_Task2Universality_1.1（8 数据条目） | 2026-08-18 | Task2-3D 普适性 | 七个真实时变数据从原始时间索引 `[20%,90%)` 均匀抽 10 片；steady `channel.vtk` 采用经 PyflowVis `referenceframe_inr_3d/frame3d.py` 验证的 3D Killing observer：`xi=R^T x-D`，`v=R s(xi)+t+w×x`，生成 10 个客观等价时刻。每片 16³ seeds、7-line primitive、FMT 最佳配置及 post-StandardScaler neighbor weight `.5`；标准全域 IVD p95 标签不在新场调参。前 8 片无标签训练，最后 2 片 held-out；raw/FMT VAE 同结构、latent16、β=.001、约 4600 optimizer steps、3 seeds | `experiments/Build_Task2_Universality_Cache.py`, `experiments/Build_Channel_Killing_Cache.py`, `experiments/Run_Task2_Universality.py`, `experiments/Summarize_Task2_Universality.py`, `FMT_Utils/NetCDF_window_3D.py`, `FMT_Utils/KillingObserver3D.py` | `config/Verify_Task2Universality_1.1.yaml` | FMT+VAE − raw+VAE：halfcylinder Re160 `+.018`（`.213±.009` vs `.195±.032`）；Re640 `−.011`（`.284±.017` vs `.295±.012`）；Re6400 `−.122`（`.203±.014` vs `.325±.003`）；tangaroa `+.084`；deltaWing-resampled `+.350`；未降采样 deltaWing `+.172`；F22 `+.119`；synthetic channel `+.002`。三个 Reynolds number 的 direct FMT 均优于 raw direct：Re160 `.259>.239`、Re640 `.496>.332`、Re6400 `.556>.366` | 新增 Re640/Re6400 推翻了此前“所有场的 FMT+VAE 均提升”结论：更高 Reynolds number 下，FMT feature 自身更可聚类，但当前重构型 VAE 会损失这些判别信息，且 Re6400 损失显著。旧结论只覆盖当时六项，不能外推。保守统计将三个 half-cylinder 合并为一个 family、两个 deltaWing 合并、排除 synthetic channel；因此结论应拆开：Task1 的 direct FMT 在三个 cylinder Reynolds number 均成立，Task2 的当前 VAE 改善不普适。 |
| Verify_VAEFailureHighRe_1.1 | 2026-08-18 | Task2-3D 失败诊断 | 冻结 Task2 数据、split、FMT、IVD、hidden layers、β和约4600步预算；Re160/640/6400 扫 latent `[4,8,16,32,64]`×3 seeds。按 center/neighbor 和 real norm/imaginary norm/cosine/chirality 报 test reconstruction MSE、normalized MSE、loss fraction，并做 block-only direct KMeans | `experiments/Diagnose_VAE_HighRe.py` | `config/Verify_VAEFailureHighRe_1.1.yaml` | Re640 最佳 latent32 `.332±.061`，高于 latent16 `.284±.021`；Re6400 最佳 latent32 仅 `.210±.008`，远低于 direct FMT `.556`。Re6400 MSE 从 latent4 `.1100` 降至 latent32 `.0367`，F1 几乎不变。latent16 下 neighbor/center normalized-MSE 比为 Re160 `2.06`、Re640 `2.55`、Re6400 `2.05`；chirality 始终是最难重构语义 block。Re6400 chirality-only direct F1=`.580`，是所有 block 最高 | Re640 有容量不足成分；Re6400 主要是 reconstruction MSE 与涡判别信息不一致。post-StandardScaler neighbor weight `.5` 使每个 neighbor slot 的 MSE 惩罚降至 center 的 1/4；同时 VAE 最差地重构了 Re6400 最有判别力的 chirality。不能以总体重构 MSE 下降推断 feature quality 提高；下一步应做 block-weighted loss 因果验证。 |
| Verify_VAEObjective3D_1.1 | 2026-08-19 | Task2-3D VAE 改造 | 冻结最佳 pathline `.25/48/32`、数据 split、β、4600 steps 和 3 seeds；对 Re640/Re6400 测试无标签 block-balanced reconstruction、与 Raw+VAE 参数量匹配的 hidden `[480,256]`、两者组合及 latent32+balanced。平衡权重仅由训练 FMT 的 7 line slots×4 semantic blocks 能量计算，不用 IVD 标签 | `experiments/Verify_VAEObjective3D.py`, `FMT_Utils/VAE_3D.py`, `docs/Verify_experiments.md` | `config/Verify_VAEObjective3D_1.1.yaml` | Re640：Raw+VAE `.2952±.0119`；latent32+balanced `.3314±.0171`，逐 seed 差 `+.0005,+.0714,+.0367`；capacity matched 均值最高 `.3463` 但 std `.0859`。Re6400：Raw+VAE `.3253±.0033`；所有 FMT+VAE 均更差，最佳仍为既有普通-MSE latent32 `.2102±.0061`。balanced 将 neighbor/center normalized-MSE 从 Re640 `2.55→1.12`、Re6400 `2.05→.87`；Re6400 latent32+balanced chirality normalized MSE `.115`，但 F1 `.1864` | VAE 调整后 Task2 在 Re640 可成立，在 Re6400 仍明确失败。修订上一诊断：neighbor/chirality 重构偏置是真实因素，但纠正逐块 MSE 不足以恢复 Re6400；模型参数量也不是主因。普通重构目标不保持稀少涡样本的 latent 邻域结构，下一步应验证无标签 metric-preserving objective，不能再把 reconstruction MSE 当 feature quality。 |
| Verify_PathlineHyperparams3D_1.1 | 2026-08-18 | Task2-3D 路径线超参数 | 8 个数据条目、固定 10 个 source times；7 个 variant 比较积分步长、总步数/观察窗和积分后采样数。所有 F1 使用 7 个 variant 共同有效的 seed，逐片验证 source index 与 IVD label 相同。前8片训练/后2片测试；VAE约4600步×3 seeds；physical aggregate 将三个 halfcylinder 和两个 deltaWing 分别先合并，排除 synthetic channel。Direct KMeans 最终用单线程确定性重算 | `experiments/Run_Pathline_Hyperparams_3D.py`, `experiments/Summarize_Pathline_Hyperparams_3D.py`, `docs/Verify_experiments.md` | `config/Verify_PathlineHyperparams3D_1.1.yaml` | physical-family macro：baseline `(dt=.25, steps=48, samples=32)` direct `.6076`、VAE `.5345±.0075`，是最高 VAE；dt=.125 VAE `.5126` 且构建1.75×；dt=.375 VAE `.5213`、构建.745×。steps32 direct `.6179`、VAE `.5277±.0221`、构建.689×；steps96 direct `.5746`、VAE `.4196±.0448`、构建1.827×、原生有效率77.53%。samples16 direct `.6174` 但 VAE `.5091±.0127`；samples48 direct `.6024`、VAE `.5258±.0044` | Task2 最佳已测试配置仍为 `.25/48/32`。短时间窗是便宜的近似备选，但逐流场差异大；24-frame 长窗明确有害。16点对 direct FMT 的小收益被 VAE 破坏，32点最适合当前 VAE。重构 MSE 与 aggregate F1 的描述性 Pearson correlation 仅 `.075`，再次说明 reconstruction loss 不是 feature quality 指标。F22 共同集合只覆盖27.07%，其结论限于长窗存活子集。 |
| Verify_HighReTask2_Controlled_1.1 | 2026-08-19 | Task2-3D 高 Reynolds 受控复核 | Re640/Re6400 固定 max-dim160、20³ seeds、`dt_scale=.125/steps=32/samples=32`、几何平均网格间距×.5 邻居半径；前6片训练/7–8片选参，最终前8片训练/9–10片留出。核心两臂使用完全相同的随机初始化 linear latent2 beta-VAE（β=1e-5、MSE+KL、exact 4600 steps、3 seeds）；Raw 为672维 center/relative geometry，FMT 为36维 neighbor real-spectrum norm | `experiments/Verify_HighReVAE.py`, `experiments/Confirm_HighReTask2_1.1.py`, `experiments/Summarize_HighReTask2_Controlled.py`, `FMT_Utils/DFT_FMT_3D.py`, `FMT_Utils/RawPathline_3D.py`, `docs/Verify_experiments.md` | `config/Verify_HighReSampling3D_1.5.yaml`, `config/Confirm_HighReVAE_Controlled_1.1.yaml`, `config/Confirm_HighReTask2_1.5.yaml` | held-out：Re640 Raw+PCA direct `.6154`、FMT direct `.7208`、Raw+VAE `.5032±.0156`、FMT+VAE `.6724±.0347`，paired gain `+.1692`；Re6400 `.6315/.6387/.5134±.0105/.6410±.0045`，gain `+.1276`。FMT gain 六个 paired seeds 全为正；FMT+VAE 所有 seed >.61 | 修订旧结论：原16³/max96/full161/nonlinear latent16 VAE 失败是真实的；提高高湍流空间采样、缩短窗口、选择 neighbor real-spectrum block、改用 linear latent2 后，Task2 在 Re640/Re6400 成立。结论是 FMT 输入提高同一 VAE，相反命题“VAE提高FMT”不成立：Re640 FMT direct 仍高于 FMT+VAE。Raw+VAE 均值略过.5但若干单 seed略低，不能宣称每次运行都过线。 |
| Verify_Task3FMTClassifier_1.1 | 2026-08-20 | Task3-3D 监督二分类 | local IVD 标签暂定 `IVD>0.9×mean_11³(IVD)`；Re640/Re6400 各10片，0–5 train/6–7 validation/8–9 test；shared temporal-convolution geometry backbone。三臂 Raw、参数更多的 Raw-wide、Raw+冻结161D FMT；train-only normalization/class weight；validation-only checkpoint与阈值；25 epoch×3 seeds | `FMT_Utils/LocalIVDLabel_3D.py`, `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Build_Task3_LocalIVD_Cache.py`, `experiments/Verify_Task3_FMTClassifier.py`, `experiments/Summarize_Task3_FMTClassifier.py`, `docs/Verify_experiments.md` | `config/Verify_Task3FMTClassifierLiteral_1.1.yaml` | held-out F1：Re640 Raw `.6752±.0126`、Raw-wide `.6704±.0105`、Raw+FMT `.7509±.0009`；Re6400 `.5848±.0046/.5834±.0056/.6974±.0021`。Average Precision：Re640 `.7437/.7482/.8517`；Re6400 `.5359/.5322/.7811` | 固定训练预算下，FMT 在两个 Reynolds number、全部6个 paired seeds 上提高监督分类，且不是参数量增加造成。但若干 Raw 最佳 epoch 触及25上限，因此该版用于多-seed稳定性，不作为最终收敛性能。 |
| Verify_Task3FMTConvergence_1.2 | 2026-08-20 | Task3-3D 收敛与标签敏感性审计 | 同 Task3 split/model，max epoch 60、patience10、seed20；同时比较字面标签与更稀疏的 `IVD>local percentile90_11³(IVD)`。物理比例空间错误图只显示 TP/FP/FN | `experiments/Verify_Task3_FMTClassifier.py`, `experiments/Visualize_Task3_FMTClassifier.py`, `docs/Verify_experiments.md` | `config/Verify_Task3FMTConvergenceLiteral_1.2.yaml`, `config/Verify_Task3FMTConvergenceP90_1.2.yaml` | 字面标签长预算 F1：Re640 Raw/Raw-wide/Raw+FMT `.7003/.6913/.7556`，Re6400 `.5989/.5977/.6940`；Average Precision `.7791/.7669/.8538` 与 `.6231/.6056/.7863`。稀疏标签 F1：Re640 `.4930/.4457/.5680`，Re6400 `.2800/.2786/.4911` | 修订1.1：短预算低估 Raw，字面标签 FMT−Raw F1 应由单-seed `+.0698/+.1097` 修正为长预算 `+.0553/+.0951`；结论方向不变。两种标签解释均支持 FMT，但收敛审计与稀疏标签目前仅单 seed；`n=11` 仍需用户冻结且应进一步改成物理半径。 |
| mainExp_Task3Universality_1.1 | 2026-08-20 | Task3-3D 跨流场主实验 | 统一 local-IVD 字面标签 `IVD>.9×mean_11³(IVD)`；8 数据条目、5 独立 family、10 slices 的 6/2/2 temporal split、3 paired seeds。先发现 joint Raw+FMT 在 F-22 失败，再改为冻结 Raw 主干的 additive FMT residual；validation 在含 alpha=0 的固定网格选 residual 权重。Raw+FMT residual 总参数125,506，低于 Raw-wide 148,225 | `experiments/Build_Task3_Universality_Labels.py`, `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Verify_Task3_FMTResidual.py`, `experiments/Summarize_Task3_Universality.py`, `docs/mainExp_Task3Universality_1.1.md` | `config/Verify_Task3Universality_1.1.yaml`, `config/Confirm_Task3UniversalityBaselines_1.1.yaml`, `config/Confirm_Task3UniversalityResidual_1.1.yaml` | 24/24 dataset-seed 均满足 residual 的 F1/AP 同时高于 Raw/Raw-wide。mean F1/AP：Re160 `.610/.592`、Re640 `.716/.792`、Re6400 `.644/.695`、Tangaroa `.913/.967`、delta-resampled `.710/.717`、delta-LBM `.709/.725`、F22 `.635/.652`、channel `.700/.740`。F22 最小配对 F1 增益对 Raw-wide 仅 `+.00159`；channel AP 对 Raw 平均 `+.231` | 在当前 5 类已测试 3D flow 与 IVD proxy label 下，FMT 对监督分类的增益方向一致，且不能由参数量解释。修订旧 joint 结论：旧模型仅7/8通过，F22失败；FMT residual 保留 Raw 路径后使F22三个seed均为正增益。不能表述成任意3D flow的数学普适性，也不能把8条目算成8个独立family。 |
| mainExp_Task3Universality_2.1（失败） | 2026-08-21 | Task3-3D 明显增益确认 | 在 161D FMT 上增加严格逐帧刚体不变的 63D time-local Gram DFT 与 44D pathline kinematic DFT，共268D；冻结 Raw 主干，残差总参数132,354 < Raw-wide 148,225。先按 validation F1 选 epoch，再选 Pareto alpha；用第三套新 seed times 确认 | `FMT_Utils/DFT_FMT_3D.py`, `experiments/Verify_Task3_FMTResidual.py`, `experiments/Evaluate_Task3_FrozenConfirmation.py` | `config/mainExp_Task3Universality_2.1_train.yaml`, `config/mainExp_Task3Universality_2.1_evaluate.yaml` | stronger Raw 三-seed method mean 判据要求 F1/AP 均 `>=+.02`。7/8 通过；Tangaroa F1 `+.02720`，AP 仅 `+.01620`，明确失败 | 不降低阈值，不接受7/8。失败原因是 F1-first checkpoint selection 对排序质量约束不足；该 confirmation 只用于否定2.1，未用于训练2.2权重。 |
| mainExp_Task3Universality_2.2 | 2026-08-21 | Task3-3D 跨流场最终确认 | 模型、268D FMT、loss、训练数据和容量全部沿用2.1；唯一方法变化是统一 constrained-AP validation selection：先要求 validation F1 相对 stronger Raw `>=+.02`，再最大化AP；无可行候选时最大化两指标最小增益。规则在旧train/validation固定后，一次性评估第四套新 seed times | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Evaluate_Task3_FrozenConfirmation.py`, `experiments/Plot_Task3_Universality.py`, `docs/mainExp_Task3Universality_2.2.md` | `config/mainExp_Task3Universality_2.2_{train_seed20,train_seeds21_22,cache,labels,evaluate}.yaml`, `config/Verify_Task3ConstrainedAPSelection_1.{1,2}.yaml` | 8/8 数据条目在 stronger Raw 三-seed method mean 上同时满足 F1/AP `>=+.02`。F1/AP gain：Re160 `+.0744/+.1046`、Re640 `+.0721/+.0919`、Re6400 `+.0998/+.1466`、Tangaroa `+.0297/+.0208`、delta-resampled `+.0463/+.1054`、delta-LBM `+.0450/+.1033`、F22 `+.0257/+.0462`、channel `+.1960/+.4050` | 在当前统一协议的5个独立flow family、8个数据条目上，FMT对监督式Raw分类器的明显增益具有一致性；不能外推为任意3D流场的数学普适性。Tangaroa AP只比阈值高`.00078`；逐seed oracle下F22 F1增益仅`.01632`，两者是优先复核边界。 |
| mainExp_Task1_3D_2.1 | 2026-08-23 | Task1-3D 跨流场无监督聚类 | 10片开发数据作6片拟合/2片表示选择/2片匿名簇校准；另4片新时间最终确认。每物理族从 cached161、Gram、kinematic及组合中按validation选FMT block/PCA；KMeans仅用无标签train feature拟合，最终cluster-ID映射冻结；5 KMeans seeds | `experiments/Run_Task1_3D_Main.py`, `FMT_Utils/Task12{Data,Evaluation}_3D.py`, `docs/mainExp_Task1_3D_2.1.md` | `config/mainExp_Task1_3D_2.1.yaml` | FMT F1：channel `.261`、Re160 `.596`、Re640 `.569`、Re6400 `.533`、Tang `.745`、delta-resampled `.745`、delta-LBM `.766`、F22 `.314`；均值`.566`。FMT−独立调PCA的Raw在7/8为正，F22 `−.343` | Task1只要求FMT+KMeans达到可量化的无监督涡/非涡聚类性能；该要求得到跨5个family的证据。不能声称所有场高准确率或都优于Raw；F22是明确比较反例，channel绝对F1也偏低。 |
| mainExp_Task2_3D_2.2（附录压力测试） | 2026-08-23 | Task2-3D strongest-Raw 比较 | Raw+VAE与FMT+VAE分别在validation独立选择linear/MLP VAE；6片训练/2片选架构/2片校准匿名簇/4片新时间确认；3 paired seeds。该协议允许Raw按自身validation单独变强，现不作为“只改变输入”的Task2主表 | `experiments/Run_Task2_3D_Main.py`, `FMT_Utils/VAE_3D.py`, `docs/mainExp_Task2_3D_2.2.md` | `config/mainExp_Task2_3D_2.2.yaml` | paired F1 gain：channel `+.168`、Re160 `−.042`、Re640 `+.198`、Re6400 `+.126`、Tang `−.040`、delta-resampled `+.437`、delta-LBM `+.338`、F22 `−.238`；5/8正，条目均值`+.119`；按family为3/5正 | **结论修订：**该结果真实保留，但它回答“各自调优后谁更强”，不是预定same-VAE主命题。此前把它称为“更严格主实验”是我擅自改变比较；2.3取代它成为主表，2.2仅作为strongest-Raw压力测试。 |
| mainExp_Task2_3D_2.3 | 2026-08-23–24 | Task2-3D same-VAE 跨流场主实验 | 每个physical-family冻结FMT validation winner对应的一个VAE；Raw/FMT两臂使用相同hidden、latent、β、学习率、4600 optimizer steps和3 paired seeds，只改变输入表示及其必需的输入/输出层宽度。8片训练/2片校准匿名簇/4片新时间确认；family-specific FMT block允许不同，IVD-p95标签不变 | `experiments/Run_Task2_3D_Main.py`, `FMT_Utils/VAE_3D.py`, `docs/mainExp_Task2_3D_2.3.md` | `config/mainExp_Task2_3D_2.3.yaml` | **Ibex A100主表：**Raw→FMT F1：channel `.053→.222` (`+.169`)；Re160 `.555→.554` (`−.001`)；Re640 `.473→.671` (`+.198`)；Re6400 `.564→.690` (`+.126`)；Tang `.704→.729` (`+.025`)；delta-resampled `.313→.821` (`+.507`)；delta-LBM `.303→.831` (`+.528`)；F22 `.630→.477` (`−.152`)。6/8条目、4/5 family正；条目平均增益`+.1750` | 支持“在当前大多数3D flow中，FMT是比Raw pathline更有效的同一VAE输入”；不能声称全部普适。F22为稳定反例，Re160近似持平。修订本地RTX3090的7/8：Re160由`+.0140`变为A100的`−.0013`，原因是VAE训练未启用CUDA确定性算法且效应接近零；两套结果并列保留。2.2仍为strongest-Raw附录。 |
| mainExp_Task1_3D_2.2_newflows | 2026-08-24 | Task1-3D 新 family confirmation | 新增 Boeing747 与 SmokeBuoyancy；沿用2.1的6片拟合/2片表示选择/2片匿名簇校准和4片独立confirmation；全域IVD-p95标签、5个KMeans seeds；两个family可在开发片独立选择FMT block/PCA | `experiments/Run_Task1_3D_Main.py`, `docs/mainExp_Task1_3D_2.2_newflows.md` | `config/mainExp_Task1_3D_2.2_newflows.yaml` | **Ibex A100：**Boeing FMT `.8408±.0002` vs Raw `.4705`，gain `+.3703`；Smoke FMT `.7608` vs Raw `.5454`，gain `+.2154` | 两个新增family均强支持FMT+KMeans。合并2.1后覆盖10条目、7 family；9/10条目高于Raw，FMT条目平均F1 `.6130`。F22仍为唯一比较反例。 |
| mainExp_Task2_3D_2.4_newflows | 2026-08-24 | Task2-3D 新 family same-VAE confirmation | 新增 Boeing747 与 SmokeBuoyancy；每family以FMT开发validation winner冻结VAE，再将完全相同hidden/latent/β/学习率/4600步/3 seeds用于Raw与FMT；4片独立confirmation不参与选择 | `experiments/Run_Task2_3D_Main.py`, `docs/mainExp_Task2_3D_2.4_newflows.md` | `config/mainExp_Task2_3D_2.4_newflows.yaml` | **Ibex A100：**Boeing Raw `.6194±.0425`、FMT `.8474±.0120`、gain `+.2280±.0307`；Smoke Raw `.8090±.0029`、FMT `.7437±.0096`、gain `−.0653±.0100`。合并2.3后7/10条目、5/7 family为正，条目平均`+.1562`、family-macro`+.1185` | Boeing强支持FMT改善同一VAE输入；Smoke稳定反对“所有flow都提高”。负结果保留，未为Raw单独搜索更强VAE。 |
| mainExp_Task3NewFlows_2.3 | 2026-08-24 | Task3-3D 新 family confirmation | 新增 Boeing747 与 SmokeBuoyancy；沿用 `.9×mean_11³(IVD)` 标签、Raw/Raw-wide基线、冻结Raw后的268D FMT residual和constrained-AP validation选择；6片train/2片validation，另4片新时间只作confirmation；3 seeds | `experiments/Verify_Task3_FMTClassifier.py`, `experiments/Verify_Task3_FMTResidual.py`, `experiments/Evaluate_Task3_FrozenConfirmation.py`, `docs/mainExp_Task3NewFlows_2.3.md` | `config/mainExp_Task3NewFlows_2.3_{baselines,residual,evaluate}.yaml` | **Ibex A100：**Boeing stronger Raw→Raw+FMT F1 `.6551→.7203` (`+.0651`)，AP `.6379→.7996` (`+.1617`)；Smoke F1 `.6406→.6706` (`+.0299`)，AP `.6833→.7267` (`+.0433`)。两项均通过F1/AP最小增益`.02` | 两个新增family支持FMT提高监督IVD二分类。合并2.2后10/10条目F1/AP正增益、7/7 family均值为正，family-macro F1/AP增益`+.0677/+.1280`。本地RTX3090数值稍高但方向及通过判据一致；论文使用A100数字。 |
| mainExp_Task3_3D_3.1 | 2026-08-24 | Task3-3D 论文主表去偏确认 | 去除显式要求FMT增益的constrained-AP选择；所有方法仅按自身validation AP选epoch，阈值仅按自身validation F1冻结，residual alpha全局固定1.0。增加train-only 268D Raw-PCA residual作为完全同训练结构、同可训练参数量的Raw-only对照；7个physical family的Raw方法只由development validation冻结，最终均选Raw-PCA。10条目×8个新起始时间×5训练seeds | `experiments/Build_Task3_Main_Confirmation.py`, `experiments/Verify_Task3_FMTClassifier.py`, `experiments/Verify_Task3_FMTResidual.py`, `experiments/Evaluate_Task3_MainTable.py`, `docs/mainExp_Task3_3D.md` | `config/mainExp_Task3_3D_3.1_*.yaml` | **Ibex V100完成。**相对原始Raw，Raw+FMT在10/10条目的F1/AP均提高，条目平均增益`+.0701/+.1176`；相对同构同参数量Raw-PCA residual，仅4/10条目、3/7 family的F1/AP为正，条目平均`-.0009/+.0117`，family-macro`+.0024/+.0129`。完整5-seed标准差、paired bootstrap 95% CI和逐片表见`outputs/mainExp_Task3_3D_3.1_ibex_v100/final_confirmation/` | 修订2.2+2.3：旧10/10结果受“validation选择显式要求FMT增益”及缺少同构Raw-only residual强对照限制。3.1支持“FMT改善原始Raw监督网络”，但反对“FMT在所有当前3D场上都优于强Raw-only特征扩展”。本版本无条件取代旧版作为论文主表。 |
| mainExp_Task3_3D_3.2_global_ivd | 2026-08-26 | Task3-3D whole-field IVD p95统一标签确认 | 模型、268D FMT、train-only 268D Raw-PCA residual、训练预算和6/2 development split均沿用3.1；唯一科学变量是把Task3标签改为与Task1/2逐位相同的`IVD(seed)>=percentile95(IVD volume)`。标签直接复制source cache的冻结`reference`；5个新训练seeds 40–44；每条目8个不参与本轮训练/选择的起始时间作confirmation。checkpoint只按自身development AP选择，threshold只用development，alpha固定1.0 | `experiments/Build_Task3_GlobalIVD_Labels.py`, `experiments/Verify_Task3_FMTClassifier.py`, `experiments/Verify_Task3_FMTResidual.py`, `experiments/Evaluate_Task3_MainTable.py`, `docs/mainExp_Task3_3D.md` | `config/mainExp_Task3_3D_3.2_global_ivd_*.yaml`；commits `dc3c844`,`d2cf6b2` | **Ibex V100：**dataset macro Raw/Raw-PCA/FMT F1 `.6887/.7818/.8320`，FMT−Raw-PCA `+.0502`；AP `.7444/.8575/.9020`，gain `+.0445`。family macro F1/AP gain `+.0548/+.0490`。相对Raw-PCA，F1 9/10条目、6/7 family正；AP 8/10、6/7正。F-22稳定为负（F1/AP `−.0269/−.0129`）；Re640 F1 `+.0028`区间跨0且AP `−.0049` | **修订3.1的标签依赖结论：**换成Task1/2的whole-field IVD p95后，FMT从“未超过Raw-PCA”变为macro-average明确超过，并在多数条目/family为正；支持广泛提升，不支持逐flow严格普适，F-22仍是明确反例。confirmation未参与本轮训练/选择，但这些起始时间此前用于其他任务，故不称全项目从未查看的sealed set。证据：`outputs/mainExp_Task3_3D_3.2_global_ivd/final_confirmation/`，归档SHA-256 `583ec77c…4c040`。 |
| mainExp_Task2_3D_3.2（失败） | 2026-08-25 | Task2-3D VAE大网格后独立确认 | development-only搜索36套VAE×7 physical family×Raw/FMT两臂×2 seeds，共1440次训练；每family两臂冻结同一VAE。开发全局精确Pareto组合先满足`FMT+VAE > Raw+VAE > Task1`，再用5个新训练seeds在既有4个confirmation时间片确认 | `experiments/Sweep_Task2_VAE_3D.py`, `experiments/Build_Task2_VAE_Confirmation.py`, `experiments/Run_Task2_3D_Main.py` | `config/Verify_Task2_VAEGrid_3D_3.1.yaml`, `config/mainExp_Task2_3D_3.2.yaml` | development：Task1 `.60090`、Raw `.61580`、FMT `.63007`。独立confirmation：Task1 `.61305`、Raw `.63970`、FMT `.63518`；Raw−Task1 `+.02665`，FMT−Raw `−.00452`。逐条目主要负值：F22 `−.13817`、Re6400 `−.09765`、Smoke `−.04682` | **完整目标失败，保留负结果。**只支持`Raw+VAE > Task1`；不支持`FMT+VAE > Raw+VAE`。开发选择只用2 seeds，Re6400 confirmation FMT std `.1493`，说明选择不稳。下一版必须增加development seeds并使用seed-robust选择；3.2 confirmation不得参与3.3参数选择。 |
| mainExp_Task2_3D_3.3 | 2026-08-25–26 | Task2-3D 稳健VAE选择与全新时间片确认 | 36套共享VAE×7 physical family×Raw/FMT两臂；在development复用2 seeds并新增5 seeds，共7 seeds。用混合整数优化最大化7个seed中最差的两级全局平均间隔；每family两臂冻结同一VAE。3.2 confirmation完全禁止参与3.3选择。最终在Task3 3.1缓存中的8个全新时间片/数据集、5个新训练seeds 9068–9072上确认；Task1也在相同8片重算 | `experiments/Sweep_Task2_VAE_3D.py`, `experiments/Build_Task2_VAE_Confirmation.py`, `Run_Task{1,2}_3D_Main.py`, `experiments/Summarize_Task2_3D_Hierarchy.py` | `config/Verify_Task2_VAEGrid_3D_3.2.yaml`, `config/mainExp_Task2_3D_3.3.yaml` | development 7/7 seeds均满足层级：Task1 `.600897` < Raw+VAE `.613055` < FMT+VAE `.628046`，最差seed最小间隔`+.010635`。独立confirmation严格平均层级通过：Task1 `.606360` < Raw+VAE `.631958` < FMT+VAE `.634191`；两级增益`+.025598`与`+.002233`。FMT−Raw逐seed 3/5正、逐条目5/10正；主要负项F22 `−.11614`、Smoke `−.05320`、Re160 `−.03317` | **用户指定的“平均F1性能”目标得到独立确认。**3.3取代3.2作为Task2当前平均主结果；但FMT相对Raw的平均优势很小，且不具有逐seed/逐场普适性，论文只能陈述10条目macro-average层级，必须同时展示逐条目结果与反例。证据：`outputs/mainExp_Task2_3D_3.3/{paper_table.csv,summary.json,hierarchy.json}`。 |
| mainExp_Task5_3D_1.1 | 2026-08-26 | Task5-3D 不同尺度 IVD 监督二分类 | 在 Task3-3.2 的 whole-field IVD p95 标签、网络、268D FMT/Raw-PCA、预算和选择规则上，只改变 primitive 尺度：中心邻居距离、RK4 步长和积分步数可变，积分后仍固定为 `7×32×3`。train/validation/confirmation 分别使用 18/6/9 个互斥尺度 tuple；每片按与位置和标签独立的固定随机排列均衡分配尺度。10条目各4 train、2 validation、4晚期confirmation时间片；5个训练seeds 40–44 | `FMT_Utils/MultiscalePathline_3D.py`, `experiments/Build_Task5_Multiscale_Cache.py`, `experiments/Verify_Task3_FMTClassifier.py`, `experiments/Verify_Task3_FMTResidual.py`, `experiments/Evaluate_Task5_Multiscale.py`, `docs/mainExp_Task5_3D_1.1.md` | `config/mainExp_Task5_3D_1.1*.yaml`；implementation commits `a01d057`,`f1b05c0`,`3307389` | **Ibex V100：**相对同维度 Raw-PCA，dataset-macro F1 `.5835→.6727`（`+.0891`），AP `.6063→.7180`（`+.1116`）；F1 10/10、AP 9/10条目为正。相对 development 冻结的 strongest Raw，F1/AP gain `+.0849/+.1026`，均为9/10条目、7/7 family正。9/9未见尺度 tuple 的平均F1/AP增益均正。Task5 FMT相对fixed-scale Task3 FMT transfer的dataset-macro F1提高`+.1515`，8/10条目正 | **支持 Task5 核心命题：**在所测不同尺度范围和多数/宏平均3D场景中，FMT提高可变尺度IVD监督识别；同维度F1改善覆盖10/10条目，包含F-22。限制：Re640的AP为`−.0063`；相对strongest Raw，Re160略负；相对fixed-scale FMT transfer，Re160/Smoke为负。最宽空间尺度与最长积分跨度组合增益最小，不支持无限尺度外推。证据归档SHA-256 `6053ed15e66b8f7438133fa92e7086bcd9b139e42a6a5cdfc6f7e682d86c58ec`。 |
| Verify_Task5_CylinderHyperparams_1.1 | 2026-08-26–27 | Task5 Re160/Re640 development-only超参数搜索与冻结outer | 30个FMT配方/融合/训练候选；每个候选配同维度Raw-PCA residual；train ordinal0–2、checkpoint与候选选择ordinal3、冻结后outer ordinal4–5；seeds60–61。physical kinematic 使用真实重采样时间，另测signed-log与pseudoinverse cutoff。旧Task5 confirmation未读取 | `experiments/Search_Task5_CylinderHyperparams.py`, `FMT_Utils/Task5FeatureRecipes_3D.py`, `docs/Verify_experiments.md` | `config/Verify_Task5_CylinderHyperparams_1.1.yaml`；commits `75e07f8`,`aff6ce3` | development冻结`c24_physical_log`，Re160/Re640最小gain `+.09202/+.10289`。outer：Re160 FMT−Raw-PCA F1/AP `+.01416/+.03629`，FMT−strong Raw `+.02468/−.03520`；Re640分别`+.09838/+.12161`与`+.15269/+.20305` | **预注册整体目标失败，负结果保留。**Re640获得明显增益；Re160未满足四项`>=+.03`且均超过strong Raw。ordinal4–5已经暴露，后续只能作development诊断。第一次outer因checkpoint路径bug失败且无指标；修复后同一冻结候选成功，选择未改变。证据：selection SHA-256 `ad7e26fc…c598`，outer audit `c5807a35…8b5e`。 |
| Verify_Task5_Re160FreshTimes_1.1 | 2026-08-27 | Task5 Re160 family-specific适应性选择与全新时间/尺度检验 | 在fresh cache不存在时，仅用已暴露development ordinal3–5重排原30候选，按12项比较的最差增益冻结44D `physical_kinematic`；旧ordinal0–4训练、5验证，5个新seeds70–74。之后首次生成starts85/96；两个10帧source window与旧development/confirmation均不重叠，6个尺度tuple均为全新；标签仍为whole-field IVD p95。Raw-PCA严格匹配44D及同residual结构 | `experiments/Verify_Task5_Re160FreshTimes.py`, `FMT_Utils/Task5FeatureRecipes_3D.py`, `docs/Verify_experiments.md` | `config/Verify_Task5_Re160FreshTimes_1.1*.yaml`；commits `5ef5a7c`,`f417057`,`a081b44`；本地RTX3090 24GB | fresh FMT/Raw-PCA/strong Raw F1 `.7400/.3984/.4028`，AP `.8906/.3222/.3592`；FMT−Raw-PCA `+.34157/+.56845`，FMT−strong Raw `+.33723/+.53148`。5/5 seed、6/6新尺度的matched F1/AP增益均正；paired 95% CI F1 `[+.30677,+.37637]`、AP `[+.52521,+.61169]` | **Re160明显增益得到全新时间与尺度确认。**与冻结Re640 outer的`+.09838/+.12161`合并后，两种低/中Re half-cylinder均有明确FMT增益，但使用不同family-specific配方，不能声称统一超参数普适。旧Re160失败结果不删除；starts85/96不得再用于后续选参。fresh audit SHA-256 `6eb8ace2…abff`。 |
| Verify_Task3_F22Hyperparams_1.1 | 2026-08-27 | Task3 F-22 family-specific超参数搜索与fresh confirmation | 14个FMT block Stage1后冻结`kin2/kin6/kin4`；Stage2搜索3 blocks×10 residual/width/lr/fusion组合，全部与同宽同结构Raw-PCA配对，seeds40–42；0–5 train、6–7 validation、冻结后8–9 outer。selector前预注册8个新索引`[34,47,60,74,89,104,117,141]`；标签保持whole-field IVD p95 | `experiments/Prepare_Task3_F22_FocusedSearch.py`, `experiments/Search_Task3_FMTResidual_Stage2_3D.py`, `experiments/Evaluate_Task3_F22_FocusedConfirmation.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_FMTResidualFamilySearch_4.1.yaml`, `config/Confirm_Task3_F22Hyperparams_1.1_{cache,labels,evaluate}.yaml`；commits `b967b7b`,`68e847c`,`e376ec1` | 冻结`kin6 + geometry_fmt residual + aux64 + lr3e-4`。development FMT−Raw-PCA F1/AP `+.02077/+.02874`；outer `+.02314/+.04131`。fresh 8片：FMT/Raw-PCA/strong Raw F1 `.89371/.89368/.87628`，AP `.94425/.94507/.93077`；FMT−strong Raw `+.01744/+.01349`且3/3 seeds两项均正；FMT−Raw-PCA `+.00003/−.00082` | **F-22上的标准Task3命题得到fresh确认：FMT网络稳定优于Raw/Raw-wide no-FMT网络。**修订3.2中F-22反例：family-specific 44D kinematic block消除了对strong Raw的不明显提升。但相对额外的同构Raw-PCA压力测试仅持平，不能声称FMT明显优于所有Raw-only特征扩展。selection SHA-256 `46aa4f46…cb4f`，fresh audit SHA-256 `1dc5ead9…`。 |
| Verify_Task3_F22AnchoredFeatures_1.2 | 2026-08-27 | Task3 F-22 seed-time anchored FMT与全新时间确认 | 由7-line pathline cross估计速度梯度和逐时刻涡量偏差；19个无参数anchored Fourier候选先用固定Logistic Regression按`min(F1 gain, AP gain)`筛前三，再以同结构同宽Raw-PCA residual、seeds50–54按最差配对增益冻结10D `aivd2w8`。12片development使用8/4 train/validation；选择前冻结且未打开8个新索引`[36,55,76,91,106,120,129,140]`；whole-field IVD p95标签不变 | `FMT_Utils/DFT_FMT_3D.py`, `experiments/Screen_Task3_F22AnchoredFeatures.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Select_Task3_F22AnchoredFeatures.py`, `experiments/Evaluate_Task3_F22_AnchoredConfirmation.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_F22AnchoredFeatures_1.2_*.yaml`, `config/Confirm_Task3_F22AnchoredFeatures_1.2_*.yaml`；本地RTX3090；实现commits `07ee64b`,`7568ac6` | development FMT/Raw-PCA F1 `.95370/.94270`、AP `.98776/.97918`，gain `+.01100/+.00858`，五seed最差双指标gain `+.00667`。Fresh 8片 FMT/Raw-PCA F1 `.95458/.94569`、AP `.98954/.98179`，gain `+.00889/+.00775`；5/5 seeds的F1/AP均为正。相对Raw与Raw-wide也均明显更高 | **修订1.1的“相对Raw-PCA仅持平”：针对seed-time IVD保留首值和早期窗口的anchored FMT在另一组fresh时间上稳定优于同宽Raw-PCA。**提升来自表示设计；Raw-PCA训练预算、结构和随机种子未削弱。该配方是F-22 family-specific extension，不静默覆盖3.2冻结主表。selection SHA-256 `5a0ed082…2daa9`，fresh audit SHA-256 `8472bbcd…7e2f`。 |
| Verify_Task2_FMTVAEFamilySearch_4.1 | 2026-08-27 | Task2-3D family-specific开发搜索 | 7个physical family分别搜索14种无标签FMT feature block与VAE；Stage1用4个代表容量筛每family前3个feature，Stage2交叉12种线性/多层感知机VAE。Raw与FMT在每个候选内严格共用hidden、latent、KL权重、学习率、optimizer steps及paired seed；0–5训练、6–7选择，冻结selection后才打开8–9 outer；Task1 direct只报告、不参与排名 | `experiments/Search_Task2_FMTVAE_3D.py`, `experiments/Search_Task2_FMTVAE_Stage2_3D.py` | `config/Verify_Task2_FMTVAEFamilySearch_4.1.yaml`；implementation commits `f8f5675`,`7bdb511` | development dataset-macro F1 gain `+.21956`，9/10条目正；冻结outer为`+.21557`，仍9/10正；F22分别`−.04654/−.01934`。selection SHA-256 `439d9e5d…97795`，outer audit SHA-256 `4cd41145…12776` | family-specific配置在未参与选择的outer ordinals上保持明显宏平均增益，因此进入一次性新空间population确认。F22负值从development到outer均保留，未为得到全正结果而改数据或单独削弱Raw。 |
| mainExp_Task2_3D_4.1 | 2026-08-27 | Task2-3D 新空间primitive population确认 | 冻结4.1 selection后，以空间seed-grid phase`[.31,−.23,.17]`重建10条目×4切片的新primitive population；原development 0–7重训同一VAE两臂，8–9仅校准KMeans簇编号；5个新VAE seeds 9080–9084。时间切片曾用于历史实验，故不是全项目sealed temporal test | `experiments/Build_Task23_FamilySearch_Confirmation.py`, `experiments/Run_Task2_FMTVAE_Frozen_4_1.py`, `docs/mainExp_Task23_3D_4.1.md` | `config/mainExp_Task2_3D_4.1.yaml`；Ibex jobs `50929941`,`50929943` | Raw+VAE/FMT+VAE dataset-macro F1 `.46165/.63077`，配对增益`+.16912`，达到预注册`.15`；9/10条目、6/7 family正，family-macro gain`+.16126`。F22`−.07123`；其余从Re160最低`+.03674`到delta-resampled最高`+.39761`。100/100唯一记录；summary/per-run SHA-256 `06c7de3e…be8122`/`c06992f5…c0c78` | **Task2主命题得到明显宏平均支持：在同一VAE内把Raw输入换为FMT，确认集F1提高`.169`。**相对3.3的`+.00223`，提升来自只读development进行的family-specific FMT block/VAE搜索；不是confirmation后调参。不能声称逐flow普适，F22仍是稳定反例。 |
| Verify_Task3_FMTResidualFamilySearch_4.1 | 2026-08-27 | Task3-3D family-specific开发搜索 | Stage1在固定residual架构下比较14种FMT block；Stage2将每family前3个block与10种residual input route、宽度、学习率及alpha选择组合交叉。每个FMT候选都与同宽、同结构、同训练过程的train-only Raw-PCA residual配对；0–5训练、6–7选择，冻结selection后打开8–9 outer；Raw/Raw-wide只作额外诊断 | `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Search_Task3_FMTResidual_Stage2_3D.py` | `config/Verify_Task3_FMTResidualFamilySearch_4.1.yaml`；implementation commits `f8f5675`,`7bdb511` | development dataset-macro F1/AP gain `+.11963/+.12805`；outer `+.14027/+.14623`；两阶段均10/10条目为正，但F1均未达到`.15`。selection SHA-256 `9eb04414…122c`，outer audit SHA-256 `7f54e05d…b649` | 搜索稳定提高了相对Raw-PCA的监督识别，但预注册`.15`开发目标没有达到，未降低阈值。冻结配方仍进入新空间population确认，以检验方向和效应大小。 |
| mainExp_Task3_3D_4.1 | 2026-08-27 | Task3-3D 新空间primitive population确认 | 使用与Task2-4.1逐位相同的新空间primitive population和whole-field IVD p95标签；冻结family-specific FMT residual配方，5个paired seeds40–44；主对照为同宽同结构Raw-PCA residual，另报告Raw与Raw-wide。训练、checkpoint和阈值只读development；confirmation只评估 | `experiments/Build_Task23_FamilySearch_Confirmation.py`, `experiments/Run_Task3_FMTResidual_Frozen_4_1.py`, `docs/mainExp_Task23_3D_4.1.md` | `config/mainExp_Task3_3D_4.1.yaml`；Ibex jobs `50929944`,`50929946` | Raw-PCA/FMT dataset-macro F1 `.71406/.81406`，gain`+.10000`；AP `.78341/.90237`，gain`+.11896`。F1/AP均10/10条目、7/7 family正；family-macro gain`+.11147/+.13232`。相对Raw/Raw-wide诊断，dataset-macro gain`+.15828/+.19479`。200/200唯一记录；summary/per-run SHA-256 `44d37070…6c25c`/`8ebfc2fd…a74be` | **Task3核心方向得到所有当前条目的支持，但预注册的Raw-PCA对照F1增益`.15`目标失败。**相对3.2的F1/AP `+.0502/+.0445`，family-specific搜索将效应提高到`+.1000/+.1190`并消除F22负例；不能把相对较弱Raw的`+.1583`冒充为Raw-PCA目标达标。 |
| Ablation_Task23IVDPercentile_1.1 | 2026-08-27 | Task2/Task3-3D whole-field IVD百分位敏感性（旧主表复现） | 完整扫描p80/p85/p87.5/p90/p92.5，并以逐位复现的p95作参考。Task2复现当时的3.3 same-VAE配方，每个dataset/method/seed只训练一次、跨百分位只改变匿名簇映射和评分标签；Task3复现3.2网络并在每个百分位重新训练Raw、Raw-wide、同宽Raw-PCA residual和FMT residual。180/180源切片p95阈值误差0、标签mismatch 0 | `experiments/Build_Task23_IVDPercentile_Labels.py`, `experiments/Run_Task2_IVDPercentile_Sweep.py`, `experiments/Prepare_Task3_IVDPercentile_Configs.py`, `experiments/Summarize_Task23_IVDPercentile.py` | `config/Ablation_Task23IVDPercentile_1.1.yaml`；Ibex jobs `50929637`–`50929643`；commit `fc89c97` | p80/p85/p87.5/p90/p92.5的Task2 FMT−Raw F1分别`−.1006/−.0990/−.0877/−.0694/−.0401`，p95参考`+.0038`。Task3 FMT−Raw-PCA F1分别`+.0160/+.0322/+.0400/+.0454/+.0520`，AP分别`+.0200/+.0273/+.0344/+.0391/+.0418`；p95参考F1/AP`+.0502/+.0445`。Task3绝对FMT F1/AP在p87.5最高，为`.8574/.9370` | **旧3.3 Task2的冻结latent簇只在窄p95核心附近与FMT有微弱优势；降低阈值不会自动扩大FMT增益。旧3.2 Task3在全部标签下macro增益为正，请求范围内p92.5的相对F1/AP增益最大。**该行只描述旧主表敏感性；4.1已在并行完成并成为当前主表，因此由`Ablation_Task23IVDPercentile_1.2`另行复现，不能将本行数值冒充当前结论。结果归档SHA-256 `a869eb06…f9e9`。 |
| Ablation_Task23IVDPercentile_1.2 | 2026-08-27 | 当前Task2/Task3-4.1 whole-field IVD百分位敏感性 | 在4.1新空间population上完整扫描p80/p85/p87.5/p90/p92.5，并保留p95参考。Task2固定同一family VAE，训练预测跨标签复用；Task3每个请求百分位重新训练Raw、Raw-wide、同宽Raw-PCA residual和FMT residual。40/40 source slices的p95 mask与原4.1缓存逐位一致；Task3共500/500 baseline模型与1000/1000 final结果行 | `experiments/Run_Task2_IVDPercentile_Frozen_4_1.py`, `experiments/Prepare_Task3_IVDPercentile_Frozen_4_1.py`, `experiments/Run_Task3_FMTResidual_Frozen_4_1.py`, `experiments/Summarize_Task23_IVDPercentile_Frozen_4_1.py`, `docs/Ablation_Task23IVDPercentile_1.2.md` | `config/Ablation_Task23IVDPercentile_1.2.yaml`；implementation commit `7cba571`；Ibex jobs `50934345`–`50935581`（含登记的调度拆分）；结果SHA-256 `abd5bb5c…224a1c9` | p80/p85/p87.5/p90/p92.5的Task2 FMT−Raw F1为`−.0590/+.0169/+.0510/+.0901/+.1337`；Task3 FMT−Raw-PCA F1为`+.0109/+.0455/+.0619/+.0675/+.0783`，AP为`+.0282/+.0470/+.0626/+.0720/+.0937`。p95参考分别为`+.1687/+.1000/+.1190`。p92.5的Task3 F1/AP均10/10数据条目正，Task2为7/10正 | **p92.5是请求的较低阈值中Task2 F1、Task3 F1/AP增益都最大的候选，但纳入原p95后，p95仍给出最大的相对增益。**p92.5将seed正标签比例从`.066`扩大到`.094`，同时Task3 FMT绝对F1/AP保持为`.8151/.9101`，是视觉边界与性能的合理折中候选；不能仅凭模型增益判定物理上最正确的阈值。若据此更换主标签，需在fresh population重新确认。 |
| Verify_Task3_AnchoredRobust_5.1 | 2026-08-27 | Task3-3D anchored FMT与同宽Raw-PCA的family-specific开发搜索 | 在4.1已公开空间population上增加pathline-only anchored vorticity-deviation Fourier blocks；Stage1为18 features×10 datasets×2 seeds，Stage2为每family前三feature×10 residual设置×3 seeds。每个FMT候选均配train-only、同宽、同结构、同训练过程Raw-PCA residual；旧ordinals与4.1相位全部只作development，5.1 final phase在selection冻结前未生成 | `FMT_Utils/DFT_FMT_3D.py`, `experiments/Screen_Task3_AnchoredRobust_3D.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Search_Task3_FMTResidual_Stage2_3D.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_AnchoredRobust_5.1.yaml`；commits `81c7f3a`,`15837c6`；Ibex jobs `50931950`,`50931968`,`50931970`,`50931980` | Stage2 development dataset-macro F1/AP gain `+.16025/+.17163`，达到development目标；10/10条目正。Stage1/Stage2 selection SHA-256 `0fca4fde…f7b76`/`8341272e…a2ab` | anchored block与family-specific residual在development超过`.15`，因此冻结进入一次性新空间确认；该行不把development指标当论文结论。 |
| mainExp_Task3_3D_5.1 | 2026-08-27 | Task3-3D anchored FMT新空间primitive population确认 | Stage2 selection冻结后，以预注册空间相位`[-.37,.29,-.11]`重建10条目×4切片、155157个有效primitive；whole-field IVD p95标签不变且40/40与source reference逐位相同；paired seeds40–44。主对照为同宽同结构train-only Raw-PCA residual；confirmation不选择feature、网络、epoch、threshold或alpha | `experiments/Build_Task3_AnchoredRobust_Confirmation_5_1.py`, `experiments/Run_Task3_FMTResidual_Frozen_5_1.py`, `docs/mainExp_Task3_3D.md` | `config/mainExp_Task3_3D_5.1*.yaml`；commits `15837c6`,`a32ad47`；Ibex jobs `50939374`–`50939376` | Raw-PCA/FMT dataset-macro F1 `.71683→.85274`，gain`+.13591`；AP `.77720→.92692`，gain`+.14971`。F1/AP均10/10条目、7/7 family、5/5 paired seeds正；family-macro gain`+.14269/+.15985`。200/200唯一记录；summary/per-run SHA-256 `7468a38b…11912`/`864bf268…21340` | **修订4.1：anchored FMT把相对Raw-PCA的F1/AP增益由`+.10000/+.11896`提高到`+.13591/+.14971`，且保留全条目正方向；但预注册F1 `+.15`仍差`.01409`，必须按失败记录。**若进入5.2，本confirmation只能作development，最终必须换全新空间population。 |
| Verify_Task3_SpatialRobust_5.2 | 2026-08-28 | Task3-3D空间增强与二阶端点FMT开发搜索 | 4.1公开phase同时加入FMT/Raw-PCA residual共同训练；5.1公开phase加入共同validation；冻结Raw checkpoint normalization。Stage1搜索30个label-free pathline blocks（含二阶单边端点导数），Stage2为family top4×18个route/width/lr/alpha设置；paired seeds40–42 | `FMT_Utils/DFT_FMT_3D.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Search_Task3_FMTResidual_Stage2_3D.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_SpatialRobust_5.2.yaml`；commits `04c093a`,`59aedf5`；Ibex jobs `50940819`–`50940822`,`50957870`,`50958095` | Stage1为1200/1200配对结果，dataset-macro F1/AP增益`+.13585/+.14244`且10/10条目正。Stage2 selector于2026-08-28T21:58:32+03:00完成：family-specific development F1/AP增益`+.14275/+.14782`，10/10条目正；预注册F1目标`.15`未达到，差`.00725`。7个family分别冻结配方；selection SHA-256 `a3be38ed…b450`。最终phase仍冻结为`[.318359375,.4561042524,-.3352]`且未被本搜索打开 | **Stage2把5.1 confirmation开发化后的F1增益提高到`+.14275`，但仍没有达到`.15`。**该结果只支持已公开development population上的候选选择；`confirmation_opened=false`，不能写成论文最终确认结论，也不能因差`.00725`而改目标。 |
| （预注册）mainExp_Task3_3D_5.2 | 2026-08-28 | Task3-3D第三空间primitive population一次性确认 | 与4.1/5.1相同physical times，仅使用预注册新phase；10条目×4切片×5 paired seeds；IVD p95及主对照不变；confirmation不选择任何超参数 | `experiments/Build_Task3_SpatialRobust_Confirmation_5_2.py`, `experiments/Run_Task3_FMTResidual_Frozen_5_2.py`, `docs/mainExp_Task3_3D.md` | `config/mainExp_Task3_3D_5.2*.yaml`；commit/job待生成 | 尚无结果；目标dataset-macro absolute F1 gain `>=+.15` | 无论结果是否达标均保留，不得换phase重跑后替换。 |
| Verify_Task3_NetworkArchitecture_6.1 | 2026-08-28 | Task3-3D配对网络结构开发搜索 | 冻结5.1 family-specific FMT feature；共同使用原development、4.1训练空间population和5.1 validation空间population，不读取5.2 final population。比较7种head：linear、historical shallow MLP、deep MLP、residual MLP、gated fusion、rank-32 bilinear fusion和4-head attention fusion。每个结构均按完全相同的冻结Raw backbone、输入宽度、seed、optimizer、epoch/early-stop和alpha网格，配对比较train-only同宽Raw-PCA与FMT；10条目×3 seeds×2 arms，共420次训练 | `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Search_Task3_NetworkArchitecture_6_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_NetworkArchitecture_6.1.yaml`；实现commit `57d6d6c`；registry commit `ad57dde`；Ibex jobs `50962559`–`50962561`；feature selection SHA-256 `8341272e…a2ab` | 420/420次训练均完成。全局胜者为2层×64 hidden units的deep MLP：Raw-PCA dataset-macro F1 `.74763`，FMT `.88526`，配对F1增益`+.13763`，AP增益`+.14617`；10/10 datasets、7/7 physical families及3/3 seeds均为正增益，最差数据集增益`+.04817`，最差seed增益`+.13624`。预注册目标`>=+.15`未达到，差`.01237` | 7种结构在10/10 datasets上均取得正F1增益，结构间增益范围仅`+.13294`–`+.13763`；deep MLP比historical shallow MLP高`+.00469`，attention仅低`.00022`且AP高`.00063`。因此FMT增益对所测结构稳定，但结构选择只能小幅放大增益，不能宣称已达到`+.15`。本结果仅用于development选择，`confirmation_opened=false`，5.2 final population仍未打开。 |
| Verify_Task3_LossOptimization_7.1 | 2026-08-28 | Task3-3D配对损失与训练超参数开发搜索 | 以已冻结5.2 Stage2 family配方和6.1胜出的2层×64 deep MLP为起点。25个候选对FMT与train-only同宽Raw-PCA两臂共同应用完全相同的类别权重、weighted BCE/focal loss、Dropout、weight decay、batch size、training alpha、epoch/patience和cosine learning-rate设置；10条目×3 seeds×2 arms。允许按physical family选择配方。稳定focal实现避免fractional gamma奇异梯度 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_loss_optimization.py` | `config/Verify_Task3_LossOptimization_7.1.yaml`；commits `c4f5beb`,`8ecc156`,`defb302`,`be6ac1c`；Ibex jobs `50966266`,`50966269`,`50967898`,`50967944`,`50972238`；selection SHA-256 `cc2d2dbb…c502` | selector完整通过。Raw-PCA/FMT dataset-macro F1 `.73602/.88666`，gain `+.15064`；AP `.78459/.94423`，gain `+.15964`。family-macro F1/AP gain `+.16122/+.17581`；10/10 datasets为正，最小dataset F1 gain `+.05371`。family胜者依次为channel focal γ3、halfcylinder focal γ1、deltaWing positive-weight `.5`、F22 positive-weight `2`、Boeing focal γ2、Smoke training-alpha `2`、Tangaroa positive-weight `.35`。`o07_focal_g05/halfcylinder`为唯一数值失效项 | **首次在development达到预注册F1增益`>=+.15`目标，较5.2的`+.14275`提高`.00789`。**这是超参数选择证据，不是final confirmation；`confirmation_opened=false`，不得将`.15064`直接写成论文最终泛化结果。 |
| Verify_Task3_RawHardness_8.1 | 2026-08-29 | Task3-3D冻结Raw难样本加权开发搜索 | 冻结5.2 family配方与6.1的2层×64 deep MLP。每个mini-batch用冻结Raw logit和训练标签计算mean-one难度权重，并对FMT与同宽同结构train-only Raw-PCA residual两臂逐位共用；权重与两种auxiliary feature均断开。搜索hardness scale、power、temperature、Raw误分类boost及交互共14候选；10条目×3 paired seeds×2 arms，共840次训练。只使用development population，不读取7.1半程排名或5.2 final population | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_loss_optimization.py` | `config/Verify_Task3_RawHardness_8.1.yaml`；implementation commit `e1052fb`；Ibex jobs `50969029`,`50969056`,`50969057`；selection SHA-256 `8197fba7…d678` | 140/140 array children、840/840 runs完整；selector stderr为空。Raw-PCA/FMT dataset-macro F1 `.73728/.88653`，gain `+.14926`；AP `.78766/.94540`，gain `+.15774`；family-macro F1/AP gain `+.16015/+.17409`；10/10 datasets为正，最小dataset F1 gain `+.05295`。family赢家为channel误分类boost4、halfcylinder hardness scale`.5`、deltaWing scale1、F22 scale2+temperature`.5`、Boeing scale8、Smoke scale2+power2、Tangaroa scale4 | 支持冻结Raw难度加权可在部分family显著放大相对增益，并保留所有数据集正方向；dataset-macro F1距`.15`目标差`.00074`，严格记为未达到，且略低于7.1的`+.15064`。该结果只作为11.1组合输入，`confirmation_opened=false`。 |
| Verify_Task3_ResidualCorrection_9.1 | 2026-08-29 | Task3-3D冻结Raw margin残差修正开发搜索 | 冻结5.2 family配方与6.1的2层×64 deep MLP。对冻结Raw尚未达到正确类margin的训练样本，构造“缺少的有号logit修正”目标；easy Raw样本目标为0，Smooth-L1辅助项与原weighted BCE相加。FMT与同宽同结构train-only Raw-PCA残差两臂使用完全相同的Raw logit、标签、margin、loss weight与Huber参数。搜索5个loss weight、3个margin及Huber/cap交互，共13候选、10条目×3 seeds×2 arms=780次训练 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_loss_optimization.py` | `config/Verify_Task3_ResidualCorrection_9.1.yaml`；implementation commit `2bb2836`；Ibex jobs `50969187`,`50969235`,`50969236`；selection SHA-256 `9c8c6b2b…1079` | 130/130 children、780/780结果完成。Raw-PCA/FMT dataset-macro F1增益`+.14262`，AP增益`+.14930`；10/10 datasets正。halfcylinder family选`r02_weight003`后F1增益`+.15869`，deltaWing选`r11_margin20_cap8`；channel、Tangaroa、Boeing选择control。预注册`.15`目标未达到 | **总体不支持Raw-margin修正进一步扩大FMT增益：F1/AP均低于7.1的`+.15064/+.15964`。**它对halfcylinder局部有效，因此按预注册设计保留为11.1组合输入；该结果仅为development，`confirmation_opened=false`。 |
| Verify_Task3_PairwiseRanking_10.1（已完成） | 2026-08-29 | Task3-3D正负样本配对排序开发搜索 | 冻结5.2 family配方与6.1的2层×64 deep MLP。每个mini-batch对全部正/负样本logit差施加稳定的smooth pairwise ranking loss，直接约束正样本排在负样本之前；FMT与同宽同结构train-only Raw-PCA两臂使用完全相同的标签、pair构造、loss weight、margin与temperature，参数量不变。搜索5个loss weight、4个margin及temperature交互，共14候选、10条目×3 seeds×2 arms=840次训练 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_loss_optimization.py` | `config/Verify_Task3_PairwiseRanking_10.1.yaml`；implementation commit `4cc4c9f`；Ibex jobs `50969371`,`50969571`,`50969582`；selection SHA-256 `33ba23a0…068b`，leaderboard `bc4aff8d…e07b` | 140/140数组任务与selector均exit 0、stderr为空；840/840唯一run（14候选、10 datasets、3 seeds、两臂各420）。family-specific选择后Raw-PCA/FMT dataset-macro F1=`.74413/.88747`，增益`+.14334`；AP=`.79704/.94684`，增益`+.14980`；10/10数据集为正，最小数据集增益`+.05427`，最差配对seed增益`+.05538`；`confirmation_opened=false` | **不支持**pairwise ranking把目标增益提高到`>=+.15`，也未超过7.1的`+.15064`；但支持FMT相对同设置Raw-PCA的正增益遍及10/10数据集。该结果保留为负开发结果，不替代7.1。 |
| Verify_Task3_CombinedOptimization_11.1（已完成） | 2026-08-29 | Task3-3D四类训练改进的冻结组合开发搜索 | 在8.1–10.1结果出现前预先声明完整`2^4` factorial：control、7.1 loss、8.1 Raw-hardness、9.1 Raw-margin correction、10.1 pairwise ranking的4个单因素、6个二因素、4个三因素和all-four。每个physical family只读取四个上游完整selector冻结的赢家，再按无冲突嵌套配置合并；FMT与同宽同结构train-only Raw-PCA两臂共用同一组合、2层×64 deep MLP、paired seeds40–42。16候选×10条目×3 seeds×2 arms=960次训练 | `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_loss_optimization.py` | `config/Verify_Task3_CombinedOptimization_11.1.yaml`；implementation commit `c1103d7`；Ibex jobs `50972698`,`50972700`,`50972701`；selection SHA-256 `c095fd78…bf51`，leaderboard `e4c7362a…754b` | 160/160数组任务与selector均exit 0、stderr为空；960/960唯一run（16候选、10 datasets、3 seeds、两臂各480），无失效候选。family-specific选择后Raw-PCA/FMT dataset-macro F1=`.73352/.88711`，增益`+.15359`；AP=`.78129/.94396`，增益`+.16268`；10/10数据集为正，最小数据集增益`+.05385`，最差paired-seed增益`+.05519`；`confirmation_opened=false` | **未达到**预注册`>=+.16`，但较7.1的`+.15064`提高`+.00295`，是当前development最佳。channel选loss+hardness+correction，Boeing选hardness，Tangaroa选loss+hardness，F22选loss+ranking；其余family保留loss单因素。该赢家已在12.1打开任何confirmation数据前冻结。 |
| Confirm_Task3_CombinedOptimization_12.1（失败，无性能结果） | 2026-08-29 | Task3-3D组合赢家的第三空间population独立确认 | 在11.1选择出现前封存confirmation：只读取11.1完整selector冻结的逐family赢家；以预注册phase `[.318359375,.4561042524005485,-.3352]`重建10 datasets×4 slices，使用paired seeds40–42及同结构同宽train-only Raw-PCA/FMT两臂，共60次评估。冻结后不得选择feature、loss、hardness、margin、ranking、threshold、epoch或alpha | `experiments/Confirm_Task3_CombinedOptimization_12_1.py`及6个`ibex_bash/confirm_task3_combined_12.1_*.sh` | `config/Confirm_Task3_CombinedOptimization_12.1.yaml`；implementation commit `163e7fa`；Ibex jobs `50974357`,`50974359`,`50974362`,`50974364`,`50974368`,`50974372`；frozen manifest SHA-256 `cf15aae9…d9d2` | static-preflight与freeze成功；10/10 cache children均在读取首个数据源前因本地Windows绝对路径不可用于Ibex而`FileNotFoundError`，没有生成任何confirmation cache或性能artifact；labels/evaluate/summary于10:20:46取消。`confirmation_data_opened=false` | 12.1不支持也不反对FMT性能结论；失败是可复现的部署路径错误，不得删除。12.2只修复数据源staging，不改变已冻结的开发赢家、时间、phase、标签、seeds或目标。 |
| Confirm_Task3_CombinedOptimization_12.2（已完成） | 2026-08-29 | 12.1的纯部署修订：以等值source pack恢复第三空间population独立确认 | 对4个Ibex缺失的大场保存production loader实际读取的4个独立时间窗，逐窗验证float32速度、物理边界、source time与time step；其余5个NetCDF和channel使用Ibex现有原文件。只把source path和pre-strided pack的effective index改为可执行值，原始10 datasets×4 times、phase、11.1逐family赢家、paired seeds40–42、IVD-p95与确认目标完全不变 | `experiments/Prepare_Task3_Confirmation_SourcePacks_12_2.py`, `experiments/Confirm_Task3_CombinedOptimization_12_1.py`, `experiments/Build_Task3_SpatialRobust_Confirmation_5_2.py` | `config/Confirm_Task3_CombinedOptimization_12.2.yaml`；commits `50a7067`,`10f880d`；初始jobs `50987111`–`50987117`，重试`50987320`–`50987325`，path重试`50988094`,`50988095`；frozen/source-preflight SHA `3b63c5b4…963e`,`abd2fc73…41be`；summary/per-run SHA `3bca2314…1667`,`2ea47be2…da1` | 修复后evaluation 10/10与summary均exit 0。第三空间population的dataset-macro Raw-PCA/FMT F1=`.73202/.86297`，gain=`+.13094`；AP=`.78637/.93656`，gain=`+.15018`。family-macro F1/AP gain=`+.14917/+.17070`；F1为9/10 datasets、7/7 families正，唯一负项Re160=`−.05799`；F22转为`+.09449`，channel为`+.57875` | **支持**11.1冻结赢家在新的空间population上相对同宽同结构Raw-PCA有广泛且较大的总体增益；**反对**预注册dataset-macro F1 `>=+.15`及aspirational `>=+.16`（相对development `+.15359`下降`.02265`），也反对逐dataset严格普适。12.2从不用于任何超参数选择；全部部署失败、取消与修复作业均保留。 |
| Verify_Task3_ClassBalancedBatches_13.1（已完成） | 2026-08-29 | Task3-3D按类精确平衡mini-batch开发搜索 | 冻结5.2 family配方与6.1的2层×64 deep MLP；只读取training标签，以可复现的exact class-balanced sampler令每批正类比例为`.05/.10/.20/.33/.50`，并比较batch size 128/256。为避免重复类别补偿，正类BCE权重从全训练集负/正比改为采样分布的`(1-q)/q`；FMT与同宽同结构train-only Raw-PCA逐位共用相同batch索引、optimizer和训练预算。10候选×10条目×3 seeds×2 arms=600次训练 | `experiments/Verify_Task3_FMTClassifier.py`, `experiments/Verify_Task3_FMTResidual.py`, `tests/test_task3_balanced_batches_13_1.py` | `config/Verify_Task3_ClassBalancedBatches_13.1.yaml`；implementation commit `be2c2fb`；Ibex jobs `50974957`,`50974962`,`50974963`；selection SHA-256 `fc07f3eb…b201`，leaderboard `9706e897…f8fc` | 100/100数组任务与selector均exit 0、stderr为空；600/600唯一run（10候选、10 datasets、3 seeds、两臂各300），无失效候选。family-specific选择后Raw-PCA/FMT dataset-macro F1=`.74245/.88741`，增益`+.14496`；AP=`.79616/.94548`，增益`+.14932`；10/10数据集为正，最小数据集增益`+.05272`，最差paired-seed增益`+.05503`；`confirmation_opened=false` | **不支持**class-balanced batch达到`>=+.15`目标或超过7.1的`+.15064`。它对channel（q=.50,batch256）、halfcylinder（q=.05）等family局部有效，保留为14.1组合输入；但总体不能替代7.1，也不打开confirmation。 |
| Verify_Task3_BalancedCombination_14.1（已完成） | 2026-08-29 | Task3-3D四因素赢家与class-balanced batch的低成本组合开发搜索 | 在11.1与13.1 selector完成前预声明两个候选：冻结的11.1逐family赢家，以及该赢家叠加冻结的13.1逐family赢家。组合器只合并`training/model`，保留源selector SHA并拒绝冲突键；FMT与同宽Raw-PCA共用完全相同的组合。2候选×10条目×3 seeds×2 arms=120次训练 | `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_balanced_combination_14_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_BalancedCombination_14.1.yaml`；commits `e028a59`,`81d0e23`；Ibex jobs `50975754`,`50975756`,`50975758`；selection SHA-256 `e6449071…654f` | 20/20数组任务与selector均exit 0；Raw-PCA/FMT dataset-macro F1=`.73301/.88663`，增益`+.15361`；AP增益`+.16433`；10/10数据集正，最小数据集增益`+.05385`；`confirmation_opened=false` | **未达到**预注册`>=+.165`；相对11.1只增加`+.00002`，不构成实质改善。平衡batch仅被halfcylinder family选择，不能作为新的总体方法结论。 |
| Verify_Task3_OverlapLoss_15.1（已完成） | 2026-08-29 | Task3-3D Soft Dice/Tversky重叠损失开发搜索 | 冻结5.2 family配方与6.1的2层×64 deep MLP；在原逐样本weighted BCE之外，对mini-batch正类集合加入可微Soft Dice或Tversky损失。FMT与同宽同结构train-only Raw-PCA两臂逐位共用相同损失、标签、batch、optimizer和训练预算。12候选×10条目×3 seeds×2 arms=720次训练 | `experiments/Verify_Task3_FMTResidual.py`, `tests/test_task3_overlap_loss_15_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_OverlapLoss_15.1.yaml`；implementation commit `d9a3c9f`；Ibex jobs `50976132`,`50976133`,`50976134`；selection SHA-256 `c5182cc9…604` | 120/120数组任务与selector均exit 0；Raw-PCA/FMT dataset-macro F1=`.74301/.88723`，增益`+.14422`；AP增益`+.15041`；10/10数据集正，最小数据集增益`+.05294`；`confirmation_opened=false` | **反对**达到`>=+.155`及超过7.1/11.1。重叠损失提高两臂部分绝对分数，但没有扩大FMT相对增益。 |
| Verify_Task3_OverlapBalancedCombination_16.1（已完成） | 2026-08-29 | Task3-3D core×balanced×overlap低成本组合开发搜索 | 在11.1、13.1、15.1三个selector产生前预声明围绕11.1 core的完整2×2组合：core、core+balanced、core+overlap、core+balanced+overlap；FMT与同宽Raw-PCA逐位共用相同组合。4候选×10条目×3 seeds×2 arms=240次训练 | `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_overlap_balanced_combination_16_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_OverlapBalancedCombination_16.1.yaml`；implementation commit `825340e`；Ibex jobs `50976236`,`50976237`,`50976238`；selection SHA-256 `fb165780…7bd` | 40/40数组任务与selector均exit 0；Raw-PCA/FMT dataset-macro F1=`.73366/.88715`，增益`+.15350`；AP增益`+.16435`；10/10数据集正，最小数据集增益`+.04509`；`confirmation_opened=false` | **反对**达到`>=+.17`或超过11.1。balanced与overlap对core没有可重复的总体叠加增益。 |
| Verify_Task3_AuxiliaryBottleneck_17.1（预检失败） | 2026-08-29 | Task3-3D FMT/Raw-PCA同宽辅助瓶颈开发搜索 | 冻结5.2 family配方、6.1 deep MLP及全部训练协议；原网格为4/8/16/24/32/48/64/96/128，64为control。逐dataset/seed预检要求两臂同结构参数预算并严格低于Raw-wide容量 | `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_auxiliary_bottleneck_17_1.py` | `config/Verify_Task3_AuxiliaryBottleneck_17.1.yaml`；implementation commit `4795e57`；Ibex jobs `50976865`,`50976866`,`50976867` | 预检在`deltaWing_resampled/d08_aux128`失败：模型超过Raw-wide容量上限；GPU数组0/90启动、无性能结果、confirmation未打开。失败证明容量守卫生效；原config和job保留，修订17.2只删除违规aux128 | 17.1不支持任何FMT性能结论。17.2保留4–96八个合规宽度和原选择协议；仍须并列报告绝对FMT/Raw-PCA F1/AP。 |
| Verify_Task3_AuxiliaryBottleneck_17.2（已完成） | 2026-08-29 | Task3-3D合规同宽辅助瓶颈开发搜索 | 17.1预检尚未产生任何性能结果时，只删除超出Raw-wide参数上限的aux128；保留4/8/16/24/32/48/64/96、64 control、数据、标签、seeds、deep MLP、训练与选择协议。FMT与train-only Raw-PCA两臂使用同一宽度、结构和参数预算。8候选×10条目×3 seeds×2 arms=480次训练 | `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_auxiliary_bottleneck_17_2.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_AuxiliaryBottleneck_17.2.yaml`；implementation commit `4efef70`；Ibex jobs `50977331`,`50977332`,`50977337`；selection SHA-256 `8e79059d…bd1` | 80/80数组任务与selector均exit 0；Raw-PCA/FMT dataset-macro F1=`.72231/.88209`，增益`+.15977`；AP=`.77069/.94058`，增益`+.16989`；10/10数据集正，最小数据集增益及最差paired-seed增益均`+.06273`。halfcylinder/channel/deltaWing/Boeing/Smoke选aux4，Tangaroa选aux8，F22选aux96 | 比11.1增益扩大`+.00618`并成为当前development最大，但距`+.16`差`.00023`。FMT绝对F1较11.1低约`.00502`，Raw-PCA低约`.01121`；支持“同宽窄瓶颈下FMT保留更多IVD信息”，不支持“FMT绝对分类性能提高”。 |
| Verify_Task3_SupervisedContrastive_18.1（已完成） | 2026-08-29 | Task3-3D辅助embedding监督式对比损失开发搜索 | 在冻结5.2 family配方和6.1 deep MLP上，对现有trainable auxiliary projection加入监督式对比目标；FMT与train-only Raw-PCA两臂使用相同loss weight、temperature、batch、Raw-derived weights、结构和seed。11候选×10条目×3 seeds×2 arms=660次训练 | `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Verify_Task3_FMTResidual.py`, `tests/test_task3_supervised_contrastive_18_1.py` | `config/Verify_Task3_SupervisedContrastive_18.1.yaml`；commits `0f825b2`,`c557809`；Ibex jobs `50977777`,`50977778`,`50977779`；selection SHA `0e380671…8d5` | 110/110数组任务和selector均exit 0；Raw-PCA/FMT F1=`.74502/.88745`，增益`+.14244`；AP=`.79755/.94604`，增益`+.14849`；10/10数据集正，最小数据集增益`+.05272`。Boeing与deltaWing选择control，其他family选择不同小权重/temperature | **反对**达到`>=+.16`及扩大17.2/11.1增益。FMT绝对F1与11.1近似持平（高约`.00034`），但Raw-PCA提高更多，因此对比损失单独不能作为主改进。 |
| Verify_Task3_RepresentationCombination_19.1 | 2026-08-29 | Task3-3D core×bottleneck×contrastive完整交互搜索 | 在读取18.1 selector前预声明完整`2^3` factorial：base、11.1 core、17.2同宽bottleneck、18.1 contrastive三个单因素、三个二因素及all-three。preflight只在三个完整selector存在后读取并冻结逐family配方及SHA；冲突参数直接拒绝。FMT与train-only Raw-PCA两臂共用同一组合、网络、batch、optimizer、训练预算与paired seeds40–42。8候选×10条目×3 seeds×2 arms=480次训练 | `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_representation_combination_19_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_RepresentationCombination_19.1.yaml`；implementation commit `894c4a9`；archive SHA `c5edced1…8728`；Ibex jobs `50987289`,`50987290`,`50987291`；selection/leaderboard SHA `67bc937c…d197`,`be60cf11…f546` | selector在80/80 children、480/480 paired trainings完成后运行；dataset-macro Raw-PCA/FMT F1=`.71042/.88283`，gain=`+.17240`；AP=`.75640/.94036`，gain=`+.18396`；10/10 datasets为正、最小dataset gain=`+.06381`，达到预注册`+.165`目标。family赢家：Boeing/F22选core，half-cylinder/Smoke选bottleneck，deltaWing选core+bottleneck，channel/Tangaroa选all-three；`confirmation_opened=false` | **相对增益目标达到，但绝对FMT几乎未提高。**相对17.2，FMT F1只`+.00074`，Raw-PCA F1下降`.01189`，因此gain增加`.01263`主要来自Raw下降。channel的Raw/FMT F1=`.18591/.81090`、gain=`+.62499`；去掉该条目的描述性macro gain仅`+.12212`。支持“组合配方下FMT比同结构Raw-PCA更稳健”，不支持“FMT绝对分类器显著增强”；不得将19.1作为新确认结论，必须在新空间population评估且完整报告逐dataset。 |
| Verify_Task3_UltraNarrowBottleneck_20.1 | 2026-08-29 | Task3-3D同宽辅助瓶颈低于4维的边界细化 | 17.2有五个physical families选择当时最小width 4，因此在不读取19.1结果的前提下预声明width 1/2/3，并保留4/8/96为17.2精确controls、加入6/12覆盖过渡。FMT与train-only Raw-PCA使用相同auxiliary width、2×64 deep MLP、训练协议、labels、splits及paired seeds40–42；每个candidate低于Raw-wide容量上限。8 widths×10 datasets×3 seeds×2 arms=480次训练 | `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_ultranarrow_bottleneck_20_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_UltraNarrowBottleneck_20.1.yaml`；implementation commit `c491d99`；archive SHA `933cdaf3…c042`；Ibex jobs `50987539`,`50987541`,`50987542`；selection SHA `3698877b…afb2` | development dataset-macro Raw-PCA/FMT F1=`.70497/.87872`，gain=`+.17375`；AP=`.75670/.92704`，gain=`+.17035`；10/10 datasets为正，最小F1 gain=`+.06535`，达到预注册`+.165`目标。family widths：channel 2、halfcylinder/Boeing 3、deltaWing/Smoke 4、F22 6、Tangaroa 8。相对19.1，F1 gain仅再增`.00135`，但FMT绝对F1降`.00411`、Raw-PCA降`.00545`，FMT AP降`.01331`；`confirmation_opened=false` | 支持FMT在2–8维同宽瓶颈下比Raw-PCA更稳健，也产生当前最大development F1差值；不支持“超窄瓶颈提高绝对FMT分类器”。差值扩大主要伴随两臂退化，不能替代绝对表现更高的方法。 |
| Verify_Task3_AuxiliaryProjection_21.1（完成，development） | 2026-08-29 | Task3-3D同宽辅助投影结构搜索 | 历史`Linear→LayerNorm→GELU`在auxiliary width 1时逐样本严格输出0，width 2也移除逐样本均值与尺度；在完整20.1 selector冻结family-specific width后，共同比较历史control、linear-only、无归一化GELU/SiLU、RMSNorm以及64维非线性pre-projection。FMT与train-only Raw-PCA逐candidate使用相同投影、输出width、2×64 residual head、训练协议及paired seeds40–42；8 projections×10 datasets×3 seeds×2 arms=480次训练 | `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_auxiliary_projection_21_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_AuxiliaryProjection_21.1.yaml`；implementation commit `c3fb5c2`；archive SHA `dd76801e…facb7`；Ibex jobs `50988984`,`50988985`,`50988990`；selection/leaderboard SHA `d14031f2…981`,`98d4723c…1fc` | 80/80 children、480/480 paired trainings及selector全部exit 0。Raw-PCA/FMT dataset-macro F1=`.70402/.87860`，gain=`+.17457`；AP=`.75389/.92744`，gain=`+.17355`；10/10 datasets F1正增益，达到预注册`+.170`目标，`confirmation_opened=false`。仅Boeing与Tangaroa选择`Linear→RMSNorm→GELU`，其余5个family保留历史投影 | 相对20.1，F1 gain仅提高`+.00082`，但Raw-PCA/FMT F1分别变化`−.00095/−.00012`；绝对FMT F1未改善且仍低于`.887`。AP gain提高`+.00320`，来自Raw-PCA AP下降`−.00281`与FMT AP小幅提高`+.00039`。因此支持“修复投影退化可微增差值”，不支持“投影搜索显著提高FMT绝对性能”；不得把主要由Raw下降形成的差值扩张称为方法改善。 |
| Verify_Task3_AnchoredFeatureDecomposition_22.1（已完成，development） | 2026-08-29 | Task3-3D anchored IVD 表示分解 | 在与5.2 Stage1完全相同的10 datasets、两套exposed spatial populations、split、训练预算及paired seeds40–41上，保留5个exact family controls，并比较first/early/DFT/core/stats共11个新表示；FMT与train-only Raw-PCA逐candidate同宽、同trainable residual architecture。selection先要求FMT F1和AP各自在冻结5.2 control的`−.002`内，再最大化FMT−Raw-PCA F1 gain | `FMT_Utils/DFT_FMT_3D.py`, `FMT_Utils/Task12Data_3D.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Preflight_Task3_AnchoredFeatureDecomposition_22_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_AnchoredFeatureDecomposition_22.1.yaml`；implementation commit `8271645`；Ibex jobs `50991000`–`50991002`；selection/leaderboard SHA `b65f0f65…0fb9`/`800d52a9…44bf` | 160/160 children、640/640 paired trainings和selector均exit 0。Raw-PCA/FMT dataset-macro F1=`.70232/.88922`，gain=`+.18691`；AP=`.74639/.95044`，gain=`+.20405`；10/10 datasets、7/7 family和所有paired seeds均为正，最小dataset F1 gain=`+.04630`。相对冻结5.2 control，FMT F1/AP绝对值提高`+.00190/+.00181`。family赢家依次为Boeing first、channel/Smoke两频DFT、deltaWing control、F22 second-order core、halfcylinder early、Tangaroa one-frequency DFT；`confirmation_opened=false` | F1 gain目标`>=+.150`明确达到并成为当前development最大；absolute FMT F1=`.88922`距离`.890`仍差`.00078`，因此联合目标未完全达到。与20–21.1不同，本轮FMT绝对质量确有小幅上升，但差值的大部分仍来自Raw-PCA降至`.70232`；不得只报`+.18691`。该赢家进入24.1与25.1组合搜索，仍须fresh population确认。 |
| Verify_Task3_ConfidenceGatedResidual_23.1（已完成，development） | 2026-08-29 | Task3-3D frozen-Raw confidence-gated residual 搜索 | 冻结完整11.1逐family赢家；对FMT与train-only Raw-PCA两臂共同施加只依赖冻结Raw logit的无标签gate：`p=sigmoid(raw/T)`、`u=4p(1-p)`、`gate=floor+(1-floor)u`。比较`T∈{.5,1,2}`×`floor∈{0,.25,.5,.75}`完整factorial及no-gate exact control；同架构、width、loss、split、alpha search与seeds40–42。13 candidates×10 datasets×3 seeds×2 arms=780次训练 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_confidence_gated_residual_23_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_ConfidenceGatedResidual_23.1.yaml`；implementation commit `f2fb0a5`；archive SHA `06afe6f8…3d7e`；remote config SHA `536a02e8…bb5a`；source 11.1 selection SHA `c095fd78…bf51`；Ibex jobs `50993404`,`50993420`,`50993434`；selection/leaderboard SHA `e66965ad…2f60`/`08f96561…9871` | 130/130 children、780/780 trainings及selector均exit 0，全部130个GPU stderr和selector stderr为空。Raw-PCA/FMT dataset-macro F1=`.73151/.88667`，gain=`+.15516`；AP=`.77925/.94465`，gain=`+.16540`；10/10 datasets正，最小dataset F1 gain=`+.04978`；`confirmation_opened=false`。channel/F22选择`g10_t05_f075`，deltaWing选择`g09_t20_f050`，其余4 family保留control | **未达到**预注册F1 gain`>=+.160`及absolute FMT F1`>=.887`。相对11.1，gain只约增加`+.00157`，同时FMT F1约下降`.00044`而Raw-PCA约下降`.00201`；因此差值扩张主要来自Raw下降，不支持confidence gate成为新方法。仍支持FMT相对同结构Raw-PCA在10/10数据集有正增益，但不打开confirmation。 |
| Verify_Task3_ProjectionFeatureCombination_24.1（已完成，development） | 2026-08-29 | Task3-3D auxiliary projection × FMT feature 组合 | 在不读取21.1/22.1部分数组结果的前提下预声明四格比较：20.1 ultra-narrow width + 历史投影的exact control、21.1 projection winner、22.1 feature winner及两者组合。FMT与train-only Raw-PCA逐candidate使用相同feature维数、投影、width、head、split、训练预算与paired seeds40–42；Raw-PCA只在training数据拟合。候选特征分别加载，preflight要求Raw数组和labels逐字节相同。4 candidates×10 datasets×3 seeds×2 arms=240次训练 | `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_projection_feature_combination_24_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_ProjectionFeatureCombination_24.1.yaml`；implementation commit `dde7da4`，部署记录commit `a37dc12`；archive SHA `bda99f5b…53732`；preflight manifest SHA `ec2e38fa…1743`；Ibex jobs `50996005`,`50996006`,`50996007`；selection/leaderboard SHA `bba89b6b…d6e`/`d67f86dd…ab1` | 40/40 children、240/240 paired trainings和selector均exit 0，全部GPU stderr为空。Raw-PCA/FMT dataset-macro F1=`.70390/.87879`，gain=`+.17490`；AP=`.75455/.92721`，gain=`+.17266`；10/10 datasets为正，`confirmation_opened=false`。Boeing选择projection+feature，F22/halfcylinder/Tangaroa选择projection，其余family保留control | **未达到**预注册F1 gain`>=+.175`（差`.00010`）或absolute FMT F1`>=.887`（差`.00821`）。相对22.1，FMT F1/AP分别下降`.01043/.02324`，gain也下降`.01201`；大多数feature组合被absolute guard拒绝，channel特征变体尤其显著破坏FMT。结论是不支持把21.1投影与22.1特征普遍组合，后续搜索继续以22.1 anchored feature为独立基线。 |
| Verify_Task3_SemanticBlockProjection_25.1（已完成，development） | 2026-08-29 | Task3-3D semantic blockwise auxiliary projection | 检验“按FMT语义块投影后再送入同一deep residual head”能否把22.1的微小FMT绝对增益放大。比较历史5.2 feature与完整22.1逐family feature；每种分别配dense control、blockwise linear GELU、blockwise RMSNorm GELU、blockwise MLP GELU hidden16。FMT与train-only Raw-PCA两臂使用相同block维数、projection、width、head、split、训练预算和paired seeds40–42。8 candidates×10 datasets×3 seeds×2 arms=480次训练 | `FMT_Utils/PathlineClassifier_3D.py`, `FMT_Utils/Task12Data_3D.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_semantic_block_projection_25_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_SemanticBlockProjection_25.1.yaml`；implementation commit `3969e8d`；archive SHA `35e49775…c37`；remote config SHA `031e668b…5dba`；Ibex jobs `50996829`,`50996832`,`50996835`；selection/leaderboard SHA `528dc082…eed9`/`64a0c9ed…ddf` | 80/80 children、480/480 paired trainings及selector均exit 0。Raw-PCA/FMT dataset-macro F1=`.69666/.88717`，gain=`+.19052`；AP=`.74250/.94809`，gain=`+.20559`；10/10 datasets为正，达到预注册gain目标；`confirmation_opened=false`。family赢家中channel、halfcylinder、Boeing、Smoke选择anchored feature+blockwise linear GELU，deltaWing保留dense feature control | **差值刷新development记录，但联合目标失败：absolute FMT F1 `.88717 < .890`。**相对22.1，FMT F1/AP变化`−.00205/−.00236`，Raw-PCA下降更多，故gain扩大主要不是FMT绝对分类器变强。支持semantic block projection提高FMT相对Raw-PCA的稳健性，不支持其作为绝对FMT改进；不打开confirmation。 |
| Verify_Task3_AuxiliaryDeepSupervision_26.1（已完成，development） | 2026-08-29 | Task3-3D auxiliary representation直接深监督 | 冻结完整11.1逐family优化配方和22.1 anchored feature；在投影后的辅助表示上增加仅训练期使用的二分类头，推理仍为原frozen-Raw+residual。比较exact control与`linear/MLP(32)`×loss weight `{.01,.03,.10,.30,1}`完整网格。FMT与train-only Raw-PCA逐candidate使用相同head、loss weight、projection、batch、optimizer、预算及paired seeds40–42；辅助头在所有推理模块之后构建，不改变共享推理路径初始化。11 candidates×10 datasets×3 seeds×2 arms=660次训练 | `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_auxiliary_deep_supervision_26_1.py`, `docs/Verify_experiments.md` | `config/Verify_Task3_AuxiliaryDeepSupervision_26.1.yaml`；implementation commit `4f2a37f`；archive SHA `d7905ba9…20a5`；remote config SHA `59077b9f…d3e4`；Ibex jobs `50997126`,`50997130`,`50997133`；selection/leaderboard SHA `1693a9f9…257f`/`d49b2932…7ccd` | 110/110 children、660/660 paired trainings及selector均exit 0。Raw-PCA/FMT dataset-macro F1=`.69289/.88490`，gain=`+.19201`；AP=`.73530/.94578`，gain=`+.21049`；10/10 datasets为正，达到预注册gain目标；`confirmation_opened=false`。channel选MLP weight1、F22选MLP weight`.01`、Boeing选linear weight`.01`、Smoke选linear weight1，其余family保留control | **再次刷新development差值，但联合目标失败：absolute FMT F1 `.88490 < .892`。**相对22.1，FMT F1/AP下降`−.00432/−.00466`，Raw-PCA下降更大；因此不支持“deep supervision增强FMT绝对表示”，也不得只报`+.19201`。结果继续作为诊断，不打开confirmation。 |
| Verify_Task3_LearningRateWeightDecay_27.1（已完成，development） | 2026-08-29 | Task3-3D learning rate × weight decay全因子搜索 | 冻结22.1逐family anchored feature、5.2 population/split、2层×64 residual MLP和paired seeds40–42；搜索learning rate `{.0002,.0005,.001,.002,.005}` × weight decay `{0,.0001,.001}`完整15格。逐candidate的FMT与train-only Raw-PCA使用相同optimizer cell、feature维数、网络、split和预算。15 candidates×10 datasets×3 seeds×2 arms=900次训练 | `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_learning_rate_weight_decay_27_1.py`, `docs/Verify_experiments.md` | config/implementation/archive见实验文档；Ibex jobs `50997722`,`50997723`,`50997724`；selection/leaderboard SHA `5f340246…acdbd`/`ff8542d8…1573` | 150/150 children及selector均exit 0。Raw-PCA/FMT dataset-macro F1=`.69693/.88845`，gain=`+.19152`；AP=`.73908/.94696`，gain=`+.20788`；10/10 datasets正；`confirmation_opened=false` | **达到relative gain目标但联合目标失败：absolute FMT F1 `.88845 < .892`。**相对22.1，gain增加`+.004612`，但FMT F1/AP下降`.000770/.003483`；因此不支持“optimizer搜索增强absolute FMT”，不打开confirmation。 |
| Verify_Task3_OptimizerFamily_28.1（已完成，development） | 2026-08-29 | Task3-3D optimizer family开发搜索 | 在27.1结果产生前预注册；冻结27.1逐family feature、learning rate和weight decay，公平比较AdamW、AMSGrad、Adam、RAdam与NAdam。6 candidates×10 datasets×3 seeds×2 arms=`360` trainings | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_optimizer_family_28_1.py`, `docs/Verify_experiments.md` | implementation commit `db75746`；Ibex jobs `50999751–50999753`；60/60 children、360/360 trainings成功；selection/leaderboard SHA `9a8b392c…e5ebb`/`2f86dc11…6ff38` | Raw-PCA/FMT F1=`.696930/.888538`，gain=`+.191608`；AP=`.739405/.947280`，gain=`+.207875`；10/10 datasets正。Channel选RAdam、F22选AdamW AMSGrad，其余family保留control | **未达到**gain`>=+.195`或absolute FMT F1`>=.893`；optimizer family相对27.1为中性结果，不打开confirmation。 |
| Verify_Task3_ResidualHeadNormActivation_29.1（已完成，development） | 2026-08-30 | Task3-3D residual head normalization×activation开发搜索 | 冻结22.1 feature和5.2 protocol，比较LayerNorm/RMSNorm/none × GELU/SiLU/ReLU完整3×3因子；逐candidate两臂同head与预算。9 candidates×10 datasets×3 seeds×2 arms=`540` trainings | `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_residual_head_norm_activation_29_1.py`, `docs/Verify_experiments.md` | implementation commit `a3219db`；Ibex jobs `51001984`,`51001986`,`51001989`；90/90 children成功；selection/leaderboard SHA `1470e7f5…38de0`/`e8da4ae7…711f` | Raw-PCA/FMT F1=`.697348/.887125`，gain=`+.189777`；AP gain=`+.205972`；10/10 datasets正。相对28.1，FMT F1下降`.001414`、gain下降`.001831` | **未达到**联合目标；normalization/activation不支持成为改进方法，confirmation关闭。 |
| Verify_Task3_ResidualOutputInitialization_30.1（已完成，development） | 2026-08-30 | Task3-3D residual输出层初始化开发搜索 | 比较exact default、zero、6种small-normal及Xavier gain`.1`；仅改terminal residual输出层，逐candidate两臂同初始化。9 candidates×10 datasets×3 seeds×2 arms=`540` trainings | `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `tests/test_task3_residual_output_initialization_30_1.py`, `docs/Verify_experiments.md` | implementation commit `e1c8361`；Ibex jobs `51002453`,`51002457`,`51002460`；90/90 children成功、stderr全空；selection/leaderboard SHA `78824f30…6726e`/`5f702052…bef2b` | Raw-PCA/FMT F1=`.697335/.887125`，gain=`+.189790`；AP gain=`+.205999`；10/10 datasets正。Half-cylinder选normal std`.025`、Smoke选`.005`，其余保留default；相对29.1 gain仅`+.000013`且FMT F1不变 | **未达到**联合目标；output initialization无实质改进，confirmation关闭。 |
| Verify_Task3_ResidualHeadDepthWidth_31.1（本地预检失败） | 2026-08-30 | Task3-3D residual head隐藏宽度×深度完整因子 | 在22.1 anchored feature、5.2 development population、LayerNorm+GELU、zero dropout及paired seeds40–42上，预声明宽度`32/48/64/80/96`×深度`1/2/3`；每个cell的FMT与train-only Raw-PCA同结构、同训练流程，Raw-wide参数上限冻结为148,225 | `config/Verify_Task3_ResidualHeadDepthWidth_31.1.yaml`, `tests/test_task3_residual_head_depth_width_31_1.py`, `docs/Verify_experiments.md` | 尚未提交Ibex；本地11项结构/修订测试中的31.1部分通过 | 完整preflight在`smokeBuoyancy/c14_w96_d3`因参数量超过Raw-wide上限而失败；未生成任何训练或性能结果 | 不支持性能结论；证明容量守卫生效。31.2只删除整个width96水平，保留其余完整4×3因子，31.1失败记录不删除。 |
| Verify_Task3_ResidualHeadDepthWidth_31.2（已完成，development） | 2026-08-30 | Task3-3D capacity-safe residual head宽度×深度搜索 | 31.1纯容量修订：宽度`32/48/64/80`×深度`1/2/3`完整4×3因子；`64×2`为exact control。FMT与train-only Raw-PCA逐cell使用同一head、feature宽度、初始化、optimizer、split、预算和paired seeds40–42；selection要求FMT F1/AP均不低于control | `config/Verify_Task3_ResidualHeadDepthWidth_31.2.yaml`, `tests/test_task3_residual_head_depth_width_31_2.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_depth_width_31.2_*.sh` | implementation commit `b7b7a9b`；Ibex jobs `51003042`–`51003044`；120/120 children、720/720 trainings及selector成功，stderr全空；selection/leaderboard SHA `140f568d…f359`/`e895b2a2…dc97` | Raw-PCA/FMT F1=`.695248/.888201`，gain=`+.192953`；AP gain=`+.207882`；halfcylinder与Tangaroa选80×1，F22选64×1，其余保留64×2 control；`confirmation_opened=false` | F1 gain和absolute FMT F1均未达到`.195/.893`联合目标。相对22.1差值扩大但FMT F1下降约`.0010`，不支持更大head改善FMT绝对质量。 |
| Verify_Task3_TrainingResidualScale_32.1（已完成，development） | 2026-08-30 | Task3-3D训练期residual scale系统搜索 | 在不读取27.1–31.2 partial/final性能的前提下预注册；冻结22.1逐family anchored feature、5.2 population/split、2层×64 LayerNorm+GELU head及paired seeds40–42。比较training alpha `{.125,.25,.5,.75,1,1.5,2,3,4}`；该值只缩放训练logit中的residual及其分类梯度，validation/inference fusion alpha仍独立选择。逐candidate两臂使用相同alpha、网络、初始化、optimizer、split与预算。9 candidates×10 datasets×3 seeds×2 arms=`540` trainings | `config/Verify_Task3_TrainingResidualScale_32.1.yaml`, `tests/test_task3_training_residual_scale_32_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_training_residual_scale_32.1_*.sh` | implementation commit `eb00dc6`；Ibex jobs `51003518`,`51003520`,`51003522`；90/90 children、540/540 trainings及selector成功，stderr全空；selection/leaderboard SHA `07d46203…0248`/`9aa46d2e…b4e` | Raw-PCA/FMT F1=`.690615/.889486`，gain=`+.198870`；AP gain=`+.213712`；Channel/Tangaroa/F22/Boeing/halfcylinder选择非1 alpha，Delta Wing与Smoke保留control；`confirmation_opened=false` | 首次超过`.195`相对增益目标，但absolute FMT F1 `.889486 < .893`，联合目标失败。差值增加不能单独写成成功，confirmation保持关闭。 |
| Verify_Task3_GradientClipping_33.1（已完成，development） | 2026-08-30 | Task3-3D全局梯度范数裁剪单因素搜索 | 在不读取27.1–32.1 partial/final性能的前提下预注册；冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head及paired seeds40–42。比较no-clipping exact control与global gradient L2-norm limit `{.1,.25,.5,1,2,5,10}`；逐candidate的FMT与train-only Raw-PCA使用相同threshold、网络、optimizer、split与预算。8 candidates×10 datasets×3 seeds×2 arms=`480` trainings | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_GradientClipping_33.1.yaml`, `tests/test_task3_gradient_clipping_33_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_gradient_clipping_33.1_*.sh` | implementation commit `ba1816f`；Ibex jobs `51004078`,`51004081`,`51004084`；80/80 children、480/480 trainings及selector成功，80个stderr全空；selection/leaderboard SHA `8b4ad715…e254`/`c7e785bf…b61` | Raw-PCA/FMT dataset-macro F1=`.696592/.887917`，gain=`+.191325`；AP=`.740252/.947162`，gain=`+.206910`；Channel/halfcylinder/Tangaroa/DeltaWing分别选clip norm `2/1/.25/5`，F22/Boeing/Smoke保留no-clipping；相对exact control，FMT F1/AP仅增加`.000792/.001417`，gain增加`.001551`；`confirmation_opened=false` | **未达到**gain`>=+.195`或absolute FMT F1`>=.893`；gradient clipping仅有轻微开发集改善，不支持新的联合成功结论，也不打开confirmation。 |
| Verify_Task3_AdamBetas_34.1（已完成，development） | 2026-08-30 | Task3-3D AdamW一阶/二阶动量完整因子搜索 | 在不读取27.1–33.1 partial/final性能的前提下预注册；冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head、AdamW base learning rate/weight decay及paired seeds40–42。比较`beta1∈{.5,.9,.95}`×`beta2∈{.9,.99,.999}`完整3×3因子，`(.9,.999)`使用no-override exact control。逐candidate的FMT与train-only Raw-PCA使用相同beta pair、网络、初始化、split与预算。9 candidates×10 datasets×3 seeds×2 arms=`540` trainings | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_AdamBetas_34.1.yaml`, `tests/test_task3_adam_betas_34_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_adam_betas_34.1_*.sh` | implementation commit `4dc40ad`；Ibex jobs `51004187`,`51004190`,`51004191`；90/90 children、540/540 trainings及selector成功，stderr全空；GPU为47×P100、39×GTX 1080 Ti、3×RTX 2080 Ti、1×V100；selection/leaderboard SHA `1c71f658…c18e`/`9aa9454a…ff8a` | Raw-PCA/FMT F1=`.696919/.887689`，gain=`+.190770`；AP=`.740544/.946010`，gain=`+.205466`；halfcylinder选`(.95,.999)`，Tangaroa/DeltaWing/Boeing选`(.95,.9)`，其余保留default control；相对全control，FMT F1仅`+.000564`、gain`+.000996`，AP gain反而`−.000506`；`confirmation_opened=false` | **未达到**gain`>=+.195`或absolute FMT F1`>=.893`；Adam beta tuning只有轻微开发集变化，不支持新的联合成功结论，也不打开confirmation。 |
| Verify_Task3_LinearWarmup_35.1（已完成，development） | 2026-08-30 | Task3-3D初始线性学习率warmup长度搜索 | 在不读取27.1–34.1 partial/final性能的前提下预注册；冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head、base optimizer/loss/epoch budget及paired seeds40–42。比较no-warmup exact control与`{1,2,5,10,20,40}` warmup epochs；warmup从base learning rate的10%线性升至100%，随后保持不变。逐candidate的FMT与train-only Raw-PCA使用相同schedule、网络、初始化、split与预算。7 candidates×10 datasets×3 seeds×2 arms=`420` trainings | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_LinearWarmup_35.1.yaml`, `tests/test_task3_linear_warmup_35_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_linear_warmup_35.1_*.sh` | implementation commit `10436a4`；archive SHA `c55531af…b75c`；Ibex jobs `51004391`,`51004395`,`51004397`；70/70 children、420/420 trainings及selector成功，70个stderr全空；9×P100、61×GTX 1080 Ti；selection/leaderboard SHA `18543e09…4e0`/`57bd7b20…858` | 所有family均选择no-warmup control；Raw-PCA/FMT F1=`.697351/.887125`，gain=`+.189774`；AP=`.739773/.945745`，gain=`+.205972`；少数family的warmup仅轻微提高absolute FMT，但Raw同步改善使paired gain下降；`confirmation_opened=false` | **未达到**gain`>=+.195`或absolute FMT F1`>=.893`。不支持linear warmup改善Task3，且该因素不进入后续组合或confirmation。 |
| Verify_Task3_BatchSize_36.1（已完成，development） | 2026-08-30 | Task3-3D mini-batch size单因素搜索 | 冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head、base optimizer/loss/epoch budget及paired seeds40–42。比较batch size `{32,64,128,256,512,1024,2048}`，`512`为exact control；每个candidate的FMT与train-only Raw-PCA使用相同batch、初始化、split和训练预算。7 candidates×10 datasets×3 seeds×2 arms=`420` trainings | `config/Verify_Task3_BatchSize_36.1.yaml`, `tests/test_task3_batch_size_36_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_batch_size_36.1_*.sh` | implementation commit `26f0218`；Ibex jobs `51006631`–`51006633`；70/70 children和selector成功；selection/leaderboard SHA `8ae6af33…e5bd`/`e9ad9508…d135`；`confirmation_opened=false` | Raw-PCA/FMT dataset-macro F1=`.69468/.88764`，gain=`+.19296`；AP=`.73827/.94600`，gain=`+.20773`；10/10 datasets正。halfcylinder选batch128、Tangaroa选1024、channel选32，其他family保留或选择见selector | **未达到联合目标**：gain距`.195`差`.00204`，absolute FMT F1距`.893`差`.00536`。相对22.1，gain扩大约`.00605`但FMT F1下降约`.00158`，因此主要仍由Raw-PCA下降形成；不支持batch size作为新的绝对FMT改进，也不打开confirmation。 |
| Verify_Task3_PositiveWeightScale_37.1（完成，负消融） | 2026-08-30 | Task3-3D positive-class weight单因素扩展搜索 | 7.1中Tangaroa在下界`0.35`、F22在上界`2.0`，说明原范围未闭合。37.1冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head及paired seeds40–42；比较scale `{.10,.20,.30,.35,.40,.50,.60,.75,1,1.25,1.5,2,2.5,3,4}`，scale-1为无override exact control；每cell两臂使用相同scale、初始化、split和训练预算。15 candidates×10 datasets×3 seeds×2 arms=`900` trainings | `config/Verify_Task3_PositiveWeightScale_37.1.yaml`, `tests/test_task3_positive_weight_scale_37_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_positive_weight_scale_37.1_*.sh` | implementation commit `372a56c`；archive SHA `d6db2b28…11b2f`；本地/远端full preflight manifest SHA `976e01a1…0ceb`/`d4ec39fe…ee34`；Ibex jobs `51010975`,`51010980`,`51010981`全部exit 0；150/150 children、900/900 trainings完整；leaderboard/selection SHA `85786d39…01a9`,`25b2f8f3…30be`且远端与下载副本一致；`confirmation_opened=false` | family-specific选择后Raw-PCA/FMT F1 `.69716/.88778`，gain `+.19062`；AP `.74053/.94799`，gain `+.20746`；10/10 datasets正。Channel/DeltaWing/Smoke选择scale `.10/.50/.10`，其余family保留scale 1；相对22.1 development gain提高约`+.00371`，但absolute FMT F1下降约`.00144` | **未达到**联合目标：gain低于`+.195`且absolute FMT F1低于`.893`。支持正类权重可小幅扩大差值，不支持其提高整体FMT分类质量；不进入fresh confirmation。 |
| Verify_Task3_ResidualDropout_38.1（已完成，development） | 2026-08-30 | Task3-3D residual head dropout单因素搜索 | 冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head及paired seeds40–42；完整比较dropout `{0,.025,.05,.075,.10,.15,.20,.30,.40,.50}`。每cell两臂使用相同dropout、初始化、split和训练预算；逐family选择先要求FMT F1/AP不低于zero-dropout control，再最大化paired F1 gain。10 candidates×10 datasets×3 seeds×2 arms=`600` trainings | `config/Verify_Task3_ResidualDropout_38.1.yaml`, `tests/test_task3_residual_dropout_38_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_dropout_38.1_*.sh` | implementation commit `274718f`；Ibex jobs `51012519`,`51012521`,`51012532`全部exit 0；100/100 children完整、selector stderr为空；selection/leaderboard/per-run archive SHA `f94c51b…c7616`/`a2abbb54…a5f8`/`c9cc7570…038b`且本地/远端一致；从600份逐次CSV独立重做guard、family排序与macro，误差`<1e-12`；远端600个临时`.pt`已删除 | 逐family dropout为channel `.10`、halfcylinder `.40`、Tangaroa `.30`、deltaWing `0`、F22 `.50`、Boeing `.50`、Smoke `0`。selected Raw-PCA/FMT F1=`.69188/.88836`，gain=`+.19648`，AP gain=`+.21188`；zero-dropout control=`.69768/.88708`，gain=`+.18940` | **支持Dropout可进一步增加FMT增益**：paired F1 gain提高`+.00708`且absolute FMT F1提高`+.00128`。达到gain`>=+.195`，但absolute FMT F1距`.893`仍差`.00464`，故joint target未达到；仅进入后续组合搜索，confirmation保持关闭。 |
| Verify_Task3_FocalGamma_39.1（已完成，development） | 2026-08-30 | Task3-3D focal-loss gamma单因素搜索 | 7.1中Channel、halfcylinder、Boeing分别选择gamma 3/1/2，但当时使用较弱5.2表示。39.1在冻结22.1 family feature和5.2 development protocol上比较exact weighted-BCE control与gamma `{.10,.25,.50,.75,1,1.5,2,2.5,3,4,5}`；每cell两臂同loss、gamma、head、optimizer、split、seed和预算。12 candidates×10 datasets×3 seeds×2 arms=`720` trainings | `config/Verify_Task3_FocalGamma_39.1.yaml`, `tests/test_task3_focal_gamma_39_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_focal_gamma_39.1_*.sh`, `experiments/Audit_Task3_ParameterSearch.py` | implementation commit `9e824d3`；Ibex jobs `51012669`,`51012681`,`51012699`,`51053640`全部exit 0；120/120 children、720/720 trainings完整且GPU/selector/archive stderr为空；selection/leaderboard/per-run archive SHA `1e99c432…e994`/`b76870ab…6ddc`/`58a10006…6af9`；独立重做Cartesian product、guard、family选择与macro，最大误差`1.11e−16`，audit SHA `bff22e74…48c3`；720个临时checkpoint已删除且剩余0 | family赢家仅Channel、halfcylinder选择gamma `.10`，其余5个family保留exact control。selected Raw-PCA/FMT F1=`.69734/.88750`，gain=`+.19016`；AP gain=`+.20656`；10/10 datasets正。exact control F1=`.69728/.88712`，gain=`+.18984`；选择仅使gain提高`+.00032`、absolute FMT F1提高`+.00039` | **未达到**gain`>=+.195`或absolute FMT F1`>=.893`，且弱于38.1 Dropout的`+.19648`；不单独打开confirmation，仅作为44.1/48.1预注册组合输入保留。 |
| Verify_Task3_FocalGammaLow_50.1（已完成，development） | 2026-08-30–31 | Task3-3D低gamma边界跟进 | 已完成39.1只在Channel和halfcylinder选择其下边界gamma`.10`，且更大gamma普遍触发absolute-FMT guard。50.1保持22.1 family feature、5.2 development population、同一网络/optimizer/split/seed/预算，比较exact weighted-BCE control与gamma `{.01,.025,.05,.075,.10,.15,.20}`；8 candidates×10 datasets×3 seeds×2 arms=`480` trainings | `config/Verify_Task3_FocalGammaLow_50.1.yaml`, `tests/test_task3_focal_gamma_low_50_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_focal_gamma_low_50.1_*.sh`, `experiments/Audit_Task3_ParameterSearch.py` | parent 39.1 selection SHA `1e99c432…e994`冻结；commit `cf85a72`；jobs `51056257`,`51056260`,`51056263`,`51072242`,`51056264`全部exit 0；80/80 children、480/480 trainings完整且stderr为空；selection/leaderboard/per-run archive SHA=`22b4b46f…f4ab8`/`0f8500b0…99930`/`ff85edf2…f973e`；独立审计最大差`4.44e−16`、audit SHA=`dbe33bc7…1bcfe`；52.1复制所需模型且无损evidence完成后，cleanup精确删除480个临时checkpoint并确认剩余0 | selected Raw-PCA/FMT F1=`.69743/.88757`，gain=`+.19014`；AP=`.74048/.94652`，gain=`+.20604`；10/10 datasets正、worst gain=`+.05272`。Channel/halfcylinder选择gamma`.075`，F22选`.01`，其余family为control；相对exact control仅提高FMT F1`+.00045`、gain`+.00036` | **不支持低gamma边界成为新的整体赢家**：gain距`.195`差`.00486`，FMT F1距`.893`差`.00543`，且整体弱于44.1/48.1。局部family赢家仍按预注册规则进入52.1比较；confirmation保持关闭。 |
| Verify_Task3_ResidualDropoutHigh_51.1（已完成，development） | 2026-08-30–31 | Task3-3D高Dropout边界跟进 | 38.1中F22与Boeing选择原搜索上界dropout`.50`，因此比较exact zero-dropout control与`.50/.55/.60/.65/.70/.75/.80`；冻结22.1 family feature、5.2 development population、同一网络/optimizer/split/seeds40–42/预算。8 candidates×10 datasets×3 seeds×2 arms=`480` trainings | `config/Verify_Task3_ResidualDropoutHigh_51.1.yaml`, `tests/test_task3_residual_dropout_high_51_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_dropout_high_51.1_*.sh` | jobs `51058832`,`51058833`,`51058835`,`51072260`,`51058838`全部exit 0；80/80 children、480/480 trainings完整；selection/per-run archive SHA `97fea5a2…1404`/`89f6d257…f5f5`；480 checkpoints在52.1复制后精确删除且剩余0 | Raw-PCA/FMT F1=`.69582/.88778`，gain=`+.19196`；AP=`.73974/.94577`，gain=`+.20603`。F22/Tangaroa/Boeing分别选择dropout`.60/.55/.50`，其余family保留control；10/10 datasets正 | **未达到**gain`>=+.195`或absolute FMT F1`>=.893`，不支持高Dropout成为整体新赢家；局部family结果进入52.1，confirmation始终关闭。 |
| Verify_Task3_ParameterEMA_40.1（已完成，development） | 2026-08-30 | Task3-3D residual参数指数移动平均单因素搜索 | 在33.1完整结束后、且不读取34.1–39.1 partial metrics的前提下预注册；冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head及paired seeds40–42。比较exact no-EMA control与decay `{.5,.8,.9,.95,.98,.99,.995,.999}`；EMA仅跟踪可训练residual参数，validation/early stopping/checkpoint/inference使用EMA参数，optimizer继续更新live参数，冻结Raw backbone不参与。每cell两臂同decay、初始化、batch、optimizer、split和预算。9 candidates×10 datasets×3 seeds×2 arms=`540` trainings | `experiments/Verify_Task3_FMTResidual.py`, `config/Verify_Task3_ParameterEMA_40.1.yaml`, `tests/test_task3_parameter_ema_40_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_parameter_ema_40.1_*.sh` | implementation commit `749b74a`；Ibex jobs `51016376`,`51016379`,`51016380`全部exit 0；90/90 children、540/540 trainings完整且GPU/selector stderr为空；selection/leaderboard/per-run archive SHA `f5d4952f…1c1d`/`4c8f3c94…abe5`/`0fc2bb50…f243`；从540份CSV独立重做Cartesian product、绝对FMT guard、family排序与macro，最大误差`5.6e−17`，audit SHA `0fd76781…e29`；临时`.pt`已不存在（保留540个空checkpoint目录） | 逐family EMA为channel `.99`、halfcylinder `.80`、Tangaroa `.80`、deltaWing `.95`；Boeing、F22、Smoke保留no-EMA。selected Raw-PCA/FMT F1=`.69734/.88743`，gain=`+.19009`；AP gain=`+.20574`；10/10 datasets正。no-EMA control F1=`.69782/.88708`，gain=`+.18926` | EMA仅使gain提高`+.00083`、absolute FMT F1提高`+.00035`，**未达到**gain`>=+.195`或absolute FMT F1`>=.893`，并弱于38.1 Dropout的`+.19648`；不单独打开confirmation，但作为44.1/48.1预注册组合候选保留。 |
| Verify_Task3_LabelSmoothing_41.1（已完成，development） | 2026-08-30 | Task3-3D binary label smoothing单因素搜索 | 在34.1完整结束后、且不读取35.1–40.1 partial metrics的前提下预注册；冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head及paired seeds40–42。比较exact hard-label control与epsilon `{.001,.0025,.005,.01,.02,.05,.10,.20}`；仅训练损失使用`(1−epsilon)y+epsilon/2`，validation/evaluation保留原始hard IVD labels。每cell两臂同epsilon、初始化、class weights、optimizer、split和预算。9 candidates×10 datasets×3 seeds×2 arms=`540` trainings | `experiments/Verify_Task3_FMTResidual.py`, `config/Verify_Task3_LabelSmoothing_41.1.yaml`, `tests/test_task3_label_smoothing_41_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_label_smoothing_41.1_*.sh`, `experiments/Audit_Task3_ParameterSearch.py` | implementation commit `352497c`；Ibex jobs `51016609`,`51016612`,`51016613`,`51057902`全部exit 0；90/90 children、540/540 trainings完整且GPU/selector/archive stderr为空；selection/leaderboard/per-run archive SHA `db7f3e86…2e07`/`49f2f20e…b34b`/`891aa774…d339`；独立重做Cartesian product、guard、family选择与macro，最大误差`2.22e−16`，audit SHA `ebfb5299…e0b`；540个临时checkpoint已删除且剩余0 | 仅Channel选择epsilon `.001`，其余6个family保留hard-label control。selected Raw-PCA/FMT F1=`.69783/.88728`，gain=`+.18945`；AP gain=`+.20618`；10/10 datasets正。exact control F1=`.69778/.88711`，gain=`+.18934`；选择仅使gain提高`+.00011`、absolute FMT F1提高`+.00017` | **未达到**gain`>=+.195`或absolute FMT F1`>=.893`，且弱于38.1 Dropout的`+.19648`；不单独打开confirmation，仅作为44.1预注册组合输入保留。 |
| Verify_Task3_AdamEpsilon_42.1（已完成，development） | 2026-08-30 | Task3-3D AdamW denominator epsilon单因素搜索 | 冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head及paired seeds40–42。比较exact PyTorch-default control与epsilon `{1e−12,1e−10,1e−9,1e−7,1e−6,1e−5,1e−4}`；每cell两臂同epsilon、初始化、AdamW其他参数、batch、split和预算。8 candidates×10 datasets×3 seeds×2 arms=`480` trainings | `experiments/Verify_Task3_FMTResidual.py`, `config/Verify_Task3_AdamEpsilon_42.1.yaml`, `docs/Verify_experiments.md` | jobs `51016835`,`51016836`,`51016837`,`51057984`全部exit 0且stderr为空；80/80 children、480/480 trainings完整；independent audit最大差`2.22e−16`，audit SHA `dac2f527…adbd5`；per-run archive SHA `73d563ef…32f`；480 checkpoints已删除且剩余0 | family选择：Channel/F22=`1e−4`，DeltaWing/Tangaroa=`1e−10`，halfcylinder=`1e−7`，Boeing/Smoke=control；Raw-PCA/FMT F1=`.696300/.887202`，gain=`+.190902`；AP gain=`+.206754`；10/10 datasets正增益 | 相对default control只增加`+.001065` gain和`+.000126` absolute FMT F1；未达到`.195/.893`联合目标且弱于Dropout，confirmation保持关闭。 |
| Verify_Task3_CosineMinLR_43.1（已完成，development） | 2026-08-30 | Task3-3D cosine learning-rate terminal ratio单因素搜索 | 在不读取35.1–42.1 partial metrics的前提下预注册；冻结22.1逐family anchored feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head及paired seeds40–42。比较strict constant-learning-rate control与cosine terminal/base ratio `{0,.001,.01,.025,.05,.10,.25,.50}`；每cell两臂同schedule、初始化、optimizer、batch、split和预算。9 candidates×10 datasets×3 seeds×2 arms=`540` trainings | `config/Verify_Task3_CosineMinLR_43.1.yaml`, `tests/test_task3_cosine_min_lr_43_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_cosine_min_lr_43.1_*.sh`, `experiments/Audit_Task3_ParameterSearch.py` | implementation commit `6679080`；Ibex jobs `51017332`,`51017334`,`51017338`,`51058184`全部exit 0且stderr为空；90/90 mappings、540/540 paired trainings完整；selection/leaderboard/preflight/per-run archive SHA `403d5207…d0e6`/`b6c2a0c8…0920`/`5dcb69fb…9929`/`4d02dfd9…0e24`；独立审计全部family选择与macro，最大误差`2.22e−16`，audit SHA `2d15f307…b4473`；540个临时checkpoint已删除且剩余0 | Channel/halfcylinder/Tangaroa分别选择terminal ratio`.50/.05/.01`，其余4个family保留constant control。Raw-PCA/FMT F1=`.697864/.887322`，gain=`+.189459`；AP gain=`+.206467`；10/10 datasets正，最小gain=`+.047216`。constant control F1 gain=`+.189125`、FMT F1=`.887076`，选择仅增加`+.000333` gain和`+.000247` absolute FMT F1 | **未达到**gain`>=+.195`或absolute FMT F1`>=.893`，并弱于38.1 Dropout的`+.196479`；不单独打开confirmation，仅作为44.1/48.1预注册组合输入保留。 |
| Verify_Task3_SafeFactorCombination_44.1（已完成，development） | 2026-08-30 | Task3-3D已完成单因素赢家安全组合搜索 | 在35.1完整结束且未读取36.1–43.1任何partial/final metric时冻结。使用22.1逐family anchored feature，并从32.1–34.1及36.1–43.1各自confirmation-closed selector读取逐family赢家；35.1因所有family选择exact no-warmup control而明确排除。固定28 candidates：feature control、11个单因素、alpha与其余10因素的两两组合、optimizer/loss-regularization/stability stacks、all-non-scheduler、all及all-without-alpha。若上游后来选择exact control，对应candidate仍保留，不做结果后删减。28×10 datasets×3 seeds×2 arms=`1680` trainings | `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_SafeFactorCombination_44.1.yaml`, `tests/test_task3_safe_factor_combination_44_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_safe_factor_combination_44.1_*.sh`, `experiments/Audit_Task3_ParameterSearch.py` | implementation commit `db1c353`；array/selector/evidence jobs `51018920`,`51018923`,`51071240`全部exit 0；280/280 children与1680/1680 trainings完整；selection/leaderboard/archive SHA `48dfa466…5f12`/`b9e96f9e…baad`/`ac55850e…82e4`；独立审计最大差`1.11e−16`，audit SHA `f1fcff66…f0ac`；1680个checkpoint完整保留供52.1 | Raw-PCA/FMT dataset-macro F1=`.68569/.89046`，gain=`+.20476`；AP=`.72987/.94905`，gain=`+.21918`；10/10 datasets正、worst gain=`+.05385`。exact control F1=`.69723/.88712`、gain=`+.18990`；选择使FMT F1提高`+.00333`、gain提高`+.01487`。达到`+.195`增益目标，但absolute FMT F1距`.893`仍差`.00254`，联合目标未达到；`confirmation_opened=false` | **当前最高development F1增益与绝对FMT F1。**相对38.1，FMT F1提高`.00210`且gain提高`.00829`，所以不只是Raw变弱；相对48.1也分别高`.00028/.00076`。逐family赢家进入52.1五路portfolio，必须经7.2全新空间population确认。 |
| Verify_Task3_HeadAlphaClipCombination_45.1（已完成，development） | 2026-08-30–31 | Task3-3D head×training-alpha×gradient-clipping完整组合 | 在31.2、32.1、33.1均完整结束后，且不读取36.1–44.1任何partial metric时冻结；补足44.1未包含31.2 head winner的明确缺口。使用22.1逐family anchored feature；对head、training alpha、clipping做完整`2^3=8`候选因子，exact anchored-feature control为第一个候选。8×10 datasets×3 seeds×2 arms=`480` trainings | `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_HeadAlphaClipCombination_45.1.yaml`, `tests/test_task3_head_alpha_clip_combination_45_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_head_alpha_clip_combination_45.1_*.sh`, `experiments/Audit_Task3_ParameterSearch.py` | implementation commit `3fe1aee`；jobs `51020733`,`51020778`,`51071279`全部exit 0；80/80 children与480/480 trainings完整；selection/leaderboard/archive SHA `cf80d0d6…43d9`/`19606af4…fa8`/`08e3788f…1bcd`；独立审计最大差`0.0`，audit SHA `58947d24…0b2`；480个checkpoint完整保留供52.1 | Raw-PCA/FMT dataset-macro F1=`.68795/.88973`，gain=`+.20178`；AP=`.73148/.94815`，gain=`+.21667`；10/10 datasets正、worst gain=`+.05272`。exact control F1=`.69724/.88706`、gain=`+.18983`。达到`+.195`增益目标，但absolute FMT F1距`.893`差`.00327`，联合目标未达到；`confirmation_opened=false` | **支持head、alpha与clipping组合能提高配对增益，但不刷新44.1的绝对FMT或增益记录。**逐family赢家进入52.1五路portfolio；只有7.2全新空间population可以形成论文级结论。 |
| Verify_Task3_AnchoredFeatureSpatialReplay_46.1（完成，已公开development回放） | 2026-08-30 | Task3-3D 22.1冻结表示的第三空间population回放 | 12.2已经打开第三空间population并使其成为development数据；46.1在该已公开population上原样加载22.1逐family赢家、seeds40–41两臂checkpoint、阈值、alpha和Raw-PCA transform，不训练、不调参、不生成checkpoint。10 datasets×2 seeds×2 arms=`40`次冻结推理 | `experiments/Replay_Task3_AnchoredFeatureSpatial_46_1.py`, `config/Verify_Task3_AnchoredFeatureSpatialReplay_46.1.yaml`, `tests/test_task3_anchored_feature_spatial_replay_46_1.py`, `docs/Verify_experiments.md` | implementation/CPU-route commits `b80bf1e`,`671f979`；Ibex jobs `51025812`,`51026592`,`51026610`均exit 0；preflight/per-run/summary SHA `c2be783b…d3be`,`26d475a8…ef26`,`23e777a1…2b1` | Raw-PCA/FMT F1 `.69586/.86586`，dataset-macro增益`+.17000`；AP `.75339/.94038`，增益`+.18699`；family-macro F1/AP增益`+.18231/+.20229`；10/10 datasets、7/7 families均为正，最小dataset增益`+.05186`；Re160由12.2旧方法的负增益修复为`+.13876` | 达到预注册描述性目标`>=+.15`，支持22.1表示在第三空间population上保持普遍优势；相对22.1 development增益下降`.01690`。因该population此前已公开，46.1不得冒充最终测试证据；下一步冻结22.1方法并在全新第四空间population上作唯一final confirmation。 |
| mainExp_Task3_3D_6.1（完成，历史第四population确认；当前主表为8.1） | 2026-08-30 | Task3-3D 22.1冻结方法的第四空间population最终确认 | 原样冻结22.1逐family feature、seeds40–41两臂checkpoint、阈值、alpha、Raw normalization与train-only Raw-PCA transform；10 datasets×2 seeds×2 arms=`40`次推理，不训练、不调参。物理时刻及积分设置不变，仅以预注册SHA-256→Halton规则生成第四seed-grid phase `[.021484375,−.3422496571,.0328]`；所有模型及输入哈希均在生成primitive/IVD前写入recipe manifest | `experiments/Confirm_Task3_AnchoredFeature_6_1.py`, `experiments/Build_Task3_AnchoredFeature_Confirmation_6_1.py`, `experiments/Prepare_Task3_AnchoredFeature_SourceManifest_6_1.py`, `config/mainExp_Task3_3D_6.1.yaml`, `tests/test_mainexp_task3_3d_6_1.py`, `docs/mainExp_Task3_3D.md` | implementation commit `f7c567b`；Ibex jobs `51028047`,`51028079`,`51028080`,`51028082`,`51028083`,`51028084`,`51028087`,`51028091`全部exit 0；frozen recipe/evaluation-preflight/per-run/summary SHA `0469dcc8…1bd5`,`1c1232f7…e7bf`,`bc5a61b9…ef591`,`f46a307a…fb756`；本地独立复算40/40唯一记录、10 datasets、40 models及全部macro指标误差`<1e−12` | Raw-PCA/FMT F1 `.69480/.86546`，dataset-macro增益`+.17066`；AP `.75263/.94465`，增益`+.19202`；family-macro F1/AP增益`+.18148/+.20598`；10/10 datasets、7/7 families均正，最小dataset F1增益Smoke `+.04851` | **支持并达到核心结论**：FMT对当前10个3D数据条目的监督IVD-p95二分类带来平均`+17.07`个百分点F1增益，超过主目标`+.15`并达到扩展目标`+.17`。这是此前未打开的第四空间population，确认数据未参与模型、feature、阈值或alpha选择。 |
| Verify_Task3_TrainingHorizon_47.1（已完成，development） | 2026-08-30 | Task3-3D共同训练轮数单因素搜索 | 冻结22.1逐family feature、5.2 development population/split、2层×64 LayerNorm+GELU residual head及paired seeds40–42；比较exact 100-epoch control与`{20,40,60,80,125,150,200,300}`。每cell两臂同epoch、初始化、batch、optimizer、loss、split和decision protocol；9 candidates×10 datasets×3 seeds×2 arms=`540`次训练，confirmation始终关闭 | `config/Verify_Task3_TrainingHorizon_47.1.yaml`, `tests/test_task3_training_horizon_47_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_training_horizon_47.1_*.sh`, `experiments/Audit_Task3_ParameterSearch.py` | implementation commit `f7c567b`；jobs `51028066`,`51028067`,`51028071`,`51065692`,`51065693`全部exit 0；初次归档`51058293`失败记录保留；90/90 mappings、540/540 paired trainings完整；selection/leaderboard/per-run archive SHA `79706980…cda9`/`1d0061a7…7e3`/`aca0ab63…5c250`；独立审计540份CSV与selector最大差`2.22e−16`，audit SHA `01e49d6a…fede`；540个临时checkpoint已删除且剩余0 | 逐family epoch为Channel/Tangaroa/F22/Boeing `100`、halfcylinder/Smoke `40`、deltaWing `80`。Raw-PCA/FMT dataset-macro F1=`.69527/.88709`，gain=`+.19182`；AP=`.73770/.94557`，gain=`+.20787`；10/10 datasets均正，最小gain=`+.05272`；`confirmation_opened=false` | **训练时长不是当前主要增益来源。**相对100-epoch control，选择仅把gain从`+.18923`提高到`+.19182`，absolute FMT F1仅`.88708→.88709`；未达到`.195/.893`且弱于38.1 Dropout的`+.19648/.88836`。只作为48.1完整训练栈组合的预注册输入，不单独打开confirmation。 |
| Verify_Task2_LatentBottleneck_5.1（已完成，development） | 2026-08-30 | Task2-3D共同latent dimension单因素搜索 | 每physical family冻结Task2-4.1已选FMT block、hidden layers、KL weight、学习率、optimizer steps和KMeans协议；只扫描共同latent dimension`{1,2,3,4,6,8,12,16,24,32,48,64}`。Raw/FMT逐cell使用相同VAE设置、seeds83–85、split和聚类协议；12×10 datasets×3 seeds×2 arms=`720`次训练；只打开development ordinal0–7 | `experiments/Search_Task2_LatentBottleneck_5_1.py`, `config/Verify_Task2_LatentBottleneck_5.1.yaml`, `tests/test_task2_latent_bottleneck_5_1.py`, `docs/Verify_experiments.md` | 初版失败链`51029073`,`51029104`,`51029108`保留；修复链`51032357`,`51032757`,`51032779`全部完成。selection/leaderboard SHA `cf1c546f…e3dc`/`4326a268…9b8`，内部结果SHA `ca6335b4…9dc`；120/120 children、720/720配对训练成功，ordinal8–9和confirmation未打开 | 逐family赢家latent为channel 8、halfcylinder 64、Tangaroa 1、deltaWing 12、F22 1、Boeing 6、Smoke 24。Raw/FMT dataset-macro F1=`.33703/.63498`，gain=`+.29794`，10/10 datasets正、worst gain=`+.08599`；control为`.43103/.64502`、gain`+.21399` | **支持development上的同一VAE窄瓶颈结论并超过`+.22`目标。**但赢家FMT绝对F1比control低`.01005`，增益扩大部分来自Raw变弱；不得替代4.1论文结果。赢家只作为预先固定输入进入全新第五空间population确认5.2。 |
| mainExp_Task2_3D_5.2（已完成，独立第五空间population） | 2026-08-30 | Task2-3D 5.1 latent赢家的独立确认 | 冻结5.1 selection SHA、逐family latent、FMT block及全部VAE/KMeans设置；development ordinal0–7训练、8–9仅校准cluster ID；新paired seeds9090–9094。主方法为selected latent，4.1配方control只作同population诊断，不是历史4.1实跑结果。10 datasets×2 recipes×2 arms×5 seeds=`200`次训练 | `experiments/Confirm_Task2_LatentBottleneck_5_2.py`, `experiments/Build_Task2_LatentConfirmation_5_2.py`, `experiments/Prepare_Task2_LatentConfirmation_SourceManifest_5_2.py`, `config/mainExp_Task2_3D_5.2.yaml`, `tests/test_mainexp_task2_3d_5_2.py`, `docs/mainExp_Task2_3D_5.2.md` | Ibex链`51036980–51036987`全部完成；selection SHA `cf1c546f…e3dc`；独立Halton phase`[-.4833984375,−.0281207133,.4344]`；200/200结果完整、不保存checkpoint；summary/per-run SHA `739b48ea…3095`/`20d9de15…111f`，独立复算误差`<1e-12` | selected Raw/FMT dataset-macro F1=`.39227/.63143`，gain=`+.23916`，10/10 datasets、7/7 families、5/5 seeds均正，worst dataset=`+.05812`，family-macro gain=`+.26598`。diagnostic control=`.48594/.65216`，gain=`+.16622`，9/10 datasets正 | **支持Task2核心结论，并超过预注册`+.15`主目标和`+.22`扩展目标。**selected相对control多获得`+.07293`增益，但绝对FMT F1低`.02074`；因此结论严格限定为“同一VAE下FMT相对Raw提升更大”，不声称selected latent使FMT绝对质量最高。 |
| Verify_Task3_HeadFullStackCombination_48.1（已完成，development） | 2026-08-30 | Task3-3D head×完整训练栈组合搜索 | 在38.1完整结束后、读取39.1–47.1任何final metric前预注册；冻结22.1 feature与5.2 development协议，把head、alpha、clipping、Adam betas、batch、positive weight、Dropout、focal、EMA、label smoothing、Adam epsilon、cosine和training horizon的逐family selector组成16个固定候选。每cell两臂同recipe、初始化、split、seed和预算；16×10 datasets×3 seeds×2 arms=`960` trainings，confirmation关闭 | `config/Verify_Task3_HeadFullStackCombination_48.1.yaml`, `tests/test_task3_head_full_stack_combination_48_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_head_full_stack_combination_48.1_*.sh`, `experiments/Audit_Task3_ParameterSearch.py` | implementation commit `eb2cd44`；array/selector/evidence jobs `51041338`,`51041339`,`51071147`全部exit 0；160/160 children与960/960 trainings完整；selection/leaderboard/archive SHA `5693e802…a983`/`0660706b…ac2e`/`aebbaa7a…91c0`；独立审计最大差`3.33e−16`，audit SHA `0ec09b9a…9c0e`；960个checkpoint完整保留供52.1 | Raw-PCA/FMT dataset-macro F1=`.68617/.89018`，gain=`+.20401`；AP=`.72844/.94904`，gain=`+.22060`；10/10 datasets正、worst gain=`+.05385`。exact control F1=`.69741/.88708`、gain=`+.18966`；选择使FMT F1提高`+.00310`、gain提高`+.01435`。达到`+.20`增益目标，但absolute FMT F1距`.893`仍差`.00282`，联合目标未达到；`confirmation_opened=false` | **刷新development增益和绝对FMT双重记录。**相对38.1，FMT F1提高`.00182`且gain提高`.00753`，所以不只是Raw变弱；但仍不能把development结果当论文最终结果。逐family赢家进入52.1五路portfolio，必须经7.2全新空间population确认。 |
| Verify_Task3_FinalPortfolio_49.1（已完成，development；由52.1扩展替代） | 2026-08-30–31 | Task3-3D三路开发赢家的training-free逐family择优 | 不训练、不读取confirmation；只在44.1、45.1、48.1均完整结束后，按预先固定的dataset-macro paired F1 gain及六个tie-breakers，为7个physical family分别选择已通过absolute-FMT guard的赢家。冻结selection文件及seed40–41两臂共40个result/checkpoint SHA-256；seed42只参与development选择 | `experiments/Select_Task3_FinalPortfolio_49_1.py`, `config/Verify_Task3_FinalPortfolio_49.1.yaml`, `tests/test_task3_final_portfolio_49_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_final_portfolio_49.1_select.sh` | implementation commit `6d47151`；job `51043495` exit 0、stderr为空；portfolio SHA `c20586d8…498f`；40个seed40–41 model/result/checkpoint身份冻结；旧7.1入口保持user hold | Raw-PCA/FMT dataset-macro F1=`.68470/.89022`，gain=`+.20552`；AP=`.72858/.94872`，gain=`+.22014`；10/10 datasets正。通过`+.20`增益目标，但absolute FMT F1未达`.893`，联合目标失败；`confirmation_opened=false` | 三路组合支持development增益可超过20个百分点，但不能替代独立确认。五路52.1额外纳入50.1/51.1并扩展替代49.1，后续只运行52.1→7.2。 |
| mainExp_Task3_3D_7.1（已提交，待49.1冻结） | 2026-08-30 | Task3-3D最终调参栈的第五空间population独立确认 | 冻结49.1逐family赢家、seed40–41两臂checkpoint、阈值、residual scale、Raw normalization与train-only Raw-PCA transform；确认阶段不训练、不选feature/threshold/scale。10 datasets×4全新空间切片×2 seeds×2 arms；物理时刻与IVD-p95不变，仅用预注册SHA-256→Halton index 187生成新phase `[.36328125,.1378600823,−.0024]` | `experiments/Confirm_Task3_FinalTuned_7_1.py`, `experiments/Build_Task3_FinalTuned_Confirmation_7_1.py`, `experiments/Prepare_Task3_FinalTuned_SourceManifest_7_1.py`, `config/mainExp_Task3_3D_7.1*.yaml`, `tests/test_mainexp_task3_3d_7_1.py`, `docs/mainExp_Task3_3D.md`, `ibex_bash/mainexp_task3_3d_7.1_*.sh` | commit/archive同上；本地5/5、远端静态4/4 contracts、Python编译及8个脚本语法通过；phase-key SHA `73f9e8ba…0ba3`；Ibex依赖链`51043506→51043510→51043512→51043525→51043526→51043534→51043535→51043546`逐项核实 | 尚无性能结果；primitive/label生成严格晚于49.1 selection与40个模型哈希冻结 | 主目标dataset-macro F1 gain`>=+.15`，扩展目标`>=+.20`；无论是否超过6.1的`+.17066`，均保留并报告。 |
| Verify_Task3_AdaptivePortfolio_52.1（已完成，development） | 2026-08-30–31 | Task3-3D五路开发赢家的training-free逐family择优 | 不训练、不读取confirmation；从44.1、45.1、48.1、50.1、51.1按预注册排序为7个physical family选择通过absolute-FMT guard的赢家，并复制校验seed40–41两臂共40个result/checkpoint | `experiments/Select_Task3_AdaptivePortfolio_52_1.py`, `experiments/Audit_Task3_AdaptivePortfolio.py`, `config/Verify_Task3_AdaptivePortfolio_52.1.yaml`, `docs/Verify_experiments.md` | jobs `51059320`,`51073078` exit 0；portfolio/audit SHA `257893ee…4393`/`289de6cd…600`；80/80冻结文件哈希和配对容量通过；独立重算最大差`2.22e−16`；两个早期来源身份失败job保留且未读性能 | development Raw-PCA/FMT F1=`.68470/.89022`，gain=`+.20552`；AP=`.72858/.94872`，gain=`+.22014`；10/10 datasets正。Boeing/Smoke来自48.1，Tangaroa来自45.1，其余来自44.1 | **通过`+.20`development增益目标但absolute FMT F1未达`.893`。**不得作为论文最终结果；其冻结赢家已在fresh 7.2确认。 |
| mainExp_Task3_3D_7.2（已完成，当前方法的第六population复现） | 2026-08-30–31 | Task3-3D adaptive portfolio的第六空间population独立确认 | 冻结52.1逐family赢家及seed40–41两臂40个模型；confirmation不训练、不选择。保持10 datasets、物理时刻、pathline和IVD-p95，仅以预注册SHA-256→Halton index678生成phase `[-.1044921875,−.365569273,.11632]` | `experiments/Confirm_Task3_AdaptiveTuned_7_2.py`, `experiments/Audit_Task3_AdaptiveTuned_7_2.py`, `experiments/Build_Task3_AdaptiveTuned_Confirmation_7_2.py`, `experiments/Prepare_Task3_AdaptiveTuned_SourceManifest_7_2.py`, `config/mainExp_Task3_3D_7.2*.yaml`, `docs/mainExp_Task3_3D.md` | 完整链`51060409→51060410→51060411→51060464→51060466→51060467→51060468→51060469→51073430`全部exit 0；summary/per-run/audit SHA `e89751f9…5248`/`7c1ecc37…3901`/`cdadc4d6…6350`；独立重算最大差`2.78e−17` | Raw-PCA/FMT F1=`.68584/.86106`，dataset-macro gain=`+.17522`；AP=`.73624/.93770`，gain=`+.20146`；family-macro F1/AP gain=`+.18926/+.22090`；10/10 datasets、7/7 families、2/2 seeds正；worst DeltaWing-LBM=`+.03857` | **支持Task3核心结论并通过预注册`+.15`目标；未达到`+.20`扩展目标。**这是此前未使用的第六空间population，不能被development搜索回写；与6.1并列证明稳定约17个百分点增益。 |
| Verify_Task3_AuxiliaryDropout_53.1（已完成，development） | 2026-08-31 | Task3-3D投影后辅助表示Dropout单因素搜索 | 冻结22.1逐family anchored feature、5.2 development population/split、2层×64 residual head、optimizer、预算及paired seeds40–42；仅在FMT/Raw-PCA同宽辅助投影后施加同概率Dropout。比较exact `p=0`与`{.025,.05,.10,.15,.20,.30,.40,.50,.60,.70}`；11×10×3×2=`660` trainings | `FMT_Utils/PathlineClassifier_3D.py`, `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Audit_Task3_ParameterSearch.py`, `config/Verify_Task3_AuxiliaryDropout_53.1.yaml`, `docs/Verify_experiments.md` | commit `6ab3c98`；jobs `51069690`,`51069691`,`51069692`,`51072617`全部exit 0；110/110 children与660/660 trainings完整；selection/leaderboard/archive SHA `9fddfe44…d3ce`/`0ea3afc0…313`/`92dadd08…f75e`；独立审计最大差`1.11e−16`、来源哈希和配对容量全部通过 | Raw-PCA/FMT dataset-macro F1=`.69642/.88775`，gain=`+.19133`；AP=`.73957/.94615`，gain=`+.20658`；10/10 datasets正、worst gain=`+.05409`。control F1=`.69738/.88712`、gain=`+.18975`；Dropout只把FMT F1提高`+.00062`、gain提高`+.00158` | **可信的未达目标结果。**F1 gain未达`+.20`且absolute FMT F1未达`.893`；Boeing/channel/smoke保留`p=0`，其余family虽选非零Dropout，但54.1未选择任何53.1来源，故不支持辅助Dropout成为主改进。 |
| Verify_Task3_ExtendedPortfolio_54.1（已完成，development） | 2026-08-31 | Task3-3D六路开发赢家training-free逐family择优 | 在53.1产生任何性能前冻结；扩展52.1并加入53.1，保持同一guard和排序，0 trainings，复制并校验seed40–41两臂40 models/40 results | `experiments/Select_Task3_ExtendedPortfolio_54_1.py`, `experiments/Audit_Task3_AdaptivePortfolio.py`, `config/Verify_Task3_ExtendedPortfolio_54.1.yaml`, `docs/Verify_experiments.md` | commit `71313b0`；source gate/selector/audit jobs `51074578`,`51074606`,`51074616`均exit 0、stderr为空；portfolio SHA `9dee2647…128`；独立审计核验80/80冻结文件、最大差`2.22e−16` | Raw-PCA/FMT development F1=`.68470/.89022`，gain=`+.20552`；AP=`.72858/.94872`，gain=`+.22014`；10/10 datasets正。Boeing/Smoke来自48.1，Tangaroa来自45.1，其余来自44.1；53.1没有family胜出 | **达到`+.20`增益目标，但absolute FMT F1 `.89022<.893`，联合目标未达到。**这是development择优而非论文结果；冻结配方已送入8.1第七空间population。 |
| mainExp_Task3_3D_8.1（已完成，第七空间population） | 2026-08-31 | Task3-3D六源portfolio的第七空间population独立确认 | 在53.1产生指标前冻结；10 datasets、IVD-p95、物理时刻和积分协议不变，仅使用Halton index798的新phase `[-.0283203125,−.2155921353,.26992]`；40次冻结推理，无训练或选择。初次evaluate因54.1包遗漏residual所引用的Raw文件而在推理前失败；原始20个Raw checkpoint仍完整，故以SHA-256精确复制建立依赖闭包，不重训、不改模型 | `experiments/Build_Task3_ExtendedTuned_Confirmation_8_1.py`, `experiments/Confirm_Task3_ExtendedTuned_8_1.py`, `experiments/Audit_Task3_ExtendedTuned_8_1.py`, `experiments/Freeze_Task3_RawDependencyClosure_8_1.py`, `config/mainExp_Task3_3D_8.1*.yaml`, `docs/mainExp_Task3_3D.md` | 原生成链`51074813–51074841`成功；首次evaluate `51074843`的10/10 children因缺失可移植依赖而exit 1且0指标，旧summary/audit取消。修复commit `a38a102`；新链`51088068→51088083→51088109→51088124`全exit 0；closure/summary/per-run/audit SHA `2e5881bf…e59b5`/`0066299c…07bd`/`78d19ec3…bc52`/`eabc384a…edb3`，独立重算最大差`1.11e−16` | Raw-PCA/FMT F1=`.69131/.87136`，dataset-macro gain=`+.18005`；AP=`.74392/.94383`，gain=`+.19991`；family-macro F1/AP gain=`+.19440/+.21523`；10/10 datasets、7/7 families、2/2 seeds正；worst DeltaWing-LBM=`+.03633` | **支持Task3核心结论并通过预注册`+.15`目标；未达到`+.20`扩展目标。**相对第六population 7.2的`+.17522`，第七population为`+.18005`，两套独立空间确认一致支持约18个百分点增益。 |
| Verify_Task3_RawDependencyRebuild_8.1（已拒绝的基础设施后备方案） | 2026-08-31 | Task3-3D 8.1缺失Raw依赖的重建预案 | 原计划用冻结的3.2代码、cache、标签、split和seeds 40–41在两张V100上重建20个Raw checkpoint，并要求两次重建`state_dict`逐张量一致、validation指标误差`<=1e-12`且normalization与40个residual checkpoint完全一致；只修依赖，不训练或修改FMT residual | `experiments/Repair_Task3_RawDependencies_8_1.py`, `config/Verify_Task3_RawDependencyRebuild_8.1.yaml`, `tests/test_task3_raw_dependency_rebuild_8_1.py`, `ibex_bash/verify_task3_raw_dependency_rebuild_8.1_*.sh`, `docs/Audit_experiments.md` | CPU preflight job `51088368`于13:55:52开始、13:56:03以exit 1结束；严格门发现8.1的10个shard、`per_run.csv`和`summary.json`已由更安全的SHA-256精确复制链完成，因而在提交任何V100重建前主动拒绝。没有模型、指标或confirmation artifact被改写 | 无性能指标；精确复制依赖闭包`51088068→51088083→51088109→51088124`已先完成并独立审计通过 | **安全拒绝，不支持也不反对FMT性能。**旧诊断关于“cleanup删除原始Raw checkpoint”的判断错误；原始20个文件始终存在。保留该失败版本用于证明未以重训练结果替换已审计的8.1证据。 |
| Verify_Task23CoreGain_1.1（本地独立审计） | 2026-08-31 | Task2/Task3核心`>=+.15`目标的逐次记录交叉审计 | 不调用两个主实验的summary实现，直接读取Task2-5.2 selected recipe的100条Raw/FMT同VAE记录及Task3-6.1的40条Raw-PCA/FMT冻结推理记录；检查dataset×seed×arm唯一性，重算dataset/family/seed macro并与正式summary逐项比较 | `experiments/Audit_Task23_CoreGain.py`；输入`output/mainExp_Task2_3D_5.2_ibex/{per_run.csv,summary.json}`与`output/mainExp_Task3_3D_6.1_ibex/{per_run.csv,summary.json}` | Python编译和真实完整证据审计通过；Task2 summary/per-run SHA `739b48ea…3095`/`20d9de15…111f`，Task3为`f46a307a…b756`/`bc5a61b9…f591`；审计JSON SHA `909ee699…b4d1` | Task2 F1 gain=`+.2391553`，10/10 datasets、7/7 families、5/5 seeds正；Task3 F1/AP gain=`+.1706578/+.1920209`，10/10 datasets、7/7 families、2/2 seeds正；两任务最小F1 gain=`+.1706578`，与原summary最大差`2.50e−16` | **在该审计时点，独立证明Task2-5.2和Task3-6.1均已超过用户要求的多流场平均`+.15` F1增益。**正在运行的Task3搜索只负责尝试继续提高，不是用development结果替代现有确认。 |
| Verify_Task3_AuxiliaryLearningRate_55.1（完成并独立审计） | 2026-08-31 | Task3-3D辅助投影学习率单因素搜索 | 冻结22.1逐family feature、5.2 development population/split、2层×64 residual head、基础optimizer及paired seeds40–42；FMT与train-only Raw-PCA两臂共同改变辅助投影学习率倍率，downstream residual head保持基础学习率。扫描`{.05,.10,.25,.5,1,2,4,8,16}`，9×10×3×2=`540` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_AuxiliaryLearningRate_55.1.yaml`, `experiments/Audit_Task3_ParameterSearch.py`, `docs/Verify_experiments.md` | — | 90/90 array children与540/540 trainings均exit 0。逐family赢家得到Raw-PCA/FMT F1=`.69671/.88772`、gain=`+.19101`；Average Precision gain=`+.20629`；10/10 datasets为正，worst gain=`+.04950`。control gain=`+.18939`、FMT F1=`.88706`，故调学习率相对自身control仅提高gain `+.00162`、FMT F1 `+.00066`。独立审计最大差`2.22e-16`；本地从SHA匹配的540份逐次CSV重算得到相同选择与macro | **反对“单独调辅助投影学习率能改进当前最佳portfolio”。**未达预注册`+.200/.893`联合目标，且弱于54.1的`+.20552/.89022`。family赢家：channel/deltaWing/f22=`4×`，halfcylinder=`.25×`，tangaroa=`16×`，boeing/smoke保留`1×`。该负结果不削弱FMT相对Raw-PCA仍为正的结论。 |
| Verify_Task3_AuxiliaryLearningRatePortfolio_56.1（完成并独立审计） | 2026-08-31 | Task3-3D训练后training-free逐family择优 | 在55.1产生性能指标前冻结；只比较已审计54.1 portfolio与55.1赢家，不训练、不读取confirmation。复制seed40–41两臂共40 model/40 result，并独立重建选择、macro及80个文件哈希 | `experiments/Select_Task3_AuxiliaryLearningRatePortfolio_56_1.py`, `experiments/Audit_Task3_AuxiliaryLearningRatePortfolio_56_1.py`, `config/Verify_Task3_AuxiliaryLearningRatePortfolio_56.1.yaml`, `docs/Verify_experiments.md` | — | 7/7 families全部选择`current_portfolio`（54.1），故结果保持Raw-PCA/FMT F1=`.68470/.89022`、gain=`+.20552`，Average Precision gain=`+.22014`。40 results/40 checkpoints冻结且配对容量相同；独立审计最大差`0`，selection/audit SHA=`e9acac41…40d`/`d1eb8164…3e9`，stderr为空 | **确认55.1没有给54.1增加可保留配方。**精确值`+.2055168`略低于预注册四舍五入门槛`+.20552`，且FMT F1`.89022<.893`，所以`target_reached=false`、`joint_target_reached=false`。56.1审计后55.1的540个临时checkpoint已删除，逐次CSV与无模型归档保留。 |
| Verify_Task3_AuxiliaryWeightDecay_57.1（已预注册，等待56.1独立审计） | 2026-08-31 | Task3-3D辅助投影weight decay单因素搜索 | 在55.1/56.1无任何性能结果时冻结；56.1逐family完整配方作为control，仅把辅助投影weight decay相对downstream head乘以`{0,.01,.05,.10,.25,.50,1,2,4,10,100}`。两臂同倍率、参数组、初始化、split、seed和预算；11×10×3×2=`660` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_AuxiliaryWeightDecay_57.1.yaml`, `tests/test_task3_auxiliary_weight_decay_57_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_weight_decay_57.1_*.sh` | — | 尚无性能指标；57.1的9项契约测试及41项相关optimizer回归测试通过；config canonical SHA=`6e98ee03…7bfa`；入口必须严格晚于56.1独立审计 | **仅为预注册假设，无结果结论。**联合目标为dataset-macro F1 gain `>=+.207`且absolute FMT F1 `>=.893`；无论成功或失败均保留。57.1的checkpoint在后续training-free portfolio完成前暂存，逐次CSV将无损归档。 |
| Verify_Task3_AuxiliaryRegularizationPortfolio_58.1（已预注册，等待57.1独立审计） | 2026-08-31 | Task3-3D辅助投影优化器搜索后的training-free逐family择优 | 在57.1产生任何性能指标前冻结；只比较已独立审计的56.1 portfolio与57.1 weight-decay赢家，沿用相同zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，再由不导入58.1 selector的审计器重建选择与macro并核验80个文件，之后才允许删除57.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryRegularizationPortfolio_58_1.py`, `experiments/Audit_Task3_AuxiliaryRegularizationPortfolio_58_1.py`, `config/Verify_Task3_AuxiliaryRegularizationPortfolio_58.1.yaml`, `tests/test_task3_auxiliary_regularization_portfolio_58_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_regularization_portfolio_58.1_*.sh` | — | 尚无性能指标；56.1 config canonical SHA=`c62b3925…0572`、57.1=`6e98ee03…7bfa`、58.1=`693284e5…db23`；source-identity阶段只能读取两个配置且必须报告`performance_artifacts_read=false`；本地58.1与56.1共13项选择/审计测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.207`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryBeta2_59.1（已预注册，等待58.1独立审计） | 2026-08-31 | Task3-3D辅助投影Adam二阶矩参数单因素搜索 | 在57.1/58.1无任何性能结果时冻结；58.1逐family完整配方作为exact control，仅把辅助投影Adam `beta2`改为`{0,.5,.8,.9,.95,.98,.99,.995,.999,.9999}`，并保留无局部覆盖的来源control。辅助投影继承来源`beta1`，downstream residual head保留来源两项beta；两臂同候选、参数组、初始化、split、seed和预算。11×10×3×2=`660` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_AuxiliaryBeta2_59.1.yaml`, `tests/test_task3_auxiliary_beta2_59_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_beta2_59.1_*.sh` | — | 尚无性能指标；34.1是全局Adam beta搜索，本实验只改变辅助投影beta2，二者不是重复实验；59.1 config canonical SHA=`2579e0de…6d6d`，入口必须严格晚于58.1独立审计 | **仅为预注册假设，无结果结论。**联合目标为dataset-macro F1 gain `>=+.208`且absolute FMT F1 `>=.893`；未达目标也必须保留。不得读取未完成数组的partial指标，checkpoint须等后续training-free portfolio冻结后再清理。 |
| Verify_Task3_AuxiliaryBeta2Portfolio_60.1（已预注册，等待59.1独立审计） | 2026-08-31 | Task3-3D辅助投影beta2搜索后的training-free逐family择优 | 在59.1产生任何性能指标前冻结；只比较已独立审计的58.1 portfolio与59.1 beta2赢家，沿用相同zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，再由不导入60.1 selector的审计器重建选择与macro并核验80个文件，之后才允许删除59.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryBeta2Portfolio_60_1.py`, `experiments/Audit_Task3_AuxiliaryBeta2Portfolio_60_1.py`, `config/Verify_Task3_AuxiliaryBeta2Portfolio_60.1.yaml`, `tests/test_task3_auxiliary_beta2_portfolio_60_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_beta2_portfolio_60.1_*.sh` | — | 尚无性能指标；60.1 config canonical SHA=`5f853c3c…ddb4`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；本地28项59.1/60.1/portfolio相关测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.208`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryBeta1_61.1（已预注册，等待60.1独立审计） | 2026-08-31 | Task3-3D辅助投影Adam一阶矩参数单因素搜索 | 在59.1/60.1无任何性能结果时冻结；60.1逐family完整配方作为exact control，仅把辅助投影Adam `beta1`改为`{0,.25,.5,.7,.8,.85,.9,.95,.98,.99}`，并保留无局部覆盖的来源control。辅助投影保留来源`beta2`，downstream residual head保留来源两项beta；两臂同候选、参数组、初始化、split、seed和预算。11×10×3×2=`660` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_AuxiliaryBeta1_61.1.yaml`, `tests/test_task3_auxiliary_beta1_61_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_beta1_61.1_*.sh` | — | 尚无性能指标；34.1是全局Adam beta搜索，本实验只改变辅助投影beta1；61.1 config canonical SHA=`4d457081…03ff`，入口必须严格晚于60.1独立审计 | **仅为预注册假设，无结果结论。**联合目标为dataset-macro F1 gain `>=+.209`且absolute FMT F1 `>=.893`；未达目标也必须保留，不得读取未完成数组的partial指标。 |
| Verify_Task3_AuxiliaryBeta1Portfolio_62.1（已预注册，等待61.1独立审计） | 2026-08-31 | Task3-3D辅助投影beta1搜索后的training-free逐family择优 | 在61.1产生任何性能指标前冻结；只比较已独立审计的60.1 portfolio与61.1 beta1赢家，沿用相同zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，再由不导入62.1 selector的审计器重建选择与macro并核验80个文件，之后才允许删除61.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryBeta1Portfolio_62_1.py`, `experiments/Audit_Task3_AuxiliaryBeta1Portfolio_62_1.py`, `config/Verify_Task3_AuxiliaryBeta1Portfolio_62.1.yaml`, `tests/test_task3_auxiliary_beta1_portfolio_62_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_beta1_portfolio_62.1_*.sh` | — | 尚无性能指标；62.1 config canonical SHA=`b29dfd4d…4805`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；本地32项相关测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.209`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryEpsilon_63.1（已预注册，等待62.1独立审计） | 2026-08-31 | Task3-3D辅助投影Adam分母epsilon单因素搜索 | 在57.1–62.1无任何性能结果时冻结；62.1逐family完整配方作为exact control，仅把辅助投影Adam epsilon改为`{1e-12,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}`，并保留无局部覆盖的来源control。downstream residual head保留来源epsilon；FMT/Raw-PCA两臂同候选、参数组、初始化、split、seed和预算。11×10×3×2=`660` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_AuxiliaryEpsilon_63.1.yaml`, `tests/test_task3_auxiliary_epsilon_63_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_epsilon_63.1_*.sh` | — | 尚无性能指标；42.1改变完整optimizer epsilon，本实验只改变辅助投影epsilon；63.1 config canonical SHA=`f9d2cf07…5833`；39项辅助optimizer回归测试通过，入口必须严格晚于62.1独立审计 | **仅为预注册假设，无结果结论。**联合目标为dataset-macro F1 gain `>=+.210`且absolute FMT F1 `>=.893`；未达目标也必须保留。不得读取未完成数组的partial指标，checkpoint须等64.1冻结后再清理。 |
| Verify_Task3_AuxiliaryEpsilonPortfolio_64.1（已预注册，等待63.1独立审计） | 2026-08-31 | Task3-3D辅助投影epsilon搜索后的training-free逐family择优 | 在63.1产生任何性能指标前冻结；只比较已独立审计的62.1 portfolio与63.1 epsilon赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入64.1 selector的审计器重建选择、macro并核验80个文件，之后才允许删除63.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryEpsilonPortfolio_64_1.py`, `experiments/Audit_Task3_AuxiliaryEpsilonPortfolio_64_1.py`, `config/Verify_Task3_AuxiliaryEpsilonPortfolio_64.1.yaml`, `tests/test_task3_auxiliary_epsilon_portfolio_64_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_epsilon_portfolio_64.1_*.sh` | — | 尚无性能指标；64.1 config canonical SHA=`8ea80ea1…dbbc`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；32项63.1/64.1及旧portfolio回归测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.210`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryNormScale_65.1（已预注册，等待64.1独立审计） | 2026-08-31 | Task3-3D辅助投影归一化affine scale初值单因素搜索 | 在57.1仍未完整时冻结；64.1逐family完整配方作为exact control，仅把辅助投影内trainable LayerNorm/RMSNorm affine weight的初值由历史`1`改为`{0,.01,.03,.10,.25,.50,.75,1.50,2,4}`。不改变linear或residual head初始化且不消耗额外随机数；两臂同candidate、参数量、split、seed和预算。11×10×3×2=`660` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryNormScale_65.1.yaml`, `tests/test_task3_auxiliary_norm_scale_65_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_norm_scale_65.1_*.sh` | — | 尚无性能指标；21.1搜索投影architecture但未搜索归一化scale初值；65.1 config canonical SHA=`6fa97ac2…f30b`；20项实现/旧投影回归测试通过，入口必须严格晚于64.1独立审计 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.211`且absolute FMT F1 `>=.893`。不得读取未完成数组的partial指标，checkpoint须等66.1冻结后再清理。 |
| Verify_Task3_AuxiliaryNormScalePortfolio_66.1（已预注册，等待65.1独立审计） | 2026-08-31 | Task3-3D辅助归一化scale搜索后的training-free逐family择优 | 在65.1产生任何性能指标前冻结；只比较已独立审计的64.1 portfolio与65.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入66.1 selector的审计器重建选择、macro并核验80个文件，之后才允许删除65.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryNormScalePortfolio_66_1.py`, `experiments/Audit_Task3_AuxiliaryNormScalePortfolio_66_1.py`, `config/Verify_Task3_AuxiliaryNormScalePortfolio_66.1.yaml`, `tests/test_task3_auxiliary_norm_scale_portfolio_66_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_norm_scale_portfolio_66.1_*.sh` | — | 尚无性能指标；66.1 config canonical SHA=`13558556…8cec`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；46项65.1/66.1及旧portfolio回归测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.211`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryNormBias_67.1（已预注册，等待66.1独立审计） | 2026-08-31 | Task3-3D辅助投影LayerNorm affine bias初值单因素搜索 | 在57.1仍未完整时冻结；66.1逐family完整配方作为exact control并保留其normalization scale，仅把辅助投影LayerNorm bias初值由历史`0`改为`{-2,-1,-.5,-.25,-.1,.05,.1,.25,.5,1}`。不改变linear、scale或residual head初始化且不消耗额外随机数；两臂同candidate、参数量、split、seed和预算。11×10×3×2=`660` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryNormBias_67.1.yaml`, `tests/test_task3_auxiliary_norm_bias_67_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_norm_bias_67.1_*.sh` | — | 尚无性能指标；此前未单独搜索GELU前归一化bias；67.1 config canonical SHA=`0801a3ae…300f`；22项scale/bias/旧投影回归测试通过，入口必须严格晚于66.1独立审计 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.212`且absolute FMT F1 `>=.893`。不得读取未完成数组的partial指标，checkpoint须等68.1冻结后再清理。 |
| Verify_Task3_AuxiliaryNormBiasPortfolio_68.1（已预注册，等待67.1独立审计） | 2026-08-31 | Task3-3D辅助归一化bias搜索后的training-free逐family择优 | 在67.1产生任何性能指标前冻结；只比较已独立审计的66.1 portfolio与67.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入68.1 selector的审计器重建选择、macro并核验80个文件，之后才允许删除67.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryNormBiasPortfolio_68_1.py`, `experiments/Audit_Task3_AuxiliaryNormBiasPortfolio_68_1.py`, `config/Verify_Task3_AuxiliaryNormBiasPortfolio_68.1.yaml`, `tests/test_task3_auxiliary_norm_bias_portfolio_68_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_norm_bias_portfolio_68.1_*.sh` | — | 尚无性能指标；68.1 config canonical SHA=`64b26950…0953`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；66项67.1/68.1及旧portfolio回归测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.212`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryNormEpsilon_69.1（已预注册，等待68.1独立审计） | 2026-08-31 | Task3-3D辅助投影归一化epsilon单因素搜索 | 在57.1仍未完整时冻结；68.1逐family完整配方作为exact control并保留其normalization scale/bias，仅把辅助投影LayerNorm/RMSNorm的方差分母epsilon改为`{1e−12,1e−10,1e−8,1e−7,1e−6,1e−5,1e−4,1e−3,1e−2,1e−1}`。该字段不进入checkpoint且不消耗随机数；两臂同candidate、参数量、split、seed和预算。11×10×3×2=`660` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryNormEpsilon_69.1.yaml`, `tests/test_task3_auxiliary_norm_epsilon_69_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_norm_epsilon_69.1_*.sh` | — | 尚无性能指标；该epsilon属于归一化层而非42.1/63.1的Adam epsilon；69.1 config canonical SHA=`a4a6a679…cdd5`；118项跨版本回归测试通过，入口必须严格晚于68.1独立审计 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.213`且absolute FMT F1 `>=.893`。不得读取未完成数组的partial指标，checkpoint须等70.1冻结后再清理。 |
| Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1（已预注册，等待69.1独立审计） | 2026-08-31 | Task3-3D辅助归一化epsilon搜索后的training-free逐family择优 | 在69.1产生任何性能指标前冻结；只比较已独立审计的68.1 portfolio与69.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入70.1 selector的审计器重建选择、macro并核验80个文件，之后才允许删除69.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryNormEpsilonPortfolio_70_1.py`, `experiments/Audit_Task3_AuxiliaryNormEpsilonPortfolio_70_1.py`, `config/Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1.yaml`, `tests/test_task3_auxiliary_norm_epsilon_portfolio_70_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_norm_epsilon_portfolio_70.1_*.sh` | — | 尚无性能指标；70.1 config canonical SHA=`971b77d5…7600`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；118项跨版本回归测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.213`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryGaussianNoise_71.1（已预注册，等待70.1独立审计） | 2026-08-31 | Task3-3D辅助投影训练期Gaussian noise单因素搜索 | 在57.1仍未完整且70.1没有性能结果时冻结；70.1逐family完整配方作为exact control，仅在辅助投影与既有Dropout之后加入训练期逐元素零均值Gaussian noise，标准差为`{.005,.010,.025,.050,.100,.200,.300,.500,.750,1.000}`；`std=0`不抽随机数、无参数和checkpoint状态，构成精确控制。两臂同candidate、参数量、split、seed和预算。11×10×3×2=`660` trainings，evaluation无噪声，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryGaussianNoise_71.1.yaml`, `tests/test_task3_auxiliary_gaussian_noise_71_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_gaussian_noise_71.1_*.sh` | — | 尚无性能指标；71.1 config canonical SHA=`c608ff33…ac3d`；本地132项跨版本回归测试通过，入口必须严格晚于70.1独立审计 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.214`且absolute FMT F1 `>=.893`。不得读取未完成数组的partial指标，checkpoint须等72.1冻结后再清理。 |
| Verify_Task3_AuxiliaryGaussianNoisePortfolio_72.1（已预注册，等待71.1独立审计） | 2026-08-31 | Task3-3D辅助Gaussian noise搜索后的training-free逐family择优 | 在71.1产生任何性能指标前冻结；只比较已独立审计的70.1 portfolio与71.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入72.1 selector的审计器重建选择、macro并核验80个文件，之后才允许删除71.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryGaussianNoisePortfolio_72_1.py`, `experiments/Audit_Task3_AuxiliaryGaussianNoisePortfolio_72_1.py`, `config/Verify_Task3_AuxiliaryGaussianNoisePortfolio_72.1.yaml`, `tests/test_task3_auxiliary_gaussian_noise_portfolio_72_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_gaussian_noise_portfolio_72.1_*.sh` | — | 尚无性能指标；72.1 config canonical SHA=`4159e7b1…df53`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；本地132项跨版本回归测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.214`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryFeatureScale_73.1（已预注册，等待72.1独立审计） | 2026-08-31 | Task3-3D辅助表示固定幅度单因素搜索 | 在57.1仍未完整且72.1没有性能结果时冻结；72.1逐family完整配方作为exact control，仅将辅助投影、Dropout和可选Gaussian noise之后、residual fusion之前的表示乘以固定`{0,.01,.03,.10,.25,.50,.75,1.50,2,4}`；`scale=1`绕过乘法，精确保留原计算图。该字段无参数、checkpoint状态或随机数消耗；两臂同candidate、参数量、split、seed和预算。11×10×3×2=`660` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryFeatureScale_73.1.yaml`, `tests/test_task3_auxiliary_feature_scale_73_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_feature_scale_73.1_*.sh` | — | 尚无性能指标；32.1改变最终residual logit scale，65.1改变可训练normalization weight初值，均未测试本固定post-projection幅度；73.1 config canonical SHA=`eeb54b88…ad50`；本地146项跨版本回归测试通过，入口必须严格晚于72.1独立审计 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.215`且absolute FMT F1 `>=.893`。不得读取未完成数组的partial指标，checkpoint须等74.1冻结后再清理。 |
| Verify_Task3_AuxiliaryFeatureScalePortfolio_74.1（已预注册，等待73.1独立审计） | 2026-08-31 | Task3-3D辅助表示幅度搜索后的training-free逐family择优 | 在73.1产生任何性能指标前冻结；只比较已独立审计的72.1 portfolio与73.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入74.1 selector的审计器重建选择、macro并核验80个文件，之后才允许删除73.1的660个临时checkpoint | `experiments/Select_Task3_AuxiliaryFeatureScalePortfolio_74_1.py`, `experiments/Audit_Task3_AuxiliaryFeatureScalePortfolio_74_1.py`, `config/Verify_Task3_AuxiliaryFeatureScalePortfolio_74.1.yaml`, `tests/test_task3_auxiliary_feature_scale_portfolio_74_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_feature_scale_portfolio_74.1_*.sh` | — | 尚无性能指标；74.1 config canonical SHA=`582c6eda…8120`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；本地146项跨版本回归测试通过 | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.215`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryPostNorm_75.1（已完成并独立审计，development） | 2026-08-31–09-01 | Task3-3D辅助投影后固定归一化单因素搜索 | 冻结74.1逐family配方，仅在辅助投影activation后加入无参数`center`、root-mean-square normalization或LayerNorm形式变换；`none`为exact control。两臂同candidate、split、seed、结构和预算；4×10×3×2=`240` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryPostNorm_75.1.yaml`, `experiments/Audit_Task3_ParameterSearch.py`, `docs/Verify_experiments.md` | jobs `51139428`,`51139429`,`51139431`,`51139432`,`51139434`,`51139440`全部exit 0；40/40 children、240/240 trainings完整且stderr为空；selection/archive/remote-audit SHA=`2f9d7df3…58077`/`e5b76cf3…8fe49`/`8fdddcf7…590`；本地独立重算选择完全一致、宏平均最大差`2.22e−16`；cleanup在76.1审计后精确删除240 checkpoints并确认剩余0 | control Raw-PCA/FMT F1=`.68412/.89084`、gain=`+.20672`；选择DeltaWing `rms`、F22 `layer`后为`.68397/.89086`、gain=`+.20689`，AP gain=`+.22326`；10/10 datasets正 | **不支持投影后固定归一化成为新的整体赢家。**相对control仅增加F1 gain `+.00017`与FMT F1 `+.00003`，未达`+.216/.893`联合目标；局部结果进入76.1但不回写confirmation。 |
| Verify_Task3_AuxiliaryPostNormPortfolio_76.1（已完成并独立审计，development） | 2026-08-31–09-01 | Task3-3D辅助投影后归一化搜索后的training-free逐family择优 | 在75.1产出指标前冻结；比较已审计74.1与75.1，沿用zero-tolerance FMT guard和七项排序；0 trainings、confirmation关闭；冻结seed40–41两臂40 models/40 results并由独立实现核验80个文件 | `experiments/Select_Task3_AuxiliaryPostNormPortfolio_76_1.py`, `experiments/Audit_Task3_AuxiliaryPostNormPortfolio_76_1.py`, `config/Verify_Task3_AuxiliaryPostNormPortfolio_76.1.yaml`, `docs/Verify_experiments.md` | jobs `51139435`,`51139436`,`51139437`全部exit 0、stderr为空；portfolio/audit/source-identity SHA=`f8da77b7…65a4`/`a2fd86c0…31ad`/`4515ffca…d1c7`；独立重建差为0、40 models和40 results哈希全部通过 | 七个physical family全部保留74.1 `current_portfolio`；Raw-PCA/FMT F1=`.68346/.89057`、gain=`+.20711`，AP gain=`+.22338` | **不支持75.1进入组合模型，76.1没有提高74.1。**未达`+.216` gain或`.893` absolute FMT F1，因此未打开新confirmation population。 |
| Verify_Task23PaperTableConsistency_1.1（本地历史表审计） | 2026-08-31 | Task2/Task3历次confirmation与当前论文表交叉核对 | 不调用正式汇总代码，直接读取Task2-4.1、Task2-5.2 control/selected及Task3-4.1/6.1/7.2/8.1逐次CSV，重建dataset/family/seed宏平均并核对七份summary；另逐项比较Task3-7.2/8.1的checkpoint、feature、阈值和residual scale身份 | `experiments/Audit_Task23_PaperHistory.py`；合并表`docs/paper_tables_tasks_3d.md` | Python编译及520条核心比较记录审计通过；七份summary最大绝对差`2.22e−16`；7.2/8.1的40个逐次模型身份和冻结manifest身份完全一致 | Task2-4.1实跑为Raw/FMT `.46165/.63077`、gain`+.16912`；5.2/control的`.48594/.65216`是4.1配方在5.2 population上的复测，不是旧4.1结果；Task3历次F1 gain为`+.10000/+.17066/+.17522/+.18005` | **未发现数值矛盾；发现并修正两类文字矛盾：5.2/control实验身份误称，以及历史结果仍标作“当前”。**不同population不得冒充同一paired对照；只有Task3-7.2/8.1因模型身份完全一致可作为同方法空间复现合并解释。 |
| Verify_Task3_AuxiliaryChain_57.1–74.1（完成并独立审计；上方57–74单行保留运行前预注册快照） | 2026-09-01 | Task3-3D辅助投影优化器、归一化与正则化连续development搜索 | 依次比较auxiliary weight decay、Adam beta2、Adam beta1、Adam epsilon、normalization affine scale、LayerNorm bias、normalization epsilon、training Gaussian noise、fixed feature scale；每轮11 candidates（含exact control）×10 datasets×3 paired seeds×2 arms=`660` trainings，后接training-free逐family guarded portfolio；全程confirmation关闭 | 57–74各自selector/auditor/config/doc；本地下载9份完整无checkpoint逐次归档，以`experiments/Audit_Task3_ParameterSearch.py`逐份独立重算 | 9个GPU arrays共990/990 children完成、stderr全空：47 A100、741 GTX1080Ti、15 RTX2080Ti、71 P100、116 V100；9×660=`5940` paired trainings。9份搜索审计最大差`<=2.22e−16`（远端记录最大`4.44e−16`）；9个portfolio审计最大差`0`且各40模型+40结果SHA通过；9份归档本地SHA与远端一致；每轮660 checkpoints均在下游portfolio审计后精确清为0 | portfolio F1 gain依次：56.1 `+.205517`→58.1 `+.205584`→60.1 `+.206914`→62.1 `+.207017`→64.1 `+.207113`；66.1、68.1、70.1、72.1、74.1均保持`+.207113`。74.1 Raw-PCA/FMT F1 `.683459/.890572`，AP gain `+.223384`，10/10 datasets F1 gain正 | **57–64给development F1 gain带来小幅累积提高`+.001597`；65–74的scale/bias/normalization-epsilon/noise/fixed-scale均未产生通过zero-tolerance FMT guard的增量。**当前development最佳为64.1并由74.1原样继承；`+.216/.893`联合目标未达到，负结果保留。该development链不得替代6.1/7.2/8.1的独立空间confirmation。 |
| Verify_Task3_AuxiliaryGradientClip_77.1（已终止，数组不完整） | 2026-09-01 | Task3-3D辅助投影梯度裁剪单因素搜索 | 在75.1运行期间且未读取任何75.1 partial metric时冻结；76.1逐family完整配方作为exact control。只在既有全模型gradient clipping之后，对`fmt_encoder`参数施加`{.01,.03,.10,.30,1,3,10,30}`的梯度范数上限；control为严格no-op。FMT与同宽train-only Raw-PCA两臂同cap、结构、参数量、split、seed、batch和预算；原计划9×10×3×2=`540` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_AuxiliaryGradientClip_77.1.yaml`, `tests/test_task3_auxiliary_gradient_clip_77_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_gradient_clip_77.1_*.sh` | — | 用户于17:10–17:12终止；90个array children中75个完成、3个运行中取消、12个未启动，parent/selector/evidence/78.1 portfolio/audit/cleanup均取消；未读取任何candidate metric，未形成完整selection或审计结果 | **没有性能结论。**不得从75个已完成children读取或汇总partial指标；预注册`+.218/.893`目标既未达到也未被完整检验。 |
| Other_Task3_Post74SearchTermination_1.1（用户终止） | 2026-09-01 | Task3-3D后续超参数搜索终止与结果边界 | 按用户要求终止77.1–110.1整条已提交依赖链、旧7.1/17.1遗留作业，并放弃尚未提交的111.1/112.1；Slurm最终队列为空。79.1–110.1性能阶段从未启动，111.1/112.1仅有本地预注册代码；未读取77.1 partial metrics | `docs/experiment_log.md`, `docs/ibex_run_registry.md`；终止前代码commit `dcdbbba9` | — | 已完成development最佳仍为64.1、由74.1/76.1原样继承：Raw-PCA/FMT F1 `.683459/.890572`、gain `+.207113`、AP gain `+.223384`；57.1–64.1相对56.1只增加F1 gain `+.001597`，65.1–76.1没有进一步可保留提高。当前论文级确认仍为8.1：Raw-PCA/FMT F1 `.69131/.87136`、gain `+.18005`、AP gain `+.19991`，10/10 datasets与7/7 families正 | **终止扩展调参，不再追逐`+.216/.893`及后续更高目标。**后续搜索没有产生可替代8.1的新确认结果；这不反对Task3核心结论（独立population的F1 gain约`+.171–+.180`且均超过`+.15`），但明确反对把development `+.207`或未运行目标写成论文最终性能。 |
| mainExp_Task1_3D_4.1（统一配方；已确认并独立审计） | 2026-09-01 | Task1跨全部10个3D数据条目的单一FMT配方 | 只重排Task1-2.1/2.2已运行的11个FMT block×4个PCA维数；development ordinals 0–7按dataset-macro F1、worst-dataset F1、ARI排序，冻结一套全局FMT和全局Raw；confirmation ordinals 8–9使用5个新KMeans seeds、`n_init=20`，不得回改配方 | `experiments/Run_Task1_3D_Uniform.py`, `experiments/Audit_Task1_Uniform_4_1.py`, `config/mainExp_Task1_3D_4.1_uniform.yaml`, `tests/test_uniform_fmt_selection.py`; 初始链`51158642→51158643`因selector预加载未授权ordinals 8–9而取消；正式链`51158920→51158921`；首版审计`51158981`因把不可执行组合计入预期数而在比较前失败，修正审计`51159203`为`PASS`；selector/audit SHA=`a62fdaa3…912d/cf3c6247…0dea`；无checkpoint证据归档SHA=`acc33a56…07b0` | A100 `gpu102-09`用于confirmation；selector与审计在CPU节点`cn604-18` | development冻结`fmt_all+kin4→PCA8`，第二名`fmt_all→PCA16`仅低`.001079`；独立confirmation的500条记录给出统一Raw-PCA2/FMT dataset-macro F1 `.451024→.601384`，增益`+.150360`，9/10数据条目为正；F-22为反例`−.344272`。修正审计独立重建46个合法候选，核验460条selection、500条confirmation、证据哈希与全部宏平均 | **支持论文主张：固定一套Task-wide FMT配方即可在当前多数3D流场中提高training-free二类聚类。**不声称所有流场均提高；旧逐family最优`.6130`保留为补充结果，统一主表`.6014`低`.0116`。首版审计失败不构成科研矛盾，因为它在读取性能结果前因记录数断言退出。 |
| Verify_Task2_UniformFMT_6.1（development全局选择已独立审计） | 2026-09-01 | Task2跨全部10个3D数据条目的单一FMT block与单一same-VAE架构 | 仅使用4.1已经逐dataset完整交叉过的14个FMT block和4个VAE架构；3个新training seeds `90–92`；每个候选在10个datasets中保持完全相同，Raw/FMT两臂共享架构、latent、学习率、KL权重和预算；按dataset-macro paired F1 gain选择 | `experiments/Search_Task2_FMTVAE_3D.py`, `experiments/Audit_UniformFMT_3D.py`, `config/Verify_Task2_UniformFMT_6.1.yaml`, `tests/test_uniform_fmt_selection.py`; Ibex `51158646/51158645→51158647→51159061(audit)`；selection/leaderboard/audit SHA=`e4c2566d…b90f/52a26cff…5ba6/85d0dfa3…e7e0` | Ibex混合GPU完成600个候选children；selector/audit在CPU `cn113-35-l` | 560个FMT与40个Raw children全部成功，600个stderr均为0 byte；56套全局候选、1800条完整development记录中冻结`fmt_all+kin4 + l512_l64_b1e-6`。Raw/FMT dataset-macro F1 `.506074/.598943`，增益`+.092868`，8/10 datasets正，三个seed增益均正；Boeing747 `−.087713`、F-22 `−.120743`为development反例。独立审计核验603份证据并确认outer/confirmation未打开 | **完成Task2单一全局配方选择，但这仍不是论文最终确认值。**不引入新超参数、不允许逐family选择；只有6.2新population confirmation及其独立审计PASS后才能替换旧5.2主表。 |
| Verify_Task3_UniformFMT_9.1（development全局选择已独立审计） | 2026-09-01 | Task3跨全部10个3D数据条目的单一FMT block | 仅比较8.1六套distinct FMT winners与4.1三个既有shared baselines的并集，共9套；固定4.1 residual结构和优化器；使用已有冻结Raw backbone的3个paired seeds `40–42`；以非负dataset-macro F1 gain为约束，再最大化absolute FMT F1 | `experiments/Search_Task3_FMTResidual_3D.py`, `experiments/Audit_UniformFMT_3D.py`, `config/Verify_Task3_UniformFMT_9.1.yaml`, `tests/test_uniform_fmt_selection.py`; 初始`51158644→51158649`因错误请求不存在的Raw seeds `93–95`而26个children启动即失败，余下取消；修正链`51158922→51158925→51159062(audit)`；selection/leaderboard/audit SHA=`e9d50522…60e4/071b94ea…921c/fd0a3270…22b8` | Ibex 90个children全部使用Tesla V100-SXM2-32GB；selector/audit在CPU节点`cn604-15/cn604-18` | 初始链未训练且未产生科学结果。修正链90/90 children、540条完整development记录全部成功，90个stderr均为空；冻结统一`aivd1w3_dft`。Raw-PCA/FMT dataset-macro F1 `.696917/.895653`，增益`+.198736`；AP `.731893/.949229`，增益`+.217336`；10/10 datasets及3/3 seeds均为正，最差数据集F1增益仍为`+.077370`。独立审计PASS，核验9套候选、543份证据并确认outer/confirmation未打开 | **支持一套固定Task-wide FMT residual在development的全部10个条目提高Task3。**这仍不是论文最终confirmation值；只有9.2在ordinals 8–9独立确认并审计PASS后才能替换8.1。旧family-specific结果仅保留为补充材料。 |
| mainExp_Task2_3D_6.2（统一配方confirmation已完成并独立审计） | 2026-09-01–09-02 | Task2单一FMT block与单一same-VAE在全部10个3D条目上的独立确认 | 严格依赖`Verify_Task2_UniformFMT_6.1` selector与独立selection audit；冻结后在train ordinals 0–5训练、6–7只校准匿名KMeans簇方向、8–9只确认；5个新seed `100–104`，KMeans seed `7068`、`n_init=20`；Raw/FMT两臂共享全部VAE设置且不写checkpoint | `experiments/Run_UniformFMT_Confirmation_3D.py`, `experiments/Audit_UniformFMT_3D.py`, `config/mainExp_Task2_3D_6.2_uniform_confirmation.yaml`；首次freeze `51159663`和原summary `51159665`因旧脚本导入方式失败，均未改变科学配置；修复链`51163188→51159664→51164592→51164593→51164594`完整成功；manifest/per-run/summary/audit/cleanup SHA=`6ad04688…0c13/e7381d4d…624c/a15d596a…0564/cc1580ed…79f4/b43589e3…415d` | 20个GPU children：14 A100、4 P100、2 V100；运行时间`2:16–3:45`，stderr 20/20为空；summary/audit/cleanup在CPU `cn113-35-l` | 冻结统一`fmt_all+kin4 + l512_l64_b1e-6`。100条记录给出Raw/FMT dataset-macro F1 `.488444→.584040`，增益`+.095596`；8/10 datasets、5/7 families、5/5 seeds为正；Boeing `−.100903`、F-22 `−.060092`。独立审计`PASS`，Task2从未写checkpoint，cleanup确认0 | **支持更严格论文结论：一套跨全部流场不变的FMT/VAE配方仍使同一VAE在多数数据条目上提高；不支持所有流场均提高。**统一增益低于旧逐family 5.2的`+.239155`，后者只保留为补充结果。部署失败只反映入口问题，不构成性能矛盾。 |
| mainExp_Task3_3D_9.2（统一配方confirmation已完成并独立审计） | 2026-09-01–09-02 | Task3单一FMT residual配方在全部10个3D条目上的独立确认 | 严格依赖`Verify_Task3_UniformFMT_9.1` selector与独立selection audit；train ordinals 0–5、validation 6–7选择epoch/threshold/residual scale、8–9只确认；paired seeds `40–44`；FMT与train-only Raw-PCA residual保持同维度、同结构、同容量，checkpoint只临时保留到独立审计 | `experiments/Run_UniformFMT_Confirmation_3D.py`, `experiments/Audit_UniformFMT_3D.py`, `config/mainExp_Task3_3D_9.2_uniform_confirmation.yaml`；首次freeze `51159668`和原summary `51159670`因旧脚本导入方式失败；修复链`51163891→51159669→51164807→51164855→51164859`完整成功；manifest/per-run/summary/audit/cleanup SHA=`ee1896fd…c3d7/6e80adfa…50a9/f577a632…6ab9/e8af36e4…7a8e/96b5c369…e00d` | 20个GPU children：19 A100、1 RTX 2080 Ti；运行时间`2:27–5:48`，stderr 20/20为空；summary/audit在CPU `cn604-17`，cleanup在`cn604-18` | 冻结统一`aivd1w3_dft`。100条记录给出Raw-PCA/FMT F1 `.639844→.858338`，增益`+.218495`；AP `.689105→.923570`，增益`+.234465`；F1/AP均10/10 datasets正，F1 7/7 families、5/5 seeds正。独立审计`PASS`；审计前100个临时checkpoint，cleanup精确删除100、剩余0 | **支持统一FMT配方提高监督IVD-p95二分类。**F1增益高于旧逐family 8.1的`+.18005`，但绝对FMT F1`.85834`低于旧`.87136`；因confirmation population和Raw对照不同，不把差值解释为单独方法进退。部署失败不构成科研矛盾。 |
| Verify_Task3_AuxiliaryGradientClipPortfolio_78.1（已预注册，等待77.1独立审计） | 2026-09-01 | Task3-3D辅助投影梯度裁剪后的training-free逐family择优 | 在77.1产生任何性能指标前冻结；只比较已独立审计的76.1 portfolio与77.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入78.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除77.1的540个临时checkpoint | `experiments/Select_Task3_AuxiliaryGradientClipPortfolio_78_1.py`, `experiments/Audit_Task3_AuxiliaryGradientClipPortfolio_78_1.py`, `config/Verify_Task3_AuxiliaryGradientClipPortfolio_78.1.yaml`, `tests/test_task3_auxiliary_gradient_clip_portfolio_78_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_gradient_clip_portfolio_78.1_*.sh` | — | 尚无性能指标；78.1 config canonical SHA=`c8ee686f…978`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config canonical SHA固定为76.1 `af02ce49…f476`与77.1 `25cd0bea…fb56` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.218`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_EarlyStoppingPatience_79.1（已预注册，等待78.1独立审计） | 2026-09-01 | Task3-3D early-stopping patience单因素搜索 | 在75.1仍未完整且未读取任何75.1 partial metric时冻结；78.1逐family完整配方作为exact control，仅将连续无validation F1改善时允许继续训练的epoch数改为`{3,5,10,15,20,30,50,80}`，source control不覆盖原值；所有candidate保留来源max-epoch预算。FMT与同宽train-only Raw-PCA两臂同patience、结构、split、seed、batch与最大预算。9×10×3×2=`540` trainings，confirmation关闭 | `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_EarlyStoppingPatience_79.1.yaml`, `tests/test_task3_early_stopping_patience_79_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_early_stopping_patience_79.1_*.sh` | — | 尚无性能指标；47.1只搜索max epochs，7.1仅把160 epochs与patience30联合改变，均未隔离patience；通用组合器新增默认拒绝、仅对预注册`training.patience`放行的显式source override契约；79.1 config canonical SHA=`9f7f6210…3977`；35项新旧相关contract和Python编译通过 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.219`且absolute FMT F1 `>=.893`。不得读取未完成数组的partial指标，checkpoint须等80.1冻结与独立审计后再清理。 |
| Verify_Task3_EarlyStoppingPatiencePortfolio_80.1（已预注册，等待79.1独立审计） | 2026-09-01 | Task3-3D patience搜索后的training-free逐family择优 | 在79.1产生任何性能指标前冻结；只比较已独立审计的78.1 portfolio与79.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入80.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除79.1的540个临时checkpoint | `experiments/Select_Task3_EarlyStoppingPatiencePortfolio_80_1.py`, `experiments/Audit_Task3_EarlyStoppingPatiencePortfolio_80_1.py`, `config/Verify_Task3_EarlyStoppingPatiencePortfolio_80.1.yaml`, `tests/test_task3_early_stopping_patience_portfolio_80_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_early_stopping_patience_portfolio_80.1_*.sh` | — | 尚无性能指标；80.1 config canonical SHA=`9d950e04…5532`；static preflight明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为78.1 `c8ee686f…1978`与79.1 `9f7f6210…3977` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.219`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_EarlyStoppingMinDelta_81.1（已预注册，等待80.1独立审计） | 2026-09-01 | Task3-3D early-stopping minimum-delta单因素搜索 | 在79.1未产生任何性能指标时冻结；80.1逐family完整配方作为exact control，仅把validation selection score至少提高多少才更新best checkpoint的`training.min_delta`改为`{0,1e-6,3e-6,1e-5,3e-5,3e-4,1e-3,3e-3,1e-2}`；source control保留来源值，patience和max epochs不变。两臂同candidate、结构、split、seed、batch和预算；10×10×3×2=`600` trainings，confirmation关闭 | `experiments/Search_Task3_LossOptimization_7_1.py`, `config/Verify_Task3_EarlyStoppingMinDelta_81.1.yaml`, `tests/test_task3_early_stopping_min_delta_81_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_early_stopping_min_delta_81.1_*.sh` | — | 尚无性能指标；现有Task3配置使用`min_delta=1e-4`但此前从未隔离搜索；仅预注册`training.min_delta`允许覆盖source，其余字段继续默认拒绝；81.1 config canonical SHA=`e18fb481…7a73` | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.220`且absolute FMT F1 `>=.893`。不得读取未完成数组的partial指标，checkpoint须等82.1冻结与独立审计后再清理。 |
| Verify_Task3_EarlyStoppingMinDeltaPortfolio_82.1（已预注册，等待81.1独立审计） | 2026-09-01 | Task3-3D min-delta搜索后的training-free逐family择优 | 在81.1产生任何性能指标前冻结；只比较已独立审计的80.1 portfolio与81.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入82.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除81.1的600个临时checkpoint | `experiments/Select_Task3_EarlyStoppingMinDeltaPortfolio_82_1.py`, `experiments/Audit_Task3_EarlyStoppingMinDeltaPortfolio_82_1.py`, `config/Verify_Task3_EarlyStoppingMinDeltaPortfolio_82.1.yaml`, `tests/test_task3_early_stopping_min_delta_portfolio_82_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_early_stopping_min_delta_portfolio_82.1_*.sh` | — | 尚无性能指标；82.1 config canonical SHA=`806c0a71…cbb4`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为80.1 `9d950e04…5532`与81.1 `e18fb481…7a73` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.220`且absolute FMT F1 `>=.893`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryLinearWeightInitialization_83.1（已预注册，等待82.1独立审计） | 2026-09-01 | Task3-3D辅助投影Linear weight初始化单因素搜索 | 在77.1数组未完成且未读取其partial metric时冻结，因而82.1尚不可能产生portfolio指标；82.1逐family完整配方作为exact control。仅把`fmt_encoder`内Linear weight改为Xavier-uniform或orthogonal，并扫描gain `{.25,.5,1,sqrt(2)}`；control为严格no-op。bias、参数量、来源配方、optimizer、loss及residual head不变；初始化后恢复RNG状态。两臂同candidate、split、seed、结构和预算；9×10×3×2=`540` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryLinearWeightInitialization_83.1.yaml`, `tests/test_task3_auxiliary_linear_weight_initialization.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_linear_weight_initialization_83.1_*.sh` | — | 尚无性能指标；单元测试逐参数证明非control只改变辅助投影Linear weight，bias与其他state逐位不变，下一随机数也与control一致；83.1 config canonical SHA=`e5e5d883…a948` | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.222`且absolute FMT F1 `>=.894`。不得读取未完成数组的partial指标，checkpoint须等84.1冻结与独立审计后再清理。 |
| Verify_Task3_AuxiliaryLinearWeightInitializationPortfolio_84.1（已预注册，等待83.1独立审计） | 2026-09-01 | Task3-3D辅助投影初始化搜索后的training-free逐family择优 | 在83.1产生任何性能指标前冻结；只比较已独立审计的82.1 portfolio与83.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入84.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除83.1的540个临时checkpoint | `experiments/Select_Task3_AuxiliaryLinearWeightInitializationPortfolio_84_1.py`, `experiments/Audit_Task3_AuxiliaryLinearWeightInitializationPortfolio_84_1.py`, `config/Verify_Task3_AuxiliaryLinearWeightInitializationPortfolio_84.1.yaml`, `tests/test_task3_auxiliary_linear_weight_initialization_portfolio_84_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_linear_weight_initialization_portfolio_84.1_*.sh` | — | 尚无性能指标；84.1 config canonical SHA=`a761c073…d4eb`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为82.1 `806c0a71…cbb4`与83.1 `e5e5d883…a948` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.222`且absolute FMT F1 `>=.894`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryLinearBiasScale_85.1（已预注册，等待84.1独立审计） | 2026-09-01 | Task3-3D辅助投影Linear bias初始幅度单因素搜索 | 在77.1数组未完成且未读取其partial metric时冻结，因而84.1尚不可能产生portfolio指标；84.1逐family完整配方作为exact control。仅把`fmt_encoder`内现有Linear bias按倍率`{0,.1,.25,.5,2,4,8}`缩放；control为严格no-op。Linear weight、归一化参数、总参数量、来源配方、optimizer、loss、residual head及RNG状态不变。两臂同candidate、split、seed、结构和预算；8×10×3×2=`480` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryLinearBiasScale_85.1.yaml`, `tests/test_task3_auxiliary_linear_bias_scale_85_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_linear_bias_scale_85.1_*.sh` | — | 尚无性能指标；单元测试逐参数证明候选仅按固定倍率改变辅助投影Linear bias，blockwise全部分支覆盖；85.1 config canonical SHA=`c8e7cc64…e69f` | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.224`且absolute FMT F1 `>=.895`。不得读取未完成数组的partial指标，checkpoint须等86.1冻结与独立审计后再清理。 |
| Verify_Task3_AuxiliaryLinearBiasScalePortfolio_86.1（已预注册，等待85.1独立审计） | 2026-09-01 | Task3-3D辅助投影Linear bias搜索后的training-free逐family择优 | 在85.1产生任何性能指标前冻结；只比较已独立审计的84.1 portfolio与85.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入86.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除85.1的480个临时checkpoint | `experiments/Select_Task3_AuxiliaryLinearBiasScalePortfolio_86_1.py`, `experiments/Audit_Task3_AuxiliaryLinearBiasScalePortfolio_86_1.py`, `config/Verify_Task3_AuxiliaryLinearBiasScalePortfolio_86.1.yaml`, `tests/test_task3_auxiliary_linear_bias_scale_portfolio_86_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_linear_bias_scale_portfolio_86.1_*.sh` | — | 尚无性能指标；86.1 config canonical SHA=`50f66f43…abb3`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为84.1 `a761c073…d4eb`与85.1 `c8e7cc64…e69f` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.224`且absolute FMT F1 `>=.895`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryProjectionActivation_87.1（已预注册，等待86.1独立审计） | 2026-09-01 | Task3-3D辅助投影activation单因素搜索 | 在77.1数组未完成且未读取其partial metric时冻结，因而86.1尚不可能产生portfolio指标；86.1逐family完整配方作为exact control。只递归替换`fmt_encoder`中已经存在的activation，比较Identity、SiLU、ReLU、LeakyReLU(0.01)、ELU、Mish、Tanh；control为严格no-op。不新增层，投影宽度、归一化、Linear weight/bias、参数量、来源配方、optimizer、loss、residual head及RNG状态不变。两臂同candidate、split、seed、结构和预算；8×10×3×2=`480` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryProjectionActivation_87.1.yaml`, `tests/test_task3_auxiliary_projection_activation_87_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_projection_activation_87.1_*.sh` | — | 尚无性能指标；区别于21.1投影结构混合搜索和29.1 residual-head activation搜索；87.1 config canonical SHA=`c1556129…1f9e` | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.226`且absolute FMT F1 `>=.896`。不得读取未完成数组的partial指标，checkpoint须等88.1冻结与独立审计后再清理。 |
| Verify_Task3_AuxiliaryProjectionActivationPortfolio_88.1（已预注册，等待87.1独立审计） | 2026-09-01 | Task3-3D辅助投影activation搜索后的training-free逐family择优 | 在87.1产生任何性能指标前冻结；只比较已独立审计的86.1 portfolio与87.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入88.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除87.1的480个临时checkpoint | `experiments/Select_Task3_AuxiliaryProjectionActivationPortfolio_88_1.py`, `experiments/Audit_Task3_AuxiliaryProjectionActivationPortfolio_88_1.py`, `config/Verify_Task3_AuxiliaryProjectionActivationPortfolio_88.1.yaml`, `tests/test_task3_auxiliary_projection_activation_portfolio_88_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_projection_activation_portfolio_88.1_*.sh` | — | 尚无性能指标；88.1 config canonical SHA=`4318e7aa…28d0`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为86.1 `50f66f43…abb3`与87.1 `c1556129…1f9e` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.226`且absolute FMT F1 `>=.896`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryActivationInputScale_89.1（已预注册，等待88.1独立审计） | 2026-09-01 | Task3-3D辅助投影activation输入尺度单因素搜索 | 在77.1数组未完成且未读取其partial metric时冻结，因而88.1尚不可能产生portfolio指标；88.1逐family完整配方作为exact control。仅在`fmt_encoder`每个已有activation前乘固定尺度`{.125,.25,.5,.75,1.5,2,4}`；control为严格no-op。不增加parameter/buffer，不改变activation类型、投影宽度、归一化、Linear weight/bias、73.1投影后feature scale、optimizer、loss、residual head或RNG状态。两臂同candidate、split、seed、结构和预算；8×10×3×2=`480` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryActivationInputScale_89.1.yaml`, `tests/test_task3_auxiliary_activation_input_scale_89_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_activation_input_scale_89.1_*.sh` | — | 尚无性能指标；89.1 config canonical SHA=`92ec2cac…14cc`；单元测试证明scale 1逐位no-op，其他scale覆盖MLP/blockwise及Identity activation且不改变参数量和随机流 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.228`且absolute FMT F1 `>=.897`。不得读取未完成数组的partial指标，checkpoint须等90.1冻结与独立审计后再清理。 |
| Verify_Task3_AuxiliaryActivationInputScalePortfolio_90.1（已预注册，等待89.1独立审计） | 2026-09-01 | Task3-3D辅助activation输入尺度搜索后的training-free逐family择优 | 在89.1产生任何性能指标前冻结；只比较已独立审计的88.1 portfolio与89.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入90.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除89.1的480个临时checkpoint | `experiments/Select_Task3_AuxiliaryActivationInputScalePortfolio_90_1.py`, `experiments/Audit_Task3_AuxiliaryActivationInputScalePortfolio_90_1.py`, `config/Verify_Task3_AuxiliaryActivationInputScalePortfolio_90.1.yaml`, `tests/test_task3_auxiliary_activation_input_scale_portfolio_90_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_activation_input_scale_portfolio_90.1_*.sh` | — | 尚无性能指标；90.1 config canonical SHA=`0c4c2a81…f4d2b`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为88.1 `4318e7aa…28d0`与89.1 `92ec2cac…14cc` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.228`且absolute FMT F1 `>=.897`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryActivationInputShift_91.1（已预注册，等待90.1独立审计） | 2026-09-01 | Task3-3D辅助投影activation输入偏移单因素搜索 | 在77.1数组未完成且未读取其partial metric时冻结，因而90.1尚不可能产生portfolio指标；90.1逐family完整配方作为exact control。仅在`fmt_encoder`每个已有activation前加入固定偏移`{-2,-1,-.5,-.25,.25,.5,1,2}`；control为严格no-op。不增加parameter/buffer，不改变activation类型、89.1输入scale、投影宽度、归一化、Linear weight/bias、73.1投影后feature scale、optimizer、loss、residual head或RNG状态。两臂同candidate、split、seed、结构和预算；9×10×3×2=`540` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryActivationInputShift_91.1.yaml`, `tests/test_task3_auxiliary_activation_input_shift_91_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_activation_input_shift_91.1_*.sh` | — | 尚无性能指标；91.1 config canonical SHA=`9135a5ba…feaf`；单元测试要求shift 0逐位no-op、输入顺序为`activation(scale*x+shift)`、其他shift覆盖MLP/blockwise且不改变参数量和随机流 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.230`且absolute FMT F1 `>=.898`。不得读取未完成数组的partial指标，checkpoint须等92.1冻结与独立审计后再清理。 |
| Verify_Task3_AuxiliaryActivationInputShiftPortfolio_92.1（已预注册，等待91.1独立审计） | 2026-09-01 | Task3-3D辅助activation输入偏移搜索后的training-free逐family择优 | 在91.1产生任何性能指标前冻结；只比较已独立审计的90.1 portfolio与91.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入92.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除91.1的540个临时checkpoint | `experiments/Select_Task3_AuxiliaryActivationInputShiftPortfolio_92_1.py`, `experiments/Audit_Task3_AuxiliaryActivationInputShiftPortfolio_92_1.py`, `config/Verify_Task3_AuxiliaryActivationInputShiftPortfolio_92.1.yaml`, `tests/test_task3_auxiliary_activation_input_shift_portfolio_92_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_activation_input_shift_portfolio_92.1_*.sh` | — | 尚无性能指标；92.1 config canonical SHA=`7417ddc3…f1c3`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为90.1 `0c4c2a81…f4d2b`与91.1 `9135a5ba…feaf` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.230`且absolute FMT F1 `>=.898`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryActivationResidualMix_93.1（已预注册，等待92.1独立审计） | 2026-09-01 | Task3-3D辅助投影activation residual-mix单因素搜索 | 在77.1数组未完成且未读取其partial metric时冻结，因而92.1尚不可能产生portfolio指标；92.1逐family完整配方作为exact control。将`fmt_encoder`每个已有activation替换为`(1-r)·activation(x)+r·x`，扫描`r={.05,.1,.25,.5,.75,.9,1}`；control为严格no-op。不增加parameter/buffer，不改变activation类型、输入scale/shift、投影宽度、归一化、Linear weight/bias、optimizer、loss、residual head或RNG状态。两臂同candidate、split、seed、结构和预算；8×10×3×2=`480` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryActivationResidualMix_93.1.yaml`, `tests/test_task3_auxiliary_activation_residual_mix_93_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_activation_residual_mix_93.1_*.sh` | — | 尚无性能指标；93.1 config canonical SHA=`7307df51…e20e`；单元测试要求mix 0逐位no-op、输入顺序为`mix(activation(scale*x+shift),scale*x+shift)`、数值公式正确且不改变参数量和随机流 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.232`且absolute FMT F1 `>=.899`。不得读取未完成数组的partial指标，checkpoint须等94.1冻结与独立审计后再清理。 |
| Verify_Task3_AuxiliaryActivationResidualMixPortfolio_94.1（已预注册，等待93.1独立审计） | 2026-09-01 | Task3-3D辅助activation residual-mix搜索后的training-free逐family择优 | 在93.1产生任何性能指标前冻结；只比较已独立审计的92.1 portfolio与93.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入94.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除93.1的480个临时checkpoint | `experiments/Select_Task3_AuxiliaryActivationResidualMixPortfolio_94_1.py`, `experiments/Audit_Task3_AuxiliaryActivationResidualMixPortfolio_94_1.py`, `config/Verify_Task3_AuxiliaryActivationResidualMixPortfolio_94.1.yaml`, `tests/test_task3_auxiliary_activation_residual_mix_portfolio_94_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_activation_residual_mix_portfolio_94.1_*.sh` | — | 尚无性能指标；94.1 config canonical SHA=`255b7503…7e38`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为92.1 `7417ddc3…f1c3`与93.1 `7307df51…e20e` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.232`且absolute FMT F1 `>=.899`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_AuxiliaryActivationResidualGain_95.1（已预注册，等待94.1独立审计） | 2026-09-01 | Task3-3D辅助投影activation residual-gain单因素搜索 | 在77.1数组未完成且未读取其partial metric时冻结，因而94.1尚不可能产生portfolio指标；94.1逐family完整配方作为exact control。对每个完整当前activation响应增加固定旁路`f(x)+g·x`，扫描`g={.01,.03,.1,.25,.5,1,2,4}`；control为严格no-op。与93.1不同，旁路增长时不衰减原非线性；操作位于activation override、输入scale/shift及residual mix之后。不增加parameter/buffer，不改变投影宽度、归一化、Linear weight/bias、optimizer、loss、residual head或RNG状态。两臂同candidate、split、seed、结构和预算；9×10×3×2=`540` trainings，confirmation关闭 | `FMT_Utils/PathlineClassifier_3D.py`, `config/Verify_Task3_AuxiliaryActivationResidualGain_95.1.yaml`, `tests/test_task3_auxiliary_activation_residual_gain_95_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_activation_residual_gain_95.1_*.sh` | — | 尚无性能指标；95.1 config canonical SHA=`4852e22d…290e`；单元测试要求gain 0逐位no-op、组合顺序为`mix activation -> additive gain`、数值公式正确且不改变参数量和随机流 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.234`且absolute FMT F1 `>=.900`。不得读取未完成数组的partial指标，checkpoint须等96.1冻结与独立审计后再清理。 |
| Verify_Task3_AuxiliaryActivationResidualGainPortfolio_96.1（已预注册，等待95.1独立审计） | 2026-09-01 | Task3-3D辅助activation residual-gain搜索后的training-free逐family择优 | 在95.1产生任何性能指标前冻结；只比较已独立审计的94.1 portfolio与95.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入96.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除95.1的540个临时checkpoint | `experiments/Select_Task3_AuxiliaryActivationResidualGainPortfolio_96_1.py`, `experiments/Audit_Task3_AuxiliaryActivationResidualGainPortfolio_96_1.py`, `config/Verify_Task3_AuxiliaryActivationResidualGainPortfolio_96.1.yaml`, `tests/test_task3_auxiliary_activation_residual_gain_portfolio_96_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_auxiliary_activation_residual_gain_portfolio_96.1_*.sh` | — | 尚无性能指标；96.1 config canonical SHA=`8feadc2b…382a`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为94.1 `255b7503…6e38`与95.1 `4852e22d…290e` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.234`且absolute FMT F1 `>=.900`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_ResidualHeadCapacityRefresh_97.1（已预注册，等待96.1独立审计） | 2026-09-01 | Task3-3D新辅助portfolio上的residual-head容量复核 | 31.2只在旧22.1 anchored表示上扫描head容量；在77.1仍未完成且未读partial metric、96.1尚无portfolio指标时冻结本复核。96.1逐family完整配方作为exact control；另扫描width `{32,48,64,80}`×depth `{1,2,3}`，只允许覆盖`head_hidden_dim/head_depth`。FMT与同宽train-only Raw-PCA两臂使用相同容量、初始化、split、seed、optimizer和预算；13×10×3×2=`780` trainings；full preflight须逐cell验证低于Raw-wide参数上限，confirmation关闭 | `config/Verify_Task3_ResidualHeadCapacityRefresh_97.1.yaml`, `tests/test_task3_residual_head_capacity_refresh_97_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_capacity_refresh_97.1_*.sh` | — | 尚无性能指标；97.1 config canonical SHA=`d1111b16…4e1f`；契约测试证明exact source不被历史64×2默认值覆盖、12个容量cell完整且其余source recipe保持不变 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.236`且absolute FMT F1 `>=.901`。不得读取未完成数组的partial指标，checkpoint须等98.1冻结与独立审计后再清理。 |
| Verify_Task3_ResidualHeadCapacityRefreshPortfolio_98.1（已预注册，等待97.1独立审计） | 2026-09-01 | Task3-3D新容量复核后的training-free逐family择优 | 在97.1产生任何性能指标前冻结；只比较已独立审计的96.1 portfolio与97.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入98.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除97.1的780个临时checkpoint | `experiments/Select_Task3_ResidualHeadCapacityRefreshPortfolio_98_1.py`, `experiments/Audit_Task3_ResidualHeadCapacityRefreshPortfolio_98_1.py`, `config/Verify_Task3_ResidualHeadCapacityRefreshPortfolio_98.1.yaml`, `tests/test_task3_residual_head_capacity_refresh_portfolio_98_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_capacity_refresh_portfolio_98.1_*.sh` | — | 尚无性能指标；98.1 config canonical SHA=`b8687fc4…f40d`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为96.1 `8feadc2b…382a`与97.1 `d1111b16…4e1f` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.236`且absolute FMT F1 `>=.901`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_ResidualHeadNormActivationRefresh_99.1（已预注册，等待98.1独立审计） | 2026-09-01 | Task3-3D新portfolio上的residual-head normalization×activation复核 | 29.1只在旧22.1 anchored表示上扫描该因素，虽整体未达标但Channel/Smoke选择RMSNorm、half-cylinder选择ReLU，表明存在family interaction；在77.1仍未完成且未读partial metric、98.1尚无portfolio指标时冻结本复核。98.1逐family完整配方为exact control；另扫描`{LayerNorm,RMSNorm,None}`×`{GELU,SiLU,ReLU}`，只允许覆盖`head_normalization/head_activation`。两臂同设计、容量、初始化、split、seed、optimizer和预算；10×10×3×2=`600` trainings；full preflight须验证Raw-wide参数上限，confirmation关闭 | `config/Verify_Task3_ResidualHeadNormActivationRefresh_99.1.yaml`, `tests/test_task3_residual_head_norm_activation_refresh_99_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_norm_activation_refresh_99.1_*.sh` | — | 尚无性能指标；99.1 config canonical SHA=`5c3fd8f7…6ad5`；契约测试证明exact source保持其现有head设计，9个factorial cell只覆盖两个注册因素，其余98.1 recipe不变 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.238`且absolute FMT F1 `>=.902`。不得读取未完成数组的partial指标，checkpoint须等100.1冻结与独立审计后再清理。 |
| Verify_Task3_ResidualHeadNormActivationRefreshPortfolio_100.1（已预注册，等待99.1独立审计） | 2026-09-01 | Task3-3D新head-factorial后的training-free逐family择优 | 在99.1产生任何性能指标前冻结；只比较已独立审计的98.1 portfolio与99.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入100.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除99.1的600个临时checkpoint | `experiments/Select_Task3_ResidualHeadNormActivationRefreshPortfolio_100_1.py`, `experiments/Audit_Task3_ResidualHeadNormActivationRefreshPortfolio_100_1.py`, `config/Verify_Task3_ResidualHeadNormActivationRefreshPortfolio_100.1.yaml`, `tests/test_task3_residual_head_norm_activation_refresh_portfolio_100_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_norm_activation_refresh_portfolio_100.1_*.sh` | — | 尚无性能指标；100.1 config canonical SHA=`f4d4dd2d…540e`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为98.1 `b8687fc4…f40d`与99.1 `5c3fd8f7…6ad5` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.238`且absolute FMT F1 `>=.902`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_ResidualHeadLearningRateRefresh_101.1（已预注册，等待100.1独立审计） | 2026-09-01 | Task3-3D residual-head独立学习率搜索 | 27.1只搜索全局learning rate，55.1只搜索辅助投影learning-rate multiplier；此前从未把下游residual head与辅助投影的更新速度分开。在77.1仍未完成且未读partial metric、100.1尚无portfolio指标时冻结。100.1逐family完整配方为exact control；仅扫描下游head multiplier `{.05,.10,.25,.50,2,4,8,16}`。乘数作用于`fmt_encoder`之外所有可训练residual参数，冻结Raw不进入optimizer，来源auxiliary optimizer设置不变。两臂同cell、结构、split、seed与预算；9×10×3×2=`540` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `config/Verify_Task3_ResidualHeadLearningRateRefresh_101.1.yaml`, `tests/test_task3_residual_head_learning_rate_refresh_101_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_learning_rate_refresh_101.1_*.sh` | — | 尚无性能指标；101.1 config canonical SHA=`d1b07663…5ec8`；契约测试要求multiplier-one保持历史flat optimizer，非control精确拆分且不重复/遗漏head与auxiliary参数，并与来源auxiliary multiplier共存 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.240`且absolute FMT F1 `>=.903`。不得读取未完成数组的partial指标，checkpoint须等102.1冻结与独立审计后再清理。 |
| Verify_Task3_ResidualHeadLearningRateRefreshPortfolio_102.1（已预注册，等待101.1独立审计） | 2026-09-01 | Task3-3D head-learning-rate搜索后的training-free逐family择优 | 在101.1产生任何性能指标前冻结；只比较已独立审计的100.1 portfolio与101.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入102.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除101.1的540个临时checkpoint | `experiments/Select_Task3_ResidualHeadLearningRateRefreshPortfolio_102_1.py`, `experiments/Audit_Task3_ResidualHeadLearningRateRefreshPortfolio_102_1.py`, `config/Verify_Task3_ResidualHeadLearningRateRefreshPortfolio_102.1.yaml`, `tests/test_task3_residual_head_learning_rate_refresh_portfolio_102_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_learning_rate_refresh_portfolio_102.1_*.sh` | — | 尚无性能指标；102.1 config canonical SHA=`a8fa82d2…53a0`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为100.1 `f4d4dd2d…540e`与101.1 `d1b07663…5ec8` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.240`且absolute FMT F1 `>=.903`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_ResidualHeadWeightDecayRefresh_103.1（已预注册，等待102.1独立审计） | 2026-09-01 | Task3-3D residual-head独立weight decay搜索 | 27.1只搜索全局weight decay，57.1只搜索辅助投影weight-decay multiplier；此前从未把下游residual head与辅助投影的正则强度分开。在101.1尚无性能指标时冻结。102.1逐family完整配方为exact control；仅扫描下游head multiplier `{0,.01,.05,.10,.25,.50,2,4,10,100}`。乘数作用于`fmt_encoder`之外所有可训练residual参数，冻结Raw不进入optimizer，来源auxiliary optimizer设置不变。两臂同cell、结构、split、seed与预算；11×10×3×2=`660` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `config/Verify_Task3_ResidualHeadWeightDecayRefresh_103.1.yaml`, `tests/test_task3_residual_head_weight_decay_refresh_103_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_weight_decay_refresh_103.1_*.sh` | — | 尚无性能指标；103.1 config canonical SHA=`8f55aebf…8547`；契约测试要求multiplier-one保持历史flat optimizer，非control精确拆分且不重复/遗漏head与auxiliary参数，并与来源head learning-rate及auxiliary weight-decay multiplier共存 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.242`且absolute FMT F1 `>=.904`。不得读取未完成数组的partial指标，checkpoint须等104.1冻结与独立审计后再清理。 |
| Verify_Task3_ResidualHeadWeightDecayRefreshPortfolio_104.1（已预注册，等待103.1独立审计） | 2026-09-01 | Task3-3D head-weight-decay搜索后的training-free逐family择优 | 在103.1产生任何性能指标前冻结；只比较已独立审计的102.1 portfolio与103.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入104.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除103.1的660个临时checkpoint | `experiments/Select_Task3_ResidualHeadWeightDecayRefreshPortfolio_104_1.py`, `experiments/Audit_Task3_ResidualHeadWeightDecayRefreshPortfolio_104_1.py`, `config/Verify_Task3_ResidualHeadWeightDecayRefreshPortfolio_104.1.yaml`, `tests/test_task3_residual_head_weight_decay_refresh_portfolio_104_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_weight_decay_refresh_portfolio_104.1_*.sh` | — | 尚无性能指标；104.1 config canonical SHA=`85bd0eae…0c8c`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为102.1 `a8fa82d2…53a0`与103.1 `8f55aebf…8547` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.242`且absolute FMT F1 `>=.904`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_ResidualHeadBeta2Refresh_105.1（已预注册，等待104.1独立审计） | 2026-09-01 | Task3-3D residual-head独立Adam beta2搜索 | 34.1搜索全局Adam beta，59.1/61.1搜索辅助投影beta；此前从未把下游residual head的second-moment decay与辅助投影分开。在77.1仍未完成且未读partial metric、103.1尚无指标时冻结。104.1逐family完整配方为exact control；仅扫描下游head beta2 `{0,.5,.8,.9,.95,.98,.99,.995,.999,.9999}`。覆盖作用于`fmt_encoder`之外所有可训练residual参数；来源head beta1与auxiliary betas保持，冻结Raw不进入optimizer。两臂同cell、结构、split、seed与预算；11×10×3×2=`660` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `config/Verify_Task3_ResidualHeadBeta2Refresh_105.1.yaml`, `tests/test_task3_residual_head_beta2_refresh_105_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_beta2_refresh_105.1_*.sh` | — | 尚无性能指标；105.1 config canonical SHA=`200fbd8c…d4a`；契约测试要求exact source保持历史flat optimizer，非control只改变downstream group beta2，auxiliary group严格保持来源beta与其余optimizer设置 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.244`且absolute FMT F1 `>=.905`。不得读取未完成数组的partial指标，checkpoint须等106.1冻结与独立审计后再清理。 |
| Verify_Task3_ResidualHeadBeta2RefreshPortfolio_106.1（已预注册，等待105.1独立审计） | 2026-09-01 | Task3-3D head-beta2搜索后的training-free逐family择优 | 在105.1产生任何性能指标前冻结；只比较已独立审计的104.1 portfolio与105.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入106.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除105.1的660个临时checkpoint | `experiments/Select_Task3_ResidualHeadBeta2RefreshPortfolio_106_1.py`, `experiments/Audit_Task3_ResidualHeadBeta2RefreshPortfolio_106_1.py`, `config/Verify_Task3_ResidualHeadBeta2RefreshPortfolio_106.1.yaml`, `tests/test_task3_residual_head_beta2_refresh_portfolio_106_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_beta2_refresh_portfolio_106.1_*.sh` | — | 尚无性能指标；106.1 config canonical SHA=`63338791…f71`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为104.1 `85bd0eae…0c8c`与105.1 `200fbd8c…d4a` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.244`且absolute FMT F1 `>=.905`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_ResidualHeadBeta1Refresh_107.1（已预注册，等待106.1独立审计） | 2026-09-01 | Task3-3D residual-head独立Adam beta1搜索 | 34.1搜索全局Adam beta，61.1搜索辅助投影beta1；此前从未把下游residual head的一阶动量衰减与辅助投影分开。在77.1仍未完成且未读partial metric、105.1尚无指标时冻结。106.1逐family完整配方为exact control；仅扫描下游head beta1 `{0,.25,.5,.7,.8,.85,.9,.95,.98,.99}`。覆盖作用于`fmt_encoder`之外所有可训练residual参数；来源head beta2与auxiliary betas保持，冻结Raw不进入optimizer。两臂同cell、结构、split、seed与预算；11×10×3×2=`660` trainings，confirmation关闭 | `config/Verify_Task3_ResidualHeadBeta1Refresh_107.1.yaml`, `tests/test_task3_residual_head_beta1_refresh_107_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_beta1_refresh_107.1_*.sh` | — | 尚无性能指标；107.1 config canonical SHA=`1b7a0f8c…97ac`；契约测试要求exact source保持现有optimizer，非control只改变downstream group beta1，head beta2与auxiliary group严格保持来源设置 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.246`且absolute FMT F1 `>=.906`。不得读取未完成数组的partial指标，checkpoint须等108.1冻结与独立审计后再清理。 |
| Verify_Task3_ResidualHeadBeta1RefreshPortfolio_108.1（已预注册，等待107.1独立审计） | 2026-09-01 | Task3-3D head-beta1搜索后的training-free逐family择优 | 在107.1产生任何性能指标前冻结；只比较已独立审计的106.1 portfolio与107.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入108.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除107.1的660个临时checkpoint | `experiments/Select_Task3_ResidualHeadBeta1RefreshPortfolio_108_1.py`, `experiments/Audit_Task3_ResidualHeadBeta1RefreshPortfolio_108_1.py`, `config/Verify_Task3_ResidualHeadBeta1RefreshPortfolio_108.1.yaml`, `tests/test_task3_residual_head_beta1_refresh_portfolio_108_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_beta1_refresh_portfolio_108.1_*.sh` | — | 尚无性能指标；108.1 config canonical SHA=`ee82e0ca…31ba`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为106.1 `63338791…f71`与107.1 `1b7a0f8c…97ac` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.246`且absolute FMT F1 `>=.906`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_ResidualHeadEpsilonRefresh_109.1（已预注册，等待108.1独立审计） | 2026-09-01 | Task3-3D residual-head独立Adam epsilon搜索 | 42.1搜索全局Adam epsilon，63.1搜索辅助投影epsilon；此前从未把下游residual head的adaptive-step denominator与辅助投影分开。在77.1仍未完成且未读partial metric、107.1尚无指标时冻结。108.1逐family完整配方为exact control；仅扫描下游head epsilon `{1e-12,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}`。覆盖作用于`fmt_encoder`之外所有可训练residual参数；来源head betas及全部auxiliary optimizer设置保持，冻结Raw不进入optimizer。两臂同cell、结构、split、seed与预算；11×10×3×2=`660` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `config/Verify_Task3_ResidualHeadEpsilonRefresh_109.1.yaml`, `tests/test_task3_residual_head_epsilon_refresh_109_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_epsilon_refresh_109.1_*.sh` | — | 尚无性能指标；109.1 config canonical SHA=`916e2768…e654`；契约测试要求exact source保持现有optimizer，非control只改变downstream group epsilon，head betas及auxiliary group严格保持来源设置 | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.248`且absolute FMT F1 `>=.907`。不得读取未完成数组的partial指标，checkpoint须等110.1冻结与独立审计后再清理。 |
| Verify_Task3_ResidualHeadEpsilonRefreshPortfolio_110.1（已预注册，等待109.1独立审计） | 2026-09-01 | Task3-3D head-epsilon搜索后的training-free逐family择优 | 在109.1产生任何性能指标前冻结；只比较已独立审计的108.1 portfolio与109.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入110.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除109.1的660个临时checkpoint | `experiments/Select_Task3_ResidualHeadEpsilonRefreshPortfolio_110_1.py`, `experiments/Audit_Task3_ResidualHeadEpsilonRefreshPortfolio_110_1.py`, `config/Verify_Task3_ResidualHeadEpsilonRefreshPortfolio_110.1.yaml`, `tests/test_task3_residual_head_epsilon_refresh_portfolio_110_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_epsilon_refresh_portfolio_110.1_*.sh` | — | 尚无性能指标；110.1 config canonical SHA=`146167c5…9a9d`；static preflight必须明确10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为108.1 `ee82e0ca…31ba`与109.1 `916e2768…e654` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.248`且absolute FMT F1 `>=.907`；达到目标也不会自动打开新的confirmation population。 |
| Verify_Task3_ResidualHeadGradientClipRefresh_111.1（已预注册，等待110.1独立审计） | 2026-09-01 | Task3-3D residual-head独立梯度裁剪搜索 | 33.1搜索全模型梯度裁剪，77.1只搜索辅助投影梯度裁剪；此前从未隔离下游residual head。在77.1仍未完成且未读partial metric、109.1尚无指标时冻结。110.1逐family完整配方为exact control；仅扫描head-only cap `{.01,.03,.10,.30,1,3,10,30}`。来源全模型clip先执行，随后head-only cap只覆盖`fmt_encoder`之外的可训练residual参数；来源auxiliary-only cap保持且作用于不相交参数集，冻结Raw不参与。两臂同cell、结构、split、seed与预算；9×10×3×2=`540` trainings，confirmation关闭 | `experiments/Verify_Task3_FMTResidual.py`, `config/Verify_Task3_ResidualHeadGradientClipRefresh_111.1.yaml`, `tests/test_task3_residual_head_gradient_clip_refresh_111_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_gradient_clip_refresh_111.1_*.sh` | — | 尚无性能指标；111.1 config canonical SHA=`d7da8daf…dab1`；契约测试要求exact source为严格no-op，非control只裁剪downstream residual梯度且不改变`fmt_encoder`梯度，并保留来源global/auxiliary clip | **仅为预注册假设，无结果结论。**zero-tolerance FMT F1/AP guard；联合目标为dataset-macro F1 gain `>=+.250`且absolute FMT F1 `>=.908`。不得读取未完成数组的partial指标，checkpoint须等112.1冻结与独立审计后再清理。 |
| Verify_Task3_ResidualHeadGradientClipRefreshPortfolio_112.1（已预注册，等待111.1独立审计） | 2026-09-01 | Task3-3D head-gradient-clip搜索后的training-free逐family择优 | 在111.1产生任何性能指标前冻结；只比较已独立审计的110.1 portfolio与111.1赢家，沿用zero-tolerance FMT guard和七项排序。0 trainings、confirmation关闭；冻结seed40–41两臂共40 models/40 results，由不导入112.1 selector的审计器重建选择、宏平均并核验80个文件，之后才允许删除111.1的540个临时checkpoint | `experiments/Select_Task3_ResidualHeadGradientClipRefreshPortfolio_112_1.py`, `experiments/Audit_Task3_ResidualHeadGradientClipRefreshPortfolio_112_1.py`, `config/Verify_Task3_ResidualHeadGradientClipRefreshPortfolio_112.1.yaml`, `tests/test_task3_residual_head_gradient_clip_refresh_portfolio_112_1.py`, `docs/Verify_experiments.md`, `ibex_bash/verify_task3_residual_head_gradient_clip_refresh_portfolio_112.1_*.sh` | — | 尚无性能指标；112.1 config canonical SHA=`c5ccae0f…6ca8`；static preflight确认10 datasets、2 sources、0 trainings、`performance_artifacts_read=false`；源config SHA固定为110.1 `146167c5…9a9d`与111.1 `d7da8daf…dab1` | **仅为预注册选择规则，无结果结论。**联合目标为dataset-macro F1 gain `>=+.250`且absolute FMT F1 `>=.908`；达到目标也不会自动打开新的confirmation population。 |
| Other_Task4A_FMTStreamlineClustering_1.1（完成，本地探索） | 2026-09-01 | Task4-a：GT hairpin 内的无标签 streamwise/spanwise 分解 | `channel_GTs.vtk` 的73个正 `VortexIds` 实例在独立`131×98×155` cube网格上产生16,711个中心种子；每个种子生成center+x/y/z±共7条双向unit-velocity RK4 streamline，33点/线，x/y周期；使用Task1冻结的161维Gram+chirality+sort-pool FMT和全局KMeans(k=2)。KMeans只读FMT；聚类后才以局部涡量轴对簇命名。严格proxy为26连通下恰好2个streamwise+1个spanwise dominant组件；soft proxy为目标组件质量覆盖率乘组件数误差指数惩罚 | `FMT_Utils/Task4A_StreamlineClustering_3D.py`, `experiments/Run_Task4A_FMTStreamlineClustering_1_1.py`, `tests/test_task4a_streamline_clustering_3d.py`, `docs/Other_Task4A_FMTStreamlineClustering_1.1.md` | `config/Other_Task4A_FMTStreamlineClustering_1.1.yaml`；base commit `a920ee0` + local implementation SHA256 core/runner/config=`fd1adf8b…b555/1110fe73…3e2/ed017f5f…bda`；`outputs/Other_Task4A_FMTStreamlineClustering_1.1/channel_GTs/summary.json` | 16,711/16,711 primitives有效；raw clusters=`13,999/2,712`；vorticity后验命名将raw1/raw0映射为streamwise/spanwise。严格2+1=`4/73=.0548`；输入单连通子集=`3/62=.0484`；soft mean/median=`.2662/.1353`；33/73实例没有达到5%阈值的streamwise主组件。直接`‖omega_x‖`/`‖omega_y‖` sanity check也仅`13/73`，soft mean `.3576` | **不支持全局pure-FMT KMeans能把每个hairpin稳定分成两腿一头。**全局簇更多区分不同实例/局部状态；且严格体素2+1规则对neck过渡和类别碎片敏感，不能单独当物理正确率。看过1.1后计算的逐实例KMeans为`10/73`、soft `.3570`，只能作为新版本候选，禁止回写或声称为1.1提升。`VortexIds`只给实例GT，无streamwise/spanwise真实标签，所有proxy与颜色图仍需人工审核。 **2026-09-03 Ibex 复现（job 51270590，CPU `cn511-07`，33 秒）：4/73、3/62、soft 均值 .26622、raw clusters 13,999/2,712、逐 VortexId CSV 73 行零差异；核心模块当前 SHA `047b6f8c…4bc9` 与 1.1 记录的 `fd1adf8b…b555` 不同，但结果逐字段相同。证据 `outputs/Other_Task4A_FMTStreamlineClustering_1.1/ibex_repro_20260903/`。** |
| Verify_Task4A_VelocityCurlBaseline_1.1（完成，本地物理baseline） | 2026-09-01 | Task4-a：velocity–curl传统方向分割、FMT对比及proxy-ground-truth评价 | 在与Other_Task4A_FMTStreamlineClustering_1.1完全相同的73个`VortexIds`、16,711个cube中心和拓扑proxy上，插值局部速度`v`与旋度`omega`，计算轴向无符号夹角`theta=acos(‖v·omega‖/(‖v‖‖omega‖))`；固定`theta<=45°`为streamwise/parallel，否则为spanwise/perpendicular。45°是到平行/垂直等距边界，未按结果搜索。FMT语义映射不按该proxy翻转。新增按`VortexId`等权的两类macro-F1、硬标签MSE和以`‖cos(2theta)‖`降权45°附近体素的MSE | `FMT_Utils/Task4A_TraditionalBaseline_3D.py`, `experiments/Run_Task4A_VelocityCurlBaseline_1_1.py`, `tests/test_task4a_traditional_baseline_3d.py`, `docs/Verify_experiments.md` | `config/Verify_Task4A_VelocityCurlBaseline_1.1.yaml`；base commit `a920ee0` + local implementation SHA256 core/runner/config=`f7f14ea2…1828/8509031e…77ff/f11d0969…4127`；`outputs/Verify_Task4A_VelocityCurlBaseline_1.1/channel_GTs/{summary.json,per_vortex_proxy_groundtruth_metrics.csv}` | 16,711/16,711方向有效；streamwise/spanwise=`5,645/11,066`。严格2+1：FMT`4/73`→baseline`13/73`；soft mean`.2662`→`.3799`。将baseline仅作proxy GT时，hairpin等权macro-F1 mean/median=`.4682/.4577`，streamwise/spanwise class-F1=`.1948/.7415`；proxy在73/73均含两类，但FMT在24/73没有streamwise cubes；硬标签MSE mean=`.3572`；角度置信加权MSE mean=`.3164`；voxel-pooled macro-F1=`.5131`（仅描述性） | **支持velocity–curl规则作为比全局pure-FMT KMeans更强的总体传统baseline，但FMT与该proxy的一致性偏低且类别不均衡。**agreement主要来自spanwise/perpendicular，streamwise F1仅`.1948`。proxy不是人工类型GT；perpendicular类可能含wall-normal curl。73个实例来自同一流场，不把16,711个cubes当独立重复，也不报告伪精确的p值/置信区间。图形source preflight 21/21、alignment、PDF 8pt glyph floor、collision audit与最终视觉审计均通过。 |
| Verify_Task4A_PerVortexFMTProxySweep_2.1（完成，失败的development参数搜索） | 2026-09-01 | Task4-a：仅调unit-velocity streamline与逐`VortexId` KMeans以拟合velocity–curl proxy | 固定73个实例和proxy；搜索3种mean-cube步长×3种每向积分步数×5种FMT邻域权重=`45`个表示候选，再在优胜表示上比较5套KMeans初始化/收敛/标准化，共50条正式记录。每个`VortexId`独立KMeans。分别报告proxy最佳置换上界、逐ID涡量轴命名及不读velocity/curl的2+1拓扑命名。预先暂定成功门槛macro-F1`>=.90`且硬MSE`<=.10` | `FMT_Utils/Task4A_PerVortexClustering_3D.py`, `experiments/Run_Task4A_PerVortexFMTProxySweep_2_1.py`, `tests/test_task4a_per_vortex_clustering_3d.py`, `docs/Verify_experiments.md` | `config/Verify_Task4A_PerVortexFMTProxySweep_2.1.yaml`；core/runner/config SHA=`e87d92cc…5640/c65c24dc…effd/75a31b59…a677`；`outputs/Verify_Task4A_PerVortexFMTProxySweep_2.1/channel_GTs/{parameter_search.csv,summary.json}` | 最佳粗搜为step scale`.5`、8 steps/向、half-length 4 mean cubes、neighbor weight`.25`、`n_init=2`：proxy置换F1/MSE=`.5652/.3805`、ID-macro ARI`.0546`。`n_init=20`稳健复验为`.5612/.3825`；同partition的涡量轴命名F1`.5472`，纯几何拓扑命名F1`.4669`。50/50候选完成，16,711/16,711 primitives有效 | **明确失败，不满足“几乎一致”。**步长、积分长度、邻域权重和KMeans初始化/收敛不足以补回当前unit-velocity rotation-invariant FMT缺失的局部速度尺度/梯度信息；增加初始化没有改善。下一版本必须从time-parameterized streamline cross重建几何velocity/curl特征，禁止回写2.1。proxy用于全体73 IDs选参，因此2.1只能是development，不是独立确认。 |
| Verify_Task4A_GeometricCurlKMeans_2.2（完成，通过proxy-development门槛） | 2026-09-01 | Task4-a：time-parameterized streamline几何重建curl方向并逐`VortexId` KMeans | 将七线cross改为`dx/dt=v(x)` RK4，使坐标位移保留speed；用中心时间差分估计7条线的seed velocity，再用x/y/z对向cross空间差分重建velocity gradient与curl。KMeans仅读取`tanh((cos²(theta_geometry)-.5)/T)`，每个ID独立二聚类；以较高median几何score命名streamwise。Phase1搜索6 time steps×4 offsets×5 temperatures，offset1.0因17个wall cubes无效的6组保留为invalid，其余90条完整；Phase2搜索6积分长度×4 KMeans设置=`24`，总114有效候选。门槛沿用2.1的F1`>=.90`、MSE`<=.10` | `FMT_Utils/{Task4A_StreamlineClustering_3D.py,Task4A_PerVortexClustering_3D.py}`, `experiments/Run_Task4A_GeometricCurlKMeans_2_2.py`, `experiments/Visualize_Task4A_GeometricCurlKMeans_2_2.py`, `experiments/Audit_Task4A_GeometricCurlKMeans_2_2.py`, `tests/test_task4a_{streamline,per_vortex}_clustering_3d.py`, `docs/Verify_experiments.md` | `config/Verify_Task4A_GeometricCurlKMeans_2.2.yaml`；core-stream/core-perID/runner/config/visual/audit SHA=`047b6f8c…4bc9/d835ec0f…a8c0/4893dd95…fbe4/2d03841a…431c/dd030ff4…a542/55261178…466e`；`outputs/Verify_Task4A_GeometricCurlKMeans_2.2/channel_GTs/{summary.json,parameter_search.csv,geometric_curl_kmeans_result.npz,independent_audit.json}` | Winner：time scale`.5`（dt`.00838506`）、offset`.5×min cube`、1 step/向、temperature`.1`、k-means++ `n_init=5`。ID-equal macro-F1 mean/median=`.94114/.94594`，stream/span F1=`.91611/.96618`；hard MSE=`.04375`；angle-weighted MSE=`.01034`；mean IoU=`.89437`；angle MAE=`2.827°`。67/73 IDs F1`>=.90`，71/73 F1`>=.80`，71/73 MSE`<=.10`；worst IDs51/10 F1`.7244/.7453`。voxel errors737/16711=`4.41%`。1–32 steps/向逐KMeans设置指标不变，故选最短。直接45°重建阈值F1/MSE=`.94017/.04413`；独立审计确认73/73均含两类且重算F1/MSE与summary差`0` | **通过预设总体“几乎一致”门槛，并将2.1 F1从约`.56`提高到`.941`；主要原因是time-parameterized几何恢复了unit-velocity FMT丢失的speed/gradient信息。**但不是73/73逐实例成功，6个ID低于`.90`，且全部73 IDs参与proxy选参；只能声称当前volume的proxy reproduction，不能声称未知人工类型GT或独立泛化。图形source 21/21、alignment、PDF 6pt glyph floor、collision 0 fail/0 warn和最终视觉审计均PASS。 |
| Task4-b 1.1（保留，独立审计否决） | 2026-09-01 | Task4-b 初始四类proxy、FMT四聚类与监督代码 | 四类为ordinary streamwise/spanwise、hairpin head/limb，non-vortex=`ignore=-1`；hairpin按完整随机`VortexId`拆分，但ordinary按x区间拆分 | `docs/{Task4B_channel_proxy_label_protocol_1.1.md,Other_Task4B_FMTFourClassClustering_1.1.md,mainExp_Task4B_1.1.md}`；原始输出仍保留在对应1.1目录 | — | 初始KMeans test macro-F1=`.3083`；监督仅完成2-epoch工程smoke | **不得作为正式结果。**独立审计发现train primitive中33.4%在一个primitive半径内有validation seed、26.0%有test seed，且初始a/b图误把全73个hairpins标成test。1.2建立前不运行完整监督实验。 |
| Verify_Task4B_ProxyLabels_1.2（完成，空间拆分修订） | 2026-09-01 | Task4-b channel四类proxy标签 | 用train hairpins选择最小候选体积且train micro/ID-equal recall均`>=.95`的vorticity-deviation gate；ordinary由velocity–curl 45°分型，head需`omega_y_prime>0`且y分量占优，其余hairpin为limb。全类型共用周期x缓冲区；完整`VortexId`必须完全位于单一slab | `FMT_Utils/Task4B_ProxyLabels_3D.py`, `experiments/Build_Task4B_ProxyLabels_1_1.py`, `config/Verify_Task4B_ProxyLabels_1.2.yaml`, `docs/Task4B_channel_proxy_label_protocol_1.2.md`；base commit `dcdbbba9` + local core/builder/config SHA=`40998fc8…93ee/f3af71c0…b24f/61af487f…88aa`；`outputs/Verify_Task4B_ProxyLabels_1.2/channel_GTs/{summary.json,proxy_label_manifest.json,task4b_proxy_labels.npz}` | — | 选择channel-profile vorticity deviation（**非标准IVD**）p80.5，threshold=`5.388844`，候选`19.50%`；train recall micro/ID-equal=`.9500/.9532`，validation=`.9709/.9718`，test=`.9361/.9375`。四类counts=`120,996/251,043/4,275/12,436`。完整实例train/val/test=`33/8/6`，26个buffer IDs排除。标准whole-domain flow-grid IVD p95 hairpin recall=`.01275` | **支持把channel-profile deviation用于本次proxy候选门，但不修改Task1/2/3/5的whole-field IVD p95。**head存在72/73；完整x/z序24/72；四项左右符号同时39/64，故位置/符号只作QA，不能冒充人工head/limb真值。 |
| Other_Task4B_FMTFourClassClustering_1.2（完成，失败baseline） | 2026-09-01 | Task4-b：`streamline→frozen Task1 FMT→KMeans(k=4)` | 20,641个7×33×3 primitives、161维FMT；train-only StandardScaler；KMeans `n_init=20, seed=7068`；validation枚举24个置换冻结映射，test只评估。缓存运行时验证最小周期x gap `.219911` 大于两倍primitive半径 `.214442` | `experiments/Build_Task4B_FourClassCache_1_1.py`, `experiments/Run_Task4B_FMTFourClassClustering_1_1.py`, `config/Other_Task4B_FMTFourClassClustering_1.2.yaml`, `docs/Other_Task4B_FMTFourClassClustering_1.2.md`；base commit `dcdbbba9` + local cache/runner/config SHA=`9a7d756d…d939/f65bb23b…73ef/fb7abcb0…5d48`；`outputs/Other_Task4B_FMTFourClassClustering_1.2/channel_GTs/{summary.json,task4b_fmt_four_class_kmeans_result.npz,task4b_fmt_four_class_kmeans_summary.*}` | — | test `N=3,330`：macro-F1=`.23787`，balanced accuracy=`.31307`，ARI=`.03363`，NMI=`.03919`；class F1=`.49814/.03575/.00000/.41758`。6个test IDs等权head/limb F1=`.23749`；严格2+1 proxy=`3/6`、KMeans=`0/6`，soft `.67226→.05206`；test-oracle映射与validation相同 | **明确失败：pure FMT四聚类不能恢复坐标定义的四类proxy，尤其hairpin-head F1为0。**测试ordinary各固定抽样1,000，不能表述为whole-field prevalence。图形alignment、PDF最小5.8pt、collision 0 fail/0 warn及最终视觉检查PASS；source静态审计唯一remaining warning是把runner中非图用的mean/median误判为未画error bar。 |
| mainExp_Task4B_1.2（完成，Ibex正式监督实验） | 2026-09-01 | Task4-b监督四分类：Raw、Raw-wide、FMT-only、Raw+FMT | exact四类balanced batch；train-only normalization；validation macro-F1选内存best epoch；test一次；三seed `7068–7070`；sample SD `ddof=1`；cache encoder/SHA与空间证书硬校验；不写checkpoint。Job `51152830`在A100-SXM4-80GB上12/12 runs完成，`COMPLETED 0:0`，独立逐样本审计`PASS` | `FMT_Utils/Task4B_Classifier_3D.py`, `experiments/Train_Task4B_FourClassClassifier_1_1.py`, `experiments/Audit_Task4B_Supervised_1_2.py`, `experiments/Visualize_Task4B_Supervised_1_2.py`, `config/mainExp_Task4B_1.2.yaml`, `docs/mainExp_Task4B_1.2.md`；base commit `dcdbbba9` + bundle SHA=`78afca97…af1`；model/trainer/config/audit/visual SHA=`63a59e6f…e5ff/6041e462…3d7f/5dc2ded0…5d17/491e48a4…ffb0c/4f343354…d66e`；cache SHA=`4be1dced…a190`；formal summary/audit SHA=`27ab0035…37fb/f13acda2…0ac7` | — | test macro-F1（mean±sample SD）Raw/Raw-wide/FMT-only/Raw+FMT=`.38837±.02276/.41892±.01805/.40214±.00239/.40944±.02275`；Raw+FMT−Raw=`+.02107±.03091`，逐seed=`[-.00841,+.01838,+.05323]`；balanced accuracy差=`+.03443±.04288`（2/3正）；macro AP差=`+.03359±.02164`（3/3正）。Raw+FMT−Raw-wide macro-F1=`-.00948±.03820`。逐类Raw+FMT−Raw F1：ordinary stream/span=`+.07016/+.06977`（均3/3正），hairpin head/limb=`-.01117/-.04449`（均1/3正）；六ID等权hairpin F1差=`-.00131±.05838`；Raw+FMT严格2+1每seed均`2/6`，proxy自身仅`3/6`；checkpoint count=0；正式图strict source 21/21、alignment/PDF字号/collision 11/11均PASS | **仅支持Raw+FMT相对Raw的总体macro-F1平均改善，但seed间不稳定；macro AP三seed一致改善。明确不支持hairpin-head/limb已经改善：总体增益来自ordinary方向类，六实例等权hairpin F1无改善。Raw-wide的macro-F1更高，不能排除容量/优化解释，也不声称统计显著。**标签仍是单steady channel的physics proxy；固定空间holdout不是whole-field、人工anatomy或跨时间泛化。 |
| Verify_Task4B_FullVolumeMemorization_1.1（完成，本地同集过拟合验证） | 2026-09-01 | Task4-b Raw/FMT监督管线零错误 sanity check | 合并冻结cache全部20,641行，fit=evaluation、无holdout；沿用`PathlineMulticlassClassifier3D`并增容至temporal/embedding/auxiliary=`128/1024/1024`，dropout/weight decay/early stopping关闭；每epoch全样本无放回一次；Raw、FMT-only、Raw+FMT×seeds7068–7070；连续3轮零错误且minimum true-logit margin>0才PASS；不写checkpoint | `experiments/{Verify_Task4B_FullVolumeMemorization_1_1.py,Audit_Task4B_FullVolumeMemorization_1_1.py}`, `config/Verify_Task4B_FullVolumeMemorization_1.1.yaml`, `docs/Verify_experiments.md`；config/trainer/audit SHA=`769d423a…5e158/2e2f33ff…cc61e/cae74479…374ce`；cache SHA=`4be1dced…a190`；summary/audit SHA=`aea80fcf…0cdd/e8b0f116…971f`；本地RTX3090、PyTorch2.6.0+cu126 | — | 9/9 runs PASS，9份prediction/history/run JSON完整，checkpoint 0。Raw通过epoch=`553/529/558`，FMT-only=`25/25/23`，Raw+FMT=`29/29/29`；每run accuracy/balanced accuracy/macro-F1及四类precision/recall/F1=`1.0`，confusion=`diag(5000,5000,2829,7812)`，minimum margin均正。独立审计重算所有epoch permutation、全样本normalization、原三split、全部positive VortexId和artifact identity均PASS；Raw/FMT/combined均20641 exact unique、跨标签冲突与不可约最小错误0。首次audit实现因额外要求run JSON自报非门槛mean margin而两次退出，修正该审计器自身要求后所有硬门槛不变并PASS | **支持“训练/标签/索引管线能在有限cache上完全拟合”，反对统一label mapping或sampler漏样本是正式低分主因；不改变1.2的holdout失败结论。**高容量同集记忆不是test或FMT优势。正式test hairpin→ordinary三seed均值Raw/Raw-wide/FMT-only/Raw+FMT=`64.56/51.05/51.48/63.58%`，而known-hairpin条件head/limb F1约`.706/.707/.599/.714`；更符合未输入`VortexId`/absolute-z plane context下的未见实例membership泛化失败。 **2026-09-03 Ibex 复现（job 51270629，A100 `gpu202-02-r`，12m43s）：9/9 PASS，独立审计 PASS；Raw 通过 epoch 530/549/548、FMT-only 25/25/23、Raw+FMT 30/29/30。证据 `outputs/Verify_Task4B_FullVolumeMemorization_1.1/ibex_repro_20260903/`。** |
| mainExp_Task4B_ChannelToTBL_2.1（无效，审计前部分试跑） | 2026-09-02 | Task4-b channel→TBL 四分类迁移 | 使用冻结的20,641行channel curated cache训练、TBL测试；只产生FMT-only与Raw+FMT共6/12个run，Raw与Raw-wide缺失，且尚无独立审计。配置还使用近似`offset/spatial_step=.1501503378764808` | `FMT_Utils/Task4B_CrossFlow_3D.py`, `experiments/{Prepare,Train}_Task4B_ChannelToTBL_2_1.py` | `config/mainExp_Task4B_ChannelToTBL_2.1.yaml` SHA=`1995f316…8335`；partial summary SHA=`d23eb2b…7fbe` | 仅6/12 run；没有完整四方法比较或独立审计 | **不得报告性能结论。**该版本是审计前部分试跑，后由2.2/2.3取代。 |
| mainExp_Task4B_ChannelToTBL_2.2（拒绝，无训练结果） | 2026-09-02 | Task4-b channel→TBL 输入身份修订 | 增加stale-artifact、coverage-adjusted与相对chunk路径检查；preflight发现配置近似ratio`.1501503378764808`与channel cache精确ratio`.1501458034894056`不一致，绝对差`4.5343870752e-6`，在首个epoch前拒绝 | `FMT_Utils/Task4B_CrossFlow_3D.py`, `experiments/{Prepare,Train}_Task4B_ChannelToTBL_2_1.py` | `config/mainExp_Task4B_ChannelToTBL_2.2.yaml` SHA=`fec91c07…2e2`；preparation summary SHA=`01810ce7…cc6` | 目标cache准备完成；training run=`0/12` | **不支持也不反对任何性能命题。**失败原因是输入契约不精确，不得引用2.2性能；2.3只修正为cache记录的精确ratio。 |
| Verify_Task123_FixedPathline_1.1（完成，Ibex development+外层confirmation） | 2026-09-02 | Task1/Task2/Task3固定pathline参数联合搜索 | 在10个3D数据条目和全部variant公共有效seed上，围绕`.25/48/32`一次只改变积分步长、积分步数或积分后重采样点数；7个固定variant为`dt_small=.125/96/32`、`baseline=.25/48/32`、`dt_large=.375/32/32`、`steps_short=.25/32/32`、`steps_long=.25/96/32`、`samples_16=.25/48/16`、`samples_48=.25/48/48`。每个variant分别完整运行Task1、Task2、Task3，不混入Task5式逐primitive多尺度。development只打开ordinals 0–7并按三任务等权dataset-macro F1增益选参；冻结winner后用ordinals 8–9和5个新seed比较baseline/winner，不在confirmation重新选择。Task1固定`fmt_all+kin4→PCA8`与Raw-PCA2；Task2固定同一MLP `[512,256]` latent-64 VAE；Task3固定`aivd1w3_dft`与等容量25,345参数Raw-PCA residual | `experiments/{Prepare,Run,Audit}_Task123_FixedPathline*.py`、`ibex_bash/verify_task123_fixed_pathline_1.1_*.sh`；base commit `22430b2e9504`；deployment/confirmation overlay SHA=`342e6c1f…445b/a94dc6ea…36e9`；development/confirmation证据归档SHA=`63853c8c…4555/62ecd98e…bec8` | `config/Verify_Task123_FixedPathline_1.1.yaml`；canonical SHA=`ee37c6b2…281` | development三任务等权F1增益排名：`samples_48 +.142679`、`baseline +.139302`、`dt_large +.138029`、`dt_small +.136224`、`steps_short +.131468`、`samples_16 +.127932`、`steps_long +.105347`；490 shards独立审计PASS。外层confirmation共220 shards：baseline与`samples_48`的三任务等权增益为`+.152004→+.154224`，差`+.002221`。逐任务变化：Task1 `+.181547→+.180536`（`−.001011`）；Task2 `+.074706→+.079315`（`+.004608`）；Task3 F1 `+.199758→+.202823`（`+.003065`），Task3 AP `+.220199→+.227262`（`+.007063`）。平均absolute FMT F1 `.706314→.703565`（`−.002749`）。远端和本地独立审计均PASS；删除1,240个临时Task3 checkpoint后残留0 | **找到了能把三任务平均相对FMT增益小幅提高的固定参数：只把每线重采样点数32增至48，物理积分轨迹仍为`.25/48`。**但效应很小且不逐任务一致：Task1略降，Task2/3小升；absolute FMT F1整体还下降。因此它支持“平均相对增益可复现地增加`.002221`”，不支持“48点普遍更好”或“应立即替换当前`.25/48/32`论文主配置”。 |
| mainExp_Task4B_PooledInstanceSplit_3.1（完成，Ibex，独立审计 PASS） | 2026-09-03 | Task4-b：channel+TBL 合并、按 hairpin 实例留出的四分类 | 每 volume 内按 cube 数分块，块内随机 `2 test/1 validation/7 train` 分配完整 `VortexId`；ordinary cube 继承最近 hairpin 实例的拆分（Voronoi，channel x/y 周期）；删除距任一测试种子 <1 个 primitive 半径（`16×spatial_step+offset`）的 train/validation cube（2R 因删掉 77%/64% 训练 hairpin cube 被预注册否决）；每 volume 无量纲坐标 `(primitive−首点)/spatial_step`，FMT 161 维重算；normalization 仅训练行；训练配方与 1.2 相同（Raw/Raw-wide/FMT-only/Raw+FMT × seeds 7068–7070） | `FMT_Utils/Task4B_PooledSplit_3D.py`, `experiments/{Build,Train,Audit}_Task4B_PooledInstanceSplit_3_1.py`, `tests/test_task4b_pooled_split_3d.py`, `docs/mainExp_Task4B_PooledInstanceSplit_3.1.md` | `config/mainExp_Task4B_PooledInstanceSplit_3.1.yaml`（SHA `2528c8cf…2184`）；Ibex jobs 51277193/51277194；cache SHA `40f21e35…d8a3`；summary/audit SHA `64547021…9aeb`/`d1113b88…65c5` | test macro-F1 Raw/Raw-wide/FMT-only/Raw+FMT=`.3841/.3687/.3979/.4196`；Raw+FMT−Raw `+.0356±.0094`（3/3 seed 正）；Raw+FMT−Raw-wide `+.0510±.0055`（3/3）；TBL 部分 `+.0523`、channel `+.0250`；hairpin→ordinary 错误率 Raw `.399`、Raw+FMT `.422`、FMT-only `.322` | 支持：同分布未见 hairpin 实例上 Raw+FMT 稳定优于 Raw 与 Raw-wide。反对：absolute macro-F1 仅 .42，约 42% hairpin cube 仍被判成 ordinary 且未因 FMT 改善；不支持“已解决 hairpin 四分类” |
| Verify_Task4B_PooledMemorization_3.1（完成，Ibex 分片，独立审计 PASS） | 2026-09-04 | Task4-b 合并 channel+TBL cache 全量记忆 sanity check | 对 `mainExp_Task4B_PooledInstanceSplit_3.1` 的合并 cache 全部 91,711 行做 fit=evaluation（无 holdout），配方与 `Verify_Task4B_FullVolumeMemorization_1.1` 相同：`PathlineMulticlassClassifier3D` 增容 temporal/embedding/auxiliary=128/1024/1024、dropout/weight decay/early stopping 关闭、Adam 1e-3 余弦退火、每 epoch 全样本一次无放回、最多 1000 epoch；连续 3 epoch 零错误且最小 true-logit margin>0 才 PASS。串行作业 51335442 因 P100 上 Raw 臂约 13.7 秒/epoch 预计超时，改以按 (variant, seed) 分片的数组 51339155 为正式路径，合并脚本用训练器自身 `_aggregate` 汇总；训练器/审计器为 1.1 的版本化拷贝，审计的 source/voxel 唯一性改为 (volume, index) | `experiments/{Verify,Audit}_Task4B_PooledMemorization_3_1.py`, `experiments/Merge_Task4B_PooledMemorization_3_1_Shards.py`, `ibex_bash/verify_task4b_pooled_memorization_3p1_{gpu,array_gpu,merge_cpu}.sh` | `config/Verify_Task4B_PooledMemorization_3.1.yaml`（SHA `abf3ac01…0ce4`）；cache SHA `40f21e35…d8a3`；Ibex jobs 51339155[0-8]/51339348（Tesla P100-PCIE-16GB）；summary/audit SHA `c46c6bf1…4e18`/`94b9e200…be57` | 9/9 run PASS，accuracy/macro-F1 均 1.0，confusion `diag(28000,28000,6927,28784)`；通过 epoch Raw `824/864/797`、FMT-only `61/60/58`、Raw+FMT `54/55/52`；最小 margin Raw `.779/1.053/.672`、FMT-only `2.925/3.472/3.597`、Raw+FMT `4.189/4.308/3.531`；串行作业中 Raw seed 7068 同样在第 824 epoch PASS（同型 GPU 上逐 epoch 一致） | 支持：合并两 volume 后数据、标签、索引与训练循环无缺陷，三种输入都能完全记忆 proxy 标签；FMT 输入使记忆所需 epoch 从约 800 降到约 55（1.1 单场为 550→25–29）。反对/边界：只证明有限样本可记忆，不含泛化、测试性能或 FMT 优势结论 |

| Verify_Task123_StrongBaselines_1.1（完成，Ibex confirmation，独立审计） | 2026-09-02 | Task1/Task2/Task3 更强非FMT baseline | Task1仅在development 0–7搜索4种time-domain和3种普通逐坐标离散Fourier表示及冻结PCA维度，随后在confirmation 8–9复核；Task2比较已完成网格中的最强Raw-VAE、train-only Raw-PCA压到189维后使用同一VAE、普通逐坐标DFT使用同一VAE、21对轨线距离DFT使用同一VAE；Task3比较原Raw与更宽Raw-only监督网络。所有选择均早于confirmation，主FMT结果不重训、不改写 | `FMT_Utils/RobustnessFeatures_3D.py`，`experiments/Run_Task123_StrongBaselines_1_1.py`，`experiments/Audit_Task123_AdditionalEvidence_1_1.py`，`config/Verify_Task123_StrongBaselines_1.1.yaml` | jobs `51218258–51218265`及修复/审计链`51219246,51220232,51220233,51220282`；100/200/100条Task1/2/3记录；summary/audit SHA=`eb31f471…a9ef`/`fc32dcc7…5cbf` | Task1 strongest plain DFT F1`.584665`，FMT`.601384`，FMT增益`+.016719`；time-domain仅`.334718`。Task2 strongest searched Raw-VAE`.539739`，dimension-matched Raw-PCA189+同一VAE`.580365`，plain DFT`.531378`，pair-distance DFT`.240909`，FMT+同一VAE`.584040`，相对Raw-PCA189仅`+.003675`。Task3 Raw/Raw-wide F1`.637015/.633984`，FMT`.858338`，相对Raw-wide增益`+.224354`；Raw-wide AP`.685310`，FMT AP`.923570`，增益`+.238260` | **强控制显著收窄Task1和Task2主张。**Task1支持FMT略优于普通DFT，但优势不是大幅；Task2原始`+.09560`只适用于直接Raw输入，同维PCA控制下FMT近乎打平（`+.00368`），不能再把大幅增益解释为FMT独有结构贡献。Task3在增加Raw容量后仍有大幅优势，核心结论保持。原summary job因历史路径别名缺失失败，首次audit因错误禁止合法负ARI失败；均保留并由不改指标的确定性修复链完成。 |
| Ablation_Task123_FMTComponents_1.1（完成，Ibex confirmation，独立审计） | 2026-09-02 | FMT组件消融 | Task1/Task2的full与6个消融均保持189维，删除信息用零mask替代，VAE结构、latent、优化器、训练步数和名义参数量不变；消融为chirality、real/imag cosine、六条neighbor Fourier blocks、center Fourier block、kinematic block及kinematic-only。Task3对已冻结`aivd1w3`辅助表示比较`first`（无Fourier）、`early`（无Fourier）和`dft`，每个版本均配对同宽Raw-PCA并用相同监督网络 | `FMT_Utils/RobustnessFeatures_3D.py`，`experiments/Run_Task123_FMTComponentAblation_1_1.py`，`experiments/Audit_Task123_AdditionalEvidence_1_1.py`，`config/Ablation_Task123_FMTComponents_1.1.yaml` | jobs `51218268–51218274`全部exit 0；350/350/300条Task1/2/3记录；独立审计PASS且Task3容量配对通过；summary/audit SHA=`b66b6be2…9846`/`5c5a37c4…4312`；审计后临时checkpoint清零 | Task1 full`.601384`；去neighbor Fourier`−.330624`，kinematic-only`−.356284`，去chirality/cosine各`−.018886/−.018303`，去center/kinematic仅`−.005762/−.001779`。Task2 full`.585456`；对应变化为`−.113691, −.289849, −.036535, −.034568, −.010718, −.010645`。Task3 DFT full F1/AP`.858644/.923535`；无Fourier的first为`.858304/.923419`（仅`−.000340/−.000116`），early为`.855293/.921655` | **Task1/Task2中neighbor Fourier blocks是不可替代的主贡献，chirality与cosine有较小但一致贡献；center和kinematic单独移除影响较小。Task3则不支持“时间Fourier是主要增益来源”：first-only几乎复现full，因此Task3只能归因于完整几何辅助表示，不能把监督增益专门归因于其时间Fourier分量。**full replay与论文主实验的微小差异来自独立重训，不替换主表。 |
| Verify_Task123_StrongBaselines_1.2（完成，Ibex confirmation，独立审计 PASS，新种子复现） | 2026-09-03 | Task1/Task2/Task3 更强非FMT baseline 复现 | 1.1 的完整复现，只换随机种子：Task1 selection KMeans seed `7068→7168`、final KMeans seeds `7080–7084→7180–7184`；Task2 VAE seeds `100–104→110–114`、latent KMeans 与 train-only PCA189 random state `7068→7168`。Task1 新增第三个冻结 family `geometric_statistics`（审计要求的曲率/扭率/速度统计 baseline）：`geometric_statistics` 每线14项统计共98维、`geometric_sequences` 逐采样 speed/arcsinh(curvature)/arcsinh(torsion) 共630维，均为刚体运动不变、无标签；三个 family 各自只用 development 0–7 冻结一个赢家再打开 confirmation 8–9。Task3 无可训练阶段，只对冻结 Raw/Raw-wide checkpoint（seeds 40–44）做推理复放，无法换种子。1.1 输出不动 | `FMT_Utils/RobustnessFeatures_3D.py`（新增 `pathline_geometric_*_3d`），`experiments/Run_Task123_StrongBaselines_1_1.py`（family 改为 config 驱动，默认仍为1.1两族），`experiments/Audit_Task123_AdditionalEvidence_1_1.py`，`tests/test_robustness_features_3d.py`（+4测试：螺旋线解析曲率/扭率、刚体不变、反射翻转扭率符号、静止线有限），`ibex_bash/task123_replication_1p2_{cpu,gpu}.sh`，`ibex_bash/submit_task123_replication_1p2.sh`；commit `eeee26d5`（1.1 部署原版另入 `38888a9f`） | `config/Verify_Task123_StrongBaselines_1.2.yaml`；部署包 `tmp/task123_replication_1p2_source.tar` SHA-256 `7e9e3826…f568`；jobs `51277929–51277937`；Task1 选择数组在 tangaroa/f22raptor/boeing747 因静止 primitive 的 0/0 曲率失败，依赖链 51277930–51277932/51277936/51277937 取消，修复（commit `ea176909`，包 `ba5338eb…0379`）后由 `51278339–51278342, 51278347, 51278348` 重跑全部 Task1 阶段（登记见 `ibex_run_registry.md`） | 150/200/100条记录，独立审计 `PASS`；summary/audit SHA `3e494ade…`/`448b7def…`。Task1 confirmation dataset-macro F1：plain DFT `.584334`（1.1 `.584665`）、time-domain `.332903`（1.1 `.334718`）、**geometric_statistics family `.447063`**（赢家 `geometric_sequences`/PCA-2；98维统计版 development 最高仅 `.40730`）；FMT 主表 `.601384`，相对曲率/扭率/速度 baseline 增益 `+.1543`，该 baseline 只在 halfcylinderRe640 一个条目高于 plain DFT（`.4277` vs `.4016`），与 Raw-PCA2 `.4510` 相当。Task2：raw_tuned `.540702`（1.1 `.539739`）、raw_pca189_same_vae `.577034`（`.580365`）、plain_dft_same_vae `.552601`（`.531378`）、pair_distance_dft `.237629`（`.240909`）；同种子110–114的 FMT full 回放（消融1.2）为 `.578509`，相对 Raw-PCA189 仅 `+.0015`（1.1 为 `+.0037`）。Task3 Raw/Raw-wide F1 `.637003/.633984`、AP `.684761/.685312`，与1.1差≤1.2e-5（推理复放） | **1.1 的三条结论全部随种子稳定**：Task1 FMT 仅略优于普通 DFT（`+.0171`）；Task2 与维度匹配 Raw-PCA189 近乎打平（`+.0015`，1.1 `+.0037`）；Task3 增加 Raw 容量不缩小差距。新结论：曲率/扭率/速度统计不是有竞争力的 Task1 baseline（低于 FMT `.154`，低于普通 DFT `.137`），因此 FMT 相对 Raw-PCA2 的 `+.1504` 不能由简单微分几何量解释，但仍应同时报告更强的普通 DFT 对照。Task2 plain DFT 两版相差 `.021`，说明同一 VAE 下 5 个种子的 macro 波动可达 .02，Task2 中小于 .02 的差异不应解读 |
| Ablation_Task123_FMTComponents_1.2（完成，Ibex confirmation，独立审计 PASS，新种子复现） | 2026-09-03 | FMT组件消融 复现 | 1.1 的完整复现，只换随机种子：Task1 KMeans seeds `7180–7184`；Task2 VAE seeds `110–114`、latent KMeans seed `7168`；Task3 residual head 训练 seeds `140–144`，逐一配对冻结 Raw backbone seeds `40–44`（只有这5个 backbone 存在），FMT 与 Raw-PCA 两臂仍共用 backbone、宽度、结构与预算。7个189维零mask变体与3个Task3辅助表示变体与1.1相同。audit 的 full-recipe 回放容差 `0.02→0.03`（主表用的是不同种子，0.03≈五种子macro差的两个标准差）；Task3 临时 checkpoint 仅在独立审计 PASS 后由 cleanup 删除 | `experiments/Run_Task123_FMTComponentAblation_1_1.py`（新增 `task3.backbone_seeds` 配对），`experiments/Verify_Task3_FMTResidual.py`（`_train_one`/`_strong_validation_baseline` 新增可选 `raw_backbone_seed`，默认等于训练 seed，1.1 行为不变），`experiments/Audit_Task123_AdditionalEvidence_1_1.py`（`replay_tolerance` 可配置）；commit `eeee26d5` | `config/Ablation_Task123_FMTComponents_1.2.yaml`；同一部署包 SHA-256 `7e9e3826…f568`；jobs `51277938–51277944` | 350/350/300条记录，独立审计 `PASS`（summary/audit SHA `b2ff74fb…`/`c1de83f8…`），300个临时checkpoint已清理。full 回放：Task1 `.601442`（主表 `.601384`）、Task2 `.578509`（主表 `.584040`，−.0055）、Task3 dft fmt/raw_pca `.859219/.638983`（主表 `.858338/.639844`）。相对 full 的 dataset-macro F1 变化（1.1→1.2）：Task1 without_neighbor_fourier −.3306→−.3307、kinematic_only −.3563→−.3563、without_chirality −.0189→−.0190、without_cosine −.0183→−.0184、without_center_fourier −.0058→−.0058、without_kinematic −.0018→−.0019（Task1 只换 KMeans 种子，n_init=20 收敛到同一解）；Task2 without_neighbor_fourier −.1137→−.1067、kinematic_only −.2898→−.2784、without_chirality −.0365→−.0231、without_cosine −.0346→−.0302、without_kinematic −.0106→−.0120、**without_center_fourier −.0107→+.0004**；Task3 first-only F1 −.0003→−.0023（AP −.0001→−.0003）、first+early mean −.0034→−.0075（AP −.0019→−.0077） | **两条主结论随种子稳定**：neighbor Fourier blocks 仍是 Task1/Task2 的主要组件（−.331/−.107）；Task3 去掉时间 Fourier 仍只降 −.0023，不能把监督增益归因于时间 Fourier。chirality 与 real/imag cosine 在两任务两版均为负贡献（Task2 −.023–−.037）。**修订**：此前写 center Fourier block 有小幅一致贡献（1.1 Task2 −.0107），1.2 Task2 为 +.0004，因此 center Fourier block 与 local kinematic block 在 Task2 的贡献在种子噪声（约 .02）之内，不能声称一致；Task1 的 −.006/−.002 虽稳定但同样太小，不进入论文主张 |
| Verify_Task123_NoiseRobustness_1.1（完成，Ibex confirmation，独立审计 PASS） | 2026-09-02 | Task1/Task2/Task3 噪声鲁棒性（FMT vs 原 Raw 对照） | 所有可训练模型与聚类器只在 clean 数据上拟合（normalization、PCA、KMeans 中心、簇名校准、epoch/alpha/threshold 均来自 clean train/validation）；只对 confirmation pathline 施加 10 种确定性扰动：clean、各向同性高斯坐标噪声 σ=0.05/0.10/0.20/0.40×邻居偏移距离、随机丢帧 10/25/40%（端点保留、线性插回原时间网格）、只保留前 75%/50% 轨迹再重采样为 32 点。扰动实现由 `corruption_seed=17068`、数据集、切片、条件与重复种子确定，两臂配对。Task1 raw(PCA-2)/fmt_all+kin4(PCA-8) 各 5 个 KMeans seed；Task2 raw/fmt 同一 VAE 5 个训练 seed（VAE 只在内存中）；Task3 raw_pca/fmt residual 5 个 seed（临时 checkpoint 审计后删除）。首次数组的 4+15 个 child 被过严的跨 GPU FFT 绝对误差校验拒绝，修复为尺度无关的 relative-L2 阈值 5e-4 后补跑，不改变协议 | `experiments/Run_Task123_NoiseRobustness_1_1.py`，`FMT_Utils/RobustnessFeatures_3D.py::corrupt_pathline_primitives_3d`，`experiments/Audit_Task123_AdditionalEvidence_1_1.py --kind noise` | `config/Verify_Task123_NoiseRobustness_1.1.yaml`；部署包 `9f06aaae…abee5`；jobs `51218275–51218282`、修复 `51221948/51221949`；summary/audit 见 `outputs/Verify_Task123_NoiseRobustness_1.1/` | 每任务 1000 条记录（10 datasets×2 arms×5 seeds×10 conditions），独立审计 `PASS`，clean 回放与主表差 ≤.002。dataset-macro F1（Raw→FMT，括号为相对 clean 的保留率）：**丢帧 10–40%**：Task1 `.451→.605`（1.00/1.01）、Task2 `.492→.572–.584`（1.00/0.98）、Task3 `.639→.856–.859`（1.00/1.00），三任务增益与 clean 相同；**截断 75%**：Task1 `.279→.567`（0.62/0.94）、Task2 `.535→.580`、Task3 `.502→.784`（AP `.614→.926`），FMT 增益扩大；**截断 50%**：Task1 `.140→.282`、Task3 `.353→.597`（AP `.449→.882`），但 Task2 `.564→.338`（Raw 反而升至 1.15 倍，FMT 降至 0.58）；**高斯 σ=0.05**：Task1 `.166→.173`（Raw 保留 0.37，FMT 0.29）、Task2 `.386→.153`（0.79/0.26）、Task3 `.635→.597`（0.99/0.70，AP `.686→.757`）；σ≥0.10 时 Task1/Task3 两臂都接近随机（Task1 F1 .13–.16），Task3 FMT F1 保留率 .51→.23、AP 低于 Raw-PCA | **FMT 对时间采样扰动（丢帧、截断）鲁棒，且截断下增益扩大；对坐标高斯噪声脆弱。**高斯 σ=0.05×邻居偏移 就让 Task1 两臂和 Task2 FMT 崩到接近随机，Task3 FMT 比 Raw-PCA 退化更快（F1 −.04，σ=0.10 起 −.16 到 −.23）。逐数据集看，Task1 σ=0.05 下 10 个流场中 9 个 FMT 与 Raw 都跌到 .05–.15，只有 halfcylinderRe6400 的 FMT 保持 .525，说明 0.05 已超过大多数流场的相对位移信号幅度，这组高斯等级对'退化曲线'信息量不足，0–0.05 之间需要更细的等级（另立版本）。Task3 的脆弱与 `aivd1w3` 是十字模板有限差分涡量估计一致（差分放大噪声）。结论边界：无 noise-augmented 训练；未测更强 baseline（见 1.2） |
| Verify_Task123_NoiseRobustness_1.2（完成，Ibex confirmation，独立审计 PASS） | 2026-09-03 | Task1/Task2/Task3 噪声鲁棒性：更强非FMT baseline | 与1.1完全相同的10个扰动条件、`corruption_seed=17068`、重复/训练种子与数据划分，因此每条被扰动的 confirmation pathline 与1.1中 FMT/Raw 臂看到的逐位相同；新增臂：Task1 `plain_dft_magnitude`/PCA-64（StrongBaselines 1.1 冻结赢家）与 `geometric_sequences`/PCA-2（1.2 冻结赢家），KMeans seeds 7080–7084；Task2 `raw_pca189_same_vae`、`raw_tuned`（`m256_l16_lr3e-4`）、`plain_dft_same_vae`，训练 seeds 100–104，VAE 只在内存；Task3 冻结 `raw`/`raw_wide` 网络（seeds 40–44）推理。所有变换只在 clean train 上拟合，运行器核验各臂与冻结清单一致。summary 先按1.1独立审计的SHA-256核验其 per-run 行再并入，输出总表与逐条件 FMT−各baseline 配对增益 | `experiments/Run_Task123_NoiseRobustness_1_2.py`（复用1.1的扰动与记录工具、StrongBaselines 的输入变换），`experiments/Audit_Task123_AdditionalEvidence_1_1.py --kind noise-strong`，`ibex_bash/task123_noise_1p2_{cpu,gpu}.sh`，`ibex_bash/submit_task123_noise_1p2.sh`；commit `a85c3ec9` | `config/Verify_Task123_NoiseRobustness_1.2.yaml`；部署包 `bebd868a…1dab`；jobs `51299171–51299176` | 新增记录 1000/1500/1000，独立审计 `PASS`（summary/audit SHA `683463e3…`/`0836cdda…`），9 个新臂 clean 复放与强 baseline 确认值差 ≤.0016。dataset-macro F1（保留率）：**Task1** σ=.05：Raw `.166`(.37)、plain DFT `.360`(.62)、几何序列 `.259`(.58)、FMT `.173`(.29)；σ=.10：`.164/.289/.202/.134`；丢帧 40%：`.451/.583/.407/.606`；截断 50%：`.140/.512/.274/.282`。**Task2** σ=.05：Raw `.386`(.79)、Raw-PCA189 `.109`(.19)、Raw-tuned `.412`(.76)、plain DFT `.411`(.78)、FMT `.153`(.26)；丢帧 40%：`.492/.316/.541/.529/.572`；截断 50%：`.564/.305/.596/.460/.338`。**Task3** σ=.05 F1：Raw `.630`(.99)、Raw-wide `.630`(.99)、Raw-PCA `.635`(.99)、FMT `.597`(.70)，AP `.681/.683/.686/.757`；σ=.10 F1 `.594/.602/.594/.436`、AP `.662/.672/.665/.655`；丢帧 40% F1 `.637/.633/.639/.856`；截断 50% F1 `.360/.372/.353/.597`、AP `.459/.469/.449/.882`。逐数据集 Task3 σ=.05：FMT 在 boeing/channel/两个 deltaWing 崩到 .06–.52（Raw-PCA .10–.84），在 f22/Re640/Re6400/tangaroa/smoke/cylinder3d 仍高于所有 Raw 臂 | **对时间采样扰动（丢帧、截断）FMT 在三任务都是最鲁棒或并列最鲁棒的表示，增益保持或扩大；对坐标高斯噪声 FMT 是最脆弱的表示之一。**Task1：普通 DFT 在所有高斯等级和截断下都明显高于 FMT（σ=.05 `.360` vs `.173`），FMT 相对普通 DFT 的 clean 优势 +.017 在任何扰动下都消失；几何序列 baseline 也比 FMT 抗噪。Task2：Raw-tuned 与 plain DFT 在 σ=.05 保留 76–78%，FMT 只保留 26%；Raw-PCA189 是唯一比 FMT 更脆弱的臂（PCA 方向被噪声占据）；丢帧下 FMT 仍最高。Task3：三种 Raw 网络在 σ=.05 保留 99%，FMT 保留 70% 且 F1 低于全部 Raw 臂（AP 仍最高 `.757`）；σ≥.10 起 F1 与 AP 均低于 Raw；截断下 FMT 优势最大（+.22–.28 F1、+.41–.43 AP）。论文表述必须写成：FMT 的增益对轨迹时间稀疏与截断稳健，但对逐点坐标噪声不稳健，普通 DFT/Raw 网络在高斯噪声下更稳；这与 FMT 依赖邻居相对位移和 `aivd1w3` 有限差分一致。边界：无 noise-augmented 训练；0–.05 之间无采样，Task1 两臂在 σ=.05 已接近随机，退化曲线需另立低等级版本 |
| Verify_Task123_NoiseTypes_1.1（完成，Ibex confirmation，独立审计 PASS） | 2026-09-04 | Task1/Task2/Task3 噪声类型扫描：FMT vs 原 Raw | 与 NoiseRobustness 1.1 相同协议（只在 clean 上拟合，只扰动 confirmation pathline，`corruption_seed=17068`，两臂配对），条件换为 22 种：高斯白噪声 σ=.005/.01/.02/.03、时间平滑高斯 σ=.05/.10/.20、共模噪声 σ=.05/.10/.20、邻居种子偏移 σ=.05/.10/.20（中心线不动）、绕种子点刚体旋转 15°/45°/90°、脉冲离群点 2%/5%/10%（跳幅 1.0 邻居间距）、单调时间弯曲 10%/25%/50%。预注册声明：全部条件与臂无论结果如何都报告，不得只挑对 FMT 有利的扰动 | `FMT_Utils/RobustnessFeatures_3D.py::corrupt_pathline_primitives_3d`（新增 6 kinds + `_moving_average_time`/`_rodrigues`），`experiments/Run_Task123_NoiseRobustness_1_1.py`（复用），`tests/test_robustness_features_3d.py`（+1 结构化扰动不变量测试），`ibex_bash/task123_noisetypes_1p1_{cpu,gpu}.sh`，`ibex_bash/submit_task123_noisetypes_1p1.sh`；commit `789b7162` | `config/Verify_Task123_NoiseTypes_1.1.yaml`；部署包 `7143f6ca…5acb`；jobs `51312626–51312632` | 每任务 2300 条，独立审计 `PASS`（summary/audit SHA `efc8b0c8…`/`8d4f4b58…`），clean 复放与主表差 ≤.0011。FMT dataset-macro F1（Raw 对照见 NoiseTypesStrong 行的合并表）：Task1 刚体旋转 15/45/90° `.601/.601/.601`（严格不变）、邻居偏移 `.601/.601/.602`、共模 `.609/.609/.570`、时间弯曲 `.608/.537/.371`、细高斯 σ=.005/.01/.02/.03 `.336/.270/.242/.206`、平滑高斯 `.177/.134/.136`、脉冲 `.148/.138/.142`；Task2 刚体旋转 `.585×3`、邻居偏移 `.584/.584/.581`、共模 `.479/.405/.326`、时间弯曲 `.542/.361/.243`、细高斯 `.297/.270/.232/.189`、平滑 `.152/.151/.152`、脉冲 `.163/.153/.151`；Task3 F1 刚体旋转 `.745/.702/.680`、邻居偏移 `.852/.832/.801`、共模 `.859/.858/.854`、时间弯曲 `.848/.806/.723`、细高斯 `.763/.702/.668/.650`、平滑 `.613/.488/.281`、脉冲 `.431/.299/.225`（AP 细高斯 `.887/.830/.807/.792`） | 见 NoiseTypesStrong_1.1 行的合并结论。单独看 FMT：对刚体旋转和邻居种子偏移按构造完全不变（Task1/2）；Task3 的 residual 模型含 Raw 主干，旋转下降到 .68–.75 但仍远高于纯 Raw；对逐点独立噪声极敏感：Task1 在 σ=.005（平均位移 ~7e-6，约邻居间距的 0.4%）就从 .601 跌到 .336 |
| Verify_Task123_NoiseTypesStrong_1.1（完成，Ibex confirmation，独立审计 PASS） | 2026-09-04 | 噪声类型扫描：更强 baseline | 与 NoiseTypes_1.1 逐位相同的 22+1 条件与扰动实现；臂与 NoiseRobustness 1.2 相同：Task1 plain_dft_magnitude/PCA-64、geometric_sequences/PCA-2；Task2 Raw-PCA189、Raw-tuned、plain DFT 同一 VAE；Task3 冻结 raw/raw_wide。summary 核验 A 组审计哈希后合并为 13 臂×23 条件总表 | `experiments/Run_Task123_NoiseRobustness_1_2.py`（复用），`experiments/Audit_Task123_AdditionalEvidence_1_1.py --kind noise-strong`；commit `789b7162` | `config/Verify_Task123_NoiseTypesStrong_1.1.yaml`；同一部署包；jobs `51312633–51312635, 51312645–51312647` | 2300/3450/2300 条，独立审计 `PASS`（summary/audit SHA `c23e1433…`/`73470164…`），9 个臂 clean 复放差 ≤.0020。合并 13 臂×23 条件总表见 `outputs/Verify_Task123_NoiseTypesStrong_1.1/robustness_table.csv`。FMT 相对最强 baseline 的 F1 差（Task1/Task2/Task3）：clean `+.017/+.005/+.219`；刚体旋转 90° `+.154/+.412/+.491`；邻居偏移 .20 `+.013/+.056/+.231`；共模 .10 `+.024/−.132/+.256`；时间弯曲 .25 `−.048/−.167/+.174`；细高斯 σ=.005 `−.247/−.240/+.123`、σ=.03 `−.276/−.300/+.011`；平滑高斯 .10 `−.145/−.138/−.056`；脉冲 5% `−.250/−.092/−.179`。最强 baseline 的身份：Task1 几乎总是普通 DFT（幅度谱对时间弯曲和 σ≤.02 高斯几乎不变：`.584/.594`），Task2 是 Raw-tuned 或普通 DFT VAE，Task3 是 Raw-wide/Raw-PCA | **按预注册全部报告。FMT 明确更鲁棒的扰动类型**：刚体旋转（三任务都是唯一不崩的臂，baseline 在 45° 起跌到 .1–.3）、邻居种子偏移（Task2/Task3 领先 +.06/+.23，Task1 与普通 DFT持平）、Task3 下的共模噪声与时间弯曲（Raw 网络分别跌到 .53 与 .60，FMT 保持 .85 与 .72）、Task3 下 σ≤.03 的细高斯（F1 仍领先，AP 领先 +.10–.20；与 NoiseRobustness 1.1 合看，FMT 在 σ≈.03–.05 之间被 Raw 追平）。**FMT 更脆弱的类型**：逐点独立噪声（细高斯在 Task1/Task2 从 σ=.005 起、平滑高斯、脉冲离群点）——普通 DFT 的幅度谱在 Task1 对 σ≤.02 高斯几乎不变（.583–.594）而 FMT 已掉到 .24–.34；Task2 下的共模噪声与 25% 以上时间弯曲。预注册预期中“共模与时间弯曲下 FMT 仍脆弱”在 Task3 被推翻、在 Task2 成立；“细高斯下脆弱”在 Task3 被推翻（σ≤.03 仍领先）。机制：FMT 的旋转/平移不变量对坐标系与种子放置误差免疫，但其无量纲比值（余弦、手性）在相对位移信号极小时被逐点噪声翻转；Task3 的 1 维 `aivd1w3` 是有限差分量，对平滑漂移与脉冲最敏感。论文表述：FMT 的鲁棒性优势在于观察者/坐标系与采样几何误差（旋转、种子偏移、共模抖动、时间轴弯曲），不在于逐点测量噪声 |

## 2026-09-06 Task1/2/3 论文三联图预测复放

`Other_Task123_PaperTriptychs_1.1`：按用户指定先绘制 Half-cylinder Re160 与 Tangaroa。
只复放当前统一 Task1-4.1、Task2-6.2、Task3-9.2 配方，冻结实现不修改。
预先固定 Task1 seed7080、confirmation ordinal0；Task2 seed100、ordinal8；
Task3 seed40、ordinal8，均为各自原协议首个注册seed和首个确认时间片。
Task2重训同一VAE；Task3重训相同配置的两臂residual，沿用原冻结Raw backbone，
临时模型只用于导出逐点预测，导出后删除，不下载模型。复放指标只检查图的数据身份，
不覆盖主表，不按图像效果或confirmation数值选择时间片、seed或阈值。
代码 `experiments/Export_Task123_PaperTriptychs_1_1.py`；配置
`config/Other_Task123_PaperTriptychs_1.1.json`；输出
`outputs/Other_Task123_PaperTriptychs_1.1/`。2026-09-06已完成6组效果图，每组分别导出论文/PPT版，
共42个图片文件；预测job51391980[0-5]、最终渲染job51392430。原始场重建标签零错配，
独立数据审计在Ibex执行成功（退出码0），报告留在远端。该审计重算逐片指标并记录与冻结逐次表的复放差异，
未要求复放与原训练浮点结果完全相同，也未据差异选图。图中F1是预先指定单时间片的精确率与召回率调和平均，
不能替代跨流场/重复实验的主表结论。本次不新增方法优劣结论。
用户授权仅下载图片，未下载预测/模型。12份PDF字体与排版检查及逐图目视检查完成；
方向标跨白色背景边界警告已记录并接受。详见[成图说明](Other_Task123_PaperTriptychs_1.1.md)。

## 定量协议（v1，改动需升版本并注明）

- 数据：cylinder2d, doublegyre2d, beads2d, pipedcylinder2d；时间窗 [0.6, 0.8]×T；参数沿用 `PathlineFMTclustering.yaml`（dt=0.005, max_steps=300, L=30, offset=0.02, grid 0.25）。
- 参考标签：(a) IVD 阈值（`ScalarField2d.compute_ivd_2D`，阈值=全数据集固定分位，非逐切片）；(b) Vatistas 合成场解析标签（`experiments/VatistasFlowDatasetGenerator.py`）。
- 指标：ARI、NMI、对参考标签的 F1（聚类簇经匈牙利匹配后计）；KMeans 固定 random_state，特征先做显式标准化并记录方式。
- 确定性：固定全部 seed；encoder 显式 `.eval()` 或显式逐片标准化，二选一并记录（两者都测一次，作为 P1 的复核）。

## 2026-09-06 Task4-b 4.1 新标签共享网络过拟合结果

| 实验版本 | 日期 | 任务 | 技术细节 | 主要代码路径 | Config / commit / 证据 | 主要指标 | 支持、反对的结论 |
|---|---|---|---|---|---|---|---|
| Verify_Task4B_VelocityCurlMemorization_4.1（完成，Ibex及本地审计PASS） | 2026-09-06 | Task4-b：channel+TBL 单个共享FMT网络四分类过拟合 | 用户重定义四类为 ordinary-streamwise、ordinary-spanwise、hairpin-head、hairpin-leg；仅 hairpin membership × velocity-curl 45°明确二分（反平行同属平行、45°归平行），无额外head判据。重算curl，全域体积加权IVD；a为正标注原始单元中心+目标体素中心IVD最小值下一个较小float64，涡区IVD>0.9a，无手工覆盖。全流场网格、普通两类每flow各随机最多8192点、全部有效hairpin点；七线×33点、无量纲单位速度积分，FMT161→1024→1024→4；单个fmt_only、seed7068、Adam1e-3、dropout/weight decay=0、每epoch全样本无放回一次、连续3epoch零错误且正确类最小logit优势>0；fit=evaluation | FMT_Utils/Task4B_VelocityCurlLabels_3D.py；experiments/Verify_Task4B_VelocityCurlMemorization.py（复用旧_train_one而不修改旧入口）；experiments/Export_Task4B_VelocityCurlReport.py；docs/Task4B_velocity_curl_protocol_4.1.md | config/Verify_Task4B_VelocityCurlMemorization_4.1.yaml，SHA df4c4d7a…761b8；base b0fe6d27 + 19文件manifest f8ccf48a…1a98；jobs 51391665/51391666；P100 dgpu502-33；cache SHA1563a763…3a671；原fit/audit SHA1094e1c5…da460 / 1185e644…cc601；正式报告four_class_report.json SHAbe62ffd8…3c3e1 | 合并47127样本，四类支持15683/15575/10783/5086。第104–106 epoch连续零错误，第106通过；训练33.76秒，GPU作业共1分钟；参数1223684。channel26842、TBL20285样本各自accuracy/macro-F1=1.0，每类F1=1.0；最小正确类logit优势2.98224。Ibex23项审计PASS，本地21项数据+13项预测审计PASS，6项单元/流程测试通过，无checkpoint | 支持：按用户本次四类标签，单个FMT输入网络可同时完全记忆两流场的47127个有效primitive；数值标签、索引、训练输出和覆盖核对通过。边界：仅训练样本记忆，不证明泛化、真实解剖标签正确或FMT优于Raw。严格全覆盖IVD使涡区几乎覆盖全场（channel99.9612%、TBL99.8475%）；不能据此声称得到选择性涡区边界。训练丢弃无法完整积分的1593个候选primitive，TBL实例12无有效primitive，未包含在拟合成功主张中 |

数据构建细节：channel原始721811个正标注单元中心及10539个正标注体素中心的覆盖率均为1.0，a=.1694188708110282、0.9a=.15247698372992538；TBL对应655063/5413，覆盖率均为1.0，a=.01026903406092381、0.9a=.009242130654831428。channel每类有效支持8155/8148/6968/3571，TBL7528/7427/3815/1515；无效primitive分别81/1512（TBL含83个hairpin点），channel73个实例全部有有效primitive，TBL58个中57个有。输入文件SHA与用户本地四个VTK逐字节一致。两个文件的速度均方根分别.907866/.843941，实际物理积分步长.00783834/.21878458；归一化依据实际数据记录，不假定输入速度一定相差10–100倍。

与旧描述并列：此前“94.2%涡区”只是在旧缓存的逐z平面均值偏差上作的诊断；本次是从原始速度重算curl、全域体积加权IVD、全流场网格及全部原始正标注单元中心校准，得到上述99.9612%/99.8475%，两者不是同一量或同一标注支持，不能混用。旧3.1的91711样本记忆结果保持不变，不替代本次新标签结果。

输出位于 outputs/Verify_Task4B_VelocityCurlMemorization_4.1/：four_class_report.json、per_volume_fit_metrics.csv、build_summary.json、fit_summary.json、independent_audit.json、local_data_audit.json、local_fit_audit.json、histories/、predictions/及cache/标签NPZ（无模型文件）。原始复用训练器的指标字典仍沿用类别3旧键hairpin_limb；正式报告从数值预测重算并使用本版名称hairpin_leg，未改变类别定义、标签或数值结果。

## 2026-09-06 Task4-b 0.98a 代理真值图片（不训练）

| 实验版本 | 日期 | 任务 | 技术细节 | 主要代码路径 | Config / commit / 证据 | 主要指标 | 支持、反对的结论 |
|---|---|---|---|---|---|---|---|
| Other_Task4B_ProxyGTThreshold_4.2（仅标签后处理与三维出图） | 2026-09-06 | Task4-b channel+TBL ground truth | 按用户要求保留4.1每flow原始a，将严格IVD>0.9a改为IVD>0.98a；四类方向规则和hairpin成员保持不变；使用完整网格，包含无法积分有效primitive的GT；不训练、不读取预测 | experiments/Visualize_Task4B_ProxyGroundTruth_3D.py | config/Other_Task4B_ProxyGTThreshold_4.2.json SHA803a016f…40b28；输入版本4.1 base b0fe6d27；jobs51392331/51392397；outputs/Other_Task4B_ProxyGTThreshold_4.2/summary.json、local_groundtruth_audit.json；label SHA channel dd1bf629…de44、TBL ae967d8f…ff99 | channel阈值.15247698372992538→.16603049339480763，涡区99.9611977%→99.9501186%，移除219体素；TBL阈值.009242130654831428→.010063653379705334，99.8475105%→99.7958363%，移除1020体素。原始GT单元中心及GT体素覆盖均100%；本地独立完整网格核对通过 | 提高到0.98a仍将几乎全部网格视为涡区，修改幅度小；此记录仅描述用户指定阈值的标签结果，不支持模型性能或泛化结论，旧0.9a训练结果保持不变 |

四类完整网格支持数（ordinary_streamwise / ordinary_spanwise / hairpin_head / hairpin_leg）：channel 618613 / 1346550 / 6968 / 3571；TBL 599852 / 1364611 / 3883 / 1530。三维总览为真实空间比例的透视体渲染，类别最近邻采样；普通两类半透明，全部hairpin另有不透明体素几何图。figures_r2仅关闭VTK过大坐标字母，标签文件未改写，原首版图和源码保留。

## 2026-09-06 Task4-b 实际最小值 a，严格 IVD>a（仅出图）

| 实验版本 | 日期 | 任务 | 技术细节 | 主要代码路径 | Config / commit / 证据 | 主要指标 | 支持、反对的结论 |
|---|---|---|---|---|---|---|---|
| Other_Task4B_ProxyGTThreshold_4.3 | 2026-09-06 | Task4-b ground truth 后处理 | a为原始正标注单元中心与目标正标注体素中心联合IVD的实际最小值，不再取nextafter；严格IVD>a排除等值点，四类方向判据不变；不训练、不读取预测 | experiments/Visualize_Task4B_ProxyGroundTruth_StrictMinimum_3D.py | config/Other_Task4B_ProxyGTThreshold_4.3.json SHA2de747d0…df674；script SHA691f11c9…22c86；输入4.1 base b0fe6d27；job51392510；outputs/Other_Task4B_ProxyGTThreshold_4.3/summary.json与local_groundtruth_audit.json | Channel a=.16941887081102822，涡区99.946121998%；TBL a=.010269034060923811，涡区99.783171032%；相对4.2分别移除79/250体素；独立逐体素标签、阈值最小值及文件SHA检查PASS | 指定a仍接近全场涡区。严格大于排除Channel 1个原始GT单元中心（原始GT覆盖99.99986146%，网格GT100%）及TBL 1个GT体素（网格GT覆盖99.98152596%，原始GT100%）；如实保留边界，不强制补回。无模型性能结论 |

与4.2并列：此前a用实际最小值下一个更小float64、阈值0.98a，Channel/TBL涡区99.9501186%/99.7958363%；现在按用户新要求用实际最小值、阈值a，得到本版结果。这是定义变化，旧数值保留。Channel原始GT与网格GT最小IVD分别.16941887081102822/.2717325463372836，TBL分别.011468771054584097/.010269034060923811；每flow取两者最小，覆盖支持集合沿用4.1/4.2。完整四类支持依序ordinary_streamwise / ordinary_spanwise / hairpin_head / hairpin_leg：Channel 618613/1346471/6968/3571，TBL 599852/1364362/3882/1530。数据SHA：Channel e97a2fd414f5e0c4679e96d21dc9771ce827bc1248992243650313a45022226b；TBL 54ee581ccd4c407015e4e7649d9523bf9dcce0d02965f6852bf652d4b82f42c4。


## 2026-09-06 六流场Task1–3模式a/b图片预登记

`Other_Task123_PaperTriptychs_1.2`：用户指定Re160、Re640、Re6400、Tangaroa、Boeing747、Delta-wing原始LBM。
Task1恢复模式a：几何体+IVD参考面+轨线 / FMT两簇 / FMT相对IVD误差；Task2/3使用模式b：参考 / 无FMT / FMT。
复用1.1的Re160与Tangaroa预测，其余四流场沿用当前统一配方Task1-4.1、Task2-6.2、Task3-9.2；
固定seed7080/100/40、确认序号0/8/8。模式a显示与评估均用p95，区别于旧模式a显示p97；
轨线按旧IVD分层规则积分，仅展示，不变更训练输入。几何体仅使用经核实资产，不根据零速度猜测障碍物。
配置config/Other_Task123_PaperTriptychs_1.2.json；结果outputs/Other_Task123_PaperTriptychs_1.2/。
不据本次图像或确认指标选择时间、阈值、特征或模型；无新增方法级结论。仅下载图片，预测及模型不下载。


`Other_Task123_PaperTriptychs_1.2`完成更新（2026-09-06）：18组三联图、36张论文/PPT版式、126个图片格式文件及3张浏览总览已生成并下载。
预测job51392726全部12child完成，旧两流场复用1.1；最终渲染jobs51392906、51392946全部完成。
数据独立审计留Ibex、退出码0；36份PDF本地最小字号7pt/13pt、碰撞0 FAIL，方向标白底边缘WARN经目视接受。
全部36图已逐栏查看。结果仅作为预先指定时刻的效果图，不据此新增跨流场方法优劣结论，不改变主表；
Boeing Task2单片结果保留其实际对照/FMT差异，未替换种子或时间来改善图片。
说明和图注见docs/Other_Task123_PaperTriptychs_1.2.md，后续默认Task1模式a、Task2/3模式b。

## 2026-09-06 Task4-b 用户指定固定 IVD 阈值（4.4，仅标签与三维图）

| 实验版本 | 日期 | 任务 | 技术细节 | 主要代码路径 | Config / commit / 证据 | 主要指标 | 支持、反对的结论 |
|---|---|---|---|---|---|---|---|
| Other_Task4B_ProxyGTThreshold_4.4 | 2026-09-06 | Task4-b ground truth 后处理 | 用户在软件检查并咨询CFD专家后指定Channel IVD>5.785、TBL IVD>0.08；使用冻结全域IVD物理原值，不归一化阈值、不重算curl、不训练、不读取预测。hairpin成员×velocity-curl四类规则不变，阈值以下ignore=-1，不强制补回GT | experiments/Visualize_Task4B_ProxyGroundTruth_FixedThreshold_3D.py | config/Other_Task4B_ProxyGTThreshold_4.4.json SHAf07fa374…d3790；script SHAda3f000b…4305a；输入4.1 base b0fe6d27；job51395061；outputs/Other_Task4B_ProxyGTThreshold_4.4/summary.json、local_groundtruth_audit.json | Channel涡区229768/1976688=11.62388804%，原始GT保留92.85408507%、网格GT90.24575387%；TBL涡区304347/1973906=15.41851537%，原始GT保留99.93786857%、网格GT99.79678552%。本地独立逐体素、阈值、覆盖、空间网格及SHA核对PASS | 新阈值产生明显更小的涡候选集；无需、也不保证hairpin标注全覆盖。阈值来源为用户指定，不是模型性能搜索；不据此声称物理分类已由独立专家审计。无训练或泛化结论 |

结论修订并列：此前以hairpin区域最小IVD为全场涡区边界（4.3），Channel/TBL分别标为99.946%/99.783%涡区；现在按用户提供的固定界限改为11.624%/15.419%。此前不合理之处是将标注区域的极小值覆盖条件当作全场涡识别阈值依据；数值计算检查通过并不证明该阈值物理合理。旧结果保留以便追溯，不再作为当前Task4-b涡区定义。尚未据此判断用户软件与本代码IVD实现逐点一致。

完整四类支持（ordinary_streamwise / ordinary_spanwise / hairpin_head / hairpin_leg）：Channel 66681/153576/6158/3353，TBL 61563/237382/3873/1529。Channel排除51580个原始GT单元中心与1028个GT体素；TBL排除407与11。新标签SHA：Channel 17d5a9bd9a7e6548bef93ce49e0a789fb90fa41e78c6ab348bab91afed2bfadf；TBL c0dbd809d7885c9d4b69f8aa71afd6f209cf1427fe97e6fbac4cd1b9d1de9f79。三维总览保留4.3相同相机、物理比例和透明度，另输出全部保留hairpin的不透明体素图。

## 2026-09-06 Task4-b 2000个小区域共享FMT分割（5.1，完成）

| 实验版本 | 日期 | 任务 | 技术细节 | 主要代码路径 | Config / commit / 证据 | 主要指标 | 支持、反对的结论 |
|---|---|---|---|---|---|---|---|
| mainExp_Task4B_PatchSegmentation_5.1 | 2026-09-06 | Task4-b channel+TBL共享FMT四类局部分割 | 每flow900train/100test区域，含全部实例体素bbox+1体素padding和随机同尺寸窗口；连通盒实例整体分组，Channel66/7、TBL52/6；跨split盒间隔1体素，区域内可重叠。4.4固定IVD阈值；每点7×33单位速度RK4流线，seed步长受patch内原生插值域限制，FMT161→1024→1024→4，dropout0.1；train-only标准化、inverse-sqrt训练类频权重，Adam1e-3/weight_decay1e-4，200epoch固定最终模型；test最终一次；无checkpoint | FMT_Utils/Task4B_PatchSplit_3D.py；experiments/Task4B_PatchSegmentation_5_1.py；Export_Task4B_PatchVolumes_5_1.py；docs/Task4B_patch_segmentation_protocol_5.1.md | config/mainExp_Task4B_PatchSegmentation_5.1.json SHA83dd47bb…c2213；本地base aee4bd562d340158118f2e41f40129a9918e06a1+新文件，远端旧checkout b0fe6d27通过23文件SHA校验；部署manifest c3105bb6…dc2bd；patch清单3e96f032…fdf67；cache e7d9f696…22936；预测122adebd…aef82；jobs51395970/51395974/51396032/51396143；P100 dgpu502-33 | 唯一test体素（无效primitive计漏检）：Channel12302点，accuracy57.8524%、mIoU25.0146%、macro-F1 36.3563%；TBL12292点，57.8262%、22.1260%、31.9828%。对应训练抽样唯一体素mIoU89.0146%/84.0151%。远端审计及独立本地重算PASS，2000个VTK文件SHA全部核对通过 | 数据拆分、共享模型训练、密集test涡区推理及导出流程完成；当前测试分割效果较差，特别是hairpin两类，不能称为高质量分割。训练/测试差距明显，不从审计PASS推断标签物理正确或模型泛化良好；仅两个已知flow内未见空间区域/实例，不支持跨flow泛化、FMT优于Raw或whole-field涡/非涡识别。未根据本次test结果重训或选择参数 |

四类顺序为ordinary_streamwise / ordinary_spanwise / hairpin_head / hairpin_leg。Channel test IoU分别31.1309%/53.9537%/7.9845%/6.9892%，F1分别47.4806%/70.0908%/14.7882%/13.0653%；TBL IoU19.0132%/57.0455%/7.9610%/4.4843%，F1 31.9514%/72.6484%/14.7479%/8.5837%。主指标对重叠test patch的同一体素概率平均后去重；无效预测为-2计入真实类别漏检。Channel/TBL唯一test体素流线覆盖98.2198%/97.5594%（219/300点无有效primitive）。每patch出现次数指标和valid-only指标另列training_report.json，不混入主结果。

构建共236129个候选、230839个有效primitive。Channel候选136529、有效133870，train有效120498、test有效13372；TBL候选99600、有效96969，train有效83861、test有效13108。普通训练点每patch最多256，hairpin训练点全部；测试patch所有涡体素均列入候选。随机窗口不按IVD或类别筛选，因此Channel训练199/测试26个patch无涡点，TBL训练366/测试52个patch无涡点，均保留在900/100区域数中但不进入四类损失；这不是1800个非空训练区域或200个非空四类测试区域。

与旧结论并列：4.1是0.9a标签、全域固定尺度流线、fit=evaluation的有限样本100%记忆；本次是4.4固定阈值、patch内可变步长、dropout/weight_decay及固定200epoch、独立空间测试。两者定义与评估不同，旧100%结果不支持本版泛化，不作直接方法优劣归因。本版参数1223684，P100训练和最终推理108.78秒，GPU作业155秒；不保留模型。配置/种子/输入/源码清单/历史曲线/逐patch指标/完整预测均保存。

输出：outputs/mainExp_Task4B_PatchSegmentation_5.1/manifest.json、config_snapshot.json、build_summary.json、training_report.json、training_history.csv、predictions.npz、channel_test_segmentation.npz、tbl_test_segmentation.npz、test_patch_metrics.json、audit.json、local_review.json、export_report.json、figures_3d及figures_3d_r2。导出2000个含velocity/vorticity/IVD/VortexIds/proxy_label的局部VTK，测试另含predicted_label与confidence；每flow分别打包并已本地逐文件核对SHA及展开到patch_volumes。逐patch原始场curl重构IVD与冻结IVD最大差异0；VTK重读train/test各抽样检查通过。VTK速度为目标网格采样值，不能假设从该较粗局部速度再差分得到同一IVD。r2仅将普通涡显示透明度改为.04，测试patch选择、标签、预测与指标不变。

## 2026-09-06 JHTDB channel 下载与加载器验证（1.1，完成）

| 实验版本 | 技术细节 | 主要代码路径 | 配置与证据 | 指标与结论边界 |
|---|---|---|---|---|
| Verify_JHTDB_ChannelDownload_1.1 | 复制并修正 PyflowVis JHTDB_Lodader；用户确认局部64³、32时刻t=1+k×0.0065；x=[3,3.3],y=[-0.9,-0.6],z=[0.2,0.5]；getData静止壁面坐标、lag8空间/PCHIP时间插值；逐帧保存、4线程独立dataset续传 | FLowUtils/flowDatasetUtils/JHTDB_Lodader.py；JHTDB_NetCDF.py；experiments/Download_JHTDB_Channel.py；Visualize_JHTDB_Channel.py；Audit_JHTDB_Channel.py | config/Verify_JHTDB_ChannelDownload_1.1.json；outputs/Verify_JHTDB_ChannelDownload_1.1/{manifest,audit,probe}.json；两次执行源码快照与最终源码SHA记录；docs/JHTDB_channel_download_1.1.md | 完成32帧(32,64,64,64,3) float32；19项测试通过；全部帧SHA、NetCDF/逐帧/FMT读取逐值一致；首末xz平面与末帧17随机点独立查询最大差异0；三维体渲染和32帧滑块已目视验证。支持下载/排列/保存的一致性，不独立验证数据库物理精度；无模型或FMT方法效果结论，不改变已有任务数据和阈值 |

时间为插值后的物理时刻，t=1不在原始k×0.0065存储网格上；不是32个未经插值的原始快照。原始channel cutout为移动网格，本次未使用cutout。初次串行完成两帧后中断并改为4线程续传，两次源码与记录保留；全程本地执行，无Ibex作业。

## 2026-09-06 客观性验证四联图代码

`Other_FMTObjectivityTranslation_1.1`：新增纯平移Killing observer四档0/25/50/100%速度对照代码；速度样条积分得到位移，在原坐标拉回位置读取v，再减u；同一Task1-4.1拟合分类器逐档重新计算FMT和标签，不复制原预测。代码experiments/Visualize_FMT_Objectivity_Translation_1_1.py、FMT_Utils/TranslationObserver_3D.py；约定config/Other_FMTObjectivityTranslation_1.1.json；说明docs/Other_FMTObjectivityTranslation_1.1.md。完成7项参考系/轨线/分类回调数值测试及解析fixture两种版式排版检查；真实FMT接口尚未端到端运行。未指定正式目标速度，未提交Ibex作业，不新增方法客观性结论。旧三联图与冻结主实验实现不变。

## 2026-09-06 channel与isotropic重新下载为VTK（完成）

| 实验版本 | 技术细节 | 主要代码路径 | 配置与证据 | 指标与结论边界 |
|---|---|---|---|---|
| Verify_JHTDB_VTKDownload_1.1 | 用户要求新增isotropic并重下非VTK数据；channel保持原局部盒64³、t=1+k×0.0065；isotropic1024coarse取[0,0.4]³、64³、t=1+k×0.002；各32帧；lag8空间插值，channel PCHIP、isotropic原始时刻none；4线程共享总并发；直接二进制STRUCTURED_GRID、PointData velocity | FLowUtils/flowDatasetUtils/JHTDB_VTK.py；experiments/Download_JHTDB_VTK.py；Audit_JHTDB_VTK.py；Visualize_JHTDB_VTK.py | config/Verify_JHTDB_VTKDownload_1.1.json；outputs/Verify_JHTDB_VTKDownload_1.1/{channel,isotropic}/manifest.json；audit.json；source/；old_format_cleanup.json；docs/JHTDB_vtk_download_1.1.md | 两流场各32单帧VTK+1含32时间数组的合并VTK；2项新增非立方/多时间数组VTK测试通过；64帧速度及坐标、文件SHA与合并数组核对通过；原PyFlowVis VTKLoader读取全部时刻及数值完全一致；两流场首末切片和17随机点查询最大差异均0。支持下载/保存/读取一致性，不独立验证数据库物理精度，不训练或改变既有任务阈值 |

此前channel为NetCDF/NPY，现在为重新请求后直接保存的VTK；新旧32帧逐值完全一致，改动原因是用户要求格式及重新下载，并非原速度数据有误。新VTK核验通过后，已删除旧32个frame_NNN.npy与channel.nc，保留历史清单、审计和源码；逐文件旧SHA及删除范围见old_format_cleanup.json。参考OneDrive的channel_flow/channel.vtk只读未改。VTK坐标使用JHTDB原始x,y,z顺序，未复制参考文件的轴置换或未查询的涡量/标签。所有执行均为本地CPU，无Ibex作业。

## Other_Task4B_GeometrySearch_5.2 — network/LR and geometry differences (2026-09-06)

User-authorized extension of frozen mainExp_Task4B_PatchSegmentation_5.1. Earlier 5.1 conclusion remains: high training fit did not yield strong held-out four-class segmentation (test mIoU channel25.0146%, TBL22.1260%). This version tests wider/residual classifiers, learning rates0.001/0.0003, and geometry-derived gradient/curl features; it does not rewrite that result. Config SHAe507d0b9289b24173d893db59f51168e88c0d8303448d33246fd6bee6fc0da7a; protocol docs/Task4B_geometry_search_protocol_5.2.md; methods code experiments/Task4B_GeometrySearch_5_2.py and FMT_Utils/Task4B_GeometricCurl_3D.py. Eight candidates were frozen before any run. Selection uses spatially separated internal validation from original training patches only, followed by refit on all900+900 original training patches. The original 5.1 test is reused once after selection and is not fresh confirmation.

Build51397778: 204359 original valid training occurrences; fit137514, internal validation12623, buffer54222. Baseline161 features, unit/time geometry399 features each; exact cache SHA7dc8501242b915cfe1d92c8ff653fc0ea34da6629ff4a24b13d7820e6c24b0c4. All native interpolation queries stayed within their patches. On fit rows only, a direct 45-degree threshold of geometry-derived tangent/curl gives parallel-versus-perpendicular agreement: channel unit59.2359%, time94.3076%; TBL unit59.8731%, time95.0657%. This is an input/label consistency diagnostic, not four-class neural segmentation or independent test performance. Analytic shear tests explain why: equal-arclength geometry removes speed variation, while equal-time geometry retains relative speed and yields a scaled velocity-curl estimate. Stored label curl is never a network input. Final neural results pending eight preregistered candidates; no performance conclusion yet.

### 5.2 completed: frozen selection and final evaluation

All eight candidates passed the same 4096-row memorization diagnostic (zero errors for three consecutive epochs). This establishes finite-sample fit capability; it is not a claim that the selected final model perfectly fits all original training points. Each candidate then trained from a fresh initialization for 300 epochs on the internal fit set.

| 5.2 candidate | Channel validation mIoU (%) | TBL validation mIoU (%) | Mean (%) | Selected epoch | Memorization pass epoch |
|---|---:|---:|---:|---:|---:|
| baseline_fmt_mlp | 32.2205 | 27.8246 | 30.0225 | 10 | 281 |
| wide_fmt_mlp | 30.4810 | 28.1360 | 29.3085 | 1 | 283 |
| residual_fmt_lr1e3 | 30.8328 | 28.1185 | 29.4757 | 1 | 261 |
| residual_fmt_lr3e4 | 32.2953 | 27.8056 | 30.0505 | 1 | 142 |
| unit_geometry_curl | 39.3485 | 34.8974 | 37.1230 | 1 | 120 |
| time_geometry_curl_lr1e3 | 54.8817 | 56.5686 | 55.7252 | 1 | 152 |
| time_geometry_curl_lr3e4 | 53.2055 | 56.4255 | 54.8155 | 1 | 68 |
| factorized_time_geometry | 53.7086 | 56.2933 | 55.0010 | 1 | 78 |

The selected candidate is time_geometry_curl_lr1e3: equal-time 7x33 geometry, 399 input features including geometry-derived gradient/curl, width512 and four residual blocks, 2,314,244 trainable parameters, Adam learning rate0.001, no dropout/weight decay. The predeclared internal-validation rule selected epoch1 (mean mIoU55.7252%). Full refit therefore used one epoch over all204359 valid original training occurrences from900+900 patches, followed by one original-test inference. Selection SHA1154822e111544cb517564e38e3d780f8a587262786c08799b873dd8971d74cc. Later near-zero fit errors in the 300-epoch searches did not exceed the early validation maximum; no epoch was selected from final-test results.

| Flow | 5.1 test mIoU (%) | 5.2 test mIoU (%) | 5.2 accuracy (%) | 5.2 macro F1 (%) | Ordinary streamwise IoU (%) | Ordinary spanwise IoU (%) | Hairpin head IoU (%) | Hairpin leg IoU (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| channel | 25.0146 | 45.0882 | 78.7514 | 56.8642 | 68.6355 | 77.1279 | 17.9451 | 16.6445 |
| tbl | 22.1260 | 45.2979 | 82.1266 | 55.8791 | 69.0787 | 82.2853 | 18.7597 | 11.0680 |

The 5.1 result is unchanged; 5.2 improves this reused benchmark by20.0736 percentage points for channel and23.1719 for TBL. It does not establish that hairpin segmentation is solved. Confusion matrices show substantial ordinary-vortex false positives as hairpin:

- channel: hairpin membership precision 21.5553%, recall 64.9371%; 1503 ordinary voxels predicted hairpin. Final one-epoch refit training sampled unique-voxel accuracy 81.4116% and mIoU 54.9390% (valid only).
- tbl: hairpin membership precision 19.5341%, recall 76.0465%; 1347 ordinary voxels predicted hairpin. Final one-epoch refit training sampled unique-voxel accuracy 84.0269% and mIoU 53.7084% (valid only).

Evidence: outputs/Other_Task4B_GeometrySearch_5.2/{selection.json,final_report.json,final_test_predictions.npz,candidate_comparison.csv,local_review.json}; complete candidate curves and validation probabilities under search/. Final unique test supports remain12302/12292 including219/300 invalid predictions counted as errors, exactly matching5.1. Internal validation uses the frozen original training-point sample; final test is dense within its vortex regions. This sampling difference and the reused test must accompany interpretation. New geometry versus cached baseline changes feature content, neighbor offset and numerical precision; it is not an isolated FMT ablation. This is one preregistered seed per candidate, not a seed-robust ranking.

All eleven Slurm processes completed. Remote audit and independent local recomputation of every validation and final metric passed; all200 VTK hashes and8 paired perspective-camera/transparency records passed. Four fixed test patches (channel6/10, TBL26/33) were rendered without selecting favorable predictions; the first examples visibly overpredict hairpin extent. Five analytic geometry tests and repository layout/diff checks passed. No model checkpoint was saved or downloaded. Final data SHA: dc67e75eb920f87015a3a3774f1f8b0756de2afa2028bf2779e98a80d5a06b52. Results bundle SHA9b494e30e43d91dbfa33295849da4f03477ab464e20d9c74b7364bdcbb42a69f.

## 2026-09-06 Other_FMTObjectivityTranslation_1.2 — 全局平均平移 observer 预登记

用户指定Cylinder Re160 3D，沿全局平均速度平移。新版本1.2保持1.1代码与记录不变；目标observer速度定义为每时刻完整原始160×60×20网格上的体积平均，非均匀坐标采用梯形积分权重；缺失速度节点联合排除，零速度节点保留，不凭零速度猜测障碍物。自然三次样条插值速度，再积分为位移；0/25/50/100%档分别积分observed field并重新分类。沿用Task1-4.1已冻结fmt_all+kin4、PCA8、seed7080、确认ordinal0，仅在原始训练/校准集合拟合一次；不按图像或确认指标选observer。全局空间均值用于定义观察者，不用于标签或模型选择。本实验测试该平移下的分类变化，不预设客观性成立，也不把均值扣除视为消除所有环境流。代码与配置见绘图方法表P15；9项数值测试通过。Job51401240，结果待运行。

### 1.2 完成结果：Cylinder Re160 平均平移 observer

证据：计算job51401240的cylinder3d/paper.json与observed_pathlines.npz；渲染job51401324复用同一数组，未重新分类。计算代码SHA f2bbc66e7fd863905bb07903f88179dc5b85a26c5d46bcf7252ed1a211a1de60；选择文件SHA f1e76ae43e97aa1d562dd2d6a168581a9ec1a0bba77275aab3e109346ea0a0a7；确认缓存SHA b9aa07f27957023785830f74ae16136cd06af838c6d98f772814fcf97d614783。实际轨线区间t=3.900000095至5.099998951，目标速度节点覆盖3.9至5.2。时间窗平均observer速度(1.0001947954,-8.195038686e-5,1.019448375e-5)，最终位移(1.2002326099,-9.834037044e-5,1.223336883e-5)，均为源数据单位，源文件未声明SI单位。全空间各帧无缺失节点。

| Observer速度比例 | 有效中心轨线 | 预测为涡 | 相对原始标签变化 | 标签一致率 | 固定IVD p95参考F1 |
|---|---:|---:|---:|---:|---:|
| 0% | 3584 | 136 | 0 | 100% | 0.6195899772 |
| 25% | 3584 | 135 | 1 | 99.972098% | 0.6164383562 |
| 50% | 3584 | 135 | 1 | 99.972098% | 0.6164383562 |
| 100% | 3584 | 134 | 2 | 99.944196% | 0.6132723112 |

四档无额外排除；固定参考正类303。直接observed积分与原始轨线坐标变换的最大差0/2.500e-11/4.999e-11/9.998e-11。特征最大绝对变化0/1.99307275/1.99988461/1.99989939，故不能以分类高度一致推断特征不变。结论仅支持：本时间片、此纯平移observer及固定分类器下标签高度一致，但非严格不变；整体流向位移显著减弱。不证明一般observer客观性，不把均值扣除等同去掉所有环境流或背景剪切。未进一步解释两条变化的数值/方法原因；未用它们改模型或调参。

输出7个正式图片文件，论文PNG2880×4079、汇报PNG2376×3366；最终两套面板尺寸PASS，PDF字号最小7/13pt，碰撞0FAIL。各4个WARN均为面板统计文字跨白色axes背景边界；已逐栏目视，文字完整不遮挡轨线，接受并保留REVIEW REQUIRED审计原状态。所有3584条线均绘制；非涡线alpha0.12、涡线alpha0.95在四档固定，放大同一相机zoom2.1，不改数据范围。格式包outputs/Other_FMTObjectivityTranslation_1.2/cylinder3d/Cylinder_Re160_mean_observer_figures.zip；未下载预测数组或模型。1.1此前仅完成代码/解析测试，本次新增真实数据结果，不改写原记录。

## 2026-09-06 Other_FMTObjectivityTranslation_1.3 — cylinder 后半程可视化预登记

用户排除所有cylinder原始模拟前50%的初始阶段，要求初始t>=7。三组原始模拟[0,15]，统一实际下限7.5；Re640源文件已裁到[7.5,15]，不能对该文件再次截半。按已有确认缓存元数据取首个合规时间，不看标签、预测或指标：Re160 ordinal2/t10.5，Re640 ordinal0/t9.5，Re6400 ordinal2/t10.5。已有1.2使用Re160 t3.9；1.3改用后半程，因为用户更新展示时间范围，旧1.2数值不覆盖也不与新时间片作方法优劣比较。新图仍固定Task1-4.1 fmt_all+kin4/PCA8、seed7080、原始训练/校准；每流场目标observer取完整源网格每时刻空间体积均值。所有分类在同一固定分类器下独立计算。长期规则已写AGENTS.md与config/cylinder_time_policy.json；未来新实验按新时间范围建版本，冻结训练/测试划分不追溯修改。Array51407401_[0-2]；结果待运行。

### 1.3 完成：三组cylinder后半程图片

证据：Array51407401_0/_1和补跑51407452；outputs/Other_FMTObjectivityTranslation_1.3/<dataset>/paper.json及observed_pathlines.npz（远端）。前两组运行代码SHA325e7c5658fe50658469504676802dce6dece2d3270119300e77d59cfb2a1a20；Re6400为相同科学计算代码加源文件/observer身份检查，SHA9fad9e65ee9a126e8ca19a5e3139e3cab0978e764441bf1d5f598eba3800f090。完整配置与数据身份在各图JSON。

| 数据 | 初始t / 确认ordinal | 共同有效primitive | 0/25/50/100%预测为涡数 | 0/25/50/100%变化数 | 0/25/50/100%固定IVD参考F1 |
|---|---|---:|---|---|---|
| Re160 | 10.5 / 2 | 3610 | 825/836/851/863 | 0/11/26/38 | 0.52057245/0.51550044/0.50874126/0.50346021 |
| Re640 | 9.5 / 0 | 3577 | 543/562/586/626 | 0/19/43/83 | 0.62093863/0.61411765/0.60411899/0.59299781 |
| Re6400 | 10.5 / 2 | 3621 | 181/184/188/188 | 0/3/7/7 | 0.51152074/0.51716247/0.51700680/0.51700680 |

三组均无额外轨线排除。100%档observed积分与坐标变换的最大物理距离误差分别8.6926e-9、1.1436e-8、8.6889e-8，均小于预设1e-5；该数值一致性不等于分类或特征客观性证明。路径时间窗平均observer速度分别(1.000022507,-0.000212933,-0.000163566)、(1.000093525,-0.000188258,-0.000127206)、(1.000194710,0.000786879,-0.000123216)，使用源数据单位。

旧说法（1.2，Re160 t3.9）：100%档仅2/3584改变。当前（1.3，Re160 t10.5）：38/3610改变。原因是用户明确改为后半程时间片；之前的数字只描述初始阶段，不能外推到后半程。本次未修改原始冻结配方，不能把跨时间差异解释为方法退化/改进。三组本次分类均非严格不变，且相对IVD的F1变化方向不一致；不预设平均平移提升分类准确率。

3组×2版式全部导出，共21张PNG/PDF/SVG/TIFF格式文件；6份面板尺寸报告PASS、最终PDF字号最小7/13pt、碰撞0FAIL。每份4项WARN为统计文字跨白色axes背景边缘，全部24面板逐栏和整图查看后接受，保留REVIEW REQUIRED报告。全部线保留，透明度和同流场四档相机固定。新增2项时间规则测试通过；源代码静态审计无FAIL。图片ZIP为outputs/Other_FMTObjectivityTranslation_1.3/Cylinder_late_time_observer_figures.zip，21文件哈希清单image_manifest.json；仅下载图片，预测与checkpoint均未下载。

## 2026-09-07 Other_FMTObjectivityTranslation_1.4 — 双倍显示时长和数量预登记

用户要求所有当前cylinder四联图轨线加长2倍、数量增至2倍。显示积分48->96步，约1.2->2.4物理时间，数量Re160/Re640/Re6400为7220/7154/7242；原初始t10.5/9.5/10.5保留。冻结分类配方只看每条轨线前48步的原定32点，各observer独立重新编码，标签着色到整条96步延长轨线上，不表示尾段逐位置都重新判为同类。旧样本能完整积分则优先保留；越域样本明确排除后，由固定Sobol空间种子按顺序补足准确数量，不据标签/图像筛选。新显示队列不套用旧缓存IVD标签、不报告旧队列F1。目标observer仍为完整空间均值，但读取26帧覆盖更长显示窗口；Re6400本地延长原始源窗口105..130帧。配置与使用说明docs/Other_FMTObjectivityTranslation_1.4.md。Array51408807_[0-1]；结果待运行。

### 1.4 数值完成与显示裁剪修复记录

计算证据：51408807_0/_1、51408817_2，运行代码SHAfec7b07553c074d9f03cfd08f0e4d36edd7bc5d470208f32be752fdb281282fe；配置执行参数见1.4预登记；每流场JSON保留完整来源。所有显示时长均2.4000091552734375，分类时长1.2000045776367188，比例恰为2；显示97点，分类取原48步32点。

| 数据 | 旧显示数→新显示数 | 保留原种子 / 新增 | 已尝试候选 / 不完整 / 多余完整未采用 | 0/25/50/100%预测涡数 | 0/25/50/100%标签变化数 | 最大轨线变换误差 |
|---|---|---|---|---|---|---|
| Re160 t10.5 | 3610→7220 | 2986 / 4234 | 9216 / 1788 / 208 | 1925/1947/1979/2012 | 0/22/54/87 | 1.52027e-7 |
| Re640 t9.5 | 3577→7154 | 2883 / 4271 | 9216 / 1634 / 428 | 1215/1256/1318/1408 | 0/41/103/193 | 9.03370e-7 |
| Re6400 t10.5 | 3621→7242 | 2903 / 4339 | 9216 / 1944 / 30 | 368/378/390/394 | 0/10/22/26 | 1.51750e-6 |

三组四档共同有效集合均恰达目标数量，无再次丢失。旧种子分别有624/694/718个不能完成更长积分，按纯几何完整性规则排除并补足。整个补点及选择过程不读取类别，未选最有利曲线。当前变化数属于新的显示队列，不能拿旧1.3的变化数直接作性能优劣比较；不报告基于旧IVD缓存标签的F1。

显示检查修订：此前1.2/1.3记录“面板尺寸、字体检查通过并已目视”的结论只覆盖布局和文字，未发现放大三维视图中Line3DCollection仍被默认二维axes边界裁剪。1.4密集图预览暴露该问题，现对物理域内的已验证轨线关闭artist clipping，重新导出到final/。之前错在把文字/面板检查和整图目视当成轨线完整显示的充分检查；它们不能证明图元没有被裁。修复不改变积分、特征或分类数值，旧图片保留但不建议作为完整轨线展示直接使用。最终渲染SHAa38bcb1e4a44c352b99092fac7fd5f9cd4738bf67e7f00a4a4d3cfffab567d44，render-only51408915读取同一保存数组；详见1.4使用说明。
1.4最终完成：render51408915于2026-09-07T00:14:00+03:00结束。所有三组最终图均在各dataset/final/，图片包Cylinder_long_dense_observer_figures.zip含21个PNG/PDF/SVG/TIFF文件。最终六份面板尺寸检查PASS、PDF最小字号论文7pt/汇报13pt、碰撞0FAIL；每份4项统计文字跨白色axes背景边缘的WARN已逐栏接受。24个面板及六张整图目视核对，物理域内轨线完整投影显示、不再按二维方形截断。下载仅图片，zip与21图哈希已核验。2项新增密集种子选择测试通过，静态审计19PASS/2WARN/0FAIL；动态字号及单次确定性队列无需误差条分别解释两个WARN。未新增训练或基线修改。

## 2026-09-07 Other_FMTObjectivityTranslation_1.5 — seven velocity fractions

用户将1.4四联图扩展为一列七图。固定alpha=0,1/6,2/6,3/6,4/6,5/6,1；同1.4的物质种子、源场、全网格均值曲线、后半程初始时间、96步展示及48步分类窗口。每档独立积分observed field并用原冻结Task1-4.1分类器重新计算标签，速度积分为位移后在逆变换查询位置读取原场。无新模型选择，无旧队列IVD F1复用。代码experiments/Visualize_FMT_Objectivity_Translation_1_5.py，配置config/Other_FMTObjectivityTranslation_1.5.json，source manifest tmp/objectivity_1p5_sources.json；Ibex51411521_[0-2]全部完成。

| 流场 | N / t0 | 七档涡轨线数（alpha升序） | 七档相对原始标签变化数 | 最大轨线对应误差 |
|---|---|---|---|---|
| Re160 | 7220 / 10.5 | 1925,1942,1953,1979,1998,2012,2012 | 0,17,28,54,73,87,87 | 1.520271892e-7 |
| Re640 | 7154 / 9.5 | 1215,1245,1276,1318,1362,1405,1408 | 0,30,61,103,147,190,193 | 9.033696338e-7 |
| Re6400 | 7242 / 10.5 | 368,376,382,390,393,395,394 | 0,8,14,22,25,27,26 | 1.517499120e-6 |

所有七档均完整保留同一队列。三流场0/50/100%相对1.4逐条标签完全相同、坐标最大绝对差0。新中间档位显示Re6400在83.3%至100%间变化数27降至26，因此不能将分类变化描述为普遍单调；三个流场均有非零标签变化，不能声称此冻结分类器严格客观。图中显示平移剥离后的几何与这些轨线的实际分类；不从画面推断新准确率结论。

最终三流场共21份PNG/PDF/SVG/TIFF图片，论文2880x6834 PNG，PPT2376x5638 PNG；最小字体7/13pt。七栏尺寸检查六份全部PASS；本地最终PDF碰撞0FAIL、各7项统计文字跨白色axes边缘WARN，六张整图全部七栏已目视检查并接受，不遮挡文字/轨线。自动REVIEW REQUIRED原报告保留并附visual-review解释。沿用1.4禁用二维axes裁剪修复。ZIP内容、CRC及21文件SHA核对通过，科学数组只保留在Ibex；1.4结果未改。


## 2026-09-07 Verify_FMTObservedPathline_2.1 — 三步observer独立实现及特征诊断

用户质疑1.5缺少pushforward和整体平移，并要求从第一性原理复核。读完当前ReferenceFrame3d.cpp的worldline、两方向变换及场构造后，新建数值相机worldline(DOP853)、ReferenceFrameFromWorldline、ObservedVectorField三部分，然后RK4积分observed场；不依赖旧TranslationObserver的场查询/位移实现。固定参考时刻为播种时刻，变换初始恒等，不额外添加模拟0时刻累积位移。原1.4同物质种子、96步显示/48步分类、冻结Task1-4.1配方和主实验完全保留；七档系数i/6。删除全部bounding box。Ibex51447581_[0-2]全部完成；运行源manifest tmp/observed_2p1_sources.json，配置config/Verify_FMTObservedPathline_2.1.json。

| 流场 | 七档标签变化数 | 仅替换中心块变化数 | 仅替换其他块变化数 | 正确轨线最大对应误差 | 错误v(y)-u轨线最大偏差 / 完整对照数 | 双精度邻居FMT最大变化 |
|---|---|---|---|---:|---|---:|
| Re160 | 0,17,28,54,73,87,87 | 0,17,28,54,73,87,87 | 0,0,1,0,1,1,0 | 1.51778993e-7 | 1.184129208 / 128 | 4.30543379e-11 |
| Re640 | 0,30,61,103,147,190,193 | 0,29,61,103,147,190,193 | 0,0,0,0,0,0,0 | 9.05568258e-7 | 1.062239021 / 128 | 4.84513853e-11 |
| Re6400 | 0,8,14,22,25,27,26 | 0,8,14,22,25,27,26 | 0,0,0,0,0,0,0 | 1.51645437e-6 | 1.173574161 / 126 | 3.74953402e-11 |

全部21档的独立observed积分与原轨线精确坐标变换的分类逐条一致（差异数0）；新实现与1.5原七档分类逐条一致（每流场七档差异全0）。所有主图物质primitive完整，N仍7220/7154/7242。错误relative-field对照只用事先按索引选定的前128条中心线，Re6400其中2条不完整仅排除于负对照；不影响主图或分类队列。

结论修订并列：之前只报告“颜色变化所以不严格客观”，没有定位具体原因 → 现在将场变换正确性和编码配方分别检验。源码显示旧1.5本已在y+d(t)查询原场，新独立实现与精确变换控制支持原有积分语义；因此不能把错误原因写成旧版仅积分v-u。旧图线框实际是共同取景盒，容易被误解成固定lab边界，现去掉。FMT的fmt_all含前23维跨时间中心位移特征，随相机速度改变；双精度邻居相对特征仅1e-11级变化，中心块变化最高约2。中心块干预在大多数档位复现标签变化数量，但“数量相同”不自动等于变化集合相同；Re640在1/6处中心块29而完整配方30，不能将每一条变化都唯一归因于中心块。

浮点局限另记：float32输入下邻居描述符最大差约0.03–0.07，kin4某些极端样本的绝对差很大（例如Re640在5/6档3238344）；该块涉及局部微分矩阵的伪逆和高幅值量，不能仅以绝对差推断准确率。本次未全面追踪每个病态样本；不修改冻结配方或以复制标签消除差异。参考系刚体性保持同一时刻粒子间几何，不保持跨时刻中心曲线形状；匀速流在随流相机里成为静止点即反例。

理论推导和知识缺口已写入optimal-connection/docs/referenceframe/referenceFrame_overview_zh.md§2.3–2.6，用户原指定缺失路径创建入口。只改文档，无C++源码修改；当前C++ SHAe786d3bb34ae8cf7bfcd421e392cf0d264051487b848808d5392fb9811649f93。纯平移验证不推广至任意时变旋转或非Killing相机。



最终质量检查：三个流场共21份图片，全部六份面板尺寸PASS，最终PDF最小字号7/13pt、0FAIL，每份7个统计文字跨不可见白色axes背景边缘WARN已逐栏查看并接受。42栏与六张整图已查看，无bounding box、无可见轨线截断；ZIP及21个文件SHA核对通过。原始科学数组和JSON只留Ibex。整理审计时移除了从1.4继承的旧四档feature_max_absolute_change、旧显示字段，将changed_label_fraction按七档实际计数重算；旧JSON留备份，图片/轨线/标签完全不改。这个metadata修订不改变上述科学结论。

## 2026-09-07 — Verify_FMTAllV2_1.1 preregistration

User requested an objective neighbour-only `fmt_all_v2` and paired Task1/2/5 reruns. Fixed protocol `docs/Verify_FMTAllV2_1.1.md` and config `config/Verify_FMTAllV2_1.1.json` before performance evaluation. Encoder `FMT_Utils/FMTAllV2_3D.py` uses same-time six-neighbour offsets, 21 scalar Gram sequences, initial-scale normalization and initial-Gram subtraction, then 6 real-input Fourier bins (231 dimensions). Same-time vectors are rotation-covariant; their scalar Gram entries are invariant under arbitrary time-dependent rigid transformations before Fourier analysis. Five local synthetic tests pass, including a negative control showing that vector-component Fourier coefficients change under time-dependent rotation. This establishes implementation behavior on these tests, not performance on flow data. Old recipes and frozen results are unchanged. New performance remains pending; no claim of improved classification. Full Task5 retains a Raw coordinate branch and is not claimed objective.

## 2026-09-07 — Task1 portion of Verify_FMTAllV2_1.1

All50 Task1 shards completed. Independent prediction-based integer confusion matrices reproduce all100 F1 values. Paired dataset-macro old/new F1 =0.6015089286796476 /0.1429654043818444, difference -0.458543524297803; all10 entries decrease. This is a result of the frozen8-component PCA/KMeans setting with the specified231-dimensional objective Gram-Fourier input, not a claim that every possible objective encoder fails. Re160 seed7080 predicts100% positive under v2 (precision0.0832235,recall1), versus15.01% under the old recipe. DeltaWing_LBM v2 predicts99.90% positive. Labels and cluster identities were calibrated on the held-out development split. Prior task results remain unchanged. Full cross-task report pending Task2/Task5 and all-shard audit.

## 2026-09-07 — distinction between standalone v2 and core replacement

Earlier1.1 comparison: the complete prior recipe was replaced by v2 alone. Revised interpretation: this measures a standalone objective encoder, **not** the isolated effect of replacing the old fmt_all core, because it also removes kin4 (Tasks1/2) and gram2+kin6 (Task5). This is a comparison-scope limitation, not an error in computed1.1 metrics. Preserve all1.1 results and add `Verify_FMTAllV2_1.2` to answer the pipeline replacement question: retain old neighbour add-ons, replace only fmt_all by fmt_all_v2. Task1/2 new width259; Task5 new width338, so both old and new residual inputs are padded to338 for equal capacity and both retrained. Hyperparameters unchanged; old/Raw Tasks1/2 predictions reused only after exact data-identity checks. This control is registered after1.1 Task1 and partial learning-task results were seen and is explicitly not fresh confirmation or tuning. The core's exact objectivity certificate does not cover retained kinematic add-ons or the Raw branch.

Task1 control1.2 also completed50 new-arm runs: retained-kin4 v2 macro F1=0.14277179008726074, versus standalone1.1 v2=0.1429654043818444 and paired old=0.6015089286796476. Restoring kin4 therefore does not recover Task1 performance under the fixed PCA8/KMeans protocol. Independent integer-count F1 audit of the50 native control predictions passes. This narrows the interpretation of Task1's failure; it does not identify a unique causal feature or rule out other objective representations. Full Task2/Task5 controls remain pending.

## 2026-09-07 — Verify_FMTAllV2_1.1 complete, standalone objective encoder

Independent auditPASS:150shards,400 aggregate rows,1350 per-scale rows,5seeds x10datasets per task. Data-source hashes, disjoint split paths, same labels, confusion matrices, Average Precision, adjusted Rand index and normalized mutual information were checked. All Task2 runs completed7000updates. Both Task5 arms have132354parameters and share each seed's frozen Raw model. Source/config/seed evidence is in `outputs/Verify_FMTAllV2_1.1` and ibex registry.

| Task | Old recipe macro F1 | Standalone v2 macro F1 | New minus old | Raw macro F1 | Entries improved |
|---|---:|---:|---:|---:|---:|
| Task1 |0.601508929|0.142965404|-0.458543524|not rerun here|0/10|
| Task2 |0.583613270|0.139851968|-0.443761302|0.488473648|0/10|
| Task5 |0.675298050|0.615071320|-0.060226731|0.576159839|1/10|

Task5 Average Precision old0.719478009→v2 0.640993768 (delta-0.078484242), Raw0.599648549. Paired macro-F1 delta sample standard deviations across seeds are0.001991067/0.020510307/0.002383823 for Tasks1/2/5. This specific objective-neighbour-only Fourier representation substantially underperforms the old complete feature recipes on the fixed benchmark, especially unsupervised Tasks1/2; Task5 remains above Raw on macro F1, with one entry improving versus old. This does not establish that objective encoders in general fail. The standalone comparison also removes existing auxiliary blocks; control1.2 separately retains them.

Core invariance check across synthetic tests and actual training fixtures passed: maximum absolute float32-output discrepancy1.862645149230957e-9, maximum relative L2 discrepancy1.747969685661488e-10 under time-dependent rotations/translations applied in float64 to the same material particles. This certificate applies to the encoder, not Task5's Raw branch. Frozen cylinder times were replayed, including earlier simulation times; no late-only benchmark was generated.

## 2026-09-07 — Verify_FMTAllV2_1.2 complete, core replacement with retained blocks

All150 control shards and250 native predictions/metric rows pass independent audit;150 Task1/Task2 old/Raw rows from1.1 are explicitly reused after source-file/hash/label identity verification. Two Task5 networks use338 inputs and136834parameters each; their matched-width old baseline is separately rerun rather than borrowing the132354-parameter old baseline from1.1. Task2 input/output layers follow259 features while the same hidden512/256,latent64,KL1e-6,learning-rate3e-4,7000updates are preserved. GPU models for reused Task2 pairs can differ; exact devices are recorded.

| Task | Matched old macro F1 | Core-replaced macro F1 | New minus old | Entries improved/decreased |
|---|---:|---:|---:|---:|
| Task1 |0.601508929|0.142771790|-0.458737139|0/10|
| Task2 |0.583613270|0.179515022|-0.404098248|1/9|
| Task5 |0.678020588|0.676800723|-0.001219866|7/3|

Task5 AP0.719512817→0.723950925 (delta+0.004438108). Its mean F1 change is-0.12percentage points, versus a paired macro-difference sample SD of0.99points over5seeds; no significance or equivalence test was conducted. Largest Task5 loss isChannel-0.0387; largest gain isF22+0.0216. Task2's sole small mean-F1 increase isChannel+0.0022. The results support this limited statement: replacing the core withv2 substantially reduces unsupervised Task1/2 performance under frozen settings; Task5 with retained neighbour blocks has near-unchanged meanF1 and slightly higher meanAP. They do not support declaring the complete Task5 pipeline objective, because Raw and retained kinematic blocks are outside the core's exact certificate. Old paper tables and frozen recipes remain unchanged.

Reconciliation with1.1: standalone v2 also removed the old auxiliary neighbour blocks, while1.2 retains them. Restoring those blocks recovers most Task5 performance but does not recover Task1/2 performance. Therefore1.1's losses cannot all be attributed to replacing only fmt_all. The two versions and their old baselines must remain separate; there is no silent revision of the original result.

An algebraic information-loss diagnostic explains a possible limitation, not a proven causal decomposition of the benchmark losses. Because v2 stores onlyper-primitive same-time Gram scalars, it is invariant even under independently chosen rotations of each primitive, stronger than the required single global rigid observer. Uniformly rotating a rigid neighbour cross and leaving that cross stationary both yield an unchanged Gram series. The existing negative-control test establishes this behavior; an additional NumPy check using seed7099,shape[4,7,32,3],independent SO(3) rotations perprimitive/persample gives maximum output difference0.0. Consequently relative rotation between different local bundles is unavailable to this feature alone. Quantifying how much this versus normalization,frequency truncation or other information changes causes each flow's performance loss requires a separate experiment; none was selected using these test results.

Final deliverables: Chinese comparison report `outputs/Verify_FMTAllV2_report/report_zh.md`, all10-flow tables and per-run/per-scaleCSV forboth versions, and PDF/SVG/600dpiPNG figures. Both final PDFs have6.5pt minimum glyphs,0collisionFAIL/WARN; equal-panel geometry and allthree panels/fullfigures inspected. Figure source `experiments/Plot_FMTAllV2_Comparison.py`; plotting catalog entryP20 updated. Figure captions explicitly retain allfive runs of each plotted FMT arm and direct Raw controls to companion tables. Code,configs,seeds,input hashes and devices are preserved; no model was downloaded.


## Verify_AIVDTransfer_1.1 — registered 2026-09-07, results pending

User requests unchanged Task3 `aivd1w3_dft` transfer toTask1/Task2 and explicitly chooses late-cylinder t>=7.5 with old methods retrained. Freeze threeTask1 arms (`fmt_all+kin4`189, standalone`aivd1w3_dft`1, `aivd1w3_dft+kin4`29) and fourTask2 arms (same plusRaw672), all10datasets×5seeds. Same original split roles with metadata-only time filtering; Re160/Re6400 Task2 retains two eligible training slices(t7.7,8.8), same7000updates. Task1 scalar skips impossible PCA8; other arms retainPCA8. Task2 retains frozen[512,256]/latent64/beta1e-6/lr3e-4 VAE; input/output width changes, parametercounts reported. No new tuning; all outcomes reported on an alreadyused benchmark. No comparison of late new metrics against old fulltime metrics as the primary contrast.

Core code remains `FMT_Utils/DFT_FMT_3D.py` and original dispatcher; new runner `experiments/Verify_AIVDTransfer_3D.py`, config/protocol share experimentID. Five synthetic checks passed locally: independentNumPy and Task3dispatch equality, centre independence, translation/constantrotation invariance, finite-step dynamicrotation failure with refinement improvement, meancontext and fourthframe dependence. These implementation checks are not performance evidence or a strict general dynamicrotation objectivity certificate. Method explanation: `docs/aivd1w3_dft_explained_zh.md`. All comparisons and resulting conclusions remain pending. Ibex jobs/source hashes tracked in runregistry.

### Verify_AIVDTransfer_1.1 — Task1 complete, Task2 still pending

All50Task1 shards/150rows were downloaded as prediction/metric-only results and locally checked against SHA256 and independently recomputed confusion-matrixF1. Archive SHA256 `2bc6daab743124feee0c2d4e233d7698660187f69ec7a6548bd8a4534d7121a5`; local certificate `outputs/Verify_AIVDTransfer_1.1/local_Task1_prediction_audit.json`. Late-time paired Task1 macroF1: retrained old`fmt_all+kin4` **0.59585617**, standalone`aivd1w3_dft` **0.63801972** (delta **+0.04216355**,6/10entries improve), `aivd1w3_dft+kin4` with frozenPCA8 **0.24621408** (delta **−0.34964209**,2/10improve). Standalone Re160/Re640/Re6400: old/new **0.54820/0.91444**, **0.56867/0.68003**, **0.52542/0.72770**; both deltaWingentries decrease, resampled **0.76866→0.40734**, LBM **0.74676→0.37924**. This supports mean improvement in this fixedTask1 protocol, not universal improvement or a claim that retainedkin4 is generally useless. The control jointly includes its fixed originalPCA8 treatment; no further tuning follows these results. Full100-shard/350-row audit andTask2 outcomes remain pending.

### Verify_AIVDTransfer_1.1 — complete, independent audit PASS

All100 scientific shards completed without failure (50Task1,50Task2), all350 per-run metric rows independently recomputed by audit job51456207, completed2026-09-07T15:15:39–15:15:45+03:00 on cn604-18,exit0. All10 datasets×5seeds, same original roles; every cylinder role t>=7.5; all baselines retrained. No hyperparameter or candidate adjustment after observing results.

| Task | Retrained old fmt_all+kin4 | Standalone aivd1w3_dft | Paired delta F1 | Improved entries | aivd1w3_dft+kin4 | Control delta |
|---|---:|---:|---:|---:|---:|---:|
| Task1 | 0.59585617 | 0.63801972 | +0.04216355 | 6/10 | 0.24621408 | −0.34964209 |
| Task2 | 0.57717493 | 0.63344946 | +0.05627453 | 7/10 | 0.30452918 | −0.27264575 |

Task2 Raw+VAE meanF1=0.49059801; standalone is +0.14285145 aboveRaw on the equal-entry mean. Paired seed-macro delta standard deviations: Task1 standalone0.00015009/control0.00012159; Task2 standalone0.00690716/control0.01062210. These describe seed variation, not independent physical-family uncertainty or statistical significance. Both controls improve only2/10entries (Channel,Boeing); this does not establish that kin4 is universally harmful, because the finding is conditional on the frozen normalization/downstream pipeline. No follow-up tuning is performed.

Standalone improves allthreeCylinder cases in bothtasks. Task2 old → standalone: Re160 `0.2985 → 0.7691`, Re640 `0.4261 → 0.5099`, Re6400 `0.5120 → 0.6185`. BothdeltaWingentries decline in bothtasks. Task1 resampled recall0.6242→0.2558 andLBM0.5998→0.2340, with standalone precision1.0 in both; the observed F1 loss corresponds to missed positives. The numerical feature is one scalar: sum of firstthree estimated sampled-mean vorticity-deviation magnitudes, no nonzero frequency bin. Thus improved averages support transfer of this geometry-derived scalar in the fixedprotocol, not a claim of nonzero Fourier-frequency benefit, universal flow improvement, or strict finite-step objectivity.

Frozen-baseline stability check on40Task1 runs whose populations did not change: all source identities and raw/time/file hashes match1.1FMTAllV2 replay.39/40 oldF1 values match exactly; Channel seed7083 differs by+0.0000789313. Both values are preserved; no scientific recipe or data revision is inferred from this small replay difference, whose numerical cause was not isolated. Re160/Re6400 old fulltime metrics are excluded from this replication check because their populations intentionally changed.

Code/config/protocol: `experiments/Verify_AIVDTransfer_3D.py`, `config/Verify_AIVDTransfer_1.1.json`, `docs/Verify_AIVDTransfer_1.1.md`. Full per-run predictions, metrics, audits and execution records remain at Ibex `/home/zhanx0o/FMT_Uniform_3D_20260901/outputs/Verify_AIVDTransfer_1.1`. Approved local aggregates: `outputs/Verify_AIVDTransfer_1.1/{dataset_metrics,paired_comparisons,task_macro}.csv` and `report_zh.md`; local aggregate consistency audit PASS70dataset+40paired+7macro rows. The full100-shard prediction audit ran on Ibex; do not describe the local aggregate check as a local350-row prediction audit. Full report source hashes in aggregate_download_manifest.json.


## Verify_AIVDTranslationObservers_1.1 — 2026-09-07

User-requested Re160 3D seven-observer figure applies unchanged aivd1w3_dft to the existing correctly integrated observed-field paths. Evidence: config `config/Verify_AIVDTranslationObservers_1.1.json`, code `experiments/Verify_AIVDTranslationObservers_3D.py`, job51461382, output summary and observer_summary.csv. Same7,220 material primitives, t0=10.5; fractions0,1/6,2/6,1/2,2/3,5/6,1 of full-source mean translation velocity. All seven groups have697 vortex labels; changed labels=[0, 0, 0, 0, 0, 0, 0]. Each feature/label array was independently recomputed; one frozen late-time Task1 scalar StandardScaler+KMeans,seed7080,noPCA, calibration exactly reproduces Verify_AIVDTransfer_1.1. No fitting, threshold selection or label copying between observers.

Maximum across observers: float32 feature absolute difference=2.50700395554e-05; relative L2 difference=1.16275759865e-05; float64 integrated-path feature difference=4.0785358979e-09; exact-transform float64 feature difference=6.63358257214e-15. Exact-transform float32 labels also unchanged; integrated-vs-exact labels all identical. Original float32-vs-float64 label disagreement=0. Path correspondence maximum=1.5177900013e-07; same-time neighbour separation maximum difference=7.84921538877e-09. Full observer final displacement=[2.4001311224970583,0.0030356506447779836,-0.0003413597505859615]. Original feature range=[0.000882230466232,1.88573026657].

Conclusion boundary: this verifies translation invariance to numerical accuracy for this cohort and fixed classifier; it does not prove objectivity under arbitrary time-dependent rotations. The previous statement “finite-time derivative breaks strict general objectivity” remains, but must not be misread as “all temporal differences break translation invariance”: this code differentiates same-time neighbour pair separations, whose common translations cancel. Exact-transform float64 control nearly reaches machine precision; float32 coordinate rounding dominates this translation test's small feature residual. No claim of improved classification accuracy is inferred from the zero changed labels. Existing five mechanistic tests pass, including common translation/constant rotation invariance and a time-dependent-rotation finite-difference counterexample.

Figure exports: seven equal vertical panels, original3D camera and shared union bounds, all centre paths,96 display steps/48 classification steps/32 samples, no bounding box. PDF font minima7pt/13pt; both alignment gatesPASS; collision0FAIL/7WARN each, all warnings inspected as intentional overlap of count text with invisible white axes background. Both whole figures and all14 panels visually reviewed. Registered plotting method P22.

### Verify_JHTDB_DualFormatDownload_1.2 — expanded-domain fresh download in progress

2026-09-07 user requested replacement downloads for channel and isotropic: double each physical-axis span, 128^3 grid, retain 32 times, save VTK plus one complete NetCDF per flow. Config: `config/Verify_JHTDB_DualFormatDownload_1.2.json`; downloader: `experiments/Download_JHTDB_DualFormat.py`; export/audit: `experiments/JHTDB_DualFormat.py`, `experiments/Audit_JHTDB_DualFormat.py`. Channel bounds [3,3.6] x [-0.9,-0.3] x [0.2,0.8], times 1+i*0.0065; isotropic1024coarse bounds [0,0.8]^3, times 1+i*0.002; i=0..31. Fixed physical coordinates, lag8 spatial interpolation; channel PCHIP, isotropic stored times. Each span and total volume increase by factors 2 and 8 respectively; old data are not upsampled. 22 focused local loader/format tests pass, including a noncubic multi-time VTK/NetCDF exact-value test. Full-download outcome remains pending in this entry; final manifests and audit provide completion evidence. This data-acquisition version changes no frozen research benchmark or scientific conclusion. Old download files will be removed only after complete replacement validation; historical records remain.

## 2026-09-07 — aivd1w3_dft 客观性术语与计算解释纠正

这是解释修订，没有启动新实验，也没有修改冻结 encoder、分类器、标签或既有结果。

| 原表述 | 修正表述 | 原问题与依据 |
|---|---|---|
| 同时间点差“只是旋转协变”，容易被理解为相对位置不客观 | 同一时刻同一对物质点的相对位置是客观几何向量；d*=Qd 是客观向量的分量变换规律 | 此前把客观向量与分量不变的客观标量混用；基底变化不等于物理相对几何变化。不能用D*=QD否定相对位置的客观性。 |
| “时间差分不客观”的笼统表述 | 要明确差分对象、运动基底项、后续抵消和离散误差 | 连续满秩D下，A*=QAQᵀ+Q̇Qᵀ，涡量加同时间统一旋转项；扣同一物质样本均值后取模可抵消。当前有限差分的具体算法边界不由输入相对向量的客观性单独决定。 |
| “前三个标量，傅里叶零频”说明过短 | 三个是k=0,1,2的涡量偏差模长，需D0..D3；默认无时间间隔除数；零频未归一化，最终是三者的和[N,1] | 逐行依据DFT_FMT_3D.py:111–201、276–338与Task12Data_3D.py:26–54；不是涡量三个分量，也不输出振荡频率。 |

展开后的说明在docs/aivd1w3_dft_explained_zh.md；在AGENTS.md持久记录“客观向量不等于坐标分量不变”。保持Verify_AIVDTranslationObservers_1.1的七档零changed labels及有限时间差分分析原证据边界，不把术语纠正伪装成一次新实验或推翻已记录的数值事实。

## Verify_AIVDLongtime_1.1 — 2026-09-07 已冻结，尚无新性能结论

用户授权Task1/2/3/5测试全轨线估计IVD零频汇总。新名称aivd1w3_dft_longtime，仅window=None，其余沿冻结短窗口；代码/config/比较臂/晚期Cylinder重建与Task5同步均值分组已登记于Verify_experiments.md。原方法与历史性能不改写。所有200分片在同一新版本下运行；结果待独立预测审计，不从未完成任务或原全时段结果作性能判断。

### Verify_AIVDLongtime_1.1 — Task1 已完成，完整审计待运行

Array51462940的50个Task1分片全部COMPLETED/exit0，200行per_run.csv已落盘。直接汇总CSV得到10条目等权F1：Raw0.42339042244、旧fmt_all+kin4 0.59585459396、短窗口0.63801972160、全32点窗口0.33864182878，长减短为−0.29937789282。Boeing及两个DeltaWing条目提高，其余7项下降。这里仅报告已完成Task1的CSV结果；200分片最终预测独立复算尚待Task2/3/5完成后由51463287执行，不把此阶段记录描述为完整审计PASS，也不据此调整特征或超参数。

Task2 array51462941亦50/50 COMPLETED，200行CSV初步汇总：Raw0.48843485237、旧fmt_all+kin4 0.58257598489、短窗口0.63678140719、长窗口0.33081165302，差值−0.30596975416。Re160及两个DeltaWing条目提高，其余7项下降。与上轮短窗口Task2平均0.63344946089相比，本轮重训为0.63678140719；两个数值分别保留，不能把重训差异误称为原算法发生修改。当前所有短/长比较均在本轮同分片、同GPU、同种子下配对；完整预测审计仍待运行。

Task3 array51462942随后50/50 COMPLETED，250行CSV汇总：Raw/Raw-PCA/Raw+短/Raw+长的F1分别0.63268687089/0.63749876269/0.86276280632/0.76464639993；短/长Average Precision为0.93512138455/0.81662501774。长减短F1−0.09811640639、Average Precision−0.11849636682；Boeing及两个DeltaWing条目F1提高，其余7项下降。两种特征分支均115266总参数、25345可训练残差参数，共享每分片同一个新训练Raw主干。长版本仍超过Raw和同结构Raw-PCA对照，但平均低于短版本。此为完整Task3的CSV结果，最终200分片独立预测审计仍等待Task5。

### Verify_AIVDLongtime_1.1 — 全部完成，独立审计 PASS

2026-09-07，四个任务各10个3D数据条目×5种子，共200分片均成功结束，无训练失败、重试、取消或预算调整。独立审计51463287于17:05:11–17:05:38+03:00在cn604-10完成，逐预测复算1000行总体指标和3150行Task5尺度指标全部通过。核对固定划分、所有Cylinder初始t>=7.5、相同输入、Task2每臂恰7000次更新、监督短/长分支容量一致及同一Raw主干。源清单SHA256 e798e3c3c2d3e81f88fa017eca2787f12d29fbe9d8b32f09efaf9c3bc9bc7fa3；config SHA256 fce12000c39c007a941f17093cd12b9f1dcd9072145ece15ecc13c9bce628eac。

下表为各条目、各种子等权平均F1；Task3/5比较Raw加相应特征，Task1/2比较相应特征进入原聚类/VAE流程。变化为长减短。

| Task | Raw | 原aivd1w3_dft | aivd1w3_dft_longtime | ΔF1 | 配对种子均值差的标准差 | 提高条目 |
|---|---:|---:|---:|---:|---:|---:|
| Task1 | 0.42339042 | 0.63801972 | 0.33864183 | −0.29937789 | 0.00014982 | 3/10 |
| Task2 | 0.48843485 | 0.63678141 | 0.33081165 | −0.30596975 | 0.00935502 | 3/10 |
| Task3 | 0.63268687 | 0.86276281 | 0.76464640 | −0.09811641 | 0.00630296 | 3/10 |
| Task5 | 0.61067165 | 0.79359782 | 0.73282486 | −0.06077297 | 0.00580160 | 2/10 |

Task3平均精确率（Average Precision）短/长0.93512138/0.81662502，差−0.11849637；Task5短/长0.87088824/0.78347186，差−0.08741638。原版保持独立名称、实现和历史记录；长版本的本轮四任务平均均下降，不支持用本版全轨线汇总替换原版。此结论限于当前固定32点、单零频、初始时刻IVD p95标签及冻结下游配置；没有搜索中间窗口，也没有从测试结果选择下一版参数。

所有条目均报告：Task1/3提高的是Boeing及两个DeltaWing；Task2提高的是Re160及两个DeltaWing；Task5提高的是Boeing与DeltaWing resampled。Task5 DeltaWing LBM差约−0.0001，不能把这种小差异解释为可靠的物理差异；种子标准差不等于跨物理流场的不确定性或显著性检验。

Task1/2平均精确率分别从0.8001/0.7469降到0.4291/0.4233，召回率从0.6972/0.7557升到0.8102/0.8050：下降伴随误报增多，而非仅漏报。Task3/5的长版本仍高于Raw与同结构Raw-PCA；Task5 Raw-PCA/旧fmt_all+gram2+kin6/短/长F1分别0.63194187/0.72773459/0.79359782/0.73282486，不能将“低于短版”改写成“低于所有FMT或Raw对照”。

源码只新增窗口别名window=None，仍用全部32个估计IVD标量的未归一化零频和、输出1维；原DFT_FMT_3D.py未修改。Task5短/长分支使用同一物理时间分组，三种Cylinder晚期缓存是本版新增群体，因此主比较是本轮配对结果，不能将其与旧全时段Task5指标直接归因比较。模型选择只用训练/验证；没有为任何流场单独修改窗口、学习率、阈值或epoch预算。

结果根目录outputs/Verify_AIVDLongtime_1.1：report_zh.md含全部40个任务/条目对比；dataset_metrics.csv含全部200个任务/条目/方法汇总，per_run.csv有1000行，per_scale.csv有3150行，task_macro.csv/paired_comparisons.csv记录均值、种子标准差及配对差。远端完整预测审计通过；本地另对11个交付文件逐一核对远端SHA256，并复算表间均值与标准差（local_delivery_audit.json PASS）。本地核验不是重新读取预测数组；预测仍保留在Ibex。per_run.csv SHA256 2c21a012899efcf8c01ce00379e5fe82351982dba4efe2cc24c9793335c45e8c。

最终指标、报告写入后清理本实验550个临时checkpoint，共204843922bytes，远端剩余0，未下载模型。精确作业/节点/GPU/开始结束时间见execution_records.csv与scheduler_records.psv；源快照和完整配置保留在reproducibility。所有本节结论均来自本轮完整审计后的结果，取代上方阶段记录的“完整审计待运行”状态，不改写阶段数值。

### 2026-09-07 — 后续采用短窗口；七联图仅添加相机标注

用户明确要求忽略longtime，后续继续使用原aivd1w3_dft；停止继续研究本轮长窗口方案，保留已有代码和结果作为历史记录。新的Verify_AIVDTranslationObservers_1.2只修订Re160七联图的相机图标与速度标注，不产生新性能或客观性实验结论。1.1七档平移分类不变的证据仍有效，证据范围仍为该平移观察者集合。

Verify_AIVDTranslationObservers_1.2标注版完成，job51466161，2026-09-07T17:43:05–17:43:55+03:00，cn604-03，exit0。两版每格增加相机和u_cam(t)=alpha*u_mean(t)，保持原7220条路径线、697个vortex标签、七档changedlabels全0；没有重新拟合或分类。标签数组及输入文件哈希核对通过，PNG在新增标注区域之外逐像素等于原1.1图。字体、面板几何及14格视觉审查通过。此次仅提高运动参考系的图示清晰度，未添加客观性验证范围或性能证据。绘图方法P23已登记。

### Verify_JHTDB_DualFormatDownload_1.2 — complete; both formats and readers PASS

2026-09-07: both expanded domains completed at 128^3 x 32 times. All 64 newly queried velocity frames passed binary VTK coordinate/value roundtrips. Each flow now has 32 per-frame VTK files, a `.vtk.series` index, one combined VTK with 32 named time arrays, and one complete NetCDF with float32 u/v/w(time,z,y,x) and float64 physical coordinates/times. NetCDF lossless sizes: channel 586337752 bytes; isotropic 611557167 bytes. Each combined VTK is 855639617 bytes. Config, sampling, interpolation and time intervals remain as specified in the preceding entry; concurrency was changed from 4 to 8 after 16 completed frames, recorded in `resume_history.json`, without changing numerical settings.

Independent `outputs/Verify_JHTDB_DualFormatDownload_1.2/audit.json` PASS: all saved VTK coordinates/times/values, all NetCDF components at all times, all 32 frames through the original PyFlowVis VTKLoader, and all 32 frames through FMT load_jhtdb_netcdf match exactly. Both flows have 32 distinct velocity arrays. Separate first/last xz-plane queries and 17 fixed-seed random points at the final time produce maximum absolute differences 0 for both datasets. 23 focused tests pass, including a physical-coordinate analytic rotation/strain check for the additional display-only Q preview. These are data/implementation checks, not evidence of research-method performance or vortex-type ground truth.

Both 32-step HTML velocity viewers were visually checked at the final time (channel 1.2015; isotropic 1.062). Additional Q isosurfaces use the first frame only and a fixed positive-Q p90 display rule; the rule changes no research label or benchmark. After the full dual-format audit passed, `experiments/Cleanup_JHTDB_64Grid.ps1` removed exactly 66 old 64^3 VTK files, 2 old series indices and 3 old previews; all 71 targets were resolved and hash-checked before deletion. `old_grid_cleanup.json` confirms completion. Old manifests, source snapshots, configs and audits remain; the previous successful 64^3 results are not reinterpreted as incorrect. Documentation: `docs/JHTDB_dual_format_download_1.2.md`.


### Verify_AIVDTranslationObservers_1.3 — 2026-09-07 注释修订

用户要求相机旁速度用矢量箭头描述。新增六个按速度缩放的箭头与零速圆点，箭头表示t0=10.5的实际observer速度在原取景中的投影，保留完整时变矢量公式。Job51467144完成，原输入哈希与标签数组不变；7220条路径线中每格697条涡轨线，changed labels仍全部0。两版PNG注释区外像素变化0，所有PDF与视觉检查通过。此版只改注释，没有新增分类实验或方法客观性结论；代码/配置见绘图方法表P24，输出outputs/Verify_AIVDTranslationObservers_1.3。
# Verify_LargeNeighbor_1.1 — preliminary implementation checks, 2026-09-07

User-requested scheme one is implemented as center plus24 material neighbors on radii r,1.5r,2r, eight cube-vertex directions per shell. Original FMT and aivd1w3_dft remain frozen comparison arms. New encoding extends the same-time Gram construction of fmt_all_v2 to300 scalar channels, complete32-point sequences and Fourier bins0..5 (3300 real-valued outputs); no center temporal differences, IVD/curl or old kin block enters the new encoding. Protocol/config and frozen source are registered in docs/Verify_experiments.md and docs/ibex_run_registry.md.

Evidence: preflight job51477160 (22:13:22–22:13:51+03:00,cn604-02) checks32 corresponding material primitives from the first eligible training slice of each of10 datasets. Independent proper rotations and translations at every sampled time give relative feature error1.69e-13–9.44e-11, maximum absolute error7.45e-9. This numerical check supports encoder invariance and does not certify the coordinate-based Raw branch in Task3. Nonzero Fourier bins contain14.96%–46.37% of retained coefficient squared energy in these training slices, so the implementation is not DC-only. These checks do not establish prediction quality.

Integrity job51477159 passes128 cached slices:486233 common valid samples,574 excluded because the new outer shell is incomplete. Original labels/seeds/seven-line coordinates match their retained-index subset exactly; the original p95 definition is not recomputed. All Cylinder initial times satisfyt>=7.5. Performance evaluation remains pending; do not infer gains from the implementation checks. Current tasks test initial-time IVD binary labels, not geometric tokenization quality.
# Verify_LargeNeighbor_1.1 — Task1 interim aggregate, 2026-09-07 22:19+03:00

All50 Task1 children of51477161 completed/exit0,300 native metric rows across10 datasets and5 seeds. Unchanged protocol/source/config as registered above. Direct scalar aggregation gives equal-dataset/equal-seed F1: Raw7=0.42383056; Raw25=0.42614239; oldFMT=0.59589572; shortIVD=0.63875495; six-neighbor Gram Fourier=0.14169753; largeNeighbor=0.12994169. This interim complete-Task1 aggregate indicates no gain from the added shells under the frozen PCA8/KMeans pipeline. It is not the final independently audited report: all-task audit still waits for Task2/3, and no experimental selection or configuration change is made from these test outcomes. Final audited figures must be compared explicitly against this interim record.
# Verify_LargeNeighbor_1.1 — Task2 interim aggregate, 2026-09-07 22:50+03:00

All50 Task2 children of51477162 completed/exit0;300 native metric rows. Complete-dataset/seed mean F1: Raw7=0.49449154; Raw25=0.42929984; oldFMT=0.58478122; shortIVD=0.62915818; six-neighbor Gram Fourier=0.14420214; largeNeighbor=0.13977928. Under the frozen shared VAE architecture and7000 updates, the new scheme remains substantially below oldFMT and shortIVD. This is a complete Task2 scalar aggregate pending the final independent all-task prediction audit. No method or hyperparameter selection is made from it; Task3 remains running. Final values must explicitly reconcile with this interim record.
# Verify_LargeNeighbor_1.1 — remote completion; file delivery pending approval

All Task1/2/3 scientific runs and the final independent audit completed successfully. The full report, per-run metrics, aggregate comparisons, input/source/device records and checkpoint cleanup record are retained in the registered Ibex output directory. Both attempted metric-file downloads were rejected by automatic approval; no rejected files were transferred through another route or reconstructed locally. The final requested conversational performance report uses the previously authorized remote report read. The earlier Task1/Task2 interim entries remain as historical records; no new local final metric table is created pending export approval. Detailed final evidence: remote `report_zh.md`, `per_run.csv`, `dataset_metrics.csv`, `task_macro.csv`, `paired_comparisons.csv`, `independent_audit.json`. Transfer status and audit job are recorded in docs/ibex_run_registry.md.

<a id="large-neighbor-1-1-final"></a>

## Verify_LargeNeighbor_1.1 — 最终实验结果（2026-09-08 记入）

2026-09-08 用户明确要求将实验结果记入 Markdown。本节根据上一轮已经读取的 Ibex 最终报告及审计记录整理；实验于2026-09-07完成，本次没有重跑、改变配置或重新选择模型。下列数值按最终报告保留四位小数，完整精度、种子标准差和逐次预测证据仍见远端原始结果。此前“仅口头汇报、最终数值尚未记入本地”的状态由本节更新；此前文件下载拒绝记录作为历史保留，本次只执行明确要求的 Markdown 记录。

### 结论与阶段记录核对

**当前 largeNeighbor 实现不支持替代旧 FMT 或原短窗口 `aivd1w3_dft`。** Task1、Task2 的平均 F1 明显下降；Task3 相对同一 Raw25 骨干有小幅增益，但低于旧 FMT、短窗口方案和同结构 Raw-PCA 对照。与六邻居客观 Fourier 方案相比，三个任务的平均 F1 也均未提高。这里评价的是当前冻结编码器及下游配置，不能外推到全部大邻域设计。

| 记录阶段 | 当时状态 | 最终状态与核对 |
|---|---|---|
| Task1阶段记录（2026-09-07 22:19） | 50次已完成，独立总审计待运行 | 总审计已通过；旧FMT/短窗口/largeNeighbor的0.59589572/0.63875495/0.12994169与最终报告四位小数一致 |
| Task2阶段记录（2026-09-07 22:50） | 50次已完成，独立总审计待运行 | 总审计已通过；旧FMT/短窗口/largeNeighbor的0.58478122/0.62915818/0.13977928与最终报告四位小数一致 |
| 上一轮最终回复 | Task1/2/3全部完成并给出平均F1 | 本节持久记录相同最终结果，补充全部流场和比较边界；没有修订前述数值 |

### 方法、数据与比较条件

`largeNeighbor` 使用中心加24条材料邻居路径线：半径r、1.5r、2r的三个球面各8点，方向为立方体顶点；r为原始最小网格间距的一半。邻居只在初始时刻播种，之后各自积分并保持材料点身份。编码器取每时刻24个中心—邻居相对向量的300个上三角内积，按初始平均平方邻距归一化并减去初始内积，再对完整32个时刻作离散傅里叶变换，保留零频和5个非零频，共3300维。没有中心点跨时刻坐标差分，也没有混入短窗口IVD或旧kin特征。频率按采样序号及轨线窗口解释。

全场瞬时涡量偏差（Instantaneous Vorticity Deviation，IVD）的p95标签沿原缓存保留，只按共同有效样本索引取子集。三个Cylinder数据初始物理时间均t≥7.5。10个3D数据条目，每任务5个种子：Task1为7080–7084，Task2为100–104，Task3为40–44，共150次运行、950行方法指标。所有比较臂使用原7线和新25线均完整有效的同一批样本；128个缓存片共486233个共同有效primitive，新增外层轨线不完整排除574个，未按标签或分数筛样本。

| 任务 | 固定比较条件 |
|---|---|
| Task1 | 表示进入标准化、主成分分析（Principal Component Analysis，PCA）和KMeans二类聚类；PCA固定8维，单标量不做PCA；验证集确定簇含义 |
| Task2 | 同一变分自编码器（Variational Autoencoder，VAE）架构：隐藏层512/256、潜变量64、KL权重1e-6、学习率3e-4，每臂7000次更新；输入/输出宽度随表示变化，因此不能声称总参数量完全相同 |
| Task3 | 各比较臂共用本轮重训的25线Raw骨干；五个残差分支均使用268维辅助输入、相同结构和参数量；largeNeighbor仅在训练集拟合标准化/PCA268，Raw25-PCA为同结构容量对照 |

旧FMT指本轮冻结的 `fmt_all+kin4`；短窗口指原 `aivd1w3_dft`；六邻居客观方案指 `fmt_all_v2` 的同时间内积序列加Fourier编码。Task1/2的Raw7和Raw25分别直接使用原7线、新25线表示。Task3表中的旧FMT、短窗口、六邻居和largeNeighbor均表示“Raw25骨干加对应特征残差”，不能解释成仅靠该特征的分类分数。模型、标准化/PCA、epoch、融合权重及阈值仅用训练/验证集确定；测试不用于调参。既有benchmark已被使用过，本轮是配对复跑，不是新的独立confirmation。

### 平均指标

F1为精确率与召回率的调和平均；下表对每个流场的5个种子先取平均，再对10个流场等权平均。

| 任务 | 旧FMT | 原aivd1w3_dft | 六邻居客观Fourier | largeNeighbor | largeNeighbor−旧FMT（约百分点） |
|---|---:|---:|---:|---:|---:|
| Task1 | 0.5959 | 0.6388 | 0.1417 | 0.1299 | −46.60 |
| Task2 | 0.5848 | 0.6292 | 0.1442 | 0.1398 | −44.50 |
| Task3 | 0.7623 | 0.8579 | 0.6802 | 0.6790 | −8.33 |

Task3还需与Raw容量对照比较。平均精确率（Average Precision）是另一项基于预测分数排序的指标，不等同于单阈值下的precision。

| Task3方法 | 平均F1 | Average Precision |
|---|---:|---:|
| Raw25 | 0.6548 | 0.7074 |
| Raw25-wide | 0.6559 | 0.7081 |
| Raw25-PCA残差 | 0.7170 | 0.7625 |
| Raw25+旧FMT | 0.7623 | 0.8160 |
| Raw25+原aivd1w3_dft | 0.8579 | 0.9260 |
| Raw25+六邻居客观Fourier | 0.6802 | 0.7250 |
| Raw25+largeNeighbor | 0.6790 | 0.7262 |

Task3中largeNeighbor相对Raw25的F1约提高2.42个百分点，但相对Raw25-PCA残差约降低3.80个百分点。其Average Precision比六邻居方案高约0.12个百分点，而F1低约0.12个百分点；不能将“平均F1未提高”写成“所有指标都下降”。这些小差值没有经过显著性检验，不能据此作可靠的物理差异判断。变化量由报告中四位小数求差，故标“约”。

### Task1逐流场F1

各格为5个种子的均值。数据条目保留配置中的原名；`cylinder3d` 为Re160。这里展示均值，种子标准差及其它指标见远端 `dataset_metrics.csv`。

| 数据条目 | Raw7 | Raw25 | 旧FMT | 短窗口IVD | 六邻居客观Fourier | largeNeighbor |
|---|---:|---:|---:|---:|---:|---:|
| channel | 0.0651 | 0.0651 | 0.1821 | 0.2067 | 0.0557 | 0.0653 |
| cylinder3d | 0.2745 | 0.2777 | 0.5481 | 0.9150 | 0.1546 | 0.1537 |
| halfcylinderRe640 | 0.3364 | 0.3400 | 0.5682 | 0.6847 | 0.1287 | 0.1290 |
| halfcylinderRe6400 | 0.3873 | 0.3886 | 0.5236 | 0.7306 | 0.1242 | 0.1242 |
| tangaroa | 0.5868 | 0.5846 | 0.7441 | 0.8499 | 0.1182 | 0.1157 |
| deltaWing_resampled | 0.5633 | 0.5633 | 0.7687 | 0.4073 | 0.1397 | 0.0923 |
| deltaWing_LBM | 0.5156 | 0.5168 | 0.7468 | 0.3792 | 0.1037 | 0.1037 |
| f22raptor | 0.5020 | 0.5050 | 0.3139 | 0.7463 | 0.1496 | 0.1440 |
| boeing747 | 0.4619 | 0.4724 | 0.8042 | 0.7778 | 0.3165 | 0.2559 |
| smokeBuoyancy | 0.5455 | 0.5480 | 0.7592 | 0.6899 | 0.1261 | 0.1155 |
| **10项等权平均** | 0.4238 | 0.4261 | 0.5959 | 0.6388 | 0.1417 | 0.1299 |

### Task2逐流场F1

| 数据条目 | Raw7 | Raw25 | 旧FMT | 短窗口IVD | 六邻居客观Fourier | largeNeighbor |
|---|---:|---:|---:|---:|---:|---:|
| channel | 0.1945 | 0.1947 | 0.2294 | 0.1747 | 0.0692 | 0.1642 |
| cylinder3d | 0.2263 | 0.2138 | 0.2955 | 0.7631 | 0.0702 | 0.0693 |
| halfcylinderRe640 | 0.3383 | 0.3119 | 0.4327 | 0.5202 | 0.1256 | 0.1195 |
| halfcylinderRe6400 | 0.3679 | 0.3512 | 0.5157 | 0.5852 | 0.1301 | 0.1306 |
| tangaroa | 0.7139 | 0.6897 | 0.7600 | 0.9179 | 0.1197 | 0.1230 |
| deltaWing_resampled | 0.5394 | 0.2758 | 0.8216 | 0.4066 | 0.1157 | 0.1090 |
| deltaWing_LBM | 0.6340 | 0.2649 | 0.7545 | 0.3925 | 0.1522 | 0.1179 |
| f22raptor | 0.6248 | 0.6295 | 0.5736 | 0.8523 | 0.1495 | 0.1568 |
| boeing747 | 0.8534 | 0.8711 | 0.7526 | 0.8199 | 0.3444 | 0.2392 |
| smokeBuoyancy | 0.4524 | 0.4905 | 0.7123 | 0.8593 | 0.1654 | 0.1684 |
| **10项等权平均** | 0.4945 | 0.4293 | 0.5848 | 0.6292 | 0.1442 | 0.1398 |

### Task3逐流场F1

特征列均含同一个Raw25骨干。Raw25-PCA列为同结构残差对照。

| 数据条目 | Raw25 | Raw25-wide | Raw25-PCA残差 | 旧FMT | 短窗口IVD | 六邻居客观Fourier | largeNeighbor |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel | 0.1048 | 0.1079 | 0.2882 | 0.6518 | 0.7831 | 0.1915 | 0.1953 |
| cylinder3d | 0.3382 | 0.3473 | 0.4843 | 0.5059 | 0.7055 | 0.3814 | 0.3589 |
| halfcylinderRe640 | 0.7141 | 0.7044 | 0.7455 | 0.7364 | 0.8672 | 0.7160 | 0.7163 |
| halfcylinderRe6400 | 0.5929 | 0.6003 | 0.6953 | 0.6549 | 0.8742 | 0.6017 | 0.6177 |
| tangaroa | 0.7427 | 0.7484 | 0.7905 | 0.8207 | 0.9058 | 0.7633 | 0.7650 |
| deltaWing_resampled | 0.8595 | 0.8636 | 0.8891 | 0.9181 | 0.9178 | 0.8936 | 0.8996 |
| deltaWing_LBM | 0.8138 | 0.8140 | 0.8506 | 0.8850 | 0.9025 | 0.8592 | 0.8452 |
| f22raptor | 0.8244 | 0.8180 | 0.8363 | 0.8259 | 0.8706 | 0.8282 | 0.8270 |
| boeing747 | 0.8257 | 0.8268 | 0.8410 | 0.8679 | 0.8861 | 0.8385 | 0.8352 |
| smokeBuoyancy | 0.7322 | 0.7278 | 0.7493 | 0.7562 | 0.8664 | 0.7282 | 0.7294 |
| **10项等权平均** | 0.6548 | 0.6559 | 0.7170 | 0.7623 | 0.8579 | 0.6802 | 0.6790 |

逐流场核对：Task1/2的largeNeighbor均低于旧FMT（各10/10个条目）；Task3仅F-22的F1略高于旧FMT，其余9项较低。Task3相对Raw25-PCA仅DeltaWing-resampled较高，另外9项较低。所有条目均保留，没有因结果好坏删除流场。

### 客观性与解释边界

训练片客观性检查对每个流场32个相同材料primitive施加逐时刻变化的正交旋转和平移，编码器相对误差最大9.44e-11；保留系数的平方和中，非零频占14.96%–46.37%。这些结果支持当前编码器实现保持客观性且使用了非零频，不能替代分类性能证据，也不保证Task3坐标Raw分支或完整网络客观。

本描述符记录邻域形状随时间的变化。全25点邻域共同作刚体旋转时，同时间内积恒定，减去初始值后得到零信号。这是其数学性质；把它解释为本轮分类下降的原因仍是待验证假设。当前实验使用初始时刻IVD标签，没有测路径几何token化质量，因此不能据此证明或否定用户关于短窗口方案不擅长几何token化的判断。旧FMT、原短窗口和已停止采用的longtime版本均保留，本轮不修改论文冻结主表。

### 可复现依据与完成状态

| 项目 | 证据 |
|---|---|
| 实验版本/配置 | `Verify_LargeNeighbor_1.1`；[config](../config/Verify_LargeNeighbor_1.1.json) |
| 编码/数据/运行代码 | [LargeNeighbor_3D.py](../FMT_Utils/LargeNeighbor_3D.py)、[Build_LargeNeighbor_3D.py](../experiments/Build_LargeNeighbor_3D.py)、[Run_LargeNeighbor_3D.py](../experiments/Run_LargeNeighbor_3D.py) |
| 独立审计代码 | [Audit_LargeNeighbor_3D.py](../experiments/Audit_LargeNeighbor_3D.py) |
| 基础commit | `aee4bd562d340158118f2e41f40129a9918e06a1`；新增代码以冻结源归档为准，不能将基础commit单独当成完整新实现 |
| 冻结源归档SHA256 | `42d44ba73fb73cbac08eeb96f6a307ebc51000af964f0ead751c878d07ba2c8c` |
| 源清单SHA256 | `7f424f8a70d3484cb348feaa5175ed49913e7e7212354f6b3d75ac5a8fe79d3a` |
| 实际运行config SHA256 | `ff983045b66991a3f97bdaa5268b238420b3e5c8b9341e3872ea35f3430d58ce` |
| Task1/2/3数组作业 | `51477161` / `51477162` / `51477163`，各50个子作业 |
| 最终独立审计 | `51477164`，2026-09-07 23:01:04–23:01:21（UTC+03:00），cn511-15，CPU，exit0 |
| 审计范围 | 全部150次运行、950行指标，逐预测重算及相同输入/标签/源代码/参数量检查均通过 |
| 临时模型 | 最终指标落盘后删除350个checkpoint，远端剩余0，未下载模型 |
| 作业历史 | [ibex_run_registry.md](ibex_run_registry.md)；Boeing缓存曾因磁盘配额失败，迁移并校验后原配置续跑；科学训练无失败重试 |

远端结果根目录：`/home/zhanx0o/FMT_Uniform_3D_20260901/outputs/Verify_LargeNeighbor_1.1`。最终证据文件包括 `report_zh.md`、`per_run.csv`、`dataset_metrics.csv`、`task_macro.csv`、`paired_comparisons.csv`、`independent_audit.json`、`execution_records.csv` 和 `checkpoint_cleanup.json`。本节依据上一轮成功读取的最终报告及已登记审计结果写入；本次没有下载这些原始文件，也没有声称完成本地逐预测复算。此前Task1/2阶段数值及下载审批历史仍保留。

## 2026-09-08 — Other_Task2_VisualAnalysis_1.1：几何与潜在特征联动探索

用户要求参考 FlowNet 实现 Task2 的多簇可视分析。论文核对修正：之前对话中说默认直接在高维空间 DBSCAN → 现按 FlowNet §3.2、附录 PDF 第12页明确的 **t-SNE 后 DBSCAN** 作为默认，同时提供高维空间模式。此前未读论文便作出的流程假设不成立。FlowNet §3.1 使用普通卷积自编码器，并非 VAE；本工具沿用我们的 Task2 VAE，不声称复现 FlowNet 网络。

| 项目 | 实施与证据 |
|---|---|
| 实验版本/配置 | `Other_Task2_VisualAnalysis_1.1`；`config/Other_Task2_VisualAnalysis_1.1.json` |
| 主要代码 | `experiments/Task2_Visual_Analysis.py`、`FMT_Utils/Task2VisualAnalysis.py` |
| 固定配方来源 | `mainExp_Task2_3D_6.2_uniform_confirmation` 的配置/选择/审计文件通过冻结 manifest 哈希校验；`fmt_all+kin4`、hidden `[512,256]`、latent 64、两臂同 seed 100、各7000步骤 |
| 本地示例 | Re160 / `cylinder3d`；CPU；未向 Ibex 提交任务 |
| 训练时间片 | ordinal 4/5，t≈7.7/8.8；Cylinder 时间规则排除原冻结训练集中的0–3，本次是新采样下重训，不是严格复现原主表训练 |
| 可视样本 | 开发集 cluster_calibration ordinal 6，t=10.0；源3584样本，seed7068均匀抽取1500；两臂共用原始行号 |
| 处理与标签 | 确定性编码均值 mu，逐样本L1归一化，t-SNE perplexity30、seed7068、1000迭代；DBSCAN eps2/min_samples10；不读取参考标签数组，不打开confirmation/test |
| 默认参数结果 | Raw：29簇、1036/1500噪声（69.07%）；FMT：26簇、1129/1500噪声（75.27%）。只描述此次投影和分析子集，不能据簇数/噪声比例推断谁更好，未按结果调参数 |
| 逐样本证据 | `outputs/Other_Task2_VisualAnalysis_1.1/cylinder3d/reports/{raw,fmt}_tsne_eps2.0_min10/` 下 `analysis.npz`、`samples.csv`、`analysis.json` |
| 训练与源码证据 | 同目录上级 `latent_bundle.npz`、`latent_bundle.json` 含数据/配置/源码哈希、设备、依赖版本和逐臂训练损失；基础commit `aee4bd562d340158118f2e41f40129a9918e06a1`，新增代码以记录的源码哈希区分 |
| 保留规则 | 保存latent/几何及分析结果；输出目录无 `.pt/.pth/.ckpt`，训练模型仅在内存中 |
| 验证 | 12项测试通过；真实页面簇筛选、七线切换、单点选择及框选联动通过。离线文件URL预览被浏览器策略阻止，离线HTML生成与ZIP内容检查通过，未声称离线浏览器验证通过 |

本次支持的结论仅是工具能够按样本编号展示真实轨线与 VAE 表示的对应关系；尚不构成几何类别准确性或 FMT 优于 Raw 的证据。原主表、冻结训练代码和历史指标均未修改。使用见 `docs/Other_Task2_VisualAnalysis_1.1.md`。

## 2026-09-08 — Other_Task2_VisualAnalysis_1.2：交互重积分与隐藏簇

按用户要求，新增流场、积分步长dt和积分步数N控件，以及多选隐藏簇。总积分时长为dt×N，每条轨线仍采样32点。重算开发集几何后，按1.1使用的冻结统一配方重新训练Raw/FMT两臂并更新两图；隐藏簇只影响显示，不改变标签或聚类统计。实现与边界见 `docs/Other_Task2_VisualAnalysis_1.2.md`。

| 项目 | 实施与证据 |
|---|---|
| 配置/代码 | `config/Other_Task2_VisualAnalysis_1.2.json`；`FMT_Utils/Task2GeometryRecompute.py`、`FMT_Utils/Task2VisualAnalysis.py`、`experiments/Task2_Visual_Analysis.py` |
| 成功运行 | `outputs/Other_Task2_VisualAnalysis_1.2/halfcylinderRe640/20260908T111207_bd22d284/`；本地CPU，UTC 11:12:07–11:16:08，约240秒；未提交Ibex |
| 参数与数据 | Re640，dt=0.0125，N=64，总时长0.8；训练ordinal0–5，显示ordinal6、t≈12.2；全部满足Cylinder t≥7.5且源窗口早于confirmation |
| 几何/特征 | 7个时间片完成重积分；显示3832个有效primitive，paths形状[3832,7,32,3]，两臂mu均[3832,64]，样本ID一一对应；每片记录请求与实际dt均为0.0125 |
| 固定训练 | hidden[512,256]、latent64、KL权重1e-6、学习率0.0003、seed100；两臂各完成7000优化步骤；逐臂损失及源码/速度数据哈希见latent_bundle.json |
| 浏览器验证 | 从Re160自动切换到新Re640几何与特征；默认FMT投影分析1500样本，27簇、912噪声；这些仅是默认参数结果，不构成方法优劣判断 |
| 隐藏验证 | Re160隐藏Noise(-1)时显示从1500降为371，左右图同步移除；原26簇、1129噪声统计不变。新Re640隐藏Cluster 0的49个样本后显示1451/1500，聚类统计不变；清空选框恢复。全部隐藏及标签不变另有自动测试 |
| 失败保留与修正 | 首次请求`20260908T110921_0f186807`失败：目标时长0.8超过float32源时钟构造的局部窗口0.799995。保留job.json及异常；修正为必要时多读一个真实插值帧，不修改dt，并仍检查源末尾/confirmation边界 |
| 自动验证 | 21项测试通过：实际常速度轨线终点随dt/N改变、float32时钟边界、拒绝confirmation帧、双视图隐藏及全隐藏、失败记录和恢复；输出无模型checkpoint |

本次验证支持重积分、重新编码与双视图切换的实现正确性；未对10个流场逐一重训，未调整主表配置，也未使用参考标签或confirmation结果选择参数。

## 2026-09-08 — Other_Task2_VisualAnalysis_1.3：取消交互时间划分限制，增加UMAP/KMeans

用户明确纠正：自由交互界面不应受到实验划分时间限制，应允许指定起始时间并在源末尾截断。此前1.2在confirmation起始时间拒绝积分 → 1.3仅按真实源时间范围积分，允许跨越旧划分时间 → 原因是交互需求与独立评测不同，之前把评测约束擅自加到了用户的几何浏览操作中。旧实验和旧失败记录保留，1.3结果不替代主表。

| 项目 | 实施与证据 |
|---|---|
| 版本 | `Other_Task2_VisualAnalysis_1.3`，配置及使用见`docs/Other_Task2_VisualAnalysis_1.3.md` |
| 时间处理 | 任意源范围内t0、完整dt步和末尾短步；实际终点min(t0+N×dt,tmax)，按实际时间重采样为32点；界面明确请求/实际终点和是否截断 |
| 数据使用 | 仅在用户指定t0生成一批primitive，全部用于VAE拟合和显示，不做训练/校准/测试划分，不读取参考标签；逐臂`eval_*`损失在此为同批样本内指标，不是独立泛化结果 |
| 方法 | 两臂同冻结VAE配方，各7000步骤；新增真实umap-learn 0.5.12投影、sklearn KMeans（n_init=10、seed7068），支持二维投影/高维latent聚类；输出标明实际方法和坐标名称 |
| 真实运行 | `outputs/Other_Task2_VisualAnalysis_1.3/halfcylinderRe640/20260908T113546_afd19b63/`；本地CPU，UTC 11:35:46–11:38:53；未提交Ibex |
| 输入/输出 | Re640 t0=14.23、dt=0.025、N=64，请求终点15.83，实际终点15，时长0.77；3847个有效primitive，paths[3847,7,32,3]、两臂mu[3847,64]，分析子集1500；两臂各完成7000步骤，无checkpoint |
| 浏览器验证 | 任意t0重算完成后自动切换新几何与特征；UMAP+KMeans K=8生效，8簇、无噪声；最新页面明确实际终点15和时长0.77。仅验证功能，不据此声称聚类准确性 |
| 自动验证 | 24项测试通过，包括跨旧划分、源末尾截断、不足一个dt的短步、非帧起始时间的时变速度积分解析解、UMAP导出名称、KMeans高维标签不随投影改变，以及原隐藏/联动功能 |

训练仍读取冻结配方及源manifest中的几何网格设置，但不使用其split成员选择交互训练/显示时间；源数据范围内的时间浏览不受旧confirmation边界约束。

<a id="progress-2026-09-09"></a>

## 2026-09-09 — 项目进展、现有证据的范围与 Task6 讨论

本节按用户最新研究判断记录当前进展。已有数值只引用冻结实验；用户最新观察与本轮建议分开标明。本轮没有新增训练、修改冻结算法或重写历史指标。

### 原 FMT 的性能优势与客观性限制

Fourier Map Tokenizer（FMT）将邻近材料路径线组成的 primitive 编码为特征向量。原 `fmt_all` 包含中心播种点轨线的跨时间坐标差分，因此它的已有性能优势不能同时解释为任意时变刚体 observer 下的客观性。静态刚体变换下的不变性与时变 observer 客观性必须区分，源代码见 [DFT_FMT_3D.py](../FMT_Utils/DFT_FMT_3D.py)，此前 observer 推导与诊断保留。

用户概括为“Task1/2/3/5 均表明原 FMT 有性能收益，但该版本不客观”。原 FMT 的收益有实验支持，但必须按具体配方解释，不能把所有历史表格中名为 FMT 的列当成同一个编码器。

| 证据版本 | 任务及对照 | 已有结果 | 支持范围 |
|---|---|---|---|
| `Verify_LargeNeighbor_1.1` | Task1，Raw7 / 原 FMT | 平均 F1 0.4238 / 0.5959 | 固定聚类设置下原 FMT 优于 Raw |
| 同上 | Task2，Raw7+VAE / 原 FMT+VAE | 平均 F1 0.4945 / 0.5848 | 同冻结变分自编码器（VAE）训练设置下，FMT 输入的聚类效果更好 |
| 同上 | Task3，Raw25-PCA 残差 / Raw25+原 FMT | 平均 F1 0.7170 / 0.7623 | 对主成分分析（PCA）残差容量对照仍有收益 |
| `Verify_AIVDLongtime_1.1` | Task5，Raw-PCA / 原 `fmt_all+gram2+kin6` | 平均 F1 0.63194187 / 0.72773459 | 本轮晚期 Cylinder、不同尺度 IVD 监督设置下有收益 |

F1 是精确率和召回率的调和平均；上表是各数据条目等权平均，不代表每个条目都改善。前三行来自[完整 LargeNeighbor 结果](#large-neighbor-1-1-final)，最后一行来自本文件 2026-09-07 的 `Verify_AIVDLongtime_1.1` 最终审计记录及 [Longtime 配置](../config/Verify_AIVDLongtime_1.1.json)。这两组实验分别保留其输入线数、配方和对照，不合并成一次统一确认。

**需要明确的历史解释修订**：把各任务表中的“FMT 增益”一概归为原 `fmt_all` → 现在按冻结配方区分 → 依据是 Task3 当前统一主表 `mainExp_Task3_3D_9.2_uniform_confirmation` 的 [manifest](../outputs/mainExp_Task3_3D_9.2_uniform_confirmation/frozen_recipe_manifest.json) 明确选择 `g08_aivd1w3_dft` → 先前笼统说法混淆了原 Fourier 几何描述符与短窗口 IVD 估计。该主表不是原中心差分 FMT 的证据；原 FMT 的 Task3 收益由上表的独立对照支持。数值和冻结算法都不改写。

### 去中心时间差分后的客观版本

`fmt_all_v2` 与 `largeNeighbor` 使用同一时刻相对几何的内积标量序列再作 Fourier 分析，保留独立名称。`largeNeighbor` 用 24 邻居的完整 32 点序列，不是短窗口 IVD。其代码及配置见 [LargeNeighbor_3D.py](../FMT_Utils/LargeNeighbor_3D.py)、[Verify_LargeNeighbor_1.1.json](../config/Verify_LargeNeighbor_1.1.json)。

| `Verify_LargeNeighbor_1.1` | 原 FMT | 六邻居客观版本 | 24 邻居客观版本 |
|---|---:|---:|---:|
| Task1 平均 F1 | 0.5959 | 0.1417 | 0.1299 |
| Task2 平均 F1 | 0.5848 | 0.1442 | 0.1398 |
| Task3 平均 F1 | 0.7623 | 0.6802 | 0.6790 |

这支持“目前测试的客观替代方案在冻结任务上的性能较差”，尤其 Task1/2；**尚不支持“客观表示必然较差”或“原 FMT 的全部优势都由中心差分造成”**。替代版本还改变了描述符构造、维度和信息保留方式，不能把这个比较等同于只删除一个输入块。

Task5 的边界同样保留：`Verify_FMTAllV2_1.1` 中独立 v2 的 F1 从原方案 0.675298050 降到 0.615071320；保留旧邻居辅助块的 `Verify_FMTAllV2_1.2` 则是 0.678020588 → 0.676800723，平均精确率从 0.719512817 升到 0.723950925。后者没有证明完整网络客观，因为 Raw 分支和保留的运动学块不在客观核心的认证范围内。两轮含旧时间划分，不能与晚期 Cylinder 实验直接归因比较。证据与容量控制见本文件对应完成记录及 [1.2 控制定义](Verify_FMTAllV2_1.2.md)。

### 短窗口 IVD 估计的定位

Instantaneous Vorticity Deviation（IVD，瞬时涡量偏差）衡量局部涡量相对于同时间参考平均涡量的偏离。用户称作 a1ivd/1ivd 的方法，本记录统一指冻结名称 `aivd1w3_dft`。它从同步邻居间距的变化估算速度梯度及涡量，扣除同时间样本平均涡量，取偏差模长；最后只保留前三个标量的未归一化零频系数，即三者之和。前三个是时间样本，不是三个空间分量；有限差分使第三个标量还依赖第四帧坐标。详细符号、实现与离散客观性边界见 [实现说明](aivd1w3_dft_explained_zh.md)。

用户当前解释是：这个方法恰好接近现有初始时刻 IVD p95 标签所需要的物理量，因而在这些任务上很强，却未证明对整个 primitive 的通用几何信息有良好表示。作为论文表述，本记录采用“**由几何估计一个任务相关物理量，不能替代全轨线几何表示的证据**”，不把它写成数学上完全不包含任何几何信息。

其余后续时间样本不进入这个一维输出；这一实现事实及 `Verify_AIVDLongtime_1.1` 中长窗口汇总平均下降，与上述解释相符，但不能单凭分类结果确认唯一因果。保留原方案作为物理量估计对照；按用户既有决定不继续采用当前 longtime 版本，不删除其代码与结果。保留七 observer 平移验证，不将该验证单独扩大为任意时变旋转的完整数值认证。

### 新的轮廓系数观察与数据限制

**用户于 2026-09-09 报告，尚未在本轮定位到对应逐次结果与完整配置**：对 primitive 特征采用多种降维和多种聚类，汇总平均轮廓系数；FMT 优于普通 VAE 和曲率等传统微分几何基线，短窗口 IVD 估计方案较差。结合可视化，用户认为这是 FMT 抓取几何信息的积极证据。本轮明确保留该观察，不虚构数值、实验 ID、方法列表、种子或独立审计结论。

当前已定位的 `Other_Task2_VisualAnalysis_1.1–1.3` 是交互可视化/重算记录，并不含上述完整轮廓系数比较，不能用其中簇数或噪声比例代替。归档这项新实验时需要记录数据时间窗、每种特征/降维/聚类配置、随机种子、在哪个空间计算距离、噪声及无有效聚类组合的处理、逐组合分数与平均规则。

轮廓系数根据簇内距离与最近其他簇距离计算，度量选定空间里的紧凑与分离程度。[官方定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) 因此，本轮推断是：更高的分数可支持当前流程下的可聚类性；结合原轨线可视化可以加强几何解释，但仅凭不同特征/降维空间的平均分数，还不能证明 token 保留了多少完整流映射信息。这正是 Task6 希望补充的证据。

**用户最新数据判断与项目决定**：可用流体数据及可靠标签不足；用户检查 Task4 的 hairpin 人工标注时发现不少错误，目前基本放弃该方向。本轮据此将 Task4 记为暂缓后续推进，保留既有协议、标签问题与历史结果，不继续把它当作主要突破口。这里记录用户报告，不声称本轮独立审计了错标比例。

### 下一步仅讨论 Task6

新任务希望直接检验 token 对整段材料运动和相对几何的保留及使用能力，不再依赖人工涡标签。已写入[三个 Task6 候选](Task6_candidates.md)：A 可查询的局部流映射重建；B 遮挡区域的流映射补全；C 短流映射 token 的组合。当前建议优先讨论 A，用户尚未选定正式 Task6，本轮没有注册版本、运行实验或宣称新方案有效。

## 2026-09-09 — mainExp_Task678_FlowMap_1.1：全部选题、代码及部署前验证

用户随后明确选择三个方案全部执行，A/B/C 正式对应 Task6/Task7/Task8，并授权代码验证后 Git commit/push、Ibex Git 拉取和批量比较。此前“尚未选择”的记录是当时状态，不再代表当前决定。正式任务和预注册设置见 [Task6/7/8 协议 1.1](Task678_flowmap_protocol_1.1.md)，配置 SHA256 `e27e18f54c49a0da9477291f6c709f5a60cc47f4c1fb832896e5c9eaaad48e34`。

| 项目 | 实施与证据 |
|---|---|
| 数据 | 九个真实时间序列条目，Channel 单时刻 VTK 不混入；每流场八个不共享插值源帧的窗口，4/2/2 拆分；Cylinder 起始 t≥7.5 |
| 来源检查 | Ibex 只读核对九个来源均至少有八个可用窗口；稀疏导出按已写入帧清单选择。Re160/Re640 薄轴的每轴8点网格不满足区域隔离，按域宽度预先减少到3点，未读取任何新任务性能 |
| Task6 | 独立随机旋转/缩放生成邻域内查询种子；输入32点，目标63点。两个短时间段各自训练查询；不只重构可见七条线 |
| Task7 | 六个外部可见 primitive 的 token 聚合，恢复内部未输入轨线；目标与可见材料种子保持大于r的间隔，IVD均值只由可见集合计算 |
| Task8 | 在内存中复用同一 Task6 解码器，第一段预测终点转为第二段局部查询；与125点真实长轨线比较，非未知未来预测 |
| 比较 | 12种特征/诊断方案，以及真实存储量的七线仿射插值；32/64维、3种子；9×3×2=54个GPU分片，每分片39条任务/方法评估，共2106条评估、4212条测试窗口记录 |
| 核心代码 | `FMT_Utils/FlowMapData_3D.py`、`FMT_Utils/FlowMapModels_3D.py`；Build/Run/Audit/Submit入口为 `experiments/*_Task678_FlowMap_1_1.py`；未修改冻结 `DFT_FMT_3D.py` |
| 单元验证 | `python -m unittest discover -s tests -p test_task678_flowmap_3d.py -v`：10项PASS，包括解析非定常平移/剪切、RK4内部越界、缺帧和时间隔离、薄轴网格、隐藏目标不进入特征、上下文置换、初始条件和预测终点组合 |
| 完整本地验证 | `outputs/Verify_Task678_LocalSmoke_1.1/local_20260909_02/`；全部12特征和插值完成39条评估，独立保存预测复算78条窗口指标PASS；每个模型仅3次更新，用于可执行验证，不是科学性能证据 |
| 验证与保留 | `independent_audit.json`、`per_window_metrics.csv`、`summary.json`及源配置保留；0个checkpoint；早先固定查询点的开发验证`local_20260909_01`也保留，不用于选择方法 |
| 部署计划 | 独立Git分支 `codex/task678-flowmap-tokenization`；Ibex旧实验目录不是Git检出，本批使用独立Git目录；source/smoke/build/cache_audit/train/final_audit依赖链，提交ID和设备另记作业登记表 |

本节只支持实现可执行、数值/信息边界检查通过。真实流场方法性能尚未产生，不能用短步数解析流验证的误差排序评价FMT。任何真实数据失败、重试或实现修正继续追加，不覆盖原运行记录。

### 2026-09-09 15:31+03 — Git 推送与 Ibex 批量提交完成

代码 commit `237f4841c0c8f848cf69ad88f3743ad3eaf879e7` 已推送至 `codex/task678-flowmap-tokenization`，Ibex `/home/zhanx0o/FMT_Task678_20260909` clone/pull 后核对到同一 HEAD。提交 source `51630116`、smoke `51630117`、九条数据构建数组 `51630118`、缓存审计 `51630119`、54条GPU数组 `51630120`、最终审计 `51630121`，全部按 afterok 依赖连接。任务版本、提交时刻、配置/commit及设备请求已登记 `docs/ibex_run_registry.md`。首次核对为source/smoke因Priority排队，其余等待依赖，尚无真实流场性能结论。

15:33+03，Ibex实际source检查与完整smoke均已exit0：同一Git检出的10项单元测试、39条评估、78条窗口指标复算全部PASS；数据构建数组已开始运行。此为真实集群环境中的代码验证，仍不是九流场性能结果。源码执行commit保持`237f4841`，后续作业登记文档提交不改变该批运行源码。

### 15:44+03 — 九流场构建和缓存审计完成

构建数组`51630118_0–8`全部COMPLETED/exit0。缓存审计`51630119`于15:43:41–15:44:38在cn604-18完成，检查72个不重叠源时间窗口、材料点数组、隐藏区域隔离、所有特征文件与配置校验值，结果PASS。训练数组的依赖已满足，此次核对为Priority排队。

下表计数单位是一个中心区域及其配套可见支持/独立查询粒子，不是独立物理流场的数量。Task6 每个区域包含两个短时段查询实例，Task7/Task8 各一个实例；方法间使用相同区域集合。每时段八个查询粒子的随机位置在数据构建时确定。

| 数据条目 | 训练区域 | 验证区域 | 测试区域 |
|---|---:|---:|---:|
| Cylinder Re160 | 453 | 225 | 223 |
| Half-cylinder Re640 | 448 | 222 | 222 |
| Half-cylinder Re6400 | 429 | 224 | 220 |
| Tangaroa | 436 | 229 | 226 |
| DeltaWing resampled | 450 | 222 | 216 |
| DeltaWing LBM | 450 | 222 | 216 |
| F22 | 390 | 198 | 187 |
| Boeing | 450 | 222 | 216 |
| SmokeBuoyancy | 511 | 256 | 254 |
| 总计 | 4017 | 2020 | 1980 |

证据为远端`outputs/mainExp_Task678_FlowMap_1.1/build/<dataset>.json`及`cache_audit.json`，初始物理时间、半径、实际网格形状、保留/排除原因及文件SHA256逐窗口保存。上述记录属于输入/实现审计，尚不支持任何方法性能排名。


### 2026-09-10 — Verify_Task1235_ObjectiveFMTnTDO_1.1（预注册，尚无性能结论）

用户明确要求直接测试完整 objective_fmt_nTDO 1.1 在 Task1/2/3/5 的性能。配置 `config/Verify_Task1235_ObjectiveFMTnTDO_1.1.json`，执行代码 `experiments/Verify_Task1235_ObjectiveFMTnTDO_1_1.py`。模块保持 369 维（相对向量 Fourier 138 + 同时间 21 点对距离 Fourier 231），六个频率；不加 aivd、运动学块或额外归一化到编码器，不修改冻结 DFT。预处理在下游仅用训练集拟合。

| 任务 | 固定对照 | 训练与评估 |
|---|---|---|
| Task1 | Raw / 旧 fmt_all+kin4 / 完整 nTDO | 训练集 StandardScaler/PCA8/KMeans；验证集定 cluster-to-vortex；7380/7381/7382 三种子 |
| Task2 | Raw / 旧 fmt_all+kin4 / 完整 nTDO | 三臂补零到700维，同一 VAE 512/256、latent64、beta1e-6、lr3e-4、7000步；310/311/312 三种子 |
| Task3 | Raw / Raw-wide / Raw-PCA residual / Raw+旧 fmt_all+kin4 / Raw+nTDO | 各辅助分支369维、同结构138818总参数；最多100epoch、patience20；原验证集融合选择规则；40/41/42 |
| Task5 | 上述五臂（旧FMT用冻结 fmt_all+gram2+kin6）及 fixed-scale Task3 Raw transfer | 辅助分支同样369维，融合系数固定1；原18/6/9个train/validation/test尺度tuple；40/41/42 |

全部10个3D条目均跑三种子，预期120个task/dataset/seed结果包、510行逐臂指标。Task3旧FMT是非aivd历史参考，不能冒充现有aivd主表配方。每个结果包在载入test特征和标签前冻结所有模型、阈值及归一化。保存逐样本预测与分数，独立复算F1/AP；临时checkpoint仅供同一Task3→Task5依赖链使用，审计后删除，不下载。

时间策略：fixed缓存沿用既有late-cylinder划分；Task5 Re160/Re6400旧development不满足t>=7.5，按独立新YAML重建。原始索引development[75,77,79,81,95,97]，test[125,128,131,134]；Re6400使用已导出的原索引75..149片段，文件保留原始索引75..149，不减75，不再次截半。其余8组Task5复用冻结缓存。所有方法同一条目使用完全相同的有效样本与标签，p95不变。新建两组Task5内train/validation/test完整加载时间窗互不重叠。

证据边界：这是复用benchmark的配对性能诊断，不是新的独立confirmation。历史Task1/2/3和fixed Task3 transfer的不同缓存文件仍可能有物理时间窗重叠，不声称完全时间独立；不依据此次test指标改配方。性能结果不替代客观性代数验证。当前仅预注册，F1待真实运行。


执行纠正（同一方法1.1，execution revision 2）：首次build 51696322中错误地对Re6400文件索引减75，读到未填充帧，产生全正标签。原因是将“仅填充原索引75..149”误读为“裁剪为75帧”；源码 Prepare_Task5_AIVDLongtime_1_1_Source.py 明确保留完整t维并写入 var[i]。已取消51696324–51696328下游作业，未将这批错误缓存用于性能结论。原无效缓存保留在late_task5_cache，修正缓存另写late_task5_cache_r2；物理时间、尺度tuple、训练预算和nTDO配方不变。新增构建前时间坐标/有效数据检查及二分类样本检查。


### Verify_Task1235_ObjectiveFMTnTDO_1.1 — Task1 已完成（2026-09-10，execution revision 2）

运行代码commit `d04f5609923bc67a1d523e56fab3e71c18c8eb2d`；job `51696571`，10数据×3种子，90行指标；训练集StandardScaler/PCA8/KMeans，验证集定涡cluster，全部配方已预注册。`outputs/Verify_Task1235_ObjectiveFMTnTDO_1.1/Task1_local_audit.json` 本地独立逐样本F1复算PASS，30份prediction文件哈希一致。

| 方法 | 宏平均 F1 | 三个种子宏平均的标准差 |
|---|---:|---:|
| Raw | 0.423525 | 0.000058 |
| 旧 fmt_all+kin4 | 0.595744 | 0.000209 |
| 完整 objective_fmt_nTDO 1.1 | 0.411354 | 0.003247 |

本固定Task1设置下，完整nTDO宏平均比Raw低0.012171，比旧FMT低0.184390；不支持其优于这两个对照。逐数据三种子平均：nTDO高于Raw的6/10组，高于旧FMT的1/10组（f22raptor）；宏平均下降较大的条目包括Re160、Re6400和smokeBuoyancy。此处只报告性能，不把下降归因于客观性或指定某一分支，未做相应因果消融。先前只删中心的编码与当前369维“无时间预差分+距离分支”的完整nTDO不是同一方案，不能混写为同一实验结果。Task2/3/5尚待完成；不根据本次Task1结果修改后续已冻结配方。

逐次表 `Task1_per_run.csv`，汇总 `Task1_summary.csv`，原始预测包 `Task1_evidence.tar.gz`，均位于上述输出目录，无模型文件。


### Verify_Task1235_ObjectiveFMTnTDO_1.1 — Task2 已完成（2026-09-10，execution revision 2）

Job `51696572`，运行代码commit仍为 `d04f5609923bc67a1d523e56fab3e71c18c8eb2d`。10数据×3种子、90行指标，本地 `Task2_local_audit.json` 逐样本F1复算PASS，同时核对三臂同700维输入、相同参数量和各7000步训练全部通过。

| 方法 | 宏平均 F1 | 三个种子宏平均的标准差 |
|---|---:|---:|
| Raw | 0.488298 | 0.007380 |
| 旧 fmt_all+kin4 | 0.559528 | 0.005643 |
| 完整 objective_fmt_nTDO 1.1 | 0.488735 | 0.010162 |

nTDO比Raw高0.000437，宏平均接近；比旧FMT低0.070793。本设置不支持nTDO保留旧FMT在Task2的宏平均优势。逐数据三种子平均，nTDO高于Raw 4/10组、高于旧FMT 4/10组；局部改善与整体差距均保留，不按test改后续配方。此处没有进行机制消融，不归因于某一分支或客观性。记录路径为 `outputs/Verify_Task1235_ObjectiveFMTnTDO_1.1/Task2_per_run.csv`、`Task2_summary.csv`、`Task2_evidence.tar.gz`，无模型文件。Task3/5继续运行。

### Verify_Task678_DirectFMTFit_1.1 — 全部36组训练及270项预测复算完成

实验日期2026-09-10；正式运行代码commit `05cb78bbed88c75c695a05f01335cea64c03a6bd`，配置 `config/Verify_Task678_DirectFMTFit_1.1.json`，Git/Linux SHA256 `8a4be2e873b0c8919b19da5c1ef779accd2a6789dfcd9c6fdadb7aeb6ea2c9e2`。协议 `docs/Task678_direct_fmt_fit_protocol_1.1.md`。冻结161维fmt_all全部直接进入可训练网络，不使用主成分分析或变分自编码器；网络内部只做训练统计确定的可逆逐通道仿射缩放。大网络3504131参数、小网络71939参数。训练区域总数4017→32157，新增播种及积分只在原四个训练时间窗口内；原验证2020和测试1980区域不变。Cylinder起始t≥7.5。

每条件覆盖全部九个非定常3D数据条目、优化种子9100。small_base为小网络/2000更新；large_base为大网络/原数据/30000更新；large_expanded为相同大网络/扩充数据/相同30000更新。memorize16每流场使用首16个训练区域，最多40000更新，仅完整训练误差≤0.001初始半径时按预注册规则提前停止。Task6/7分别训练，Task8复用Task6网络，将第一段预测终点作为第二段查询位置，不独立训练长目标。

下表是九个数据条目等权宏平均，数值为非初始采样时刻粒子位置均方根误差除以初始半径，越低越好。训练与测试分别报告；Task8训练域数值为训练区域内的长轨迹组合误差，不是直接优化过的长目标损失。

| 条件 | Task6训练 | Task6测试 | Task7训练 | Task7测试 | Task8训练域 | Task8测试 |
|---|---:|---:|---:|---:|---:|---:|
| small_base | 0.631539661 | 1.860203419 | 0.389294800 | 2.036656419 | 1.159251476 | 3.929661370 |
| large_base | 0.081370155 | 1.855058312 | 0.036258319 | 1.888685364 | 0.238638930 | 13.946947702 |
| large_expanded | 0.157718969 | 1.579466724 | 0.078373165 | 1.818462935 | 0.370706349 | 3.615534976 |

固定小训练集的完整拟合结果如下；没有用test挑选曲线或流场。

| 数据 | Task6训练 | Task7训练 | Task8训练域组合 | Task6更新数 | Task7更新数 |
|---|---:|---:|---:|---:|---:|
| cylinder3d | 0.003759435 | 0.003548373 | 0.007064465 | 40000 | 40000 |
| halfcylinderRe640 | 0.006735118 | 0.006586879 | 0.032416061 | 40000 | 40000 |
| halfcylinderRe6400 | 0.013151533 | 0.015570956 | 0.026920888 | 40000 | 40000 |
| tangaroa | 0.016648818 | 0.004739292 | 0.035273035 | 40000 | 40000 |
| deltaWing_resampled | 0.000957728 | 0.000937620 | 0.001679918 | 10500 | 11500 |
| deltaWing_LBM | 0.001271071 | 0.001329212 | 0.002923418 | 40000 | 40000 |
| f22raptor | 0.023125945 | 0.018354236 | 0.057800551 | 40000 | 40000 |
| boeing747 | 0.000989559 | 0.000996099 | 0.001772570 | 30000 | 29000 |
| smokeBuoyancy | 0.000978781 | 0.000818219 | 0.003136123 | 24500 | 24500 |

18个Task6/7小集拟合网络最终训练误差范围0.000818219–0.023125945；DeltaWing resampled、Boeing747和smokeBuoyancy三个流场两个任务均达到0.001目标，其他六流场均完成40000更新。Re160 Task6从8.197304315降至0.003759435，Task7从9.046948019降至0.003548373。结论：在这些真实训练样本上，大网络已表现出显著拟合能力，但拟合成功不等于泛化成功。

Task6：同一原训练数据上，大网络加更多训练更新使训练域宏平均误差降低87.116%；相同大网络和更新预算下，扩充数据后的测试宏平均变化-14.856%，九流场中9/9降低。
Task7：同一原训练数据上，大网络加更多训练更新使训练域宏平均误差降低90.686%；相同大网络和更新预算下，扩充数据后的测试宏平均变化-3.718%，九流场中7/9降低。
Task8：同一原训练数据上，大网络加更多训练更新使训练域宏平均误差降低79.414%；相同大网络和更新预算下，扩充数据后的测试宏平均变化-74.077%，九流场中9/9降低。

必须保留的失败：large_base的Task8测试宏平均显著增大，其中F22从small_base的10.017358002增至98.607658321；其预测首段终点位于第二段支持域之外的比例为97.6604%，large_expanded则为87.6337%、误差9.261361481。这些是Task8_test.json直接记录的现象，表明本次固定评估包含域外查询；未做机制消融，不能把全部误差唯一归因于某一因素。预测终点没有被真实终点替换，也没有裁剪掉失败样本。

比较边界：small_base→large_base同时改变网络容量和训练预算，不能把增益单独归因于参数量；large_base→large_expanded保持网络和优化配置相同。扩充的是既有训练时间窗口中的播种样本，不是新增独立物理流场。全部条件只有一个优化种子，不能声称多种子稳定排名；本轮没有训练同容量Raw基线，不能宣称FMT优于Raw。不得用本次test结果改写冻结配置或旧mainExp结论。

Ibex最终审计51696913重读保存预测与真值，复算全部270项task/role/condition/dataset记录及相关指标PASS，预测文件哈希核对通过，模型checkpoint文件数为0。已取回 `outputs/Verify_Task678_DirectFMTFit_1.1/metrics.csv`、summary.json、independent_audit.json、job_events.jsonl及完整配置；本地local_delivery_audit.json另核对文件哈希、270行唯一性和30个九流场宏平均分组PASS。运行开始/结束、节点和GPU见docs/ibex_run_registry.md本版本完整登记。

小训练集学习曲线：`outputs/Verify_Task678_DirectFMTFit_1.1/figures/small_set_fitting.pdf`，另有SVG/PNG、全部1238个原始曲线点CSV、source_snapshot.json及figure_contract.json。绘图代码 `experiments/Plot_Task678_DirectFMTFit_1_1.py`，commit `51f4be5a971e7964e9beaeb6179a43b82e494477`。图展示九流场全部训练误差点，无平滑；画布183×157mm、最低字体7pt，文本/布局/碰撞审计无失败，九面板目视检查通过。


<a id="ntdo-task1235-final-20260910"></a>
### 2026-09-10 — Verify_Task1235_ObjectiveFMTnTDO_1.1 四任务最终结果

完整369维 objective_fmt_nTDO 1.1 已完成10个3D条目×3种子×4任务，共120份结果包、510行主指标。执行revision 2的95个Slurm作业全部COMPLETED/exit0；训练代码commit `d04f5609923bc67a1d523e56fab3e71c18c8eb2d`。编码器SHA256仍为 `9ddb3bc4471d4ef3191b78567061956cdfd5bb9c11e9ebbcff888c9f18246e4c`，运行中没有修改特征、数据划分、训练预算或选择规则。仅按实际耗时调整Slurm预留时限，记录在resource_updates.jsonl。

以下为10条目等权、每条目三种子平均的宏F1；Task3/5的FMT列均表示Raw+FMT残差分支，Raw-PCA为原始轨线经主成分分析后接相同分支的参数量对照。旧FMT在Task1/2/3采用fmt_all+kin4，在Task5采用fmt_all+gram2+kin6；本表不使用a1ivd，不替换历史aivd主表。

| 任务 | Raw | Raw-wide | Raw-PCA | 旧FMT | 完整nTDO |
|---|---:|---:|---:|---:|---:|
| Task1 | 0.423525 | — | — | 0.595744 | 0.411354 |
| Task2 | 0.488298 | — | — | 0.559528 | 0.488735 |
| Task3 | 0.632093 | 0.627990 | 0.686009 | 0.761062 | 0.753333 |
| Task5 | 0.610622 | 0.616785 | 0.620485 | 0.732271 | 0.723182 |

监督任务的平均精确率（Average Precision）：

| 任务 | Raw | Raw-wide | Raw-PCA | 旧FMT | 完整nTDO | fixed Task3 Raw transfer |
|---|---:|---:|---:|---:|---:|---:|
| Task3 | 0.680141 | 0.680030 | 0.728698 | 0.813913 | 0.803540 | — |
| Task5 | 0.646098 | 0.649700 | 0.660146 | 0.797662 | 0.780242 | 0.429579 |

Task5 fixed Task3 Raw transfer的宏F1为0.377967，variable-scale Raw为0.610622。Task3的nTDO三种子宏F1标准差为0.004130，Task5为0.008418；这些是三个种子宏平均的标准差，不是把30个异质条目混在一起的离散度，也不是置信区间。逐种子宏平均见per_seed_macro.csv，逐条目均值/种子标准差见per_dataset.csv。

结果解释：

1. Task1的nTDO低于Raw和旧FMT；Task2与Raw接近、低于旧FMT。因此完整369维nTDO没有保持旧配方在本次两个无监督任务上的宏平均优势。
2. Task3的Raw+nTDO比Raw提高0.121240、比同结构Raw-PCA提高0.067324，比旧FMT低0.007729。逐条目三种子平均高于Raw 10/10、高于Raw-PCA 9/10、高于旧FMT 6/10。30次nTDO的验证集选择融合系数全部大于0，没有退化为alpha=0的纯Raw输出。channel差距较大（nTDO 0.514060、旧FMT 0.654929），保留原定10条目宏平均，不删除该条目改变排名。
3. Task5的Raw+nTDO比variable Raw提高0.112561、比Raw-PCA提高0.102698，比旧FMT低0.009089。逐条目高于Raw及Raw-PCA各10/10，高于旧FMT 4/10。训练/验证/测试分别18/6/9个尺度tuple，10条目的三组tuple均已核查互不重叠；1620行逐尺度F1/AP从保存预测独立复算通过。
4. 这组结果支持nTDO作为额外输入表示在当前有监督Task3/5上提供有用分类信号，同时说明其无监督聚类表现没有保留旧FMT优势。它不是“仅nTDO独立输入”的监督实验，不能把Raw+nTDO结果写成仅凭nTDO的性能；也不从F1推断客观性或把性能差异归因于未经消融检验的特定分支。当前nTDO与此前“只删除中心、保留旧邻居编码”的方案不同，实验ID和代码保持分开。

证据范围仍为预注册的复用benchmark配对诊断，非新的独立confirmation；历史固定尺度窗口重叠及Task3 transfer的边界见本节预注册。Re160/Re6400的新Task5缓存使用正确的原始时间索引及t>=7.5，初次错误缓存/取消作业保留在原目录和登记中，未进入本表。

完整输出在 `outputs/Verify_Task1235_ObjectiveFMTnTDO_1.1/`：per_run_metrics.csv、summary.csv、paired_f1.csv、per_dataset.csv、per_seed_macro.csv、所有shards预测/阈值/输入证据、final_audit.json、local_independent_audit.json、slurm_accounting.psv及run_config.json。远端和本地独立审计均PASS：510行主指标F1/IoU（监督任务另复算AP）、1620行Task5分尺度F1/AP及配置/来源哈希一致。完整证据包complete_evidence.tar.gz的SHA256为 `471a80424c0a1a65f5883ace2c98d710098802da72c1d4eac3ecad628c107fc1`。90份执行链审计记录删除了300个临时checkpoint，未下载模型；最后总审计于2026-09-10 13:22:47+03完成。


## 2026-09-10 — Verify_Task678_HanSampling_1.1：Han式密集查询重跑（已预注册并提交）

用户授权依照Han 2024的数据组织改进Task6/7/8。固定源码`45eefee828f5439f6aad6940b01aafe7803aa691`，配置`config/Verify_Task678_HanSampling_1.1.json`，Linux SHA256 `dd58fefe440bc75c5eaa49d1d017a2ccb04ed457977226da167d04da15247ce9`；协议`docs/Task678_han_sampling_protocol_1.1.md`。冻结完整161维FMT和现有512宽可训练残差网络；每primitive用128条训练查询及独立验证/测试材料轨迹，仅偶数时刻监督，奇数时刻单列。50遍等效查询曝光；Raw全672维、零token、仿射插值及同窗口token置换作为对照。Task8扩大仅由可见支持确定的第二段覆盖，正式输出仍传递预测终点。9个非定常3D条目、3个种子、81个神经网络分片，仿射9个独立CPU分片。新共同数据集合不能与旧结果混为配对表。

本地9项测试通过，全部4臂短运行102条指标独立复算通过，篡改指标被审计拒绝。奇数时刻真值任意更改不影响训练统计和最终权重；解析流场验证真实梯度优化可以降低拟合误差。这些是实现正确性证据，不是实际流场FMT能力结论。Ibex流程依次为短运行、数据构建、数据隔离审计、训练/仿射、最终预测复算；尚无本轮正式性能结论。


### 2026-09-10 — HanSampling 实际数据预检与方向信息边界

本地`Verify_Task678_HanDataPreflight_1.1`检查四个已有源文件的首个预定窗口，使用正式采样配置，Re6400/DeltaWing LBM/F22/Boeing分别保留114/111/101/111个区域，角色划分和隐藏输入缓冲均通过；脚本commit`4d32f729`，逐项JSON在同名outputs目录。这是数据可构建性检查，不是性能结论。

独立解析检验`Verify_Task678_FMTDirectionalAmbiguity_1.1`（commit `eb887539`，`experiments/Verify_Task678_FMTDirectionalAmbiguity_1_1.py`）构造两个不同的平滑均匀速度场：`v_a(t)=(1/8+t/8,0,0)`、`v_b(t)=(0,1/8+t/8,0)`，相同轴向七线、相同初始查询及半径0.25。Task6和Task7的161维原FMT token、query、geometry、scales均逐比特相同，而正确坐标轨迹不同。两样本等权平方误差的最佳共同预测是均值，解析例的不可约位置误差/半径为0.2779337031。完整输入与JSON报告在`outputs/Verify_Task678_FMTDirectionalAmbiguity_1.1`；编码器SHA仍为`efff993f10a8db7b481784198beecce1c3c74b2f81cd2eddf5a523ba0a19222d`。

结论边界：此前“扩大网络能降低已有训练集误差”的实验观察仍成立；不能由此推出“原旋转不变token足以唯一恢复任意绝对坐标流映射”。新增解析证据指出原token存在方向信息缺失，增加训练数据和网络不能消除这个反例的输入歧义。这不是客观性测试，也不预判九个实际流场的排名；各方法在其数据分布上的误差仍由已提交实验测量。不能把全部泛化误差归因于没训练好。

本轮正式批次发生home存储配额失败，尚未训练或读取新测试性能；科学配置保持固定，修复存储后重跑失败分片。详见运行登记，不把执行失败作为方法性能。


### Verify_Task678_FMTDirectionalAmbiguity_1.2：补足单流场与未见粒子条件

上一版1.1为两个不同均匀流场的中心query示例，说明原token丢失方向信息；本项目每流场独立训练且Task6要求query未输入，因此不能直接将1.1误差下界称为本项目任务的单流场下界。1.2明确修正该论证范围：采用同一平滑非定常流场`v=(1/8+t/8)*(1-g(z),g(z),0)`，`g`在z≤-2为0、z≥2为1，中间用光滑过渡。两个primitive中心为(0,0,-4)/(0,0,4)，所有可见轨迹分别处于各自均匀区域；query偏移(1/32,1/64,0)与全部输入播种点不同。Task6第一段以及Task7均逐比特验证tokens/query/geometry/scales相同、正确局部轨迹不同。

同一确定性解码器的两例等权位置误差/半径下界仍为0.2779337031。源码`experiments/Verify_Task678_FMTDirectionalAmbiguity_1_2.py`，完整输入和报告`outputs/Verify_Task678_FMTDirectionalAmbiguity_1.2`，输入SHA256`1c7046458d23d19f15f9c7486024841503884ab1589eb0b7faa001c2966bfdd9`。该反例约束原161维FMT与当前合法查询元数据的普遍坐标重建能力，不替代九流场实际性能；不涉及运动观察者客观性结论。旧1.1不删除，本轮HanSampling配方不因此静默改变。


## 2026-09-10 — Verify_Task678_VectorFMT_1.1：保留方向的冻结Fourier token（已提交）

由上述独立解析1.2反例触发，在读取HanSampling实际流场测试指标之前固定本扩展：保留原6个频率，但直接保存中心及六邻居相对增量的三维Fourier实部、非DC虚部和邻居对应关系，不取旋转不变量、不排序邻居。新`vector_fmt6`为231维/924字节，不是原161维FMT；原FMT与全部对照继续独立完成，不改历史代码。编码器仍无训练参数，接同一512宽可训练网络。另有逆Fourier补零重建+仿射查询的无训练`vector_affine`对照。六频截断仍有损，不宣称任意时变观察者客观，也不预先声称查询充分性。协议`docs/Task678_vector_fmt_protocol_1.1.md`。

固定代码commit`d9d04b41f0b01af0a2236707a369e1bb2d58cbeb`，新配置`config/Verify_Task678_VectorFMT_1.1.json`，Linux SHA256 `843eee14d04bc4a2d47a5bd9e43d811dcc5f0ea96508eef46c1be5a0d3ff724d`。完整复用已审计HanSampling缓存，基础数据配置SHA256仍为`dd58fefe440bc75c5eaa49d1d017a2ccb04ed457977226da167d04da15247ce9`；同query ID、轨迹、区域、窗口、种子及50遍曝光量，不能重采样择优。新维度仅改变第一层参数量，记录实际参数而不声称等总参数。27个神经分片+9个无训练对照，预期999条评估，与原批次2376条分开独立复算后配对。

本地3项解析检查与51条全流程短运行复算通过：低频支撑重建、保持净位移、相同原FMT token的方向反例可区分、隐藏目标不进入新token。以上是实现证据，不是实际流场性能结论。

## 2026-09-10 — Verify_Task1235_ObjectiveFMTnTDO_2.1：距离标量 v2 预注册

用户授权新增 `objective_fmt_nTDO_v2` 并部署 Task1/2/3/5。本版直接对全部 21 个同时间欧氏点对距离做傅里叶编码，保留 6 个频率，共 231 维；删除第一版的相对向量傅里叶分支。依用户要求不做客观性验证。原 nTDO 1.1 和历史原始 FMT 均冻结，并在本批作为配对对照重新运行。

| 版本 | 技术及代码 | 指标状态 |
| --- | --- | --- |
| objective_fmt_nTDO_v2 2.1 | `FMT_Utils/objective_fmt_nTDO_v2.py`；无时间差分、无 IVD 估算，仅标量距离傅里叶；`experiments/Verify_Task1235_ObjectiveFMTnTDO_2_1.py` | 预注册，尚无性能结论 |

10 个三维数据 × 3 个种子 × 4 个任务；所有划分、标签、训练预算沿用完整 1.1。Task2 输入统一填充至 700，Task3/5 辅助输入统一填充至 369，以保持网络参数量相同。预计 120 分片、630 行主指标及 1890 行 Task5 分尺度指标。Task5 Re160/Re6400 只读复用已修正的前批 `late_task5_cache_r2`。本地三个实现接口测试和四任务合成数据训练/评分流程检查通过，不能作为性能证据。

完整协议：`docs/Verify_Task1235_ObjectiveFMTnTDO_2.1_protocol_zh.md`；配置：`config/Verify_Task1235_ObjectiveFMTnTDO_2.1.json`。每个 Slurm 进程提交、开始、结束分别记录，训练链结束删除临时 checkpoint；仅保留可复核的预测、指标与配置。本批复用已评估 benchmark，不声称新的独立确认。

### nTDO v2 2.1 已部署（2026-09-10 14:37 +03）

固定执行 commit `55928e99130d5209f2ee027be3a1a94b55d0be56`。Ibex 单元接口检查通过；提交 smoke `51702594`、数据预检 `51702595`、Task1 `51702596`、Task2 `51702598`、Task3→Task5 `51702599`、最终汇总及独立预测审计 `51702600`。GPU 阵列各最多并行 6 个 A100/V100 作业。当前为已提交状态，不能报告 F1。

配置 SHA256 `cc137f5104c4835330d2e2e2da7f53d56c46a47b46b0ea7f93e5a0c2a07a9598`；源文件清单 SHA256 `5e817c49a8557a84c8d3d1b04a8fd4ec949f2e7a59bab1c09de36d4d4595b2ff`。本地提交证据位于 `outputs/Verify_Task1235_ObjectiveFMTnTDO_2.1/`。

### nTDO v2 2.1 前置检查通过

Ibex 的四任务训练/评分流程检查与全部数据预检均为 PASS，记录已同步至 `outputs/Verify_Task1235_ObjectiveFMTnTDO_2.1/{smoke,preflight}.json`。性能阵列依赖已解除，目前等待 Priority 调度；尚未完成性能测试，无 F1 结论。


<a id="task678-han-vector-final-20260910"></a>
## 2026-09-10 — Task6/7/8 Han式密集查询：两套冻结实验全部完成

本轮已经证实原FMT和方向保留新token可以支持已见primitive内的新粒子查询，网络具有明确的拟合能力；预注册的未见primitive主测试仍不支持“FMT神经解码器具有足够的泛化能力”这一更强结论。所有主结果及失败条目均保留。方向token的确定性重建结果另支持其保存了可用的局部流映射信息，不能把该无训练结果写成神经网络结果。

### 方法、实现与证据

| 版本/方案 | 技术细节 | 主要代码 | 维数 / 每primitive字节 / 网络参数 |
|---|---|---|---|
| Verify_Task678_HanSampling_1.1 / fmt_all | 冻结原配方，token全部直接输入512宽可训练残差网络 | config/Verify_Task678_HanSampling_1.1.json；experiments/Run_Task678_HanSampling_1_1.py；FMT_Utils/DFT_FMT_3D.py | 161 / 644 / 3504131 |
| 同版 / raw_positions | 同一可见七线的全部坐标作为网络输入 | 同一训练器和解码器；输入第一层维数不同 | 672 / 2688 / 3765763 |
| 同版 / coordinates_only | 与FMT相同维数的全零token，仅保留合法query和尺度信息 | 同一训练器 | 161零值 / 无流信息 / 3504131 |
| 同版 / affine | 从原始可见轨线作确定性仿射查询和两段组合 | FMT_Utils/FlowMapData_3D.py::affine_query | 672 / 2688 / 0 |
| Verify_Task678_VectorFMT_1.1 / vector_fmt6 | 六频，保留三维复系数方向、相位及六邻居身份；不取模长、不排序 | FMT_Utils/VectorFlowMapFMT_3D.py；config/Verify_Task678_VectorFMT_1.1.json | 231 / 924 / 3539971 |
| 同版 / vector_affine | 从同一231维token补零未保留高频、逆傅里叶恢复支持轨线，再作冻结仿射查询 | FMT_Utils/VectorFlowMapFMT_3D.py及冻结仿射实现 | 231 / 924 / 0 |

新方向编码不是原161维FMT，也不宣称任意时变观察者客观。与Raw相比token字节减少65.625%，这一数值只算可见primitive的编码，不包含网络权重、查询位置和尺度元数据，不能称为整个系统压缩率。Task7每次使用六个外部primitive tokens。

HanSampling冻结训练commit `45eefee828f5439f6aad6940b01aafe7803aa691`，配置SHA256 `dd58fefe440bc75c5eaa49d1d017a2ccb04ed457977226da167d04da15247ce9`，原FMT编码器SHA256 `efff993f10a8db7b481784198beecce1c3c74b2f81cd2eddf5a523ba0a19222d`。一个停滞执行由commit `fddec88b91f93ca430f9ae3edd97152a01a6b9c4`恢复：八个训练/编码/预测源码与原版相同，仅替换经等价验证的全粒子对距离计算内核；重跑后的Task6/7/8拟合预测压缩文件与原输出逐字节相同。详细过程、两个实际运行commit及取消记录见运行登记，不能把审计commit冒充训练commit。

VectorFMT训练commit `d9d04b41f0b01af0a2236707a369e1bb2d58cbeb`，配置SHA256 `843eee14d04bc4a2d47a5bd9e43d811dcc5f0ea96508eef46c1be5a0d3ff724d`，新编码器SHA256 `85e472644473501c21d310dcea7ecb62f13829e67b3095ebc6b9643f3e2b4b86`。该版本在读取真实流场test之前依据已记录的解析方向歧义反例冻结；没有使用本轮test选频率、模型或epoch。

### 数据组织与训练预算

借鉴Han 2024的密集播种/时间查询、分段独立播种和到达位置组合，保留当前512宽网络来隔离输入表示与数据组织的作用；不是对Han原网络的逐项复现。每流场独立训练，覆盖九个非定常3D条目和种子9110/9111/9112。每流场八个真实时间窗口，前四训练、随后两个时间验证、末两个时间测试；原模拟Cylinder起始t≥7.5。每拟合primitive在同一覆盖约束下取固定Sobol序列的128训练、32验证、64测试query，划分在积分/训练前固定。七条输入线各32点，query真值63点。只监督偶数非初始时刻，31个奇数时刻不进入损失、尺度统计、探针或停止条件。

共2516个拟合primitive；Task6两段合计644096条训练短query轨迹，Task7为322048条。每流场只有243–319个不同拟合primitive：密集query数量不能等同于不同局部流映射数量。相同AdamW配置、50遍等效查询曝光，Task6每流场47082–61807更新，Task7为23541–30904。Task8没有独立长目标训练，将首段预测终点传入第二段已知token；真实终点只用于单列诊断。测试不裁剪失败预测或域外到达。

Task6主测试query没有出现在可见输入中。Task7按初始空间缓冲隔离隐藏种子和全部外部邻居；不声称后续整条轨迹包络互不相交。未见primitive按(window,id)定义，时间测试有独立窗口；不声称跨所有时间完全未见全局xyz坐标或跨流场泛化。query_test表示已见primitive内的新粒子，primitive_test才是预注册主结果。所有方案使用完全相同的原缓存与query真值。

### 已见primitive内的拟合和新粒子查询

以下全部位置值均为非初始时刻三维位置欧氏均方根误差除以初始primitive半径r，越小越好。每流场先对三个神经网络种子平均，再对九流场等权平均；确定性方法只有一个结果。

拟合数据（Task8仅是训练区域内组合，不是直接优化的长目标）：

| 方案 | Task6 | Task7 | Task8 |
|---|---:|---:|---:|
| 原FMT + 网络 | 0.027155 | 0.036625 | 0.146806 |
| 方向保留FMT + 网络 | 0.013289 | 0.024013 | 0.114727 |
| Raw + 网络 | 0.012785 | 0.022425 | 0.111604 |
| 零token + 网络 | 1.518227 | 2.045974 | 3.670087 |
| Raw仿射插值 | 0.027086 | 0.093232 | 0.206390 |
| 方向token逆傅里叶 + 仿射插值 | 0.033014 | 0.094723 | 0.209284 |

已见primitive内独立新粒子query_test：

| 方案 | Task6 | Task7 | Task8 |
|---|---:|---:|---:|
| 原FMT + 网络 | 0.027311 | 0.036365 | 0.146498 |
| 方向保留FMT + 网络 | 0.013496 | 0.023774 | 0.114387 |
| Raw + 网络 | 0.013022 | 0.022422 | 0.111091 |
| 零token + 网络 | 1.518625 | 2.046613 | 3.670770 |
| Raw仿射插值 | 0.027177 | 0.093048 | 0.207053 |
| 方向token逆傅里叶 + 仿射插值 | 0.033089 | 0.094529 | 0.209960 |

原FMT的Task6/7新粒子误差为2.731%r/3.637%r，新方向版为1.350%r/2.377%r，Raw为1.302%r/2.242%r。这些数字支持“网络已学会在已见局部流映射内查询新粒子”，不是仅报告小批次训练loss。新方向版在这两个任务上接近Raw，但本轮没有证明其精度优于Raw。

将同窗口token无固定点置换后，原FMT的新粒子Task6/7误差从0.027311/0.036365升至2.319727/2.894371；新方向版从0.013496/0.023774升至2.380600/2.894714。配合零token对照，这证明预测确实依赖流场token，不能说明已泛化到未见primitive。

原FMT在新粒子上的监督偶数时刻/未监督奇数时刻误差分别为Task6 0.027707/0.026908、Task7 0.036988/0.035731；新方向版为0.013670/0.013320、0.024236/0.023302。两类误差处于相近范围，支持区间内时间插值；这些奇数时刻不代表区间外未来预测。

### 未见primitive主测试

| 方案 | Task6 | Task7 | Task8 |
|---|---:|---:|---:|
| 原FMT + 网络 | 1.382312 | 1.496823 | 3.487696 |
| 方向保留FMT + 网络 | 0.959180 | 0.899417 | 8.403275 |
| Raw + 网络 | 0.161853 | 0.308596 | 0.558887 |
| 零token + 网络 | 1.454789 | 1.958378 | 3.624228 |
| Raw仿射插值 | 0.033814 | 0.087329 | 0.279385 |
| 方向token逆傅里叶 + 仿射插值 | 0.040242 | 0.089058 | 0.283920 |

方向版相对原FMT在Task6/7降低误差，但三任务的两个FMT神经方案在每个流场均落后于Raw网络（各任务0/9胜出），也都落后于Raw仿射插值。主测试不能支持“原FMT或新方向FMT神经解码器已经足够用于通用局部流映射查询”的结论。

九流场逐项主误差，每个单元按“原FMT / 方向FMT / Raw网络”排列：

| 流场 | Task6 | Task7 | Task8 |
|---|---:|---:|---:|
| cylinder3d | 1.476387 / 1.028392 / 0.107449 | 1.747799 / 0.753463 / 0.138330 | 3.673970 / 2.609905 / 0.288352 |
| halfcylinderRe640 | 1.178553 / 0.787408 / 0.101591 | 1.182502 / 0.505137 / 0.244777 | 2.813325 / 1.665809 / 0.322293 |
| halfcylinderRe6400 | 4.282879 / 3.177548 / 0.338860 | 4.267770 / 2.455937 / 0.388804 | 10.702469 / 58.152762 / 1.257069 |
| tangaroa | 0.947075 / 0.588808 / 0.170484 | 0.997848 / 0.921151 / 0.432635 | 2.229887 / 2.078503 / 0.533705 |
| deltaWing_resampled | 0.177714 / 0.140677 / 0.046214 | 0.151665 / 0.147360 / 0.070340 | 0.409908 / 0.308575 / 0.106527 |
| deltaWing_LBM | 0.168413 / 0.160696 / 0.046934 | 0.194504 / 0.178891 / 0.071967 | 0.398092 / 0.342676 / 0.106310 |
| f22raptor | 3.355215 / 2.310645 / 0.567079 | 4.262204 / 2.710152 / 1.255752 | 9.200781 / 9.425151 / 2.203466 |
| boeing747 | 0.556552 / 0.286663 / 0.030659 | 0.279865 / 0.253883 / 0.076079 | 1.181754 / 0.644858 / 0.077140 |
| smokeBuoyancy | 0.298021 / 0.151780 / 0.047403 | 0.387251 / 0.168784 / 0.098680 | 0.779076 / 0.401236 / 0.135121 |

物理族等权宏平均也保留，不用修改平均方式隐藏失败：

| 方案 | Task6 | Task7 | Task8 |
|---|---:|---:|---:|
| 原FMT + 网络 | 1.137330 | 1.245715 | 2.866830 |
| 方向保留FMT + 网络 | 0.770876 | 0.794651 | 5.729972 |
| Raw + 网络 | 0.149193 | 0.305138 | 0.507624 |
| 零token + 网络 | 1.263315 | 1.702138 | 3.130330 |
| Raw仿射插值 | 0.031811 | 0.078981 | 0.257423 |
| 方向token逆傅里叶 + 仿射插值 | 0.036372 | 0.080245 | 0.260213 |

### 未见时间窗口测试

| 方案 | Task6 | Task7 | Task8 |
|---|---:|---:|---:|
| 原FMT + 网络 | 1.655329 | 1.905395 | 4.261378 |
| 方向保留FMT + 网络 | 1.278485 | 1.129941 | 5.881276 |
| Raw + 网络 | 0.228240 | 0.387167 | 0.648514 |
| 零token + 网络 | 1.837858 | 2.352536 | 4.925724 |
| Raw仿射插值 | 0.039532 | 0.116866 | 0.229098 |
| 方向token逆傅里叶 + 仿射插值 | 0.046622 | 0.121363 | 0.233422 |

未见时间窗口上同样没有获得FMT神经方案相对Raw的宏平均优势。各方案在独立窗口内仍拥有该窗口的可见支持轨线，所以这是已知token条件下的流映射查询，不能写成未知未来预测。

### 几何、压缩重建与组合诊断

下表主测试单元依次为“同时间粒子云去质心平均位置误差 / 全部粒子对间距平均绝对误差 / 平均终点误差”，均除以r；这三者不是位置均方根误差，不混为同一个指标。

| 方案 | Task6 | Task7 | Task8 |
|---|---:|---:|---:|
| 原FMT + 网络 | 0.035690 / 0.023147 / 1.395509 | 0.031190 / 0.020194 / 1.540039 | 0.070115 / 0.049009 / 3.971187 |
| 方向保留FMT + 网络 | 0.016073 / 0.011343 / 0.484984 | 0.015861 / 0.011011 / 0.402600 | 0.044752 / 0.041764 / 3.117132 |
| Raw + 网络 | 0.016099 / 0.011588 / 0.087728 | 0.020323 / 0.014653 / 0.156524 | 0.034517 / 0.027578 / 0.316367 |
| 零token + 网络 | 0.029928 / 0.020754 / 1.413364 | 0.030974 / 0.021684 / 2.055636 | 0.059818 / 0.045763 / 3.775760 |
| Raw仿射插值 | 0.005631 / 0.004422 / 0.016433 | 0.005488 / 0.004403 / 0.055948 | 0.011627 / 0.011407 / 0.143309 |
| 方向token逆傅里叶 + 仿射插值 | 0.005742 / 0.004516 / 0.016433 | 0.005596 / 0.004503 / 0.055948 | 0.011741 / 0.011497 / 0.143309 |

同一231维方向token经确定性逆傅里叶和仿射查询，在未见primitive的Task6/7/8分别达到0.040242/0.089058/0.283920，未压缩Raw插值为0.033814/0.087329/0.279385。该token以原坐标数据34.375%的字节保留了可用于当前局部查询的流映射信息；Task7/8宏误差接近Raw插值，Task6有可测的截频损失。它在九流场的三个任务均优于当前Raw网络，但没有一致优于Raw插值（分别1/9、3/9、3/9）；这是确定性解码能力证据，不是可训练网络获胜，也不是“充分保留任意流映射”的理论证明。原161维FMT不能共享这一结论。

Task8主测试方向网络宏误差8.403275包含Re6400的58.152762、F22的9.425151，没有删除这两个条目；其余七条目的误差低于原FMT，宏平均仍更差。原FMT、方向网络、Raw网络的预测首段到达位置越出第二段支持域的九流场平均比例分别26.507%、7.031%、0.694%。较少越界不保证较小组合误差，不能仅凭越界比例解释全部误差。真实到达输入的第二段诊断分别1.058535/0.616561/0.077495，但它除以第二段半径且仅评价第二段，不能与上面的整段/r0误差直接相减。

### 结论边界和下一轮假设

此前“小训练集扩大网络可以拟合”的结论保持；本轮在严格区分新粒子和新primitive后补充：拟合与区间内粒子/时间插值已经成功，未见primitive和时间窗口泛化仍不足。以前若将拟合能力直接表述为“足以成为通用flowmap tokenizer”，现在应缩小到上述实验已经支持的范围，原因是新增了同数据、同预算Raw及确定性解码对照。

我认为下一轮应优先检验“token确定性重建 + 可训练残差查询器”，并增加不同primitive/不同窗口的数量，而不仅继续增加同一primitive的query。依据是同一token的确定性重建远好于当前神经直接解码、且每流场有效拟合token只有243–319个；这是待验证假设，不能写成已确认失败原因。任何新网络或数据方案必须另建版本，只用训练/验证选择，并把本轮已读取test明确作为复用benchmark或另建未读取测试；本轮冻结结果不修改。

### 完整性、图与复现文件

原实验2376、新方向实验999，共3375条task/role/variant/dataset/seed记录全部独立重读实际预测及真值复算PASS，九流场齐全。审计源码commit `ea1054f9af40e68ab318c7e96d02406c535f56d9`，含全粒子对距离，没有抽样或删掉高误差粒子。原/新审计JSON的SHA256分别为 `dce4789c5c8dccf95b96cf53ea089d7ce6175136dea4c1dba472479ca8cda110` / `2a35a41c7e0adafbf806f35a7f9a758a3c6a431706a19a2c37340d9aa483c0f1`。元数据包SHA256为 `86dd3b8be6b5ca4aa55490f1d42b8ee1a6cab6ccbc8f0a1f12fb5d37607a449a` / `82115d4457503da09646e31b5b98dfa3df162091c749f017f24c6879d3bfb94a`；本地分别2797/1161个文件逐个哈希验证PASS。模型checkpoint保存和下载数均为0。

完整结果在 `outputs/Task678_HanAndVector_1.1/`：per_flow.csv、macro.csv、paired_differences.csv、diagnostics.csv、training_fit.csv、training_curve_source.csv；各实验目录保留原metrics.csv、summary.json、independent_audit.json、完整config和evidence_metadata。绘图代码及方法索引见 `docs/Task678_plotting_methods_1.1.md`。共同三任务测试图held_out_comparison、九流场训练曲线training_fit以及fixed_examples/figures内九套轨迹图，全部有PDF/SVG/600dpi PNG。固定轨迹样例来自每流场第一个主测试primitive、前8个query、seed9110；Task6只展示第一段，定量测试包含两段；Task7显示全部42条外部支持线。每行共享完整取景范围和正交相机，未裁剪错误预测，样例只作解释不替代全量统计。

最终图形审计：11张完整图、120个面板已逐一目视检查；全部PDF最低字体7pt，碰撞与面板布局审计均PASS且无WARN，实际宽度全部183mm。测试对比/训练曲线/轨迹图高度分别100/164/157mm。静态预检的宽度解析、log检测与不确定性编码提示均已人工核查；本次交付PDF/SVG及600dpi PNG，没有要求TIFF。审计JSON、覆盖层、预览及逐项说明见figure_qa/final_visual_audit.json和同目录文件。

<a id="ntdo-v2-task1235-final-20260910"></a>

## 2026-09-10 — Verify_Task1235_ObjectiveFMTnTDO_2.1 完整结果

Task1/2/3/5 各 30/30，120 个任务/数据/种子分片全部完成。93 个 Slurm 进程均 COMPLETED。630 行主指标、1890 行 Task5 分尺度指标通过 Ibex 和本地独立预测复核。以下全是本次 2.1 同批复跑的宏平均 F1，不替换或混用 1.1 历史值。

| 任务 | Raw | Raw 主成分残差 | 原始 FMT | nTDO 1.1 | nTDO v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task1 | 0.423539 | — | 0.595775 | 0.411354 | 0.305793 |
| Task2 | 0.488126 | — | 0.559912 | 0.488721 | 0.238937 |
| Task3 | 0.629017 | 0.681284 | 0.763586 | 0.750525 | 0.668700 |
| Task5 | 0.615603 | 0.629601 | 0.733821 | 0.729262 | 0.661876 |

| 任务 | Raw AP | Raw 主成分残差 AP | 原始 FMT AP | nTDO 1.1 AP | nTDO v2 AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task3 | 0.678259 | 0.727559 | 0.813825 | 0.806733 | 0.709402 |
| Task5 | 0.651294 | 0.673811 | 0.800243 | 0.780464 | 0.691802 |

AP 指 Average Precision（平均精确率），用于衡量分类分数的排序表现。宏平均每任务覆盖相同 10 数据 × 3 种子；逐数据和逐种子结果分别在 `per_dataset.csv`、`per_seed_macro.csv`，summary 中 std 跨 30 次运行，不应解释为仅随机种子误差。

**本批支持的结论：**距离标量 v2 在冻结的 Task1、Task2 管线中低于 Raw、原始 FMT 和第一版 nTDO；Task3、Task5 相对 Raw 仍有提升，但均低于原始 FMT 和第一版 nTDO。Task3 还低于同结构 Raw 主成分残差，而 Task5 高于该对照。该结果只评估本版距离傅里叶特征及既定分类/聚类流程的性能，不说明所有客观特征都无效，也未单独识别下降的机制。本轮遵照用户要求未进行客观性变换验证。

| 任务 | v2−Raw F1 | v2−第一版 F1 | v2−原始 FMT F1 |
| --- | ---: | ---: | ---: |
| Task1 | -0.117746 | -0.105561 | -0.289982 |
| Task2 | -0.249189 | -0.249784 | -0.320975 |
| Task3 | +0.039684 | -0.081825 | -0.094886 |
| Task5 | +0.046273 | -0.067387 | -0.071946 |

逐数据均值胜出数量（先平均 3 个种子；不以 test 选配方）：

- Task1：raw: 1/10, old_fmt: 1/10, ntdo: 2/10。
- Task2：raw: 0/10, old_fmt: 0/10, ntdo: 1/10。
- Task3：raw: 10/10, raw_pca: 5/10, old_fmt: 1/10, ntdo: 1/10。
- Task5：raw: 10/10, raw_pca: 7/10, old_fmt: 1/10, ntdo: 0/10。

**可复核证据：**固定执行 commit `55928e99130d5209f2ee027be3a1a94b55d0be56`；配置 SHA256 `cc137f5104c4835330d2e2e2da7f53d56c46a47b46b0ea7f93e5a0c2a07a9598`；源清单 SHA256 `5e817c49a8557a84c8d3d1b04a8fd4ec949f2e7a59bab1c09de36d4d4595b2ff`。

本地完整证据目录 `outputs/Verify_Task1235_ObjectiveFMTnTDO_2.1/`；`summary.csv` SHA256 `8ff23680f07fe59f896dfacb2ff579a4731677b47910b4dcd1bfb45919c3bb66`；结果包 `complete_evidence.tar.gz` SHA256 `64a17ae3dd5c554141ee1fe6e7bcc711c2ae80f80afed4bb2f9dac9c9fe90207`。归档 1457 个证据文件，无模型。360 个依赖链临时 checkpoint 全部删除，Ibex 剩余 checkpoint 为 0。Task5 的 18/6/9 个训练/验证/测试尺度组合互不重叠。继承旧 benchmark 的时间窗口与复用评估边界，不声称新的独立确认。


## 2026-09-10 — Task6任务定义更新：mainExp_Task6_PrimitiveVAE_2.1

用户要求当前只专注Task6，并将目标改为单流场内大量多时间、多位置、多积分时长和多邻居距离的七线primitive自重建。输入是完整七线geometry的冻结FMT token，VAE编码器接收token、VAE解码器直接输出同一簇完整七线geometry；测试整个未见primitive，不再查询未输入播种粒子的轨迹。每个网络及预处理统计只属于一个流场。旧查询任务的定义、源码、3375项审计记录和负结果保留原编号，不追溯修改，也不能与新任务混表。

新协议为 `docs/Task6_primitive_vae_protocol_2.1.md`，并已同步AGENTS.md和唯一任务定义。一般流场播种时刻为裁剪前原始时段的10%–80%，Cylinder为50%–80%且保持t>=7；当前三个[0,15]源数据对应[7.5,12]。80%是播种上限，完整积分区间和插值源帧需另行验证。尺度完整记录邻居距离r、数值积分时间步长h、积分步数N及总时长H=N*h，积分后统一成7×32×3，参考Task5接口而不复用其旧时间切片或IVD标签。

首个核心比较为原fmt_all token→VAE→geometry与Raw geometry→相同隐藏结构及潜变量维数的VAE→geometry；共享同一物理primitive和几何目标，不将token重建误差冒充几何重建误差。训练覆盖完整采样时间区间，未见primitive按完整源时间组隔离；未见尺度重建另列。用户关于FMT更适合几何模式学习的解释作为新任务研究假设，不作为已证实的性能结论。

本次仅完成协议更新；新Task6的训练/数据构建代码及数值运行配置尚未实现或冻结，没有提交新Ibex作业或产生新指标。Task7/8暂不推进。旧Task8仍使用其冻结的旧Task6逐粒子查询器，不能自动替换为新VAE重建网络。


### 2026-09-10 — mainExp_Task6_PrimitiveVAE_2.1 实现与部署前冻结

- 用户要求部署Ibex测试新Task6。实现为单流场七线geometry→冻结原161维FMT→可训练VAE→同一七线geometry，Raw-VAE使用相同隐藏结构/潜变量/解码器及训练预算。Task7/8不启动。
- 数值配置：`config/mainExp_Task6_PrimitiveVAE_2.1.json`。九流场各60,000 train+6,000 validation+6,000同尺度test+6,000未见尺度test；18训练尺度、9未见尺度；512宽度、64潜变量、编码解码各3残差块，beta=1e-5；512样本4,000更新拟合检查后重新初始化、120epoch，3随机种子。详细时间表与代码路径见`docs/Task6_primitive_vae_protocol_2.1.md`第7节。
- 本地验证：`tmp/jhtdb-venv/Scripts/python.exe -m unittest tests.test_task6_primitive_vae_2_1`，7项通过（8.616s）。覆盖解析仿射场积分终点、均匀时间重采样、Cylinder原时间范围、稀疏源帧隔离、masked voxel不补零、VAE梯度/均值确定性、数据构建审计与损坏检测。合成小样本训练误差1.131→0.166仅验证代码可以学习，不构成真实流场的FMT优越性证据。
- 原FMT源码SHA256保持`efff993f10a8db7b481784198beecce1c3c74b2f81cd2eddf5a523ba0a19222d`；没有修改冻结算法或旧Task6/7/8结果。当前尚无新真实流场训练结果，不能预写性能提升结论。


### 2026-09-10 — mainExp_Task6_PrimitiveVAE_2.1：54组训练与最终审计完成

**结论：当前原FMT-VAE没有在七线完整坐标重建任务中优于Raw-VAE。九个流场的主测试均为FMT-VAE误差更高。** 这是对本次冻结原FMT配方和相同VAE隐藏结构/预算的实测结论，不推广成所有傅里叶编码或所有tokenizer都失败；此前“适合几何模式学习”是待验证假设，本次没有证实其完整几何重建优势。

证据：代码`59d74fc42904ea45ab75775e11d84c492ca10124`，配置`config/mainExp_Task6_PrimitiveVAE_2.1.json`，SHA256 `a20f11d93d462df22b1af96b414de1e0d6e09e1715bea267f869a8b71f70836d`。9个数据构建和数据审计全部通过；54/54训练完成，全部使用Tesla V100-SXM2-32GB；每组先在512个train样本上拟合4000更新，再全新初始化正式训练120epoch=14160更新、720万primitive曝光。最终审计`51713660`于2026-09-10 19:12:13(+03:00)成功结束，重新从预测核算1566条指标记录。本地报告再次逐项核对CSV与54个result.json、配置、commit和预算，一致。

主指标为七条对应路径线31个非初始时刻的三维位置均方根误差，除以初始邻居半径r；越小越好。下表为每流场三个优化种子的均值±样本标准差；不旋转预测、不重排邻居、不按真实目标挑选潜变量样本。

| Flow | FMT-VAE test RMSE/r | Raw-VAE test RMSE/r |
|---|---:|---:|
| cylinder3d | 5.199786 ± 0.032362 | 0.321845 ± 0.040424 |
| halfcylinderRe640 | 3.696060 ± 0.011607 | 0.388623 ± 0.053366 |
| halfcylinderRe6400 | 13.747953 ± 0.315997 | 1.679649 ± 0.096684 |
| tangaroa | 4.418128 ± 0.018612 | 0.411473 ± 0.002665 |
| deltaWing_resampled | 0.421725 ± 0.008864 | 0.031181 ± 0.000222 |
| deltaWing_LBM | 0.467661 ± 0.014513 | 0.034554 ± 0.000341 |
| f22raptor | 11.746878 ± 0.051456 | 1.947621 ± 0.234189 |
| boeing747 | 0.961566 ± 0.028935 | 0.063205 ± 0.003495 |
| smokeBuoyancy | 1.642650 ± 0.016142 | 0.049045 ± 0.000678 |

九流场等权平均（先对各流场三个种子平均，再平均九流场，不是所有点合并RMSE）：主test FMT-VAE 4.700267，Raw-VAE 0.547466；FMT误差为Raw的8.59倍。独立未见尺度test：FMT 3.940238，Raw 0.600285。未见尺度仅DeltaWing LBM的整体RMSE中FMT略低（0.370366 vs 0.372950）；该集合混合组合插值和半径外推，不能将这个局部结果改写为原FMT普遍优于Raw。逐尺度数字保留在metrics.csv。

拟合诊断（512样本拟合与正式训练是不同初始化的模型）：

| 流场 | FMT小集拟合 | Raw小集拟合 | FMT正式train probe | Raw正式train probe |
|---|---:|---:|---:|---:|
| cylinder3d | 0.377525 | 0.254418 | 2.438337 | 0.275078 |
| halfcylinderRe640 | 0.537474 | 0.358633 | 2.636909 | 0.350661 |
| halfcylinderRe6400 | 2.034719 | 2.234147 | 10.684015 | 1.364500 |
| tangaroa | 0.515096 | 0.248833 | 1.950374 | 0.347521 |
| deltaWing_resampled | 0.060701 | 0.018792 | 0.172570 | 0.023254 |
| deltaWing_LBM | 0.056910 | 0.023213 | 0.183989 | 0.024325 |
| f22raptor | 2.392846 | 2.366213 | 8.190403 | 1.676505 |
| boeing747 | 0.096259 | 0.062317 | 0.532395 | 0.049787 |
| smokeBuoyancy | 0.071628 | 0.015909 | 0.436312 | 0.030130 |

正式train probe固定抽4096个训练primitive，并非全部60000样本误差。以Re160为例，FMT小集误差0.377525，正式train probe 2.438337，test 5.199786；Raw分别0.254418、0.275078、0.321845。小集拟合改善不能替代未见primitive泛化。两臂均真实训练、均使用可训练VAE，结果不是“只有FMT没有神经网络”或作业未跑完造成的。

误差位置诊断：主test宏平均中心线误差FMT 4.691805 vs Raw 0.397423；同时间21对粒子间距离RMSE为0.602123 vs 0.499212。绝对对应坐标误差的差距远大于同时间距离误差的差距，与原FMT未保留完整朝向、轨迹对应信息的解释相容；**这只是解释，不是本实验对唯一原因的证明**。本实验无法单独排除网络结构、优化预算等影响，也不能据此声称FMT完全没有几何信息。不得读取本test后更改超参数再把原test当新确认集。

结果文件：`outputs/mainExp_Task6_PrimitiveVAE_2.1/{summary.json,per_flow_summary.csv,metrics.csv,final_audit.json,config.frozen.json,slurm_status_final.txt}`；各次训练曲线和fit_check在`runs/`。报告代码`experiments/Report_Task6_PrimitiveVAE_2_1.py`。下载只包含元数据/指标/曲线，不含模型；原始几何缓存与预测仍在Ibex，最终审计确认未生成checkpoint文件。配置原始字节另存config.frozen.json，Windows工作副本仅CRLF换行不同，解析配置与部署内容一致。


### 2026-09-10 — Verify_Task6_ReconstructionAudit_2.2 与 Task6修复3.1冻结

用户要求从第一性原理解决高误差，目标各流场测试RMSE/r<1；误差>3按工程失败处理。原2.1结果不改写；本次只读train/validation诊断。诊断代码commit `7ca9fe12`，作业`51716649_0..8`全部成功。

| 流场 | 缓存回放最大误差/r | 独立DOP853最大误差/r | 六频带符号逆变换validation | 十频带符号逆变换validation | 192维PCA validation |
|---|---:|---:|---:|---:|---:|
| cylinder3d | 5.3570506e-07 | 0.00032854434 | 0.100623 | 0.044704 | 0.002175 |
| halfcylinderRe640 | 4.3535301e-07 | 0.0001749829 | 0.081523 | 0.036313 | 0.005219 |
| halfcylinderRe6400 | 1.8056664e-06 | 0.0011078637 | 0.285038 | 0.128106 | 0.011890 |
| tangaroa | 2.4151741e-06 | 7.2413095e-05 | 0.096004 | 0.042357 | 0.003925 |
| deltaWing_resampled | 1.2132446e-07 | 1.0748568e-07 | 0.002106 | 0.000925 | 0.000038 |
| deltaWing_LBM | 1.2208135e-07 | 1.7254599e-06 | 0.002517 | 0.001105 | 0.000060 |
| f22raptor | 5.5074122e-06 | 0.0030142406 | 0.367646 | 0.158323 | 0.058191 |
| boeing747 | 1.7380997e-07 | 1.644266e-06 | 0.005328 | 0.002342 | 0.000120 |
| smokeBuoyancy | 2.566811e-08 | 1.6212128e-07 | 0.017988 | 0.007917 | 0.000508 |

以上每流场仅4个固定train样本用于独立积分，并非穷举全部轨迹；未发现能解释13量级误差的坐标轴、缓存匹配或积分错误。RMSE由独立公式复算一致。Re6400的静止初始簇/训练均值簇/随机训练簇validation误差分别105.460/59.340/83.517，作为明确定义且同单位的控制记录，不据此停止修复。

确切的重建接口问题：旧fmt_all针对分类保留Gram/模长与排序，完整位置重建需要的带符号方向/相位/粒子身份没有完整保留。测试构造两种等速不同方向的合法七线簇，旧token一致，signed复系数不同；全16频复系数回环达到浮点精度。它不证明所有真实样本都达到某个不可达下界，却证明旧接口不能普遍唯一解码。本次把信息保留问题作为实现修复，未改旧DFT_FMT_3D.py。

另一处应修复的是重建训练起点：原随机初始化VAE尚未达到train拟合的简单几何低秩表示能达到的精度。新方案先用train拟合线性潜空间/初始化，再训练真实残差VAE；不把初始化所得精度全部归于神经网络学习。诊断新代码的float64输入原地居中问题在单元测试中发现并修正为复制数组；旧诊断使用float32缓存会生成独立float64副本，其已提交结果不受此问题影响。

下一轮`mainExp_Task6_Reconstruction_3.1`训练前冻结：signed_fmt10(399维)→192潜变量VAE→672几何，与同结构Raw-VAE对照；旧161维FMT仍是2.1历史对照，不能给它记新编码成绩。每流场240000train+12000validation+12000test+12000未见尺度，新的固定数据种子；十频与192维仅按本次validation诊断选择（分别满足全流场逆变换<0.25、PCA<0.1）。32epoch、同曝光预算、正KL；只选正步数的validation最低模型，validation>=0.8不读test。另评估原2.1固定legacy_test，检查同一轨迹/同一单位的误差是否下降。完整冻结见`docs/Task6_reconstruction_protocol_3.1.md`和`config/mainExp_Task6_Reconstruction_3.1.json`。

本地11项测试通过，包括完整物理积分/缺失值处理、复系数回环、潜变量确实影响输出、解码器仅收潜变量、后验方差梯度、正训练步模型选择；约9.8秒。此时新神经网络真实流场测试尚未运行，目标<1尚不能宣称达到。

### 2026-09-10 — Task6重建3.1阶段结果：Re6400原高误差benchmark已低于1

这是运行中结果，54组模型仍在继续，尚未完成统一最终审计，不据此宣布九流场全部达标。编码是新399维signed_fmt10，不是原161维fmt_all。

Re6400首个种子93110：新test 0.023288174，未见尺度 0.024521196，原2.1固定legacy_test 0.025753942。旧2.1在该固定test的三种子均值13.747953；此处明确是旧三种子均值与新首种子的阶段比较，完整新三种子结果结束后另列。误差公式/单位/对应点与旧版不变。

真实训练证据：初始化validation 0.131825545 → 正训练步选择后的validation 0.022886942；选择step 15008，总更新15008，训练primitive曝光7680000。模型仅使用训练数据拟合初始化，再由validation选择；配置、潜变量及频率在读取新test前已冻结。

代码9d73a0736b907478d1dc7cc3ac75d764a9aeefce，config SHA256 eaf8c0cf6e902589cd5c68b8811601b593fe040e51d660c354e320b705cca6cb；实际job 51719635 / array 51717898_12，节点gpu211-18，设备Tesla V100-SXM2-32GB。证据文件`outputs/mainExp_Task6_Reconstruction_3.1/runs/halfcylinderRe6400/signed_fmt_vae/93110/result.json`；最终仍需对保存的预测独立复算。

### 2026-09-10 — mainExp_Task6_Reconstruction_3.1：54组训练与最终审计完成

证据：训练代码`9d73a0736b907478d1dc7cc3ac75d764a9aeefce`，配置`config/mainExp_Task6_Reconstruction_3.1.json`，实际配置SHA256 `eaf8c0cf6e902589cd5c68b8811601b593fe040e51d660c354e320b705cca6cb`。54/54训练完成，最终从预测独立复算2592条指标；本地再次核对CSV、逐次JSON、训练更新/曝光、正步数模型选择及validation门槛。

主指标保持原定义：七条对应路径线31个非初始时刻的三维位置均方根误差，除以初始邻居半径r。下表为三个优化种子的均值±样本标准差；时间、点身份及坐标不做目标辅助变换。目标按每个流场/种子的全测试集RMSE判断，不是每个点或每个primitive的最大误差。

- 新test：27个signed-FMT模型是否全部RMSE/r<1：**True**；最大单次模型指标0.0721537603。
- 原固定legacy_test：27个signed-FMT模型是否全部RMSE/r<1：**True**；最大单次模型指标0.067632901。
- 未见尺度test：27个signed-FMT模型是否全部RMSE/r<1：**True**；最大单次模型指标0.069712356。

同一历史测试轨迹的比较（该benchmark已使用过）：

| 流场 | 原FMT-VAE 2.1 | 新signed-FMT-VAE 3.1 | 同版Raw-VAE 3.1 |
|---|---:|---:|---:|
| cylinder3d | 5.19979 ± 0.0324 | 0.00452687 ± 1.42e-05 | 0.00273802 ± 2.33e-05 |
| halfcylinderRe640 | 3.69606 ± 0.0116 | 0.00605046 ± 3.86e-05 | 0.00442113 ± 3.39e-06 |
| halfcylinderRe6400 | 13.748 ± 0.316 | 0.0261244 ± 0.000755 | 0.0142735 ± 0.000234 |
| tangaroa | 4.41813 ± 0.0186 | 0.00507157 ± 8.42e-05 | 0.00360043 ± 5.98e-06 |
| deltaWing_resampled | 0.421725 ± 0.00886 | 0.000374961 ± 2.2e-05 | 0.000148297 ± 1.45e-05 |
| deltaWing_LBM | 0.467661 ± 0.0145 | 0.00045184 ± 9.11e-06 | 0.000184122 ± 9.08e-06 |
| f22raptor | 11.7469 ± 0.0515 | 0.0675079 ± 0.000215 | 0.055504 ± 2e-05 |
| boeing747 | 0.961566 ± 0.0289 | 0.000458923 ± 8.63e-06 | 0.000211391 ± 7.65e-06 |
| smokeBuoyancy | 1.64265 ± 0.0161 | 0.00137754 ± 3.25e-05 | 0.000670977 ± 2.18e-05 |

新采样的测试primitive及未见尺度：

| 流场 | signed-FMT test | Raw test | signed-FMT未见尺度 | Raw未见尺度 |
|---|---:|---:|---:|---:|
| cylinder3d | 0.0043304 ± 6.69e-05 | 0.00231377 ± 5.8e-06 | 0.004429 ± 1.92e-05 | 0.00237225 ± 5.09e-06 |
| halfcylinderRe640 | 0.00628821 ± 1.69e-05 | 0.004669 ± 1.09e-05 | 0.00593557 ± 9.65e-05 | 0.0044797 ± 9.12e-06 |
| halfcylinderRe6400 | 0.0229239 ± 0.000364 | 0.0118132 ± 4.17e-05 | 0.0248443 ± 0.000371 | 0.0144801 ± 0.000166 |
| tangaroa | 0.00488355 ± 5.43e-06 | 0.00342583 ± 1.58e-06 | 0.004893 ± 3.62e-05 | 0.00352048 ± 6.89e-06 |
| deltaWing_resampled | 0.000353496 ± 8.44e-06 | 0.000130816 ± 1.25e-05 | 0.000381467 ± 5.75e-06 | 0.000143467 ± 1.11e-05 |
| deltaWing_LBM | 0.000376601 ± 7.75e-06 | 0.000150601 ± 9.32e-06 | 0.00040543 ± 5.27e-06 | 0.000161794 ± 1.21e-05 |
| f22raptor | 0.0717226 ± 0.000379 | 0.0593226 ± 6.97e-05 | 0.0694663 ± 0.000307 | 0.0595102 ± 5.23e-05 |
| boeing747 | 0.000482003 ± 6.75e-06 | 0.000221249 ± 6.5e-06 | 0.000527819 ± 3.67e-06 | 0.000242842 ± 6.38e-06 |
| smokeBuoyancy | 0.0012937 ± 1.53e-05 | 0.000623275 ± 1.93e-05 | 0.00133098 ± 1.63e-05 | 0.000647317 ± 2.23e-05 |

初始化和神经网络训练分别列出（signed-FMT三个种子的均值）：

| 流场 | 初始化validation | 正训练步选中模型validation | 512样本拟合误差 |
|---|---:|---:|---:|
| cylinder3d | 0.0445882826 | 0.0040921865 | 0.0003295427 |
| halfcylinderRe640 | 0.0352491155 | 0.00639304297 | 0.00032631985 |
| halfcylinderRe6400 | 0.131825541 | 0.0229119099 | 0.000846431943 |
| tangaroa | 0.0405080184 | 0.00518883683 | 0.00051046768 |
| deltaWing_resampled | 0.00102766192 | 0.000346675624 | 6.46858046e-05 |
| deltaWing_LBM | 0.00101353877 | 0.000344274459 | 6.07999793e-05 |
| f22raptor | 0.16617709 | 0.0733472438 | 0.0010440551 |
| boeing747 | 0.00226383093 | 0.000436589157 | 9.07145358e-05 |
| smokeBuoyancy | 0.00788287882 | 0.00093034862 | 8.08954978e-05 |

本轮新test按流场三种子均值比较，Raw-VAE误差低于signed-FMT-VAE的流场数为9/9。九流场等权宏平均：signed-FMT 0.012517155，Raw 0.00918559423。这次回答完整几何能否被重建；该结果中的Raw对照必须共同保留。

修订说明：旧2.1使用原161维Gram/模长与邻居排序特征，且VAE从随机初始化开始；本版使用保留方向、相位、七线身份的399维signed_fmt10，192维潜变量及只在train拟合的线性几何初始化，再训练编码/解码残差网络和后验方差，每流场train扩大至240000。新成绩属于新编码，原fmt_all源码及2.1历史结果保持。多项设计同时改变，单项贡献需要独立消融；初始化已达到的精度单独报告。

所有几何输出经过192维潜变量，解码器只接收潜变量。每组512个train primitive拟合8000更新，然后全量训练重新初始化，32epoch=15008更新、768万训练样本曝光；beta=1e-5全程为正，validation选择step>0且误差<0.8后才读test。仅保留指标/预测，未写出模型checkpoint。

第一性原理排查的可核验证据见Verify_Task6_ReconstructionAudit_2.2：独立积分和缓存回放只检查每流场4个固定train样本，未发现能解释13量级误差的积分错误；原分类编码不能普遍唯一恢复带符号坐标的反例及完整复系数回环已通过测试。原问题是重建输入信息与训练设计需要修复，不能把分类表征的成绩直接当成坐标重建能力。

新test使用新primitive采样种子，源时间块沿用原协议；legacy_test为同一已用benchmark。训练、模型选择与评价角色保持区分。额外按真实几何总转角选择的弯曲图只作可视化核对，原固定样例和全量指标均保留。

完整证据：`outputs/mainExp_Task6_Reconstruction_3.1/{summary.json,per_flow_summary.csv,metrics.csv,final_audit.json,config.frozen.json,slurm_status_final.txt}`及`runs/`逐次结果；复算代码`experiments/Report_Task6_Reconstruction_3_1.py`，图形规则`docs/Task6_reconstruction_artifacts_3.1.md`。

### 2026-09-10 — Verify_Task6_ScarceGeneralization_4.1 预注册

用户要求尝试训练数据匮乏与dropout等泛化操作，并以测试优于Raw为目标。前一轮3.1新test九流场宏平均Raw 0.009185594、signed_fmt10 0.012517155，Raw在9/9流场更低；此事实和旧算法保持。本版按训练数量256/1024/4096比较，1024为预注册主设置；标准化与几何主成分初始化也仅使用小训练集。

候选为10频/完整16频signed编码各搭配无正则化、dropout+权重衰减、较强正则化、dropout+权重衰减+干净目标去噪，共8个FMT；Raw运行同样4种正则化。九流场共324个搜索模型只读train/validation，先按1024样本的九流场平均validation选择跨流场统一候选。测试前冻结选择，再使用三个新子集/优化种子；报告Raw原配方、相同正则化和单独validation最佳Raw。不能把16频可逆表示说成频谱压缩。完整配置和方法见docs/Task6_scarce_generalization_protocol_4.1.md。

本地9项行为测试通过，包括嵌套相同子集、train-only初始化、相同物理输入扰动且干净目标不变、dropout推断关闭、完整频率与Raw初始化几何一致、正步数validation选择和独立误差公式。尚无4.1真实流场结果，不预写胜利结论。

### 2026-09-11 — Verify_Task6_ScarceGeneralization_4.1 完整验证选择

324个候选全部完成。代码`bc874a1eb4508293b07f0989e07e170b4a102140`；选择文件SHA256 `81a6c189d87e06dba679a7e8ef60ac9517651fef0992b44b0f8a51f2cea05ebe`在读取test前固定。搜索种子94110，每个初始化和网络只使用指定训练子集。

| 候选 | 256样本validation | 1024样本validation（主设置） | 4096样本validation |
|---|---:|---:|---:|
| r0 | 0.0412080884 | 0.0259181997 | 0.0166857023 |
| r1 | 0.0375477651 | 0.0222161015 | 0.0151659693 |
| r2 | 0.0361545962 | 0.0219120476 | 0.014905681 |
| r3 | 0.0494783126 | 0.0265137379 | 0.0165063424 |
| c0 | 0.0758175508 | 0.0403489497 | 0.0246487029 |
| c1 | 0.0737309205 | 0.0364012549 | 0.0220553164 |
| c2 | 0.0734623769 | 0.0352283851 | 0.0209823485 |
| c3 | 0.0656665142 | 0.0358355825 | 0.0227508609 |
| c4 | 0.041517248 | 0.0258190057 | 0.0165538729 |
| c5 | 0.0377055296 | 0.0221986287 | 0.0152024963 |
| c6 | 0.0367572162 | 0.021829113 | 0.0149241805 |
| c7 | 0.0497797521 | 0.0264643183 | 0.0165669597 |

全局选择c6：16频、dropout=0.25、权重衰减0.01、不加输入噪声。独立选择的Raw为r2，与FMT正则化相同；两个Raw角色共享模型。

1024样本时，所选FMT的validation误差比原Raw低15.777%，比同正则化/独立选优Raw低0.378489%。但256/4096样本时，同一个FMT候选的validation分别比r2高1.66679%/0.124111%。这些是搜索种子的validation结果，不能充当测试证据。

10频候选总体仍落后Raw；增加噪声也没有在预注册主设置获胜。最终阶段固定c6/r0/r2，以三个新训练子集及随机种子重新拟合，独立测试。

### 2026-09-11 — Verify_Task6_ScarceGeneralization_4.1 最终测试结果

证据：训练代码`bc874a1eb4508293b07f0989e07e170b4a102140`、配置SHA256 `2b6006ac4f01441dfd44b7f70655f115b82061fdac5b891bb9fdd33eb6b069c8`、选择SHA256 `81a6c189d87e06dba679a7e8ef60ac9517651fef0992b44b0f8a51f2cea05ebe`；独立复算9396条比较/尺度指标通过，实际唯一模型数243。

此前3.1在24万训练primitive下Raw九流场均优于signed_fmt10；本版改变为有限训练数据及预注册正则化，并搜索10频与完整16频，结论需按本版条件报告，旧结果不改写。小训练集包含全部标准化/初始化/网络训练输入。测试仍为3.1已用benchmark，不是新的确认集。

选择只使用1024样本的九流场平均validation；同一候选用于全部流场与三个样本量。实际选择：

- fmt: `{"id": "c6", "frequencies": 16, "regularizer_index": 2, "dropout": 0.25, "weight_decay": 0.01, "noise_sigma": 0.0}`
- raw_frozen: `{"id": "r0", "frequencies": 16, "regularizer_index": 0, "dropout": 0.0, "weight_decay": 0.0, "noise_sigma": 0.0}`
- raw_matched: `{"id": "r2", "frequencies": 16, "regularizer_index": 2, "dropout": 0.25, "weight_decay": 0.01, "noise_sigma": 0.0}`
- raw_selected: `{"id": "r2", "frequencies": 16, "regularizer_index": 2, "dropout": 0.25, "weight_decay": 0.01, "noise_sigma": 0.0}`

每流场三个新训练子集/优化种子的均值，再等权平均九流场。RMSE/r定义保持不变。

| 训练数量 | 测试集合 | FMT | Raw原配方 | Raw相同正则化 | Raw验证最优 | FMT优于原Raw流场数 |
|---|---|---:|---:|---:|---:|---:|
| 256 | test | 0.0366882337 | 0.0417294205 | 0.0368026819 | 0.0368026819 | 9/9 |
| 256 | unseen_scale | 0.0400943554 | 0.0451047116 | 0.041203906 | 0.041203906 | 9/9 |
| 1024 | test | 0.0219111617 | 0.025636275 | 0.0219784636 | 0.0219784636 | 9/9 |
| 1024 | unseen_scale | 0.0235727652 | 0.0276490588 | 0.0236643384 | 0.0236643384 | 9/9 |
| 4096 | test | 0.0149392634 | 0.0165421824 | 0.0149292777 | 0.0149292777 | 9/9 |
| 4096 | unseen_scale | 0.0160410095 | 0.0179059286 | 0.0160333276 | 0.0160333276 | 9/9 |

主比较n=1024：相对raw_frozen的误差降低率14.5306%，获胜流场9/9；负值表示FMT更差。三个配对子集种子的宏平均误差差值(FMT−Raw)为[-0.0038002451929068253, -0.0037504588610085265, -0.003624635787551892]。

主比较n=1024：相对raw_matched的误差降低率0.306218%，获胜流场5/9；负值表示FMT更差。三个配对子集种子的宏平均误差差值(FMT−Raw)为[-6.316984628649556e-05, -1.1920351846546945e-05, -0.0001268155661201726]。

主比较n=1024：相对raw_selected的误差降低率0.306218%，获胜流场5/9；负值表示FMT更差。三个配对子集种子的宏平均误差差值(FMT−Raw)为[-6.316984628649556e-05, -1.1920351846546945e-05, -0.0001268155661201726]。

结论边界：相对原Raw的总改善包含正则化收益；FMT本身的额外收益应看相同正则化Raw及独立选择Raw，不能把总改善全部归给傅里叶编码。三个随机种子的方向不等于统计显著性，也不能替代逐流场结果。

主设置逐流场（均值±样本标准差）：

| 流场 | FMT | Raw原配方 | Raw相同正则化 | Raw验证最优 |
|---|---:|---:|---:|---:|
| cylinder3d | 0.0080257063 ± 0.00044 | 0.010839533 ± 0.000683 | 0.0081586646 ± 0.000493 | 0.0081586646 ± 0.000493 |
| halfcylinderRe640 | 0.010531104 ± 0.000707 | 0.013363223 ± 0.00102 | 0.010534083 ± 0.000719 | 0.010534083 ± 0.000719 |
| halfcylinderRe6400 | 0.030079816 ± 0.000383 | 0.038931447 ± 0.00156 | 0.030134958 ± 0.000306 | 0.030134958 ± 0.000306 |
| tangaroa | 0.0093439145 ± 0.000511 | 0.0099254699 ± 0.000424 | 0.0097959704 ± 0.000581 | 0.0097959704 ± 0.000581 |
| deltaWing_resampled | 0.00055317534 ± 4.73e-05 | 0.00071243655 ± 2.06e-05 | 0.00050605045 ± 5.64e-05 | 0.00050605045 ± 5.64e-05 |
| deltaWing_LBM | 0.0004650722 ± 8.11e-05 | 0.00057161838 ± 6.79e-05 | 0.00043204496 ± 3.78e-05 | 0.00043204496 ± 3.78e-05 |
| f22raptor | 0.13651353 ± 0.00759 | 0.15434574 ± 0.0075 | 0.13664898 ± 0.00728 | 0.13664898 ± 0.00728 |
| boeing747 | 0.00052722475 ± 3.45e-05 | 0.00062450695 ± 3.29e-05 | 0.00048492676 ± 7.38e-05 | 0.00048492676 ± 7.38e-05 |
| smokeBuoyancy | 0.001160915 ± 0.000116 | 0.0014125026 ± 7.99e-05 | 0.0011104908 ± 8.25e-05 | 0.0011104908 ± 8.25e-05 |

主设置的初始化与训练后validation（九流场/三子集种子等权均值；不用于追加测试选择）：

| 方法角色 | 初始化validation | 训练后validation | 训练集误差 | 训练后优于初始化次数 |
|---|---:|---:|---:|---:|
| fmt | 0.0209298738 | 0.0215470104 | 0.000696264697 | 2/27 |
| raw_frozen | 0.0209298716 | 0.0250035914 | 0.000737448644 | 0/27 |
| raw_matched | 0.0209298716 | 0.0216667923 | 0.000685427725 | 2/27 |
| raw_selected | 0.0209298716 | 0.0216667923 | 0.000685427725 | 2/27 |

初始化诊断必须保留：如果训练后validation仍高于初始化，说明该神经网络训练尚未改善线性初始化的泛化；不能把Raw/FMT间的排名改善等同于网络已优于线性几何表示。这里没有根据初始化诊断重新选择测试模型。

所选16频编码保留全部651个独立系数，相对31间隔信号不做频谱压缩；最终192维VAE潜变量才是压缩。结果不能归因于丢弃高频，也不属于旧161维fmt_all。

不同Raw角色若候选相同，使用同一模型预测，不能当成独立证据。完整训练曲线、选择记录与逐次指标保存在`outputs/Verify_Task6_ScarceGeneralization_4.1/`；不保留checkpoint。

### 2026-09-11 — Verify_Task6_DirectNeural_5.1：纠正逆变换解释与预注册

用户追问：是否只是先傅里叶变换再逆变换才有效，能否说明tokenizer能力；随后授权实现下一版并部署Ibex。

| 此前表述 | 当前准确解释 | 代码证据与修订原因 |
|---|---|---|
| 3.1/4.1使用PCA初始化，残差网络继续训练 | 解析还原与PCA组成的固定矩阵在每次推理仍然存在，不是仅初始化后丢弃；完整16频下固定通路满足P(F^{-1}(F(x)))=P(x) | 科学commit `bc874a1eb4508293b07f0989e07e170b4a102140`，`FMT_Utils/Task6Recovery_3D.py`、`FMT_Utils/Task6Scarce_3D.py`；先前若以“初始化”让人理解为后续不依赖逆变换，则该解释不完整 |
| 4.1的重建效果支持FMT作为tokenizer | 它支持含固定几何PCA通路的整个系统可重建；不能独立证明FMT提供了学习几何token的优势 | 4.1 primary test FMT .0219111617、同正则Raw .0219784636，增益仅0.306%；主要改善来自正则化。原指标及结论边界保留，不把它改写成纯神经编码实验 |

新版本检验 `geometry → frozen signed_fmt16 → learned encoder → 192 latent → learned decoder → geometry`。不调用逆傅里叶或累加还原，不使用PCA/SVD构造网络权重；无输入到输出绕过潜变量的通路。只有训练集逐通道均值/尺度、全局输出标量尺度与固定初始种子可作缓冲区。Raw/FMT输入均651维，参数数量和种子初始权重相同。独立PCA192只作比较。signed_fmt16仍含中心时间差分，未宣称客观性；本版不是旧161维fmt_all。

| 版本 | 技术与训练 | 主代码 | 结果状态 |
|---|---|---|---|
| Verify_Task6_DirectNeural_5.1 | 宽512、编码/解码各3残差块、192潜变量；训练20,000步；4个学习率/正则候选；train-only统计量；先32样本过拟合门槛 | `FMT_Utils/Task6DirectNeural_3D.py`, `FMT_Utils/Task6PCABaseline_3D.py`, `experiments/Task6_DirectNeural_5_1.py`, `config/Verify_Task6_DirectNeural_5.1.json` | 预注册；尚无真实测试指标，不能宣称胜出 |

沿用3.1/4.1九流场、256/1024/4096嵌套训练集，1024主设置，搜索94110，最终94111/94112/94113。72个搜索模型仅使用validation；统一FMT配置，主Raw采用相同配置，另保留独立validation最佳Raw；选择JSON冻结后才最终评估。测试仍是已用benchmark，不是新confirmation。数据时间、积分长度及物理RMSE/r均不变。变更包含取消固定通路、数值标准化及训练预算，不能视为单因素消融。

本地7项测试通过：完整前向FMT系数一致、Raw/FMT初始可训练权重相同、统计量仅来自train、清零网络消除全部输入相关重建、禁用逆傅里叶/PCA函数仍能实际训练、独立PCA基线、合成小数据完整部署链。完整链覆盖配置选择冻结、Raw重复配置只拟合一次、禁止提前读test，以及篡改选择或预测时审计拒绝。合成拟合误差1.146439→.010191只是程序测试，不是流场实验性能。真实小样本GPU拟合检查及正式结果以Ibex登记/输出为准。

真实训练数据32样本拟合检查（不是test性能）：

| 流场 | 输入 | 未训练RMSE/r | 拟合RMSE/r | 误差下降比例 | 潜变量清零/正常误差 | 通过 |
|---|---|---:|---:|---:|---:|---|
| cylinder3d | signed_fmt16_neural | 11.71856281 | 0.01060726 | 99.909% | 1104.92 | True |
| cylinder3d | raw_neural | 11.71856281 | 0.01666272 | 99.858% | 703.30 | True |
| halfcylinderRe640 | signed_fmt16_neural | 11.86779772 | 0.00774584 | 99.935% | 1532.41 | True |
| halfcylinderRe640 | raw_neural | 11.86779772 | 0.00991118 | 99.916% | 1197.43 | True |
| halfcylinderRe6400 | signed_fmt16_neural | 38.84255944 | 0.02146553 | 99.945% | 1809.81 | True |
| halfcylinderRe6400 | raw_neural | 38.84255944 | 0.02984566 | 99.923% | 1301.52 | True |
| tangaroa | signed_fmt16_neural | 15.03150517 | 0.00749647 | 99.950% | 2005.74 | True |
| tangaroa | raw_neural | 15.03150517 | 0.00755313 | 99.950% | 1990.12 | True |
| deltaWing_resampled | signed_fmt16_neural | 3.53901189 | 0.00018418 | 99.995% | 19215.64 | True |
| deltaWing_resampled | raw_neural | 3.53901189 | 0.00147974 | 99.958% | 2391.65 | True |
| deltaWing_LBM | signed_fmt16_neural | 3.95821481 | 0.00015564 | 99.996% | 25433.90 | True |
| deltaWing_LBM | raw_neural | 3.95821481 | 0.00121934 | 99.969% | 3246.20 | True |
| f22raptor | signed_fmt16_neural | 58.83550697 | 0.00454862 | 99.992% | 12936.75 | True |
| f22raptor | raw_neural | 58.83550697 | 0.05080833 | 99.914% | 1158.05 | True |
| boeing747 | signed_fmt16_neural | 2.98054267 | 0.00029286 | 99.990% | 10180.60 | True |
| boeing747 | raw_neural | 2.98054267 | 0.00317809 | 99.893% | 937.84 | True |
| smokeBuoyancy | signed_fmt16_neural | 1.47963361 | 0.00433093 | 99.707% | 341.75 | True |
| smokeBuoyancy | raw_neural | 1.47963361 | 0.00209452 | 99.858% | 706.69 | True |

来源：科学commit `1496fd7c31a6dae3d71fd25228dfba6bafa3b289`，作业51724746_0..8，outputs/Verify_Task6_DirectNeural_5.1/fit_check/*/result.json。该检查只证明随机初始化网络可以学习这些训练几何，不证明未见primitive上的泛化或优于Raw。

### 2026-09-11 — Verify_Task6_DirectNeural_5.1 最终失败与独立审计

**结论：5.1实验未通过工程性能要求，未证明FMT优于Raw。记录与数据核查通过，不能等同于方法有效。**

科学commit `1496fd7c31a6dae3d71fd25228dfba6bafa3b289`；原config SHA256 `6b3c62a72994b7d870259895ee3b5d74c65e80f0ce7b5fd2044df19fe3ec9227`；选择SHA256 `97654b01dd5e2208bb8eebd6205dc7fb22f9c4a1c203878d9ad74d99a761ce64`。
原作业51724744–51724751及结束记录51724775全部结束。最终训练243个神经网络（81 FMT、81同配置Raw、81独立选择Raw），另拟合81个独立PCA。37个FMT模型因validation RMSE/r>=3未生成test预测；其余44个FMT模型及全部Raw/PCA共287组模型产生预测，累计8323行尺度/比较记录。

FMT由验证集选择c2（学习率1e-4、dropout .25、权重衰减 .01）；同配置Raw使用c2；独立选择Raw使用c1（3e-4、.10、.0001）。选择仅由九流场搜索种子94110的1024样本validation等权平均决定，选择后才启动最终三个种子。

| 训练样本数 | FMT验证RMSE/r | 同配置Raw验证RMSE/r | 独立选择Raw验证RMSE/r | FMT失败/总数 | Raw失败/总数（每组） |
|---:|---:|---:|---:|---:|---:|
| 256 | 15.903919 | 0.540609 | 0.554899 | 17/27 | 0/27 |
| 1024 | 4.801351 | 0.341982 | 0.337596 | 12/27 | 0/27 |
| 4096 | 2.243565 | 0.234245 | 0.216030 | 8/27 | 0/27 |

每档包含九流场×三个种子，全体模型均计入验证均值；误差是七线31个非初始时刻对应点的三维位置均方根误差除以初始邻居半径。81次FMT验证均差于各自同配置Raw。

| 训练样本数 | FMT总体test均值 | 同配置Raw test | 独立选择Raw test | 独立PCA192 test | 实际有FMT测试的配对比较 |
|---:|---|---:|---:|---:|---|
| 256 | 不报告：失败模型缺失 | 0.551276 | 0.565075 | 0.029783 | FMT胜出0/10 |
| 1024 | 不报告：失败模型缺失 | 0.351586 | 0.347227 | 0.021294 | FMT胜出0/15 |
| 4096 | 不报告：失败模型缺失 | 0.243961 | 0.223633 | 0.014894 | FMT胜出0/19 |

44个实际完成test的FMT模型在位置RMSE/r上全部差于相同样本、种子、正则配置的Raw；unseen_scale同样0/44。次指标同时间点间距误差有7/44个test配对、5/44个unseen_scale配对优于Raw，这些局部胜出不构成预注册主指标达标。这个配对计数仅限实际评估模型，不把失败的37个模型删除后重算总体FMT平均。PCA为独立主成分分析线性压缩/重建，与神经网络无依赖。完整对照与未见尺度数值见`outputs/Verify_Task6_DirectAudit_1.1/complete_validation_and_controls.csv`。

| 之前已报告的事实 | 现在的完整解释 | 修订依据 |
|---|---|---|
| 18项真实32样本拟合检查通过，FMT训练误差0.000156–0.02147 | 该事实仍成立，只证明记住小训练集；不能用它证明正式训练/泛化问题已经解决 | 主1024设置FMT训练均值0.110127、validation均值4.801351；12/27未通过门槛 |
| 5.1删除固定逆变换、使用全部可训练的编码/解码 | 独立检查确认确实删除，但去掉后本轮性能失败 | 神经模块没有逆FFT/PCA/固定恢复矩阵；所有输入相关映射通过192维潜变量 |
| 3.1/4.1的重建成绩很好 | 原指标保持，但不能转记到纯神经5.1；完整16频仍不是旧161维fmt_all | 5.1另含标准化及训练预算变更，不是只删除一个分支的单因素消融 |

**独立审计核查范围与结果。** 新审计`Verify_Task6_DirectAudit_1.1`、commit `f89adae13cbd8b9ac0a98adcd0c9f69cf63f64f0`、CPU作业51736122_0..8全部成功。审计代码不调用原geometry_metrics，重新从全部预测计算位置、中心、邻居、端点、同时间点间距及逐时刻误差；最大数值差`1.0613732115416497e-13`。全部原始/子集缓存哈希一致，四个随机子集与原始train几何、ID逐值一致；train/validation/test完整源时间片互不交叉；test与unseen_scale允许共享测试时间段，二者primitive ID不同。Cylinder播种范围仍为[7.5,12]。未发现单位更换、测试用于配置选择或失败模型伪造测试结果。两臂参数均4,267,990，随机种子相同时初始可训练权重相同；20,000更新、5,120,000样本曝光量一致；选中步与正训练步的最低validation记录一致。

**编码没有在这轮丢失几何信息。** 仅在独立审计中，用原float32完整16频系数解析还原训练/验证几何，九流场validation相对RMSE为`2.3205e-8–2.6881e-8`，最大绝对RMSE/r为`2.74384e-6`。因此本轮失败不是完整频率编码不可逆或缺失位置/相位造成。审计中的逆变换未接入任何训练或测试推理，不能把该数值当作神经网络成绩。

**训练设计与流程的限制及排查方向。**

1. 当前送入网络的两种标准化表示，其方差分布差异明显。搜索训练集1024样本中，达到99%输入方差所需维数：Raw仅3–8，FMT为35–154；FMT标准化后的validation最大绝对分量117–426，而Raw为14–36。下表是描述性诊断；我认为标准化后小幅度频率通道及离群值的放大值得优先排查，但这些统计还不能单独证明因果。

| 流场 | Raw 99%方差维数 | FMT 99%方差维数 | Raw验证最大标准化分量 | FMT验证最大标准化分量 |
|---|---:|---:|---:|---:|
| cylinder3d | 6 | 115 | 19.036 | 339.462 |
| halfcylinderRe640 | 8 | 154 | 14.299 | 117.436 |
| halfcylinderRe6400 | 6 | 141 | 17.295 | 426.146 |
| tangaroa | 5 | 44 | 24.170 | 124.238 |
| deltaWing_resampled | 3 | 38 | 25.279 | 138.499 |
| deltaWing_LBM | 3 | 35 | 23.755 | 124.422 |
| f22raptor | 8 | 112 | 23.991 | 232.667 |
| boeing747 | 3 | 77 | 19.615 | 195.769 |
| smokeBuoyancy | 7 | 38 | 35.763 | 140.843 |

2. `FMT_Utils/Task6DirectNeural_3D.py`的encoder_projection和decoder_projection是绕过Dropout的可训练线性分支；dropout .25只用于非线性分支。因此该参数不能解释为整个编码通路受25%Dropout。两臂结构仍公平一致；它是否导致FMT泛化失败，需要下一版分别消融，不能由本审计直接归因。
3. 搜索阶段最优FMT的validation宏平均已为4.322062，`select()`仍然无条件启动完整最终训练。当前门槛只在每个最终模型训练后阻止test读取。未来应增加最终批次启动前的验证门槛；单靠32样本记忆检查不足。此次如实保留失败，不更改冻结5.1协议。
4. VAE后验的对角方差是全局可训练参数，初值logvar=-18，并不由每个输入单独预测；KL权重1e-5。本版确实包含变分目标，但没有证明其生成采样能力，不能把重建任务结果扩张到这类结论。

**边界：** 本次未重训模型，训练/验证误差核对的是保存日志和选择逻辑；由于按协议不保留checkpoint，不能对原最优权重重新运行validation。测试和未见尺度预测已逐项独立复算。未重新积分原始流场，物理积分正确性仍引用3.1既有验证。本审计没有证明单一失败原因，也没有修改5.1算法或开始下一版训练。

审计输出：`outputs/Verify_Task6_DirectAudit_1.1/scalar_audit.json`、`per_model_validation.csv`、`complete_validation_and_controls.csv`以及九流场独立JSON。5.1应作为失败版本保留，不能用于论文中声称FMT已优于Raw或已证明其tokenizer优势。

### 2026-09-11 — Verify_Task6_PNNTrans_1.1 实施与预注册

用户要求实现 `pnn_trans`：每个时间步用固定 Point-NN 空间算子编码七点，再经注意力 Transformer 压缩为小 latent，最后由全连接网络重建完整 primitive。已阅读原论文及固定源码 `a85bfc365258a2c65a5f6ac289537b4d7a7cec0f`，适配与协议见 `docs/Task6_pnn_trans_protocol_1.1.md`。此版不修改或解释性重命名4.1/5.1历史结果，也不预先将5.1失败归因于傅里叶变换的信息丢失。

| 版本/方法 | 技术细节 | 主要代码 | 当前证据 |
|---|---|---|---|
| Verify_Task6_PNNTrans_1.1 / pnn_trans | 每帧7个48维位置编码+96维邻域聚合，保留身份得到1008维；3层128宽Transformer；192维latent；512宽全连接解码器 | FMT_Utils/Task6PNNTrans_3D.py | 本地4项算子/梯度/拟合测试和1项完整数据/选择/评价流程测试通过；尚无真实流场性能结论 |
| Verify_Task6_PNNTrans_1.1 / raw_trans | 每帧21个坐标；相同Transformer、latent、解码器和训练配置，输入投影维数不同 | 同上 | 相同测试通过；这是新结构配对对照，不能替换冻结Raw-VAE基线 |

固定前端没有可训练参数、没有跨批次统计、没有时间差分/时间离散傅里叶变换。单时刻邻域标准化不抹去全局漂移，因为原始位置编码仍进入latent。仅192维latent进入解码器，无解析逆变换、PCA或原始几何跳接。本版为确定性自编码器，未使用VAE后验/KL。

沿用九流场、三种训练规模、原冻结数据及Cylinder时间政策；先真实训练集拟合，随后1024样本的36模型validation搜索。只有选定候选两臂在九流场全部validation<3，才提交162模型最终阶段；否则保留全部失败验证结果且不访问test。实际运行与结果待补，不保证优于既有FMT/Raw/PCA。

### 2026-09-11 — Verify_Task6_PNNTrans_1.1 Ibex部署

科学commit `21108d729e36a5d4186c673d361d9a913171ea0e`，配置SHA256 `8c91aa9abfec86f493ccb38c4033437d090001d799652eb7d49ea07113223fe2`。Ibex独立目录`/ibex/user/zhanx0o/FMT_Task6PNNTrans_20260911`，作业51737881–51737886。已提交CPU预检、九流场数据核验、18模型小集拟合及36模型验证搜索，之后冻结选择；最终162模型只在预注册validation门槛通过后自动提交。

参数审计：pnn_trans 3,523,403个可训练参数，raw_trans 3,397,067个；空间Point-NN前端0参数。两臂下游Transformer/decoder配置相同，差126,336个输入投影参数（相对Raw多约3.72%）。相对旧5.1还改变了架构、归一化、确定性目标及优化预算，不能把跨版本变化当作单因素消融。

Ibex预检51737881已于2026-09-11 12:48:24 +03结束，cn604-18，exit 0；五项测试全部通过，见本地`outputs/Verify_Task6_PNNTrans_1.1/logs/preflight_51737881_4294967294.err`。初次数据核验已通过三个Cylinder流场，其余阶段继续运行/等待依赖；目前尚无真实流场的测试性能结论。

### 2026-09-11 — Verify_Task6_PNNTrans_1.1 验证完成，未通过测试启动门槛

结论：当前pnn_trans 1.1未通过预注册门槛。九流场选中配置下的验证误差全部高于原始坐标+相同Transformer/解码器对照；因此本轮不支持固定Point-NN前端改善Task6重建泛化。训练作业没有崩溃，后续测试启动器按协议拒绝提交，不把该拒绝误报为硬件故障。

科学commit `21108d729e36a5d4186c673d361d9a913171ea0e`，配置SHA256 `8c91aa9abfec86f493ccb38c4033437d090001d799652eb7d49ea07113223fe2`，冻结选择SHA256 `ac2938acce4c363ba42d38914292d8163e3caa3563b11e65da90647f1230c0df`。每流场1024个训练primitive、12000个validation，种子94110；每模型12000更新、batch128（共1,536,000样本曝光）。九流场18个32样本拟合模型均通过；正式36个搜索模型全部完成。所有GPU任务实际设备为Tesla V100-SXM2-32GB。

九流场等权validation均值选择c1：学习率0.0003、Dropout 0.1、权重衰减0.001；Raw使用同一配置。c0/c1的pnn_trans验证均值分别为1.222370852/1.077592128。以下均为各自最低validation正训练步的结果，**不是最后一步误差，也不是test误差**。误差为完整7条路径线、31个非初始采样点的三维位置均方根误差除以初始邻居半径（RMSE/r）。

| 流场 | pnn_trans训练误差 | 原始坐标对照训练误差 | pnn_trans验证误差 | 原始坐标对照验证误差 |
|---|---:|---:|---:|---:|
| Cylinder Re160 | 0.190434 | 0.193009 | 0.735556 | 0.446428 |
| Half-cylinder Re640 | 0.192045 | 0.188742 | 0.794155 | 0.492713 |
| Half-cylinder Re6400 | 0.707811 | 0.805397 | 2.374840 | 1.614463 |
| Tangaroa | 0.204444 | 0.233860 | 1.329173 | 0.662348 |
| DeltaWing resampled | 0.028446 | 0.025456 | 0.092278 | 0.078844 |
| DeltaWing LBM | 0.039431 | 0.034015 | 0.085203 | 0.068532 |
| F22 | 0.661581 | 0.762105 | 3.162694 | 1.962421 |
| Boeing747 | 0.046506 | 0.054238 | 0.150338 | 0.098527 |
| Smoke buoyancy | 0.209150 | 0.037218 | 0.974092 | 0.199941 |
| 九流场等权均值 | 0.253317 | 0.259338 | 1.077592 | 0.624913 |

pnn_trans验证均值比配对Raw高72.44%，胜场0/9。训练均值相近且pnn_trans略低，但validation更差；这表明本配置有明显泛化问题，不能仅凭32样本拟合或训练误差宣称tokenizer有效。尚未确定是空间位置编码、邻域聚合、归一化或优化中的哪一步造成，不据此直接否定Point-NN方法或归因于傅里叶信息丢失。

F22验证误差3.162693753≥3，触发门槛。选择作业51737885于13:05:23 +03正常完成；启动器51737886于13:07:31 +03以exit 1结束，原因明确为`Validation gate failed: final training/testing forbidden`。最终162模型批次没有提交，256/4096样本与三个最终种子的测试尚未执行，本轮没有test或unseen_scale指标。目前无本实验运行中作业。

核验：`experiments/Report_Task6_PNNTrans_1_1.py`逐一比对36份fit/result/selection、正训练步最低validation、样本数、优化步数、配置/选择哈希和6条提交记录，全部一致。仅是保存记录的一致性核对，没有重放模型权重或重新积分。完整标量见`outputs/Verify_Task6_PNNTrans_1.1/validation_results.csv`和`validation_summary.json`；旧4.1/5.1结果保持不变。

### 2026-09-11 — Verify_Task6_PNNTrans_1.2 扩大网络与正则化搜索

用户要求增加网络大小、层数、Transformer深度、Dropout及训练正则化，以超过原始坐标对照。新协议`docs/Task6_pnn_trans_protocol_1.2.md`保留1.1空间Point-NN算子和192维压缩；旧1.1结果保持冻结。代码`FMT_Utils/Task6PNNTuning_3D.py`，流程`experiments/Task6_PNNTuning_1_2.py`。

| 版本/方法 | 技术细节 | 对照与选择 | 当前证据 |
|---|---|---|---|
| Verify_Task6_PNNTrans_1.2 / pnn_trans | 16候选，Transformer 6–12层、宽128–512；解码器宽512–1024、4–6残差块；Dropout .1–.4、权重衰减.001–.03；部分候选有训练集特征尺度处理、额外全连接、去噪/几何混合增强 | 三流场48模型筛选，前4候选在九流场训练PNN与配对Raw共72模型；只用validation选择 | 新增4项算子/正则化/训练测试与1项完整两阶段、三臂流程测试通过；尚无真实流场新结果 |
| Verify_Task6_PNNTrans_1.2 / raw_trans | 与PNN相同的入选网络和正则化；每时刻输入原始21坐标 | 同时保留冻结1.1 Raw c1，不能用调弱Raw替换baseline | 最终需分别报告胜负 |
| 冻结1.1 / raw_frozen | 原128宽、3层Transformer和原12000更新训练代码不改 | 搜索参照已冻结validation=0.624913022，最终重训对应子集/种子 | 该数值不是test |

新PNN和配对Raw完整训练均24000更新，batch128；统计量只从相应训练子集拟合，去噪仅训练启用。混合增强为训练几何凸组合后重新执行Point-NN，不混合三角函数特征，不声称新样本来自真实流场积分。更大网络同时改变多个设计，不把搜索结果直接视为单因素消融。

仅在选中候选全部九流场两臂validation<3，且PNN宏平均同时优于配对Raw及冻结Raw时自动提交最终243模型。最终test才能回答超过baseline与否；如果未达到，则完整保留失败，不能改标签、测试集、单位或筛掉困难流场。模型checkpoint不落盘。

### 2026-09-11 — Verify_Task6_PNNTrans_1.2 部署记录

科学commit `4a206bc1d02f74e006643ac20b85425f846e7673`，配置SHA256 `ef8ea7cda2498d7bebcfaec42997fb14e4f4f1a0728db0d5f3dbbd089c6e78c1`。Ibex独立目录`/ibex/user/zhanx0o/FMT_Task6PNNTransTuning_20260911`。2026-09-11 14:11:23 +03提交51740510–51740517：预检、九流场数据准备、最大网络容量核验、48模型筛选、选前4候选、72模型九流场配对验证、冻结选择及有条件提交最终测试。此次只是部署完成，尚无新性能结论；目标不能当成已达到的结果。

Ibex预检51740510于14:12:16 +03正常完成，cn604-17，九项测试全部通过；数据准备51740511于14:14:08 +03完成，九流场与冻结Raw基线哈希核对成功。结构计数：新PNN候选5,710,667–55,230,283个参数；最大c14的配对Raw为54,724,939参数，空间前端仍0参数。此时GPU容量检查与正式搜索等待调度，尚无新验证性能数字。

### 2026-09-11 — Verify_Task6_PNNTrans_1.2 最大网络GPU检查通过

51740512_0/1均于14:16:05 +03开始，分别14:16:21/18结束，exit 0。最大c14为12层、512宽Transformer，PNN 55,230,283参数，Raw 54,724,939参数；均在Tesla V100-SXM2-32GB上完成batch128反向传播。PNN峰值显存2.572 GiB，Raw 2.515 GiB。只运行8步用于有限值/显存检查，不是拟合或泛化性能结果；正式候选仍按4000/24000步协议执行。

### 2026-09-11 — Verify_Task6_PNNTrans_1.2 验证完成，仍未超过两种Raw对照

**结论：本轮选中pnn_trans c11，九流场validation均值0.991092193，比相同网络/训练预算的Raw高29.49%，比冻结1.1 Raw高58.60%，两种比较均0/9胜场。** 相比旧pnn_trans 1.1均值1.077592128，新版本下降8.03%；这只是跨版本开发集改善，不是超过基线，也不是test结果。

48个初筛模型及72个完整配对模型均完成。九流场1024个训练primitive、12000个validation和搜索种子94110保持不变；新版本每模型24000更新、batch128，冻结Raw保持原12000更新。按预注册九流场PNN等权均值选唯一c11，不逐流场拼接最优架构或改变单位。全部数字为正训练步中最低validation对应模型的三维位置均方根误差/初始邻居半径（RMSE/r）；覆盖7条路径线、31个非初始时刻。

| 流场 | 旧pnn_trans 1.1 | 新pnn_trans 1.2 c11 | 同结构Raw 1.2 c11 | 冻结Raw 1.1 c1 |
|---|---:|---:|---:|---:|
| cylinder3d | 0.735556 | 0.763287 | 0.546707 | 0.446428 |
| halfcylinderRe640 | 0.794155 | 0.703574 | 0.573745 | 0.492713 |
| halfcylinderRe6400 | 2.374840 | 2.707400 | 1.949308 | 1.614463 |
| tangaroa | 1.329173 | 1.175693 | 0.819787 | 0.662348 |
| deltaWing_resampled | 0.092278 | 0.124551 | 0.103410 | 0.078844 |
| deltaWing_LBM | 0.085203 | 0.106366 | 0.079524 | 0.068532 |
| f22raptor | 3.162694 | 2.808281 | 2.444317 | 1.962421 |
| boeing747 | 0.150338 | 0.145666 | 0.129180 | 0.098527 |
| smokeBuoyancy | 0.974092 | 0.385010 | 0.242585 | 0.199941 |
| 九流场等权均值 | 1.077592 | 0.991092 | 0.765396 | 0.624913 |

所有进入完整比较的候选均保留：

| 配置 | PNN验证均值 | 配对Raw验证均值 | 验证误差≥3的模型数 |
|---|---:|---:|---:|
| c12 | 1.042314 | 0.924343 | 0/18 |
| c11 | 0.991092 | 0.765396 | 0/18 |
| c08 | 1.513818 | 1.366945 | 4/18 |
| c15 | 1.316168 | 1.410384 | 4/18 |

选中c11：Transformer宽256、6层、8头，192维latent；解码器宽768、4个残差块；Dropout 0.2、权重衰减0.01、学习率0.0003；训练集特征尺度处理、额外逐帧全连接、特征噪声0.01、50%训练几何混合。PNN 12,017,483参数、配对Raw 11,764,811参数；固定Point-NN前端仍0参数。原空间算子不变，代码`FMT_Utils/Task6PNNTuning_3D.py`。两臂训练误差均值分别0.394703/0.357332，并非全部困难场景已经充分拟合。

与1.1并列解释：Smoke从0.974092降到0.385010、F22从3.162694降到2.808281；但Re160从0.735556升到0.763287、Re6400从2.374840升到2.707400。新配对Raw也从冻结Raw均值0.624913升到0.765396。因此本轮不能把“更大网络+更强正则化”说成全面改善；多个因素同时改变，不能单凭本结果确定因果，也不能据此声称固定空间算子必然丢失几何信息。

选择作业51740516于16:05:24 +03正常结束。选中两臂全部九流场validation<3，但PNN未超过两种Raw，故`numeric_gate_passed=true`、`beats_both_baselines=false`、`gate_passed=false`。启动器51740517于16:07:31 +03以exit 1结束，日志明确为`Validation gate failed: final training/testing forbidden`；这是预注册性能门槛拒绝提交，不是训练崩溃。最终243模型没有提交，256/4096训练规模、三个最终种子的test/unseen_scale没有运行；截至16:34 +03本轮无排队/运行任务。

科学commit `4a206bc1d02f74e006643ac20b85425f846e7673`，配置SHA256 `ef8ea7cda2498d7bebcfaec42997fb14e4f4f1a0728db0d5f3dbbd089c6e78c1`，冻结选择SHA256 `301b56e8e3b3f0fa635a2185ce834f29058ba23db3d4129443d92bdd4fd57a07`。Ibex目录`/ibex/user/zhanx0o/FMT_Task6PNNTransTuning_20260911`。`experiments/Report_Task6_PNNTuning_1_2.py`对120个模型的fit/result、最低validation选择步、更新数、样本曝光量、配置/选择哈希和九流场均值完成一致性复核；这是保存记录核对，没有重放权重或重新积分。只同步JSON/CSV/文本日志，没有保存/下载checkpoint。

标量证据：`outputs/Verify_Task6_PNNTrans_1.2/checked_training_results.csv`（完整120模型）、`selected_validation_comparison.csv`、`selection.json`、`status_summary.json`。本版作为未超过基线的结果保留，原1.1及4.1/5.1指标不改写。

### 2026-09-11 — Verify_Task6_PNNTrans_1.3 三倍数据与学习率调度预注册

用户要求训练集扩大约三倍并增加学习率/调度器实验。新协议`docs/Task6_pnn_trans_protocol_1.3.md`：每流场1024→3072个真实primitive，保留旧训练子集前缀及固定validation/test；九流场分别训练。固定1.2最佳c11网络和正则化，搜索学习率0.0001/0.0003/0.001×原余弦、OneCycleLR、ReduceLROnPlateau，共9候选。

| 版本/方法 | 技术及代码 | 对照及当前证据 |
|---|---|---|
| Verify_Task6_PNNTrans_1.3 / pnn_trans | 6层256宽Transformer、192维latent不变；新学习率调度`FMT_Utils/Task6PNNSchedule_3D.py`，三倍数据与预算 | 筛选12000更新、完整72000更新；本地14项算子/调度/拟合/流程测试通过，无真实新性能结论 |
| Verify_Task6_PNNTrans_1.3 / raw_trans | 相同候选、结构、数据及72000更新，仅输入为原始坐标 | 用于表示比较，不替换冻结Raw |
| Verify_Task6_PNNTrans_1.3 / raw_frozen | 原1.1 c1网络、原代码及12000更新，使用相同3072项重新训练 | 不沿用1024数据上的0.624913充当同数据基线 |

9个冻结Raw重训＋27个三流场筛选＋72个九流场配对模型，共108个开发模型；保留旧学习率/余弦配置l01进入完整比较。按PNN九流场验证均值选统一候选，通过每流场<3且胜过两种Raw后，才提交九流场×三种子×三臂的81个最终模型，训练规模只用3072。原结果不改写。

14项本地测试验证了原余弦逐步学习率及同种子权重/指标与1.2逐值一致、单周期的起点/峰值/终点与不改变Adam动量、平台调度忽略第0步并按停滞降学习率、三种调度实际训练、三倍嵌套索引、重新训练Raw及测试门槛和最终三臂流程。测试中平台调度器内部Infinity哨兵导致严格JSON保存失败，已仅对该内部状态以字符串记录；真实损失仍要求有限。最佳权重仅RAM保存，未下载checkpoint。

### 2026-09-11 — Verify_Task6_PNNTrans_1.3 Ibex部署

科学commit `9b33826c91ece524d3ed36b9f42e288a8658eb3c`，配置SHA256 `67ac3e6397689f71e4cef9360a3629c2ea365871c0fc5063296e9f658c46703a`。独立目录`/ibex/user/zhanx0o/FMT_Task6PNNDataSchedule_20260911`。2026-09-11 18:07:00–01 +03提交预检51743949、数据核验51743950、九流场Raw重训51743953、27模型学习率筛选51743954、保留参考组的四候选选择51743955、72模型配对验证51743956、冻结选择51743957、条件启动最终测试51743958。每GPU作业独占一张分配到的GPU，最多18个作业并行；不代表一个模型使用18张GPU。最终81模型尚未提交，需验证门槛通过。

Ibex预检51743949已于2026-09-11 18:08:22–18:09:15 +03在cn113-35-l运行，14项测试全部通过，exit 0。代码同时经过本地CPU和Ibex PyTorch 2.6.0+cu118预检；这只是实现/流程核验，尚无新真实流场性能结论。日志已同步到`outputs/Verify_Task6_PNNTrans_1.3/logs`。

### 2026-09-11 — Verify_Task6_PNNTrans_1.3 验证完成：误差改善，仍未超过Raw

**结果：选中l00（学习率0.0001、原余弦衰减），PNN九流场平均validation误差0.706685554；同结构Raw为0.566086842，重新在3072项上训练的冻结Raw为0.460250170。** PNN分别高24.84%/53.54%，两种对照均0/9胜场。该版本不支持PNN前端优于原始坐标的Task6重建泛化，最终81模型未启动。

108个开发模型全部正常训练完成：9个冻结Raw、27个筛选PNN、72个完整配对模型。数据准备证明3072为旧1024的三倍嵌套扩展；每流场3072训练、12000validation，种子94110。PNN和同结构Raw每模型72000更新、batch128，冻结Raw保留原代码/配置12000更新。结构仍为6层256宽Transformer、192维latent，Dropout0.2、权重衰减0.01、噪声0.01及50%几何混合保持不变。

下表均为各模型正训练步最低validation对应的三维位置RMSE/r，覆盖7条线31个非初始时刻。**不是test结果**。选择全局同一l00，不逐流场拼接最佳候选。

| 流场 | 旧1.2 PNN（1024训练） | 新1.3 PNN（3072） | 同结构Raw（3072） | 冻结Raw重训（3072） |
|---|---:|---:|---:|---:|
| cylinder3d | 0.763287 | 0.535571 | 0.436589 | 0.326318 |
| halfcylinderRe640 | 0.703574 | 0.543901 | 0.460640 | 0.346515 |
| halfcylinderRe6400 | 2.707400 | 1.906598 | 1.396810 | 1.176699 |
| tangaroa | 1.175693 | 0.789103 | 0.593337 | 0.479529 |
| deltaWing_resampled | 0.124551 | 0.060855 | 0.057342 | 0.050503 |
| deltaWing_LBM | 0.106366 | 0.057744 | 0.054778 | 0.047616 |
| f22raptor | 2.808281 | 2.180147 | 1.880296 | 1.520521 |
| boeing747 | 0.145666 | 0.094467 | 0.081694 | 0.080025 |
| smokeBuoyancy | 0.385010 | 0.191785 | 0.133297 | 0.114525 |
| 九流场等权均值 | 0.991092 | 0.706686 | 0.566087 | 0.460250 |

全部四个完整比较候选：

| 配置 | PNN验证均值 | 同结构Raw验证均值 |
|---|---:|---:|
| l03 | 0.729966 | 0.575843 |
| l00 | 0.706686 | 0.566087 |
| l04 | 0.914783 | 0.596245 |
| l01 | 0.893650 | 0.590091 |

l00/l01为学习率0.0001/0.0003的原余弦；l03/l04为对应峰值学习率的单周期调度。平台调度和0.001学习率候选在三流场12000更新筛选中未晋级；不能将短筛选结果推广为这些调度器在其他预算或数据上必然较差。全部27个筛选模型和72个配对模型保留于标量表。

与上一轮并列解释：旧PNN=0.991092，新PNN=0.706686，降低28.70%；旧冻结Raw=0.624913，新同算法Raw=0.460250，也改善。数据、PNN训练预算、学习率及验证频率共同变化，不能将28.70%全部归因于数据量。上一轮与本轮均未超过两种Raw，科学结论没有反转。新PNN/配对Raw训练误差均值分别0.439437/0.347653；Re6400和F22的PNN训练误差仍分别1.425889/1.168355，不能宣称所有场景已充分拟合。

Ibex实际时间（+03）：Raw 18:12:37–18:17:05；筛选18:19:05–18:35:30；配对训练18:38:28–20:52:59；选择51743957于20:54:20完成。71个GPU作业使用Tesla V100-SXM2-32GB，1个使用NVIDIA A100-SXM4-80GB（51743956_21）；最多18个单GPU作业并行。所有训练进程成功，选中三臂每流场validation均<3，但PNN均值未低于两种Raw，`gate_passed=false`。启动器51743958于20:56:25以exit 1结束，明确报`Validation gate failed: final training/testing forbidden`，是协议性能门槛拒绝提交，不是训练崩溃。截至23:37 +03无本轮排队/运行作业，三种子的test/unseen_scale没有运行。

科学commit `9b33826c91ece524d3ed36b9f42e288a8658eb3c`；配置SHA256 `67ac3e6397689f71e4cef9360a3629c2ea365871c0fc5063296e9f658c46703a`；冻结选择SHA256 `545e07fca5cb0bfa331d2c95984d6dfbd8512f129a845a9a5ee65aeaaee21039`。Ibex目录`/ibex/user/zhanx0o/FMT_Task6PNNDataSchedule_20260911`。`experiments/Report_Task6_PNNDataSchedule_1_3.py`复核108个模型的候选配置、训练样本数、更新数、样本曝光、学习率日志、最低验证步、配置/选择哈希、全局选择及门槛，均一致；此为保存记录核对，未重放权重/重新积分。只下载JSON/CSV/文本，不下载模型或轨迹数组。

证据：`outputs/Verify_Task6_PNNTrans_1.3/checked_training_results.csv`、`selected_validation_comparison.csv`、`selection.json`、`status_summary.json`及`runtime_events.jsonl`。旧1.1/1.2及其他任务历史结果保持不变。

### 2026-09-12 — Verify_Task6_FMTGeometryMoE_1.1 原 FMT 与几何双分支预注册

用户提出频域分支 A（原 FMT）与几何分支 B 的混合专家重建。协议见 `docs/Task6_fmt_geometry_moe_protocol_1.1.md`；每流场使用冻结 3072 样本，九流场各自训练，输入七条完整路径线并重建它们。保持 192 维 token，仅 token 输入共享全连接解码器。

| 版本/方法 | 技术与代码 | 当前性能证据 |
|---|---|---|
| Verify_Task6_FMTGeometryMoE_1.1 / fmt_moe + raw_mlp | 原 161 维 fmt_all → 可训练全连接专家 A；原几何 → 全连接专家 B；scalar/channel 门控；`FMT_Utils/Task6FMTGeometryMoE_3D.py` | 本地原配方一致性、容量/路由检查、实际小样本拟合和完整微型流程五项测试通过；无真实新测试结果 |
| Verify_Task6_FMTGeometryMoE_1.1 / fmt_moe + pointnn | B 换为逐帧冻结 Point-NN + 6 层 256 宽 Transformer；其余相同 | 同上，性能待 Ibex 验证 |
| Verify_Task6_FMTGeometryMoE_1.1 / geometry_moe | A 换成原几何全连接专家；与对应 FMT MoE 参数量、数据、预算相同 | 控制增加专家容量的作用 |
| Verify_Task6_FMTGeometryMoE_1.1 / geometry_only | B + 相同解码器，保留相同 B/解码器初始参数 | 判断融合是否优于单几何分支 |
| Verify_Task6_FMTGeometryMoE_1.1 / raw_frozen | 冻结 1.1 c1 Raw Transformer，原代码及 12000 更新 | 开发复用经配置/索引哈希核验的 1.3 同 3072 基线；最终三种子按冻结代码重训 |

首先 18 个训练集过拟合检查，之后 24 个初筛模型、108 个完整开发模型。两种几何 B 各用验证选择一套跨九流场的统一候选；若两个家族均通过数值门槛，最终 189 个模型。学习率 0.0001/0.0003；dropout 0.1、weight decay 0.001；完整新方法均 72000 更新。门控依赖输入，初始 A/B=0.1/0.9，允许自行舍弃无益专家；早期共享解码辅助损失逐步归零。记录强制关 A、关 B 及错配 A 的验证误差作为依赖性诊断。

本次与 1.3 并列的协议修订：旧版要求验证误差超过两种 Raw 才进入测试；新版仅要求所选家族各流场各方法验证误差有限且 <3，再诚实比较测试胜负。超过 Raw 是结果，不是测试准入条件。全连接 B 是新候选；此前最强 Raw 是 Transformer，不将它误称为已验证的纯全连接网络。

原 FMT 的具体冻结调用包含六频率截断、范数/夹角/手性与邻居排序，不可逆；不能把完整有符号 DFT 也说成丢失信息。本版本无逆 DFT、无 PCA、无原坐标到输出的旁路。五项本地测试包含破坏预测文件后审计必失败；最终位置误差独立复算，缺方法/种子/流场不出完整平均值。无 checkpoint 文件。此条仅记录实现与预注册，不证明 MoE 优于基线。

### 2026-09-12 — Verify_Task6_FMTGeometryMoE_1.1 Ibex 首次部署

实现已提交并推送：`4859c0caec1ac519f5932ed8772a8c775e71f688`。Ibex 独立工作目录已检出该 commit，Bash 语法检查通过，8 个阶段作业于 2026-09-12 01:40:34（+03:00）登记提交：51758242–51758249。本地五项测试通过；截至首次提交仅为排队，尚无真实 MoE 性能结果。配置 SHA256 为 `2ad8965a5240f59ef8543f44c53a7d41d5f55e265117c1a49c059a400e76249c`。完整协议见 `docs/Task6_fmt_geometry_moe_protocol_1.1.md`；作业详情见 `docs/ibex_run_registry.md`。

### 2026-09-12 — Verify_Task6_FMTGeometryMoE_1.1 集群预检通过

Ibex 独立 checkout（科学 commit `4859c0caec1ac519f5932ed8772a8c775e71f688`）上五项测试全部通过，26.646 秒；作业 `51758242`、节点 `cn604-18`、CPU。作业运行脚本起止时间为 01:42:02.584436–01:42:32.289479（+03:00），exit=0。证据：远端 `outputs/Verify_Task6_FMTGeometryMoE_1.1/logs/preflight_51758242_4294967294.err` 与 `runtime_events.jsonl`。截至此次核验，数据准备作业 `51758243` 等待优先级调度，GPU 训练尚未开始，不能把预检拟合结果当作真实流场性能。

### 2026-09-12 — Verify_Task6_FMTGeometryMoE_1.1 最终性能：全连接几何有效，FMT 混合未带来净收益

**最终测试已完成，无失败或缺项。** 每流场 3072 训练样本，九流场 × 三种子，每种方法 27 次最终训练。总计 339 个模型：18 个拟合检查、24 个初筛、108 个完整开发模型、189 个最终模型。每种几何 B 按冻结验证集选择一套全局配置：全连接 B=m01，Point-NN B=m05；两者均为标量门控、学习率 0.0003。其他超参数遵循预注册 1.1。

科学代码 `4859c0caec1ac519f5932ed8772a8c775e71f688`；配置 SHA256 `2ad8965a5240f59ef8543f44c53a7d41d5f55e265117c1a49c059a400e76249c`；测试前冻结选择 SHA256 `9505190c47cc772127d6dc8bed842e11192454680009334d81757f34a28f1475`。位置均方根误差除以初始邻居半径 r，越低越好；不计固定初始点，三维坐标平方求和不再除以 3。下表均值为九流场与三种子的等权平均，± 为三个种子各自九流场平均值之间的样本标准差，不是置信区间。

| 方法 | 普通测试误差 | 未见尺度测试误差 |
|---|---:|---:|
| 冻结 Raw Transformer | 0.482664 ± 0.009935 | 0.454378 ± 0.009738 |
| 仅全连接几何分支 | 0.211436 ± 0.002634 | 0.220112 ± 0.005294 |
| 两个全连接几何专家混合 | 0.214204 ± 0.002419 | 0.222143 ± 0.004684 |
| 原 FMT + 全连接几何混合 | 0.220306 ± 0.003531 | 0.229040 ± 0.007615 |
| 仅 Point-NN + Transformer | 0.629852 ± 0.013902 | 0.599075 ± 0.005959 |
| 全连接几何 + Point-NN 混合 | 0.289105 ± 0.007559 | 0.300016 ± 0.013396 |
| 原 FMT + Point-NN 混合 | 0.737732 ± 0.022269 | 0.698124 ± 0.025270 |

**结果解释与旧结论的区别。** 上一轮 1.3 中最强的是冻结 Raw Transformer；本版本增加的纯全连接网络取得更低重建误差。旧版网络、配置与结果保持不变，不能把旧版“最强 Raw”解读为它将永远优于新结构。新方法训练 72000 次，冻结 Raw 12000 次，且网络结构不同；相对旧 Raw 的优势不能单独归因于 FMT。

原 FMT + 全连接混合相对冻结 Raw 的测试误差降低 54.36%，九流场均降低，但比相同预算的仅全连接几何分支高 4.19%，比相同参数量的双几何专家高 2.85%。与仅全连接相比，普通测试仅 Tangaroa 小幅更好（0.228845 vs 0.229146），其余八个流场更差；未见尺度九流场全部更差。这一版本没有证明原 FMT 为重建带来净收益。

原 FMT + Point-NN 混合相对仅 Point-NN 的测试误差高 17.13%，九流场全部更差；相同参数量的全连接几何 + Point-NN 混合为 0.289105，明显更低。不能将“增加专家”得到的性能变化混同为 FMT 提供了有效附加信息，也不能把此结果推广为所有频域方法或所有 MoE 结构均无效。

逐流场普通测试误差（三种子平均；方法顺序固定）：

| 流场 | 冻结 Raw | 仅全连接 | 双全连接 | FMT+全连接 | 仅 Point-NN | 全连接+Point-NN | FMT+Point-NN |
|---|---:|---:|---:|---:|---:|---:|---:|
| cylinder3d | 0.371721 | 0.177559 | 0.180777 | 0.190027 | 0.442958 | 0.226111 | 0.459671 |
| halfcylinderRe640 | 0.346945 | 0.150217 | 0.152559 | 0.153696 | 0.447183 | 0.241188 | 0.465053 |
| halfcylinderRe6400 | 1.295991 | 0.491087 | 0.498953 | 0.534751 | 1.549371 | 0.722615 | 1.718208 |
| tangaroa | 0.437533 | 0.229146 | 0.230225 | 0.228845 | 0.782787 | 0.248339 | 1.150076 |
| deltaWing_resampled | 0.046471 | 0.029121 | 0.029336 | 0.030514 | 0.062406 | 0.030702 | 0.105618 |
| deltaWing_LBM | 0.052281 | 0.032815 | 0.032316 | 0.034228 | 0.057565 | 0.033333 | 0.127099 |
| f22raptor | 1.571051 | 0.696013 | 0.706091 | 0.708861 | 2.012095 | 0.993476 | 2.195751 |
| boeing747 | 0.088993 | 0.026788 | 0.026812 | 0.028175 | 0.076713 | 0.033558 | 0.136536 |
| smokeBuoyancy | 0.132986 | 0.070179 | 0.070767 | 0.073653 | 0.237589 | 0.072620 | 0.281576 |

逐流场未见尺度测试误差（三种子平均）：

| 流场 | 冻结 Raw | 仅全连接 | 双全连接 | FMT+全连接 | 仅 Point-NN | 全连接+Point-NN | FMT+Point-NN |
|---|---:|---:|---:|---:|---:|---:|---:|
| cylinder3d | 0.349583 | 0.181761 | 0.185109 | 0.192744 | 0.429294 | 0.222249 | 0.462575 |
| halfcylinderRe640 | 0.366850 | 0.177855 | 0.180901 | 0.181085 | 0.467085 | 0.284375 | 0.484040 |
| halfcylinderRe6400 | 1.261847 | 0.554728 | 0.558300 | 0.598394 | 1.484224 | 0.771426 | 1.665502 |
| tangaroa | 0.408291 | 0.218596 | 0.219908 | 0.219954 | 0.667860 | 0.240927 | 0.973988 |
| deltaWing_resampled | 0.044793 | 0.031426 | 0.031568 | 0.031496 | 0.073787 | 0.031685 | 0.125463 |
| deltaWing_LBM | 0.050443 | 0.033908 | 0.034599 | 0.036471 | 0.086886 | 0.057689 | 0.141354 |
| f22raptor | 1.402637 | 0.680382 | 0.685815 | 0.694133 | 1.888401 | 0.981200 | 2.038990 |
| boeing747 | 0.076177 | 0.025850 | 0.025833 | 0.026924 | 0.072267 | 0.030452 | 0.131383 |
| smokeBuoyancy | 0.128782 | 0.076499 | 0.077257 | 0.080161 | 0.221867 | 0.080146 | 0.259819 |

门控与分支依赖诊断（**最终模型的验证集结果，不是测试结果**；27 模型等权平均）：

| 几何 B | FMT 平均门控权重 | 正常融合误差 | 强制仅 B | 强制仅 A | 错配 A |
|---|---:|---:|---:|---:|---:|
| raw_mlp | 0.724% | 0.216303 | 0.227656 | 7.973678 | 0.237575 |
| pointnn | 36.568% | 0.711752 | 8.960206 | 6.297019 | 1.429051 |

全连接版本给 FMT 的平均门控权重约 0.724%；该权重不能换算成信息贡献百分比，因为两分支隐向量的尺度不同。关闭/错配 A 会让已经联合训练的网络变差，说明存在依赖；然而独立从头训练的仅几何对照更好，所以依赖不等于泛化收益。强制单专家是对融合网络的干预，不能把其误差当作单专家独立训练的性能。Point-NN 的门控权重较大，同样未转化为优于几何对照的性能。

复核：Ibex 已从实际预测数组独立复算位置误差；本地再从逐次 CSV 检查七种方法每种 27 个 dataset/seed 组合与尺度完整性，复算所有宏平均值，与 final_audit.json 一致；冻结配置/选择哈希一致，全部开发与最终拟合使用 3072 项（训练集小样本拟合检查为 32 项），189 次最终拟合齐全，无失败，未保存或下载 checkpoint。结果仅支持本版本数据、结构与训练预算下的结论。

证据位置：`outputs/Verify_Task6_FMTGeometryMoE_1.1/{metrics.csv,final_audit.json,selection.json,status_summary.json,analysis_verified.json}`；复算代码 `experiments/Summarize_Task6_FMTGeometryMoE_1_1.py`，原训练代码 `FMT_Utils/Task6FMTGeometryMoE_3D.py`，协议 `docs/Task6_fmt_geometry_moe_protocol_1.1.md`。完整标量指标和日志已下载；轨迹数组保留 Ibex 审计，模型无落盘。

### 2026-09-12 — Verify_Task36_MultiGate_1.1 联合分类与重建的任务专用门控

用户提出 Task3＋Task6 联合训练，以分类目标和重建目标训练不同任务的分支选择。本次已实现 `Verify_Task36_MultiGate_1.1`，协议 `docs/Task36_multigate_protocol_1.1.md`。共享原 FMT 专家 A 与全连接几何专家 B，分类和重建各有独立门控/输出头，端到端同时训练。原 FMT 161 维、每专家 192 维；两个任务各用自己的 192 维混合 token，同时服务两任务所需专家对合计 384 维，不能误称为统一 192 维存储。

| 方法版本 | 具体技术、代码与对照 | 当前性能证据 |
|---|---|---|
| Verify_Task36_MultiGate_1.1 / joint_task_gates | `FMT_Utils/Task36MultiGate_3D.py`；共享双专家、分类/重建独立门控，两个归一化损失相加 | 六项本地测试通过；未取得真实新性能 |
| 同版 / joint_shared_gate | 两个任务共用同一门控，专家/输出头相同 | 判断任务专用路由是否必要 |
| 同版 / joint_raw_task_gates | A 改为原几何输入；与主方法参数量及初始化相同 | 判断是否为 FMT 收益 |
| 同版 / joint_fixed_routes | 分类固定用 FMT A，重建固定用几何 B | 与预设分工比较 |
| 同版 / joint_geometry_only | 只有几何 B，共同训练两任务 | 与纯几何多任务表示比较 |
| 同版 / single_task3、single_task6 | 共享结构分别只优化分类或重建，均重新训练 | 判断联合学习的收益和负迁移 |

九流场各使用相同冻结 3072 训练 primitive；根据原中心种子、原时刻，从原加载空间整场 IVD p95 补算分类标签，保持原划分。`FMT_Utils/Task36Labels_3D.py` 复用旧 IVD 标量与插值算子；比较使用 float64 的 p95 阈值，缓存的值/阈值可直接逐项核验。标签、IVD、物理位置和时间均不作为网络输入。分类包含可变尺度，因此未见尺度分类是 Task5 式扩展；不直接覆盖旧固定尺度 Task3 主表。整个联合编码器获得分类监督，不再称为纯自监督 Task6。

训练固定原验证选择的全连接结构、学习率 0.0003、dropout 0.1、weight decay 0.001、72000 更新。分类采用按训练类别比例加权的交叉熵/log(2)，重建采用三维平方误差/训练集平均几何方差，相加；去除上一版每个专家都承担重建的辅助项，允许自然任务分工。门控都从 0.5/0.5 开始，不在主方法中强制分工。联合 checkpoint 按同一个验证损失和选取，分类阈值在该 checkpoint 的 validation 上选 F1；两个测试指标分别报告。

六项测试检查独立门控、容量匹配/固定路由、损失归一化、整场 IVD 阈值、真实联合拟合和完整七方法标签/训练/评估流程，包括破坏预测后审计拒绝。首轮单元测试的解析场端点恰与整场 p95 相同，导致用“查询点 p95 必不相同”检验标签域的断言不成立；改为查询仅中心点来检验整场阈值不依赖查询集合，生产算法未因此改变。后续六项测试全部通过。

开发 63 模型、最终 189 模型，全部方法保留；先开发，再冻结 selection 并生成 test 标签，之后最终训练/独立审计。任务相关门控是可检验机制：需要在 AP–重建误差两维对照单任务、共享门控、双几何和固定分工，不能只凭门控权重不同就宣布有效。历史 MoE 1.1 结果不改写。无 checkpoint 落盘。

### 2026-09-12 — Verify_Task36_MultiGate_1.1 Ibex 联合任务部署

代码已提交并推送 `c3f95c9b9d5d89750fd7ff38f7ff9467c7aa1885`；Ibex 独立 checkout Bash 语法检查通过。2026-09-12 12:50:42–43（+03:00）提交五阶段：预检 51809306、九流场标签 51809307、63 开发模型 51809310、冻结验证 51809311、自动后续提交 51809312。本地六项测试通过，预检已在 cn605-27-l 开始；尚无真实联合训练性能。完整协议见 `docs/Task36_multigate_protocol_1.1.md`。

### 2026-09-12 — Verify_Task36_MultiGate_1.1 预检通过与标签准备记录

截至 2026-09-12T12:57:04.537901+03:00，独立 Ibex checkout 的六项测试通过；标签准备 5/9 完成，GPU 训练等待依赖，未报告模型优劣。已完成流场的开发集/验证集正类样本数如下；这些是全场 p95 标签的抽样计数，不是分类性能，未重采样以强制正类恰好占 5%。

| 流场 | 开发训练正类 / 总数 | 验证正类 / 总数 |
|---|---|---|
| cylinder3d | 193 / 3072 | 701 / 12000 |
| halfcylinderRe640 | 157 / 3072 | 658 / 12000 |
| halfcylinderRe6400 | 158 / 3072 | 635 / 12000 |
| deltaWing_resampled | 122 / 3072 | 566 / 12000 |
| smokeBuoyancy | 154 / 3072 | 625 / 12000 |
