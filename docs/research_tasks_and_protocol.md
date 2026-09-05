# FMT 研究任务与统一协议

本文件是 Task1–Task5 的唯一任务定义。旧文档若与本文件冲突，以本文件为准。协议自 2026-08-23 起生效；Task5 自 2026-08-26 起加入；历史实验 ID 和输出目录不追溯改名。

## 1. 总体研究命题

研究对象是 pathline cross primitive。2D primitive 通常为中心线和 `x±、y±` 共 5 条线；3D primitive 为中心线和 `x±、y±、z±` 共 7 条线。

FMT 是由 Fourier 变换、`sin/cos`、几何不变量和 aggregation 构成的 **training-free encoder**。这里“training-free”只描述 encoder 本身没有通过标签或重构损失更新的参数；KMeans、VAE 和监督分类器仍然需要训练拟合。
当前研究包含 **3D Task1、Task2、Task3、Task5**，以及仅限 3D 的 **Task4-b proxy-label 实验**。Task4-b 1.1 因跨 split primitive 空间重叠被审计否决；channel 内标签与空间拆分协议冻结为 1.2。`mainExp_Task4B_ChannelToTBL_2.3` 只研究一个 channel source volume 到一个 TBL target volume 的跨 volume 迁移，不把 Task4-b 扩展解释为一般跨流场结论。Task1–Task3 和 Task5 在 2D、3D 都有定义，但 2D 扩展暂不进入当前实验计划；Task4 只在 3D 中成立。

## 2. 五项任务的固定定义

| 任务 | 维度 | 输入与方法 | 核心比较 | 主要输出 | 允许的核心结论 |
|---|---|---|---|---|---|
| **Task1：无监督涡区域聚类** | 2D、3D | `primitive -> FMT -> feature -> KMeans(k=2)` | Raw geometry + KMeans、FMT + KMeans，以及必要的无参数几何 baseline | ARI、NMI、固定映射后的 F1/IoU；逐流场与 family macro | FMT feature 是否足以区分涡区域和非涡区域；是否优于 Raw 聚类 |
| **Task2：FMT 作为 VAE 输入** | 2D、3D | 无监督 VAE 编码后，对 latent feature 做 KMeans 二类聚类 | **主比较：Raw+VAE vs FMT+VAE**；FMT direct 只作诊断 | held-out ARI、NMI、F1/IoU；多 VAE seed 分布 | FMT 是否是比 Raw pathline 更好的 VAE 输入 |
| **Task3：有监督 IVD 涡识别** | 2D、3D | IVD 标签监督的涡/非涡二分类网络 | Raw、参数量控制 Raw、Raw+FMT | F1、Average Precision、AUROC、precision、recall；多训练 seed | 加入 FMT 是否提高有监督涡区域识别 |
| **Task4：有监督涡类型分类** | **仅 3D** | 对已定义的 3D 涡型标签做多分类 | 不使用 FMT vs 加入 FMT | macro-F1、每类 F1、balanced accuracy、confusion matrix | 加入 FMT 是否提高 streamwise、spanwise、hairpin 等涡型分类 |
| **Task5：不同尺度几何学习** | 2D、3D | 每个 primitive 的邻居距离、积分步长和积分步数可变；积分后统一重采样为固定 `K×L×C`，再做 IVD 监督二分类 | 固定尺度 Task3 迁移、variable-scale Raw、结构匹配 Raw-PCA residual、variable-scale Raw+FMT | unseen-scale confirmation 的 F1、Average Precision；逐尺度、逐流场及 family macro | 模型能否学习跨尺度 primitive；FMT 是否提高 variable-scale IVD 涡识别 |

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

- 只允许 3D 数据；必须报告每类样本数和 class-balanced 指标。
- 涡区域定位误差与涡型分类误差应分开统计。
- Task4 开始前必须单独建立标签来源、类别定义和跨数据集名称映射文档。
- Channel Task4-b 1.2 的标签定义、阈值选择、空间/实例拆分和物理质检冻结在 `docs/Task4B_channel_proxy_label_protocol_1.2.md`。其中 `norm(omega-mean_xy(omega)(z))` 必须称为 channel-profile vorticity deviation，不得称为标准 IVD，也不得据此改写 Task1/Task2/Task3/Task5 的 whole-field IVD p95 协议。
- `VortexIds>0` 只提供人工 hairpin support 与实例编号。ordinary streamwise/spanwise orientation 是 velocity–curl 45°规则产生的 proxy；hairpin head/limb 是 velocity–curl、`omega_y_prime` 及分量占优规则产生的 proxy。四类标签都不得表述为人工 anatomical ground truth；位置序和左右涡量符号只作逐实例物理质检。
- `mainExp_Task4B_ChannelToTBL_2.3` 的 TBL 测试总体是 **oracle-enriched population**：先按冻结的 per-volume p80.5 规则产生 vortex candidates，再显式加入全部人工 `VortexIds>0` hairpin support。该实验衡量已知人工 support/identity 条件下的四类 proxy 迁移，不衡量 blind whole-field hairpin detection。
- Channel→TBL 2.3 使用全部 channel source rows 训练，并以连续三 epoch source 零错误和正 minimum true-logit margin 作为进入 TBL 一次评估的门禁；它回答“完全拟合 source 后能否迁移”，不是常规 source holdout 泛化。必须报告 streamline coverage，并把无有效 streamline 的人工 hairpin 行及完整 VortexId 计入 coverage-adjusted 失败指标。即使 Ibex 独立审计 `PASS`，证据仍只来自一个 source volume 和一个 target volume。

### Task5

- Task5 是 Task3 的尺度扩展，监督目标仍是冻结的 IVD 涡/非涡二分类；不得同时改变标签定义后把差异归因于尺度学习。
- “尺度”至少完整记录三项：中心到邻居种子的距离、数值积分步长、积分步数；总积分时间由后两者的乘积给出。最终送入网络的线数 `K` 和每线采样数 `L` 必须固定。当前 3D 主协议使用 7 条线，2D 使用 5 条线。
- 空间尺度、时间步长和积分步数的组合必须在 train、validation、confirmation 间按 tuple 拆分。主 confirmation 至少包含训练中未出现的组合，并明确区分尺度插值与尺度外推。
- 尺度 tuple 的分配必须与空间 seed 和 IVD 标签独立，且每个时间片中各 tuple 数量近似均衡；不得把大尺度主要分给涡区、小尺度主要分给背景。
- 主表至少包含 variable-scale Raw、同结构同维度 Raw-PCA residual、variable-scale Raw+FMT；同时报告固定尺度 Task3 模型直接迁移到 variable-scale confirmation 的结果。
- 除总体 F1 和 Average Precision 外，必须输出逐尺度 tuple 的指标，避免总体平均掩盖某一尺度范围的系统失败。
