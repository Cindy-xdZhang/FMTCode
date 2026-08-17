# FMT 实验记录（唯一结论载体）

规则（与全局研究规范一致）：
1. 版本命名：主实验 `mainExp_x.y`；组件验证 `Verify_[xxx]_x.y`；非主线探索 `Other_[xxx]_x.y`；消融 `Ablation_[xxx]_x.y`。x = 大迭代，y = 小迭代。
2. 每行必须可追溯：技术细节 + 主要代码路径 + config + commit + 指标。没有数字的结论不进表。
3. 修订旧结论必须新旧并列写明变更原因，禁止静默翻转。
4. Baseline 冻结：VortexTransformer（已发表）冻结在 PyflowVis 仓库，不复制不修改；本仓库的 legacy 聚类管线（mainExp_1.1 待跑）一经记录即冻结。

## 版本表

| 版本 | 日期 | 任务 | 技术要点 | 主要代码路径 | config | 指标（协议见下） | 结论 |
|---|---|---|---|---|---|---|---|
| （待跑）mainExp_1.1 | — | Task1 | 旧法冻结复跑：FMT(PosE+同时刻跨线分组+LGA+Pool, stages=2, embed24, alpha1000, beta19, temporal_head=None) + KMeans(k=2)，4 输入视图；补定量协议与确定性设置。注：分组已从 GeoLinePicker 重构为 `group_same_timestep`，经测试与旧实现逐位一致（commit ca653fd），故仍视为"旧法冻结"。复跑前已修复两处数据侧 bug（commit 944d206）：时间重采样改为先过滤无效 primitive；积分后端 CPU/CUDA 语义统一——它们改变的是"数据正确性"而非方法本身。eval/train 模式两臂均测（协议要求） | `FMT_Clustering.py`, `FMT_Utils/FMT_encoder.py` | `config/PathlineFMTclustering.yaml` | ARI / NMI / F1(vs IVD阈值) / F1(vs Vatistas解析标签) | — |
| （待跑）Verify_objectivity_1.1 | — | Task1 | Killing 观察者不变性测试：随机时变刚体观察者变换场 → 重积分 → 重编码 → 特征漂移量化 | `FLowUtils/KillingObserver2D.py`（符号先校准，见问题分析 P7） | — | 特征相对漂移 / 聚类标签翻转率 | — |
| mainExp_3DFMT_1.1 | 2026-08-17 | Task1-3D | 每个种子生成 7 条 pathline（center、x±、y±、z±）；中心位移与邻居相对位移先做时间差分；三维实序列逐坐标做 Fourier 变换，每个频率取 Gram 不变量并附加旋向三重积；邻居特征排序池化；StandardScaler + KMeans(k=2)；输出真实物理比例的 3D/正交投影/Z截面/pathline 图，并计算同一时刻 3D IVD 体数据及 p90/p95/p97.5 等值面 | `FMT_Clustering_3D.py`, `FMT_Utils/DFT_FMT_3D.py`, `FMT_Utils/FMT_3D_pipeline.py` | `config/PathlineFMTclustering3D.yaml` | 合成 smoke：180/180 valid，feature `[180,77]`，cluster `[60,120]`。真实 `halfcylinderRe160Resampled.nc`：两次均为 7200/8000 valid、feature `[7200,161]`。默认 `t=3`：cluster `[7000,200]`，少数 cluster 1 对 IVD 的 p90/p95/p97.5 F1=.347/.558/.833。指定 `--seed-time 7`：cluster `[828,6372]`，少数 cluster 0 对 IVD：p90 F1=.674/IoU=.509/P=.774/R=.597；p95 F1=.736/IoU=.582/P=.615/R=.917；p97.5 F1=.468/IoU=.306/P=.306/R=1.000 | `t=3` 少数簇只对应近圆柱最强 IVD 核；`t=7` 尾迹已发展为明显的波动结构，少数簇沿近尾迹展开，和 p95 IVD 区域最接近。它覆盖全部 p97.5 强 IVD 点但相对该窄阈值过分割，因此不能只凭最高分位判定成败。KMeans 簇编号每次运行可交换；这里的“少数簇”按样本数识别。IVD 百分位只是可追溯参考阈值，不是唯一 ground truth。只保证常数旋转/常数平移不变，不保证时变观察者下的完整客观性。 |
| Verify_IVDSearch_1.1 | 2026-08-17 | Task1-3D 诊断 | 定义局部扩展 `IVD_a=||ω-mean_a(ω)||`；搜索 `a=[3,5,7,9,11,15,global]`，有限 `a` 是 `a³` 体素 box mean，`global` 是标准全域 IVD；对每个 `a`、两个簇和种子处全部可区分阈值精确搜索 F1。为防止“几乎全场为涡”的多数类退化解，涡候选要求簇和 IVD 正类均不超过 50%；无约束最优仍落盘供审计 | `FMT_Utils/IVD_parameter_search_3D.py`, `FLowUtils/ScalarField3d.py`, `FMT_Clustering_3D.py` | `visualization.ivd_search_averaging_sizes` | `halfcylinderRe160Resampled.nc`, `t=7`：涡候选最优 `a=11`, level=`0.913665`, cluster 0，F1=`.853`, IoU=`.743`, P=`.835`, R=`.871`。无约束最优 `a=3`, level=`3.77958e-7`, cluster 1，F1=`.939`，但 IVD 正类覆盖 100% 种子、所选多数簇占 88.5% | 局部 11³ 体素平均比标准全域平均更接近当前少数簇；全域基线对 cluster 0 的最佳 F1=`.757`。该搜索使用聚类标签选择 IVD 参数，属于事后相似度诊断，不能作为独立 ground truth 或无偏性能指标。 |
| Verify_3DFMTHyperparam_1.1 | 2026-08-17 | Task1-3D 超参数 | 冻结 `halfcylinderRe160Resampled, t=7, local-IVD a=11, level=.913665` 和同一批 7200 个 primitive；搜索 7 个频率数 × 2 种 Fourier 不变量 × chirality 开/关 × 3 种邻居池化 × 5 个邻居权重，共 420 组。粗筛 KMeans `n_init=1`；前 10 名用 5 个 random state、每次 `n_init=10` 复核。`neighbor_scale` 因会被逐列 StandardScaler 严格抵消而不列入搜索 | `Verify_3DFMTHyperparam.py` | `config/Verify_3DFMTHyperparam_1.1.yaml` | 最佳：`num_freq=6, mode=gram, chirality=true, pool=sort, neighbor_weight=.5`，5 次 mean/min/max F1 均为 `.852750`，std≈0。第二名关闭 chirality：F1=`.851697`；第三名 `freq=8` 且其余同最佳：F1=`.850450`。420 组加稳定性复核耗时 100.76 s（CUDA） | 当前 `mainExp_3DFMT_1.1` 配置恰为本搜索空间内稳定最优，不修改冻结基线。chirality 收益仅 `.00105`，很小但在 5 个 KMeans seed 上稳定。该结论只来自一个场的一个时刻，并且参考 IVD 参数曾在同一数据上选择；它是训练集内超参数结果，必须在其他时间和其他 3D 场上做 held-out 验证后才能称为可泛化最佳。 |
| mainExp_3DFMTVAE_1.1 | 2026-08-17 | Task2-3D | 固定 cylinder3d `t=7` 的 7200 个 primitive 和 local-IVD `a=11, level=.913665`；无标签随机 80/20 train/test。四臂：raw local primitive direct、raw→VAE、FMT direct、FMT→VAE。raw 先减中心线起点；两种输入的 StandardScaler 只 fit train。两种 VAE 共用 hidden `[256,128]`、latent 16、β=.001、200 epoch；5 个训练 seed。KMeans 只 fit train latent，IVD 仅用于 test F1 | `Train_3DFMT_VAE.py`, `FMT_Utils/VAE_3D.py` | `config/mainExp_3DFMTVAE_1.1.yaml` | held-out F1：raw direct `.5935`；raw+VAE mean `.5644±.0241`（`.5407–.5961`）；FMT direct `.7961`；FMT+VAE mean `.7815±.0634`（`.6563–.8254`）。FMT+VAE − raw+VAE=`+.2171`；FMT+VAE − FMT direct=`−.0145` | FMT tokenizer 对 VAE 有明显价值：相同 VAE 主体下比 raw 输入高 `.217`。但本版不能证明“VAE 提高 FMT feature”：平均值低于 FMT direct，且一个 seed 明显退化；低重建误差没有保证 latent 聚类稳定。该负结果冻结保留，下一版需用独立 validation 选择 latent 维度/β，再在 untouched test 上报告。 |

## 定量协议（v1，改动需升版本并注明）

- 数据：cylinder2d, doublegyre2d, beads2d, pipedcylinder2d；时间窗 [0.6, 0.8]×T；参数沿用 `PathlineFMTclustering.yaml`（dt=0.005, max_steps=300, L=30, offset=0.02, grid 0.25）。
- 参考标签：(a) IVD 阈值（`ScalarField2d.compute_ivd_2D`，阈值=全数据集固定分位，非逐切片）；(b) Vatistas 合成场解析标签（`VatistasFlowDatasetGenerator.py`）。
- 指标：ARI、NMI、对参考标签的 F1（聚类簇经匈牙利匹配后计）；KMeans 固定 random_state，特征先做显式标准化并记录方式。
- 确定性：固定全部 seed；encoder 显式 `.eval()` 或显式逐片标准化，二选一并记录（两者都测一次，作为 P1 的复核）。
