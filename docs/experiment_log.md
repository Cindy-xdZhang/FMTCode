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
| mainExp_3DFMT_1.1 | 2026-08-17 | Task1-3D | 每个种子生成 7 条 pathline（center、x±、y±、z±）；中心位移与邻居相对位移先做时间差分；三维实序列逐坐标做 Fourier 变换，每个频率取 Gram 不变量并附加旋向三重积；邻居特征排序池化；StandardScaler + KMeans(k=2)；输出真实物理比例的 3D/正交投影/Z截面/pathline 图，并计算同一时刻 3D IVD 体数据及 p90/p95/p97.5 等值面 | `FMT_Clustering_3D.py`, `FMT_Utils/DFT_FMT_3D.py`, `FMT_Utils/FMT_3D_pipeline.py` | `config/PathlineFMTclustering3D.yaml` | 合成 smoke：180/180 valid，feature `[180,77]`，cluster `[60,120]`。真实 `halfcylinderRe160Resampled.nc`：7200/8000 valid，feature `[7200,161]`，cluster `[7000,200]`。cluster 1 对 IVD：p90 F1=.347/IoU=.210/P=1.000/R=.210；p95 F1=.558/IoU=.387/P=1.000/R=.387；p97.5 F1=.833/IoU=.714/P=1.000/R=.714 | 少数簇在全部 Z 截面稳定集中于近圆柱强 IVD 区，代表性轨线明显弯曲/回旋。它对最高 IVD 区 precision=1，但覆盖不完整；当前更像最强涡核候选区，而非宽阈值下的完整涡区域。IVD 百分位只是可追溯参考阈值，不是唯一 ground truth。只保证常数旋转/常数平移不变，不保证时变观察者下的完整客观性。 |

## 定量协议（v1，改动需升版本并注明）

- 数据：cylinder2d, doublegyre2d, beads2d, pipedcylinder2d；时间窗 [0.6, 0.8]×T；参数沿用 `PathlineFMTclustering.yaml`（dt=0.005, max_steps=300, L=30, offset=0.02, grid 0.25）。
- 参考标签：(a) IVD 阈值（`ScalarField2d.compute_ivd_2D`，阈值=全数据集固定分位，非逐切片）；(b) Vatistas 合成场解析标签（`VatistasFlowDatasetGenerator.py`）。
- 指标：ARI、NMI、对参考标签的 F1（聚类簇经匈牙利匹配后计）；KMeans 固定 random_state，特征先做显式标准化并记录方式。
- 确定性：固定全部 seed；encoder 显式 `.eval()` 或显式逐片标准化，二选一并记录（两者都测一次，作为 P1 的复核）。
