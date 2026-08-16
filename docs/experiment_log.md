# FMT 实验记录（唯一结论载体）

规则（与全局研究规范一致）：
1. 版本命名：主实验 `mainExp_x.y`；组件验证 `Verify_[xxx]_x.y`；非主线探索 `Other_[xxx]_x.y`；消融 `Ablation_[xxx]_x.y`。x = 大迭代，y = 小迭代。
2. 每行必须可追溯：技术细节 + 主要代码路径 + config + commit + 指标。没有数字的结论不进表。
3. 修订旧结论必须新旧并列写明变更原因，禁止静默翻转。
4. Baseline 冻结：VortexTransformer（已发表）冻结在 PyflowVis 仓库，不复制不修改；本仓库的 legacy 聚类管线（mainExp_1.1 待跑）一经记录即冻结。

## 版本表

| 版本 | 日期 | 任务 | 技术要点 | 主要代码路径 | config | 指标（协议见下） | 结论 |
|---|---|---|---|---|---|---|---|
| （待跑）mainExp_1.1 | — | Task1 | 旧法冻结复跑：FMT(PosE+GeoLinePicker+LGA+Pool, stages=2, embed24, alpha1000, beta19, temporal_head=None) + KMeans(k=2)，4 输入视图；补定量协议与确定性设置 | `FMT_Clustering.py`, `FMT_Utils/FMT_encoder.py` | `config/PathlineFMTclustering.yaml` | ARI / NMI / F1(vs IVD阈值) / F1(vs Vatistas解析标签) | — |
| （待跑）Verify_objectivity_1.1 | — | Task1 | Killing 观察者不变性测试：随机时变刚体观察者变换场 → 重积分 → 重编码 → 特征漂移量化 | `FLowUtils/KillingObserver2D.py`（符号先校准，见问题分析 P7） | — | 特征相对漂移 / 聚类标签翻转率 | — |

## 定量协议（v1，改动需升版本并注明）

- 数据：cylinder2d, doublegyre2d, beads2d, pipedcylinder2d；时间窗 [0.6, 0.8]×T；参数沿用 `PathlineFMTclustering.yaml`（dt=0.005, max_steps=300, L=30, offset=0.02, grid 0.25）。
- 参考标签：(a) IVD 阈值（`ScalarField2d.compute_ivd_2D`，阈值=全数据集固定分位，非逐切片）；(b) Vatistas 合成场解析标签（`VatistasFlowDatasetGenerator.py`）。
- 指标：ARI、NMI、对参考标签的 F1（聚类簇经匈牙利匹配后计）；KMeans 固定 random_state，特征先做显式标准化并记录方式。
- 确定性：固定全部 seed；encoder 显式 `.eval()` 或显式逐片标准化，二选一并记录（两者都测一次，作为 P1 的复核）。
