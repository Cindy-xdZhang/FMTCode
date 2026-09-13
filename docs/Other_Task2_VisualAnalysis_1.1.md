# Task2 visual analysis

实验版本：`Other_Task2_VisualAnalysis_1.1`。独立的开发集探索工具，观察 VAE 潜在特征与轨线几何的对应关系。已有 Task2 的 KMeans 二分类代码、标签阈值、主表与模型选择均不修改。

## 论文依据及实现边界

本地 FlowNet PDF 第 3–4 页（§3.1–3.3）、第 11–12 页附录：FlowNet 使用体素输入的普通卷积自编码器，得到 1024 维描述子；逐向量 L1 归一化后做 t-SNE，并在 **t-SNE 二维空间**进行 DBSCAN。并非 VAE。两视图通过颜色和选择事件联动。

这里保留 Task2 的变分自编码器（Variational Autoencoder，VAE：用概率潜在表示重建输入），使用确定性的编码均值 `mu`，不使用随机采样的 `z`，不引入体素化或三维卷积。

DBSCAN（Density-Based Spatial Clustering of Applications with Noise：按邻域密度分簇，稀疏样本记为噪声）可在 `tsne` 或 `latent` 空间计算。默认 `tsne` 符合 FlowNet；`latent` 用于检查投影之前的分组。t-SNE（t-distributed Stochastic Neighbor Embedding：保留局部邻近关系的二维投影）的位置、距离和密度不能直接解释为真实高维距离或物理类别。

一个点对应一个含中心线和六条邻线的 primitive。左图默认绘制中心线，也可显示完整七线，始终沿用同一 primitive 标签。缓存 `raw_features` 已减去中心种子的初始位置，物理坐标必须加回 `seeds`；“Shapes at common origin” 则用于去除平移后观察形状，不消除旋转或尺度。

## 固定配置

| 版本 | 技术细节 | 主要代码 | 指标/状态 |
|---|---|---|---|
| Other_Task2_VisualAnalysis_1.1 | 已冻结统一 Task2 VAE 配方；Raw/FMT 相同架构、种子与训练参数；mu → L1 normalization → t-SNE → DBSCAN；也支持直接在归一化 mu 上聚类 | `experiments/Task2_Visual_Analysis.py`、`FMT_Utils/Task2VisualAnalysis.py` | 探索工具；不计算或替换主表 F1；结果由每次导出的 JSON/CSV 记录 |

读取 `mainExp_Task2_3D_6.2_uniform_confirmation` 冻结 manifest，校验配置、选择与审计文件哈希。当前固定配方为 `fmt_all+kin4`、hidden dimensions `[512,256]`、latent dimension `64`；具体数值以被校验的文件为准。两臂只有输入表示及其必要输入层宽度不同，分别训练，同一 seed。直接复用 `_prepare_inputs` 和 `_train`，不保存 checkpoint。

交互式探索仅打开 `train` 和 `cluster_calibration`。默认按元数据选择后者中最早的合规时间片，**不读取 confirmation/test**，不读取参考标签数组。Cylinder 按原始 `[0,15]` 限制 `t >= 7.5`，同时排除更早的训练片。这是本探索版本的采样变化，记录原训练划分和实际划分，不宣称严格复现原主表训练。非 Cylinder 保持冻结训练片。

全部导出样本保留原始局部样本行号。为控制浏览器负担，分析前按固定 seed 均匀无放回抽取最多 1500 个 primitive；Raw/FMT 共用相同行号。t-SNE 和 DBSCAN 都在这组样本上计算，**不是先聚类全部样本再抽样显示**。结果中的簇数和噪声比例只描述该分析子集。取消子采样可把 `max_samples` 设为大于源样本数的值，但高样本数和大 eps 会增加 DBSCAN 内存和浏览器负担。

默认 L1 归一化及 t-SNE seed/perplexity/iterations 均写入配置；修改它们需重新启动或渲染。UI 中调整 DBSCAN 不会重新训练或重新计算同一臂的 t-SNE。`eps` 使用所选空间的距离单位，两种空间之间不能直接比较其数值。全噪声、单簇均如实显示；不按图像好看程度自动寻找参数。不同臂的 cluster ID 和颜色不表示类别身份相同。

## 使用

在仓库根目录、包含项目训练依赖的 Python 环境运行。交互依赖已在项目 `requirements_fmt.txt` 中包含：NumPy、scikit-learn、Plotly、Dash；导出 latent 还需要原 Task2 的 PyTorch 等依赖。

```powershell
# 只需一次：重训两臂并导出 mu 和真实几何，不写模型文件。
python experiments/Task2_Visual_Analysis.py export --dataset cylinder3d

# 启动本机交互工具，然后打开 http://127.0.0.1:8052。
python experiments/Task2_Visual_Analysis.py serve --bundle outputs/Other_Task2_VisualAnalysis_1.1/cylinder3d/latent_bundle.npz

# 无服务的可携带 HTML 报告，支持点选/套索联动，内嵌 Plotly，无 CDN。
python experiments/Task2_Visual_Analysis.py render --bundle outputs/Other_Task2_VisualAnalysis_1.1/cylinder3d/latent_bundle.npz --arm fmt

# 直接在归一化高维潜在空间聚类的对照视图；eps 是明确的探索参数。
python experiments/Task2_Visual_Analysis.py render --bundle outputs/Other_Task2_VisualAnalysis_1.1/cylinder3d/latent_bundle.npz --arm raw --cluster-space latent --eps 0.1
```

可选 `--config`、`--output`、`--device cpu/cuda/auto`、`--port`。`export --output` 是 `.npz` 文件路径；`render --output` 是报告目录。已有 latent bundle 不会被覆盖。重新探索先复用已有 bundle；要重训时指定新输出路径并保留旧结果。

UI：选择输入臂/聚类空间、填写 eps 与 min_samples 后点击 **Apply clustering**；当前已应用参数显示在状态栏。多选 cluster 筛选几何，右侧点选/套索或左侧点选联动；Reset selection 清除样本选择，清空 cluster 下拉框恢复全部簇。Noise (-1) 使用灰色，始终可单独选择。每个视图可旋转/缩放；物理坐标保持等比例。

界面名称已明确为 **t-SNE 投影空间（2D）** 和 **VAE 潜在特征空间（实际维数D）**，删除容易误解为网络选择的 “FlowNet” 标记。界面常显说明列出两条处理顺序、当前归一化方法、距离与密度的区别及 eps 单位。相同输入臂的右图坐标不因切换聚类空间改变，改变的是分组颜色；模型和计算方法未因此更改。

**Export current view** 下载 ZIP，包含可离线打开的 `index.html`、全部分析样本的 `samples.csv`、记录当前选择和参数的 `analysis.json`。离线报告只对已导出的几何联动，Reset 恢复导出时的几何集合；修改 DBSCAN 需回到在线工具。CLI `render` 另存 `analysis.npz`，包含 mu、二维坐标、簇标签、核心点标记、对应几何与原始行号。

每份 bundle 记录配置、数据与源码哈希、git commit、依赖版本、设备、训练种子、时间片、训练损失；每份分析报告记录子采样、投影和聚类参数。新参数/投影/种子比较请用 `--output` 指定不同目录以保留历史。UI 操作不自动写实验目录，需导出保存当前状态。方法级结论只记入 `docs/experiment_log.md`。

## 验证

`python -m pytest tests/test_task2_visual_analysis.py` 验证物理坐标恢复、聚类空间区别、核心点和噪声、时间范围和开发集限制、原样本编号、折线分隔、非法参数及 Dash 服务端入口。合成测试仅验证程序行为，不能作为真实流场的几何聚类证据。

2026-09-08 本机验证：`tmp/jhtdb-venv/Scripts/python.exe` 环境下 **12 tests passed**，包含当前选择 ZIP 导出的内容检查。系统默认 `python` 缺少依赖；本工作区可以将上方命令中的 `python` 换成 `./tmp/jhtdb-venv/Scripts/python.exe`。

已生成 Re160 的真实示例：`outputs/Other_Task2_VisualAnalysis_1.1/cylinder3d/latent_bundle.npz`，以及 `reports/fmt_tsne_eps2.0_min10/index.html`、`reports/raw_tsne_eps2.0_min10/index.html`。样本时间 `t=10.0`，3584 个源 primitive，固定抽样 1500 个。两臂均完成 7000 个优化步骤，训练只用合规的 ordinal 4/5。具体聚类统计及证据边界见实验日志，不据簇数判断表示优劣。

浏览器人工检查：两视图完整显示、簇筛选、完整七线切换、单点选择、框选联动通过。离线 HTML 的文件 URL 预览被浏览器策略阻止，未声称完成离线页面的浏览器验证；HTML 自包含检查和 ZIP 内容检查通过。服务只监听 `127.0.0.1:8052`。
