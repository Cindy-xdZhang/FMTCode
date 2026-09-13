# Task2 visual analysis 1.2：重积分和隐藏簇

版本 `Other_Task2_VisualAnalysis_1.2`。在1.1的联动界面上增加两项功能；原冻结实验和1.1结果均保留。

## 操作

1. 在“重新计算轨线几何与特征”中选择流场、输入积分步长 `dt` 和步数 `N`，点击“重新计算并切换流场”。`dt` 使用源数据时间坐标的单位，总积分时长为 `dt × N`，不是空间弧长。F-22 数据的源时间是帧索引，因此该项的单位也是源帧索引单位，不能声称是秒。
2. 后台顺序执行：读取源流场、重积分开发集训练及显示时间片、计算Raw/FMT输入、按同一冻结配方分别训练两臂VAE、生成潜在均值、t-SNE及DBSCAN，再切换两视图。进度会显示当前阶段。训练不是即时操作；期间旧视图仍可使用。失败时保留旧视图和失败记录。
3. “隐藏簇”支持多选任意cluster及Noise (-1)。两图都移除相应对象，不改变簇标签、噪声统计和t-SNE坐标。清空该选框恢复显示；原“只查看选中簇”仍独立可用。重新聚类或切换数据时清空旧簇选择，以免把旧簇编号用于新数据。

界面列出固定 Task2 配方中的10个流场，只有原始源文件可用的条目可执行重算。普通流场使用已有NetCDF窗口读取器；Channel使用已有VTK读取器和冻结缓存manifest中的时变观察者参数，不把稳态VTK冒充原有非稳态流场。可视化对象是轨线primitive，并非重新生成物体表面网格。

## 计算及实验边界

| 项目 | 1.2约定 |
|---|---|
| 可变项 | 流场、物理/源时间单位下的dt、积分步数N |
| 固定项 | RK4（四阶Runge–Kutta积分方法）、七条邻近轨线、每线32采样点、源配置中的种子网格和邻居距离、统一FMT配方和VAE训练参数 |
| 输入长度 | 沿用原积分器的均匀步索引抽样，每条线32点；N至少31，不改变VAE输入维数 |
| 模型处理 | 每次在新几何生成的开发集上训练两臂，不复用旧latent、不只替换左图几何；VAE架构和训练步数保持冻结配方 |
| 采样 | 源缓存manifest中的训练ordinal和最早合规的开发集显示ordinal；Cylinder继续限制t≥7.5 |
| 数据边界 | 所有重积分和插值所需的源帧均早于最早confirmation起始时间；不读取参考标签数组。开发集内已有训练/校准窗口可能重叠，本工具不报告独立泛化分数 |
| 失败条件 | 无效dt/N、窗口越界或跨入confirmation、所需流场数组超过1GiB配置上限、完整有效primitive不足100等，返回明确原因；不自动缩短轨线 |
| 数值误差 | float32源时间坐标可能导致读取器的局部均匀时间窗口略短；必要时多读一个真实帧，仍需满足划分和源边界，不修改请求dt |
| 并发 | 单个后台工作线程，避免两次训练并发改变随机状态；新数据按bundle身份切换，避免用旧几何解释新latent |
| 输出 | 每次请求在版本目录下新建时间戳+随机后缀子目录；不覆盖旧结果，不写模型checkpoint |

主入口：`experiments/Task2_Visual_Analysis.py`；重算实现：`FMT_Utils/Task2GeometryRecompute.py`；绘图：`FMT_Utils/Task2VisualAnalysis.py`；配置：`config/Other_Task2_VisualAnalysis_1.2.json`。默认CLI配置现在为1.2；仍可用 `--config` 明确读取1.1配置（无重算配置时重算按钮禁用）。

```powershell
./tmp/jhtdb-venv/Scripts/python.exe experiments/Task2_Visual_Analysis.py serve --bundle outputs/Other_Task2_VisualAnalysis_1.1/cylinder3d/latent_bundle.npz
```

打开 `http://127.0.0.1:8052/`。每次成功运行后，可用同一命令的 `--bundle` 指向新目录中的 `latent_bundle.npz` 重新打开该结果。

## 可复现记录与验证

每次重算保存 `job.json`（请求、进度、成功/失败及原因）、训练时间片的 `geometry_slice_*.npz`、`latent_bundle.npz/json` 和两臂的默认交互HTML/NPZ/CSV/JSON报告。记录实际dt/N、源文件路径/大小/修改时间、实际载入速度数组SHA256、源配置/冻结manifest/重算代码哈希、原始种子行号、种子和逐臂训练损失。没有下载或保存训练checkpoint。

导出当前视图的JSON增加 `hidden_clusters`。CSV保留完整分析子集的标签，隐藏只是显示操作，不能将隐藏后的图当作全体样本。

21项测试通过，涵盖实际常速度场积分、dt/N与轨线终点、float32窗口边界、拒绝confirmation帧、隐藏所有簇/噪声的双视图处理、失败任务记录及恢复。真实数据浏览器验证和结论记录在 `docs/experiment_log.md`。
