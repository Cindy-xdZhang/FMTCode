# FMT — Pathline Geometry Features

**2026-09-17硬性协议：FMT完整预测路径禁止空间卷积，包括原始坐标/冻结骨干中的Conv1d、Conv2d、Conv3d及转置卷积。** 旧Raw＋FMT残差、Task4-b raw_fmt、FMT＋U-Net及P35卷积融合方案已撤销。旧名称只会报错，旧成绩不得进入有效FMT表。详见 [项目协议](docs/FMT_no_spatial_convolution_protocol_1.1.md)；本条优先于以下历史状态。

FMT 将中心轨线及其邻居组成的轨线组编码为几何特征。编码器没有可训练参数，
用于涡区域聚类、无监督表示学习和监督识别；后续聚类器与神经网络仍需拟合。

目标会议为 ICLR。**2026-09-13 当前状态**：保留 Task1/2/3/5 已冻结的分类研究，停止继续追求现有客观化路线，
暂停 Task6/7/8 几何 tokenizer 及混合专家调参。用户随后选定并修订 [Task4-c 2.1](docs/Task4C_hairpin_binary_protocol_2.1.md)：
从Channel局部线簇几何判断Hairpin / Non-hairpin，比较FMT＋多层感知机与Conv3D＋多层感知机。
首轮六次训练与独立复算已完成，结果见[Task4-c记录](docs/experiment_log.md#task4c-binary-2026-09-13)。
原 FMT 不能整体宣称对任意时变刚体参考系客观，也未证明具有通用几何重建优势。
研究进展、关键反例和修订依据集中在[项目进展记录](docs/experiment_log.md#progress-2026-09-13)；
[验证总表](docs/Verify_experiments.md#status-20260913)提供按问题查找的入口。

当前主实验覆盖十个三维数据条目，采用任务级统一配置：

- Task1：直接用轨线特征做涡区/非涡区聚类。
- Task2：比较原始轨线与 FMT 作为同一变分自编码器输入的效果。
- Task3：有监督涡区域二分类。
- Task5：在训练未见的空间与时间尺度组合上进行二分类。

Task4-a/b 保留原代理标签实验。Task4-c 2.1以GT区域提供二分类标签，旧1.1四分类和BiLSTM仅保留历史实现。
项目研究对象为有局部信息的线簇，不限定pathline、streamline或vortex line；各实验明确自己的积分场与采样设置。
准确的任务定义和评测限制见 [研究协议](docs/research_tasks_and_protocol.md)。

## 代码结构

```text
FMT_Utils/       FMT 编码器、几何特征和任务模型
FLowUtils/       流场读取、轨线积分和涡判据
DeepUtils/      配置等基础工具
pnn/            既有点云模型与可视化支持
assets/         积分所需的 CUDA 源码
experiments/    实验执行、数据准备、评估和审计
config/         当前实验配置及必要的冻结依赖
ibex_bash/      集群提交入口
tests/          单元测试与当前流程检查
docs/           研究协议、结果和维护说明
tools/          仓库检查与历史归档工具
```

## 运行

从仓库根目录执行。现有研究环境的依赖记录在 `requirements_fmt.txt`；其中包含平台相关
依赖与 CUDA 版本，尚未整理为跨平台安装包。数据需要自行准备，并修改配置中的数据路径。
输出默认写入 `outputs/`，不提交数据、训练模型或运行日志。

普通三维聚类入口：

```bash
python -m experiments.FMT_Clustering_3D --input /path/to/field3d.nc
```

当前论文实验的入口与配置：

- Task1：`experiments/Run_Task1_3D_Uniform.py`，`config/mainExp_Task1_3D_4.1_uniform.yaml`。
- Task2：直接特征交互分析，见 `docs/Task2_feature_visual_analysis_protocol_2.1.md`；旧FMT＋VAE指标流程已撤销。
- Task3/5：无卷积直接分类 `experiments/ASAPFMT_Task35_1_1.py`，配置 `config/mainExp_ASAPFMT_Task35_1.1.json`；包括fmt_c156和ASAP。旧Raw＋FMT卷积融合/残差入口已撤销。
- Task4-c：`experiments/Task4C_P35GTHead_1_1.py` 对应原p35/h0全连接小网络；c156为加宽残差全连接网络。独立Conv3D对照不是FMT方案。
- 历史更强对照、消融与噪声配置仅保留核对记录；涉及卷积FMT的分支不再获准运行。

例如查看统一确认流程的参数：

```bash
python -m experiments.Run_UniformFMT_Confirmation_3D --help
```

## 结果与历史

[Task2 visual analysis](docs/Other_Task2_VisualAnalysis_1.1.md) 提供 FlowNet 风格的
轨线几何与 t-SNE 联动视图，支持 DBSCAN 多簇探索、Raw/FMT 切换、套索选择及离线导出。
入口为 `experiments/Task2_Visual_Analysis.py`；使用开发集，不修改主实验二分类结果。
[1.2版本](docs/Other_Task2_VisualAnalysis_1.2.md) 新增流场/dt/积分步数重算、后台进度和双视图隐藏簇。

[论文结果](docs/paper_tables_tasks_3d.md) 汇总当前主实验。
[绘图方法与代码索引](docs/plotting_methods_catalog.md) 记录三联图、密集近景和性能图的入口、特点、数据版本与复用限制。
三联图现有代码：Task1使用模式a（背景/两簇/误差），Task2、Task3与Task5使用模式b（参考/无FMT/FMT）。
本次指定的Task1/3/5背景—分类—误差布局与现有实现的区别见[绘图索引](docs/plotting_methods_catalog.md#triptychs-20260913)。[六流场最新成图](docs/Other_Task123_PaperTriptychs_1.2.md)。
[验证实验总表](docs/Verify_experiments.md) 按问题概括以往验证、负结果及未完成项。
需要精确复现时查看 [完整实验流水](docs/experiment_log.md) 和
[Ibex 作业登记](docs/ibex_run_registry.md)。

不再使用的版本扫描脚本、配置和配套测试已移出工作目录，保留为可恢复的本地源文件归档。
被当前流程调用的历史实现仍保留，避免更改冻结基线。详见
[整理与恢复说明](docs/repository_maintenance.md)。

初始代码来自 PyflowVis，原始研究背景与代码来源记录保留在 `docs/from_pyflowvis/`。
目前是开源前整理中的研究仓库，尚未完成许可证、依赖和数据分发检查。

Task2 交互分析新增起始时间、源末尾截断、UMAP和KMeans：[使用说明](docs/Other_Task2_VisualAnalysis_1.3.md)。
