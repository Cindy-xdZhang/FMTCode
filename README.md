# FMT — Training-free Objective Flowmap Tokenizer

FMT 将中心轨线及其邻居组成的轨线组编码为几何特征。编码器没有可训练参数，
用于涡区域聚类、无监督表示学习和监督识别；后续聚类器与神经网络仍需拟合。

当前主实验覆盖十个三维数据条目，采用任务级统一配置：

- Task1：直接用轨线特征做涡区/非涡区聚类。
- Task2：比较原始轨线与 FMT 作为同一变分自编码器输入的效果。
- Task3：有监督涡区域二分类。
- Task5：在训练未见的空间与时间尺度组合上进行二分类。

Task4 是独立的三维涡类型探索，目前使用代理标签，不能与 Task3 二分类结果混用。
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
- Task2/3：`experiments/Run_UniformFMT_Confirmation_3D.py`，分别使用
  `config/mainExp_Task2_3D_6.2_uniform_confirmation.yaml` 和
  `config/mainExp_Task3_3D_9.2_uniform_confirmation.yaml`。
- Task5：`experiments/Evaluate_Task5_Multiscale.py`，`config/mainExp_Task5_3D_1.1_evaluate.yaml`。
- 更强对照、消融与噪声测试：`experiments/Run_Task123_*.py`，对应配置和提交入口保留在
  `config/` 与 `ibex_bash/`。这些流程需要已有数据缓存及冻结选择记录，不是单命令下载即跑。

例如查看统一确认流程的参数：

```bash
python -m experiments.Run_UniformFMT_Confirmation_3D --help
```

## 结果与历史

[论文结果](docs/paper_tables_tasks_3d.md) 汇总当前主实验。
[验证实验总表](docs/Verify_experiments.md) 按问题概括以往验证、负结果及未完成项。
需要精确复现时查看 [完整实验流水](docs/experiment_log.md) 和
[Ibex 作业登记](docs/ibex_run_registry.md)。

不再使用的版本扫描脚本、配置和配套测试已移出工作目录，保留为可恢复的本地源文件归档。
被当前流程调用的历史实现仍保留，避免更改冻结基线。详见
[整理与恢复说明](docs/repository_maintenance.md)。

初始代码来自 PyflowVis，原始研究背景与代码来源记录保留在 `docs/from_pyflowvis/`。
目前是开源前整理中的研究仓库，尚未完成许可证、依赖和数据分发检查。
