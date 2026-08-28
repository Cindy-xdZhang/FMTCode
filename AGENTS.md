# FMT 项目研究协议

## 唯一任务定义

- **Task1（2D/3D）**：training-free FMT encoder 作用于 pathline primitive，输出 feature vector；使用 KMeans 做涡区域/非涡区域二类聚类。FMT encoder 无可训练参数，KMeans 仍需拟合聚类中心。
- **Task2（2D/3D）**：无监督表示学习。核心比较固定为 `Raw pathline -> VAE -> latent -> KMeans` 与 `FMT(pathline) -> VAE -> latent -> KMeans`；同一 physical-family 内两臂必须使用为 FMT 开发并冻结的同一个 VAE（架构、latent dimension、KL 权重、学习率、训练步数相同），不得为 Raw 单独搜索更强 VAE 后替换主 baseline。主结论只回答 FMT 是否改善该同一 VAE 的输入表示；独立优化的 strongest-Raw 只能作附录压力测试。
- **Task3（2D/3D）**：有监督 IVD 涡识别。核心比较固定为不使用 FMT 与加入 FMT 的神经网络在涡/非涡二分类上的性能。
- **Task4（仅 3D）**：有监督涡类型多分类，例如 streamwise、spanwise、hairpin。Task4 与 Task3 严格分开。
- **Task5（2D/3D）**：Task3 的不同尺度扩展。primitive 的邻居距离、积分步长和积分步数可变，但输出保持固定线数和每线采样点数；使用同一 IVD 二分类监督。核心比较为 fixed-scale Task3 transfer、variable-scale Raw、同结构 Raw-PCA residual 与 variable-scale Raw+FMT，并在训练未见尺度组合上确认。

当前研究范围包含 **3D Task1、Task2、Task3、Task5**；Task4 尚未开始。旧实验 ID 不因本协议改名。

当前论文中所有采用 whole-field IVD 二分类的 3D 实验统一固定为 **IVD p95**：Task1/Task2 将其用于评估，Task3/Task5 将其用于监督。`Ablation_Task23IVDPercentile_1.2` 表明，在 p80、p85、p87.5、p90、p92.5、p95 的完整扫描中，p95 给出最大的 Task2 F1 增益以及 Task3 F1、Average Precision 增益；较低百分位只作为标签敏感性分析。任何后续阈值变更必须建立新实验版本，不得重写已有 p95 结果。

完整定义和评测边界见 `docs/research_tasks_and_protocol.md`。

## 实验记录

- 方法级结论只写入 `docs/experiment_log.md`。
- 每个向 Ibex 提交的进程必须登记到 `docs/ibex_run_registry.md`；失败、取消、超时和无效实验也不得删除。
- 提交后立即登记 job ID、实验版本、任务、提交时间、config、git commit 和预期设备；开始后补实际开始时间、节点和 GPU；结束后补主要结果及其支持/反对的结论。
- 不得用 test/confirmation 数据选模型、阈值、特征、epoch 或超参数。任何修订必须新旧并列记录。
