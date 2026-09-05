# FMT 项目研究协议

## 唯一任务定义

- **Task1（2D/3D）**：training-free FMT encoder 作用于 pathline primitive，输出 feature vector；使用 KMeans 做涡区域/非涡区域二类聚类。FMT encoder 无可训练参数，KMeans 仍需拟合聚类中心。
- **Task2（2D/3D）**：无监督表示学习。核心比较固定为 `Raw pathline -> VAE -> latent -> KMeans` 与 `FMT(pathline) -> VAE -> latent -> KMeans`；同一 physical-family 内两臂必须使用为 FMT 开发并冻结的同一个 VAE（架构、latent dimension、KL 权重、学习率、训练步数相同），不得为 Raw 单独搜索更强 VAE 后替换主 baseline。主结论只回答 FMT 是否改善该同一 VAE 的输入表示；独立优化的 strongest-Raw 只能作附录压力测试。
- **Task3（2D/3D）**：有监督 IVD 涡识别。核心比较固定为不使用 FMT 与加入 FMT 的神经网络在涡/非涡二分类上的性能。
- **Task4（仅 3D）**：有监督涡类型多分类，例如 streamwise、spanwise、hairpin。Task4 与 Task3 严格分开。
- **Task5（2D/3D）**：Task3 的不同尺度扩展。primitive 的邻居距离、积分步长和积分步数可变，但输出保持固定线数和每线采样点数；使用同一 IVD 二分类监督。核心比较为 fixed-scale Task3 transfer、variable-scale Raw、同结构 Raw-PCA residual 与 variable-scale Raw+FMT，并在训练未见尺度组合上确认。

当前研究范围包含 **3D Task1、Task2、Task3、Task5**，并已开始仅限 channel flow 的 **Task4-b proxy-label 探索**。Task4-b 1.1 因跨 split primitive 空间重叠被审计否决；当前有效协议为带缓冲空间拆分的 1.2。Task4-b 与 Task3 的 whole-field IVD 二分类结果严格分开；旧实验 ID 不因本协议改名。

当前论文中所有采用 whole-field IVD 二分类的 3D 实验统一固定为 **IVD p95**：Task1/Task2 将其用于评估，Task3/Task5 将其用于监督。`Ablation_Task23IVDPercentile_1.2` 表明，在 p80、p85、p87.5、p90、p92.5、p95 的完整扫描中，p95 给出最大的 Task2 F1 增益以及 Task3 F1、Average Precision 增益；较低百分位只作为标签敏感性分析。任何后续阈值变更必须建立新实验版本，不得重写已有 p95 结果。

论文主表采用任务级统一 FMT 配方：Task1、Task2、Task3 必须分别从既有候选中选择一套跨全部10个3D数据条目不变的FMT feature blocks、FMT侧权重/缩放及后处理维度；允许用新随机种子复跑既有候选，不得新增未搜索过的超参数。旧逐flow/逐physical-family最优配置、代码和结果只作补充表与可视化，不得混入统一主表macro。Task5的FMT encoder配方同样固定；只有其研究问题预注册的邻居距离、积分步长和积分步数可按尺度tuple变化。

完整定义和评测边界见 `docs/research_tasks_and_protocol.md`。

## 实验记录

- 方法级结论只写入 `docs/experiment_log.md`。
- 每个向 Ibex 提交的进程必须登记到 `docs/ibex_run_registry.md`；失败、取消、超时和无效实验也不得删除。
- 提交后立即登记 job ID、实验版本、任务、提交时间、config、git commit 和预期设备；开始后补实际开始时间、节点和 GPU；结束后补主要结果及其支持/反对的结论。
- 不得用 test/confirmation 数据选模型、阈值、特征、epoch 或超参数。任何修订必须新旧并列记录。

## 实验文件保留规则

- 训练模型 checkpoint 不属于长期实验结果。默认不得在本地保存、下载或归档 `.pt`、`.pth`、`.ckpt` 及包含这些文件的压缩包。
- 如果 Ibex 上的后续依赖作业必须读取 checkpoint，可以只在该实验依赖链运行期间临时保存；最终评估的逐次 CSV/JSON 和总实验表写完后必须删除 checkpoint。
- 可复现依据是代码 commit、完整 config、随机种子、数据时间片/划分、设备记录、逐次指标和汇总表，不是训练好的模型文件。只有用户明确要求保留某个模型时才可例外。
