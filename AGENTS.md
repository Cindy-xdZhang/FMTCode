# FMT 项目研究协议

## 客观性表述（2026-09-07 用户强调）

- 同一时刻、同一对物质点的相对位置是客观几何向量；任意时变刚体观察者不会改变点间距离及同时间向量之间的夹角。
- 必须区分几何向量与坐标分量：在 x*=Q(t)x+c(t) 的分量约定下，d*=Q(t)d 是客观向量的正确变换规律，不能据此声称相对位置不客观或物理几何改变。客观标量要求数值不变，客观向量/张量要求遵守对应的基底变换规律；不得把两者混为一谈。
- 对后续计算另行逐步说明：客观向量的普通时间导数会包含运动基底项，不能直接从输入的客观性推断任意跨时间分量运算的结果。分析 aivd1w3_dft 时，明确连续涡量偏差公式、代码有限差分、同一时刻样本均值和最终标量之间的区别。
- 解释纠正不得静默改写冻结算法或历史指标。详细实现说明见 docs/aivd1w3_dft_explained_zh.md，方法结论与修订证据记入 docs/experiment_log.md。

## 唯一任务定义

- **Task1（2D/3D）**：training-free FMT encoder 作用于 pathline primitive，输出 feature vector；使用 KMeans 做涡区域/非涡区域二类聚类。FMT encoder 无可训练参数，KMeans 仍需拟合聚类中心。
- **Task2（2D/3D）**：无监督表示学习。核心比较固定为 `Raw pathline -> VAE -> latent -> KMeans` 与 `FMT(pathline) -> VAE -> latent -> KMeans`；同一 physical-family 内两臂必须使用为 FMT 开发并冻结的同一个 VAE（架构、latent dimension、KL 权重、学习率、训练步数相同），不得为 Raw 单独搜索更强 VAE 后替换主 baseline。主结论只回答 FMT 是否改善该同一 VAE 的输入表示；独立优化的 strongest-Raw 只能作附录压力测试。
- **Task3（2D/3D）**：有监督 IVD 涡识别。核心比较固定为不使用 FMT 与加入 FMT 的神经网络在涡/非涡二分类上的性能。
- **Task4（仅 3D）**：有监督涡类型多分类，例如 streamwise、spanwise、hairpin。Task4 与 Task3 严格分开。
- **Task5（2D/3D）**：Task3 的不同尺度扩展。primitive 的邻居距离、积分步长和积分步数可变，但输出保持固定线数和每线采样点数；使用同一 IVD 二分类监督。核心比较为 fixed-scale Task3 transfer、variable-scale Raw、同结构 Raw-PCA residual 与 variable-scale Raw+FMT，并在训练未见尺度组合上确认。
- **Task6（当前3D）**：局部流映射查询。将可见 primitive 编码成 token，预测未输入播种点在同一观察区间内的完整轨迹，以原流场积分自监督。
- **Task7（当前3D）**：遮挡区域补全。仅由隐藏区域外的可见 primitive tokens 恢复内部材料轨迹；必须防止重叠邻居泄漏。
- **Task8（当前3D）**：短流映射组合。使用各段已知 tokens，把第一段预测到达位置传入下一段查询，评估真实长轨迹；不是未知未来预测。

2026-09-09 用户选择 Task6/7/8 全部执行，并授权代码验证、Git commit/push、Ibex Git 拉取及批量实验；当前版本见 `docs/Task678_flowmap_protocol_1.1.md`。Task4 暂缓推进，保留以下历史协议及结果。Task6/7/8 不使用 IVD p95 或人工类别标签，不能把 Task1–5 的分类指标直接沿用为新任务目标。

当前研究范围包含 **3D Task1、Task2、Task3、Task5**，以及 **Task4-b 有监督四分类**。2026-09-06 用户重新定义 Task4-b：channel+TBL 合并训练同一个网络；涡区按 hairpin 标注内/外及 velocity–curl 更接近平行/垂直明确二分为 ordinary-streamwise、ordinary-spanwise、hairpin-head、hairpin-leg（平行含反平行，45°归平行；hairpin 内平行为 leg、垂直为 head），不加额外 head 条件。当前先执行 `Verify_Task4B_VelocityCurlMemorization_4.1` 过拟合验证，阈值使用标注全覆盖约束下的每流场 IVD>0.9a，详见 `docs/Task4B_velocity_curl_protocol_4.1.md`。历史 Task4-b 1.1 因跨 split primitive 空间重叠被否决，1.2/2.x/3.x 结果保留原协议与证据边界，不能替代新标签实验。Task4-b 与 Task3 的 whole-field IVD 二分类结果严格分开；旧实验 ID 不因本协议改名。

当前论文中所有采用 whole-field IVD 二分类的 3D 实验统一固定为 **IVD p95**：Task1/Task2 将其用于评估，Task3/Task5 将其用于监督。`Ablation_Task23IVDPercentile_1.2` 表明，在 p80、p85、p87.5、p90、p92.5、p95 的完整扫描中，p95 给出最大的 Task2 F1 增益以及 Task3 F1、Average Precision 增益；较低百分位只作为标签敏感性分析。任何后续阈值变更必须建立新实验版本，不得重写已有 p95 结果。

论文主表采用任务级统一 FMT 配方：Task1、Task2、Task3 必须分别从既有候选中选择一套跨全部10个3D数据条目不变的FMT feature blocks、FMT侧权重/缩放及后处理维度；允许用新随机种子复跑既有候选，不得新增未搜索过的超参数。旧逐flow/逐physical-family最优配置、代码和结果只作补充表与可视化，不得混入统一主表macro。Task5的FMT encoder配方同样固定；只有其研究问题预注册的邻居距离、积分步长和积分步数可按尺度tuple变化。

完整定义和评测边界见 `docs/research_tasks_and_protocol.md`。

## 实验记录

Task4-b 当前阈值以2026-09-06用户最新指定为准：**Channel IVD>5.785，TBL IVD>0.08**，取代上述4.1及后续4.2/4.3基于hairpin最小IVD的选择。当前 `Other_Task4B_ProxyGTThreshold_4.4` 仅生成ground truth和三维图，不训练；四类velocity-curl规则保持不变，不强制补回阈值以下hairpin标注。旧版数据和训练结果保留，后续训练需新版本。该更新不修改其他任务的p95。

- 方法级结论只写入 `docs/experiment_log.md`。
- Task4-b 的 `mainExp_Task4B_PatchSegmentation_5.1` 已冻结：channel/TBL各900训练+100测试小区域、实例66/7与52/6、共享FMT四分类网络；跨集合patch隔离、流线不出patch，固定200epoch后test一次。沿用4.4固定IVD阈值；详见 `docs/Task4B_patch_segmentation_protocol_5.1.md`。
- 当前用户授权的 `Other_Task4B_GeometrySearch_5.2` 尝试网络结构、学习率及几何点差估计的梯度/curl特征。只在5.1原训练区域内部划分有空间间隔的拟合/验证集来选候选和epoch，再恢复全部900+900训练patch重训；最后使用原test一次，明确其为已用过的benchmark，非新的confirmation。仍为共享四分类，不把真实curl、IVD、实例ID或标签作为输入。详见 `docs/Task4B_geometry_search_protocol_5.2.md`；原5.1结果不改写。
- 每个向 Ibex 提交的进程必须登记到 `docs/ibex_run_registry.md`；失败、取消、超时和无效实验也不得删除。
- 提交后立即登记 job ID、实验版本、任务、提交时间、config、git commit 和预期设备；开始后补实际开始时间、节点和 GPU；结束后补主要结果及其支持/反对的结论。
- 不得用 test/confirmation 数据选模型、阈值、特征、epoch 或超参数。任何修订必须新旧并列记录。

## Cylinder 默认时间范围（2026-09-06 用户更新）

- 所有 cylinder/half-cylinder 数据（含Re160、Re640、Re6400）后续可视化及新实验默认忽略原始模拟前50%的初始阶段，同时要求初始物理时间t>=7。当前三个3D数据原始模拟区间均为[0,15]，故有效起始下限为t=7.5。
- 按原始模拟时段计算50%，不能对已裁剪的文件再次截半；Re640现有文件[7.5,15]已经符合这一范围。
- 已有确认缓存用于新图时，按时间选择首个合规时间片，不依据图像、标签或指标择优：当前Task1 Re160/Re640/Re6400分别为t=10.5/9.5/10.5。
- 已冻结实验、训练/测试划分与旧结果不追溯修改；后续改变实验采样范围必须建新版本并记录。机器可读默认值见config/cylinder_time_policy.json。

## 实验文件保留规则

- 训练模型 checkpoint 不属于长期实验结果。默认不得在本地保存、下载或归档 `.pt`、`.pth`、`.ckpt` 及包含这些文件的压缩包。
- 如果 Ibex 上的后续依赖作业必须读取 checkpoint，可以只在该实验依赖链运行期间临时保存；最终评估的逐次 CSV/JSON 和总实验表写完后必须删除 checkpoint。
- 可复现依据是代码 commit、完整 config、随机种子、数据时间片/划分、设备记录、逐次指标和汇总表，不是训练好的模型文件。只有用户明确要求保留某个模型时才可例外。
