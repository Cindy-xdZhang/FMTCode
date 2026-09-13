# Verify_Task4B_VelocityCurlMemorization_4.1

2026-09-06 预注册。用户重新定义 Task4-b，明确 channel+TBL 合并训练一个四分类网络。
本实验重新生成标签与 primitive，不替换任何旧版本结果；方法级结果只记入 experiment_log.md。

## 冻结标签

名称：velocity–curl baseline as proxy ground truth。curl 指速度的旋度（涡量），不是速度梯度张量。
x=streamwise、y=spanwise、z=vertical；两个流场环境流沿 x+。

| hairpin 标注 | 速度与 curl 更接近平行（含反平行） | 更接近垂直 |
|---|---|---|
| 不在标注内 | 0 ordinary_streamwise | 1 ordinary_spanwise |
| 在标注内 | 3 hairpin_leg | 2 hairpin_head |

锐角 θ=acos(|u·ω|/(‖u‖‖ω‖))，θ≤45°归平行，θ>45°归垂直。全部可计算方向的涡区点明确二分；
不设角度排除带，不使用额外涡量正负、分量占优或位置条件。零向量/非有限值只按数值无效报告。

从 velocity 重算 curl：二阶有限差分；channel 的 x/y 用周期中心差分，z 按真实非均匀坐标，
TBL 全轴按非周期真实坐标。原存储涡量只用于数值比较，不进入标签规则。

IVD（Instantaneous Vorticity Deviation，瞬时涡量偏差）=‖ω−全域体积加权平均 ω‖。
非周期轴使用梯形积分权重，channel 周期 x/y 等权；不减去每个 z 平面的均值。
每个流场独立计算 a：原始正 VortexIds 单元中心与所有正标注目标体素中心上 IVD 的最小值，
再取其下一个较小 float64，使离散标注点严格满足 IVD>a。固定涡区条件 IVD>0.9a，
无 hairpin 强制纳入。零 IVD 或非有限标注值导致构建失败，不擅自放宽覆盖规则。
这里的全覆盖指上述离散支持，不宣称连续单元内部每一点都有数值保证。

## 输入与 primitive

输入为 channel_flow/channel.vtk、channel_flow/channel_GTs.vtk 与 tbl_flow/tbl.vtk、tbl_flow/tbl_GTs.vtk。
完整原始输入 SHA-256 写入构建报告。目标规则网格覆盖完整流场域，不只覆盖 GT 文件包围盒；
由标注原始建议分辨率按完整域长度扩展后上限 200 万体素。记录原始、体素化及流线有效的实例 ID。

每 volume 两类 ordinary 各固定随机抽取最多 8192 点；hairpin 两类全部保留，seed=7068+volume_code。
先冻结抽样，再剔除不能完成积分的 primitive，不根据网络表现筛样；每类每 volume 至少 32 点。
标签图中的非涡区域使用 -1 标记而非第五类，非涡点不参与监督。

每点生成中心线与沿 ±x/±y/±z 偏移的六条线，共七条。双向四阶 Runge–Kutta 单位速度积分，
每方向 16 步、共 33 点；物理步长 h=0.5×目标网格平均间距，偏移/h=0.1501458034894056。
积分先转 q=(x−流场原点)/h，速度除以每 volume 的均方根速度，步长为 1，避免大坐标 float32 相减；
单位速度方向积分消除统一速度倍率。FMT 输入=(q−第一条线首点)。这消除单位和统一倍率差异，
不宣称不同流场的网格间距对应同一物理涡尺度。FMT 特征标准化仅在本次合并拟合样本上计算。

FMT 固定为 6 频、Gram、chirality、sorted neighbors、161 维，neighbor_scale=100，
标准化后 neighbor block 权重 0.5。FMT 无可训练参数。

## 单个共享网络与过拟合判据

仅运行 fmt_only：161→1024→1024→4，LayerNorm/GELU，dropout=0，沿用已验证的
PathlineMulticlassClassifier3D 与旧版 _train_one 训练循环，不改旧冻结入口。
输入不含速度/curl/标注/坐标/volume ID。Adam lr=1e-3、weight decay=0、无 label smoothing，
余弦退火至 1e-6，最多 1000 epoch，batch=512，seed=7068；每 epoch 对全部样本无放回遍历一次。
拟合与评估使用完全相同样本，无 test/validation，不保留模型文件。
连续三个 epoch 合并样本零错误且最小正确类 logit 优势>0才通过；同时报告 channel、TBL 四类混淆矩阵。

审计重新根据角度计算标签与阈值，核对覆盖、cache 索引、输入/标签/预测哈希、标准化、全样本遍历、
合并及逐 volume 指标、连续零错误和 checkpoint 零残留。审计指标从保存预测独立计算；共享历史训练
循环及标准化函数不意味着完全独立重实现端到端 pipeline。

成功只证明有限样本记忆与该数据/训练路径可运行，不能证明泛化、人工解剖真值或 FMT 相对 Raw 优势。

## 代码

- config/Verify_Task4B_VelocityCurlMemorization_4.1.yaml
- FMT_Utils/Task4B_VelocityCurlLabels_3D.py
- experiments/Verify_Task4B_VelocityCurlMemorization.py
- tests/test_task4b_velocity_curl_labels_3d.py
- ibex_bash/verify_task4b_velocity_curl_4p1_{build,gpu}.sh

正式可读报告由 `experiments/Export_Task4B_VelocityCurlReport.py` 从数值预测重算，输出
`four_class_report.json`。复用的历史训练器在原始指标字典中把数字类别3写作 `hairpin_limb`；
本版类别3的定义始终是 `hairpin_leg`，正式报告使用这个名字。保留原始结果文件，不改变任何标签、
预测或数值指标。实验结果与通过状态见 `docs/experiment_log.md`，运行状态见 `docs/ibex_run_registry.md`。

## 0.98a 标签可视化预览（2026-09-06）

用户要求下一版涡区条件采用严格 IVD>0.98a，保留每个流场原有a。当前仅生成完整网格代理真值与三维图片，版本 `Other_Task4B_ProxyGTThreshold_4.2`，配置 `config/Other_Task4B_ProxyGTThreshold_4.2.json`；不运行训练或模型性能实验，不读取4.1预测，不修改4.1的0.9a标签与已完成结果。四类的velocity-curl规则不变，完整GT不受训练抽样或流线有效性过滤。今后若采用该阈值训练，应使用新实验版本。

## IVD>a 标签可视化预览（2026-09-06，取代0.98a作为当前请求）

用户进一步指定 a 为 hairpin 区域的实际最小 IVD，涡区为严格 IVD>a。新标签版本 Other_Task4B_ProxyGTThreshold_4.3 保持此前覆盖标注支持的定义（原始正标注单元中心与目标正标注体素中心的联合），直接取该集合实际最小值，不作nextafter下移。等于a的极小值样本按严格比较排除，不以人工GT强制补回；该边界的覆盖率如实报告。四类方向规则保持不变，仅出完整网格ground truth与三维图，不训练。旧4.1/4.2配置、标签及结果保留。

## 固定物理 IVD 阈值（当前请求：2026-09-06）

用户在其软件检查并咨询CFD专家后指定 Channel IVD>5.785、TBL IVD>0.08，取代基于hairpin最小值的阈值选择。当前标签/可视化版本为 Other_Task4B_ProxyGTThreshold_4.4。阈值直接应用于冻结的全域IVD原值，不作归一化换算；不通过模型性能选阈值。不再要求全部hairpin标注通过涡区阈值，也不以成员标注强制补回低IVD点；报告原始单元与目标网格的各自覆盖率。四类velocity-curl规则不变；无涡区仍为ignore=-1，不新增分类。继续仅出ground truth三维图，不训练；旧版结果保留，未来训练需新版本。
