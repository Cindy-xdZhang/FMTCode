# Other_FTLEUpsampling2D_1.1

2026-09-14：用户要求移植 PyFlowVis 可用 FTLE 超分辨代码，明确二维每个时刻只有
中心、x+、x−、y+、y−五个对应物质点；后续确认采用当前 Fourier 方法体系，
三种编码器共用 U-Net 解码器，加入 Raw 路径线对照。

## 研究问题与冻结来源

输入低分辨率有限时间李雅普诺夫指数（Finite-Time Lyapunov Exponent，FTLE，
衡量有限时间内粒子分离速率），恢复同一时间窗的高分辨率 FTLE。
几何分支额外输入同一低分辨率格点的五条路径线。直接监督 FTLE，
不将 flow-map 坐标误差冒充 FTLE 超分辨误差。

PyFlowVis 源 commit：`3040ace90b34483fd80387166f17d97305d7ae45`。
本地继承的 `FMT_Utils/model_zoo.py` 中 `DoubleConv/Down/Up/ESPCN_SR/UNet_SR`
类主体原样提取到 `FMT_Utils/FTLE_Baselines_2D.py`；哈希见
`config/ftle_2d_baseline_provenance.json`。旧模块、旧配置和历史结果均冻结。
旧 `FTLE_experiment.py` 的周期性测试选模型、分别生成不重合网格、无效路径线置零
后作为真值的执行方式不复用。新入口独立版本化，修复项不追溯改写历史数字。

论文依据：

- [Shi et al., ESPCN, 2016](https://arxiv.org/abs/1609.05158)：
  Efficient Sub-Pixel Convolutional Neural Network，通过低分辨率卷积和像素重排上采样。
- [Ronneberger et al., U-Net, 2015](https://arxiv.org/abs/1505.04597)：
  带跳跃连接的下采样、上采样网络。这里使用 PyFlowVis 的两层下采样超分辨改编版，
  不称作原医学分割网络的逐项复现。
- [Jakob et al., Neural Flow Map Interpolation, 2020](https://doi.org/10.1109/TVCG.2020.3028947)：
  流映射是空间节点采样；这里保留其低、高网格对应原则，但目标改为 FTLE，不能直接比较论文 flow-map PSNR。

## 方法版本

| 方法 | 固定输入和技术 | 主代码 | 结果 |
|---|---|---|---|
| ESPCN 1.1 | 低分辨率 FTLE，3×3卷积128/128通道，像素重排；来自 PyFlowVis 配方 | `FTLE_Baselines_2D.py::ESPCN_SR` | 已完成，见`paper_tables_ftle_2d.md` |
| U-Net 1.1 | 低分辨率 FTLE；基础宽32，两层下采样及跳跃连接，像素重排 | `FTLE_Baselines_2D.py::UNet_SR` | 已完成，见`paper_tables_ftle_2d.md` |
| Raw+U-Net 1.1 | 五线32时刻；中心减其起点，四邻居减同时刻中心，320维 | `FTLE_Encoders_2D.py::encode` | 已完成，见`paper_tables_ftle_2d.md` |
| fmt 2D adapter 1.1 | 冻结二维 `DCT_FMT` 的中心运动及四邻居相对运动差分，正负6对频率及DC，65维 | 同上及`DCT_FMT_encoder.py` | 已完成，见`paper_tables_ftle_2d.md` |
| fmt_objective_ntod_v2 2.2, 2D adapter 1.1 | 10对同时间点距 + 6个同时间中心夹角，各保留6个实系数、5个虚系数，176维 | `FTLE_Encoders_2D.py` | 已完成，见`paper_tables_ftle_2d.md` |
| fmt_v5 5.1, 2D adapter 1.1 | 原65维FMT逐项保留，仅追加66维夹角频谱，131维 | 同上 | 已完成，见`paper_tables_ftle_2d.md` |
| bilinear / bicubic | 在真实低网格节点坐标上插值，非图像半像素坐标 | `FTLE_Data_2D.py::interpolation` | 已完成，见`paper_tables_ftle_2d.md` |

二维原谱采用本仓库已修复正、负频率的复数 x+iy 变换；不是把二维路径线送入三维
Gram 不变量，也不把 (x,y,t) 中的 t 当作 z。`DCT_FMT` 历史类名虽含 DCT，
实际使用离散 Fourier 变换。单 primitive 窗口的 max+mean 聚合及其2倍因子原样保留。
没有额外运动学、涡量或真实梯度输入。FMT 不整体宣称对任意时变刚体观察者客观；
距离＋夹角输入在对应点身份和时间固定时为客观标量。U-Net 的空间网格操作不因此
自动具有任意观察者下的整体客观性。

四种几何方法均在每个低网格节点编码，无8×8空间降采样。输入标准化后补零到320维，
统一可训练1×1卷积映射到32通道，再与一通道FTLE拼接进入同一个U-Net。
四种几何方法的参数张量形状、总参数量及同种子初始化一致；补零列不携带额外信息，
实际活跃输入列数仍不同。普通 U-Net 没有几何投影层，额外输入收益以Raw对照判断。
整个网络都不使用测试数据估计均值、标准差或动态范围。

## 数据与时间划分

本轮选择 PyFlowVis 旧超分配置中的四个流场：cylinder2d、boussinesq、
pipedcylinder2d、doublegyre2d。前三者读取已有 NetCDF；doublegyre 使用旧
`double_gyre_2D` 相同解析公式（eps=.25、A=.1、omega=.2π），直接求速度，
并记录这一区别，避免把粗64帧插值误差混入积分。
旧配置另有beads/rfc，本轮不纳入，不声称覆盖全部历史数据。

每个流场单独训练一个模型；三种子97111/97112/97113；2×、4×两个倍率；
四流场×两倍率×六学习方法×三种子=144次训练。每流场12个训练、3个验证、
3个测试起始时刻。同一split内部窗口可以重叠，不能把这些时刻视作独立流场。

一般流场从原时段10%开始；名称含cylinder的从原时段50%且t≥7开始。
剩余时段按60%/20%/20%分成完整积分块。分界两侧各留2.1个原始源帧间隔，
每个种子时刻必须使整个1.0物理时间窗落在自身块内。训练、验证、测试
不共享轨迹积分时间窗或源时间插值支撑；空间区域共享。因此只评价同流场时间留出，
不宣称未见流场或未见物理实例泛化。实际时刻、源文件哈希逐流场记录。

高网格形状分别为65×513、385×129、129×385、129×257（顺序为y,x）；
低网格严格取高网格`[::scale,::scale]`，覆盖相同端点。网络像素重排后裁掉末端
多出的scale−1行/列，输出精确为`((Hlow−1)scale+1,(Wlow−1)scale+1)`。
插值同样按节点坐标计算，绝不使用错误的半像素插值作为有利对照。

## 积分与真值

每中心固定偏移0.005物理单位，五线顺序为中心、x+、x−、y+、y−；32个等间距
时间样本，含首尾。四阶Runge–Kutta积分（RK4，每一步计算四次速度）每个采样间隔
细分到步长≤0.005，因此当前实际217步，dt=1/217，积分时长严格1.0。
数值速度场用原始空间节点和时间的三线性插值，积分及距离/角度计算使用float64。
每个RK4中间阶段均检查空间边界及源文件给出的圆形障碍物；任一线失败即整束无效。

FTLE=`log(sigma_max(J))/|tau|`，J为首尾两对相反邻居确定的二维流映射导数。
真实tau来自积分配置与保存时刻；不把采样数当作物理时间，不把负FTLE强制改为0。
高分辨率路径线只在数据生成过程中形成真值；缓存仅保留低分辨率路径线。
高、低网格共享相同FTLE公式、偏移、物理时长；各倍率输入值严格等于高网格对应节点。

训练只用全部有效的65×65高分辨率patch，步幅32个高网格间隔，完整遍历全部patch。
验证/测试在整幅场推理：统一有效高网格掩码，并剔除距低网格无效点两格内的支撑。
无效位置输入填0，但不作为0真值参与损失或评价。所有方法共享掩码与样本。

## 训练、指标与审计

AdamW学习率0.001、weight decay 1e-6、batch 8；最多100epoch。
ReduceLROnPlateau仅按验证均方误差降学习率（patience5，factor.5）；
连续15epoch无改善停止。最小整场验证均方误差选模型，然后读取测试一次。
不按测试调整网络、epoch、阈值、掩码、数据范围或超参数。

FTLE物理单位的均方根误差（Root Mean Squared Error，RMSE）和平均绝对误差
（Mean Absolute Error，MAE）为主要误差指标；峰值信噪比
（Peak Signal-to-Noise Ratio，PSNR）使用同流场同倍率训练真值max−min作为固定动态范围；
结构相似性（Structural Similarity，SSIM）使用7×7均匀窗与样本协方差，并剔除窗内无效点。
逐slice先计算，三slice等权平均，再报告三种子均值和总体标准差。
不同流场分开报告，不用大网格的更多像素主导平均。插值只报一次确定性结果。

预检包括冻结网络类哈希、5/7线拒绝、时间列忽略、65维原FMT保留、176维标量
对时变刚体变换不变、解析拉伸的FTLE、完整时长、出界排除、节点一致性、六模型优化
检查，以及四个真实训练窗口的小样本核验。数据审计独立用二维Cauchy–Green
矩阵最大特征值公式复算低FTLE并验证点顺序、特征、掩码、划分和文件哈希。
结果审计从保存的逐slice预测独立复算RMSE/MAE/PSNR/SSIM并复核最佳epoch。

模型权重只在单个进程内存中保存，禁止写入或下载checkpoint。长期结果包括
源数据元信息、完整config、科学commit、每次训练history、逐slice预测、指标、
汇总与审计。全部Ibex调度任务提交即登记，开始/结束的独立runtime文件记录节点和GPU；
失败、取消、超时保留。科学commit与部署哈希在运行记录中补齐。

## 入口

`python -m experiments.FTLE_Upsampling_2D_1_1 submit --config config/Other_FTLEUpsampling2D_1.1.json`

依赖链为preflight → prepare[4] → audit-data → train[24，每项6方法] → summarize → audit-results。
初始本地预检使用`tmp/ftle_2d_local_preflight.json`，仅覆盖数据根目录和输出位置。
方法级结论仅写入`docs/experiment_log.md`；本协议表记录版本与结果指针。

## 完成记录（2026-09-14）

科学commit`9943fcfe`，144次训练、864份预测、960行逐次指标、64行分流场汇总均已完成并复算PASS。32个调度进程全部COMPLETED/0:0，64条开始/结束事件、配置与科学源文件哈希一致；源结果压缩包SHA256为`f845b08d6e8effc2c1312daf32e462de707310a330e59d15bc81bfd5465d188e`。无checkpoint文件。正式时间范围17:24:12–17:41:22 Asia/Riyadh。

训练设备为108次V100、36次A100；同一流场/倍率/种子的六方法在同一GPU上顺序执行。跨种子的标准差也包含设备差异，不能将小均值差自动称为显著改善。

完整表见[FTLE结果表](paper_tables_ftle_2d.md)，方法判断见[实验记录](experiment_log.md#ftle-upsampling-2d-results-2026-09-14)。结果目录`outputs/Other_FTLEUpsampling2D_1.1`；报告生成入口`experiments/Report_FTLE_Upsampling_2D_1_1.py`。正式完成后未追加调参。
