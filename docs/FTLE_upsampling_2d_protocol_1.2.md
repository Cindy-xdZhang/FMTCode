# Other_FTLEUpsampling2D_1.2

2026-09-14：用户要求增加 dropout 正则化，严格在未见 timeslice 上测量指标，增加8×。
本版本比较固定 dropout 概率0与0.15；不搜索概率，不根据测试结果修改训练规则。
此前1.1的源码、配置、数据和结果冻结。本协议继承
`docs/FTLE_upsampling_2d_protocol_1.1.md`，以下为明确新增或变更项。

## 数据与未见时间的含义

沿用1.1四个流场、原始源文件、12/3/3个训练/验证/测试起始时刻及完整积分窗。
每个窗口长1.0；训练、验证、测试的所有窗口两两分离，连分界端点也不共享。
`FMT_Utils/FTLE_Temporal_Audit_2D.py::certify` 读取真实时间坐标，逐窗枚举速度插值
所需的首尾源帧及中间帧，再检查跨集合源帧索引交集为空；不能只按播种时刻去重。
正式数据审计逐文件核对保存时间数组、manifest起始时刻、积分终点和split身份。
每次训练启动前重新核对时间证书，失败即停止，不进入训练。

测试切片从未参与本版本模型拟合、归一化、学习率调整或epoch选择。它们已经用于
1.1评价，因此仍是重复使用的benchmark，不称作此前无人评估的新确认集。
最初工作更新中“此前未评估过”措辞已明确纠正，不能通过轻微移动起始时刻伪造新的独立时间块。
同一split内部窗口可重叠，空间区域共享；不声称独立物理实例或未见流场泛化。
Cylinder仍按原始模拟区间[0,15]计算50%且t≥7，下限7.5，不重新截半。

## 方法版本和正则化

Dropout2d指训练时以固定概率随机置零整个特征通道，并缩放保留通道；验证和测试关闭。
实现参考[PyTorch Dropout2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html)。
冻结基线类`FMT_Utils/FTLE_Baselines_2D.py`不改；1.2在独立入口显式增加dropout前向运算。

| 方法版本 | 技术和位置 | 主代码 | 结果 |
|---|---|---|---|
| ESPCN SR 1.2 p0/p15 | 仅低分辨率FTLE；两次隐藏卷积ReLU之后各一次Dropout2d | `FTLE_Upsampling_2D_1_2.py::Model` | 待完成审计 |
| U-Net SR 1.2 p0/p15 | 五个DoubleConv模块输出后各一次Dropout2d，包括三层编码和两层解码；跳跃连接使用同一已dropout的编码输出 | 同上 | 待完成审计 |
| Raw/FMT/objective2.2/FMTv5 + U-Net 1.2 p0/p15 | 与普通U-Net相同位置和概率；32通道几何投影与输入、输出头均不额外dropout | 同上 | 待完成审计 |
| bilinear/bicubic 1.2 | 固定节点坐标插值；相同评价区域；无训练或dropout | `FTLE_Data_2D.py::interpolation` | 待完成审计 |

ESPCN为Efficient Sub-Pixel Convolutional Neural Network，通过低分辨率卷积和像素重排上采样。
U-Net是有跳跃连接的编码、解码卷积网络。本次保留PyFlowVis改编网络的参数形状。
四种几何输入仍为320/65/176/131维；primitive严格为中心、±x、±y五线，32时刻，时间不是第三空间坐标。
原始FMT的65维在FMTv5中逐项保留，只追加六个中心夹角的66维频谱。距离＋夹角采用冻结2.2二维适配。
所有几何方法仍共用相同U-Net、320→32投影和参数总量。同种子同参数初始化；dropout不增加参数。

## 倍率与统一指标区域

倍率为2、4、8。高网格保持65×513、385×129、129×385、129×257；低网格取高网格
`[::scale,::scale]`，两端节点严格重合。8×时分别为9×65、49×17、17×49、17×33。
训练patch仍覆盖64个高网格间隔，即65×65节点；8×输入为9×9节点，能通过两层下采样。

每个切片先按原有效轨迹规则生成各倍率掩码，再取2/4/8×的交集。
该切片全部方法、两个dropout概率、三个倍率使用完全相同的高网格评价像素。
掩码仅来自物理有效性与固定空间支撑规则，不用FTLE数值或误差选择区域。
结构相似性（Structural Similarity Index，SSIM）仍要求完整7×7邻域有效。
掩码与1.1不同，因此新无dropout基线也全部重新训练和评价，不将1.1数字当成配对对照。
跨倍率变化和dropout收益以本版本内部配对结果判断。

## 训练、选择与证据

四流场×三倍率×六学习方法×两个dropout概率×三种子97111/97112/97113，共432次训练。
每个流场/倍率/种子的12次训练在同一GPU进程顺序运行，保证概率对照和方法对照共享设备。
AdamW学习率0.001，weight decay 1e-6，batch 8，最多100epoch；验证MSE最低选模型，
连续15epoch不改善停止，ReduceLROnPlateau patience5、factor0.5。其余1.1设置不变。
MSE是Mean Squared Error，即平方误差均值。训练数据单独拟合全部均值、标准差和PSNR数据范围。

选择完成后、首次加载测试数据之前写入`selection.json`：最佳epoch、验证MSE、完整归一化量、
训练/验证文件清单、配置哈希、data manifest哈希、history哈希及锁定时间。
`result.json`保存首次测试读取时间及selection哈希。此后不再运行优化、调度或选择。
审计核对每份预测split身份、原真值和掩码，独立重算指标及全部汇总。
主表仅报告完整留出测试切片；validation指标只作选择证据，不混入测试均值。
权重只在内存保存，禁止落盘checkpoint或下载模型。

报告均方根误差（Root Mean Squared Error，RMSE，FTLE物理单位）、平均绝对误差
（Mean Absolute Error，MAE）、峰值信噪比（Peak Signal-to-Noise Ratio，PSNR，固定训练范围下越高越好）、SSIM。
先等权平均三个测试切片，再报告三个种子的均值与总体标准差；四流场宏平均在每个种子内等权计算。
同时报告0.15减0的配对变化，包含改善和变差情形，不预设dropout或FMT必须胜出。

## 执行入口

`python -m experiments.FTLE_Upsampling_2D_1_2 submit --config config/Other_FTLEUpsampling2D_1.2.json`

依赖链：preflight → prepare[4] → audit-data → train[36] → summarize → audit-results，
共44个Slurm进程。每个提交立即登记；结束后逐项核对实际作业ID、节点、GPU、状态和退出码。
独立预检覆盖旧基线哈希、0概率与冻结网络逐元素相同、eval关闭dropout、train随机性、
全部倍率输出大小和反向梯度、五线几何、故意重叠时间窗及共享源帧的拒绝检查。

科学提交、验证、运行和最终结果在完成后补入专门验证记录；方法结论仅写入`docs/experiment_log.md`。
