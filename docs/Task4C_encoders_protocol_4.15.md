# Task4-c 4.15：在4.14数据上比较纯编码器与Conv3D

2026-09-14用户要求将`fmt_v5`和`objective_fmt_nTDO_v2`用于Task4-c 4.14，与Conv3D比较。实验版本`Verify_Task4C_Encoders_4.15`；保留原4.14数据、代码和结果，不重新采样或改变积分长度。方法结论仅记`experiment_log.md`。

## 方法身份

用户在提交前明确选择刚开发的距离+夹角版2.2，实际名称为`fmt_objective_ntod_v2`。冻结纯距离版2.1名称为`objective_fmt_nTDO_v2`，本次不运行，二者不可混称。

| 实验输入 | 代码 | 每条线特征维数，不含mask | 分类器参数数 |
|---|---|---:|---:|
| fmt_v5 5.1 | FMT_Utils/fmt_v5.py | 326：原161维+同采样序号的15个中心夹角六频率165维 | 100,418，预检实测确认 |
| fmt_objective_ntod_v2 2.2 | FMT_Utils/fmt_objective_ntod_v2.py | 396：21对距离231维+15个中心夹角165维 | 109,378，预检实测确认 |
| Conv3D | FMT_Utils/Task4C_LinePooling_4_3.py::WiderConv3D | 4通道24³体素 | 219,602，冻结 |

每条有效线作为局部中心，以播种点距离选择六条最近的有效邻居；完整沿用4.3/4.14的稳定排序。每条线32个弧长重采样点，最多27条线、至少10条有效线；完整原线簇保持，不丢弃不利样本。逐线调用独立编码器，额外一维仅作有效线mask。编码器不含可训练参数，不调用kin4/kin6、速度梯度/curl/IVD估计，也不拼接4.14旧FMT所用72维坐标/切向方向频谱。该原方向分支属于几何特征，不应与IVD估计混同。

两个新编码器使用原4.3逐线全连接映射、masked mean/max聚合及分类头，除首层输入维数外拓扑和隐藏宽度保持。参数量因输入维数不同而不同，逐次记录，不宣称与Conv3D参数量相同。Conv3D按原结构、输入和三种子重新训练；原4.14 FMT（161+72维）和Conv3D成绩仅另列历史参照，不能将旧FMT简写为纯161维。

此处曲线沿涡量场积分，是静态快照的涡线；采样序号对应归一化弧长，不是物理时间。距离编码器的数学不变性核验保持七线对应关系及采样序号固定；不将静态涡线编码实验描述为任意时变观察者下重新播种、积分的客观性证明。全束归一化、近邻选择及采样操作与独立编码器分别说明。

## 数据与训练

只读复用个人Ibex的`FMT_Task4C_PhysicalLength_20260914/outputs/mainExp_Task4C_PhysicalLength_4.14`。父科学commit`5c9c904228d5185447a084953b2652a8938c0485`，父配置SHA256`3909eaab4093c0cb85f9bbd400fcf81e9dc036c879df7b8b9f6f214b6a281e6d`，父summary SHA256`7104bcbf34e637a6462c5eb77936e7b37f6731f2bd13368d0eed45b9d82b9889`。逐文件复核实际geometry、seeds、metadata和Conv体素。两流场Channel/TBL各13,500训练、1,500验证、5,000测试，合计27,000/3,000/10,000；标签、线簇顺序、尺度组合完全相同。

共享候选头区、GT实例和源快照，评价中心距训练中心规则仍为4.14的1h至同头区4h范围；这是已用benchmark上的比较，不是独立涡实例确认。原始涡量及lambda2用于4.14的既定播种/积分/标签流程，模型仅接收保存几何的编码或体素；禁止将“没有IVD输入分支”扩展为“数据构建从未用物理量”。

三种子96611/96612/96613；dropout0.15、weight decay0.0001，AdamW初始学习率0.001，batch128、最多500epoch，验证交叉熵15epoch平台后学习率降半，50epoch无改进停止。普通未加权交叉熵，每epoch全部训练样本一次；直接调用冻结`Task4C_NearbySampling_4_13.py::train`，不复制或修改训练算法。FMT输入沿用signed-log、训练集有效线通道标准化、截断至±8；不得用验证或测试拟合归一化。

验证固定0.5阈值F1选择epoch，同分以Average Precision（平均精确率，按召回率增量加权的精确率）排序；选择锁写入后测试一次。主表固定0.5阈值，验证选阈值仅补充。报告全部九次及三种子均值/样本标准差、合并和分流场成绩，不按测试选择方法参数或重试种子。旧4.14测试已被使用，此次不称为新封存测试。

## 实现、核验与部署

新适配器`FMT_Utils/Task4C_Encoders_4_15.py`，入口`experiments/Task4C_Encoders_4_15.py`，独立复算`experiments/Audit_Task4C_Encoders_4_15.py`，配置`config/Verify_Task4C_Encoders_4.15.json`。

先用两个流场各64个训练样本验证原161维前缀逐位一致、直接编码输出、padding、距离标量在固定对应关系下的不变性、原Conv输入与初始化；从这些训练样本划分工程小集合，仅运行两轮以验证三个完整训练/选择/推断流程，不计科研成绩。全量编码前核对所有父数据哈希，随后九次训练、汇总、独立预测复算与调度核对。逐进程记录提交、开始、结束、节点、GPU、配置及源码哈希，失败不删除。使用逐事件独立文件及带文件锁的汇总记录。

不写模型权重文件；仅内存保存最佳epoch。预测、JSON/CSV及运行证据保留于个人Ibex，新实验不推送到公开GitHub。方法代码和冻结配置本地commit确定，不覆盖父实验。

首次工程预检51886086在体素重算的逐位相等检查失败，正式编码/训练尚未提交。原CUDA体素构造含scatter-add累加，重算结果不要求与float16缓存逐位相同；改为记录全部差异、要求占据mask完全一致且最大绝对误差不超过[-1,1]内一个float16间隔2^-11。模型实际输入始终直接读取原缓存，文件哈希要求完全相同，工程训练也使用缓存；未改变体素表示、数据或网络。失败记录保留，重新预检确认数值差异量级。

## 完成记录

九次正式训练、汇总与独立预测复算全部通过；F1和Average Precision的复算最大差异0。调度及逐事件核对PASS：13个成功进程、1个保留的早期失败预检，28条开始/结束记录完整；81行逐次指标和九份预测保留，无模型权重文件。summary SHA256 `a5a11208d800dca0b40297ae4a5da4007a3d2df00baae9859a5404bbb6783a5f`；证据包SHA256 `5b2b4f0cf423b1a70b55c570fda99016c261d7bbf7f5da6ca1ac529a3ae3c3fc`。本地`outputs/Verify_Task4C_Encoders_4.15`，完整源输入及结果在个人Ibex `/ibex/user/zhanx0o/FMT_Task4C_Encoders_20260914/outputs/Verify_Task4C_Encoders_4.15`。独立数值复算`experiments/Audit_Task4C_Encoders_4_15.py`；调度复核`experiments/Audit_Task4C_EncodersRuntime_4_15.py`，后者提交99546af9。

冻结三种子训练全部完成；科学配置没有按测试更改。指标见paper_tables_tasks_3d，方法结论仅见experiment_log的task4c-encoders-4-15-2026-09-14。
