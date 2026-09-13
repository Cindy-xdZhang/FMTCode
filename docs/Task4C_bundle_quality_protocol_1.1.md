# Task4-c：Channel涡线束的四类形态识别

**历史协议**：2026-09-13用户随后明确改为[2.1 Hairpin / Non-hairpin二分类](Task4C_hairpin_binary_protocol_2.1.md)，比较FMT与Conv3D＋MLP。
下文保留1.1定义和当时状态，不再作为当前执行协议。BiLSTM保留但不运行；临时测试按用户要求删除，验证报告保留。

2026-09-13用户授权新任务，版本`mainExp_Task4C_BundleQuality_1.1`。Task4-a/b的部位分类代码与历史结果冻结。
本任务输入一整束涡线，输出Hairpin、Quasi-hairpin、Fragment或Non-hairpin；不输出头/腿逐点分割。
本轮只使用Channel，不训练或评估TBL。Task6/7/8及Task36仍暂停。

## 1. 论文依据与不能混用的定义

用户提供的匿名稿为《A Deep Learning Framework for Hairpin Vortex Identification in Wall-Bounded Turbulent Flows》，
共11页，首页标注Online Submission ID 1017。源文件位于用户指定的Zotero目录；方法依据第3–6页，评价依据第7–8页。
本文没有论文官方源码或原始人工四类标注，不能声称已经复现其报告数字。
稿件SHA256：`3662a1fa9d4e2fd0b7692720bfe21ef568a436469bb554732b11d020fffccdc9`。

双向长短期记忆网络（Bidirectional Long Short-Term Memory，BiLSTM）分别沿点序列、线序列学习表示。
论文沿**涡量场**`ω=curl(u)`双向积分vortex lines；这与沿速度场积分的streamlines不同。
它的bundle最多256条线，每线按弧长重采样32点，也不是Task1–5的中心加六邻居pathline cross。

| 标签 | 论文3.1.3与图3的含义 |
|---|---|
| hairpin | 清晰拱头、明显抬升、紧密且连贯的线束。图3把左右单侧partial hairpin和马蹄形也列入本类 |
| quasi_hairpin | 总体完整且保留弯曲头部，但有轻微不对称、形变或倾斜 |
| fragment | 不完整、衰减、稀疏、不连贯或抬升不足的局部拱形片段；论文把合成reverse hairpin也放入本类 |
| non_hairpin | 一般流向、展向、竖向涡，或随机、严重破碎的非hairpin结构 |

这些是作者人工定义的操作类别，没有给出能自动生成全部标签的几何阈值。
因此用户给出的理想对称hairpin规则只用于解释与检查，不能替代四类人工标注，也不能把不满足对称规则的样本自动判为非hairpin。

## 2. 数据、坐标与当前标签缺口

输入为`channel_flow/channel.vtk`，标注参考为`channel_flow/channel_GTs.vtk`。
原场维度`256×192×256`，范围约`[0,3.129321]×[0,1.171961]×[-1,-0.003080]`。
`x`为主流方向，`y`为展向，`z`为竖向，壁面位于`zmin=-1`。这是局部裁剪数据，本版本不将裁剪边界当周期接缝。

原场含`velocity`、`vorticity`、`vorticity_mag`、`oyf`和`lambda2`。
GT的cell数组为`RegionIds`和`VortexIds`，后者有74个不同编号，包含编号0，没有四类质量字段。
编号0在新任务中不自动当作背景；该检查不追溯修改旧Task4-a/b的73个正编号实例协议。
GT也不能自动区分完整hairpin与quasi-hairpin，GT外区域不能未经标注全部当作non-hairpin。

论文用8个时间快照构造约4万候选，约4600束有人工作业标签，并追加约500个运行阶段候选标签。
我们只有一个Channel局部快照，既无对应四类标签，也无论文修订后的99个Channel等值面GT。
本版评估目标因此限定为**单快照中留出整束、隔离空间重叠组的四分类**，不复现论文跨时间训练或最终曲面检测表。
用户的标签来源选择尚待确认。在标签缺失时，训练入口直接报错，不补造标签或F1。

## 3. 候选束准备

按照论文训练输入的基本逻辑，使用原场`lambda2 < -13.395`与`oyf > 0`寻找头部种子。
`-13.395`来自论文表1的Channel small设置，不沿用Task3的IVD p95或Task4-b的IVD阈值。
`oyf`使用文件已有的展向涡量脉动字段，不擅自在局部裁剪上重算并替换原始平均值。

本地准备版采用6连通头部区域，至少10个头点。固定种子94013在全部合格区域中均匀抽取最多64个候选，每区最多256个种子。
这个上限用于首批标注准备，不是论文的数据量，也不是根据模型性能挑选样本。
在所有顶点均通过λ2阈值的网格单元中，以五阶Runge–Kutta自适应积分（RK45）沿`vorticity`双向追踪。
源域不做周期拼接。双向半线按seed identity合并，并保留沿正涡量方向的点序。

每方向最大弧长为局部壁法向范围的0.5倍，初始步长0.002倍，积分误差参数0.000001倍。
论文没有提供这些具体数值，它们是本地准备默认值。去掉点数≤2或任一坐标轴零方差的轨迹，少于10条有效线的束明确登记为准备失败。
线数不足256时补零，超过时均匀随机子采样并保留原seed顺序。
本版没有实现Zafar的拓扑分割，也没有实现论文3.1.4的signed-swirl头部细分/合并或3.2的闭合曲面构建，不能称为完整端到端复现。

每束所有有效重采样坐标先减全束中心，再除最大中心半径。附加单位前向切向量，末点沿用倒数第二点切向量，得到`256×32×6`输入。
坐标与切向量不含绝对位置和速度幅值。FMT也使用同一归一化坐标，不能靠Channel与TBL的量级差异获取类别信息。

输出集中为`bundles.npz`、`bundle_manifest.json`与`annotations.csv`，不为每个小检查另建脚本。
GT不参与候选选择、曲线积分或网络特征。

## 4. 冻结的论文网络基线

实现`FMT_Utils/Task4C_PaperBaseline_1_1.py`，以论文3.1.2/3.1.3的尺寸与公式为准。
该文件作为本任务冻结基线，后续FMT迭代不得静默改写。
基线文件SHA256：`500ee0d4409fa1897c6fa1fa21b33b5411356ac9acc2160778817ac96514c4c7`。

| 部分 | 实现 |
|---|---|
| 单线编码 | 6维点特征，单层BiLSTM，每方向隐藏64，拼接两方向最终隐藏状态得到128维线表示 |
| 整束编码 | 单层BiLSTM，每方向隐藏128，拼接得到256维，再线性投影到64维潜变量 |
| 解码 | 潜变量投影为256维，沿256线槽重复，整束BiLSTM后沿32点重复，单线BiLSTM和线性输出，Tanh约束6维输出 |
| 无监督损失 | 公式10：0.5倍均方误差加0.5倍六维向量余弦距离；排除补零点 |
| 分类器 | 公式11的**加性余弦间隔**：归一化特征与类别权重，训练时仅真实类别减间隔`[0.3,0.3,0.3,0.1]`，乘30 |
| 监督损失 | 训练集逆类别频数加权交叉熵；推理不减类别间隔 |
| 置信排序 | `p_hairpin + 0.7 p_quasi + 0.4 p_fragment`，不是经过概率校准的hairpin概率 |

稿件图注称ArcFace，但公式11为加性cosine margin，代码遵循公式，不自行加入`acos`再加角度的变换。
论文未披露部分实现细节：本版使用单层、无dropout、固定顺序零填充线槽，不额外引入packed sequence。
“15 epoch warmup”本地解释为至少完成15个微调epoch再允许early stopping，不声称作者采用了相同学习率warmup。

## 5. FMT对照与版本身份

| 方法 | 输入相关操作 | 比较目的 |
|---|---|---|
| paper_bilstm | 原始归一化坐标/切向量，论文两级网络 | 论文架构基线 |
| raw_pca_residual | 相同主干，加每线原始192维输入的训练集主成分分析（Principal Component Analysis，PCA）161维表示，经线性层映射128维并加到单线表示 | 控制FMT分支新增的网络参数 |
| fmt_residual | 相同主干和161→128辅助层，输入换成冻结FMT七线特征 | 判断FMT是否优于同容量原始几何对照 |

FMT适配：每条涡线的头部seed选六个最近的其他seed，组成中心加六邻居小束。
将同一束归一化坐标送入原`DFT_FMT_3D.py::pathline_dft_features_3d`，冻结六频、`mode=gram`、手性项、邻居排序、scale100与weight0.5，得到每线161维。
这沿**弧长采样序号**编码涡线几何，是新Task4-c适配，不冒充原物质轨线在同时间的相对位置，也不宣称任意时变参考系客观。
辅助层初始化为零，初始主干输出与论文基线一致。两辅助臂参数个数、初始化、潜变量和训练预算相同。
PCA与特征标准化只读取训练组，不能用整场、验证或测试数据拟合。

## 6. 标签与拆分

人工填写`annotations.csv`中的`label`、`split`、`annotator`，保留`bundle_id`和`group_id`。
label取第1节四个固定字符串；split为`train`、`validation`或`test`。
使用整束密集积分轨迹的包围盒：若两盒在每个轴上的间隔均不超过该轴两倍最大网格间隔，就连接为同组。
所有直接/间接连接束组成同一组，整组只能属于一个split。
该保守规则可能把当前候选连成很少的组；若不能形成含四类的三个集合，则停止性能实验，追加适当数据或另立协议，不能打散重叠组凑成绩。
预训练也只用train组，验证选epoch，test不参与预训练、PCA、标准化或模型选择。

## 7. 已声明的本地训练补充设置

论文明确的微调设置为AdamW、学习率1e-4、weight decay 1e-4、梯度范数上限1、验证宏平均F1、warmup15与patience50。
宏平均F1（Macro F1）是四类F1的等权平均；它与论文表2基于曲面交并比的检测F1不是同一指标。

论文未给出的预算采用本地固定值：batch8、预训练100epoch/学习率1e-3、微调最多250epoch，三个种子94021/94022/94023。
三臂逐一在同样本上重新预训练和微调。预训练按最低验证重建损失选择，微调按最高验证宏平均F1选择，严格改善才更新，平局保留最早epoch。
每次微调前使用`run_seed+100000`重置分类器初始化，避免辅助层构造消耗随机数造成三臂分类器初值不同。
数据增强按论文类别规则作用于坐标、切向量和seed，之后再计算FMT/PCA：Hairpin采用展向镜像及绕X小角度旋转；
本地固定25%概率执行X反射加绕Y正90度旋转并转为Fragment；Non-hairpin可绕三个轴旋转和反射。
25%、镜像概率及Non-hairpin角度分布是论文未披露的本地默认，不能称为作者原参数。

输出逐次类别F1、宏平均F1、平衡准确率、混淆矩阵、预测概率、训练曲线、配置/源码/输入/标注哈希和设备。
所有权重只在内存中保存，不写`.pt`、`.pth`或`.ckpt`。
当前没有真实四类标签，故尚无任何Task4-c性能结果，不预设FMT胜出。

## 8. 入口与检查

```powershell
& tmp/jhtdb-venv/Scripts/python.exe -m experiments.Task4C_BundleQuality_1_1 inspect
& tmp/jhtdb-venv/Scripts/python.exe -m experiments.Task4C_BundleQuality_1_1 build
& tmp/jhtdb-venv/Scripts/python.exe -m experiments.Task4C_BundleQuality_1_1 review
# 当时的合并测试已执行；后按用户要求删除，日志保留于outputs/Verify_Task4C_Pipeline_1.1/report.json。
# 标注与空间组拆分满足条件后才可执行；入口会拒绝空标签和跨集合重叠。
& tmp/jhtdb-venv/Scripts/python.exe -m experiments.Task4C_BundleQuality_1_1 train
```

本地环境已有PyTorch CPU版本；虽然机器有RTX3090，当前环境不能调用CUDA。此次没有向Ibex提交任何进程。
实现核验另记`Verify_Task4C_Pipeline_1.1`，只能支持代码数值与流程有效，不能作为真实四类性能。
所有方法级观察仅写`experiment_log.md`。

`review`导出单个`bundle_review.vtp`，保留原始物理坐标与每条线的`bundle_index`、`bundle_id`。
用ParaView打开后可按cell字段`bundle_index`着色，通过Threshold选中某一编号并Reset Camera查看；
`bundle_id`对应`annotations.csv`。每线保持正涡量方向的点序，x为主流方向、y为展向、z向上，壁面在z=-1。
此文件仅供查看候选形态，不代表检测结果或人工标注。首批准备上限64，只是流程样例；完整性能实验还需足够的四类样本及独立空间组。
