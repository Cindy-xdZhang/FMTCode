# Ablation_Task4C_FMTv6v7_4.20

## 用户指定方法与冻结比较

用户在4.19单种子验证集逐块消融后指定两种组合，均重新训练三个原4.14种子。方法选择未使用新的test结果。4.14数据和历史结果不改写；Conv3D沿用4.14原结果，仅V6/V7重跑。

| 方法 | 方法版本 | 实际特征顺序（Python左闭右开） | 维数（不含有效线标记） | 网络参数 |
|---|---|---|---:|---:|
| fmt_V6 | 6.1 | 中心步进[0:23]＋邻居变化[23:161]＋坐标/切向方向[161:233] | 233 | 88,514 |
| fmt_V7 | 7.1 | 中心步进[0:23]＋坐标/切向方向[23:95] | 95 | 70,850 |

保留原特征数组顺序方便逐位复核，用户所列方向72＋中心23＋邻居138的内容完全相同。两版无角度、kin、curl或IVD输入；不宣称客观性。V6就是原4.14的233维组合，不能作为新方法超越原FMT的证据。V7仍共享原束的质心/Rmax归一化，不能解释为完全独立的单线表示。

## 实现与训练

- 编码器：`FMT_Utils/Task4C_FMTv6v7_4_20.py`，提供`fmt_V6`和`fmt_V7`。输入为归一化涡线束、原种子与有效线数，输出每线特征并附最后一维有效线标记。V7公共函数通过原计算后取列；实验直接从冻结缓存取列，不重新积分/编码。
- V6直接复用4.14的完整234列缓存；V7按原列`[0:23] + [161:234]`复制为96列（最后一列为标记）。有效/补零线、逐样本标签及划分不变。原缓存、物理数据和配置全量校验哈希。
- 分类器沿用`Task4C_LinePooling_4_3.LearnedLinePooling`。V6不重新替换首层，确保相同种子初始权重完全一致；V7使用既有`EncoderLinePooling(95)`适配首层，其余隐藏层同种子一致。V7首层采用PyTorch默认初始化，参数量减少17,664；不同于4.19保留398维输入后的置零诊断。
- 原`experiments/Task4C_NearbySampling_4_13.py::train`直接复用，不复制或改写优化循环。27000训练/3000验证/10000测试，种子96611/96612/96613；AdamW初始学习率0.001、weight decay0.0001、dropout0.15、batch128，最多500轮、验证F1/AP早停50轮，学习率按验证交叉熵降低。signed-log和训练集逐维标准化保持。
- 所有训练固定Tesla V100；6个独立训练并行。工程检查只用真实训练样本，检查几何重算、精确取列、有效标记、V6初始化/前向一致及两网梯度。
- 验证集选择权重后固定0.5阈值测试一次；测试为已用过的4.14已知头区局部空间benchmark，不是独立实例确认集。不据test再选特征或超参数。
- 最优权重只存内存，不写模型文件。逐样本train/validation/test预测独立复算F1及Average Precision（按预测排序计算的平均精确率），验证选择与每轮27000样本覆盖独立检查。V6三个新训练均与原4.14逐样本预测比较。

## 记录

科学运行配置：`config/Ablation_Task4C_FMTv6v7_4.20.json`。代码commit、任务ID及设备写入`docs/ibex_run_registry.md`，方法结果写入`docs/experiment_log.md`及`docs/paper_tables_tasks_3d.md`。

## 完成记录

科学commit `63a3cf4dfadfee2a4a71d5edc81ac9a7adb7aa7c`；配置`config/Ablation_Task4C_FMTv6v7_4.20.json`，SHA256 `018454b18df73b98b25b468a9768409f71eb95f58a8967562e313d9c1527dc5d`；summary SHA256 `778ea4f48a9d34f42dfae00e0c329287bbceef0f6aff6cc67d4f2a052a3120e1`。

预检51890379、训练51890400[0–5]、汇总及独立预测复算51890401全部完成；8个Slurm进程、16条生命周期事件核对PASS。6份预测逐样本身份、每轮训练覆盖、F1/Average Precision和三种子汇总独立复算，最大指标误差0。V6三种子逐样本概率与原4.14最大误差0。审计代码commitdb2f5d8f。未保存模型权重；72份结果证据在`outputs/Ablation_Task4C_FMTv6v7_4.20`及个人Ibex `/ibex/user/zhanx0o/FMT_Task4C_FMTv6v7_20260914/outputs/Ablation_Task4C_FMTv6v7_4.20`。无模型证据包SHA256 `0e9185d70504a36eaa2d1c43c159ef3794af2893f1b48a60c8ed54beec818a38`。

完整指标和方法判断见experiment_log的task4c-fmtv6v7-4-20-2026-09-14。
