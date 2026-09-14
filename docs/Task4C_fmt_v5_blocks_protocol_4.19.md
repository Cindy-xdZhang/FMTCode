# Task4-c 4.19：fmt_v5五个特征块的快速消融

用户要求最快判断Task4-c 4.14中fmt_v5五个特征块的重要性。`Ablation_Task4C_FMTv5Blocks_4.19`采用固定seed96611的逐块移除后重新训练，完整398维参照复用已核验的4.16同种子结果；只需五次新训练，不重新编码/采样。旧4.14/4.16结果不改。

| 标识 | 被置零的Python切片 | 特征维数 |
|---|---|---:|
| center_step | [0:23] | 23 |
| neighbor_change | [23:161] | 138 |
| coordinate_tangent | [161:233] | 72 |
| angle_real | [233:323] | 90 |
| angle_imaginary | [323:398] | 75 |

每组只在训练和评价时将指定块的标准化输入置零，保留其余通道。末尾[398]有效线标记始终保持。每组网络仍398输入、109,634训练参数，相同随机初始化；被遮挡列不承载输入信息，不能宣称有效自由度相同。比较衡量给定训练配方下移除一块后的条件贡献，相关/冗余特征可互相替代，单种子不构成稳定普适排序，也不是单个通道的重要性。

全量27,000训练和3,000验证，沿用原signed-log和训练有效线逐通道标准化、clip±8，统计必须与完整参照完全相同。原训练循环逐句保留并以抽象语法树检查相等：AdamW学习率0.001、weight decay0.0001、dropout0.15、batch128、最多500epoch、验证交叉熵平台15轮降半、验证F1/AP无改善50轮停止。固定0.5阈值F1选模型，同分按Average Precision（按召回增量加权的精确率）选择。限定原V100环境。

重要性定义为“完整参照验证F1－去块后验证F1”：正值越大，当前实验中该块越重要；负值表示移除该块后该训练配方的验证表现更好。同时报告训练F1、验证Average Precision及Channel/TBL分别结果。只使用验证集排序，不加载测试特征或新增测试推断，已使用的历史测试结果不用于排序。

完整参照为4.16科学commitdfdc87b1的fmt_v5_4_14 seed96611，验证F1=0.5873015873；不是原233维FMT，也不是三种子均值。原result SHA256 c22a9a1b03e0600d2b55617eb6f6ac45d620a100290dcbbb1934fdd197f9f180。配置固定于`config/Ablation_Task4C_FMTv5Blocks_4.19.json`，入口`experiments/Task4C_FMTv5Blocks_4_19.py`。

预检只用真实训练样本，核验完整包装模型与原初始化/输出一致、五组初始训练参数一致、被移除块的输入扰动不能改变输出、其首层权重梯度为零、mask不变、原优化循环相等、缓存哈希。正式结果逐样本独立复算及每epoch全样本覆盖检查。进程登记提交/开始/结束/节点/GPU/配置和源码哈希，失败保留；不写模型权重、不公开推送。结果和方法判断写experiment_log，指标另记paper_tables_tasks_3d。

## 完成记录

科学commit `c0d33df0021f5bb036f32038e6393f8cbbe08d23`，配置`config/Ablation_Task4C_FMTv5Blocks_4.19.json`，配置SHA256 `2bf2806babfb1f2a0e04c5e4eb7c9fbca226b9b00cb088ee599415922bf853b7`。summary SHA256 `890e4a02b9b6bdb2c4b8d60f9edfa6d2aa56e3bd9c4c704321d1e090b9f1306a`。

预检51889118、训练51889196[0-4]、汇总及独立预测复算51889197全部完成，7个进程与14条开始/结束事件核对PASS。五次训练分别用时262.6/181.6/106.9/173.4/187.6秒（中心/邻居/方向/角度实部/角度虚部），并行执行；选择/停止epoch依次98/148、74/124、10/60、90/140、98/148。全部27,000训练样本逐epoch覆盖、逐样本身份及F1/Average Precision独立复算通过。审计代码commitf02ec804。无模型权重；63份证据文件位于本地`outputs/Ablation_Task4C_FMTv5Blocks_4.19`及个人Ibex `/ibex/user/zhanx0o/FMT_Task4C_FMTv5Blocks_20260914/outputs/Ablation_Task4C_FMTv5Blocks_4.19`；证据包SHA256 `82e981222eef55571bc54f682736024ea571f8d384e712fb68008f1bcf2beb08`。

指标和方法判断见experiment_log的task4c-fmtv5-blocks-4-19-2026-09-14。
