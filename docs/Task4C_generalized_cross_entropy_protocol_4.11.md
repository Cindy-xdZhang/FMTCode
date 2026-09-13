# Task4-c：广义交叉熵对照 4.11

版本 `Ablation_Task4C_GeneralizedCrossEntropy_4.11`。原固定10,000束测试上两个方法三种子平均Hairpin F1均≥0.6的目标保持；本版仅用训练/验证开发，不自动重测4.6。4.6实际测试FMT0.335526、Conv3D0.304615，目标未达成。

## 研究问题与来源

检验降低低置信度样本的监督梯度贡献能否改善当前几何分类。现有GT的错误率未知，不把低F1、头区标签与中心成员差异或GT外几何案例视为错标证明，也不据此删除、重标样本。

来源：[Zhang and Sabuncu, Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels, NeurIPS 2018](https://papers.nips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf)，式(6)：

\[
L_q(p_y)=(1-p_y^q)/q,\qquad 0<q\le1.
\]

q趋近0时为交叉熵；q=1时为平均绝对误差的比例形式。硬标签单项的梯度相对于交叉熵乘以p_y^q，困难但正确的样本也会被减弱。本文不使用原论文截断损失及样本剔除算法。原论文噪声假设及理论保证不能直接套用于本数据或以下加权/平滑改编。

## 冻结对照与唯一变化

保持4.1原27,000训练、3,000验证与10,000测试、头区/GT实例/源节点空间隔离、全部物理尺度及几何清洗。复制原4.4训练/验证编码，逐文件哈希相同，不加载test。保持4.3分类器：FMT逐线161维局部特征加72维有符号系数、学习聚合，88,514参数；Conv3D四通道24³体素、219,602参数。

保持4.5同中心物理尺度成对训练：学习率0.001，AdamW weight decay0.001，dropout0.15，label smoothing0.02，batch128，200epoch上限、35epoch耐心、5epoch预热及余弦学习率；每epoch27,000前向视图。尺度一致性权重FMT10、Conv1，独立配对随机数与5epoch渐增不变。没有SAM、辅助对比投影器或新增推断视图。

仅将原加权平滑交叉熵替换为：

\[
\mathcal L_q=\frac{\sum_i\sum_{c=0}^1 w_c[(1-\epsilon)\mathbf1(y_i=c)+\epsilon/2](1-p_{ic}^q)/q}{\sum_i w_{y_i}}.
\]

w与epsilon沿用4.5，分母保留PyTorch硬标签加权交叉熵约定。q=0直接调用冻结原损失，保证对照逐值一致；q>0用expm1/log_softmax稳定求值。总损失仍为监督损失加原尺度一致性。q也会改变监督项相对于固定一致性项的强度，因此结果不能单独归因为抗错标能力。

## 搜索、复核与证据

q仅取0、0.3、0.7；首种子96311共六次训练。每模型始终保留q=0和验证F1最好的正q，以96312/96313配对复核，共八次。平分时按验证平均精确率选择；三种子汇总规则同上。阈值、epoch及q只用验证，不依据test调整。全部原样本仍参与原抽样总体。

runner：`experiments/Task4C_GeneralizedCrossEntropy_4_11.py`；配置：`config/Ablation_Task4C_GeneralizedCrossEntropy_4.11.json`；作业脚本：`ibex_bash/task4c_generalized_cross_entropy_4p11.sh`。4.8配置以SHA256固定，仅作为公共设置来源；真正训练复用冻结4.5单步AdamW训练函数。损失回调只在一次调用内替换，异常时也恢复；每个Slurm任务独立进程，历史源码未改。

工程核验集中保存至`outputs/Verify_Task4C_GeneralizedCrossEntropyCode_4.11/checks.json`：独立NumPy损失/有限差分梯度、q极限/非法值/极端logits、两实际模型零q连续三步参数与dropout随机数等价、正q梯度、尺度项不变、回调恢复、test拒绝及真实小缓存14次训练/排名/选择。pilot F1不计科研成绩。临时核验内联执行，不新增一次性test/verify源码。

科学结果与提交身份记入`experiment_log.md`，全部Slurm任务记入`ibex_run_registry.md`。模型只在内存保留；落盘逐次指标、预测、配置和汇总，最终独立复算。详细预测与审计保留在原Ibex目录，当前仅读取获准的汇总和运行元数据。
