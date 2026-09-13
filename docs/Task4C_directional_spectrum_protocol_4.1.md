# Verify_Task4C_DirectionalSpectrum_4.1

2026-09-13，当前多尺度主实验的十个原FMT候选验证F1为0.285–0.306；早期正则化Conv3D验证F1约0.42。这不足以达到用户目标。下面只检查训练/验证中的表示问题，不访问测试，不替代原FMT主结果。

代码检查事实：冻结 `dft_rotation_invariants_3d` 保留复向量实/虚部范数、夹角和相邻频率手性，但不保留整体空间方向。用户定义的wall-bounded流具有固定streamwise/spanwise/vertical轴。**待检验假设**是保留方向有助于当前hairpin标签分类；现有差距本身不能证明这一假设。

数据完全复用 `mainExp_Task4C_Multiscale_4.1` 的27,000拟合和3,000验证样本，几何、标签、头区、尺度和空间隔离均不改变。编码和训练入口显式拒绝test。
输入是已经整束质心/最大半径归一化的32点曲线。每线沿弧长序列执行正交归一化实数DFT。在线之间分别聚合mean/std/max/min，掩码排除补零。

| 版本/表示 | 每线频率信息 | 输出维数 | 实现 |
|---|---|---:|---|
| gram6 control | 6频；冻结Gram与手性公式，输入改为本协议相同坐标序列 | 92 | `spectrum_features(kind="gram")` |
| signed6 | 6频复系数的有符号xyz实部/虚部 | 144 | `spectrum_features(kind="signed")` |
| signed17 | 全17个独立频率的有符号xyz实部/虚部 | 408 | 同上 |

gram6与signed6使用相同坐标序列/频率数/聚合规则，构成保留方向信息的对照；signed17另检查保留完整频率。参数量随输入维数变化，分别报告。整个新系列相对原322维FMT还改变了输入差分与聚合，不能把它与原FMT的差异全部归因于方向，不追溯改名原算法。

每表示四个候选：lr1e−4/3e−4 × dropout0.3/0.5；对应weight decay1e−3/1e−2，label smoothing0.02。seed94611。MLP宽度256/128，LayerNorm/GELU，训练均匀抽头区、正类权重、signed-log及train-only标准化/±8截断、batch128、200epoch上限、5epoch预热、余弦学习率、35epoch早停、验证F1/阈值选择均与主实验新网络相同。

唯一科学入口 `experiments/Task4C_DirectionalSpectrum_4_1.py`，配置 `config/Verify_Task4C_DirectionalSpectrum_4.1.json`。三个表示的掩码、线顺序置换、旋转响应和真实小规模两epoch训练核验通过，test访问被显式拒绝。工程核验证据 `outputs/Verify_Task4C_DirectionalCode_4.1/checks.json`，以内联命令完成，没有临时测试源码。

Ibex encode → train[12] → merge，最多4 GPU并发，每作业10分钟。所有提交和实际设备登记。只保存配置、清单、训练历史和验证概率，不写模型权重。本轮即使验证达到0.6也不自动启动测试；先将所得方法以新主实验协议固定，再与Conv3D共同进行预定最终评估。
