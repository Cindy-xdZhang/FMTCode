# 第一性原理分析：原有 FMT 实现的重大问题（2026-08-16）

> 逐条给出：问题、证据（代码位置）、为什么重要、建议。凡属推断而非实证之处，均以"我认为/存疑"标出。
> 排序按"对研究结论的威胁程度"，不按发现顺序。

## P0. 先把词定义清楚：什么叫"客观（objective）特征"

流体可视化里的客观性有精确定义：设欧氏参考系变换 `x* = Q(t)·x + c(t)`（Q(t) 为随时间变化的旋转，c(t) 为随时间变化的平移），一个量若在所有此类变换下不变（标量）或按张量规则协变，才叫 objective。以此为尺子逐一检查现有 pipeline 的每个环节：

| 环节 | 在变换下的行为 | 客观？ |
|---|---|---|
| 原始坐标 (x,y,t) | 直接被 Q(t), c(t) 改写 | ✗（连伽利略不变都没有） |
| `normalizeLines`（domain min-max 归一化） | 域框本身不随观察者变，但轨线形状已变 | ✗ |
| `LocLines`（减种子点） | 只消掉**常数**平移；时变 c(t) 会改变轨线形状本身 | ✗ |
| PosE 正弦嵌入 sin/cos(β·x/α^k) | 对坐标的任何等距变换都改变输出 | ✗ |
| 同时刻跨线取差 d_ij(t) = x_i(t) − x_j(t)（原 GeoLinePicker，现 `group_same_timestep`） | c(t) 被消掉，d* = Q(t)·d，**范数‖d‖不变** | 差向量协变；**‖d‖ 严格客观** ✓ |
| LGA 里 diff/std 归一化后的差向量 | 仍随 Q(t) 旋转 | ✗ |
| `DCT_FMT` 的 \|FFT(x+iy)\| | 对**常数**旋转/平移/时移不变；对时变 Q(t) 不变 ✗ | 部分（比 PosE 强得多） |
| cross primitive 的有限差分 Jacobian ∇Φ 的奇异值（Cauchy-Green 特征值 → FTLE） | σ_i(∇Φ) 在两端各乘正交阵下不变 | **✓ 严格客观** |
| IVD（瞬时涡量偏差） | Haller 定义即客观 | ✓ |

**由此得出本次重启最重要的两个第一性原理结论：**

1. **cross primitive 天生就是 flow map 梯度 ∇Φ 的有限差分模板**：(x+ 线 − x− 线)/2ε 和 (y+ 线 − y− 线)/2ε 在每个时刻 t 给出 ∇Φ_{t0→t}(seed) 的两列。对它做 SVD，奇异值 σ1(t), σ2(t) 的整条时间曲线是**严格客观的标量信号**；σ1·σ2 是局部面积变化，σ1/σ2 是剪切/拉伸各向异性，(log σ1)/(t−t0) 就是 FTLE。旋转性信息可用同样客观的量补充：相对位移向量 d(t) 的**转角增量**（d(t) 与 d(t+dt) 的夹角——两个同时刻差向量的夹角在 Q(t) 下不变……严格说夹角跨了两个时刻，仅在 Q(t) 缓变时近似客观；严格客观的旋转量应当用 IVD 沿线采样，或用相对速度对 d 的分解）。**一个"客观的 FMT"应当以这些量为原子，而不是以裸坐标为原子。**（此为设计判断，待实验验证。）
2. 旧实现声称 objective，实际客观性从未被架构保证，也从未被测试过。VortexTransformer 论文的客观性是**数据侧**给的（Killing 观察者增强）。FMT 若要把"training-free 且 objective"当卖点，必须有**不变性单元测试**：用 `FLowUtils/KillingObserver2D.py` 生成随机时变刚体观察者，变换场→重积分 pathline→重编码，断言特征漂移 < 容差。这个测试现在写就能写，不依赖任何新方法。

## P1. "training-free / 无参数"目前名不副实（最危险的表述问题）

**证据**：
- `FMT_Utils/FMT_encoder.py:139` `Pooling` 含 `nn.BatchNorm1d(out_dim)`——有可学习 affine 参数 + running statistics。聚类用的 `FMT`（stages=2）每个 stage 都过它。
- `FMT_encoder.py:238` `TemporalDFT` 有 `weight_real/weight_imag` 可学习复数滤波 + BatchNorm（聚类实验 `temporal_head=None` 未用到，但它在"FMT"这个名字底下）。
- 真正零参数的只有：强制 stages=0 的 `FTLEUpsamplingFMT_Unet`、以及 `DCT_FMT`。

**更隐蔽的一层**：`FMT_Clustering.py` 构建 encoder 后**从未调用 `.eval()`**，且推理在 `torch.no_grad()` 下逐 (场,时刻) 组整批前向。BatchNorm 在 train 模式下用的是**当前 batch 的统计量**——即每个 primitive 的特征依赖于同批次其他 primitive。后果：
1. 同一条 primitive 单独编码 vs 随全场编码，特征**不同**；
2. 换 batch 组成（比如换 grid_sampling）结果漂移；
3. 我认为这次"聚类成功"可能部分**依赖**这个意外行为——train 模式 BN 恰好等价于对该时间片全部 primitive 做逐通道标准化，这正是 KMeans 需要的特征白化。若切到 eval 模式（未训练的 BN ≈ 恒等），特征分布会变，聚类结果可能变差。**这是"成功"最需要复核的一根柱子**（存疑，待 mainExp 复跑验证）。

**建议**：重启版把归一化从"隐式 BN"改成**显式、确定性的逐时间片特征标准化**（或干脆去掉，把尺度处理放进特征定义），使"无参数"字面成立，且单样本编码 = 批量编码。

## P2. DCT_FMT 只取正频率：顺时针涡被系统性错编码（实打实的 bug）

**证据链**：`DCT_utils.py:70` 用 `torch.fft.fft`（完整复谱，排列为 [0, +1, …, +N/2, −N/2+1, …, −1]）；`DCT_FMT_encoder.py:115/124` 取 `mag[:, :dct_k]`——即 **DC + 前几个正频率**。对复信号 z(t)=x+iy，逆时针圆周运动能量在正频率，**顺时针在负频率**。取前 k 个 bin 意味着：逆时针涡特征强烈、顺时针涡几乎全零（只剩泄漏）。而冯·卡门涡街的脱落涡**自旋方向交替**——一半的涡会被编码成"接近无旋转"，和层流区难以区分。
另外：类名叫 DCT 实为 FFT，`dct_1d/idct_1d` 是死代码；`neighbor_diff_scale=100.0` 与 `dct_weight=0.5` 是无物理依据的魔数（应改用 `2·offset_dist` 归一化）；`dct_k` 超过 L−1 时被静默 clamp（L=4 时特征从 30 维悄悄变 15 维）。
**建议**：同时取正负频率末端 `|Z[1..k]|` 与 `|Z[N−k..N−1]|`（保旋向可分性），或报告 |Z[+k]|±|Z[−k]| 的对称/反对称组合（反对称部分就是净旋向强度）。3D 推广时此构造（z=x+iy）失效，需另行设计——暂按用户指示不展开。

**状态（2026-08-16）：已修复**（commit bd55d45）。`dft_complex_lowfreq_mag` 按 `[|Z[0]|, |Z[+1]|, |Z[−1]|, …]` 成对保留正负频率；修复前顺/逆时针刚体旋转 primitive 的特征范数为 9.64 vs 116.93（比值 0.082），修复后两者相等（117.09），旋向仍可区分。测试见 `tests/test_dct_fmt.py`（含常数旋转/平移不变性）。命名问题（名为 DCT 实为 FFT）保留未改，已在 docstring 注明。

## P3. Task1 的"成功"没有定量地基

- `FMT_Clustering.py` 的评估 = 4 种输入视图 KMeans(k=2) 的并排可视化，人眼判断。无 ARI（Adjusted Rand Index，调整兰德指数）、无 NMI（Normalized Mutual Information，归一化互信息）、无对客观判据标签的 F1/IoU。
- 唯一的定量 A/B 框架 `compare_DCT_FMT_vs_FMT.py` 比的是**另一个任务**（FTLE 超分 PSNR），且证据显示 FMT/DCT 两臂**从未跑通**（无图、无日志、断言与 config 冲突、import 已断）。
- 旧评测函数还有硬 bug：`test.py:62` 只评了每样本第 0 条线。
- **可用的 GT 其实现成**：(a) Vatistas 合成场自带解析涡核标签（`VatistasFlowDatasetGenerator.py`，已复制）；(b) 真实场可用 IVD/Q 判据阈值当参考标签（`FLowUtils/ScalarField2d.py` 的向量化 `compute_ivd_2D` 等；注意 IVD 本身客观，适合做"客观涡"的参照）。
**建议**：mainExp_1.x 的定义里必须包含固定的定量协议：每场每时间片报 ARI/NMI + 对 IVD 阈值标签的 F1，Vatistas 合成场报对解析标签的 F1；聚类先做特征标准化，KMeans 固定 seed 多次取稳。

## P4. 采样与数据构造存在系统性偏差

1. **涡心被剔除**：CUDA 积分核对种子处 `|v|<1e-9` 直接 `valid_steps=1` 提前返回（`assets/cuda_kernal/PathlineIntegration2D.cu:103` 一带），下游又要求 5 条线全部积满 max_steps 才保留 primitive——**恰好把驻点/涡心（零速度）的 primitive 系统性丢掉**。对"区分涡/非涡"的任务这是朝着任务要害的采样偏差。
2. **边界/快速区丢失**：任何一条线出域即整个 primitive 弃用 → 标签图有洞，且丢弃概率与流速相关（非随机缺失）。
3. **AngleAwareSampling 的三个问题**（`FlowlinePostProcessing.py:91`）：(a) 显著性对全部 primitive 和线取平均（docstring 说"组内共享"，实际**全批共享**），逐 primitive 的自适应性为零，对比 Regular 等间距的优势存疑；(b) 在 `FMTClusteringDataset` 里它在**过滤无效 primitive 之前**调用——未积满的线在零填充段产生恒定 π/2 转角，污染全局显著性排序；(c) 选出的时间索引非均匀且随 (场,时刻) 变化——喂给任何按索引做 DFT 的模块（TemporalDFT / DCT_FMT）时，频率轴的物理含义已经丢失，不同切片的特征也不再同分布。
4. **只积分前向**：backward 分支被注释（`flowlineIntegral.py:842`）。吸引型结构（后向 FTLE 脊）天然看不见。
5. `t_target` 被静默 clip 到 `[tmin, tmax]`（`flowlineIntegral.py:582` 一带）：时间窗设置越界时不报错，全场线提前终止 → "FTLE 全 0 / primitive 全弃"的历史事故源头。

## P5. 尺度与量纲从未被处理（PosE 对此极度敏感）

(x, y, t) 三个通道量纲不同（cylinder2d 空间跨度 ~O(1)、时间窗 ~O(1) 但单位无关联），却共用同一组 PosE 超参 alpha=1000, beta=19——这组数是 Point-NN 为归一化到 [-1,1]³ 的 ModelNet 点云调的。beta·x 进 sin/cos 后，尺度差直接决定各通道的有效频率覆盖，等于给三个轴随意加权。4 视图对比（原始/归一化/局部化/两者）实质上是在黑盒试探这个问题，但从未从量纲角度正面解决。
**建议**：特征定义处显式无量纲化：空间用 offset_dist 或局部弧长归一，时间用积分窗长归一；PosE 若保留，按无量纲输入重推 alpha/beta 的合理范围（可解析算出频率覆盖）。

## P6. HierachyFMT 的多感受野设计在算法复杂度上不可行（所以从未真正用过）

`GeoLinePicker` 把"邻居"定义为窗口内**全部** P=M·5 条线的同时刻点，构造 `[B, N, P, C]`（N=M·5·L）的邻接张量——内存 O(M²·L·C)，对 M 二次爆炸。receptive_field=16（M=256, L=30, C=96 一档）单窗口就要 ~10¹⁰ 级别元素。这就是 config 里 `receptive_fields:[1]` 的真实原因：**层级设计存在但被内存锁死在"1 个 primitive"档**，等价于没有层级。
**建议**：跨 primitive 的空间聚合不要在点级做全连接邻接；先做 per-primitive token（5 条线内 O(25·L)），再在 token 网格上做卷积/池化——复杂度回到线性。

**状态（2026-08-16）：`HierachyFMT_encoder` 与 `GeoLinePicker` 已按用户决定移除**（commit ca653fd）。FMT 改为内联的 `group_same_timestep`（纯 reshape/expand），与旧实现在 eval/train × 有/无 DFT head 四种模式下**逐位一致**（max|diff|=0，等价性脚本 + `tests/test_fmt_encoder.py`）。同时清理了 `buld_FMT_encoder`、`kNN`/`knn_point` 等死代码与 yaml 里的 `receptive_fields` 残留。未来若需要多感受野，按上面建议在 token 网格层面重新设计。

## P7. 工程腐化清单（会直接浪费下一次实验时间的）

| 位置 | 问题 | 状态 |
|---|---|---|
| `FMT_Clustering.py:9` | import 的 `generate_FLowMap_SLICE` 已被改名 → ImportError | **本仓库已修** |
| `FMT_Utils/vortexExtraction_utiles.py` | 整个文件**没有任何 import 语句**，引用的 loader 函数也不存在；且其 IVD 标签用逐切片 50 分位阈值——每张切片恒有 50% 像素是"涡"，物理语义不成立 | 保留原样，仅作参考；重写前不可用 |
| `FTLE_fitting_utils.py:435` | `compute_ivd_2D` 未 import → `generate_IVD_SLICE` 必崩 | **已修**（944d206；且 IVD 本体已改为有号涡量，见 code_review §F） |
| `FTLE_fitting_utils.py:896` | `temporal_downsamplePathlineCrossPrimitive` 不存在（只有 …Regular/…Random）→ `PointWiseFTLETrainDataset` 必崩 | 未修（属失败 SR 线） |
| `FTLE_fitting_utils.py:126` | 对**带符号**的 dx0 用 `clamp_min(1e-12)`——若邻居次序颠倒，Jacobian 爆到 1e12 且无声 | **已修**（944d206，符号保持；交换 x± 后 FTLE 精确不变，有测试） |
| `FTLE_fitting_utils.py:245` | 低分辨率 linspace 网格不是高分辨率网格子集（`flowmap_sr.py` 的 `hi[::k,::k]` 才是对的） | 未修 |
| `debug_checks.py:29` | `_LEVEL=1` 硬编码，`FMT_DEBUG` 环境变量失效。〔修正 2026-08-16：本行原写"且每个 train step 强制一次 GPU→CPU 同步"——该说法在本仓库不成立，`check_train_step` 在本仓库无调用点（原说法来自 PyflowVis 的 FTLE_experiment.py 训练循环，未复制）；实际代价在数据集生成路径的 `check_ftle_field`（每张 FTLE 切片约 6 遍全量扫描且无法关闭）。详见 docs/code_review_2026-08-16.md §D〕 | **已修**（944d206，恢复 `FMT_DEBUG` 环境变量，默认关闭） |
| `pnn/models/point_nn.py:111,135` | `torch.arange(...).cuda()` 硬编码，CPU 上崩；`embed_dim % 6 != 0` 时静默错维 | `.cuda()` **已修**（944d206，设备无关，有 CPU 测试）；整除假设仍在 |
| `FLowUtils/VectorField2d.py:250,545`、`VectorField3d.py:194,209` | `SteadyVectorField2D.get_vector`、`UnsteadyVectorField2DTrainable.__init__`、3D 两个构造/切片方法均为必崩代码（参数错位/公式错） | 未修，3D 重启前必须先修 |
| `FLowUtils/KillingObserver2D.py` | 瞬时角速度符号疑与场定义相反（`0.5(du/dy−dv/dx)` vs +c 逆时针），做不变性测试前需先用 `constant_rotation` 解析场校准符号 | 待验证 |
| requirements_fmt.txt | 原为 UTF-16LE，pip 读不了 | **本仓库已转 UTF-8** |
| `pointnet2_ops` 本地路径安装 | 换机器必断；实际上当前代码已不需要它（import 被注释） | **已修**（944d206，已从 requirements_fmt.txt 移除） |

## P8. 方法论问题：结论没有载体

失败的 SR 线留下的唯一"结论"是一条 commit message（27dB vs bilinear 39dB）；聚类线的"成功"没有数字；DCT vs FMT 的对比从未跑完。对照 INR 压缩线（实验编号 + spec/results 双文档 + job id + 记分板），差距是纪律性的，不是能力性的。本仓库以 `docs/experiment_log.md` 的版本表为唯一结论载体，规则见该文件。

## P9. 重启的最小可信路径（建议，非结论）

1. **mainExp_1.1**：复跑修复后的旧聚类管线（本仓库现状 = 旧法冻结版），但补上定量协议（P3）与确定性（固定 seed、显式 eval/标准化二选一并记录）。它是 baseline，好坏都要先钉死。
2. **Verify_objectivity_1.1**：对 mainExp_1.1 的特征做 Killing 观察者不变性测试（P0），量化"现在到底有多不客观"。
3. **mainExp_2.x**：客观原子特征版 FMT——对每个 primitive 输出 [σ1(t), σ2(t), ‖d_i(t)‖, 旋向谱(P2 修正版), IVD 沿线采样] 的无参数编码，再聚类。与 1.1 同协议对比。
4. Task2/Task3 在 1.x/2.x 钉死之后再动。
