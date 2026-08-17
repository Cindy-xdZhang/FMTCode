# Code Review 记录（2026-08-16）

**范围**：本仓库从 PyflowVis 复制的全部代码（FLowUtils / FMT_Utils / pnn / DeepUtils / FMT_Clustering.py / FittingVatistasParam.py / VatistasFlowDatasetGenerator.py / config）。
**方法**：8 个独立视角扫描（Task1 主链路逐行、积分与 CUDA 内核、数据加载与判据、Vatistas 标签管线、pnn 与工具、复制边界完整性、跨文件张量契约、清理/效率/规范）→ 约 40 个候选 → 逐条独立验证（含数值实验），结论分 CONFIRMED（确认）/ PLAUSIBLE（成立但依赖触发条件）/ REFUTED（驳回）。
**好消息先说**：四个核心契约逐位验证一致——种子网格 nx/ny 排列、5 线顺序 (center,x+,x−,y+,y−)、flatten=line·L+t 约定、AngleAwareSampling 维度语义；IVD/Q 的 gradient 物理间距与轴序正确；Vatistas 形变正逆互洽、标签与场的仿射一致、t0 恒等成立、随机种子齐全。

## A. 最高优先级 10 条（已通过 ReportFindings 上报，此处为存档）

| # | 位置 | 结论 | 要点 |
|---|---|---|---|
| 1 | `VatistasFlowDatasetGenerator.py:281` | CONFIRMED（数值实验） | 观察者速度变换的平移速率项用 `(−a,−b)`，正确应为 `−Q(t)·(a,b)`（对 `c(t)=Os−Q·p` 求导，交叉项精确抵消）。数值实验：修正式对真值误差 3.4e-8；代码式误差=\|(Q−I)(a,b)\|，观察者采样范围 MC：典型 ~10%、p90 22%、最坏 74% 的伪均匀背景流；"观察者看自己的 Killing 场应为零"残差 0.524；落盘的 (场, 观察者6标量) 互不自洽。t0 切片不受影响 |
| 2 | `FLowUtils/ScalarField2d.py:51,101` | CONFIRMED | 2D IVD 建立在 `np.abs(curl)` 上，算的是 \|\|ω\|−mean\|ω\|\| 而非定义 \|ω−mean(ω)\|；反号涡被折叠、阈值零点偏移。3D 版（ScalarField3d.py:81）是有号的、正确——两版不一致证明 2D 是笔误。**这是 Task1 定量协议计划用的参考标签，必须先修** |
| 3 | `FLowUtils/flowDatasetUtils/NetCDF_AmiraLoader.py:324-343` | CONFIRMED（内存内复现） | 真实 AmiraMesh 的数据段是 `# Data section follows\n@1\n<binary>`，主分支正则不吃 `@1\n`，np.frombuffer 从这 3 个 ASCII 字节开始按 float32 解释，整场错位且 count 校验必然通过。本仓库自写文件走 fallback 恰好正确 → 往返自测掩盖 bug |
| 4 | `FMT_Clustering.py:96` | CONFIRMED | `AngleAwareSampling` 在有效性过滤**之前**对全批调用；未积满的线在零填充段贡献恒定 π/2 转角，污染全局显著性 topk，所有保留 primitive 的采样时刻被垃圾数据决定。修法：先 `valid_index` 过滤再采样 |
| 5 | `FMT_Clustering.py:159-160` | CONFIRMED | encoder 从未 `.eval()`；Pooling 内 BatchNorm 以 train 模式用当前批统计量（`torch.no_grad` 不切换模式），特征依赖同批其他 primitive，running stats 跨组/跨视图漂移；聚类结果依赖遍历顺序与批组成 |
| 6 | `FLowUtils/flowlineIntegral.py:789`（伞形） | CONFIRMED | CPU 与 CUDA 后端语义不一致，同一 config 产出不同数据集：(a) GPU 对 t0 零速度种子早退 valid=1（涡心 primitive 被整组丢弃），CPU 无此判据（涡心变成 300 步全同点的"静止 primitive"进 KMeans）；(b) CPU 先判旧点后无条件写新点，提前终止的线多带一个域外点；(c) GPU 对 t_target 做 `np.clip(tmin,tmax)`，CPU 不裁剪（越界后在冻结末帧场里外推且判满长度）；(d) CPU fallback 把种子降为 float32 累加（GPU 全程 double）；(e) CPU fallback 对 `method="euler"` 因大小写落到 `raise ValueError` 直接崩（当前调用点全为 "rk4"，潜伏） |
| 7 | `NetCDF_AmiraLoader.py:82-99` | CONFIRMED | 分量名候选 `['x','y']` 排在 `['a','b']`/`['Component1','Component2']`/`['velocity_x','velocity_y']` 之前，而能走到这一步的文件必含 x/y 坐标变量 → 用后三种命名的文件把坐标轴当速度读（T≥X==Y 时静默成功且 y 铺错轴；否则 ValueError 被上层 except 吞掉、场无声消失）。全部候选不匹配时 `field=None` 无任何报错（3D 版有 raise，2D 漏了） |
| 8 | `FittingVatistasParam.py:178-184` | CONFIRMED | `zip(field_specs, fields)` 无长度校验，而 loader 加载失败是 continue 不 append（`if vf is None` 是死分支）；Amira 目录分支还会一个 name 追加 N 个场。任何缺失/目录数据集都让 spec（含时间窗）与场错位，meta 标错场名，拟合分布静默污染 |
| 9 | `FittingVatistasParam.py:116-120` | CONFIRMED | `p2n.clamp(max=30)` 使 `2n·ln(r/rc)>30` 后 g 冻结、速度随 r **线性增长**（正确远场 ~1/(2πr)）；边界参数 (rc=0.05,n=6) 下 71.7% 的渲染网格落入线性区，拉回放大后更大。修法：softplus 恒等式 `relu(p2n)+log1p(exp(−|p2n|))`，已验证数值安全 |
| 10 | `requirements_fmt.txt:74` | CONFIRMED | `pointnet2_ops @ file:///C:/Users/xingdi/sources/PyflowVis/...` 指向仓库外绝对路径，换机器 `pip install -r` 必失败；运行期实际不需要它（仅两个死文件 import） |

## B. 其余已验证发现（按主题）

**Task1 主链路（跑 mainExp_1.1 前须知）**
- `FMT_Clustering.py:176`：整片切片当一个 batch，B≈3200 时 stage-2 峰值 ≈6.8 GB，8 GB 卡临界 OOM（PLAUSIBLE，逐张量估算）。`config` 里 `batch_size: 12` 无人读取；`dataset.t_target` 在 timesliceCount=1 时失效；`pathlines.num_cross_points_per_seeding` 全链路无人读取（5 是硬编码）。
- `FMT_Clustering.py:168-170` + `FlowlinePostProcessing.py:49`：`normalizeLines(LocLines(x))` 组合在数学上不成立——对已局部化的位移量再减绝对 domain min 并做各向异性缩放（cylinder2d x/8 vs y/1），送进对平移/缩放敏感的 PosE 后，"4 视图对比"的结论不可解释（CONFIRMED as design flaw）。
- 时间窗守卫缺失：`t_start+dt·max_iterations` 无人校验 ≤ tmax；当前 4 个场恰好放得下（0.4·T ≥ 1.5），改任一参数或换短时间窗场即触发 A6(c) 的后端分叉。建议在 `generate_Flowmap_SLICE` 加显式 assert。
- `FTLE_fitting_utils.py:126`：对带符号 `dx0/dy0` 用 `clamp_min(1e-12)`，邻序颠倒或零填充时产出 ~1e12 的假 Jacobian 而非报错（CONFIRMED，当前调用序恰好为正）。
- `FTLE_fitting_utils.py:435`：`compute_ivd_2D` 未导入 → `generate_IVD_SLICE` 一执行即 NameError（CONFIRMED；IVD 评测路径端到端断裂，连同 §A2 一起修）。
- `FTLE_fitting_utils.py:99-100`：组内 5 线等长检查被注释掉（`same` 计算后丢弃），混长 group 的尾点零填充会进 FTLE 差分（CONFIRMED，函数当前无调用者）。

**数据加载/缓存**
- `ScalarField2d.py:313-319`：ScalarFieldManager 单例缓存键只有 (name, op)，不含分辨率/时间/场身份；`resample2UnsteadyField` 原地改场后同名请求拿陈旧结果（PLAUSIBLE，当前零调用点）。
- `ScalarField2d.py:435-575`：`load_scalar_field_from_file` 声明返回 ScalarField2D 实返 tuple，两处消费方按对象用（潜伏 AttributeError）；`build_scalar_filename` 调不存在的 `self._sanitize_name`（传 extra_tags 即崩）（CONFIRMED）。
- `AnalyticalFlowCreator.py:59`：`'local_dict' not in locals()` 在"x 为 callable、y 为字符串"混用时把 t 冻结在第一帧（CONFIRMED，当前 4 个内置场不触发）。
- `stable_hash.py` + `FTLE_fitting_utils` 缓存键：键漏 `dat_dir` 与积分方法；ndarray 落到截断 repr（大数组必碰撞）；set 迭代序不稳定（PLAUSIBLE）。`EasyConfig`：空 yaml → DispatchError；与 dict 方法同名的键（hash/update/get…）被方法遮蔽（PLAUSIBLE）。

**Vatistas 管线（除 A1/A8/A9 外）**
- `FittingVatistasParam.py:609-616`：间距判据用未形变 rc（应为 max(sx,sy)·rc），m=2 场中"判据通过但形变椭圆重叠"占 9.6%~71%（依分布）；50 次重试耗尽后静默 append（CONFIRMED）。
- 涡核可整体落在渲染域外（bounds t=1.5 vs 域 [-1,1]²）：标签按定义是"正确的空"，但构成未声明的额外负样本，占比 0.7%~7.2%（PLAUSIBLE，定性经对抗验证修正）。
- θ 统计：模型对 θ→θ+π 规范对称 → 拟合分布双峰；圆均值配线性 std 采样 ≈ 均匀取向（CONFIRMED as statistics flaw）。
- 旋转角积分是一阶右矩形和（漏 i=0 项恰好保住 t0 恒等），与 RK4 世界线、解析 Q̇ 阶数不一致，t=1 处 ~0.75° 系统偏差（CONFIRMED，低危）。
- 标签哨兵值不一致（正常 rc−r / Killing 恒 −1.0 / 无 profile −1e9）；`observer.make_labels` 读入后未用；默认 unsteady_subset=400 只覆盖 20 个 steady 场。

**pnn / 工具 / 效率**
- `pnn/models/point_nn.py:111,135`：`.cuda()` 硬编码；与 `FMT_encoder.py` 的 device 无关版**已漂移**（同源四份拷贝：point_nn / point_nn_seg / point_pn / FMT_encoder，前三份带 `.cuda()`，其中两份是死文件）。model_zoo 从 point_nn 导入 EncNPNew，但 model_zoo 本身当前无调用方 → PLAUSIBLE（一旦在 CPU/非 0 号卡用 model_zoo 即崩）。
- `model_zoo.py:1002+`：`build_model` 需要 `config.pcds/pnn/lowResX/lowResY/dataset.UPsampling`，本仓库任何 yaml 都没有 → 调用必 AttributeError；`PathlineFMTclustering.yaml` 的 `model.NAME` 是无人消费的误导残留（CONFIRMED）。
- `pnn/libs/flows.py:393-540`：`multi_points_vis_fast` 的 `nerbos` 参数纯摆设，线数由 `max_step` 反推；当前调用传 max_step=299 但走 4 维分支被忽略（侥幸）；3 维分支在 P%L==0 且 L≠真实线长时会静默取错点+颜色错位；姊妹函数 `multi_lines_vis` 的第 5 条线从不绘制（作者注释自认）（CONFIRMED，当前实参组合是响亮报错而非静默错画）。
- `pnn/libs/parallel_flows.py:153-154`：`gen_starts` 从 3 元边界解包 2 值 → full/areas 分支必崩；`:65-69` nearest 时间插值按种子物化整帧场 `[M,H,W,2]`（full 采样下 TB 级，靠调小 batch 掩盖）；空返回形状声明 max_steps+1 与实际 max_steps 差一位（CONFIRMED，该路径当前未被 Task1 使用）。
- `pnn/models/point_nn.py:255`：`Point_NN.__init__` 引用被整段注释的 `EncNP` → 实例化即 NameError（CONFIRMED，死链）。
- 效率/重复（CONFIRMED）：`_tiling_starts` 五份拷贝三种语义（建议保留 FTLE_fitting_utils 版提为公共函数）；model_zoo 三个上采样类的滑窗索引每 forward 重建+逐窗 H2D（应 `register_buffer` 预建）；`cache_coordGrid` 非 buffer 每步 H2D 拷贝且 `repeat` 应为 `expand`；`point_nn_seg.py`+`point_pn.py` 共 535 行死代码；`debug_checks` 五个检查四个无调用点、`_LEVEL=1` 硬编码使 `FMT_DEBUG` 失效（实际代价落在数据集生成的 `check_ftle_field` 全量扫描，不可关闭）。

## C. 仓库外的重要发现（PyflowVis 的 C++，影响已发表数据集，需另行数值确认）

审查 §A1 时核对了 NumPy 移植声明的参照物 `PyflowVis/CppProjects/src/transformation.cpp`：

1. `killingABCtransformation`（transformation.cpp:211-237，世界线版）：与 NumPy 移植同公式同错（`translationTdot={−a,−b}`）；另有 line 224 的 `u` 计算后未使用（死变量）。**该函数的调用已被注释，未用于生成论文数据集。**
2. **生成论文 60k 数据集的是 `Tobias_ObserverTransformation`**（flowGenerator.cpp 5 处调用）。其平移项自洽（Ṫ 恰为其自身积分的 ċ），但旋转项 `Q_dot = Q_t·[[0,ω],[−ω,0]]`（transformation.cpp:314-319）对它自己的 `Q_t=R(θ)` 约定**符号反了**（正确为 `Q·[[0,−ω],[ω,0]]=Ṙ`）。笔算判例：静止流体 + 纯旋转观察者，真观测场为 `+ωJx`，该公式给 `−ωJx`（旋向相反）。后果：生成场 = 真观测 − 2ωJ(x−c(t))，**不是任何单一刚体观察下的推前**。
3. **谨慎定性（未验证的推断，明确标注）**：多出的项是空间刚体旋转场，对 IVD 这类"减去均值涡量"的客观判据不改变结构位置，t0 的 Vatistas 参数标签也在 t0 恒等下定义——所以**论文标签本身可能幸存**；但"60k 场是 steady 场的刚体观察"这一客观性论证不再严格成立，pathline 几何与记录的观察者参数互不自洽。**建议用 20 行 numpy 数值实验确认符号问题后，再决定是否影响任何已发表结论**（本仓库 review 不下这个结论）。

## D. 对本仓库既有文档的修正

`first_principles_analysis.md` P7 原写"`debug_checks._LEVEL=1` 硬编码……**且每个 train step 强制一次 GPU→CPU 同步**"。修正：后半句在本仓库不成立——`check_train_step` 在本仓库无调用点（该说法源自 PyflowVis 的 FTLE_experiment.py 训练循环，未复制）；实际代价位于数据集生成路径的 `check_ftle_field`（每张 FTLE 切片 ~6 遍全量扫描且无法用环境变量关闭）。P7 表已同步更新。

## F. 修复状态（2026-08-17，commit 944d206）

按"只修有实锤且修法无歧义"的原则，以下发现已修复并带回归测试（全部通过）：

| 发现 | 修复与证据 |
|---|---|
| A1 观察者平移速率项 | 改为对 c(t)=Os−Q·p 的精确链式求导（任意 center 成立）。测试：观察者看自身 Killing 场残差 0.56 → **6.7e-16**；pathline 等变性终点误差 0.2+ → **0.0012**（tests/test_observer_transform.py） |
| A2 IVD 用 \|ω\| | 新增 `compute_vorticity_2D`（有号），IVD 改用之；curl_magnitude 语义不变。测试：反向旋转双半平面 IVD 中位数 ~0 → **1.000**；刚体旋转 IVD=0（tests/test_labels_and_loaders.py） |
| A3 Amira 3 字节错位 | 正则消费 `@N` 行；真实格式与自写格式均逐位往返（同上测试） |
| A4 采样先于过滤 | FMT_Clustering 改为先 `keep_full` 过滤再 AngleAwareSampling |
| A6 CPU/CUDA 五处分歧 | kernel 移除 t0 零速度早退（GPU 涡心 primitive 不再被丢，valid 1 → 100）；CPU 先验新点再记录、float64、方法名大小写不敏感；t_target 双后端统一裁剪+响亮警告；.cu 路径改为模块相对。冒烟：GPU/CPU **valid_steps 完全一致，端点差 6e-8**（tests/test_integrator_and_utils.py + parity smoke） |
| A7 NetCDF ['x','y'] 回退 | 移除该候选（在本 loader 中必然只会命中坐标轴）；全不匹配改为 raise |
| A8 zip 错位 | extract_patches 对 loader/spec 长度不一致 fail fast |
| A9 p2n clamp | 改稳定 softplus。测试：远场衰减比 1/(2πr) 的比值 **1.000**（此前线性增长） |
| A10 requirements 本地路径 | 已删除该行（运行期已不需要 pointnet2_ops） |
| B 组已修 | FTLE 基线符号保持（交换 x± 后 FTLE 仍= ln2，此前 ~1e12 垃圾）；`compute_ivd_2D` 导入；间距判据椭圆化+耗尽告警；AnalyticalFlowCreator 逐帧重建 eval 字典；pnn PosE 设备无关；gen_starts 解包；stable_hash ndarray/set；EasyConfig 空 yaml；debug_checks 恢复 FMT_DEBUG；ScalarField2d `_sanitize_name`/tuple 消费点；yaml 残留 model 块删除 |

**刻意未修**（非"百分百无歧义"）：A5 缺 `.eval()`（eval vs train 模式是 mainExp_1.1 协议要两臂实测的科学决策）；B 组的 normalize∘LocLines 组合、整片 batch 显存、θ 圆统计、涡核出域构成、缓存键弱化（零调用）、multi_lines_vis 第 5 线、死代码/效率项；C 组 C++ 侧问题（另行验证）。

## E. 建议修复顺序

1. **先修标签与数据正确性**（否则 mainExp_1.1 的定量协议无地基）：A2 IVD 有号化（对齐 3D 版）+ `compute_ivd_2D` 导入；A1 观察者平移项 `−Q(a,b)`；A9 p2n softplus；A8 zip 长度断言；B 组的 enforce_spacing 椭圆化判据。
2. **再修 Task1 主链路确定性**：A4 先过滤后采样；A5 显式 `.eval()`（或显式标准化，二选一入协议）；时间窗 assert；分块前向控显存。
3. **再修加载器**：A3 Amira 偏移（消费 `@1\n` 行）；A7 候选表把 `['x','y']` 移除或置末 + 全不匹配时 raise。
4. **环境与卫生**：A10 删 requirements 该行；`.cuda()` → device 参数并统一四份拷贝为一份；清死代码与重复实现（B 组效率条目）。
5. C 组 C++ 问题在 PyflowVis 侧另行验证与决策（不在本仓库范围）。
