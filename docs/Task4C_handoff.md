# Task4-c Hairpin 分类：最新交接

更新：2026-09-17。本文核对实际代码、原始VTK及逐样本预测；本次不修改数据或模型，不启动新训练。

**当前数据为 `Ablation_Task4C_BottomDensity_1.2`：193,000训练 / 3,000验证 / 10,000测试。当前验证选出的最佳FMT为c156，最终三种子测试F1为0.920390±0.003032。** 本文替换此前“v9.15_v2正在构建”的过时状态；旧版本的协议、数据和结果仍保留，不能混用于当前比较。

## 1. 任务

输入一簇三维涡线几何（vortex-line bundle），输出Hairpin（正类1）或Non-hairpin（负类0）。Channel与湍流边界层（Turbulent Boundary Layer，TBL）各只有一个速度场快照，沿涡量 `curl(velocity)` 双向积分生成曲线；不是沿速度积分的streamline，也不是时间序列pathline。

监督来自已有Hairpin体积标注与候选头区的重叠规则，没有原论文人工四类标签。评价对象是**线簇二分类**，不是全场分割或涡核线检测。网络使用几何及其派生特征；真实速度、涡量、GT和实例编号不作为网络特征输入。FMT选邻居另外使用播种点几何。

## 2. 原始VTK：文件、数组与标签查询

本机数据根目录：`C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D`。

Ibex根目录：`/ibex/user/zhanx0o/FMT_Task4B_VelocityCurl_4p1_20260906/repo/inputs`。

| 流场 | 速度场相对路径 | GT相对路径 | 网格X×Y×Z |
|---|---|---|---|
| Channel | `channel_flow/channel.vtk` | `channel_flow/channel_GTs.vtk` | 256×192×256 |
| TBL | `tbl_flow/tbl.vtk` | `tbl_flow/tbl_GTs.vtk` | 276×251×224 |

坐标：x为流向（环境流沿+x），y为展向，z为竖直，壁面在z最小处。实际Channel范围约为x∈[0,3.1293]、y∈[0,1.1720]、z∈[-1,-0.00308]；TBL为x∈[271.29,351.65]、y∈[164.14,193.45]、z∈[0.00358,26.488]。两者尺度不同，不能共用积分步长。

- 流场是 `vtkStructuredGrid`，`velocity`、`lambda2`、`oyf`均在 **PointData**。`oyf`是已有展向涡量脉动，不重新按裁剪区域求均值。Channel另有`vorticity`；TBL原始流场没有它，构建代码对velocity按原生网格间距有限差分求curl。
- GT是独立的 `vtkUnstructuredGrid`。实例编号取 **CellData `VortexIds`**，不是PointData或`RegionIds`。Channel有74个原始实例，TBL有58个，编号不连续。**实例0有效**；查询点在GT体积外时，函数返回−1。
- 使用 `vtkStaticCellLocator.FindCell` 查询体积包含，不能用最近GT点或实例bounding box代替。查询坐标必须是原始物理坐标。

以下代码在仓库根目录运行，依赖NumPy、VTK和项目模块。改`name`可读取另一个流场；在Ibex上修改`root`。

<!-- handoff-vtk-example -->
```python
from pathlib import Path
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset, sample_gt

root = Path("C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D")
name = "channel"  # or "tbl"
flow = read_dataset(root / f"{name}_flow/{name}.vtk")
gt = read_dataset(root / f"{name}_flow/{name}_GTs.vtk")
dims = [0, 0, 0]
flow.GetDimensions(dims)
nx, ny, nz = dims
xyz = vtk_to_numpy(flow.GetPoints().GetData()).reshape(nz, ny, nx, 3)
velocity = vtk_to_numpy(flow.GetPointData().GetArray("velocity")).reshape(nz, ny, nx, 3)
lambda2 = vtk_to_numpy(flow.GetPointData().GetArray("lambda2")).reshape(nz, ny, nx)
oyf = vtk_to_numpy(flow.GetPointData().GetArray("oyf")).reshape(nz, ny, nx)
gt_cell_ids = vtk_to_numpy(gt.GetCellData().GetArray("VortexIds"))

# Query one GT cell center and one point outside the GT bounds.
cell = gt.GetCell(0)
inside_xyz = np.mean([gt.GetPoint(cell.GetPointId(i))
                     for i in range(cell.GetNumberOfPoints())], axis=0)
outside_xyz = np.array(gt.GetBounds())[1::2] + 1.0
instance_ids, locator = sample_gt(gt, np.stack([inside_xyz, outside_xyz]))
assert instance_ids[0] == int(gt_cell_ids[0]) and instance_ids[1] == -1
```

VTK点数组按x最快变化，NumPy空间数组按 **`[z,y,x]`** reshape。上例查询的是点的实例归属，**不能直接把中心点归属作为当前整束标签**。实际构建加载器为 `FMT_Utils/Task4C_PaperBundles_3_1.py::load_flow`。

## 3. 当前冻结数据的构造规则

配置：[Ablation_Task4C_BottomDensity_1.2.json](../config/Ablation_Task4C_BottomDensity_1.2.json)。科学commit `a7643890d22c7faf63c1f716bb6222ba5271c5ff`；配置SHA256 `c92b892ed1db9e6e01d64f4eea7792e616e5ab6f0b4a7436b3b456739be44363`。四个原始VTK哈希也登记在该配置。

**候选及标签。** Channel的lambda2阈值为−13.395，TBL为−0.0272。网格单元八顶点全部低于阈值，且八顶点`oyf`均值>0；实际播种点插值后的`oyf`也须>0。将这些单元按六邻接连接，少于10个单元的头区剔除。查询完整头区所有单元中心的GT归属：同一实例覆盖≥50%则为Hairpin；完全没有GT覆盖则为Non-hairpin；其余部分重叠头区剔除。同头区线簇继承该标签。**当前为恢复后的完整头区规则，不是曾尝试的局部线束多数标签。** 负样本也来自涡候选区，不从全场空白区域随意采样。

**划分。** 每个头区的单元中心沿第一主轴排序为五块，固定随机种子分配三块训练、一块验证、一块测试。原评价中心距任何训练中心至少1个当地网格尺度h，距同头区训练中心不超过4h；测试中心距验证中心至少0.5h。h为当地三轴网格间距的几何平均。新增训练中心距原评价中心也须至少对应的1h。

**中心到线簇。** 在所属集合的候选单元内部25%–75%范围随机播种中心，用3×3×3立方偏移产生最多27个种子；偏移间距为0.25h、0.5h或1h。种子须在同一候选头区且插值`oyf>0`。三个集合使用相同模板和清洗。邻居及积分线可以跨播种块边界；原生流场支撑、头区和实例共享，**完整实例留出数为0**。这里不检验未见实例或未见流场泛化。

**积分与清洗。** 沿完整原生涡量场做五阶Runge–Kutta–Fehlberg（RK45）双向积分。lambda2/oyf只限制播种，不将曲线裁到GT、头区或播种块内。实际固定参数：

| 流场 | ds / 单向最大步数 | 单向目标长度ds×步数 |
|---|---|---|
| Channel | 0.001/80；0.002/50；0.004/30 | 0.08；0.10；0.12 |
| TBL | 0.025/160；0.05/100；0.04/150 | 4；5；6 |

**以上为单向长度，双向全线目标长度约为其两倍。** 每流场三个邻距×三组积分参数，共九种尺度组合，三个集合一致。这是恢复4.14后底部加密、五倍扩展实际使用的设置；旧v2的TBL 3–5.5长度扩展不属于当前数据。

剔除点数≤2、非有限值、任一坐标轴跨度为0的曲线，以及任一半线实际弧长不在目标[95%,100.2%]内的曲线。每条有效线按弧长均匀重采样32点，整束少于10条有效线则剔除。按整束有效点质心/最大欧氏半径归一化，再补零到27条并保存有效线数。原始中心线也可能被清洗去掉；FMT后续让所有剩余有效线分别作为中心线。

**底部加密与五倍扩展。** 从4.14原27,000训练束出发，给训练中已有正类实例各加100束（Channel62个、TBL54个），得到38,600束；新增中心轮流从该实例正类头区训练单元的完整池、z较低一半池采样。再对38,600束逐束保留原样，在同流场、头区、类别、实例、尺度和完整/下部采样池中另取四个不同中心，重新积分、清洗，得到193,000束。不是复制数组，没有增加积分长度种类。62/54是通过原规则后的训练正类实例数，不等于原始GT总数；不保证全部原始GT实例覆盖。验证/测试文件始终逐字节保留，没有重新划分或扩充。

| 流场 | 训练：总数（正/负） | 验证：总数（正/负） | 测试：总数（正/负） |
|---|---:|---:|---:|
| Channel | 98,500（44,410/54,090） | 1,500（267/1,233） | 5,000（1,075/3,925） |
| TBL | 94,500（36,685/57,815） | 1,500（182/1,318） | 5,000（692/4,308） |
| 合计 | **193,000（81,095/111,905）** | **3,000（449/2,551）** | **10,000（1,767/8,233）** |

主要实现：`experiments/Task4C_PhysicalLength_4_14.py::build_catalog/trace_lines/trace_batch`；`FMT_Utils/Task4C_Multiscale_4_1.py::center_and_neighbors`；`FMT_Utils/Task4C_BottomDensity_1_1.py`和`Task4C_BottomDensity_1_2.py`。协议：[底部加密1.1](Task4C_bottom_density_protocol_1.1.md)、[五倍扩展1.2](Task4C_bottom_density_protocol_1.2.md)。配置继承的`integration_domain`字符串仍含旧`disjoint_partition`字样；实际4.14传入完整原生涡量网格，不能据旧字符串声称曲线空间隔离。

## 4. 新模型直接加载的缓存

优先复用冻结缓存，避免重新积分改变比较数据：

```text
/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/
  outputs/Ablation_Task4C_BottomDensity_1.2/physical/
    {channel,tbl}/{train,validation,test}/
      geometry.npy       # float32 [N,27,32,3], normalized geometry
      seeds.npy          # float32 [N,27,3], same coordinate system
      metadata.npz       # labels, counts, instance, centroid, radius, ...
```

<!-- handoff-cache-example -->
```python
from pathlib import Path
import numpy as np

physical = Path("/ibex/user/zhanx0o/FMT_Task4C_BottomDensity_1p2_20260916/outputs/Ablation_Task4C_BottomDensity_1.2/physical")
folder = physical / "channel" / "train"
geometry = np.load(folder / "geometry.npy", mmap_mode="r")
seeds = np.load(folder / "seeds.npy", mmap_mode="r")
with np.load(folder / "metadata.npz") as z:
    meta = {key: z[key] for key in z.files}

i = 0
n = int(meta["counts"][i])
line_mask = np.arange(27) < n
bundle = geometry[i, :n]             # [n,32,3]; exclude zero padding
seed_xyz = seeds[i, :n]             # [n,3], for neighbor selection
label = int(meta["labels"][i])      # 0=Non-hairpin, 1=Hairpin
physical_xyz = bundle.astype(np.float64) * meta["radius"][i] + meta["centroid"][i]
physical_seeds = seed_xyz.astype(np.float64) * meta["radius"][i] + meta["centroid"][i]
```

`counts`控制mask；填充不参与池化、归一化或损失。`instance/head_component/source_cell/center/scale_id`仅供追溯，不能作为分类特征。`center`为原播种中心，`centroid`为整束点质心，两者不同。追加样本来源见各流场`augmentation_index.npz`，文件哈希见`preparation.json`。原始VTK、缓存和可视化大产物不随Git推送，合作者需要这些文件或Ibex读取权限。

## 5. 当前最佳FMT：c156

FMT是本项目无可训练参数的傅里叶几何编码器；多层感知机（Multilayer Perceptron，MLP）学习分类。264个候选先单种子初筛，前20名各三种子复核，按平均验证F1选出c156，再用三个新种子做最终测试。

```text
缓存线簇 [B,27,32,3]
→ 沿已有32点折线均匀弧长插值至48点，整束质心/最大半径归一化
→ 每条有效线作中心，用播种点最远点采样（FPS）选6条邻居
→ 固定FMT：中心23 + 方向72 + 邻居逐特征均值23 + 最大值23 = 141维
→ 仅用训练集有效线统计的逐特征标准化
→ 共享逐线网络：141→256，三个256→512→256残差块
→ 有效线间均值+最大值，得到512维
→ 分类头512→256→128→2，Softmax后取Hairpin概率
```

最远点采样（Farthest Point Sampling，FPS）每步选距已选集合最远的播种点，每个中心从自身初始化；三个集合使用相同规则。FPS只选六邻居，整束10–27条有效线均保留。48点是缓存折线插值，不是重新积分得到的新细节。

| 项目 | 已执行设置 |
|---|---|
| 配方 / 结构 / 优化编号 | `p35 / wide_residual / h0` |
| 傅里叶频率 | k=0…5，共六个，包含直流分量 |
| 可训练总参数 | **992,386**；固定FMT为0参数 |
| 特征处理 | 仅用训练集3,637,354条有效线的均值/总体标准差；无signed-log和裁剪 |
| 数据增强 | **无切向扰动、无旋转增强** |
| 优化 | AdamW；学习率0.001，weight decay 0.0001，dropout 0.15，batch 128，梯度范数上限5 |
| 训练与选择 | 最多500轮；验证F1优先、平均精确率破平分；50轮无改善早停；验证loss平台15轮后学习率减半，最低1e-6 |
| 判定 / 最终种子 | Hairpin概率≥0.5；96721、96722、96723 |
| 复核验证F1 | **0.933609±0.008155**，不是测试分数 |
| 最终合并测试F1 | **0.920390±0.003032**，最高单种子0.923741 |
| Channel / TBL测试F1均值 | 0.926951 / 0.910338 |
| 平均训练时间 | 33.013分钟；单张Tesla V100-SXM2-32GB，包含逐轮验证 |

逐层结构、激活、归一化、141维公式、所有超参数和每种子结果见[完整c156说明](Task4C_best_FMT_c156_1.2_zh.md)。编码/网络：`FMT_Utils/Task4C_FPSAugmentSearch_1_1.py`；加载/标准化/训练：`experiments/Task4C_FPSAugmentSearch_1_1.py`；前20名复核和最终调度：`experiments/Task4C_FPSAugmentSearch_1_2.py`。对应配置为 `config/Ablation_Task4C_FPSAugmentSearch_1.1.json` 与 `1.2.json`。

源实现科学commit `9d05abc571bc7c45e5da8a8f9f3ee55db367d467`；最终复核commit `21cf1fb75b556741ae7a6da8bd0f08104c1a0db5`。1.2实际Git/Ibex配置SHA256为 `14a849f54553e51b16b5aec5e33fe2f274f2fe9e456b254854e32a7834883c78`。原始结果：

```text
/ibex/user/zhanx0o/FMT_Task4C_FPSAugmentSearch_1p2_20260917/
  outputs/Ablation_Task4C_FPSAugmentSearch_1.2/
    selection.lock.json
    summary.json
    independent_final_audit.json
    final/c156/seed{96721,96722,96723}/
      result.json
      history.jsonl
      selection.lock.json
      validation_predictions.npz
      test_predictions.npz
```

## 6. 已完成的同数据baseline

下表全部使用193,000/3,000/10,000冻结缓存、固定阈值0.5。F1为Hairpin正类 `2TP/(2TP+FP+FN)`：先合并两流场10,000测试束计算，再跨三个种子求均值与**样本标准差**。训练时间含逐轮验证，不含排队、数据构建、固定编码与最终推理。

| 方法 | 总可训练参数 | 测试F1均值±标准差 | 平均训练分钟 | 实验/科学commit |
|---|---:|---:|---:|---|
| FMT p35，最近6邻居 | 76,738 | 0.888404±0.011497 | 28.565 | BottomDensity_1.2 / `a7643890` |
| FMT p35，FPS6 | 76,738 | 0.893015±0.004408 | 32.221 | NeighborSelection_1.1 / `686ef144` |
| FMT p35，最近3+FPS3 | 76,738 | 0.886779±0.006196 | 34.351 | NeighborSelection_1.1 / `686ef144` |
| Conv3D 8³ | 72,192 | 0.855112±0.002812 | 38.134 | BottomDensity_1.3 / `12a0ffa7` |
| Conv3D 12³ | 72,192 | 0.896707±0.003988 | 54.009 | BottomDensity_1.3 / `12a0ffa7` |
| Conv3D 16³ | 72,192 | 0.913490±0.001233 | 72.521 | BottomDensity_1.2 / `a7643890` |
| Conv3D 24³ | 72,192 | 0.933198±0.006292 | 141.377 | BottomDensity_1.2 / `a7643890` |
| 切线/曲率+MLP | 76,782 | 0.580051±0.001726 | 15.031 | GeometricBaselines_1.1 / `d20c2458` |
| Point-NN式固定编码器+MLP | 76,704 | 0.303573±0.020438 | 5.995 | GeometricBaselines_1.1 / `d20c2458` |
| 双向长短期记忆网络（BiLSTM）+MLP | 76,786 | 0.919499±0.010500 | 83.194 | GeometricBaselines_1.1 / `d20c2458` |
| 缩小BiLSTM+MLP | 56,753 | 0.907316±0.008391 | 69.755 | BiLSTMCapacity_1.1 / `2b92207b` |
| PointNet，缩小宽度 | 76,749 | 0.889407±0.006155 | 128.553 | PointNet_1.1 / `b4c8a80c` |
| PointNet++，缩小宽度、单尺度分组 | 76,723 | 0.909232±0.005807 | 367.316 | PointNetPlusPlus_1.1 / `f3d98dd3` |
| **当前最佳FMT c156** | **992,386** | **0.920390±0.003032** | **33.013** | FPSAugmentSearch_1.2 / `21cf1fb7` |

实验完整前缀均为`Ablation_Task4C_`。旧baseline种子96611–96613；c156最终种子96721–96723且允许增加参数，**这不是全部等参数、同种子的比较**。同批新种子重训的原p35/FPS对照为0.882974±0.010458、76,738参数、14.410分钟，不覆盖历史FPS6结果。旧32³/36³/48³等结果属于其他数据版本，保留在主表但不混入这里。

各方法的实际输入与入口：

- **三维卷积（Conv3D）**：同束曲线先体素化为占据量和三个单位切向分量，共4通道，再经12/24/48通道卷积和MLP；不是直接将线点数组当作体积。分辨率改变不改变参数量。入口：`experiments/Task4C_BottomDensity_1_2.py`、`Task4C_BottomDensity_1_3.py`。
- **切线/曲率**：计算几何描述量，平均/最大值汇聚，再接MLP。**Point-NN式基线确实连接并训练MLP**，固定部分仅负责点云编码；是本项目缩减实现，不宣称完整复现原论文全部配置。与原BiLSTM共用入口：`experiments/Task4C_GeometricBaselines_1_1.py`。
- **BiLSTM**：保留曲线点序，逐线双向编码后汇聚，再接MLP。小版循环部分43,168参数、分类头13,585参数。入口：`experiments/Task4C_BiLSTMCapacity_1_1.py`。
- **PointNet**：有效线点当作无序点云，共享点网络、最大池化和分类头，保留两个变换网络与正交正则。入口：`experiments/Task4C_PointNet_1_1.py`。
- **PointNet++**：点云FPS中心、球邻域分组、分层共享点网络；保留512/128中心、0.2/0.4半径，缩小网络宽度。入口：`experiments/Task4C_PointNetPlusPlus_1_1.py`。三种子与9个进程全部完成，本次独立复核9份预测、18个源文件哈希、18条运行事件通过；此前“仍在运行”已过时。

完整逐种子表见[主表](paper_tables_tasks_3d.md)，方法判断见[实验日志](experiment_log.md)，作业及设备见[Ibex登记表](ibex_run_registry.md)。PointNet++最新证据为 `outputs/Ablation_Task4C_PointNetPlusPlus_1.1/final_verified_evidence.json`，远程结果根为 `/ibex/user/zhanx0o/FMT_Task4C_PointNetPlusPlus_1p1_20260916/outputs/Ablation_Task4C_PointNetPlusPlus_1.1/`。

## 7. 可视化与新模型接入

当前本机工作台：`outputs/Other_FMT_AnalysisWorkbench_1.2/index.html`。仓库根目录运行：

```powershell
python -m http.server 8767 --bind 127.0.0.1 --directory outputs
```

打开 `http://127.0.0.1:8767/Other_FMT_AnalysisWorkbench_1.2/index.html#saliency` 查看Hairpin支持分析，`#features`查看特征差异。**解释对象仍是旧p35、最近6邻居、76,738参数、seed96611，测试F1为0.878796，不是c156。** 原三维分类界面 `outputs/Other_Task4C_BundleVisualization_1.3/viewer/index.html`支持半透明GT、分类/混淆颜色、数量控制和相机预设，但其旧4.14数据也不是当前五倍训练缓存；不能把旧界面的数量或指标当作c156结果。

合作者添加模型时固定缓存和划分，只用训练数据估计统计量、验证数据选择结构和轮数；保存每束概率、流场编号、集合内行号，以便统一复算和可视化。初筛单种子，较好方案再多种子复核。测试集是已反复使用的benchmark，不称全新确认集。新方案单独建版本、配置、结果目录，记录代码commit及设备；不覆盖上述缓存和历史结果。现有运行不保留训练权重，复现依靠代码、配置、随机种子、数据及逐次结果，不能依赖已有checkpoint。
