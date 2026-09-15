# Task4-c Hairpin分类交接

## 1. 任务

输入一簇三维涡线几何，分类Hairpin/Non-hairpin。两个流场均为单帧，沿涡量curl(u)双向积分；GT、速度和涡量用于数据构建，不作为网络输入。

## 2. 最新数据：task4-c-v9.15_v2（已纠正）

**原commit `9769c68d` 的实例留出、局部标签和固定150轮运行已废弃，不是有效v2结果。当前修正版正在重新构建。**

- 恢复4.14候选及标签：lambda2阈值Channel −13.395、TBL −0.0272，候选单元八顶点过阈值且展向脉动涡量为正。完整候选头区单个GT实例覆盖≥50%为Hairpin，零覆盖为Non-hairpin，混合头区排除。
- 恢复同一头区内空间分块，训练/验证/测试共享涡实例，**完整实例留出数为0**。评价中心距训练中心至少1个网格间距、距同头区训练中心不超过4个。总132,000束，按旧比例为89,100训练、9,900验证、33,000测试；不设每实例正负配额。
- 负样本采样沿用旧实现。正类头区每隔一个采样编号使用其原分区内z较低的一半单元，其余仍用完整单元池，使底部采样更密。
- Channel **单向长度0.08–0.12**、ds 0.001–0.005、11档；TBL **单向长度3–5.5**、ds 0.01–0.05、12档。双向全线长度分别0.16–0.24、6–11。邻距仍为网格间距的0.25/0.5/1倍。RK45积分并清洗，每束10–27线、每线32点，整束质心/最大半径归一化。

原始数据根目录：Windows `C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D`；Ibex `/ibex/user/zhanx0o/FMT_Task4B_VelocityCurl_4p1_20260906/repo/inputs`（需读取权限，不随Git发布）。

|流场|速度场|GT|
|---|---|---|
|Channel|`channel_flow/channel.vtk`|`channel_flow/channel_GTs.vtk`|
|TBL|`tbl_flow/tbl.vtk`|`tbl_flow/tbl_GTs.vtk`|

VTK速度为`point_data['velocity']`，GT为`cell_data['VortexIds']`，实例0有效、GT外−1。x流向、y展向、z竖直，壁面zmin。

修正版Ibex目录：`/ibex/user/zhanx0o/FMT_Task4C_v2_Restored_20260915`。生成后读取其下`outputs/mainExp_Task4C_InstanceCoverage_9.15_v2/restored_4.14/physical/<flow>/<train|validation|test>/`：`geometry.npy [N,27,32,3]`、`seeds.npy`；`metadata.npz`中的`labels/counts/head_component/instance`给出标签、有效线数及来源。沿用固定集合。

比较p35/n0_k06与32³/36³/48³ Conv3D；恢复验证集选最佳轮次、平台降学习率和早停。入口：[配置](../config/mainExp_Task4C_InstanceCoverage_9.15_v2.json) · [数据](../FMT_Utils/Task4C_InstanceCoverage_9_15_v2.py) · [训练](../experiments/Task4C_InstanceCoverage_9_15_v2.py)。

## 3. 可视化

`outputs/Other_Task4C_BundleVisualization_1.3/viewer/index.html`及旁边`geometry/`目录：支持模型/集合/实例切换、两类数量、标准相机视角、半透明GT、Head模式及TP/TN/FP/FN分析色。**目前仍展示旧4.14结果，未混入废弃运行。**
