# Task4-c Hairpin分类交接

## 1. 任务

输入一簇三维涡线的几何，二分类：Hairpin=1、Non-hairpin=0。Channel与TBL都是单帧流场，沿涡量`curl(u)`积分。新增模型使用同一几何和实例划分；GT、速度、涡量及实例ID用于构建标签，不作为模型输入。

## 2. 最新数据：task4-c-v9.15_v2

**状态：2026-09-15已确认单向长度，正在部署正式数据构建。**

- Channel 74个、TBL 58个Hairpin实例；每实例500正＋500负，共132,000束。相交采样盒分组后约90%实例训练、10%测试：119,000/13,000束；测试实例的全部束只进测试，轨线穿入另一集合GT则剔除。
- 在实例扩张包围盒内以`lambda2 < −13.395`（Channel）/`−0.0272`（TBL）及`ω′y > 0`筛选种子。GT内速度与涡量的锐角>45°视为head，每实例必须有3个不同中心的head束。
- Runge–Kutta–Fehlberg（RK45）双向积分；剔除短线、任一轴零方差等退化线，每束保留10–27线，每线按弧长重采样32点，整束以质心平移并除以最大半径。邻居距离多尺度；Channel**单向长度[0.08,0.12]**、ds[0.001,0.005]、11档；TBL**单向长度[3,5.5]**、ds[0.01,0.05]、12档。双向整条曲线长度分别为[0.16,0.24]、[6,11]。
- 清洗后有效种子≥50%属于唯一同一GT实例则为正；全部在GT外为负；其余混合束和并列归属剔除。

原始数据根目录（不随Git发布）：Windows为`C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D`；Ibex为`/ibex/user/zhanx0o/FMT_Task4B_VelocityCurl_4p1_20260906/repo/inputs`，需目录读取权限。

| 流场 | 速度场文件 | GT文件 |
|---|---|---|
| Channel | `channel_flow/channel.vtk` | `channel_flow/channel_GTs.vtk` |
| TBL | `tbl_flow/tbl.vtk` | `tbl_flow/tbl_GTs.vtk` |

读取VTK的`point_data['velocity']`；GT为`cell_data['VortexIds']`，实例0有效，GT外记−1。坐标：x流向、y展向、z竖直，壁面在zmin。

生成后读取`outputs/mainExp_Task4C_InstanceCoverage_9.15_v2/physical/<flow>/<instance>/`：`geometry.npy`形状`[N,27,32,3]`；`metadata.npz`的`labels`为二分类、`counts`为有效线数（其后是补零）、`mandatory_head`标记必选head；`coverage.json`的`role`给出train/test。新模型沿用此划分。

入口：[配置](../config/mainExp_Task4C_InstanceCoverage_9.15_v2.json) · [数据构建](../FMT_Utils/Task4C_InstanceCoverage_9_15_v2.py) · [训练接口](../experiments/Task4C_InstanceCoverage_9_15_v2.py)。

当前比较：p35逐特征均值＋最大值池化，最大半径归一化、训练集逐特征标准化、6个频率（含零频）；与参数量接近的32³/36³/48³ Conv3D共用几何和划分。

## 3. 可视化分析

本地打开`outputs/Other_Task4C_BundleVisualization_1.3/viewer/index.html`，须保留旁边的`geometry/`目录。可切换流场、集合、模型、实例和标准相机视角；分别控制两类数量，叠加半透明GT。Head特殊模式保留GT head附近的束（含漏检）＋Non-hairpin；分析勾选显示TP正确正类、TN正确负类、FP误报、FN漏检。

**当前页面展示旧4.14数据；v2完成后以新几何和预测重新构建。** 查看器入口：[Visualize_Task4C_Bundles_3D.py](../experiments/Visualize_Task4C_Bundles_3D.py)，输入`viewer_package`后运行`build`。
