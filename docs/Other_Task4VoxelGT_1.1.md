# Other_Task4VoxelGT_1.1：手动标注 VTK 的独立 voxel/cube 可视化

## 目的与边界

本版本是 Task4 的数据准备工具，不是 Task4 分类实验。它载入手动标注的
`channel_GTs.vtk`，把标签离散采样到一个与输入流场格点相互独立的规则三维
voxel 网格，并以可旋转 cube 或固定多视角 PNG 显示。

严格标签约定：`label > 0` 是涡实例，且不同正整数表示不同实例；`label <= 0`
是背景。这里的 `VortexIds` 是实例编号，不是 streamwise、spanwise、hairpin
类别名。不得直接把 1–80 当成 Task4 的 80 个涡类型；Task4 开始前仍须按
`docs/research_tasks_and_protocol.md` 单独冻结标签来源、类别定义和实例到类别的映射。

## 来源核对

- 用户给出的 `optimal-connection/src/flow3d/ObjectTopologicalVortexSegmentation3D.cpp`
  计算 Bremer topological relevance 标量场，不含 cube 渲染。
- cube 行为的实际 C++ 基准是
  `optimal-connection/src/voxel3d/ObjectVoxelAnnotation3D.cpp`：独立规则网格、voxel
  中心采样、非正标签作为背景、六邻域 exposed-voxel 裁剪和 shrink cube。
- Python 实现位于 `FMT_Utils/VoxelSegmentation_3D.py`；独立入口是
  `experiments/Visualize_Task4_VoxelGT_3D.py`。它不改动另一个 session 正在移植的 PyflowVis
  渲染引擎。

## 真实 `channel_GTs.vtk` 审计（2026-08-31）

来源：
`C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D/channel_flow/channel_GTs.vtk`

- 文件大小：48,456,419 bytes。
- VTK 拓扑：`vtkUnstructuredGrid`；218,205 points，723,188 cells。
- 物理边界：x `[0, 3.1293208599]`，y `[0, 1.1719599962]`，z
  `[-0.9992340207, -0.3812069893]`。
- 目标数组：cell-data `VortexIds`，整数范围 0–80。
- 源文件实际出现 73 个正实例编号；721,811 cells 为正，1,377 cells 为 0。
- 按各轴 median cell bounds 估计的原生 fidelity 是 `255×191×302`
  （14,708,910 voxels）。默认配置只为快速检查把三轴同比例压到不超过
  2,000,000 voxels；`--native-resolution` 可关闭该预览上限。

## 数值语义

1. voxel 网格的 domain 固定为 GT 文件自身 bounds；分辨率可独立指定，绝不从
   `channel.vtk` 的离散格点偷换。
2. cell-data 标签按 voxel center 通过 `vtkResampleToImage` 做分片常数采样；
   `vtkValidPointMask=0` 的稀疏网格外部位置置 0。
3. point-data 标签使用 nearest-point sampling；标签不做线性插值。
4. NumPy 存储顺序是 `(Z,Y,X)`；VTK/用户接口分辨率顺序是 `(Nx,Ny,Nz)`。
5. 默认 `surface_mode=outer` 复现 C++ 六邻域规则，只上传存在非可见邻居的
   cube。`interfaces` 还保留相邻正标签不同的内部界面；`all` 显示全部正 cube。

## 运行

默认输出四个固定视角和 JSON 审计：

```powershell
python -m experiments.Visualize_Task4_VoxelGT_3D
```

打开可拖动旋转、滚轮缩放的 VTK 窗口，同时保留 PNG：

```powershell
python -m experiments.Visualize_Task4_VoxelGT_3D --interactive
```

使用完整 source-fidelity voxel 网格，或显式指定与流场分辨率无关的网格：

```powershell
python -m experiments.Visualize_Task4_VoxelGT_3D --native-resolution
python -m experiments.Visualize_Task4_VoxelGT_3D --resolution 160 120 192
```

只看指定实例并显示不同实例的内部接触界面：

```powershell
python -m experiments.Visualize_Task4_VoxelGT_3D --label 8 --label 26 --surface-mode interfaces
```

默认结果目录：`outputs/Other_Task4VoxelGT_1.1/channel_GTs/`。

## 验证

```powershell
python tests/test_voxel_segmentation_3d.py
python -m py_compile FMT_Utils/VoxelSegmentation_3D.py experiments/Visualize_Task4_VoxelGT_3D.py
```

单测覆盖 point/cell 标签、非正标签归零、任意分辨率离散采样、六邻域内部
cube 裁剪、不同标签界面保留、label filter 和 preview voxel budget。

真实文件验证结果（2026-08-31）：

- 默认 preview：`131×98×155 = 1,989,890` voxels；16,711 个正 voxel；
  `outer` 模式输出 15,628 个 cube；源文件全部 73 个正实例编号均保留。
- 完整 fidelity：`255×191×302 = 14,708,910` voxels；125,242 个正 voxel；
  125,958 个采样中心位于输入几何内；源文件全部 73 个正实例编号均保留。
- 四个默认视角均由真实 GT 离屏渲染成功并完成目视检查；JSON 审计位于
  `outputs/Other_Task4VoxelGT_1.1/channel_GTs/voxel_segmentation_summary.json`。
