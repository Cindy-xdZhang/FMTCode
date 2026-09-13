# Verify_JHTDB_DualFormatDownload_1.2

用户要求重新下载 channel 和 isotropic，将每轴物理跨度扩大两倍，离散网格改为 128×128×128，并同时保存 VTK 和包含完整 unsteady 时间序列的单个 NetCDF。

本版本保留原区域下界，向各轴正方向扩展。channel 若以原中心对称扩展，y 下界将越过壁面 -1。各轴跨度为原来的 2 倍，体积和每帧采样点数均为原来的 8 倍。

| 数据集 | x | y | z | 时间 | 网格 |
|---|---|---|---|---|---|
| channel | [3,3.6] | [-0.9,-0.3] | [0.2,0.8] | 1 + i×0.0065，i=0…31 | 128³ |
| isotropic1024coarse | [0,0.8] | [0,0.8] | [0,0.8] | 1 + i×0.002，i=0…31 | 128³ |

使用官方 getData 查询固定物理 xyz 坐标及三分量速度 u/v/w。空间插值均为 lag8；channel 使用 PCHIP 时间插值，因为 t=1 不在其 0.0065 的存储时间网格上；isotropic 使用其原存储时刻，无时间插值。不得把 channel 原始移动计算网格 cutout 当作固定物理坐标。没有对旧下载数据升采样。

每流场输出 32 个二进制 legacy STRUCTURED_GRID VTK 文件及 `.vtk.series` 时间索引；另有一个包含 32 个 `velocity_<time>` 数组的总 VTK，兼容 PyFlowVis 的时间数组读取方式。单个 `.nc` 包含完整 32 时刻，u/v/w 各自维度为 `(time,z,y,x)`，float32；坐标与时间为 float64。NetCDF 使用无损压缩，不量化速度。两种格式存储同一批新查询结果。

## 代码与复现

- 配置：`config/Verify_JHTDB_DualFormatDownload_1.2.json`。
- 下载：`experiments/Download_JHTDB_DualFormat.py`。
- NetCDF 导出与逐值验证：`experiments/JHTDB_DualFormat.py`。
- 独立审计：`experiments/Audit_JHTDB_DualFormat.py`。
- NetCDF FMT 读取：`FLowUtils/flowDatasetUtils/JHTDB_NetCDF.py`。
- 可视化：`experiments/Visualize_JHTDB_VTK.py`，交互显示抽稀到 20³，保存数据仍为完整 128³。
- 结构预览：`experiments/Visualize_JHTDB_Structure.py`，固定首个时刻 t=1，在完整 128³ 速度上按物理坐标用二阶差分计算 `Q=(||Omega||²-||S||²)/2=-trace(grad(u)^2)/2`。剔除外侧两层样点，仅绘制正 Q 的第 90 百分位等值面；阈值只用于显示，不定义研究标签。每流场 `structure_3d.html`、`structure_preview.json` 记录阈值、网格、源帧哈希和方法。固体旋转 Q=4、纯应变 Q=-9 的独立解析场测试通过；此额外测试使相关测试总数为 23。
- 输出：`outputs/Verify_JHTDB_DualFormatDownload_1.2/`。

```powershell
$env:MPLCONFIGDIR=(Join-Path $PWD 'tmp/matplotlib')
& tmp/jhtdb-venv/Scripts/python.exe -m experiments.Download_JHTDB_DualFormat --config config/Verify_JHTDB_DualFormatDownload_1.2.json --token-source 'C:\Users\xingdi\sources\PyflowVis\FLowUtils\flowDatasetUtils\JHTDB_Lodader.py'
& tmp/jhtdb-venv/Scripts/python.exe -m experiments.Audit_JHTDB_DualFormat --reference-loader 'C:\Users\xingdi\sources\PyflowVis\FLowUtils\flowDatasetUtils\VTKLoader.py'
```

凭据仅从本地源文件读取，不写入配置、输出或归档。运行最初使用 4 个工作线程；两流场各完成 8 帧后，确认本机有约 210 GiB 可用内存，将并发改为 8 并续传，未完成请求重新执行。该调整仅改变传输并发，采样及插值不变；记录见 `resume_history.json`，原配置另存 `source/config_workers4.json`。每次请求最多 524288 个空间点，低于官方 getData 的 2000000 点限制。下载支持通过文件哈希检查后续传当前版本。

## 验证与状态

**2026-09-07 已完成。** 两套各 32 帧 128³ 速度重新下载成功；总计 66 个二进制 VTK（64 个逐帧文件、2 个总文件）和 2 个完整 NetCDF。channel.nc 为 586337752 字节（559.18 MiB），isotropic.nc 为 611557167 字节（583.23 MiB）；两个总 VTK 均为 855639617 字节（816.00 MiB）。

`audit.json` 为 PASS：全部 VTK 坐标、时间、速度均精确一致；原 PyFlowVis VTK 读取器和 FMT NetCDF 读取器均正确读取全部 32 帧，包括末帧。NetCDF 的每个 u/v/w 值与对应 VTK 逐值相等；每个流场均有 32 个不同的速度数组。首末 xz 切面与 17 个独立随机点在线核对的最大绝对误差均为 0。

两个 `flow_3d.html` 已更新为 32 步，浏览器实际切换至末帧，分别显示 channel t=1.2015 和 isotropic t=1.062。首帧结构预览也已检查三维渲染。完整审计通过后，已删除旧版 66 个 64³ VTK 文件、2 个时间索引及 3 个旧预览，共 71 个文件；清单、删除前哈希及确认结果见 `old_grid_cleanup.json`。旧版完整 NPY/NetCDF 已在上一版本删除，没有再次删除研究基准数据。

完成情况以各流场 `manifest.json` 与总 `audit.json` 为准。只有全部 VTK 和 NetCDF 坐标、时间、速度逐值相等，首末时刻 xz 平面及独立随机点在线核对成功，才允许删除旧版本下载数据。旧版配置、审计、源代码与文档保留；旧数据被删除不改变其既有正确性记录。此版本不更改任何冻结研究实验或训练数据。

官方依据：[isotropic 数据集](https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic)、[channel 说明](https://turbulence.pha.jhu.edu/docs/README-CHANNEL.pdf)、[JHTDB 官方配置](https://raw.githubusercontent.com/sciserver/giverny/main/metadata/configs/jhtdb-config.json)。
