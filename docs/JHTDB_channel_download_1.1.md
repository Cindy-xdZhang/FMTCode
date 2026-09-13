# JHTDB channel 下载与加载器检查

**当前版本（2026-09-07）**：两流场已重新下载为扩大范围的 128³×32 步，并同时保存 VTK 和完整 NetCDF；当前入口见 [1.2 下载与审计记录](JHTDB_dual_format_download_1.2.md)。以下内容保留原版本历史，旧数据与旧预览已删除。

**保存格式更新（2026-09-06）**：用户要求重新下载为VTK，现已完成并逐值核对。下文描述历史NetCDF/NPY版本；32个NPY与一个NetCDF已删除，实验清单及审计保留。当前数据和入口见 [VTK下载记录](JHTDB_vtk_download_1.1.md)。

本次为 `Verify_JHTDB_ChannelDownload_1.1`，只验证数据下载和坐标/时间映射，不训练模型，不变更已有 Task1–5 数据或阈值。

## 来源与采样

用户提供路径的实际文件是 PyflowVis 的 `FLowUtils/flowDatasetUtils/JHTDB_Lodader.py` 和 `misc/DEMO_Getdata_local.ipynb`。
已复制加载器并保留原类名，另提供正确拼写 `JHTDBLoader`。原有硬编码凭据未复制进 FMT 源码。

依据：[官方 Python 示例入口](https://turbulence.idies.jhu.edu/database/local/python)、[官方实现](https://github.com/sciserver/giverny)、[channel 数据说明](https://turbulence.pha.jhu.edu/docs/README-CHANNEL.pdf)。

用户确认的配置为局部 `64×64×64`、32 个时间步，起点 `t=1`。空间盒沿用 notebook：
`x=[3,3.3]`、`y=[-0.9,-0.6]`、`z=[0.2,0.5]`，包含端点、各轴均匀采样。
时间为 `1+k×0.0065, k=0,...,31`，终点 `1.2015`。

channel 原始存储时刻为 `k×0.0065`，因此上述时刻使用分段三次 Hermite 插值（PCHIP）。它们不是原始连续快照编号。
空间采用 `lag8`（每方向八点 Lagrange 插值），查询三分量速度。
channel 原始 y 网格非均匀，本输出是均匀物理坐标上的插值结果；增加输出分辨率不会增加原始数据的独立信息。

使用 `getData` 的静止壁面坐标系。官方说明指出 cutout 的原始网格以 x 方向速度 0.45 移动；不能把原始 cutout 直接当作固定物理位置的时间序列。
本实现不手动再次施加坐标平移或速度修正。

## 对原实现的检查

| 项目 | 原实现 | 当前处理 |
|---|---|---|
| 三维点排列 | `i*Ny*Nz+j*Nz+k` 与官方示例一致 | 保留，转换成 FMT `(T,Z,Y,X,3)`；用非立方网格验证 |
| 二维分量 | xy→u,v；xz→u,w；yz→v,w 正确 | 保留，三个平面分别验证；记录平面及固定坐标 |
| 时间参数 | `none` 拒绝多个时刻；`integerTime` 忽略 `time_res`；部分非法输入静默降为一帧 | 返回恰好请求数量，验证范围和整数；channel 禁止把整数模式解释成快照编号 |
| PCHIP option | 计算 `[end,dt]` 却不传入；逐时刻查询本身有效 | 保留逐时刻查询，明确不使用 time-series option，避免在时间循环内重复请求时间序列 |
| 返回形状 | 任意响应强制 reshape 成三分量，可能误接受非速度变量 | 只允许 velocity/field；检查每批 `(N,3)`、有限值及缺失值标记 |
| 内存与续传 | 所有时间结果在内存中堆叠 | 保留原内存 API，额外下载入口逐帧落盘、校验和续传；大型请求不要一次调用内存 API |
| 凭据 | 源码硬编码 | 环境变量或显式参数；兼容从用户指定旧源码只读取字面量凭据，不执行该文件 |

`return_array=True` 返回 NumPy 数组，下载不需要 PyTorch。
默认仍返回既有 `UnsteadyVectorField2D/3D` 对象，需安装 FMT 对应依赖。
二维对象表达的是平面内投影速度，不能解释为三维粒子被限制在该平面上的真实轨迹。

## 运行

配置：`config/Verify_JHTDB_ChannelDownload_1.1.json`。
输出：`outputs/Verify_JHTDB_ChannelDownload_1.1/`（git 忽略）。

```powershell
# Set JHTDB_AUTH_TOKEN in your environment, then:
python -m experiments.Download_JHTDB_Channel --probe
python -m experiments.Download_JHTDB_Channel --workers 4
python -m experiments.Visualize_JHTDB_Channel
python -m experiments.Audit_JHTDB_Channel
python -m pytest tests/test_jhtdb_loader.py tests/test_jhtdb_netcdf.py -q
```

本地隔离环境为 `tmp/jhtdb-venv/Scripts/python.exe`。下载依赖 `givernylocal==3.3.3`、NumPy、netCDF4；交互查看器使用 Plotly。
完整环境记录写入输出目录 `requirements_runtime.txt`。
下载最多 4 个线程，各线程使用独立 dataset 对象，清单由主线程单独写入。
本次先串行完成两帧，随后中断以改为 4 线程续传；保留首次源码快照和每次运行的源码校验和。

文件：逐帧 `frame_NNN.npy`，合并 `channel.nc`，可独立打开的 `channel_3d.html`，及 `manifest.json`。
NetCDF 变量 `u,v,w` 均按 `(time,z,y,x)` 排列，坐标为 float64，速度为 float32。
完整速度数组未压缩为 96 MiB；逐帧文件与合并文件同时保留，以支持续传及直接使用。
交互图显示 20³ 个采样点、完整三个速度分量和时间选择；显示抽样不修改下载数组。

读取下载完成的文件：

```python
from FLowUtils.flowDatasetUtils.JHTDB_NetCDF import load_jhtdb_netcdf
flow = load_jhtdb_netcdf('outputs/Verify_JHTDB_ChannelDownload_1.1/channel.nc')
# flow.field: (32, 64, 64, 64, 3), float32
```

该专用读取器默认包含全部帧，`start/stop` 为左闭右开；拒绝非均匀坐标。

## 验证证据

`tests/test_jhtdb_loader.py` 的 17 项测试通过，包含非立方网格排列、xy/xz/yz 分量、时间数量、非法响应拒绝、单快照默认行为和实际 FMT 对象取值。
加上 NetCDF 读取器的两个测试，共 19 项通过。
在线小网格 `3×4×5` 在 t=1 查询成功，见 `probe.json`。
完整下载的完成状态与逐帧统计以 `manifest.json` 为准。
下载结束时还会重新查询首末帧 xz 平面及末帧 17 个固定随机种子采样点，比较二维加载器、三维结果和直接官方调用。
这验证查询/排列/保存的一致性，不是独立验证 JHTDB 原始仿真或插值精度。

下载期间另完善了非 channel 单快照的默认时间方法、流式文件校验和及错误脱敏。
本次已显式指定 `pchip`，这些完善不改变此次查询；下载时实际执行的两个源码版本保存在 `source_attempt_001/002`，最终代码另记于 `audit.json`。

## 完成状态（2026-09-06）

32 帧下载及完整审计通过：`(32,64,64,64,3)`、float32、t=1 至 1.2015。
32 个逐帧文件的 SHA-256 均核对通过；NetCDF 与逐帧数组、FMT 读取结果逐值完全一致。
首末帧独立 xz 平面查询和末帧 17 个独立随机点查询的最大绝对差异均为 0。
首末帧速度分量差的均方根为 0.0847707242，仅记录时间变化，不作为物理精度结论。
19 项测试通过。三维图已在浏览器核对体渲染与时间滑块，末帧正确显示 t=1.2015。
具体证据为输出目录的 `manifest.json`、`audit.json`、`probe.json` 和源码快照。
