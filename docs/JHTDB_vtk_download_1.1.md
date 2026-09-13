# JHTDB 二进制 VTK 下载（1.1）

**2026-09-07 数据已替换**：用户要求各轴跨度翻倍并改为 128³，现已完成新 VTK 和完整 NetCDF，见 [1.2 下载与审计记录](JHTDB_dual_format_download_1.2.md)。本页所述 64³ VTK 和旧预览已删除，历史配置及审计保留。以下结果为原版本的历史记录。

**已完成（2026-09-06）**：两套各32帧64³、全部二进制VTK；逐帧/合并文件与原始返回值逐值一致。
原PyFlowVis读取器正确识别全部时间与数值，两个流场的独立切片与随机点查询最大差异均为0。
新channel与上一轮32帧完全一致。旧32个NPY及一个NetCDF已按用户要求删除；证明见 `audit.json` 和 `old_format_cleanup.json`。

版本 `Verify_JHTDB_VTKDownload_1.1`。按用户新要求，channel 重新请求 JHTDB，isotropic 为新下载，直接输出 VTK，不通过修改扩展名转换格式。
采样配置在 `config/Verify_JHTDB_VTKDownload_1.1.json`。

| 数据 | API 名称 | 空间盒（64³，含端点） | 时间（32帧） | 时间方法 |
|---|---|---|---|---|
| channel | channel | x=[3,3.3], y=[-0.9,-0.6], z=[0.2,0.5] | 1+k×0.0065，末帧1.2015 | PCHIP |
| isotropic | isotropic1024coarse | x,y,z=[0,0.4] | 1+k×0.002，末帧1.062 | none，原始存储时刻 |

isotropic 沿用前轮的局部64³、32帧及从t=1开始的规模；未另指定空间盒时采用上述局部区域。
两个流场均用 lag8 空间插值（每方向八点 Lagrange 插值），仅下载三分量速度；物理坐标与速度分量保留 JHTDB 的 x,y,z 和 u,v,w 顺序。
isotropic 是 [0,2π)³ 周期域中的局部盒，不是整个周期域。
channel 采用静止壁面坐标的 getData，时间起点1需插值。

## 参考格式与文件

实际参考文件为用户目录的 `flowData3D/channel_flow/channel.vtk`，而非 `channel/_flow/channel.vtk`。
读取确认其为 binary legacy `STRUCTURED_GRID`、256×192×256 点，点数据有 `velocity` 三分量。
新文件匹配该网格类型和速度存储位置；坐标采用 float64、速度采用 float32。
未把参考文件的涡量、lambda2、区域标签或其坐标置换复制到新下载数据。

输出目录 `outputs/Verify_JHTDB_VTKDownload_1.1/` 下，每个流场提供：

- `channel/channel.vtk` 或 `isotropic/isotropic.vtk`：同一网格的32个速度数组，命名 `velocity_1.0000000` 等，后缀为物理时间；用于支持命名时间数组的 PyFlowVis 读取器。
- `channel_000.vtk` 至 `channel_031.vtk`（isotropic同理）：每个文件单帧，点速度名为 `velocity`，全局字段 `TimeValue` 记录物理时间。
- `channel.vtk.series` / `isotropic.vtk.series`：为 ParaView 指定逐帧文件与实际时刻；单个合并 VTK 的多个数组本身不是 ParaView 时间轴。
- `flow_3d.html`：直接读取逐帧 VTK 生成的三维查看器。
- `manifest.json`：逐帧文件与原始速度校验和、源码校验和、配置、时间、在线核对结果。

## 执行与审计

```powershell
python -m experiments.Download_JHTDB_VTK --token-source PATH_TO_EXISTING_LOADER
python -m experiments.Audit_JHTDB_VTK --reference-loader PATH_TO_PYFLOWVIS_VTKLOADER --old-channel outputs/Verify_JHTDB_ChannelDownload_1.1
python -m experiments.Visualize_JHTDB_VTK outputs/Verify_JHTDB_VTKDownload_1.1/channel
python -m experiments.Visualize_JHTDB_VTK outputs/Verify_JHTDB_VTKDownload_1.1/isotropic
```

凭据只从用户指定原加载器的字面量赋值读取，用于 JHTDB；不写入代码或清单。
本地环境 `tmp/jhtdb-venv/Scripts/python.exe`，4个线程总量在两个流场之间共享，各线程有独立的 JHTDB dataset 对象。
读写测试使用非立方网格的坐标编码速度，检查轴顺序、二进制文件结构和多个时间数组。
每个真实帧写完后立即逐值读回；结束后重新独立查询首末 xz 平面与17个随机点，并核对所有合并数组。
使用原 PyFlowVis VTKLoader 再读取合并文件，检查时间及每一个速度数值。

旧 channel 的32个NPY与NetCDF仅在新VTK全部核对成功后删除，保留原实验清单、审计和源码记录。
删除记录已写入新输出目录 `old_format_cleanup.json`。之后复查VTK无需 `--old-channel` 参数。
这次仅改变下载/文件保存，不训练、不修改既有研究数据或方法阈值。

## 官方依据

- [Isotropic 数据说明](https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic)：coarse 数据覆盖 t=1，fine 只覆盖很短的初始时间。
- [Isotropic 参数说明](https://turbulence.idies.jhu.edu/docs/isotropic/README-isotropic1024.pdf)：空间域、存储间隔0.002。
- [ParaView 时间序列文件说明](https://docs.paraview.org/en/latest/UsersGuide/dataIngestion.html)：JSON .series 元文件指定真实时间。
