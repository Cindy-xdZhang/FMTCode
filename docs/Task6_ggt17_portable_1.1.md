# GGT17：从原速度序列产生一个新时刻的 v−u 场

这里的GGT17指项目已验证的局部最小二乘参考系优化：求瞬时刚体参考系速度u，使观察到的速度尽量定常，返回v−u。入口源于 `GenericLocalOptimization3d::computeWithOutFilter`，实际代码、适配层和输入来源均随包提供。本文是当前实现的调用记录，不声称重新独立复现了一篇论文。

## 固定调用

1. 从原流场读取索引i−1、i、i+1三个时刻的原速度，数组顺序为 `[3,Nz,Ny,Nx,3]`，float32。
2. 保留原空间网格与物理范围，不自动降采样；当前原实现适配器要求均匀空间网格和均匀时间步长。
3. 设置 `mInvariance=EInvariance::Objective`、`NeighborhoodU=11`、`UseSummedAreaTables=true`，调用 `computeWithOutFilter(field,1,1,u,relative,diagnostic)`，输出中间帧的v−u。
4. `NeighborhoodU=11` 是原C++参数，代码据此取x/y/z的±11邻域端点，并按原代码裁剪、求和。它不是“总共11个点”。保留原prefix-sum分支，不自行替换窗口和边界算法。
5. NetCDF时间步长使用整段时间轴 `(tmax−tmin)/(Nt−1)`，三帧窗口起点用 `tmin+(i−1)*dt`，物理时间另外记录原时间变量值。这与已存场的生成方式相同。
6. SquareCylinder从 `SquareCylinder.fileseries` 读取时间范围和文件顺序，不能把文件名后缀当物理时间。
7. xy平面格点数可被64整除时默认8线程，否则单线程，避开原代码packed `vector<bool>` 有效性掩码的并发问题。稳态tornado跳过优化，直接输出v。

原计算函数只将内部时间遍历限制为三帧窗口的中间帧；保留原空间代数与求解、prefix-sum代码。适配器实现原流场的三线性和时间插值。C++函数不是从另一个不同的直接窗口求解器替换而来。具体原文件哈希见 `tools/native/source_audit.json`，已有帧记录见各帧的 `observer_provenance.json`。

## 命令

先安装 `requirements.txt`，使用Python 3.11或兼容环境。Windows需要x64 Microsoft C++运行库和OpenMP运行库；当前附库与原数据所用库完全一致。系统已有这些运行库时直接运行：

```bash
python tools/ggt17_observe.py --flow halfcylinderRe640 --source "D:/flowdata/halfcylinderRe640resampled.nc" --index 67 --out "D:/new_frames/re640_067"
```

其他流名为 `cylinder3d`、`halfcylinderRe320`、`deltaWing_resampled`、`SquareCylinder`、`tornado3d`，原文件名和坐标变量见 `tools/flow_recipes.json`。传入的新输出目录必须尚不存在，以防覆盖人工标签。非定常首末帧因缺少相邻时刻而被拒绝。输入有非均匀网格、非均匀时间轴或缺失数据时明确报错，不默默改变算法。

输出 `relative_velocity.npy`、`coordinates.npz`、`generation.json`。生成记录含参数、实际使用的速度切片哈希、库哈希和线程数。输出不含核线标签，也不会保存流线。`--threads 1` 可显式使用单线程；不对未对齐平面开放不安全的多线程模式。

## 重编译

原C++源、固定的计算函数、Eigen头文件及其许可证都在 `tools/native/`，不依赖作者机器的绝对路径。

```bash
cmake -S tools/native -B build/ggt17 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ggt17 --config Release
```

Windows通常生成 `build/ggt17/Release/reference.dll`，Linux通常生成 `build/ggt17/libreference.so`；用 `--library <该文件>` 明确选择重编译库。重编译后的浮点结果应与保存的场另行比较，不能仅凭编译成功声称逐位一致。Windows源码构建已经通过；附带原库经CLI重算Re160索引81、DeltaWing索引85、SquareCylinder索引65与稳态tornado，全部与已有场逐位一致。Linux构建说明不代表已完成Linux数值验证。
