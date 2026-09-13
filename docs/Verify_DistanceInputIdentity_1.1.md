# Verify_DistanceInputIdentity_1.1 验证协议

验证目标：直接检查同时间点间距离经过标量时间差分、Fourier、固定网络和固定聚类流程是否保持观察者不变性，并捕获旧编码器实际读取的数组。方法级结论与修订仅记入 `docs/experiment_log.md`。

输入固定为 `C:/Users/xingdi/sources/FLowlineClusteringVisualAnalysis/outputs/Other_FeatureSilhouette_1.1/*/geometry/geometry_bundle.npz` 的全部8份 `paths`，每个primitive含中心及六邻居，每线32点。载入后转换为float64；结果保存每个源文件SHA256。不读IVD标签、不选择样本。另设64个整数坐标primitive构造以消除旋转三角函数舍入误差。

所有材料点共享同一个随时间变化的旋转和平移，不重新播种。计算七点全部21对同时间欧氏距离及其标量时间差分。距离序列取前六个实Fourier频率，拼接实部和非零频率虚部；标准化、8维主成分分析及KMeans只在原观察者下拟合，变换后复用参数。固定随机小网络采用231→32→8、tanh、双精度、eval模式，种子12091；它只检验函数恒等性，不用于分类性能比较。

临时在内存中包装 `DFT_FMT_3D.dft_rotation_invariants_3d`，记录旧编码器送入的数组；finally恢复原函数。断言实际邻居输入等于三个坐标分量的时间差分。冻结磁盘文件不修改。输出逐数据绝对误差、相对误差、逐位相等标志、簇标签变化数、实际输入形状，以及原算法/Gram对照误差。

执行入口：

```powershell
./tmp/jhtdb-venv/Scripts/python.exe -m experiments.Verify_DistanceInputIdentity_1_1
```

结果：`outputs/Verify_DistanceInputIdentity_1.1/results.json`。测试只验证预定义变换下的数值行为；一般距离不变性来自欧氏范数的正交不变性。此实验没有训练任务或Ibex作业，没有checkpoint。
