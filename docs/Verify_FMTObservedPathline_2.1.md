# Observer worldline 与 observed pathline 独立核对

依用户要求，按 `optimal-connection/src/flow3d/ReferenceFrame3d.cpp` 的三步顺序新建实现：observer field中数值积分相机worldline；用observed→lab变换构造observed场；在observed场中积分粒子轨线。当前只验证纯平移，旋转矩阵为单位阵。

源代码：`FMT_Utils/ObservedFieldWorldline_3D.py`；实验入口：`experiments/Verify_FMT_Observed_Pathline_2_1.py`；配置：`config/Verify_FMTObservedPathline_2.1.json`。

每个流场保留1.4同一批物质种子、后半程初始时间、96步展示和48步分类窗口；不改冻结Task1-4.1分类器。七档observer速度比例为0至1、间隔1/6。相机参考时刻等于粒子播种时刻，所以初始变换为恒等；此后逐时刻累积相机位移。去掉全部bounding box，保留所有档位共同的取景尺度，不逐幅重置中心。

验证内容：

- 独立积分的observed pathline与原轨线精确坐标变换比较。
- 同一时刻邻居相对位置、双精度邻居FMT特征，以及中心轨线FMT特征分别比较。
- 对精确变换轨线重新分类，排查积分器数值误差是否导致标签变化。
- 固定其他特征只替换中心块，以及固定中心块只替换其他块，分别计算标签变化数量。相同数量不自动代表逐条变化集合相同。
- 错误对照直接积分`v(y,t)-u(t)`，使用按原顺序选定的前128条中心轨线；只在该对照完整者上比较误差，报告其数量；不改变主图队列。

11项本地数值测试通过，新增4项覆盖参考时刻/相机起始点、逆查询错误对照、越出原盒、同刻和跨时刻几何的区别。实际数据结论和全部诊断数值只写入 `docs/experiment_log.md`；不推断benchmark准确率，不用复制标签或修改冻结配方制造不变性。

理论文档已更新至实际正式路径 `optimal-connection/docs/referenceframe/referenceFrame_overview_zh.md` 的§2.3–2.6；用户指定的 `referenceFrame/_overview/_zh.md` 建为通向正式文档的入口。保留原文其余内容；没有修改C++源码，因此不需要C++编译。

最终21份图片已下载并检查，ZIP为outputs/Verify_FMTObservedPathline_2.1/Cylinder_observed_worldline_figures.zip。全部七栏大小检查通过，六份PDF均0FAIL；最小字体论文7pt/PPT13pt，42栏及六张整图完成目视复核。各图7个统计文字跨不可见白色axes背景边缘WARN已接受；无可见边框或轨线裁剪。运行后清除了继承的旧四档审计字段，保留原JSON备份，不改变图像和数值结果。
