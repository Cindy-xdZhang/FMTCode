# Verify_AIVDTranslationObservers_1.1

2026-09-07 用户要求：将原样 `aivd1w3_dft` 用于 Re160 3D 七档平移 observer，按真实 changed labels 检查平移不变性。

## 冻结协议

- 复用 `Verify_FMTObservedPathline_2.1` 的 7,220 个物质 primitive、七组独立 observed-field 积分轨迹；t0=10.5，96 步显示、前48步重采样32点用于分类。重新验证每组轨迹等于 lab 轨迹减去积分相机位移。没有重新积分或修改这些原始轨迹，也不读取旧预测用于着色。
- observer 速度是原场全局空间平均速度的 0、1/6、2/6、3/6、4/6、5/6、1 倍；积分速度得到相机位移。禁止逐幅重新居中、原域 bounding box 和按标签筛选轨线。
- 分类器采用 `Verify_AIVDTransfer_1.1` Task1/cylinder3d/seed7080 标量分支：只使用晚期 train ordinals4–7、validation8–9，一次 StandardScaler+KMeans(n_init20)，无 PCA；验证原运行的 calibration 指标和类别对应完全复现后冻结。
- 特征函数和 float32 运算保持原样；与旧七联图相同，先把绝对坐标转float32，再减各primitive初始中心。所有7,220个primitive共同计算每时刻涡量均值，不按batch重算。
- 对照报告精确坐标变换、float64运算、同一时刻邻居差、分类距离差。对照只诊断误差，不能替换主图算法、拷贝标签或调整阈值来制造不变性。
- 图只验证平移；不用于证明任意时变旋转下的严格客观性。方法结论只写入 experiment_log.md。

## 图形约定

问题：固定分类器下，平移 observer 是否改变同一物质 primitive 的特征或标签？图形为纵向七幅轨迹及计数，按用户要求保留全部速度档。a 为原参考系，b–f 为平移速度过渡，g 为全部平均速度抵消环境平移；不将它们当独立统计重复，无误差棒。

结构复用已有七联图 renderer；输入是 [7,N,7,97,3] 轨迹、[7,N] 独立分类和每档计数，显示所有中心轨迹的整条标签，统一物理范围和相机。蓝=非涡、红=涡，不作裁剪。输出 paper 7.2×17.085英寸/7pt、slides10.8×25.6275英寸/13pt、PDF/SVG/PNG和paper600dpiTIFF；这是用户指定的长条比较图，不声明符合单页期刊高度。Python backend。最终输出需字体、面板尺寸、碰撞及逐幅和全图检查。

配置：`config/Verify_AIVDTranslationObservers_1.1.json`。计算：`experiments/Verify_AIVDTranslationObservers_3D.py`。绘图：`experiments/Plot_AIVDTranslationObservers_3D.py`。不写模型checkpoint。
