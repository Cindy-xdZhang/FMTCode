# Task6/7/8 流映射查询绘图索引

本页记录2026-09-10密集查询实验的绘图接口。全部采用Python；源数据和网络预测必须通过独立审计后才能生成正式图。新方向保留编码`vector_fmt6`与原161维`fmt_all`分别标记。模式a/b的旧三联图定义保持不变。

| 绘图方法 | 代码 | 数据接口 | 特点与适用范围 |
|---|---|---|---|
| 三任务、三类测试的共同数据对比 | `experiments/Report_Task678_HanAndVector_1_1.py` | 两个实验的`metrics.csv`、`runtime_config.json`、`independent_audit.json`，要求2376及999条审计记录 | 三栏为Task6/7/8，横轴区分新粒子、未见primitive、未见时间窗口；六种方案全部保留。九流场等权均值、三种子单独标记；无训练方法只有一个确定性结果。完整逐流场及physical-family汇总另存CSV。 |
| 九流场的密集查询训练曲线 | 同上，`plot_training()` | 两实验`evidence_metadata/runs/<dataset>/<arm>/seed<seed>/Task6_training.json`和Task7对应文件 | 九宫格；原FMT、新方向编码、Raw三种网络；实线/虚线表示Task6/7，细线为全部种子、粗线为均值。只使用监督时刻的固定训练探针，不平滑，不挑选epoch。Task8没有独立训练损失。 |
| 原FMT/Raw/零token/仿射逐流场误差图 | `experiments/Plot_Task678_HanSampling_1_1.py` | 已审计HanSampling的2376条结果 | 每类测试独立一张三栏图；逐流场显示全部种子及其均值，便于检查宏平均是否掩盖失败条目；附原FMT与Raw的训练曲线。 |
| 固定未见primitive的真实三维轨迹 | `experiments/Export_Task678_FixedExamples_1_1.py`、`experiments/Plot_Task678_FixedExamples_1_1.py` | 导出的`fixed_examples.json`及九个小型NPZ；原缓存/预测逐文件SHA256校验 | 每流场3×4面板，三行Task6/7/8，四列真值、原FMT、新方向编码、Raw网络。固定第一个测试primitive、前8个保存的query、seed9110，不按误差或外观选图。同一行统一正交相机与包含所有轨迹的取景范围；不画bounding box，浅灰显示可见支持轨线。Task6只展示第一段，定量评价仍含两段。 |

正式图输出可编辑PDF/SVG和600 dpi PNG，附图形合同、源CSV或样本清单、面板尺寸记录、字体与碰撞检查。静态代码预检不能代替最终图片检查；最终渲染结果与审计状态以实验输出文件为准。

统一指标为三维位置欧氏均方根误差除以初始primitive半径；Task8正式结果传递预测终点。真实到达位置的第二段诊断使用第二段半径归一化，不能直接与整段Task8误差相减来估计误差来源。Token字节数只描述每个可见primitive的编码大小，不包含神经网络参数和查询几何信息。

主要输出目录：`outputs/Task678_HanAndVector_1.1/`。方法性能与其支持的结论只记录在`docs/experiment_log.md`。
