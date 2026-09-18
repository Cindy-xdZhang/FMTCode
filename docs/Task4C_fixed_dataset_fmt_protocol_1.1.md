# mainExp_Task4C_FixedDatasetFMT_1.1：固定数据集 v1 上的固定中心 FMT 四臂比较

数据：`mainExp_Task4C_FixedDataset_1.1` 通过 `audit-data` 后的只读文件（180,000 / 3,000 / 12,000 行），`verify-source` 逐文件核对 `data_audit.json` 的 30 份冻结哈希后才允许训练；不修改、不重排、不重新积分。

方法：与 `Ablation_Task4C_FPS16_1.1/1.2` 完全相同的四臂——`p35_h0_fps16`（141 维、76,738 参数、以第 0 槽中心为起点 FPS 选 16 邻居）、`c156_fps16`（48 点重采样、加宽残差网络、992,386 参数、FPS16）、`p35_h0_six_control`（最近 6）、`c156_six_control`（FPS6）。全部固定中心、无卷积、无轮换；编码器与网络代码 `FMT_Utils/Task4C_FPS16_1_1.py`、`FMT_Utils/Task4C_OriginalCenter_1_1.py` 不改。

训练：seed 96611/96612/96613 各一次完整训练（共 12 次，V100 并发 4），batch 128、AdamW 0.001/0.0001、Dropout 0.15、最多 500 轮、50 轮早停、验证 F1（平均精度打破平局）选轮、阈值 0.5、训练集逐特征标准化；模型只驻内存。

报告：合并测试 F1/AP、分流场、**未见实例行**、**覆盖实例行**、覆盖实例偏移正类、GT 头部行、放松行，以及每个 GT 实例的正类召回（供可视化页面逐实例展示）；三种子均值与样本标准差。`audit-results` 复算 24 份预测的哈希、行覆盖、标签/实例/分组一致性与 F1/precision/recall。

链：`verify-source` → `gpu-check`（四臂 32 束小拟合）→ `train`（12 次）→ `audit-results`。代码 `experiments/Task4C_FixedDatasetFMT_1_1.py`；配置 `config/mainExp_Task4C_FixedDatasetFMT_1.1.json`（`source_audit_sha256` 在数据冻结后填入）；启动器 `ibex_bash/task4c_fixed_dataset_fmt_1p1.sh`。非 FMT baseline（Conv3D、PointNet、PointNet++、BiLSTM）在同一数据上以各自版本运行，见后续协议。
