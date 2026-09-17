# 二维FTLE：FMT卷积方案已撤销

2026-09-17起，FMT＋U-Net、FMT v5＋U-Net、距离＋夹角＋U-Net以及P35卷积融合超分辨全部退出有效FMT方法和结果表。完整模型使用二维空间卷积，违反用户现行协议，不能再称为当前最佳FMT或纯傅里叶方案。

[硬性协议](FMT_no_spatial_convolution_protocol_1.1.md)规定完整预测路径无卷积。当前没有在本页用新网络替换或重算这些成绩。原始逐次预测和日志保留；[撤销前完整表](archive/retired_fmt_convolution_20260917/paper_tables_ftle_2d.md)仅供核对历史，其中明确标名的独立ESPCN/U-Net/Raw和插值对照仍是原实验的历史结果，不并入FMT名下。
