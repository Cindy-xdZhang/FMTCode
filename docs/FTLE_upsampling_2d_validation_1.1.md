# FTLE 2D 1.1 工程验证

> **2026-09-17撤销状态：** 本文涉及的Raw卷积＋FMT/残差或二维FMT卷积网络，已按用户要求从可用方案及有效FMT证据中删除；旧数值仅为撤销历史，不得用于当前主表、附录或“最佳FMT”结论。固定傅里叶特征本身及明确无卷积的Task1/Task4-c分支不受此撤销影响。以 [FMT禁止空间卷积协议1.1](FMT_no_spatial_convolution_protocol_1.1.md) 为准。


本地运行环境：Python3.11，PyTorch2.14.0+cpu，NumPy2.4.6；运行目录`outputs/`。
科学实验由独立的冻结提交在Ibex执行；以下不属于性能结果。

2026-09-14 初次预检通过：

- PyFlowVis原始源文件与提取后5个基线类的抽象语法树逐项相同。
- 五线输入、十对距离、六个夹角、时间列排除、七线拒绝全部通过。
- FMT65维与v5前65维逐位一致；距离＋夹角176维通过时变旋转及平移的数值不变检查。
- 常应变率0.4解析场的217步RK4和两种FTLE公式一致，误差≤1e-10。
- 六网络35步固定批次优化的最终归一化均方误差约0.00055–0.00237，
  相对于各自初始误差均降低至少75%。这是梯度与执行检查，不是超分辨成绩。
- 四流场第一训练窗口各16中心探针的有效FTLE，与独立二维最大特征值公式
  最大差异≤1e-8；源路径、哈希和有效数量见`preflight.json`。

完整流程小预检 `Verify_FTLEUpsampling2D_EndToEnd_1.1` 也通过：
只在doublegyre的开发时段[0,0.9]、tau0.1做缓存、数据审计、两倍率×六方法共12次
三epoch训练、验证选模型、预测、汇总及独立复算。正式doublegyre实验从t=1开始，
这些开发窗口不进入正式训练/验证/测试。该预检没有改正式超参数。
临时驱动为`tmp/ftle_end_to_end_check.py`；报告位于
`outputs/Verify_FTLEUpsampling2D_EndToEnd_1.1/{data_audit,result_audit}.json`。
全部模型仅驻留内存，无checkpoint。

训练前仍会在Ibex重新执行完整preflight与正式缓存audit-data；测试后执行
独立audit-results，复算每次预测、最佳epoch、逐次指标表与汇总表。
