# Task4-c 4.17：单个基线种子的原设备复核

4.16九次预定训练中，原FMT seed96611在A100节点执行，其他两个原FMT种子及全部六次新方法训练均在V100节点。原实验4.14为V100。96611的归一化均值/标准差最大差异为7.3217e-9/2.9369e-9，测试F1为0.6833225而原值0.6653747；96612、96613的归一化与训练/验证/测试F1完全复现。原输入缓存哈希不变；差异发生在不同节点环境下，不把它断言为单一GPU运算原因。signed-log在CPU通过NumPy执行，其指令实现也可能随CPU变化。

新增`Verify_Task4C_BaselineDevice_4.17`仅复跑原FMT seed96611，明确限定原Tesla V100-SXM2-32GB。数据、233维输入、模型、初始化种子、优化器、训练/选择/测试规则全部使用原实现。不重跑新方法、不选更好成绩、不修改4.16记录。运行依赖4.16原核验结束，不因该核验发现基线差异而隐藏失败。

验收要求归一化、验证选定epoch、训练/验证/测试F1均与原4.14完全一致。完整对比使用冻结4.14 FMT/Conv3D参照与4.16六次预定V100新方法结果；A100基线复跑作为保留的环境差异记录另列，不替换原基线。复核全40,000条原特征保留、各新方法原通道归一化、设备、逐样本指标及原Conv3D预测。

配置`config/Verify_Task4C_BaselineDevice_4.17.json`；入口`experiments/Task4C_BaselineDevice_4_17.py`。单个进程登记提交、开始、结束、节点/GPU、源码和配置哈希。仅内存保存最佳权重，无模型文件。4.16独立指标复算证据保留，最终有效对比写`validated_comparison.json`。方法结论仅写experiment_log。

## 完成及原设备复核记录

4.16科学commit `dfdc87b10f0d3de21d694e20f3dc97f7295dc66f`，配置SHA256 `660f8582c63a2d17ef332596685cec3249b59ba270e5ecd40ecedd1258e7faf1`；单个原设备复核4.17科学commit `e5de2877ca756973116d6dd2bfc8254c16d4ac4f`，配置SHA256 `6083ddbefd98e413ee4d7326df42b42d585cff9db73a0a0d01644b66f572d6f3`。最终有效对比`outputs/Verify_Task4C_BaselineDevice_4.17/validated_comparison.json`，SHA256 `c782f1f879d3e719d04fb77a70d2f230126edfe2d5299dda37f587fffab70968`。

4.16预检51887792、编码51887818、九次训练51887819[0-8]、汇总51887822完成；指标独立复算最大差异0。原核验51887823因检测到A100基线的归一化差异失败，失败和A100结果保留。4.17仅以V100重跑原FMT seed96611，job51888313完成；原233维归一化、验证选定epoch及训练/验证/测试F1完全复现。其余两个原FMT种子原已完全复现，两个新方法全部六次均为预定V100运行，未重跑/挑选新方法。最终检查覆盖全部40,000条特征保留、原通道标准化、预测、原Conv3D参照；14个调度进程（13成功、1保留失败）、28条生命周期记录核对PASS，无权重文件。运行审计代码commit7302a46b。

本地证据在`outputs/Verify_Task4C_RetainDirection_4.16`（124文件）及`outputs/Verify_Task4C_BaselineDevice_4.17`（15文件）。无模型证据包SHA256分别`7ce3ba4faf10ba266f5f44cc566c5aace86229f9e6bea6dc92b1399243b7454d`、`12f550fae5fce136e7c068f0cde88d0827ebc1a2ef99deb194b42dc270c8737b`。个人Ibex对应根目录为`/ibex/user/zhanx0o/FMT_Task4C_RetainDirection_20260914`及`/ibex/user/zhanx0o/FMT_Task4C_BaselineDevice_20260914`。

成绩及方法结论见experiment_log的task4c-retain-direction-4-16-2026-09-14。
