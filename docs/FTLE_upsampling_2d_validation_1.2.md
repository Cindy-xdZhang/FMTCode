# FTLE 2D 1.2 验证记录

本地CPU环境：PyTorch 2.14.0+cpu、NumPy 2.4.6。
`outputs/Verify_FTLEUpsampling2D_Preflight_1.2/preflight.json`：原基线类哈希、五线输入、
频谱维数与前缀、物理积分、全部36种方法/倍率/dropout组合的输出、梯度与train/eval行为通过。
使用四个真实流场的首个训练窗口小样本作物理核验，不访问正式测试场。

`outputs/Verify_FTLEUpsampling2D_EndToEnd_1.2/`：只在doublegyre解析开发时段[0,0.9]，
2/1/1切片、tau0.1、3epoch、65×97网格、seed918，36次训练完成。
该时段早于正式doublegyre训练下限1.0，不用正式测试数据调试。
数据审计和72份预测独立复算通过；84行指标、42行汇总，无checkpoint。
完整本地执行日志：`tmp/ftle12_checks.log`。本地结果只验证工程正确性，不作性能结论。

正式Ibex科学提交、调度记录及最终独立复算将在运行后补充。
