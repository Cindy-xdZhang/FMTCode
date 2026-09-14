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

正式科学commit：`25578a8fe650e15b87f8d58731ced2f485df5cee`。
主分支等价集成commit：`51df52e3`。Linux正式config SHA256：
`527de631b4b7e043fd76c04b9c5fe8250ef2a13418231d107794e018fe285ff5`；
Windows工作文件可能使用CRLF，正式哈希以科学commit检出的Linux字节和runtime记录为准。

正式Ibex目录：`/ibex/user/zhanx0o/FMT_FTLE2D_Dropout_20260914`。
作业51890330、51890331[0-3]、51890332、51890333[0-35]、51890334、51890335，
共44进程；全部COMPLETED、退出0:0，88条runtime事件逐项核对通过。
完整节点、起止时间、设备、提交命令与配置见output的`submission.json`及
`scheduler_and_evidence_audit.json`，登记见`docs/ibex_run_registry.md`。

432次正式训练、2592份预测、2736行逐切片指标、168行汇总独立复算PASS。
每个流场/倍率/种子的12次训练共享同一GPU进程；每种方法dropout0/0.15的参数量和
训练归一化量完全相同。所有432次运行的科学commit、config与源代码哈希一致。
逐次selection锁定记录、验证最优轮次和测试读取时间均已核验；测试文件身份、真值、
掩码与原数据一致。所有流场的积分窗和源插值帧跨split均无交集。

独立新旧数据核验：1.1的144份2/4×记录在1.2中保留相同时间、源文件、有效性、
低/高FTLE值；最大逐点FTLE差为0。记录`frozen_time_and_target_audit.json`。
各倍率训练patch数量固定：Cylinder38，Boussinesq207，Piped cylinder193，Double gyre252。
三个倍率评价掩码取交集，只覆盖报告列出的共同区域；不可把新旧指标直接相减归因于dropout。

本地结果目录：`outputs/Other_FTLEUpsampling2D_1.2`；已保存逐次预测、指标、训练曲线、
选择证据、manifest、调度日志与审计；未下载大数据缓存，不包含任何模型文件。
模型权重始终只在进程内存中。报告由`experiments/Report_FTLE_Upsampling_2D_1_2.py`
从审计后的表生成；完整结果见`docs/paper_tables_ftle_2d.md`的1.2部分。
方法结论仅写入`docs/experiment_log.md`的`ftle-upsampling-2d-dropout-results-2026-09-14`。

结果包`tmp/ftle2d1p2-results.tar.gz`：691,100,567字节、4,035个结果文件；SHA256
`3df157708ac3eff991a4185c60204b37a43306559847bfff360353526bd6a60d`。
下载后独立校验哈希，并检查归档所有路径都在本实验output目录内、均为普通文件，
才解压到本地。训练设备统计为A100 180次、V100 252次；同一配对组始终使用相同GPU。
调度起止时间与runtime UTC时间交叉核对通过。
