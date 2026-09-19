# Task4-c 3h ball query 2.5：Ibex执行记录

科学commit：`ea1cf5d31ee5b3b6e64bce55996814f5eeb3f790`。私有部署目录：`/ibex/user/zhanx0o/FMT_Task4C_BallQuery_2p5_20260919`。未公开推送；旧1h和其他独立实验不受本轮影响。

五模型×两邻居数，共十次正式拟合，唯一seed96611。p35/h0、c156、Conv3D16³、BiLSTM、PointNet均测试k6/k16；起步3h，h固定Channel0.003909492、TBL0.117239990。科学规则见 `Task4C_ball_query_protocol_2.5.md`。

## 提交与索引

| 阶段 | Slurm作业 |
|---|---|
| 冻结源数据/共享邻居核验 | 52094397 |
| 四FMT GPU预检 | 52094398 |
| 四FMT正式训练 | **52094399[0–3]** |
| FMT预测复算 | 52094400 |
| k6/k16 × Channel/TBL导出 | 52094401[0–3] |
| 基线只读复用和文件核验 | 52094402[0–1] |
| 六基线GPU预检 | 52094418[0–5] |
| Conv16体素编码，两k×两流场 | 52094419[0–3] |
| 六基线正式训练 | **52094420[0–5]** |
| 六基线预测复算 | 52094421[0–5] |

FMT索引0=p35/h0-k6、1=p35/h0-k16、2=c156-k6、3=c156-k16。基线索引0=Conv16-k6、1=BiLSTM-k6、2=PointNet-k6、3=Conv16-k16、4=BiLSTM-k16、5=PointNet-k16。全部按成功依赖衔接，预检失败不会继续训练。

提交后调整调度依赖以缩短等待：52094420_1/2/4/5（BiLSTM、PointNet）仅依赖各自52094418同索引预检成功，不等待不使用的Conv体素缓存；Conv编码52094419并发上限改为2，整个实验仍最多十张GPU。Conv两组训练仍等待四份编码全部成功。只调整Slurm依赖和并发，未修改科学源码、配置、数据、训练预算或种子；实际`scontrol`记录保存在`scheduler_adjustments.json`。

## 已核验的证据

本地14项测试通过；远端同14项通过（3.38秒）。66份科学源码/配置/测试文件与Git文件内容的SHA-256逐一吻合；上传的h文件和邻居NPZ字节哈希吻合。源核验52094397已COMPLETED/0:0，用时8秒。

初始3h计数与2.4可视化的387829个样本逐点一致；另对每流场34个球心直接扫描所有点，用原NumPy FPS复核k6/k16邻居编号和扩张半径，全部通过。初始球Channel平均13.543756、范围0–32，TBL平均30.357648、范围0–85。需要扩球的数量：Channel k6=14659、k16=125532；TBL k6=4398、k16=31768。

2026-09-19 12:41 UTC启动核验：四FMT GPU预检52094398和六基线预检52094418_0–5均COMPLETED/0:0；18份源核验、预检、pilot及数据核验报告全部通过，科学identity均为ea1cf5d3。四FMT正式训练52094399_0–3均RUNNING，四组都已经写入至少一轮、每轮826005行的训练历史；Conv编码52094419_0/1 RUNNING，_2/3等待数组并发槽；四BiLSTM/PointNet训练等待Priority，Conv两组等待编码。所有错误日志为空，尚无正式测试F1。

当前各阶段的实际状态以 `outputs/Ablation_Task4C_BallQuery_2.5/deployment_status.json` 及其`remote_evidence/`副本为准。提交不等于训练完成，GPU小拟合结果不作为正式测试F1。

证据目录：`outputs/Ablation_Task4C_BallQuery_2.5/remote_evidence/`（实际提交清单和远端阶段报告）、`independent_geometry_check.json`（本地全量计数/独立选邻居复核）。原始训练输出分别在FMT的`arms/`及基线的`outputs/Ablation_Task4C_BallQueryBaselines_2.5/k6|k16/<family>/`。正式指标只在预测复算通过后记录。
