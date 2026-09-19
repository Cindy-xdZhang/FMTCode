# Task4-c ball query 2.2 执行记录

2026-09-19，科学提交 `a5af710d72a74bbbceb8bd5a48db8b26c1f16c74`，执行修订 `r2_global_h_user_confirmed`。用户最终确认 h 使用全流场最小相邻网格间距。早期局部h草案 `e0616ee5` 仅做本地诊断，虽然曾上传部署包，但未执行其部署脚本、未提交训练；本轮训练全部使用固定全局h版本。

远端私有目录 `/ibex/user/zhanx0o/FMT_Task4C_BallQuery_2p2_20260919_r2`。复用 `/ibex/user/zhanx0o/FMT_Task4C_FixedDatasetBaselines_2p1_20260919/outputs/mainExp_Task4C_FixedDataset_2.1`，只读符号链接，无数据重建。输出子目录 `outputs/Ablation_Task4C_BallQuery_2.2/r2_global_h/`。

| 阶段 | 作业 | 已观察状态 |
|---|---|---|
| 源文件/数据/邻居表核验 | 52090742 | COMPLETED，0:0，30秒 |
| V100 工程预检 | 52090743 | COMPLETED，0:0，46秒 |
| p35/h0 ball6，seed96611 | 52090744_0 | RUNNING |
| p35/h0 ball16，seed96611 | 52090744_1 | RUNNING |
| c156 ball6，seed96611 | 52090744_2 | RUNNING |
| c156 ball16，seed96611 | 52090744_3 | RUNNING |
| 预测独立复算 | 52090745 | PENDING，成功依赖四臂训练 |

以上是检查时的状态，不是自动更新的实时页面。目前没有新正式测试F1。不得把小拟合F1=1.0当正式成绩。无需再提交；下一步读这条作业链和对应输出即可。

## 核验

- 本地七项ball query测试、四项原v2接口测试、九项构建测试通过；远端运行其中七项ball query与四项v2接口测试，全部通过。
- 41份代码/配置/协议与科学Git提交的blob哈希一致；本机九份文件与Git blob仅有CRLF/LF差异，已逐文件核对，未隐瞒成原始字节全同。
- 数据六文件与冻结audit核对通过；训练/验证/测试行数826,005/91,779/245,703一致。
- 远端和本地邻居NPZ文件的压缩包SHA-256不同，统计均值有约1e-18的求和舍入差别。此前整体JSON严格相等检查因此失败；改为下载原文件逐数组检查后，全部字段（包括邻居编号、初始数量、最终半径、距离、扩张标记）**逐值完全相同，最大差0**。差异没有改变训练输入。证据 `remote_evidence/neighbor_array_comparison.json`。
- GPU预检：Tesla V100-SXM2-32GB；四臂输入 `[96,1,142]`，参数76,738/992,386，递归无卷积检查通过；跨batch编码检查通过；150步小拟合均F1=1.0（仅工程检查）；峰值CUDA已分配内存1,570,728,960字节。

## 几何结论与展示

两流场全体样本固定1h球均无其他点。自适应扩大后，全部样本的邻居恰好是最近6/16个点（无第k距离并列），本轮实际没有需要FPS削减的球。最终半径中位数：Channel k6=0.008322958（504.10h）、k16=0.012495897（756.85h）；TBL k6=0.175785320（24.30h）、k16=0.264877431（36.61h）。半径无上限，仍不能保证同一hairpin内取邻居。

统计页 `http://127.0.0.1:8767/Verify_Task4C_BallQueryCounts_2.2/viewer/index.html`，默认固定全局1h；可选0.25–2048h的预计算半径、流场、集合、中心标签，显示分布及最小/中位/最大值。命令行 `Analyze_Task4C_BallQuery_2_2 --h-mode global_min --radii ... --output ...` 可算任意正半径，禁止用测试F1自动挑半径。页面已实际检查初始值、流场切换、滑块和绘图。

本地证据：本输出目录的 `remote_evidence/{submission,source_verification,gpu_check,deployment_verification}.json`，以及 `deployment_status.json`。代码/文档审查另见 `docs/Task4C_v2_review_2026-09-19.md`。
