# Task4-c 数据集 v3 执行记录

科学配置：`config/mainExp_Task4C_FixedDataset_3.1.json`；协议：[Task4C_fixed_dataset_protocol_3.1.md](Task4C_fixed_dataset_protocol_3.1.md)。科学 commit `e536ff7f00ce568af01f4f5bbeb1599e9e9b50ea`。私有部署目录 `/ibex/user/zhanx0o/FMT_Task4C_FixedDataset_3p1_20260919`。

目标两流场总计 40,000,000 个合格初始点，经 Poisson disk 降采样到 8,000,000 个目标样本；每流场各半。只增密，保持 v2 r3 科学规则和具体实例折归属。积分清洗后不补采，最终保留数以审计为准。没有训练或 GPU 作业，只有一套固定构建随机数设置。

## 已完成并冻结：2026-09-19 18:01:39 UTC 核查

Channel 构建 `52102609_0`、TBL 构建 `52102609_1`、全量审计 `52102610` 均 COMPLETED / 0:0，分别用时 **54分24秒、1小时42分9秒、9分39秒**。先前16:41快照中的“TBL正在积分、尚未冻结”已结束；本次依据最终审计更新，规则与数据未变。

最终保留 **7,762,378** 个样本点，每点三种长度，共 **23,287,134** 条曲线。Channel **3,793,240**（正605,155 / 负3,188,085），TBL **3,969,138**（正613,656 / 负3,355,482）。8,000,000个目标点中清洗删除237,622个，按协议不补采。两场具体实例→折映射均继承v2 r3，逐行规则核验与各500点独立重积分通过。

冻结审计 `data_audit.json`：`complete=true`，SHA-256 **`b584fc8e8f42dcc305754e550804eb3f8fb75c254fd76cf7272a9c6623312584`**。本地证据 `outputs/mainExp_Task4C_FixedDataset_3.1/remote_evidence/data_audit.json`；全部物理文件哈希记录在其中。后续训练另立 `Ablation_Task4C_BallQuery_3.1` / `Ablation_Task4C_BallQueryBaselines_3.1`，见 [Task4C_v3_training_execution_3.1.md](Task4C_v3_training_execution_3.1.md)，不修改已冻结数据。

## 提交

| 阶段 | Slurm 作业 | 依赖 | 资源 |
|---|---|---|---|
| 输入、来源与单元测试 | 52102607 | 无 | 4 CPU / 16 GiB |
| 两流场试构建及核验 | 52102608_0–1 | 输入核验成功 | 各 4 CPU / 32 GiB |
| 两流场正式构建 | 52102609_0–1 | 两个试构建均成功 | 各 4 CPU / 96 GiB |
| 全量数据审计 | 52102610 | 两个正式构建均成功 | 4 CPU / 96 GiB |

链按 `afterok` 放行，失败依赖自动取消；正式构建入口额外检查两份 `pilot_check.json`。未重复提交或追加种子。

## 启动核验

- 本地 15 项测试通过；远端输入核验作业再次运行全部 15 项，通过，耗时 2.171 秒。
- 部署包 SHA-256 `f875219d47f2a75411b17382d524b12c6e5ea18228e67494e4e2946cc76fbe1b`；18 份科学源码/配置/测试文件的 Git 内容哈希核对一致，导入与配置约束通过。
- `52102607` COMPLETED / 0:0，用时 19 秒；原始四个 VTK 哈希、v2 审计哈希与实例折映射来源通过。
- 提交前可用磁盘约 708 GB；运行时可用量另记 `resource_check.json`。
- 2026-09-19 15:23:42 UTC：两个试构建均 COMPLETED / 0:0，分别用时 65 / 87 秒；Channel 保留 37,871 / 40,000、TBL 保留 39,673 / 40,000。两个旧算法重建检查通过，每流场 500 点重积分一致。
- 正式构建 `52102609_0/1` 均 RUNNING（各已运行 15 秒），审计 `52102610` 等待构建成功；无错误堆栈。此时尚无全量 v3，不称已冻结。

实际证据：`outputs/mainExp_Task4C_FixedDataset_3.1/remote_evidence/`。`progress_latest.json` 是带 UTC 时间戳的状态快照，`submission.json` 是提交记录；最终数据冻结须以 `data_audit.json` 的 `complete=true` 和文件哈希为准。
