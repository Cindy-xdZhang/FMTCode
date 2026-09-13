# 仓库维护与历史验证归档

最新一轮见[2026-09-13继续整理](#cleanup-20260913)。下方前两轮计数是各轮当时的快照，不能相加当作本轮删除数量。

## 2026-09-05 整理

应开源前清理要求，将不再被保留工作流程引用的历史验证文件移出工作目录：
84 个实验脚本、299 个配置、453 个 Slurm 提交脚本和 161 个配套测试，共 997 个文件。
原始文件逐字节保存为本地 `.research_archive/retired-validation-20260905.zip`，
所有成员经过 SHA-256 校验后才从原路径移除。归档另外保存了整理前的 README 和验证摘要。

归档目录被 Git 忽略，不属于开源代码分发；不要把本地唯一备份当作远端已经保存的备份。
[完整清单](research_archive_manifest.json) 记录原路径、字节数、每个文件及压缩包校验值。
这份清单是源文件维护记录，不是实验指标。
历史方法与负结果归入 [同一张验证总表](Verify_experiments.md)，
[实验流水](experiment_log.md) 与 [作业登记](ibex_run_registry.md) 不删行。

未移动算法库、实验输出、数据、模型或 Ibex 上的文件；没有提交 Git，也没有重新跑实验。
旧文件名如 `Verify_HighReVAE.py` 仍会少量出现，因为当前主实验调用其中的训练函数。
本次不重写这些冻结实现；保留的历史依赖列在清单的 `retained_historical_dependencies` 中。

## 校验与恢复

从仓库根目录执行，工具只需 Python 标准库：

```bash
python tools/research_archive.py verify
python tools/research_archive.py check
python tools/research_archive.py restore
```

`restore` 按原路径恢复全部 997 个源文件；已有相同文件会跳过，遇到已修改文件会拒绝覆盖。
README 和摘要快照不自动恢复，避免覆盖整理后的文档；需要时可从 ZIP 中单独取出查看。
只有本机归档仍存在时才能使用该恢复命令。公开发行若要求重跑全部历史搜索，须另行发布
这份源文件归档或相应完整 Git 历史；默认发行只保留当前代码、必要依赖和历史结论。

`plan` 是只读依赖检查；`apply` 是本次整理的执行入口，已有归档或清单时拒绝重复执行。
它保留源文件引用、Python 模块引用及可用冻结清单中的间接引用，并一并归档依赖旧文件的测试。
动态构造路径仍须结合实际运行检查；静态检查不能代替数据和 GPU 上的端到端复跑。

本次检查：997 个归档成员及两份文档快照全部通过校验；598 个保留的代码与配置文件
未发现对已移除文件的静态引用，Python 语法检查通过；5 项归档安全/恢复测试通过。
仓库布局检查通过，检查了 33 份 Markdown。顺带将 10 份保留的 Slurm 脚本中的
60 处直接文件调用统一为 `python -m experiments.<module>`，修正作业登记表两行竖线转义；
执行模块、参数和实验算法不变。噪声实验两个运行器及扰动实现的 SHA-256 与整理前一致。
本机 Python 缺少 PyTorch 等研究依赖，未运行完整科学计算测试或重新提交 GPU 实验。

## 后续约定

实验版本属于配置、指标和实验流水，不等于必须新建一个执行脚本。同一执行逻辑应复用入口，
以配置描述变体；仅当算法确实改变且需保留旧实现时才增加版本实现。
仍被当前模型、审计或数据准备调用的旧实现不得按名称批量删除。
历史配置可在完整备份并检查依赖后归档，失败与取消结论必须保留。

发布前仍需单独核查数据获取说明、依赖环境、许可证和机器专用路径。本次文件整理不等于已完成
完整的开源发布检查。

## 2026-09-13 整理：删除已完结实验的审计/汇总/校验脚本与对应测试

按用户要求，从工作目录删除 `experiments/` 下 38 个 `Audit_`、`Summarize_`、`Verify_` 脚本和 `tests/` 下 32 个测试，共 70 个文件。
删除条件：文件已提交到 Git 且本地无改动；对应实验的结论与证据已写入 `experiment_log.md`、`ibex_run_registry.md`
或对应协议文档；删除后 `experiments/`、`tests/`、`FMT_Utils/`、`FLowUtils/`、`DeepUtils/`、`pnn/`、`tools/`、`config/`
中没有任何 Python 或配置文件再引用它们；docs 中没有 Markdown 链接指向它们。
恢复方式：`git checkout ee576e26 -- <path>`（删除前最后一次提交）。未提交 Git，未改动算法库、实验输出、数据或 Ibex 文件。

保留且未按名称删除的例外：
- 仍被其他代码导入的旧实现：`Verify_HighReVAE.py`、`Verify_HighReSampling3D.py`、`Verify_3DFMTHyperparam.py`、
  `Verify_Task3_FMTClassifier.py`、`Verify_Task3_FMTResidual.py`、`Verify_Task4B_FullVolumeMemorization_1_1.py`、
  `Verify_Task4B_PooledMemorization_3_1.py`。
- 实验尚未完结或结论尚无记录：`Audit_GeometryParameterStress.py`（`Verify_Task123_GeometryParameterStress_1.1` 登记表仍为 RUNNING/QUEUED，日志写“暂无性能结论”）、
  `Audit_Task678_PerDataset_1_1.py` 及其依赖 `Audit_Task678_HanFast_1_1.py`、`Audit_Task678_VectorFast_1_1.py`（文档无任何记录）。
- 算法库单元测试、Task6/Task36 当前部署链在 Ibex 上作为预检运行的测试、绘图脚本契约测试、数据准备（标签/缓存/分片）测试，不属于实验结论测试。
- 从未提交到 Git 的 47 个候选文件（25 个脚本、22 个测试，`git status` 中为 `??`）未删除：删除后无法从历史恢复，需先提交或另行决定。

副作用：32 个历史 Slurm 启动脚本（`ibex_bash/`）仍以 `python -m experiments.<module>` 调用已删除模块，它们对应的作业均已完结登记；
如需重跑须先恢复模块。校验：`python tools/research_archive.py check` 通过（836 个保留文件无对已移除文件的引用，语法有效）；
`experiments/`、`tests/`、`tools/` 全部 `py_compile` 通过。`tools/Validate_Repository_Layout.py` 报告的失败项
（`docs/Verify_*.md` 拆分文档、`ibex_bash` 直接脚本调用、`third_party/point_nn/README.md`）在本次整理前已存在，与删除无关。
本机 Python 未安装 pytest 与研究依赖，未运行测试。

<a id="cleanup-20260913"></a>

## 2026-09-13继续整理：进展汇总、诊断归档与调用链修复

本轮开始时工作树干净，源版本为 `67139f80`（完整提交号见下方清单的 `source_commit`）。
按用户本次要求，将已经记录的检查及分散验证说明合并整理，并保留可恢复源文件。

| 操作 | 文件数 | 范围 |
|---|---:|---|
| 移出一次性检查/报告 | 22 | 客观性直接核验、FMTAllV2报告、Task6已完成报告与记录器等；仍被调用的历史训练辅助函数保留 |
| 移出过时提交脚本 | 46 | 原执行模块已被上一轮删除，或只调用本轮退役诊断；保留的提交链没有依赖这些入口 |
| 移出历史测试 | 20 | 已完成或暂停方向的独立检查；核心编码、积分、标签、绘图及被集群调用的预检保留 |
| 合并验证文档 | 12 | 原文集中到`Verify_experiments.md`末尾，配套最新问题索引；原始字节另存归档 |
| 修复上一轮误删的必要依赖 | 恢复10 | 从`f13a774942ac43ed8fad3b99216831a2021019ae`原样恢复7个审计模块及3个集群预检测试 |

共移出100个文件，恢复10个必要文件，新增1份机器可读清单；工作代码与文档文件净减少89个。
所有移出文件都有Git历史，且在移除前另做逐文件SHA-256归档校验。历史作业登记、逐次指标、数据、配置和冻结算法库不改写。
没有提交Git、访问Ibex或启动新实验。

[逐文件清单](research_cleanup_20260913_manifest.json)记录路径、原始大小、校验值、记录中可匹配的实验ID、移出原因与恢复依赖的源commit。
`recorded_ids`为空不表示运行成功；可能是通用测试、图形检查或失效入口，其状态仍须查实验流水。
未完成的GeometryParameterStress检查、Task678逐数据审计及其依赖继续保留，不能按名字当成已完结实验删除。
`Verify_HighReVAE`、`Verify_Task3_FMTResidual`等被调用实现、全部三联图渲染和数据资产构建代码保留。
部分Task6报告也被部署文件清单引用，继续保留；未通过修改冻结训练链来强行减少文件。

### 本轮归档恢复

归档为本地 `.research_archive/retired-diagnostics-20260913.zip`，只含源文件与整理前文档快照，不含模型或数据。
该目录仍被Git忽略，清单随代码保留；因此本地ZIP不是已经同步到远端的备份。
已有Git历史也可恢复单个原始文件；不要将恢复旧脚本解释为可以在当前版本随意重跑旧实验。

```bash
python tools/research_archive.py verify --manifest docs/research_cleanup_20260913_manifest.json
python tools/research_archive.py check --manifest docs/research_cleanup_20260913_manifest.json
# 需要完整恢复这100个文件时执行；已修改的同名文件不会被覆盖。
python tools/research_archive.py restore --manifest docs/research_cleanup_20260913_manifest.json
```

不带`--manifest`仍操作09-05原归档，原归档和原清单没有改动。恢复本轮会重新产生十二份独立验证文档，
因此日常阅读使用合并总表；完整恢复适合历史复现，恢复后布局检查会重新提示拆分文档。

### 文档与维护工具的实际变更

最新研究状态统一登记到AGENTS、README、总协议与实验流水；论文主表数字保持。
合作者简介中混淆原FMT/同时间Gram函数，以及把nTDO两版都当作距离编码的错误已纠正，前后解释并列记入实验流水。
三联图索引明确用户指定的背景/分类/误差布局与既有Task3/5参考/Raw/FMT实现的区别，没有重新渲染。

归档工具增加批次清单选择、文档恢复和根目录启动器/测试模块引用检查；AIVD报告打包器改为附带合并后的验证文档。
两个旧相机渲染提交入口改为`python -m`调用，模块与参数不变。布局检查区分第三方README与科研文档，
并允许实际存在的根目录Submit启动器记录运行事件；没有关闭实验脚本位置与文档链接检查。

### 验证范围

100个归档成员校验通过，已在仓库内隔离临时目录完整恢复并逐字节比对；恢复不会覆盖冲突文件的测试通过。
归档工具7项标准库测试通过；817个保留源文件/配置的两批归档引用检查、411个Python文件语法检查、
Slurm字面模块/预检引用和72份Markdown布局检查通过，`git diff --check`通过。
实验流水1051行历史表格与论文表164行历史表格均与整理前逐行相同。
本轮没有运行依赖PyTorch与真实流场的科学计算测试，静态检查不等于端到端科研实验复跑。
