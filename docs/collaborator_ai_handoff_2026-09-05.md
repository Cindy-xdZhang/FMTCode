# 协作 AI 交接说明（2026-09-05，Task4 接手会话）

本文供并行工作的 Codex 会话与后续 AI 会话阅读。权威定义仍是 `AGENTS.md` 与
`docs/research_tasks_and_protocol.md`；本文只记录 2026-09-03 至 09-05 由 Claude 会话接手
Task4 后新增的状态、约定与待办，不覆盖任何协议。全部实验行已按既有表头追加到
`docs/experiment_log.md`（8 列）与 `docs/ibex_run_registry.md`（10 列），
`tools/Validate_Repository_Layout.py` 校验 PASS。

## 1. 本轮完成的实验（均已落账）

| 实验 ID | Ibex job | 结果 | 证据目录 |
|---|---|---|---|
| `Other_Task4A_FMTStreamlineClustering_1.1`（Ibex 复现） | 51270590 | 4/73、soft .2662，与本地逐字段一致 | `outputs/Other_Task4A_FMTStreamlineClustering_1.1/ibex_repro_20260903/` |
| `Verify_Task4B_FullVolumeMemorization_1.1`（Ibex 复现） | 51270629 | 9/9 零错误 PASS，审计 PASS | `outputs/Verify_Task4B_FullVolumeMemorization_1.1/ibex_repro_20260903/` |
| `mainExp_Task4B_PooledInstanceSplit_3.1` | 51277193 / 51277194 | Raw+FMT macro-F1 .4196，相对 Raw +.0356（3/3），相对 Raw-wide +.0510（3/3）；审计 PASS | `outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/` |
| `Verify_Task4B_PooledMemorization_3.1` | 51339155[0-8] / 51339348（正式）；51335442（串行，TIMEOUT） | 9/9 零错误 PASS，审计 PASS | `outputs/Verify_Task4B_PooledMemorization_3.1/pooled_sharded/` |

详细协议与结果：`docs/mainExp_Task4B_PooledInstanceSplit_3.1.md`；跨实验结论已并入
`docs/Verify_experiments.md` Task4 小节；协议文档 §1 与 Task4 小节各加一句 3.1 的边界。

## 2. 新增代码与文件（已随 Codex 的整理 commit `894fe1aa` 于 2026-09-05 14:18 提交）

- `FMT_Utils/Task4B_PooledSplit_3D.py`：实例分块拆分、Voronoi 归属、缓冲证书。
- `experiments/Build_Task4B_PooledInstanceSplit_3_1.py`、`Train_Task4B_PooledInstanceSplit_3_1.py`
  （复用 1.2 训练循环 `Train_Task4B_FourClassClassifier_1_1._train_one`）、
  `Audit_Task4B_PooledInstanceSplit_3_1.py`（不导入训练器）。
- `experiments/Verify_Task4B_PooledMemorization_3_1.py`、`Audit_Task4B_PooledMemorization_3_1.py`：
  1.1 记忆脚本的版本化拷贝，只改契约字面量；审计的 source/voxel 唯一性改为 (volume, index)。
  `experiments/Merge_Task4B_PooledMemorization_3_1_Shards.py` 合并分片 summary。
- `experiments/Render_Task4A_FMTStreamlineClustering_1_1_Views.py`（从 NPZ 渲染 4A 视图）、
  `experiments/Render_Task4B_ChannelToTBL_2_3_Views3D.py`（2.3 的三维立方体渲染，含 x 窗口特写）。
- `config/mainExp_Task4B_PooledInstanceSplit_3.1.yaml`、`config/Verify_Task4B_PooledMemorization_3.1.yaml`。
- `ibex_bash/repro_task4a_fmt_streamline_1p1_cpu.sh`、`repro_task4b_memorization_1p1_gpu.sh`、
  `mainexp_task4b_pooled_3p1_{build_cpu,train_gpu}.sh`、
  `verify_task4b_pooled_memorization_3p1_{gpu,array_gpu,merge_cpu}.sh`。
- `tests/test_task4b_pooled_split_3d.py`（可用 pytest 或直接 `python` 运行；本地与 Ibex 均无 pytest）。

与 `docs/repository_maintenance.md`“同一执行逻辑复用入口、以配置描述变体”的约定相比，
`Verify_Task4B_PooledMemorization_3_1.py` / `Audit_Task4B_PooledMemorization_3_1.py` 是 1.1 的
版本化拷贝：原因是 1.1 脚本把实验名、行数与类别支持写死在 `_validate_spec_contract` /
`_validate_config` 里，且 1.1 已冻结不可改。后续若再做记忆验证，建议把这些契约字面量改由 config
提供，合并回单一入口；本轮未改动 1.1 原件。

## 3. Ibex 部署约定（与既有 tar 快照方式一致）

- Task4 部署目录：`/ibex/user/zhanx0o/FMT_Task4_Repro_20260903/FMT_Task4_Repro_20260903`
  （home 配额 180G/200G 接近上限，故放 `/ibex/user`）。快照 tar SHA `51703a78…fef3`，
  base commit `22430b2e`；之后的增量文件用 scp 覆盖并以独立清单核对：
  `DEPLOYMENT_MANIFEST.sha256`（复现）、`DEPLOYMENT_MANIFEST_Task4B_3p1.sha256`、
  `DEPLOYMENT_MANIFEST_Task4B_3p1_memo.sha256`。
- 该目录内放了一个仅含 HEAD 的浅克隆 `.git`（`git rev-parse HEAD` = `22430b2e`），因为记忆脚本
  用 `git rev-parse HEAD` 记录 `git_head` 且失败即退出。
- 数据：`channel.vtk`/`channel_GTs.vtk` 首次上传至 `/ibex/user/zhanx0o/FLowDataFolder/channel_flow/`
  （SHA `c1e3c18d…d564`/`b9fdd93a…ac09`）；channel proxy 标签 NPZ 与 Task4-b 1.2 冻结 cache 已放在
  部署目录的原相对路径下；TBL 2.3 target cache 直接引用
  `/ibex/user/zhanx0o/FMT_Task4B_ChannelToTBL_2_3/repo/outputs/.../target_cache/`（逐 chunk SHA 核对）。
- GPU 排队经验：`--constraint=a100|v100` 在拥堵时预计等待 7 小时以上；放宽为 `a100|v100|p100`
  后立刻分配到空闲 P100。P100 上 1024 宽记忆模型的 Raw 臂约 13.7 秒/epoch（91,711 行），
  三臂串行会超 6 小时时限，应按 (variant, seed) 分片提交数组再合并。

## 4. 遗留待办（接手时已存在，本轮未处理）

1. `mainExp_Task4B_ChannelToTBL_2.3` 的 Ibex 正式复跑 job 51221314 已 COMPLETED（55 分钟），输出在
   `/ibex/user/zhanx0o/FMT_Task4B_ChannelToTBL_2_3/repo/outputs/`，尚未下载、审计、落账。
2. `Verify_Task123_NoiseRobustness_1.1` 全部作业 COMPLETED、独立审计 PASS，结果仍只在远端
   `/home/zhanx0o/FMT_Uniform_3D_20260901/outputs/Verify_Task123_NoiseRobustness_1.1/`，未落账。
   （Codex 会话 09-03 起在 Task1–3 上继续做 StrongBaselines 1.2 等复制实验，registry 已有其行。）
3. 本轮新增文件与文档改动已包含在 Codex 的整理 commit `894fe1aa` 中（该 commit 同时把 997 个历史验证文件移入本地归档，见 `docs/repository_maintenance.md`）；本文自身与协议文档中新增的两句 3.1 边界说明尚未提交。

## 5. 对 Task4 现状的判断（供讨论，不是结论）

合并数据可被三种输入完全记忆，说明容量与管线不是瓶颈；同分布留出完整实例的四分类只有
macro-F1 .42，约 42% 的 hairpin cube 被判为 ordinary 且不因 FMT 改善。缺的是单 primitive 之外的
实例级上下文。可选方向：邻域/连通块聚合后再判别，或把相邻实例合并成"包"拆分以少删训练数据。
