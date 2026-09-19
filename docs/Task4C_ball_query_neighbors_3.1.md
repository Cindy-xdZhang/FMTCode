# Verify_Task4C_BallQuery_3.1：v3 共享邻居表

冻结数据 `mainExp_Task4C_FixedDataset_3.1`：7,762,378 点（Channel 3,793,240；TBL 3,969,138），审计 SHA-256 `b584fc8e8f42dcc305754e550804eb3f8fb75c254fd76cf7272a9c6623312584`。构建与全量审计全部成功。

保持 BallQuery 2.5 的全部科学邻居规则：h 为 Channel 0.003909492 / TBL 0.117239990；初始半径 3h；在同流场全部样本查球，排除原中心；不足 k 时扩大到第 k 个其他点，多于 k 时在完整球内以原中心开始做最远点采样（farthest point sampling，FPS），距离并列取小样本号；k6/16 独立处理。几何可跨折查询，标签与折不参与选邻居。所有后续模型共享本表。

只改内存与循环开销：每次至多 2048 个球心物化候选列表，保持旧浮点比较与边界；Numba FPS 将短向量差改为逐坐标差，不启用 fastmath。稀疏、稠密、边界并列及不同分块尺寸与 2.5 所有输出逐项完全相等。每流场另抽 32 个球心全域直接计算距离，并用旧 NumPy FPS 核对。

入口 `experiments/Prepare_Task4C_BallQuery_3_1.py`，配置 `config/Verify_Task4C_BallQuery_3.1.json`，核心 `FMT_Utils/Task4C_BallQuery_3_1.py`。输出每流场 `order6.npy`、`order16.npy`、初始计数、扩球标记、半径和最大选中距离，以及含文件哈希的 `manifest.json`。

沿用测试折 0，训练折内用原固定种子 98401 和原规则划出 10% 验证样本；每点三长度同行划分。最终行数：训练 16,521,207、验证 1,835,691、测试 4,930,236。此阶段仅 CPU 邻居预处理，无训练、无测试性能选参。
