# 配置使用约束

2026-09-17起，FMT完整预测路径禁止任何一维、二维、三维及转置卷积，包括冻结Raw骨干。以 [FMT禁止空间卷积协议](../docs/FMT_no_spatial_convolution_protocol_1.1.md) 为准。

本目录包含历史配置，文件存在不表示方案仍获准运行。以下配置中的对应分支已撤销：Task3/5的raw_fmt与FMT残差（包括nTDO、p35/h0、n0_k06）、Task4-b raw_fmt、FTLEUpsampling2D的FMT＋U-Net、FTLEP35Fusion的卷积方案，以及model_zoo的FMT/DCT-FMT U-Net名称。相关构造入口直接报错；不允许修改名称或忽略错误后重跑。保留原配置仅供核对历史科学提交与哈希，不改写旧结果。

Task4-c的p35/h0、c156和Task3/5直接fmt_c156/ASAP全连接方案仍使用原配置与原网络。独立非FMT卷积基线不能混入FMT方案或指标名下。
