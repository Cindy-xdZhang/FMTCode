# Other_FTLEP35Fusion_2.2 核验记录

## 提交前工程检查（2026-09-16）

本地解析/导入检查通过。复用原2.1科学预检，2×的五线二维形状、逐特征均值/最大值、直接三角函数计算傅里叶系数、零频形变关系、冻结参考网络及几何梯度检查全部通过。

16个冻结科学源文件按Linux换行后的SHA256与2.1锁记录相同；正式Ibex锁使用实际文件字节哈希再次核验。新程序的`infer`、`inference_time`、`train`、`merge`和独立结果审计`audit`共五个函数与原2.1抽象语法树完全一致；编码、网络、优化直接调用冻结实现。矩阵为12个流场/种子组合，每组6方法，共72次拟合。

工程输出：`outputs/Verify_FTLEP35Scales_2.2/preflight.json`和`checks.json`。工程检查不是性能结果；GPU预检、正式源数据和结果检查在调度任务中执行。

科学commit：64378a0554416a84cb1c5f907425edc613b1c940。
配置SHA256：5ee322ef61486da07409e3a21eb0c9a7f03914d6f42f9b651211a5bd967163a7。

后续完成状态在本文件追加，不覆盖失败证据。


<!-- ftlep35-scales-validation-complete -->
## 正式完成

GPU预检51929949通过，2×参数数：ESPCN153,476；U-Net467,716；四组融合网络均208,228。60份训练/验证特征文件与原数据/五线编码一致，完整时间隔离证书通过。72次正式训练全部GTX1080Ti。

独立重算72次训练的432预测：PASS；训练归一化、验证最优轮次、测试读取边界和节点约束：PASS。两种插值在四流场/三倍率/验证及测试上的144预测、120行总表及CSV：PASS。三倍率真值与mask逐元素相同。4×/8×来源文件与本地已核验2.1文件哈希相同；24组插值均值复现1.2历史结果；全部方法和倍率训练确定的PSNR范围相同。

24进程全部COMPLETED/0:0，48条runtime事件与Slurm核对PASS。825归档文件（824条内容清单＋清单自身）已取回并逐一核对，SHA256为c33e3855d9ba47dbfbf38a66fb6c87a0eb23d719d16efd506d273bec37f001b5。没有模型文件；原始缓存未下载。

工程检查与科学指标分别保存；完整记录见outputs/Other_FTLEP35Fusion_2.2的audit.json、combined_audit.json、runtime_audit_development.json及collection.json。方法结论只在experiment_log中给出。
