# mainExp_Task3_3D_5.1：anchored FMT 的新空间确认

## 结论

相对同宽、同结构、同训练过程的 train-only Raw-PCA residual，anchored FMT
residual 在 10 个 3D 数据条目上的 dataset-macro F1 从 `.71683` 提高到
`.85274`，绝对增益 `+.13591`；Average Precision 从 `.77720` 提高到
`.92692`，绝对增益 `+.14971`。F1 与 Average Precision 均为 10/10 条目、
7/7 physical family、5/5 paired seed 正增益。

该结果明显高于 `mainExp_Task3_3D_4.1` 的 F1/Average Precision 增益
`+.10000/+.11896`，支持 FMT 提高监督式 IVD 二分类性能。但预注册的
dataset-macro F1 `+.15` 目标仍未达到，差 `.01409`，不得写成达到 15 个
F1 百分点。若按基线比例计算，F1 的相对提升为 `18.96%`；该相对百分比不能
替代预注册的绝对 F1 增益判据。

## 冻结协议

1. whole-field IVD p95 标签保持不变；40/40 个最终切片的独立标签与 source
   cache 中的 `reference` 逐位相同。
2. Stage 1 比较 18 个无参数 FMT feature block；Stage 2 将每个 family 的前三个
   block 与 10 个 residual route、宽度、学习率和融合设置组合。每个 FMT 候选
   都配一个只在训练 primitive 上拟合、与 FMT 等宽的 Raw-PCA residual。
3. Stage 1 使用 paired seeds 40--41；Stage 2 使用 40--42。所有旧时间 ordinal
   与 `mainExp_Task3_3D_4.1` 的空间相位都只作 development。
4. Stage 2 selection 写入 SHA-256 后，才生成预注册空间相位
   `[-.37,.29,-.11]` 的新 primitive population。最终包含 10 个数据条目、
   每条目 4 个切片、共 155,157 个有效 primitive。
5. 最终使用 paired seeds 40--44。confirmation 不选择 feature、网络、epoch、
   threshold、residual alpha 或标签。

`aivd{k}w{L}` 是 anchored vorticity-deviation Fourier block：它只从 7-line
pathline primitive 的局部 flow-map differential 估计 velocity gradient 和
vorticity-deviation sequence，再保留首值、早期统计量和前 `k` 个离散 Fourier
频率；它不读取 IVD volume、p95 threshold 或二值标签。

## 冻结的 family 配方

| Physical family | FMT feature | Residual 设置 |
|---|---|---|
| Channel observer | `aivd1w3` | dual input, auxiliary 64, validation minimum-gain selection |
| Half-cylinder Re160/Re640/Re6400 | `aivd1w3` | dual input, auxiliary 64, validation minimum-gain selection |
| Tangaroa | `aivd1w3` | dual input, auxiliary 64, validation minimum-gain selection |
| Delta wing | `fmt_all+aivd2w8` | FMT-only residual, auxiliary 64, fixed alpha 1 |
| F-22 | `aivd2w8` | geometry+FMT residual, auxiliary 96, validation minimum-gain selection |
| Boeing 747 | `fmt_all+kin2` | FMT-only residual, auxiliary 64, fixed alpha 1 |
| Smoke buoyancy | `aivd1w3` | dual input, auxiliary 64, validation minimum-gain selection |

Stage 2 development 的 dataset-macro F1/Average Precision 增益为
`+.16025/+.17163`。该值只用于冻结配方，不是论文最终指标。

## 最终结果

表中 `±` 是五个 paired training seeds 的 population standard deviation；增益
按同 seed 的 `FMT − Raw-PCA` 后再统计。

| Flow | Raw-PCA F1 | FMT F1 | F1增益 | Raw-PCA AP | FMT AP | AP增益 |
|---|---:|---:|---:|---:|---:|---:|
| Boeing 747 | .7649±.0147 | .8349±.0063 | **+.0701±.0170** | .8392±.0130 | .9197±.0026 | **+.0804±.0140** |
| Channel observer | .3308±.0243 | .8067±.0057 | **+.4759±.0222** | .2778±.0046 | .8733±.0183 | **+.5955±.0146** |
| Half-cylinder Re160 | .7288±.0162 | .8120±.0096 | **+.0832±.0179** | .7916±.0294 | .8659±.0077 | **+.0742±.0236** |
| Delta-wing original LBM | .8017±.0065 | .8773±.0046 | **+.0757±.0096** | .9012±.0062 | .9601±.0012 | **+.0588±.0054** |
| Delta-wing resampled | .8157±.0059 | .9054±.0045 | **+.0897±.0056** | .9108±.0079 | .9710±.0024 | **+.0602±.0075** |
| F-22 | .8116±.0120 | .9156±.0076 | **+.1040±.0151** | .8760±.0075 | .9703±.0058 | **+.0944±.0068** |
| Half-cylinder Re640 | .7176±.0041 | .8929±.0133 | **+.1752±.0143** | .7833±.0051 | .9635±.0082 | **+.1801±.0095** |
| Half-cylinder Re6400 | .6256±.0021 | .7836±.0037 | **+.1580±.0056** | .6465±.0063 | .8701±.0038 | **+.2237±.0089** |
| Smoke buoyancy | .8084±.0054 | .8610±.0071 | **+.0526±.0056** | .8970±.0116 | .9524±.0062 | **+.0554±.0060** |
| Tangaroa | .7631±.0072 | .8379±.0056 | **+.0748±.0055** | .8486±.0053 | .9230±.0088 | **+.0744±.0051** |
| **Dataset macro** | **.7168** | **.8527** | **+.1359** | **.7772** | **.9269** | **+.1497** |
| **Family-macro gain** | — | — | **+.1427** | — | — | **+.1598** |

五个 seed 的 dataset-macro F1 增益为 `.13941/.13287/.13327/.14061/.13338`；
Average Precision 增益为 `.15782/.14528/.14369/.15186/.14991`。因此正方向
不依赖某一个训练 seed。

## 审计与复现

- Stage 1 jobs：`50931950[0-179%24]`；selector：`50931968`。
- Stage 2 jobs：`50931970[0-299%24]`；selector：`50931980`。
- 最终标签：`50939374[0-1%2]`；最终评估：`50939375[0-9%10]`；汇总：
  `50939376`。最终使用 1×GTX 1080 Ti、1×P100 与 8×V100；10/10 children
  exit 0，所有 stderr 为空。
- Stage 1 selection SHA-256：
  `0fca4fde801c898827517ad072df2918704eb61e2f56c8bc65555715259f7b76`。
- Stage 2 selection SHA-256：
  `8341272e5984008cb0d39059f6fb84dbeea4251989b0a37040e931481968a2ab`。
- Frozen manifest SHA-256：
  `5e4d30c81f07e78dcfad240ac321f30bfe364be8e2589df57bc0a3cd7aa68d71`。
- `summary.json` SHA-256：
  `7468a38b850df1f278c48f60678d226acbce4ec9139be2927098e6f7f1611912`。
- `per_run.csv` SHA-256：
  `864bf2683c3b671e6483debb395e308545fd65346e97649a9cc8a22834f21340`。
- 实现 commits：`81c7f3a`、`15837c6`、`a32ad47`。完整失败重试与设备记录见
  `docs/ibex_run_registry.md`。

5.1 confirmation 已经打开。任何下一轮方法或超参数选择都必须把它明确降为
development，并在另一组预注册、未生成的新空间 primitive population 上确认。
