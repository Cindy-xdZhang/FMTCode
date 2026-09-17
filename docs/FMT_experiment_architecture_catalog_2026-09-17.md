# FMT实验配置完整索引：2026-09-17

本表按配置路径逐项登记，配置内容完整保存在CSV的configuration列；同一路径全部历史版本见configuration_inventory.csv。本表的结构栏指向已核查的共享实现，**不把配置文件存在等同于已执行，也不把一个多分支实验混称为同一种网络**。具体几何公式、分支差异和运行证据见主审计报告。

本地不同配置路径：573。按已核查的调用族登记；库模板不等于正式实验，历史版本见原始CSV。

| 配置 | 实验名 | 对应实现 | 中心策略 | 卷积情况 | 来源 |
|---|---|---|---|---|---|
| config/Ablation_FMTv8_NormFrequency_3.1.json | Ablation_FMTv8_NormFrequency_3.1 | V8/p00–p49/n0–n3跨任务搜索 | Task1/3/5固定slot0；Task4-c全部有效线轮换 | Task1聚类无；历史Task3/5完整预测含三层Conv1d；Task4-c FMT无 | working_tree |
| config/Ablation_FMTv8_Search_2.1.json | Ablation_FMTv8_Search_2.1 | V8/p00–p49/n0–n3跨任务搜索 | Task1/3/5固定slot0；Task4-c全部有效线轮换 | Task1聚类无；历史Task3/5完整预测含三层Conv1d；Task4-c FMT无 | working_tree |
| config/Ablation_FMTv8_Search_2.2.json | Ablation_FMTv8_Search_2.2 | V8/p00–p49/n0–n3跨任务搜索 | Task1/3/5固定slot0；Task4-c全部有效线轮换 | Task1聚类无；历史Task3/5完整预测含三层Conv1d；Task4-c FMT无 | working_tree |
| config/Ablation_Task123_FMTComponents_1.1.yaml | Ablation_Task123_FMTComponents_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Ablation_Task123_FMTComponents_1.2.yaml | Ablation_Task123_FMTComponents_1.2 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Ablation_Task23IVDPercentile_1.1.yaml | Ablation_Task23IVDPercentile_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Ablation_Task23IVDPercentile_1.2.yaml | Ablation_Task23IVDPercentile_1.2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Ablation_Task4C_BiLSTMCapacity_1.1.json | Ablation_Task4C_BiLSTMCapacity_1.1 | Task4-c独立非FMT几何基线 | 无FMT轮换；Point-NN/PointNet++有多个点邻域，BiLSTM逐线后池化 | 本仓库这些网络使用Linear/LSTM，无ConvNd；不据PointNet名称猜实现 | working_tree |
| config/Ablation_Task4C_BottomDensity_1.1.json | Ablation_Task4C_BottomDensity_1.1 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_BottomDensity_1.2.json | Ablation_Task4C_BottomDensity_1.2 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_BottomDensity_1.3.json | Ablation_Task4C_BottomDensity_1.3 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_CenterDiversity_4.9.json | Ablation_Task4C_CenterDiversity_4.9 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_ConvCapacity_4.18.json | Ablation_Task4C_ConvCapacity_4.18 | Task4-c独立Conv3D对照 | 无FMT特征 | 有Conv3d，独立对照不是FMT分支 | working_tree |
| config/Ablation_Task4C_ConvResolution_4.22.json | Ablation_Task4C_ConvResolution_4.22 | Task4-c独立Conv3D对照 | 无FMT特征 | 有Conv3d，独立对照不是FMT分支 | working_tree |
| config/Ablation_Task4C_FMTv5Blocks_4.19.json | Ablation_Task4C_FMTv5Blocks_4.19 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_FMTv6v7_4.20.json | Ablation_Task4C_FMTv6v7_4.20 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_FPSAugmentSearch_1.1.json | Ablation_Task4C_FPSAugmentSearch_1.1 | Task4-c c000–c263/c156 | 全部有效线轮换；每中心FPS选6邻居 | FMT无ConvNd；fps_graph含图消息传递，单独标注 | working_tree |
| config/Ablation_Task4C_FPSAugmentSearch_1.2.json | Ablation_Task4C_FPSAugmentSearch_1.2 | Task4-c c000–c263/c156 | 全部有效线轮换；每中心FPS选6邻居 | FMT无ConvNd；fps_graph含图消息传递，单独标注 | working_tree |
| config/Ablation_Task4C_GeneralizedCrossEntropy_4.11.json | Ablation_Task4C_GeneralizedCrossEntropy_4.11 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_GeometricBaselines_1.1.json | Ablation_Task4C_GeometricBaselines_1.1 | Task4-c独立非FMT几何基线 | 无FMT轮换；Point-NN/PointNet++有多个点邻域，BiLSTM逐线后池化 | 本仓库这些网络使用Linear/LSTM，无ConvNd；不据PointNet名称猜实现 | working_tree |
| config/Ablation_Task4C_NeighborSelection_1.1.json | Ablation_Task4C_NeighborSelection_1.1 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_PointNetPlusPlus_1.1.json | Ablation_Task4C_PointNetPlusPlus_1.1 | Task4-c独立非FMT几何基线 | 无FMT轮换；Point-NN/PointNet++有多个点邻域，BiLSTM逐线后池化 | 本仓库这些网络使用Linear/LSTM，无ConvNd；不据PointNet名称猜实现 | working_tree |
| config/Ablation_Task4C_PointNet_1.1.json | Ablation_Task4C_PointNet_1.1 | Task4-c独立非FMT几何基线 | 无FMT轮换；Point-NN/PointNet++有多个点邻域，BiLSTM逐线后池化 | 本仓库这些网络使用Linear/LSTM，无ConvNd；不据PointNet名称猜实现 | working_tree |
| config/Ablation_Task4C_RepresentationResolution_4.7.json | Ablation_Task4C_RepresentationResolution_4.7 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_Sharpness_4.8.json | Ablation_Task4C_Sharpness_4.8 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Ablation_Task4C_SupervisedContrastive_4.10.json | Ablation_Task4C_SupervisedContrastive_4.10 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Calibrate_Task3APSelected_1.1.yaml | Calibrate_Task3APSelected_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Calibrate_Task3ConstrainedAP_1.1.yaml | Calibrate_Task3ConstrainedAP_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Calibrate_Task3ConstrainedAP_1.2.yaml | Calibrate_Task3ConstrainedAP_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Confirm_HighReFMTVAE_1.1.yaml | Verify_HighReFMTVAE_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Confirm_HighReTask2_1.1.yaml | Verify_HighReTask2_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Confirm_HighReTask2_1.2.yaml | Verify_HighReTask2_1.2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Confirm_HighReTask2_1.3.yaml | Verify_HighReTask2_1.3 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Confirm_HighReTask2_1.4.yaml | Verify_HighReTask2_1.4 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Confirm_HighReTask2_1.5.yaml | Verify_HighReTask2_1.5 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Confirm_HighReVAE_Controlled_1.1.yaml | Verify_HighReVAE_Controlled_1.1_heldout | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Confirm_Task3UniversalityBaselines_1.1.yaml | Confirm_Task3UniversalityBaselines_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 20fd7945 |
| config/Confirm_Task3UniversalityKinematicLabels_2.1.yaml | Confirm_Task3UniversalityKinematicLabels_2.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Confirm_Task3UniversalityKinematic_2.1.yaml | Confirm_Task3UniversalityKinematic_2.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 20fd7945 |
| config/Confirm_Task3UniversalityResidual_1.1.yaml | Confirm_Task3UniversalityResidual_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 20fd7945 |
| config/Confirm_Task3_CombinedOptimization_12.1.yaml | Confirm_Task3_CombinedOptimization_12.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 163e7fa2 |
| config/Confirm_Task3_CombinedOptimization_12.2.yaml | Confirm_Task3_CombinedOptimization_12.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 50a70674 |
| config/Confirm_Task3_F22AnchoredFeatures_1.2_cache.yaml | Confirm_Task3_F22AnchoredFeatures_1.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Confirm_Task3_F22AnchoredFeatures_1.2_evaluate.yaml | Confirm_Task3_F22AnchoredFeatures_1.2_evaluate | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Confirm_Task3_F22AnchoredFeatures_1.2_labels.yaml | Confirm_Task3_F22AnchoredFeatures_1.2_labels | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | git_history 7568ac63 |
| config/Confirm_Task3_F22Hyperparams_1.1_cache.yaml | Confirm_Task3_F22Hyperparams_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | git_history 68e847cc |
| config/Confirm_Task3_F22Hyperparams_1.1_evaluate.yaml | Confirm_Task3_F22Hyperparams_1.1_evaluate | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history e376ec16 |
| config/Confirm_Task3_F22Hyperparams_1.1_labels.yaml | Confirm_Task3_F22Hyperparams_1.1_labels | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | git_history 68e847cc |
| config/FittingVatistas.yaml |  | 解析流场拟合/生成或GT体素化，无FMT分类器 | 不适用；观察者变换/网格采样不是primitive内部重选中心 | 不定义预测CNN；解析模型参数优化不是卷积网络 | working_tree |
| config/Other_FMTObjectivityTranslation_1.1.json | Other_FMTObjectivityTranslation_1.1 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Other_FMTObjectivityTranslation_1.2.json | Other_FMTObjectivityTranslation_1.2 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Other_FMTObjectivityTranslation_1.3.json | Other_FMTObjectivityTranslation_1.3 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Other_FMTObjectivityTranslation_1.4.json | Other_FMTObjectivityTranslation_1.4 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Other_FMTObjectivityTranslation_1.5.json | Other_FMTObjectivityTranslation_1.5 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Other_FMT_FeatureContrast_1.1.json | Other_FMT_FeatureContrast_1.1 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Other_FTLEP35Fusion_2.1.json | Other_FTLEP35Fusion_2.1 | 二维FTLE方法/对照 | DFT五线固定slot0；老Point-NN为多个点局部中心；多时间窗不换中心 | 历史FMT+U-Net/P35融合有Conv2d/转置卷积；插值对照无 | working_tree |
| config/Other_FTLEP35Fusion_2.1_final.json | Other_FTLEP35Fusion_2.1 | 二维FTLE方法/对照 | DFT五线固定slot0；老Point-NN为多个点局部中心；多时间窗不换中心 | 历史FMT+U-Net/P35融合有Conv2d/转置卷积；插值对照无 | working_tree |
| config/Other_FTLEP35Fusion_2.2.json | Other_FTLEP35Fusion_2.2 | 二维FTLE方法/对照 | DFT五线固定slot0；老Point-NN为多个点局部中心；多时间窗不换中心 | 历史FMT+U-Net/P35融合有Conv2d/转置卷积；插值对照无 | working_tree |
| config/Other_FTLEUpsampling2D_1.1.json | Other_FTLEUpsampling2D_1.1 | 二维FTLE方法/对照 | DFT五线固定slot0；老Point-NN为多个点局部中心；多时间窗不换中心 | 历史FMT+U-Net/P35融合有Conv2d/转置卷积；插值对照无 | working_tree |
| config/Other_FTLEUpsampling2D_1.2.json | Other_FTLEUpsampling2D_1.2 | 二维FTLE方法/对照 | DFT五线固定slot0；老Point-NN为多个点局部中心；多时间窗不换中心 | 历史FMT+U-Net/P35融合有Conv2d/转置卷积；插值对照无 | working_tree |
| config/Other_Task123_PaperTriptychs_1.1.json | Other_Task123_PaperTriptychs_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task123_PaperTriptychs_1.2.json | Other_Task123_PaperTriptychs_1.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task23DenseCloseup_1.1.yaml | Other_Task23DenseCloseup_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_1.2.yaml | Other_Task23DenseCloseup_1.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_1.3.yaml | Other_Task23DenseCloseup_1.3 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_1.4.yaml | Other_Task23DenseCloseup_1.4 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_1.5.yaml | Other_Task23DenseCloseup_1.5 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_1.6.yaml | Other_Task23DenseCloseup_1.6 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_1.7.yaml | Other_Task23DenseCloseup_1.7 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_1.8.yaml | Other_Task23DenseCloseup_1.8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_1.9.yaml | Other_Task23DenseCloseup_1.9 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.0.yaml | Other_Task23DenseCloseup_2.0 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.1.yaml | Other_Task23DenseCloseup_2.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.2.yaml | Other_Task23DenseCloseup_2.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.3.yaml | Other_Task23DenseCloseup_2.3 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.4.yaml | Other_Task23DenseCloseup_2.4 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.5.yaml | Other_Task23DenseCloseup_2.5 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.6.yaml | Other_Task23DenseCloseup_2.6 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.7.yaml | Other_Task23DenseCloseup_2.7 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task23DenseCloseup_2.8.yaml | Other_Task23DenseCloseup_2.8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task2_VisualAnalysis_1.1.json | Other_Task2_VisualAnalysis_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task2_VisualAnalysis_1.2.json | Other_Task2_VisualAnalysis_1.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task2_VisualAnalysis_1.3.json | Other_Task2_VisualAnalysis_1.3 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task3DenseVisualization_1.1.yaml | Other_Task3DenseVisualization_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.10.yaml | Other_Task3DenseVisualization_1.10 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.11.yaml | Other_Task3DenseVisualization_1.11 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.12.yaml | Other_Task3DenseVisualization_1.12 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.13.yaml | Other_Task3DenseVisualization_1.13 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.14.yaml | Other_Task3DenseVisualization_1.14 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.15.yaml | Other_Task3DenseVisualization_1.15 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.17.yaml | Other_Task3DenseVisualization_1.17 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.18.yaml | Other_Task3DenseVisualization_1.18 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.19.yaml | Other_Task3DenseVisualization_1.19 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.2.yaml | Other_Task3DenseVisualization_1.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.20.yaml | Other_Task3DenseVisualization_1.20 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.21.yaml | Other_Task3DenseVisualization_1.21 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.22.yaml | Other_Task3DenseVisualization_1.22 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.23.yaml | Other_Task3DenseVisualization_1.23 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.24.yaml | Other_Task3DenseVisualization_1.24 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.25.yaml | Other_Task3DenseVisualization_1.25 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.26.yaml | Other_Task3DenseVisualization_1.26 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.27.yaml | Other_Task3DenseVisualization_1.27 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.28.yaml | Other_Task3DenseVisualization_1.28 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.29.yaml | Other_Task3DenseVisualization_1.29 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.3.yaml | Other_Task3DenseVisualization_1.3 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.30.yaml | Other_Task3DenseVisualization_1.30 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.31.yaml | Other_Task3DenseVisualization_1.31 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.32.yaml | Other_Task3DenseVisualization_1.32 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.33.yaml | Other_Task3DenseVisualization_1.33 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.35.yaml | Other_Task3DenseVisualization_1.35 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.36.yaml | Other_Task3DenseVisualization_1.36 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.37.yaml | Other_Task3DenseVisualization_1.37 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.38.yaml | Other_Task3DenseVisualization_1.38 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.39.yaml | Other_Task3DenseVisualization_1.39 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.4.yaml | Other_Task3DenseVisualization_1.4 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.40.yaml | Other_Task3DenseVisualization_1.40 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.41.yaml | Other_Task3DenseVisualization_1.41 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.42.yaml | Other_Task3DenseVisualization_1.42 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.44.yaml | Other_Task3DenseVisualization_1.44 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.45.yaml | Other_Task3DenseVisualization_1.45 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.46.yaml | Other_Task3DenseVisualization_1.46 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.48.yaml | Other_Task3DenseVisualization_1.48 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.49.yaml | Other_Task3DenseVisualization_1.49 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.5.yaml | Other_Task3DenseVisualization_1.5 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.50.yaml | Other_Task3DenseVisualization_1.50 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.51.yaml | Other_Task3DenseVisualization_1.51 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.55.yaml | Other_Task3DenseVisualization_1.55 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.56.yaml | Other_Task3DenseVisualization_1.56 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.57.yaml | Other_Task3DenseVisualization_1.57 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.58.yaml | Other_Task3DenseVisualization_1.58 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.59.yaml | Other_Task3DenseVisualization_1.59 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.6.yaml | Other_Task3DenseVisualization_1.6 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.60.yaml | Other_Task3DenseVisualization_1.60 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.61.yaml | Other_Task3DenseVisualization_1.61 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.62.yaml | Other_Task3DenseVisualization_1.62 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.63.yaml | Other_Task3DenseVisualization_1.63 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.64.yaml | Other_Task3DenseVisualization_1.64 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.65.yaml | Other_Task3DenseVisualization_1.65 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.66.yaml | Other_Task3DenseVisualization_1.66 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.67.yaml | Other_Task3DenseVisualization_1.67 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.68.yaml | Other_Task3DenseVisualization_1.68 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.69.yaml | Other_Task3DenseVisualization_1.69 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.7.yaml | Other_Task3DenseVisualization_1.7 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.70.yaml | Other_Task3DenseVisualization_1.70 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.71.yaml | Other_Task3DenseVisualization_1.71 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.72.yaml | Other_Task3DenseVisualization_1.72 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.73.yaml | Other_Task3DenseVisualization_1.73 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.74.yaml | Other_Task3DenseVisualization_1.74 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.75.yaml | Other_Task3DenseVisualization_1.75 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.76.yaml | Other_Task3DenseVisualization_1.76 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.77.yaml | Other_Task3DenseVisualization_1.77 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.78.yaml | Other_Task3DenseVisualization_1.78 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.79.yaml | Other_Task3DenseVisualization_1.79 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.8.yaml | Other_Task3DenseVisualization_1.8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.80.yaml | Other_Task3DenseVisualization_1.80 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.81.yaml | Other_Task3DenseVisualization_1.81 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.82.yaml | Other_Task3DenseVisualization_1.82 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.83.yaml | Other_Task3DenseVisualization_1.83 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.84.yaml | Other_Task3DenseVisualization_1.84 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.85.yaml | Other_Task3DenseVisualization_1.85 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.86.yaml | Other_Task3DenseVisualization_1.86 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.87.yaml | Other_Task3DenseVisualization_1.87 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.88.yaml | Other_Task3DenseVisualization_1.88 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.89.yaml | Other_Task3DenseVisualization_1.89 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.9.yaml | Other_Task3DenseVisualization_1.9 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.90.yaml | Other_Task3DenseVisualization_1.90 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.91.yaml | Other_Task3DenseVisualization_1.91 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.92.yaml | Other_Task3DenseVisualization_1.92 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.93.yaml | Other_Task3DenseVisualization_1.93 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.94.yaml | Other_Task3DenseVisualization_1.94 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.95.yaml | Other_Task3DenseVisualization_1.95 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.96.yaml | Other_Task3DenseVisualization_1.96 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.97.yaml | Other_Task3DenseVisualization_1.97 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.98.yaml | Other_Task3DenseVisualization_1.98 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_1.99.yaml | Other_Task3DenseVisualization_1.99 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_2.0.yaml | Other_Task3DenseVisualization_2.0 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_2.1.yaml | Other_Task3DenseVisualization_2.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_2.2.yaml | Other_Task3DenseVisualization_2.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_2.3.yaml | Other_Task3DenseVisualization_2.3 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_2.4.yaml | Other_Task3DenseVisualization_2.4 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_2.7.yaml | Other_Task3DenseVisualization_2.7 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_2.8.yaml | Other_Task3DenseVisualization_2.8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_2.9.yaml | Other_Task3DenseVisualization_2.9 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.0.yaml | Other_Task3DenseVisualization_3.0 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.1.yaml | Other_Task3DenseVisualization_3.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.2.yaml | Other_Task3DenseVisualization_3.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.3.yaml | Other_Task3DenseVisualization_3.3 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.4.yaml | Other_Task3DenseVisualization_3.4 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.5.yaml | Other_Task3DenseVisualization_3.5 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.6.yaml | Other_Task3DenseVisualization_3.6 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.7.yaml | Other_Task3DenseVisualization_3.7 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_3.8.yaml | Other_Task3DenseVisualization_3.8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_4.0.yaml | Other_Task3DenseVisualization_4.0 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_4.1.yaml | Other_Task3DenseVisualization_4.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_4.2.yaml | Other_Task3DenseVisualization_4.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_4.3.yaml | Other_Task3DenseVisualization_4.3 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_4.4.yaml | Other_Task3DenseVisualization_4.4 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Other_Task3DenseVisualization_4.5.yaml | Other_Task3DenseVisualization_4.5 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task4A_FMTStreamlineClustering_1.1.yaml | Other_Task4A_FMTStreamlineClustering_1.1 | Task4-a逐primitive/逐实例聚类 | 固定七线中心；跨primitive池化不是同primitive重选中心 | 无；固定描述/KMeans | working_tree |
| config/Other_Task4B_FMTFourClassClustering_1.1.yaml | Other_Task4B_FMTFourClassClustering_1.1 | Task4-b固定特征聚类 | 固定七线中心；不轮换 | 无；KMeans/固定特征 | working_tree |
| config/Other_Task4B_FMTFourClassClustering_1.2.yaml | Other_Task4B_FMTFourClassClustering_1.2 | Task4-b固定特征聚类 | 固定七线中心；不轮换 | 无；KMeans/固定特征 | working_tree |
| config/Other_Task4B_GeometrySearch_5.2.json | Other_Task4B_GeometrySearch_5.2 | Task4-b纯特征MLP | 固定七线中心；不轮换 | 无；普通/残差MLP | working_tree |
| config/Other_Task4B_ProxyGTThreshold_4.2.json | Other_Task4B_ProxyGTThreshold_4.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task4B_ProxyGTThreshold_4.3.json | Other_Task4B_ProxyGTThreshold_4.3 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task4B_ProxyGTThreshold_4.4.json | Other_Task4B_ProxyGTThreshold_4.4 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task4C_BundleVisualization_1.1.json | Other_Task4C_BundleVisualization_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task4C_BundleVisualization_1.2.json | Other_Task4C_BundleVisualization_1.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Other_Task4C_Saliency_1.1.json | Other_Task4C_Saliency_1.1 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Other_Task4VoxelGT_1.1.yaml | Other_Task4VoxelGT_1.1 | 解析流场拟合/生成或GT体素化，无FMT分类器 | 不适用；观察者变换/网格采样不是primitive内部重选中心 | 不定义预测CNN；解析模型参数优化不是卷积网络 | working_tree |
| config/PathlineFMTclustering.yaml |  | 早期二维Point-NN式FMT聚类 | 每个同时间点都作局部中心；不是后来固定slot0的DFT | 聚类入口明确temporal_head=None，无卷积；可选dft头另列为频域卷积 | working_tree |
| config/PathlineFMTclustering3D.yaml |  | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/VatistasDataset.yaml |  | 解析流场拟合/生成或GT体素化，无FMT分类器 | 不适用；观察者变换/网格采样不是primitive内部重选中心 | 不定义预测CNN；解析模型参数优化不是卷积网络 | working_tree |
| config/Verify_3DFMTHyperparam_1.1.yaml | Verify_3DFMTHyperparam_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_AIVDLongtime_1.1.json | Verify_AIVDLongtime_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_AIVDLongtime_1.1_task5_cache.yaml | Verify_AIVDLongtime_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_AIVDTransfer_1.1.json | Verify_AIVDTransfer_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_AIVDTranslationObservers_1.1.json | Verify_AIVDTranslationObservers_1.1 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Verify_AIVDTranslationObservers_1.2.json | Verify_AIVDTranslationObservers_1.2 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Verify_AIVDTranslationObservers_1.3.json | Verify_AIVDTranslationObservers_1.3 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Verify_FMTAllV2_1.1.json | Verify_FMTAllV2_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_FMTAllV2_1.2.json | Verify_FMTAllV2_1.2 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_FMTObjectivityMechanism_1.1.json | Verify_FMTObjectivityMechanism_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_FMTObjectivityMechanism_1.2.json | Verify_FMTObjectivityMechanism_1.2 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_FMTObjectivityMechanism_1.3.json | Verify_FMTObjectivityMechanism_1.3 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_FMTObservedPathline_2.1.json | Verify_FMTObservedPathline_2.1 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Verify_FMT_SingleCenter_Task354C_1.1.json | Verify_FMT_SingleCenter_Task354C_1.1 | 新单中心候选，正式训练未开始 | 固定一个；Task4-c一次选有效线 | 无；141→MLP | working_tree |
| config/Verify_HighReRawNormalization_1.1.yaml | Verify_HighReRawNormalization_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReRawRepresentation_1.1.yaml | Verify_HighReRawRepresentation_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReRawWeighting_1.1.yaml | Verify_HighReRawWeighting_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReSampling3D_1.1.yaml | Verify_HighReSampling3D_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_HighReSampling3D_1.2.yaml | Verify_HighReSampling3D_1.2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_HighReSampling3D_1.3.yaml | Verify_HighReSampling3D_1.3 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReSampling3D_1.4.yaml | Verify_HighReSampling3D_1.4 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReSampling3D_1.5.yaml | Verify_HighReSampling3D_1.5 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_HighReVAE_1.1.yaml | Verify_HighReVAE_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_HighReVAE_1.2.yaml | Verify_HighReVAE_1.2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_1.3.yaml | Verify_HighReVAE_1.3 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_1.4.yaml | Verify_HighReVAE_1.4 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_1.5.yaml | Verify_HighReVAE_1.5 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_1.6.yaml | Verify_HighReVAE_1.6 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_1.7.yaml | Verify_HighReVAE_1.7 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_1.8.yaml | Verify_HighReVAE_1.8 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_1.9.yaml | Verify_HighReVAE_1.9 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_Controlled_1.1.yaml | Verify_HighReVAE_Controlled_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_HighReVAE_RawRe640_1.1.yaml | Verify_HighReVAE_RawRe640_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_JHTDB_ChannelDownload_1.1.json | Verify_JHTDB_ChannelDownload_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_JHTDB_DualFormatDownload_1.2.json | Verify_JHTDB_DualFormatDownload_1.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_JHTDB_VTKDownload_1.1.json | Verify_JHTDB_VTKDownload_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_LargeNeighbor_1.1.json | Verify_LargeNeighbor_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_ObjectiveFMTnTDO_1.1.json | Verify_ObjectiveFMTnTDO_1.1 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Verify_PathlineHyperparams3D_1.1.yaml | Verify_PathlineHyperparams3D_1.1 | 几何/观察者/客观性诊断 | 固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略 | 诊断公式无预测CNN；若复用分类模型需按该模型计 | working_tree |
| config/Verify_Task1235_AngleFeatures_1.1.json | Verify_Task1235_AngleFeatures_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task1235_ObjectiveFMTnTDO_1.1.json | Verify_Task1235_ObjectiveFMTnTDO_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task1235_ObjectiveFMTnTDO_1.1_task5_cache.yaml | Verify_Task1235_ObjectiveFMTnTDO_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_Task1235_ObjectiveFMTnTDO_2.1.json | Verify_Task1235_ObjectiveFMTnTDO_2.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task123_FixedPathline_1.1.yaml | Verify_Task123_FixedPathline_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task123_GeometryParameterStress_1.1.yaml | Verify_Task123_GeometryParameterStress_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task123_NoiseRobustness_1.1.yaml | Verify_Task123_NoiseRobustness_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task123_NoiseRobustness_1.2.yaml | Verify_Task123_NoiseRobustness_1.2 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task123_NoiseTypesStrong_1.1.yaml | Verify_Task123_NoiseTypesStrong_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task123_NoiseTypes_1.1.yaml | Verify_Task123_NoiseTypes_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task123_StrongBaselines_1.1.yaml | Verify_Task123_StrongBaselines_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task123_StrongBaselines_1.2.yaml | Verify_Task123_StrongBaselines_1.2 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task135_GeometricControls_1.1.yaml | Verify_Task135_GeometricControls_1.1 | 多个任务共享固定特征比较 | FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告 | Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d | working_tree |
| config/Verify_Task2F22FMTBlock_1.1.yaml | Verify_Task2F22FMTBlock_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/Verify_Task2Universality_1.1.yaml | Verify_Task2Universality_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_Task2_FMTVAEFamilySearch_4.1.yaml | Verify_Task2_FMTVAEFamilySearch_4.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_Task2_LatentBottleneck_5.1.yaml | Verify_Task2_LatentBottleneck_5.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_Task2_UniformFMT_6.1.yaml | Verify_Task2_UniformFMT_6.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_Task2_VAEGrid_3D_3.1.yaml | Verify_Task2_VAEGrid_3D_3.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/Verify_Task2_VAEGrid_3D_3.2.yaml | Verify_Task2_VAEGrid_3D_3.2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | git_history 11781673 |
| config/Verify_Task36_BalancedScale_1.2.json | Verify_Task36_BalancedScale_1.2 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task36_MultiGate_1.1.json | Verify_Task36_MultiGate_1.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task3APSelectedResidual_1.1.yaml | Verify_Task3APSelectedResidual_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3APSelectedResidual_1.2.yaml | Verify_Task3APSelectedResidual_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3BalancedSelection_1.1.yaml | Verify_Task3BalancedSelection_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3ConstrainedAPSelection_1.1.yaml | Verify_Task3ConstrainedAPSelection_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3ConstrainedAPSelection_1.2.yaml | Verify_Task3ConstrainedAPSelection_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTClassifierLiteral_1.1.yaml | Verify_Task3FMTClassifierLiteral_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTClassifierP90_1.1.yaml | Verify_Task3FMTClassifierP90_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTConvergenceLiteral_1.2.yaml | Verify_Task3FMTConvergenceLiteral_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTConvergenceP90_1.2.yaml | Verify_Task3FMTConvergenceP90_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTDualResidual_1.1.yaml | Verify_Task3FMTDualResidual_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTDualResidual_1.2.yaml | Verify_Task3FMTDualResidual_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTGramResidual_1.1.yaml | Verify_Task3FMTGramResidual_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTGramResidual_1.2.yaml | Verify_Task3FMTGramResidual_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTGramResidual_1.3.yaml | Verify_Task3FMTGramResidual_1.3 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTKinematicResidual_1.1.yaml | Verify_Task3FMTKinematicResidual_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTKinematicResidual_1.2.yaml | Verify_Task3FMTKinematicResidual_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTKinematicResidual_1.4.yaml | Verify_Task3FMTKinematicResidual_1.4 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTKinematicResidual_1.5.yaml | Verify_Task3FMTKinematicResidual_1.5 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTKinematicResidual_1.6.yaml | Verify_Task3FMTKinematicResidual_1.6 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTOnlyResidual_1.1.yaml | Verify_Task3FMTOnlyResidual_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FMTOnlyResidual_1.2.yaml | Verify_Task3FMTOnlyResidual_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FrozenNewTimes_1.1.yaml | Verify_Task3FrozenNewTimes_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3FrozenNewTimes_1.2.yaml | Verify_Task3FrozenNewTimes_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3Label_1.1.yaml | Verify_Task3Label_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3LabelsLiteral_1.1.yaml | Verify_Task3LabelsLiteral_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_Task3LabelsP90_1.1.yaml | Verify_Task3LabelsP90_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | archive:retired-validation-20260905.zip |
| config/Verify_Task3UniversalityClassifier_1.1.yaml | Verify_Task3UniversalityClassifier_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 20fd7945 |
| config/Verify_Task3UniversalityResidual_1.2.yaml | Verify_Task3UniversalityResidual_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3UniversalityResidual_1.3.yaml | Verify_Task3UniversalityResidual_1.3 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | archive:retired-validation-20260905.zip |
| config/Verify_Task3Universality_1.1.yaml | Verify_Task3Universality_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_AdamBetas_34.1.yaml | Verify_Task3_AdamBetas_34.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 4dc40ad1 |
| config/Verify_Task3_AdamEpsilon_42.1.yaml | Verify_Task3_AdamEpsilon_42.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 03594f71 |
| config/Verify_Task3_AdaptivePortfolio_52.1.yaml | Verify_Task3_AdaptivePortfolio_52.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_AnchoredFeatureDecomposition_22.1.yaml | Verify_Task3_AnchoredFeatureDecomposition_22.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_AnchoredFeatureSpatialReplay_46.1.yaml | Verify_Task3_AnchoredFeatureSpatialReplay_46.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history b80bf1e3 |
| config/Verify_Task3_AnchoredRobust_5.1.yaml | Verify_Task3_AnchoredRobust_5.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_AuxiliaryActivationInputScalePortfolio_90.1.yaml | Verify_Task3_AuxiliaryActivationInputScalePortfolio_90.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history b4b08a4d |
| config/Verify_Task3_AuxiliaryActivationInputScale_89.1.yaml | Verify_Task3_AuxiliaryActivationInputScale_89.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history b4b08a4d |
| config/Verify_Task3_AuxiliaryActivationInputShiftPortfolio_92.1.yaml | Verify_Task3_AuxiliaryActivationInputShiftPortfolio_92.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 23593a30 |
| config/Verify_Task3_AuxiliaryActivationInputShift_91.1.yaml | Verify_Task3_AuxiliaryActivationInputShift_91.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 23593a30 |
| config/Verify_Task3_AuxiliaryActivationResidualGainPortfolio_96.1.yaml | Verify_Task3_AuxiliaryActivationResidualGainPortfolio_96.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history cf22d968 |
| config/Verify_Task3_AuxiliaryActivationResidualGain_95.1.yaml | Verify_Task3_AuxiliaryActivationResidualGain_95.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history cf22d968 |
| config/Verify_Task3_AuxiliaryActivationResidualMixPortfolio_94.1.yaml | Verify_Task3_AuxiliaryActivationResidualMixPortfolio_94.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 48a493ee |
| config/Verify_Task3_AuxiliaryActivationResidualMix_93.1.yaml | Verify_Task3_AuxiliaryActivationResidualMix_93.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 48a493ee |
| config/Verify_Task3_AuxiliaryBeta1Portfolio_62.1.yaml | Verify_Task3_AuxiliaryBeta1Portfolio_62.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history d5283f5a |
| config/Verify_Task3_AuxiliaryBeta1_61.1.yaml | Verify_Task3_AuxiliaryBeta1_61.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 6b376017 |
| config/Verify_Task3_AuxiliaryBeta2Portfolio_60.1.yaml | Verify_Task3_AuxiliaryBeta2Portfolio_60.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history f12fdceb |
| config/Verify_Task3_AuxiliaryBeta2_59.1.yaml | Verify_Task3_AuxiliaryBeta2_59.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 1162334f |
| config/Verify_Task3_AuxiliaryBottleneck_17.1.yaml | Verify_Task3_AuxiliaryBottleneck_17.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 4795e57e |
| config/Verify_Task3_AuxiliaryBottleneck_17.2.yaml | Verify_Task3_AuxiliaryBottleneck_17.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 4efef705 |
| config/Verify_Task3_AuxiliaryDeepSupervision_26.1.yaml | Verify_Task3_AuxiliaryDeepSupervision_26.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 4f2a37f8 |
| config/Verify_Task3_AuxiliaryDropout_53.1.yaml | Verify_Task3_AuxiliaryDropout_53.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_AuxiliaryEpsilonPortfolio_64.1.yaml | Verify_Task3_AuxiliaryEpsilonPortfolio_64.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 53fb282b |
| config/Verify_Task3_AuxiliaryEpsilon_63.1.yaml | Verify_Task3_AuxiliaryEpsilon_63.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 53fb282b |
| config/Verify_Task3_AuxiliaryFeatureScalePortfolio_74.1.yaml | Verify_Task3_AuxiliaryFeatureScalePortfolio_74.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 813ddb75 |
| config/Verify_Task3_AuxiliaryFeatureScale_73.1.yaml | Verify_Task3_AuxiliaryFeatureScale_73.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 813ddb75 |
| config/Verify_Task3_AuxiliaryGaussianNoisePortfolio_72.1.yaml | Verify_Task3_AuxiliaryGaussianNoisePortfolio_72.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3d4a9aea |
| config/Verify_Task3_AuxiliaryGaussianNoise_71.1.yaml | Verify_Task3_AuxiliaryGaussianNoise_71.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3d4a9aea |
| config/Verify_Task3_AuxiliaryGradientClipPortfolio_78.1.yaml | Verify_Task3_AuxiliaryGradientClipPortfolio_78.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history a920ee0d |
| config/Verify_Task3_AuxiliaryGradientClip_77.1.yaml | Verify_Task3_AuxiliaryGradientClip_77.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history a920ee0d |
| config/Verify_Task3_AuxiliaryLearningRatePortfolio_56.1.yaml | Verify_Task3_AuxiliaryLearningRatePortfolio_56.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 62300dff |
| config/Verify_Task3_AuxiliaryLearningRate_55.1.yaml | Verify_Task3_AuxiliaryLearningRate_55.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 62300dff |
| config/Verify_Task3_AuxiliaryLinearBiasScalePortfolio_86.1.yaml | Verify_Task3_AuxiliaryLinearBiasScalePortfolio_86.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3afab36b |
| config/Verify_Task3_AuxiliaryLinearBiasScale_85.1.yaml | Verify_Task3_AuxiliaryLinearBiasScale_85.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3afab36b |
| config/Verify_Task3_AuxiliaryLinearWeightInitializationPortfolio_84.1.yaml | Verify_Task3_AuxiliaryLinearWeightInitializationPortfolio_84.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history bd27d2e5 |
| config/Verify_Task3_AuxiliaryLinearWeightInitialization_83.1.yaml | Verify_Task3_AuxiliaryLinearWeightInitialization_83.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history bd27d2e5 |
| config/Verify_Task3_AuxiliaryNormBiasPortfolio_68.1.yaml | Verify_Task3_AuxiliaryNormBiasPortfolio_68.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history f9b34230 |
| config/Verify_Task3_AuxiliaryNormBias_67.1.yaml | Verify_Task3_AuxiliaryNormBias_67.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history f9b34230 |
| config/Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1.yaml | Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history e37dde81 |
| config/Verify_Task3_AuxiliaryNormEpsilon_69.1.yaml | Verify_Task3_AuxiliaryNormEpsilon_69.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history e37dde81 |
| config/Verify_Task3_AuxiliaryNormScalePortfolio_66.1.yaml | Verify_Task3_AuxiliaryNormScalePortfolio_66.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history fd8a7308 |
| config/Verify_Task3_AuxiliaryNormScale_65.1.yaml | Verify_Task3_AuxiliaryNormScale_65.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history fd8a7308 |
| config/Verify_Task3_AuxiliaryPostNormPortfolio_76.1.yaml | Verify_Task3_AuxiliaryPostNormPortfolio_76.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 6b71af0c |
| config/Verify_Task3_AuxiliaryPostNorm_75.1.yaml | Verify_Task3_AuxiliaryPostNorm_75.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 6b71af0c |
| config/Verify_Task3_AuxiliaryProjectionActivationPortfolio_88.1.yaml | Verify_Task3_AuxiliaryProjectionActivationPortfolio_88.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 6185fac1 |
| config/Verify_Task3_AuxiliaryProjectionActivation_87.1.yaml | Verify_Task3_AuxiliaryProjectionActivation_87.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 6185fac1 |
| config/Verify_Task3_AuxiliaryProjection_21.1.yaml | Verify_Task3_AuxiliaryProjection_21.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history c3fb5c27 |
| config/Verify_Task3_AuxiliaryRegularizationPortfolio_58.1.yaml | Verify_Task3_AuxiliaryRegularizationPortfolio_58.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history a1e44d66 |
| config/Verify_Task3_AuxiliaryWeightDecay_57.1.yaml | Verify_Task3_AuxiliaryWeightDecay_57.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history f0432255 |
| config/Verify_Task3_BalancedCombination_14.1.yaml | Verify_Task3_BalancedCombination_14.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history e028a594 |
| config/Verify_Task3_BatchSize_36.1.yaml | Verify_Task3_BatchSize_36.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 26f02183 |
| config/Verify_Task3_ClassBalancedBatches_13.1.yaml | Verify_Task3_ClassBalancedBatches_13.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history be2c2fb9 |
| config/Verify_Task3_CombinedOptimization_11.1.yaml | Verify_Task3_CombinedOptimization_11.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history c1103d76 |
| config/Verify_Task3_ConfidenceGatedResidual_23.1.yaml | Verify_Task3_ConfidenceGatedResidual_23.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history f2fb0a5e |
| config/Verify_Task3_CosineMinLR_43.1.yaml | Verify_Task3_CosineMinLR_43.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 66790800 |
| config/Verify_Task3_EarlyStoppingMinDeltaPortfolio_82.1.yaml | Verify_Task3_EarlyStoppingMinDeltaPortfolio_82.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3300affc |
| config/Verify_Task3_EarlyStoppingMinDelta_81.1.yaml | Verify_Task3_EarlyStoppingMinDelta_81.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3300affc |
| config/Verify_Task3_EarlyStoppingPatiencePortfolio_80.1.yaml | Verify_Task3_EarlyStoppingPatiencePortfolio_80.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3b0b7b46 |
| config/Verify_Task3_EarlyStoppingPatience_79.1.yaml | Verify_Task3_EarlyStoppingPatience_79.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3b0b7b46 |
| config/Verify_Task3_ExtendedPortfolio_54.1.yaml | Verify_Task3_ExtendedPortfolio_54.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_F22AnchoredFeatures_1.2_baselines.yaml | Verify_Task3_F22AnchoredFeatures_1.2_baselines_v2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 07ee64b6 |
| config/Verify_Task3_F22AnchoredFeatures_1.2_cache.yaml | Verify_Task3_F22AnchoredFeatures_1.2_development | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | git_history 07ee64b6 |
| config/Verify_Task3_F22AnchoredFeatures_1.2_labels.yaml | Verify_Task3_F22AnchoredFeatures_1.2_labels | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | git_history 07ee64b6 |
| config/Verify_Task3_F22AnchoredFeatures_1.2_search.yaml | Verify_Task3_F22AnchoredFeatures_1.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_FMTResidualFamilySearch_4.1.yaml | Verify_Task3_FMTResidualFamilySearch_4.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_FinalPortfolio_49.1.yaml | Verify_Task3_FinalPortfolio_49.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_FocalGammaLow_50.1.yaml | Verify_Task3_FocalGammaLow_50.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_FocalGamma_39.1.yaml | Verify_Task3_FocalGamma_39.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 9e824d38 |
| config/Verify_Task3_GradientClipping_33.1.yaml | Verify_Task3_GradientClipping_33.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history ba1816f2 |
| config/Verify_Task3_HeadAlphaClipCombination_45.1.yaml | Verify_Task3_HeadAlphaClipCombination_45.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_HeadFullStackCombination_48.1.yaml | Verify_Task3_HeadFullStackCombination_48.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_LabelSmoothing_41.1.yaml | Verify_Task3_LabelSmoothing_41.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | git_history 352497c4 |
| config/Verify_Task3_LearningRateWeightDecay_27.1.yaml | Verify_Task3_LearningRateWeightDecay_27.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history c1cbdff8 |
| config/Verify_Task3_LinearWarmup_35.1.yaml | Verify_Task3_LinearWarmup_35.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 10436a4b |
| config/Verify_Task3_LossOptimization_7.1.yaml | Verify_Task3_LossOptimization_7.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history c4f5beb6 |
| config/Verify_Task3_NetworkArchitecture_6.1.yaml | Verify_Task3_NetworkArchitecture_6.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 57d6d6cc |
| config/Verify_Task3_OptimizerFamily_28.1.yaml | Verify_Task3_OptimizerFamily_28.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history db757462 |
| config/Verify_Task3_OverlapBalancedCombination_16.1.yaml | Verify_Task3_OverlapBalancedCombination_16.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 825340eb |
| config/Verify_Task3_OverlapLoss_15.1.yaml | Verify_Task3_OverlapLoss_15.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history d9a3c9f5 |
| config/Verify_Task3_PairwiseRanking_10.1.yaml | Verify_Task3_PairwiseRanking_10.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 4cc4c9fa |
| config/Verify_Task3_ParameterEMA_40.1.yaml | Verify_Task3_ParameterEMA_40.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 749b74af |
| config/Verify_Task3_PositiveWeightScale_37.1.yaml | Verify_Task3_PositiveWeightScale_37.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 372a56cb |
| config/Verify_Task3_ProjectionFeatureCombination_24.1.yaml | Verify_Task3_ProjectionFeatureCombination_24.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history a37dc128 |
| config/Verify_Task3_RawDependencyRebuild_8.1.yaml | Verify_Task3_RawDependencyRebuild_8.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 443fb21d |
| config/Verify_Task3_RawHardness_8.1.yaml | Verify_Task3_RawHardness_8.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history e1052fba |
| config/Verify_Task3_RepresentationCombination_19.1.yaml | Verify_Task3_RepresentationCombination_19.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 894c4a95 |
| config/Verify_Task3_ResidualCorrection_9.1.yaml | Verify_Task3_ResidualCorrection_9.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 2bb2836c |
| config/Verify_Task3_ResidualDropoutHigh_51.1.yaml | Verify_Task3_ResidualDropoutHigh_51.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_ResidualDropout_38.1.yaml | Verify_Task3_ResidualDropout_38.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 274718ff |
| config/Verify_Task3_ResidualHeadBeta1RefreshPortfolio_108.1.yaml | Verify_Task3_ResidualHeadBeta1RefreshPortfolio_108.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 61ce0e65 |
| config/Verify_Task3_ResidualHeadBeta1Refresh_107.1.yaml | Verify_Task3_ResidualHeadBeta1Refresh_107.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 61ce0e65 |
| config/Verify_Task3_ResidualHeadBeta2RefreshPortfolio_106.1.yaml | Verify_Task3_ResidualHeadBeta2RefreshPortfolio_106.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 87cf5f87 |
| config/Verify_Task3_ResidualHeadBeta2Refresh_105.1.yaml | Verify_Task3_ResidualHeadBeta2Refresh_105.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 87cf5f87 |
| config/Verify_Task3_ResidualHeadCapacityRefreshPortfolio_98.1.yaml | Verify_Task3_ResidualHeadCapacityRefreshPortfolio_98.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 314e5272 |
| config/Verify_Task3_ResidualHeadCapacityRefresh_97.1.yaml | Verify_Task3_ResidualHeadCapacityRefresh_97.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 314e5272 |
| config/Verify_Task3_ResidualHeadDepthWidth_31.1.yaml | Verify_Task3_ResidualHeadDepthWidth_31.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history b7b7a9b3 |
| config/Verify_Task3_ResidualHeadDepthWidth_31.2.yaml | Verify_Task3_ResidualHeadDepthWidth_31.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history b7b7a9b3 |
| config/Verify_Task3_ResidualHeadEpsilonRefreshPortfolio_110.1.yaml | Verify_Task3_ResidualHeadEpsilonRefreshPortfolio_110.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 746aa856 |
| config/Verify_Task3_ResidualHeadEpsilonRefresh_109.1.yaml | Verify_Task3_ResidualHeadEpsilonRefresh_109.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 746aa856 |
| config/Verify_Task3_ResidualHeadGradientClipRefreshPortfolio_112.1.yaml | Verify_Task3_ResidualHeadGradientClipRefreshPortfolio_112.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history dcdbbba9 |
| config/Verify_Task3_ResidualHeadGradientClipRefresh_111.1.yaml | Verify_Task3_ResidualHeadGradientClipRefresh_111.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history dcdbbba9 |
| config/Verify_Task3_ResidualHeadLearningRateRefreshPortfolio_102.1.yaml | Verify_Task3_ResidualHeadLearningRateRefreshPortfolio_102.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 8552a6e0 |
| config/Verify_Task3_ResidualHeadLearningRateRefresh_101.1.yaml | Verify_Task3_ResidualHeadLearningRateRefresh_101.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 8552a6e0 |
| config/Verify_Task3_ResidualHeadNormActivationRefreshPortfolio_100.1.yaml | Verify_Task3_ResidualHeadNormActivationRefreshPortfolio_100.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history dda30d9f |
| config/Verify_Task3_ResidualHeadNormActivationRefresh_99.1.yaml | Verify_Task3_ResidualHeadNormActivationRefresh_99.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history dda30d9f |
| config/Verify_Task3_ResidualHeadNormActivation_29.1.yaml | Verify_Task3_ResidualHeadNormActivation_29.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history a3219db7 |
| config/Verify_Task3_ResidualHeadWeightDecayRefreshPortfolio_104.1.yaml | Verify_Task3_ResidualHeadWeightDecayRefreshPortfolio_104.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 28b7964a |
| config/Verify_Task3_ResidualHeadWeightDecayRefresh_103.1.yaml | Verify_Task3_ResidualHeadWeightDecayRefresh_103.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 28b7964a |
| config/Verify_Task3_ResidualOutputInitialization_30.1.yaml | Verify_Task3_ResidualOutputInitialization_30.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history e1c83611 |
| config/Verify_Task3_SafeFactorCombination_44.1.yaml | Verify_Task3_SafeFactorCombination_44.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_SemanticBlockProjection_25.1.yaml | Verify_Task3_SemanticBlockProjection_25.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 3969e8d1 |
| config/Verify_Task3_SpatialRobust_5.2.yaml | Verify_Task3_SpatialRobust_5.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task3_SupervisedContrastive_18.1.yaml | Verify_Task3_SupervisedContrastive_18.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history 0f825b20 |
| config/Verify_Task3_TrainingHorizon_47.1.yaml | Verify_Task3_TrainingHorizon_47.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history f7c567b7 |
| config/Verify_Task3_TrainingResidualScale_32.1.yaml | Verify_Task3_TrainingResidualScale_32.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history eb00dc67 |
| config/Verify_Task3_UltraNarrowBottleneck_20.1.yaml | Verify_Task3_UltraNarrowBottleneck_20.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | git_history c491d995 |
| config/Verify_Task3_UniformFMT_9.1.yaml | Verify_Task3_UniformFMT_9.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task4A_GeometricCurlKMeans_2.2.yaml | Verify_Task4A_GeometricCurlKMeans_2.2 | Task4-a逐primitive/逐实例聚类 | 固定七线中心；跨primitive池化不是同primitive重选中心 | 无；固定描述/KMeans | working_tree |
| config/Verify_Task4A_PerVortexFMTProxySweep_2.1.yaml | Verify_Task4A_PerVortexFMTProxySweep_2.1 | Task4-a逐primitive/逐实例聚类 | 固定七线中心；跨primitive池化不是同primitive重选中心 | 无；固定描述/KMeans | working_tree |
| config/Verify_Task4A_VelocityCurlBaseline_1.1.yaml | Verify_Task4A_VelocityCurlBaseline_1.1 | Task4-a逐primitive/逐实例聚类 | 固定七线中心；跨primitive池化不是同primitive重选中心 | 无；固定描述/KMeans | working_tree |
| config/Verify_Task4B_FullVolumeMemorization_1.1.yaml | Verify_Task4B_FullVolumeMemorization_1.1 | Task4-b多分支分类 | 固定七线中心；不轮换 | fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有 | working_tree |
| config/Verify_Task4B_PooledMemorization_3.1.yaml | Verify_Task4B_PooledMemorization_3.1 | Task4-b多分支分类 | 固定七线中心；不轮换 | fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有 | working_tree |
| config/Verify_Task4B_ProxyLabels_1.1.yaml | Verify_Task4B_ProxyLabels_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_Task4B_ProxyLabels_1.2.yaml | Verify_Task4B_ProxyLabels_1.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_Task4B_VelocityCurlMemorization_4.1.yaml | Verify_Task4B_VelocityCurlMemorization_4.1 | Task4-b纯特征MLP | 固定七线中心；不轮换 | 无；普通/残差MLP | working_tree |
| config/Verify_Task4C_BaselineDevice_4.17.json | Verify_Task4C_BaselineDevice_4.17 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Verify_Task4C_DirectionalSpectrum_4.1.json | Verify_Task4C_DirectionalSpectrum_4.1 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Verify_Task4C_Encoders_4.15.json | Verify_Task4C_Encoders_4.15 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Verify_Task4C_P35GTHead_1.1.json | Verify_Task4C_P35GTHead_1.1 | 最新扩充集合的原p35/h0小网络 | 全部有效线轮换，每中心最近6邻居 | 无ConvNd；逐中心MLP→跨中心mean/max→MLP | working_tree |
| config/Verify_Task4C_RetainDirection_4.16.json | Verify_Task4C_RetainDirection_4.16 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Verify_Task4C_TrainingMemorization_4.12.json | Verify_Task4C_TrainingMemorization_4.12 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/Verify_Task5_CylinderHyperparams_1.1.yaml | Verify_Task5_CylinderHyperparams_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task5_CylinderHyperparams_1.1_baseline_re160.yaml | Verify_Task5_CylinderHyperparams_1.1_baseline_re160 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task5_CylinderHyperparams_1.1_baseline_re640.yaml | Verify_Task5_CylinderHyperparams_1.1_baseline_re640 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task5_Re160FreshTimes_1.1.yaml | Verify_Task5_Re160FreshTimes_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task5_Re160FreshTimes_1.1_baseline.yaml | Verify_Task5_Re160FreshTimes_1.1_baseline | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/Verify_Task5_Re160FreshTimes_1.1_cache.yaml | Verify_Task5_Re160FreshTimes_1.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_Task5_Re160FreshTimes_1.1_labels.yaml | Verify_Task5_Re160FreshTimes_1.1_fresh_labels | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/Verify_Task678_DirectFMTFit_1.1.json | Verify_Task678_DirectFMTFit_1.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task678_HanSampling_1.1.json | Verify_Task678_HanSampling_1.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task678_VectorFMT_1.1.json | Verify_Task678_VectorFMT_1.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task6_DirectAudit_1.1.json | Verify_Task6_DirectAudit_1.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task6_DirectNeural_5.1.json | Verify_Task6_DirectNeural_5.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task6_FMTGeometryMoE_1.1.json | Verify_Task6_FMTGeometryMoE_1.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task6_PNNTrans_1.1.json | Verify_Task6_PNNTrans_1.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task6_PNNTrans_1.2.json | Verify_Task6_PNNTrans_1.2 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task6_PNNTrans_1.3.json | Verify_Task6_PNNTrans_1.3 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task6_ReconstructionAudit_2.2.json | Verify_Task6_ReconstructionAudit_2.2 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_Task6_ScarceGeneralization_4.1.json | Verify_Task6_ScarceGeneralization_4.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/Verify_VAEFailureHighRe_1.1.yaml | Verify_VAEFailureHighRe_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | git_history 6baa6b2b |
| config/Verify_VAEObjective3D_1.1.yaml | Verify_VAEObjective3D_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | archive:retired-validation-20260905.zip |
| config/cylinder_time_policy.json |  | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/ftle_2d_baseline_provenance.json |  | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_3DFMTVAE_1.1.yaml | mainExp_3DFMTVAE_1.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_ASAPFMT_Task35_1.1.json | mainExp_ASAPFMT_Task35_1.1 | Task3/5直接c156三分支 | fmt_c156/asap_fmt均7中心；raw_c156只逐线处理原坐标 | 三分支均无ConvNd；全连接残差网络 | working_tree |
| config/mainExp_Task123NewFlows_1.1_confirmation_cache.yaml | mainExp_Task123NewFlows_1.1_confirmation_cache | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task123NewFlows_1.1_development_cache.yaml | mainExp_Task123NewFlows_1.1_development_cache | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task135_FMTv8_1.1.json | mainExp_Task135_FMTv8_1.1 | V8/p00–p49/n0–n3跨任务搜索 | Task1/3/5固定slot0；Task4-c全部有效线轮换 | Task1聚类无；历史Task3/5完整预测含三层Conv1d；Task4-c FMT无 | working_tree |
| config/mainExp_Task1_3D_2.1.yaml | mainExp_Task1_3D_2.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task1_3D_2.2_newflows.yaml | mainExp_Task1_3D_2.2_newflows | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task1_3D_3.3_reference_new2.yaml | mainExp_Task1_3D_3.3_reference_new2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task1_3D_3.3_reference_old8.yaml | mainExp_Task1_3D_3.3_reference_old8 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task1_3D_4.1_uniform.yaml | mainExp_Task1_3D_4.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task23_3D_4.1_labels_new2.yaml | mainExp_Task23_3D_4.1_labels_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task23_3D_4.1_labels_old8.yaml | mainExp_Task23_3D_4.1_labels_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task2_3D_2.1.yaml | mainExp_Task2_3D_2.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task2_3D_2.1_highre_confirmation_cache.yaml | mainExp_Task2_3D_2.1_highre_confirmation | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task2_3D_2.2.yaml | mainExp_Task2_3D_2.2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task2_3D_2.3.yaml | mainExp_Task2_3D_2.3 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task2_3D_2.4_newflows.yaml | mainExp_Task2_3D_2.4_newflows | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task2_3D_3.2.yaml | mainExp_Task2_3D_3.2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task2_3D_3.3.yaml | mainExp_Task2_3D_3.3 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task2_3D_4.1.yaml | mainExp_Task2_3D_4.1 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task2_3D_5.2.yaml | mainExp_Task2_3D_5.2 | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task2_3D_6.2_uniform_confirmation.yaml | mainExp_Task2_3D_6.2_uniform_confirmation | 固定特征/Task1聚类/旧Task2全连接VAE | 固定slot0；采样中心变化产生新primitive，不是内部轮换 | 特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型 | working_tree |
| config/mainExp_Task3NewFlows_2.3_baselines.yaml | mainExp_Task3NewFlows_2.3_baselines | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3NewFlows_2.3_confirmation_labels.yaml | mainExp_Task3NewFlows_2.3_confirmation_labels | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3NewFlows_2.3_development_labels.yaml | mainExp_Task3NewFlows_2.3_development_labels | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3NewFlows_2.3_evaluate.yaml | mainExp_Task3NewFlows_2.3_evaluate | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3NewFlows_2.3_residual.yaml | mainExp_Task3NewFlows_2.3_residual | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3Universality_2.1_cache.yaml | mainExp_Task3Universality_2.1 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3Universality_2.1_evaluate.yaml | mainExp_Task3Universality_2.1_evaluate | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3Universality_2.1_labels.yaml | mainExp_Task3Universality_2.1_labels | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3Universality_2.1_train.yaml | mainExp_Task3Universality_2.1_train | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3Universality_2.2_cache.yaml | mainExp_Task3Universality_2.2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3Universality_2.2_evaluate.yaml | mainExp_Task3Universality_2.2_evaluate | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3Universality_2.2_labels.yaml | mainExp_Task3Universality_2.2_labels | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3Universality_2.2_train_seed20.yaml | mainExp_Task3Universality_2.2_train_seed20 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3Universality_2.2_train_seeds21_22.yaml | mainExp_Task3Universality_2.2_train_seeds21_22 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_baselines_new2.yaml | mainExp_Task3_3D_3.1_baselines_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_baselines_old8.yaml | mainExp_Task3_3D_3.1_baselines_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_confirmation_new2.yaml | mainExp_Task3_3D_3.1_confirmation_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_confirmation_old8.yaml | mainExp_Task3_3D_3.1_confirmation_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_evaluate.yaml | mainExp_Task3_3D_3.1_evaluate | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_fmt_new2.yaml | mainExp_Task3_3D_3.1_fmt_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_fmt_old8.yaml | mainExp_Task3_3D_3.1_fmt_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_labels_new2.yaml | mainExp_Task3_3D_3.1_labels_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_3.1_labels_old8.yaml | mainExp_Task3_3D_3.1_labels_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_3.1_raw_pca_new2.yaml | mainExp_Task3_3D_3.1_raw_pca_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.1_raw_pca_old8.yaml | mainExp_Task3_3D_3.1_raw_pca_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_baselines_new2.yaml | mainExp_Task3_3D_3.2_global_ivd_baselines_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_baselines_old8.yaml | mainExp_Task3_3D_3.2_global_ivd_baselines_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_evaluate.yaml | mainExp_Task3_3D_3.2_global_ivd_evaluate | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_fmt_new2.yaml | mainExp_Task3_3D_3.2_global_ivd_fmt_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_fmt_old8.yaml | mainExp_Task3_3D_3.2_global_ivd_fmt_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_labels_confirmation_new2.yaml | mainExp_Task3_3D_3.2_global_ivd_labels_confirmation_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_labels_confirmation_old8.yaml | mainExp_Task3_3D_3.2_global_ivd_labels_confirmation_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_labels_development_new2.yaml | mainExp_Task3_3D_3.2_global_ivd_labels_development_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_labels_development_old8.yaml | mainExp_Task3_3D_3.2_global_ivd_labels_development_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_raw_pca_new2.yaml | mainExp_Task3_3D_3.2_global_ivd_raw_pca_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_3.2_global_ivd_raw_pca_old8.yaml | mainExp_Task3_3D_3.2_global_ivd_raw_pca_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_4.1.yaml | mainExp_Task3_3D_4.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_5.1.yaml | mainExp_Task3_3D_5.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_5.1_labels_new2.yaml | mainExp_Task3_3D_5.1_labels_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_5.1_labels_old8.yaml | mainExp_Task3_3D_5.1_labels_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_5.2.yaml | mainExp_Task3_3D_5.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_5.2_labels_new2.yaml | mainExp_Task3_3D_5.2_labels_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_5.2_labels_old8.yaml | mainExp_Task3_3D_5.2_labels_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_6.1.yaml | mainExp_Task3_3D_6.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_6.1_labels_new2.yaml | mainExp_Task3_3D_6.1_labels_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_6.1_labels_old8.yaml | mainExp_Task3_3D_6.1_labels_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_7.1.yaml | mainExp_Task3_3D_7.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_7.1_labels_new2.yaml | mainExp_Task3_3D_7.1_labels_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_7.1_labels_old8.yaml | mainExp_Task3_3D_7.1_labels_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_7.2.yaml | mainExp_Task3_3D_7.2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_7.2_labels_new2.yaml | mainExp_Task3_3D_7.2_labels_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_7.2_labels_old8.yaml | mainExp_Task3_3D_7.2_labels_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_8.1.yaml | mainExp_Task3_3D_8.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task3_3D_8.1_labels_new2.yaml | mainExp_Task3_3D_8.1_labels_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_8.1_labels_old8.yaml | mainExp_Task3_3D_8.1_labels_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task3_3D_9.2_uniform_confirmation.yaml | mainExp_Task3_3D_9.2_uniform_confirmation | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task4B_1.1.yaml | mainExp_Task4B_1.1 | Task4-b多分支分类 | 固定七线中心；不轮换 | fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有 | working_tree |
| config/mainExp_Task4B_1.2.yaml | mainExp_Task4B_1.2 | Task4-b多分支分类 | 固定七线中心；不轮换 | fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有 | working_tree |
| config/mainExp_Task4B_ChannelToTBL_2.1.yaml | mainExp_Task4B_ChannelToTBL_2.1 | Task4-b多分支分类 | 固定七线中心；不轮换 | fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有 | working_tree |
| config/mainExp_Task4B_ChannelToTBL_2.2.yaml | mainExp_Task4B_ChannelToTBL_2.2 | Task4-b多分支分类 | 固定七线中心；不轮换 | fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有 | working_tree |
| config/mainExp_Task4B_ChannelToTBL_2.3.yaml | mainExp_Task4B_ChannelToTBL_2.3 | Task4-b多分支分类 | 固定七线中心；不轮换 | fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有 | working_tree |
| config/mainExp_Task4B_PatchSegmentation_5.1.json | mainExp_Task4B_PatchSegmentation_5.1 | Task4-b纯特征MLP | 固定七线中心；不轮换 | 无；普通/残差MLP | working_tree |
| config/mainExp_Task4B_PooledInstanceSplit_3.1.yaml | mainExp_Task4B_PooledInstanceSplit_3.1 | Task4-b多分支分类 | 固定七线中心；不轮换 | fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有 | working_tree |
| config/mainExp_Task4C_BundleQuality_1.1.json | mainExp_Task4C_BundleQuality_1.1 | Task4-c早期论文双向LSTM体系 | FMT辅助分支逐有效线轮换中心、每中心最近6邻居 | 双向LSTM+Linear，无ConvNd；不是纯FMT+MLP | working_tree |
| config/mainExp_Task4C_FMTv8_4.21.json | mainExp_Task4C_FMTv8_4.21 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_FinalAssessment_4.6.json | mainExp_Task4C_FinalAssessment_4.6 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_GTHeadCoverage_1.1.json | mainExp_Task4C_GTHeadCoverage_1.1 | Task4-c c000–c263/c156 | 全部有效线轮换；每中心FPS选6邻居 | FMT无ConvNd；fps_graph含图消息传递，单独标注 | working_tree |
| config/mainExp_Task4C_HairpinBinary_2.1.json | mainExp_Task4C_HairpinBinary_2.1 | Task4-c七线2.1 | FMT固定slot0，一个161维向量 | FMT分支无；体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_InstanceCoverage_9.15.json | mainExp_Task4C_InstanceCoverage_9.15 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_InstanceCoverage_9.15_v2.json | mainExp_Task4C_InstanceCoverage_9.15_v2 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_LinePooling_4.3.json | mainExp_Task4C_LinePooling_4.3 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_ManifoldMixup_4.4.json | mainExp_Task4C_ManifoldMixup_4.4 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_Multiscale_4.1.json | mainExp_Task4C_Multiscale_4.1 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_NearbySampling_4.13.json | mainExp_Task4C_NearbySampling_4.13 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_PaperBundles_3.1.json | mainExp_Task4C_PaperBundles_3.1 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_PhysicalLength_4.14.json | mainExp_Task4C_PhysicalLength_4.14 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_Regularized_4.2.json | mainExp_Task4C_Regularized_4.2 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task4C_ScaleConsistency_4.5.json | mainExp_Task4C_ScaleConsistency_4.5 | Task4-c整束FMT体系（3.1起） | FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告 | FMT分支无ConvNd；同实验的独立体素对照有Conv3d | working_tree |
| config/mainExp_Task5_3D_1.1.yaml | mainExp_Task5_3D_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task5_3D_1.1_baselines_new2.yaml | mainExp_Task5_3D_1.1_baselines_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task5_3D_1.1_baselines_old8.yaml | mainExp_Task5_3D_1.1_baselines_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task5_3D_1.1_evaluate.yaml | mainExp_Task5_3D_1.1 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task5_3D_1.1_fmt_new2.yaml | mainExp_Task5_3D_1.1_fmt_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task5_3D_1.1_fmt_old8.yaml | mainExp_Task5_3D_1.1_fmt_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task5_3D_1.1_labels_confirmation_new2.yaml | mainExp_Task5_3D_1.1_labels_confirmation_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task5_3D_1.1_labels_confirmation_old8.yaml | mainExp_Task5_3D_1.1_labels_confirmation_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task5_3D_1.1_labels_development_new2.yaml | mainExp_Task5_3D_1.1_labels_development_new2 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task5_3D_1.1_labels_development_old8.yaml | mainExp_Task5_3D_1.1_labels_development_old8 | 数据/标签/展示/来源配置 | 本配置不定义中心编码；追溯所引用方法 | 本配置不定义预测网络；不能由展示名称判断 | working_tree |
| config/mainExp_Task5_3D_1.1_raw_pca_new2.yaml | mainExp_Task5_3D_1.1_raw_pca_new2 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task5_3D_1.1_raw_pca_old8.yaml | mainExp_Task5_3D_1.1_raw_pca_old8 | 旧Task3/5监督分类/残差/优化搜索 | 固定原七线中心；不因Raw逐线卷积而轮换几何中心 | 有三层Conv1d；residual_input=fmt_only仍保留Raw logit | working_tree |
| config/mainExp_Task678_FlowMap_1.1.json | mainExp_Task678_FlowMap_1.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/mainExp_Task6_PrimitiveVAE_2.1.json | mainExp_Task6_PrimitiveVAE_2.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| config/mainExp_Task6_Reconstruction_3.1.json | mainExp_Task6_Reconstruction_3.1 | Task6/7/8或Task36几何重建 | 原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心 | 已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告 | working_tree |
| pnn/configs/dev.yaml |  | 第三方Point-NN库的数据/聚类配置 | 点云邻域/FPS，非固定七线Fourier描述 | 这些YAML本身未指定Point_PN；不能把同库Conv1d/2d当作已执行证据 | working_tree |
| pnn/configs/test.yaml |  | 第三方Point-NN库的数据/聚类配置 | 点云邻域/FPS，非固定七线Fourier描述 | 这些YAML本身未指定Point_PN；不能把同库Conv1d/2d当作已执行证据 | working_tree |
