"""Index every available configuration with explicit task-family boundaries.

Family associations are an index into manually reviewed implementations;
they do not assert a historical configuration was executed or completed.
"""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/Verify_FMTArchitectureAudit_1.1'


def associate(path, data):
    name=path.lower(); text=json.dumps(data,ensure_ascii=False,default=str).lower()
    if name in ('config/fittingvatistas.yaml','config/vatistasdataset.yaml','config/other_task4voxelgt_1.1.yaml'):
        return '解析流场拟合/生成或GT体素化，无FMT分类器', '不适用；观察者变换/网格采样不是primitive内部重选中心', '不定义预测CNN；解析模型参数优化不是卷积网络', 'experiments/FittingVatistasParam.py; VatistasFlowDatasetGenerator.py; 配置task=Task4_preparation'
    if name=='config/pathlinefmtclustering.yaml':
        return '早期二维Point-NN式FMT聚类', '每个同时间点都作局部中心；不是后来固定slot0的DFT', '聚类入口明确temporal_head=None，无卷积；可选dft头另列为频域卷积', 'experiments/FMT_Clustering.py:148–154; FMT_Utils/FMT_encoder.py'
    if name.startswith('pnn/configs/'):
        return '第三方Point-NN库的数据/聚类配置', '点云邻域/FPS，非固定七线Fourier描述', '这些YAML本身未指定Point_PN；不能把同库Conv1d/2d当作已执行证据', 'pnn/models/point_nn.py; point_nn_seg.py; point_pn.py分开；未发现正式FMT入口导入point_pn'
    if any(s in name for s in ('visual','closeup','triptych','render','saliency','provenance','download','threshold','labels','_cache','time_policy')) and 'saliency' not in name:
        return '数据/标签/展示/来源配置', '本配置不定义中心编码；追溯所引用方法', '本配置不定义预测网络；不能由展示名称判断', '请读configuration字段中的source/reuse/model路径；不将数据任务当新方法'
    if 'singlecenter' in name or 'single_center' in name:
        return '新单中心候选，正式训练未开始', '固定一个；Task4-c一次选有效线', '无；141→MLP', 'FMT_Utils/FMT_SingleCenter_1_1.py'
    if 'asapfmt' in name:
        return 'Task3/5直接c156三分支', 'fmt_c156/asap_fmt均7中心；raw_c156只逐线处理原坐标', '三分支均无ConvNd；全连接残差网络', 'FMT_Utils/ASAPFMT_Task35_1_1.py; FMT_Utils/Task4C_FPSAugmentSearch_1_1.py'
    if 'ftle' in name:
        return '二维FTLE方法/对照', 'DFT五线固定slot0；老Point-NN为多个点局部中心；多时间窗不换中心', '历史FMT+U-Net/P35融合有Conv2d/转置卷积；插值对照无', 'FMT_Utils/FTLE_Encoders_2D.py; FMT_Utils/FTLE_P35_2D_2_1.py; 历史model_zoo/FTLE_Fusion_2D_2_1.py'
    if 'task4c' in name:
        if any(s in name for s in ('pointnet','geometricbaselines','bilstm')):
            return 'Task4-c独立非FMT几何基线', '无FMT轮换；Point-NN/PointNet++有多个点邻域，BiLSTM逐线后池化', '本仓库这些网络使用Linear/LSTM，无ConvNd；不据PointNet名称猜实现', 'FMT_Utils/Task4C_GeometricBaselines_1_1.py; Task4C_PointNet_1_1.py; Task4C_PointNetPlusPlus_1_1.py'
        if any(s in name for s in ('convcapacity','convresolution')):
            return 'Task4-c独立Conv3D对照', '无FMT特征', '有Conv3d，独立对照不是FMT分支', 'experiments/Task4C_ConvCapacity_4_18.py; Task4C_ConvResolution_4_22.py'
        if 'hairpinbinary_2.1' in name or 'binarydata_2.1' in name:
            return 'Task4-c七线2.1', 'FMT固定slot0，一个161维向量', 'FMT分支无；体素对照有Conv3d', 'FMT_Utils/Task4C_HairpinBinary_2_1.py'
        if 'bundlequality' in name or 'paperbaseline' in name:
            return 'Task4-c早期论文双向LSTM体系', 'FMT辅助分支逐有效线轮换中心、每中心最近6邻居', '双向LSTM+Linear，无ConvNd；不是纯FMT+MLP', 'FMT_Utils/Task4C_Bundles_1_1.py; Task4C_PaperBaseline_1_1.py'
        if 'fpsaugment' in name or 'gtheadcoverage' in name:
            return 'Task4-c c000–c263/c156', '全部有效线轮换；每中心FPS选6邻居', 'FMT无ConvNd；fps_graph含图消息传递，单独标注', 'FMT_Utils/Task4C_FPSAugmentSearch_1_1.py; experiments/Task4C_FPSAugmentSearch_1_1.py'
        if 'p35gthead' in name:
            return '最新扩充集合的原p35/h0小网络', '全部有效线轮换，每中心最近6邻居', '无ConvNd；逐中心MLP→跨中心mean/max→MLP', 'experiments/Task4C_P35GTHead_1_1.py; FMT_Utils/Task4C_LinePooling_4_3.py'
        return 'Task4-c整束FMT体系（3.1起）', 'FMT逐有效线轮换中心；最近6/FPS6按具体配置，详见主报告', 'FMT分支无ConvNd；同实验的独立体素对照有Conv3d', 'FMT_Utils/Task4C_PaperBundles_3_1.py; Task4C_LinePooling_4_3.py; Task4C_Encoders_4_15.py; FMT_P35_NormFrequency_3_1.py'
    if 'task4b' in name:
        if any(s in name for s in ('geometrysearch','patchsegmentation')) or '"variant": "fmt_only"' in text:
            return 'Task4-b纯特征MLP', '固定七线中心；不轮换', '无；普通/残差MLP', 'FMT_Utils/Task4B_Classifier_3D.py; experiments/Task4B_GeometrySearch_5_2.py'
        if 'clustering' in name:
            return 'Task4-b固定特征聚类', '固定七线中心；不轮换', '无；KMeans/固定特征', 'experiments/Task4B_FMTFourClassClustering_1_1.py'
        return 'Task4-b多分支分类', '固定七线中心；不轮换', 'fmt_only无；历史raw_fmt有三层Conv1d；Raw对照也有', 'FMT_Utils/Task4B_Classifier_3D.py; 历史PathlineClassifier_3D.py'
    if 'task4a' in name:
        return 'Task4-a逐primitive/逐实例聚类', '固定七线中心；跨primitive池化不是同primitive重选中心', '无；固定描述/KMeans', 'FMT_Utils/Task4A_StreamlineClustering_3D.py; Task4A_PerVortexClustering_3D.py'
    if any(s in name for s in ('task6','task36','task678')):
        return 'Task6/7/8或Task36几何重建', '原161维/有向FMT固定slot0；Point-NN分支另有全点局部中心', '已核查实现为Linear/Transformer/LSTM类，无ConvNd；细分见主报告', 'FMT_Utils/PrimitiveVAE_3D.py; Task6Recovery_3D.py; Task6FMTGeometryMoE_3D.py; Task6PNNTrans_3D.py; FlowMapModels_3D.py'
    if 'fmtv8' in name:
        return 'V8/p00–p49/n0–n3跨任务搜索', 'Task1/3/5固定slot0；Task4-c全部有效线轮换', 'Task1聚类无；历史Task3/5完整预测含三层Conv1d；Task4-c FMT无', 'experiments/FMTv8_Search_2_2.py; FMTv8_NormFrequency_3_1.py; FMT_Utils/FMT_P35_NormFrequency_3_1.py'
    tasks=data.get('tasks',[]) if isinstance(data,dict) else []
    if any(s in name for s in ('task123','task135','fmtallv2','largeneighbor','aivdtransfer','objectivitymechanism')) or len(tasks)>1:
        return '多个任务共享固定特征比较', 'FFT特征固定原中心；25线扩展也不轮换；观察者实验详见主报告', 'Task1聚类/Task2全连接VAE无；若含旧Task3/5则Raw卷积分支有Conv1d', '配置tasks/features字段；experiments/Run_Task1235_AngleFeatures_1_1.py及相应历史驱动'
    if 'task3' in name or 'task5' in name:
        return '旧Task3/5监督分类/残差/优化搜索', '固定原七线中心；不因Raw逐线卷积而轮换几何中心', '有三层Conv1d；residual_input=fmt_only仍保留Raw logit', '历史FMT_Utils/PathlineClassifier_3D.py; experiments/Verify_Task3_FMTResidual.py'
    if any(s in name for s in ('task2','vae','highre','3dfmt','task1','pathlinefmtclustering3d')):
        return '固定特征/Task1聚类/旧Task2全连接VAE', '固定slot0；采样中心变化产生新primitive，不是内部轮换', '特征编码、KMeans、全连接VAE无ConvNd；数据生成本身无模型', 'FMT_Utils/DFT_FMT_3D.py; Task12Data_3D.py; VAE_3D.py'
    if any(s in name for s in ('objectivity','observedpathline','translation','relativefourier','objectivefmt','pathlinehyper','featurecontrast','aivdlongtime')):
        return '几何/观察者/客观性诊断', '固定材料对应关系；七平移观察者版本另列，不能从observer词猜中心策略', '诊断公式无预测CNN；若复用分类模型需按该模型计', 'FMT_Utils/ObjectivityMechanism_3D.py; SevenTranslationObservers_3D.py; ASAPFrame_1_1.py'
    return '需按具体入口核实的早期/辅助配置', '未据名称推断', '未据名称推断', '保留完整configuration字段；本行不作已核实无卷积结论'


def main():
    records=json.loads((OUT/'configurations.json').read_text(encoding='utf-8'))
    best={}
    rank={'working_tree':0,'git_history':1}
    for row in records:
        key=row['path']
        if key not in best or rank.get(row['origin'],2)<rank.get(best[key]['origin'],2):
            best[key]=row
    catalog=[]
    for path,row in sorted(best.items()):
        data=row['configuration']
        family,centers,conv,evidence=associate(path,data or {})
        catalog.append(dict(path=path,experiment=row['experiment'],family=family,center_policy=centers,convolution=conv,
            evidence=evidence,origin=row['origin'],commit=row['commit'],sha256=row['sha256'],
            execution_status='配置定义；不据文件存在宣称运行或完成',configuration=json.dumps(data,ensure_ascii=False,default=str)))
    with (OUT/'experiment_configuration_catalog.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(catalog[0]));w.writeheader();w.writerows(catalog)
    lines=['# FMT实验配置完整索引：2026-09-17','',
      '本表按配置路径逐项登记，配置内容完整保存在CSV的configuration列；同一路径全部历史版本见configuration_inventory.csv。本表的结构栏指向已核查的共享实现，**不把配置文件存在等同于已执行，也不把一个多分支实验混称为同一种网络**。具体几何公式、分支差异和运行证据见主审计报告。',
      '',f'本地不同配置路径：{len(catalog)}。按已核查的调用族登记；库模板不等于正式实验，历史版本见原始CSV。','',
      '| 配置 | 实验名 | 对应实现 | 中心策略 | 卷积情况 | 来源 |','|---|---|---|---|---|---|']
    for r in catalog:
        cells=[r['path'],str(r['experiment']),r['family'],r['center_policy'],r['convolution'],r['origin']+(' '+r['commit'][:8] if r['commit'] else '')]
        lines.append('| '+' | '.join(s.replace('|','/').replace('\n',' ') for s in cells)+' |')
    (ROOT/'docs/FMT_experiment_architecture_catalog_2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Preserve every scheduler-table row as evidence, including old and failed jobs.
    registry=ROOT/'docs/ibex_run_registry.md'
    runs=[]
    for line_number,line in enumerate(registry.read_text(encoding='utf-8').splitlines(),1):
        if line.startswith('|') and re.search(r'\b5\d{7}\b',line):
            runs.append(dict(source=str(registry.relative_to(ROOT)),line=line_number,record=line,
                             status='original_record_preserved_not_reinterpreted_as_completed'))
    with (OUT/'scheduler_record_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['source','line','record','status']);w.writeheader();w.writerows(runs)
    print(json.dumps(dict(configurations=len(catalog),scheduler_rows=len(runs),unresolved=[r['path'] for r in catalog if r['family'].startswith('需按')]),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
