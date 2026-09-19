"""Dataset v2 viewer (2.1): the training-fold and test-fold sample points with their three lines.

Reuses the frozen 3-D bundle viewer template: GT surfaces, per-instance region selection, count
sliders for the visible samples, and the candidate-head overlay of 1.2.  Every sample is one
"bundle" of its three half-length curves; colour = label (model "样本标签") or, on the test fold,
the FMT 2.1 prediction when a finished result is given.  Nothing here changes the dataset.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np

DATASET = 'outputs/mainExp_Task4C_FixedDataset_2.1'
PACKAGE = 'outputs/mainExp_Task4C_GTHeadCoverage_1.1/viewer_package/manifest.json'
SOURCE_VIEWER = 'outputs/Other_Task4C_GTHeadCoverage_1.1/viewer'
FMT_CONFIG = 'config/mainExp_Task4C_FixedDatasetFMT_2.1.json'
INPUT_ROOT = 'C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D'
KINDS = {0: '在标注单元内', 1: 'GT 包围盒内', 2: '候选分量重叠', 3: '最近实例'}


def fmt_test_probabilities(result_dir, flows):
    """Per-sample mean over the three half-length rows of a finished FMT 2.1 test prediction."""
    result_dir = Path(result_dir); result = json.loads((result_dir/'result.json').read_text())
    with np.load(result_dir/'test_predictions.npz') as z: pred = {k: z[k] for k in z.files}
    per_flow = {}
    for fi, flow in enumerate(flows):
        m = pred['flow_index'] == fi; ids = pred['sample_id'][m]; p = pred['probability'][m]
        order = np.argsort(ids, kind='stable'); ids, p = ids[order], p[order]
        unique, start = np.unique(ids, return_index=True); mean = np.add.reduceat(p, start)/np.diff(np.r_[start, len(p)])
        per_flow[flow['name']] = dict(zip(unique.tolist(), mean.tolist()))
    model = dict(id='fmt21', label=result['candidate']['id']+' · seed'+str(result['seed'])+'（FMT 2.1，仅测试折）', seed=result['seed'], parameters=result['parameters'],
                 version='mainExp_Task4C_FixedDatasetFMT_2.1', metrics=dict(test=dict(per_flow={k: dict(f1=v['f1']) for k, v in result['test']['per_flow'].items()}), train=dict(per_flow={f['name']: dict(f1=0.0) for f in flows})),
                 note='行级预测（样本×长度）按样本取三条长度的平均概率；训练折没有预测')
    return model, per_flow


def build(dataset, package, input_root, output, fmt_result=None, chunk=512, display=200):
    from plotly.offline import get_plotlyjs
    from vtk.util.numpy_support import vtk_to_numpy
    from experiments.Visualize_Task4C_Bundles_3D import array_json, geometry_chunks, read_vtk, gt_surface, save_vtp, COLORS, ANALYSIS_COLORS, sha, write
    from experiments.Build_Task4C_CandidateHeadRegion_1_2 import load_fields
    from FMT_Utils import Task4C_FixedDataset_2_1 as data
    from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as v2
    from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
    dataset, output = Path(dataset), Path(output); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(package).read_text(encoding='utf-8')); fmt_spec = json.loads(Path(FMT_CONFIG).read_text())
    audit = json.loads((dataset/'data_audit.json').read_text())
    # The frozen template reads metrics[split].per_flow[flow].f1 on every render; placeholders keep it alive, the controls script shows the real text.
    placeholder = {f['name']: dict(f1=1.0) for f in manifest['flows']}
    models = [dict(id='labels', label='样本标签（无模型）', seed=0, parameters=0, version='mainExp_Task4C_FixedDataset_2.1', metrics=dict(train=dict(per_flow=placeholder), test=dict(per_flow=placeholder)))]
    fmt = None
    if fmt_result:
        model, fmt = fmt_test_probabilities(fmt_result, manifest['flows']); models.append(model)
    payload = dict(models=models, pending=[], colors=COLORS, analysis_colors=ANALYSIS_COLORS, threshold=.5, display_bundles=display, flows={},
                   default_hairpin_count=display//2, default_nonhairpin_count=display//2, default_view_mode='all', viewer_version='数据集 v2 · 2.1',
                   evidence_note='数据集 v2（mainExp_Task4C_FixedDataset_2.1 r3）：每个样本点一束三条曲线（短/中/长），坐标为物理尺度。训练折 = 折 1–4（含 FMT 2.1 的 10% 验证子集），测试折 = 折 0。颜色默认按标签；选 FMT 2.1 模型时只有测试折有预测。滑块控制可见样本数，几何按需加载。')
    record = dict(version='Other_Task4C_DatasetViewer_2.1', dataset_audit_sha256=sha(dataset/'data_audit.json'), flows={})
    for fi, flow in enumerate(manifest['flows']):
        name = flow['name']; native = Path(input_root)/flow['flow']; gt_path = Path(input_root)/flow['gt']
        assert sha(native) == flow['flow_sha256'] and sha(gt_path) == flow['gt_sha256']
        for filename, digest in audit['frozen_files'][name].items(): assert sha(dataset/'physical'/name/filename) == digest
        domain = list(read_vtk(native).GetBounds()); gt = gt_surface(gt_path); save_vtp(gt, output/f'{name}_ground_truth.vtp')
        points = vtk_to_numpy(gt.GetPoints().GetData()); faces = vtk_to_numpy(gt.GetPolys().GetConnectivityArray()).reshape(-1, 3); ids = vtk_to_numpy(gt.GetCellData().GetArray('VortexIds'))
        entry = dict(domain=domain, gt=dict(points=array_json(points, '<f4'), faces=array_json(faces, '<i4'), ids=array_json(ids, '<i4'), instances=np.unique(ids).tolist(), bounds=list(gt.GetBounds())), splits={})
        seeds, curves, meta = data.load(dataset/'physical'/name); axes, lam, oyf = load_fields(native); thr = float(meta['lambda2_threshold'])
        l, _, _ = interpolate_scalar(seeds, axes, lam); o, _, _ = interpolate_scalar(seeds, axes, oyf)
        masks = v2.split_masks(meta, fi, fmt_spec); validation = masks['validation']
        for split, mask in (('train', ~masks['test']), ('test', masks['test'])):
            rows = np.flatnonzero(mask); n = len(rows)
            pack = dict(geometry=np.ascontiguousarray(curves[rows], np.float32), counts=np.full(n, 3, np.int64))
            probabilities = {'labels': meta['label'][rows].astype(float).tolist()}
            if fmt is not None:
                probabilities['fmt21'] = [fmt[name].get(int(i), float(meta['label'][i])) for i in rows] if split == 'test' else meta['label'][rows].astype(float).tolist()
            entry['splits'][split] = dict(population=n, selected=n, counts=[3]*n, row_ids=rows.tolist(), labels=meta['label'][rows].astype(int).tolist(),
                center=np.round(seeds[rows], 6).tolist(), head_component=meta['component'][rows].astype(int).tolist(), instance=meta['instance'][rows].astype(int).tolist(),
                scale_id=[0]*n, radius=[1.0]*n, neighbor_distance=[0.0]*n, probabilities=probabilities, geometry_chunk_size=chunk,
                geometry_chunks=geometry_chunks(pack, output, name+'_'+split, chunk), center_is_head=meta['label'][rows].astype(bool).tolist(),
                fold=meta['fold'][rows].astype(int).tolist(), validation=validation[rows].astype(bool).tolist(), assignment_kind=meta['assignment_kind'][rows].astype(int).tolist(),
                component_size=meta['component_size'][rows].astype(int).tolist(), gt_owner=meta['gt_owner'][rows].astype(int).tolist(),
                center_lambda2=[float(f'{v:.5g}') for v in l[rows]], center_oyf=[float(f'{v:.5g}') for v in o[rows]], center_step1=[1]*n,
                fmt_test_prediction_available=bool(fmt is not None and split == 'test'))
            record['flows'].setdefault(name, {})[split] = dict(samples=n, hairpin=int(meta['label'][rows].sum()), validation_subset=int(validation[rows].sum()), chunks=len(entry['splits'][split]['geometry_chunks']))
        payload['flows'][name] = entry
        print(json.dumps(dict(flow=name, **record['flows'][name])), flush=True)
    template = Path('experiments/templates/task4c_bundles.html').read_text(encoding='utf-8')
    html = template.replace('/*__PLOTLY__*/', get_plotlyjs()).replace('/*__PAYLOAD__*/', json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    # Text of the frozen template adapted to a dataset (not a classification result) viewer.
    replacements = [('<option value="test">测试集</option><option value="train">训练集</option>', '<option value="test">测试折 0</option><option value="train">训练折 1–4</option>'),
                    ('真实涡线簇 · 预测二分类 · 半透明 Hairpin 标注曲面', '数据集 v2 样本点 · 每点三条涡线 · 半透明 Hairpin 标注曲面'),
                    ('<title>Task4-c · 涡线簇分类</title>', '<title>Task4-c · 数据集 v2 样本</title>'),
                    ('<h1>涡线簇分类 <span style="font-weight:400;color:#738480">/ Task4-c</span></h1>', '<h1>数据集 v2 样本 <span style="font-weight:400;color:#738480">/ Task4-c</span></h1>'),
                    ('<label class="control">分类模型<select id="model"></select></label>', '<label class="control">着色来源<select id="model"></select></label>'),
                    ('<div class="muted" style="margin-bottom:8px">点击线簇中心点查看预测</div>', '<div class="muted" style="margin-bottom:8px">点击样本点查看归属与标签</div>'),
                    ('<input id="analysis" type="checkbox">', '<input id="analysis" type="checkbox" checked>')]
    for old, new in replacements:
        assert html.count(old) == 1, old; html = html.replace(old, new)
    needle = "indices=selectClassIndices(indices,selectionValues,$('hairpin').checked?+$('hairpinCount').value:0,$('nonhairpin').checked?+$('nonhairpinCount').value:0);"
    assert needle in html; html = html.replace(needle, needle+'state.visibleIndices=indices.slice();')
    start = 'refreshRegions();render(true).then('; assert html.count(start) == 1
    html = html.replace(start, 'installDatasetViewer();installCandidateHead();'+start)
    style = '\n#datasetPanel{padding:10px 23px;background:#eef5f2;border-top:1px solid #dce8e2;font-size:12px}#datasetPanel p{margin:4px 0}#datasetPanel table{border-collapse:collapse;text-align:right}#datasetPanel th,#datasetPanel td{padding:3px 10px;border-bottom:1px solid #d8e2dd}\n'
    html = html.replace('</style>', style+'</style>', 1)
    # Candidate-head overlay: same surfaces as the 1.2 viewer (identical flows and thresholds).
    source = Path(SOURCE_VIEWER)
    if (output/'candidate').exists(): shutil.rmtree(output/'candidate')
    shutil.copytree(source/'candidate', output/'candidate'); shutil.copyfile(source/'candidate_head.js', output/'candidate_head.js')
    shutil.copyfile('experiments/templates/task4c_dataset_viewer_2_1.js', output/'dataset_viewer.js')
    html = html.replace('</head>', '<script src="candidate/index.js?v='+sha(output/'candidate/index.js')[:16]+'"></script><script src="candidate_head.js?v='+sha(output/'candidate_head.js')[:16]+'"></script><script src="dataset_viewer.js?v='+sha(output/'dataset_viewer.js')[:16]+'"></script></head>', 1)
    path = output/'index.html'; path.write_text(html, encoding='utf-8')
    record.update(html_sha256=sha(path), template_sha256=sha('experiments/templates/task4c_bundles.html'), controls_sha256=sha(output/'dataset_viewer.js'), candidate_source=str(source),
                  fmt_result=str(fmt_result) if fmt_result else None, models=[m['id'] for m in models], html_mb=path.stat().st_size/1e6)
    write(output/'viewer_manifest.json', record)
    print(json.dumps(dict(status='PASS', viewer=str(path.resolve()), html_mb=record['html_mb'])), flush=True)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default=DATASET); p.add_argument('--package', default=PACKAGE); p.add_argument('--input-root', default=INPUT_ROOT)
    p.add_argument('--output', default='outputs/Other_Task4C_DatasetViewer_2.1/viewer'); p.add_argument('--fmt-result', default=None)
    p.add_argument('--chunk', type=int, default=512); p.add_argument('--display', type=int, default=200)
    a = p.parse_args(); build(a.dataset, a.package, a.input_root, a.output, a.fmt_result, a.chunk, a.display)
