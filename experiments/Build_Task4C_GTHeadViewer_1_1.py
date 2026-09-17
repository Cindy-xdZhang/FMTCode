"""Build the new classification viewer with an explicit per-instance census."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np


def checked_added_membership(gt, centers, owners, added):
    """Verify each new center inside its intended GT, including overlaps.

    A global VTK locator returns only one containing cell. Overlapping GT
    volumes can therefore yield different valid IDs across locator builds.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    labels=vtk_to_numpy(gt.GetCellData().GetArray('VortexIds'))
    verified=0
    for iid in np.unique(owners[added]):
        cell_ids=vtk.vtkIdList()
        for index in np.flatnonzero(labels==iid):cell_ids.InsertNextId(int(index))
        assert cell_ids.GetNumberOfIds()>0,f'Unknown target GT {iid}'
        extract=vtk.vtkExtractCells();extract.SetInputData(gt);extract.SetCellList(cell_ids);extract.Update()
        locator=vtk.vtkStaticCellLocator();locator.SetDataSet(extract.GetOutput());locator.BuildLocator()
        points=centers[added&(owners==iid)]
        assert all(locator.FindCell(p)>=0 for p in points),f'New center outside intended GT {iid}'
        verified+=len(points)
    return verified


def decorate(output,package,input_root,require_complete=True):
    from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset,sample_gt
    from FMT_Utils.Task4C_GTHeadCoverage_1_1 import sha
    output=Path(output);path=output/'index.html';html=path.read_text(encoding='utf8')
    start=html.index('const DATA=')+len('const DATA=')
    payload,length=json.JSONDecoder().raw_decode(html[start:])
    manifest=json.loads((Path(package)/'manifest.json').read_text())
    census={};membership={};added_census={}
    for flow in manifest['flows']:
        name=flow['name'];gt=read_dataset(Path(input_root)/flow['gt']);census[name]={};membership[name]={};added_census[name]={}
        for role,s in payload['flows'][name]['splits'].items():
            centers=np.asarray(s['center']);ids,_=sample_gt(gt,centers)
            added=np.asarray(s.get('mandatory_head',[False]*len(ids)),dtype=bool)
            if require_complete:
                owners=np.asarray(s['owner_instance'],dtype=int)
                verified=checked_added_membership(gt,centers,owners,added)
                membership[name][role]=dict(verified_added_centers=verified,
                    global_locator_different_valid_GT=int(np.sum(added&(ids!=owners))))
                # Keep the sampling target only after independent containment.
                # Existing rows still use their geometric, not label, membership.
                ids[added]=owners[added]
            s['center_gt_instance']=ids.tolist()
            mask=(np.asarray(s['labels'])==1)&np.asarray(s['center_is_head'],dtype=bool)
            totals={str(i):int(np.sum(mask&(ids==i))) for i in payload['flows'][name]['gt']['instances']}
            if require_complete:assert all(totals.values()),f'Incomplete {name}/{role} GT head coverage'
            census[name][role]=totals
            added_totals={str(i):int(np.sum(mask&added&(ids==i))) for i in payload['flows'][name]['gt']['instances']}
            if require_complete:
                assert all(n==(30 if role=='train' else 10) for n in added_totals.values()),f'Incorrect {name}/{role} added GT quotas'
            added_census[name][role]=added_totals
    payload['viewer_version']='GT头部覆盖 1.1' if require_complete else '旧4.14覆盖诊断';payload['pending']=[]
    html=html[:start]+json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\/')+html[start+length:]
    assert html.count('refreshRegions();render(true).then(')==1
    html=html.replace('refreshRegions();render(true).then(','installGTCoverage();refreshRegions();render(true).then(')
    needle="indices=selectClassIndices(indices,selectionValues,$('hairpin').checked?+$('hairpinCount').value:0,$('nonhairpin').checked?+$('nonhairpinCount').value:0);"
    assert needle in html
    html=html.replace(needle,needle+'state.visibleIndices=indices.slice();')
    style='\n#coveragePanel{padding:10px 23px;background:#eef5f2;border-top:1px solid #dce8e2;font-size:12px}#coveragePanel p{margin:4px 0}#coveragePanel summary{cursor:pointer}.coverage-scroll{max-height:220px;overflow:auto}#coveragePanel table{width:100%;border-collapse:collapse;text-align:right}#coveragePanel th,#coveragePanel td{padding:4px 10px;border-bottom:1px solid #d8e2dd}#coveragePanel .missing{background:#fff0e6}#coveragePanel button{padding:3px 8px}\n'
    html=html.replace('</style>',style+'</style>',1)
    controls_sha=sha(Path('experiments/templates/task4c_gt_coverage_1_1.js'))
    html=html.replace('</head>','<script src="gt_coverage.js?v='+controls_sha[:16]+'"></script></head>',1)
    html=html.replace('<input id="analysis" type="checkbox">','<input id="analysis" type="checkbox" checked>')
    old_details="+(s.owner_instance?'<br>附近实例：'+s.owner_instance[id]:'')"
    assert old_details in html
    html=html.replace(old_details,"+(s.center_gt_instance?.[id]>=0?'<br>中心归属GT实例：'+s.center_gt_instance[id]:'')+(s.owner_instance?.[id]>=0?'<br>标签来源实例：'+s.owner_instance[id]:'')")
    html=html.replace('<br>必选 head 线束','<br>本次新增的GT头部线束')
    if require_complete:
        html=html.replace('<option value="train">训练集</option>','<option value="train">新增头部训练束</option>')
    path.write_text(html,encoding='utf8')
    shutil.copyfile('experiments/templates/task4c_gt_coverage_1_1.js',output/'gt_coverage.js')
    complete=all(v>0 for flow in census.values() for role in flow.values() for v in role.values())
    record=dict(complete=complete,html_sha256=sha(path),coverage=census,added_coverage=added_census,
        added_GT_containment_checks=membership,
        overlap_rule='New rows: independently verified sampling GT; existing rows: global containing-cell query.',
        source_manifest_sha256=sha(Path(package)/'manifest.json'))
    (output/'head_coverage_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    build=json.loads((output/'viewer_manifest.json').read_text())
    build['base_template_html_sha256']=build['html_sha256'];build['html_sha256']=sha(path)
    build['head_coverage_manifest_sha256']=sha(output/'head_coverage_manifest.json')
    build['head_controls_sha256']=sha(output/'gt_coverage.js')
    (output/'viewer_manifest.json').write_text(json.dumps(build,indent=2)+'\n',encoding='utf8')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--input-root',required=True)
    p.add_argument('--output',default='outputs/Other_Task4C_GTHeadCoverage_1.1/viewer')
    a=p.parse_args()
    from experiments.Visualize_Task4C_Bundles_3D import build_viewer
    build_viewer(a.package,a.input_root,a.output,False)
    print(json.dumps(decorate(a.output,a.package,a.input_root)),flush=True)
