"""Add feature-step diagrams and exact signed-window hairpin explanations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from FMT_Utils.Task4C_SupportView_1_2 import summarize_bundle


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(o): return json.dumps(o, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
def dump(p,o): Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def build(old, output, source, update_entry=False):
    old, output, source = Path(old), Path(output), Path(source)
    if old.resolve() == output.resolve(): raise ValueError('Keep the old analysis artifacts separate')
    features, saliency = output/'features', output/'saliency'
    features.mkdir(parents=True, exist_ok=True); (saliency/'data').mkdir(parents=True, exist_ok=True)
    # No clustering, encoding or new model evaluation: all matrices stay frozen.
    for path in (old/'features').iterdir():
        if path.name == 'index.html': continue
        if path.is_dir(): shutil.copytree(path, features/path.name, dirs_exist_ok=True)
        else: shutil.copy2(path, features/path.name)
    templates = Path(__file__).parent/'templates'
    html = (templates/'fmt_feature_contrast.html').read_text(encoding='utf-8')
    html = html.replace('<script src="data.js"></script>', '<script src="data.js"></script><script src="feature_diagram.js"></script>')
    diagram = '''<section class="card wide" aria-label="当前特征的FMT步骤图"><div class="head"><h2>所选特征来自 FMT 的哪一步？</h2><span class="note">橙色为当前分支 · 随特征与样本联动</span></div><svg id="featurePipeline" role="img" aria-label="选中特征的计算流程" viewBox="0 0 1275 370" style="display:block;width:100%;min-height:235px"></svg><div class="detail" style="min-height:0"><div id="pipelineFormula" class="formula"></div><p id="pipelineOutput"></p><p id="pipelineNote" class="note"></p></div></section>'''
    html = html.replace('<main>', '<main>'+diagram, 1)
    html = html.replace('async function drawFeature(){', 'async function drawFeature(){window.updateFeatureDiagram(D,feature,sample);')
    html = html.replace("$('sample').value=String(row);", "window.updateFeatureDiagram(D,feature,row);$('sample').value=String(row);")
    (features/'index.html').write_text(html, encoding='utf-8')
    shutil.copy2(templates/'fmt_feature_diagram_1_2.js', features/'feature_diagram.js')

    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    records=[]; diagnostics=[]
    for record in manifest['records']:
        path=source/record['file']
        assert sha(path)==record['sha256'], record['id']
        with np.load(path, allow_pickle=False) as z: values={k:z[k] for k in z.files}
        n=record['count']; info=summarize_bundle(values,n)
        data={k:values[k][:n].tolist() for k in ('geometry','normalized_geometry','gradient','smooth_gradient','local_shape_delta')}
        data['patches']=[dict(indices=idx.tolist(),margin_delta=float(d),probability=float(p),probability_delta=float(dp))
            for idx,d,p,dp in zip(values['patch_indices'],values['patch_margin_delta'],values['patch_probability'],values['patch_probability_delta'])]
        data['support']=info
        r=dict(record, support=info)
        # Use the original differentiable evaluation consistently in before/after comparisons.
        r['display_probability']=float(values['probability']);r['display_margin']=float(values['margin'])
        assert abs(r['display_probability']-record['probability'])<1e-4
        (saliency/'data'/f"{r['id']}.js").write_text('window.registerSaliency('+js(r['id'])+','+js(data)+');\n',encoding='utf-8')
        dump(saliency/'data'/f"{r['id']}.json",dict(record=r,geometry_and_scores=data))
        records.append(r);diagnostics.append(dict(id=r['id'],flow=r['flow'],**info))
    payload=dict(version='Other_FMT_AnalysisWorkbench_1.2',model=manifest['model'],records=records)
    (saliency/'data.js').write_text('window.SUPPORT_DATA='+js(payload)+';\n',encoding='utf-8')
    html=(templates/'task4c_support_1_2.html').read_text(encoding='utf-8')
    (saliency/'index.html').write_text(html,encoding='utf-8')
    shutil.copy2(features/'plotly.min.js',saliency/'plotly.min.js')
    entry=(templates/'fmt_analysis_workbench.html').read_text(encoding='utf-8').replace('__SALIENCY_URL__','saliency/index.html')
    (output/'index.html').write_text(entry,encoding='utf-8')
    report=dict(version=payload['version'],source_manifest_sha256=sha(source/'manifest.json'),
        previous_feature_arrays_sha256=sha(old/'features/analysis_arrays.npz'),
        unchanged_feature_arrays=sha(old/'features/analysis_arrays.npz')==sha(features/'analysis_arrays.npz'),
        source_bundles_checked=len(records),total_interventions=sum(d['windows'] for d in diagnostics),
        negative_absolute_max_count=sum(d['previous_absolute_max_was_negative'] for d in diagnostics),
        positive_support_bundles=sum(d['strongest_support'] is not None for d in diagnostics),
        per_flow={f:dict(samples=sum(d['flow']==f for d in diagnostics),
            negative_absolute_max_count=sum(d['flow']==f and d['previous_absolute_max_was_negative'] for d in diagnostics),
            mean_old_color_cancellation_fraction=float(np.mean([d['positive_window_covered_points_with_nonpositive_old_color_fraction'] for d in diagnostics if d['flow']==f]))) for f in ('channel','tbl')},
        bundle_diagnostics=diagnostics,score='logit_hairpin_minus_logit_nonhairpin',
        intervention='original_margin_minus_locally_straightened_margin',
        display='exact_original_windows_without_pointwise_averaging',default_patch_order='positive_delta_descending',
        model_retrained=False,source_arrays_modified=False,browser_checked=False)
    dump(saliency/'diagnostics.json',report)
    generated=[output/'index.html',*features.rglob('*'),*saliency.rglob('*')]
    files={p.relative_to(output).as_posix():sha(p) for p in generated if p.is_file()}
    dump(output/'build_manifest.json',dict(version=payload['version'],files=files,source_report='saliency/diagnostics.json',
        code={str(p):sha(p) for p in (Path(__file__),Path('FMT_Utils/Task4C_SupportView_1_2.py'),templates/'task4c_support_1_2.html',templates/'fmt_feature_diagram_1_2.js')}))
    if update_entry:
        legacy=old/'index_1.1.html'
        if not legacy.exists(): shutil.copy2(old/'index.html',legacy)
        import os
        url=Path(os.path.relpath(output/'index.html',old)).as_posix()
        (old/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>FMT analysis 1.2</title><script>location.replace('+js(url)+'+location.hash);</script><a href="'+url+'">打开更新后的分析工具1.2</a>',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('bundle_diagnostics',)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',default='outputs/Other_FMT_AnalysisWorkbench_1.1');p.add_argument('--output',default='outputs/Other_FMT_AnalysisWorkbench_1.2');p.add_argument('--source',default='outputs/Other_Task4C_Saliency_1.1/package');p.add_argument('--update-entry',action='store_true');a=p.parse_args();build(a.previous,a.output,a.source,a.update_entry)
