"""Add complete-curve intervention colors and embed the existing result viewer."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path, folder):
    return Path(os.path.relpath(path,folder)).as_posix()


def build(previous,output,results,update_entry=False):
    previous,output,results=Path(previous),Path(output),Path(results)
    if previous.resolve()==output.resolve():
        raise ValueError('Keep prior scientific and viewer artifacts separate')
    if not (results/'index.html').exists() or not (results/'geometry').is_dir():
        raise FileNotFoundError('The existing classification viewer and geometry folder are required')
    templates=Path(__file__).parent/'templates'
    inherited={}
    replaced={'saliency/index.html','saliency/color_help.js'}
    for folder in ('features','saliency'):
        shutil.copytree(previous/folder,output/folder,dirs_exist_ok=True)
        for p in (previous/folder).rglob('*'):
            rel=p.relative_to(previous).as_posix()
            if p.is_file() and rel not in replaced:
                inherited[rel]=sha(p)
                assert sha(output/rel)==inherited[rel],rel
    for source,target in [('task4c_support_1_4.html','index.html'),('task4c_color_help_1_4.js','color_help.js'),('task4c_all_segments_1_4.js','all_segments.js')]:
        shutil.copy2(templates/source,output/'saliency'/target)
    entry=(templates/'fmt_analysis_workbench_1_4.html').read_text(encoding='utf-8')
    (output/'index.html').write_text(entry.replace('__RESULTS_URL__',relative(results/'index.html',output)),encoding='utf-8')
    dependencies={relative(p,output):sha(p) for p in [results/'index.html',results/'viewer_manifest.json',*sorted((results/'geometry').glob('*.js'))]}
    report=dict(version='Other_FMT_AnalysisWorkbench_1.4',source_version='Other_FMT_AnalysisWorkbench_1.3',
        inherited_file_hashes=inherited,unchanged_inherited_files=len(inherited),scientific_data_changed=False,
        display_windows=[[0,8],[8,16],[16,24],[24,31]],color_scale='symmetric max absolute score across displayed bundles; no clipping',
        result_viewer_dependencies=dependencies,result_viewer_path=str(results),
        ui_hashes={p:sha(output/p) for p in ('index.html','saliency/index.html','saliency/color_help.js','saliency/all_segments.js')})
    (output/'build_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if update_entry:
        archive=previous/'index_1.3.html'
        if not archive.exists():shutil.copy2(previous/'index.html',archive)
        url=relative(output/'index.html',previous)
        (previous/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>FMT analysis 1.4</title><script>location.replace('+json.dumps(url)+'+location.hash);</script><a href="'+url+'">打开分析工具1.4</a>',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if 'hashes' not in k and k!='result_viewer_dependencies'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--previous',default='outputs/Other_FMT_AnalysisWorkbench_1.3')
    p.add_argument('--output',default='outputs/Other_FMT_AnalysisWorkbench_1.4')
    p.add_argument('--results',default='outputs/Other_Task4C_BundleVisualization_1.3/viewer')
    p.add_argument('--update-entry',action='store_true')
    a=p.parse_args();build(a.previous,a.output,a.results,a.update_entry)
