"""Integrate the audited, complete-GT c156 results into the three-page workbench."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def build(previous,results,output,update_entry=False):
    previous,results,output=map(Path,(previous,results,output))
    audit=json.loads((results/'head_coverage_manifest.json').read_text())
    assert audit['complete'] and audit['html_sha256']==sha(results/'index.html')
    assert all(v>0 for flow in audit['coverage'].values() for role in flow.values() for v in role.values())
    inherited={}
    for folder in ('features','saliency'):
        shutil.copytree(previous/folder,output/folder,dirs_exist_ok=True)
        for file in (previous/folder).rglob('*'):
            if file.is_file():
                rel=file.relative_to(previous).as_posix();inherited[rel]=sha(file)
                assert sha(output/rel)==inherited[rel]
    html=Path('experiments/templates/fmt_analysis_workbench_1_4.html').read_text(encoding='utf8')
    link=Path(os.path.relpath(results/'index.html',output)).as_posix()
    html=html.replace('__RESULTS_URL__',link)
    start=html.index('<div class="source-note">');end=html.index('</div>',start)+6
    note='<div class="source-note"><strong style="font-size:13px">全部GT头部覆盖 · 固定c156</strong>　132个GT实例；每个追加10束头部测试＋30束邻近训练。测试视图显示全部11,320束；训练视图只展示新增3,960束（完整训练196,960束）。seed96721，992,386参数；新增束按人工GT头部归属标正类。形状解释页仍对应原p35模型。</div>'
    html=html[:start]+note+html[end:]
    (output/'index.html').write_text(html,encoding='utf8')
    dependencies={Path(os.path.relpath(p,output)).as_posix():sha(p) for p in results.rglob('*') if p.is_file()}
    manifest=dict(version='Other_FMT_AnalysisWorkbench_1.5',previous_version='Other_FMT_AnalysisWorkbench_1.4',
        inherited_file_hashes=inherited,result_dependencies=dependencies,index_sha256=sha(output/'index.html'),
        all_GT_heads_covered=True,saliency_scientific_data_changed=False)
    (output/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    if update_entry:
        archive=previous/'index_1.4.html'
        if not archive.exists():shutil.copyfile(previous/'index.html',archive)
        url=Path(os.path.relpath(output/'index.html',previous)).as_posix()
        (previous/'index.html').write_text('<!doctype html><meta charset="utf-8"><script>location.replace('+json.dumps(url)+'+location.hash)</script>',encoding='utf8')
    print(json.dumps(dict(version=manifest['version'],inherited_files=len(inherited),result_files=len(dependencies),index=str(output/'index.html'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',default='outputs/Other_FMT_AnalysisWorkbench_1.4')
    p.add_argument('--results',default='outputs/Other_Task4C_GTHeadCoverage_1.1/viewer')
    p.add_argument('--output',default='outputs/Other_FMT_AnalysisWorkbench_1.5');p.add_argument('--update-entry',action='store_true')
    a=p.parse_args();build(a.previous,a.results,a.output,a.update_entry)
