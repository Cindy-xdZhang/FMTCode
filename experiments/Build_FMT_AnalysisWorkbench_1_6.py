"""Workbench 1.6: the three pages of 1.5 plus a fourth page with the dataset v2 sample viewer."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''): h.update(part)
    return h.hexdigest()


def build(previous, results, dataset_viewer, output, update_entry=False):
    previous, results, dataset_viewer, output = map(Path, (previous, results, dataset_viewer, output))
    assert json.loads((dataset_viewer/'viewer_manifest.json').read_text())['html_sha256'] == sha(dataset_viewer/'index.html')
    inherited = {}
    for folder in ('features', 'saliency'):
        shutil.copytree(previous/folder, output/folder, dirs_exist_ok=True)
        for file in (previous/folder).rglob('*'):
            if file.is_file():
                rel = file.relative_to(previous).as_posix(); inherited[rel] = sha(file); assert sha(output/rel) == inherited[rel]
    source_index = previous/'index_1.5.html' if (previous/'index_1.5.html').exists() else previous/'index.html'   # the entry may already redirect here
    html = source_index.read_text(encoding='utf8')
    assert '__DATASET_URL__' not in html and 'nav-results' in html
    html = html.replace('<a href="#results" id="nav-results">Hairpin 分类结果</a>', '<a href="#results" id="nav-results">Hairpin 分类结果</a><a href="#dataset" id="nav-dataset">数据集 v2 样本</a>')
    note = '<section id="datasetPanel" hidden><div class="source-note"><strong style="font-size:13px">数据集 v2 · mainExp_Task4C_FixedDataset_2.1 r3</strong>　训练折 1–4 与测试折 0 的全部样本点（Channel 148,746 / 40,670，TBL 157,182 / 41,231），每点三条涡线；按归属实例查看划归它的样本；着色按标签或 FMT 2.1 测试折预测。</div><iframe id="dataset" title="数据集 v2 样本点" data-src="__DATASET_URL__" hidden></iframe></section>'
    anchor = '</section>\n</div>'; assert html.count(anchor) == 1; html = html.replace(anchor, '</section>\n'+note+'\n</div>')
    # Same full-height flex layout as the results page; otherwise the iframe keeps its default height.
    assert html.count('</style>') == 1
    html = html.replace('</style>', '#datasetPanel{height:100%;display:flex;flex-direction:column}#datasetPanel iframe{flex:1;min-height:0}#datasetPanel[hidden]{display:none}</style>')
    html = html.replace("['features','saliency','results'].includes(key)", "['features','saliency','results','dataset'].includes(key)")
    html = html.replace("document.getElementById('resultsPanel').hidden=active!=='results';", "document.getElementById('resultsPanel').hidden=active!=='results';document.getElementById('datasetPanel').hidden=active!=='dataset';")
    html = html.replace("for(const id of ['features','saliency','results'])", "for(const id of ['features','saliency','results','dataset'])")
    assert html.count("'dataset'") >= 3
    # Results page keeps its 1.5 link; the dataset page links to the new viewer.
    html = html.replace('__DATASET_URL__', Path(os.path.relpath(dataset_viewer/'index.html', output)).as_posix())
    results_link = Path(os.path.relpath(results/'index.html', output)).as_posix(); old_link = Path(os.path.relpath(results/'index.html', previous)).as_posix()
    html = html.replace('data-src="'+old_link+'"', 'data-src="'+results_link+'"'); assert results_link in html
    (output/'index.html').write_text(html, encoding='utf8')
    manifest = dict(version='Other_FMT_AnalysisWorkbench_1.6', previous_version='Other_FMT_AnalysisWorkbench_1.5', inherited_file_hashes=inherited,
                    results_viewer_index_sha256=sha(results/'index.html'), dataset_viewer_index_sha256=sha(dataset_viewer/'index.html'), index_sha256=sha(output/'index.html'))
    (output/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    if update_entry:
        archive = previous/'index_1.5.html'
        if not archive.exists(): shutil.copyfile(previous/'index.html', archive)
        url = Path(os.path.relpath(output/'index.html', previous)).as_posix()
        (previous/'index.html').write_text('<!doctype html><meta charset="utf-8"><script>location.replace('+json.dumps(url)+'+location.hash)</script>', encoding='utf8')
    print(json.dumps(dict(version=manifest['version'], inherited_files=len(inherited), index=str(output/'index.html'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--previous', default='outputs/Other_FMT_AnalysisWorkbench_1.5')
    p.add_argument('--results', default='outputs/Other_Task4C_GTHeadCoverage_1.1/viewer'); p.add_argument('--dataset-viewer', default='outputs/Other_Task4C_DatasetViewer_2.1/viewer')
    p.add_argument('--output', default='outputs/Other_FMT_AnalysisWorkbench_1.6'); p.add_argument('--update-entry', action='store_true')
    a = p.parse_args(); build(a.previous, a.results, a.dataset_viewer, a.output, a.update_entry)
