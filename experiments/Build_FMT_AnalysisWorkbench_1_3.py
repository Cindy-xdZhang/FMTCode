"""Clarify viewer terminology while preserving every frozen analysis array."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(previous, output, update_entry=False):
    previous, output = Path(previous), Path(output)
    if previous.resolve() == output.resolve():
        raise ValueError('Keep the previous viewer in its own directory')
    templates = Path(__file__).parent / 'templates'
    inherited = {}
    for folder in ('features', 'saliency'):
        shutil.copytree(previous / folder, output / folder, dirs_exist_ok=True)
        for p in (previous / folder).rglob('*'):
            if p.is_file() and p.relative_to(previous).as_posix() != 'saliency/index.html':
                rel = p.relative_to(previous).as_posix()
                inherited[rel] = sha(p)
                assert sha(output / rel) == inherited[rel], rel
    shutil.copy2(templates / 'task4c_support_1_3.html', output / 'saliency/index.html')
    shutil.copy2(templates / 'task4c_color_help_1_3.js', output / 'saliency/color_help.js')
    entry = (templates / 'fmt_analysis_workbench.html').read_text(encoding='utf-8')
    entry = entry.replace('__SALIENCY_URL__', 'saliency/index.html').replace('Hairpin 形状显著性', 'Hairpin 形状解释')
    (output / 'index.html').write_text(entry, encoding='utf-8')
    report = dict(version='Other_FMT_AnalysisWorkbench_1.3', source_version='Other_FMT_AnalysisWorkbench_1.2',
        inherited_file_hashes=inherited, unchanged_inherited_files=len(inherited), scientific_data_changed=False,
        scope='Mode labels, visible explanations, and hiding irrelevant intervention controls and metrics',
        ui_hashes={rel: sha(output / rel) for rel in ('index.html', 'saliency/index.html', 'saliency/color_help.js')},
        code_hashes={str(p): sha(p) for p in (Path(__file__), templates / 'task4c_support_1_3.html', templates / 'task4c_color_help_1_3.js')})
    (output / 'build_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if update_entry:
        archive = previous / 'index_1.2.html'
        if not archive.exists():
            shutil.copy2(previous / 'index.html', archive)
        url = Path(os.path.relpath(output / 'index.html', previous)).as_posix()
        (previous / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>FMT analysis 1.3</title><script>location.replace('+json.dumps(url)+'+location.hash);</script><a href="'+url+'">打开分析工具1.3</a>', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('hashes')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--previous', default='outputs/Other_FMT_AnalysisWorkbench_1.2')
    p.add_argument('--output', default='outputs/Other_FMT_AnalysisWorkbench_1.3')
    p.add_argument('--update-entry', action='store_true')
    a = p.parse_args()
    build(a.previous, a.output, a.update_entry)
