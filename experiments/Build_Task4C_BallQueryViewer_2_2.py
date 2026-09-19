"""Build the adjustable radius-count viewer from exact, read-only count statistics."""
import argparse
import json
from pathlib import Path


def build(stats, output, inline=None):
    data = json.loads(Path(stats).read_text())
    # Normalize keys so JavaScript String(1.0) -> '1' finds the exact recorded counts.
    for flow in data['flows'].values():
        flow['fixed_radius_counts'] = {format(float(k), 'g'): v for k, v in flow['fixed_radius_counts'].items()}
    source = Path('experiments/templates/task4c_ball_query_counts_2_2.html').read_text(encoding='utf8')
    fragment = source.replace('__DATA__', json.dumps(data, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/'))
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out/'fragment.html').write_text(fragment, encoding='utf8')
    if inline: Path(inline).write_text(fragment, encoding='utf8')
    wrapper = '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task4-c Ball query 数量分布</title>
<style>:root{color-scheme:light dark;--foreground:light-dark(#20342e,#e5eee9);--background:light-dark(#fff,#17201c);--border:light-dark(#ccd6d0,#45534b);--viz-series-1:light-dark(#25735e,#78c5a8)}body{font:16px system-ui;color:var(--foreground);background:var(--background);max-width:1120px;margin:32px auto;padding:0 18px}h2{font-weight:500}.viz-controls{display:flex;flex-wrap:wrap;gap:18px;margin-bottom:22px}.form-label{display:block}select{display:block;font:inherit;padding:7px;min-width:160px}input[type=range]{width:100%;margin:14px 0}.table-responsive{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{text-align:right;padding:10px 8px;border-bottom:1px solid var(--border);white-space:nowrap}th{font-weight:500}.text-small{font-size:14px}.tabular-nums{font-variant-numeric:tabular-nums}a{color:var(--foreground)}</style></head><body>'''
    footer = '<p><a href="../../Other_FMT_AnalysisWorkbench_1.6/index.html#dataset">返回数据集 v2 工作台</a></p></body></html>'
    (out/'index.html').write_text(wrapper+fragment+footer, encoding='utf8')
    print(str((out/'index.html').resolve()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--statistics', required=True); p.add_argument('--output', required=True); p.add_argument('--inline')
    a = p.parse_args(); build(a.statistics, a.output, a.inline)
