"""Compare both flows with the user's fixed per-flow h; expand histogram bins for large balls."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.Analyze_Task4C_BallQuery_2_2 import sha
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as baseline


def build(config_path, inline=None):
    config = json.loads(Path(config_path).read_text(encoding='utf8')); root = Path(config['output'])
    stats_path = root/'radius_statistics.json'; report = json.loads(stats_path.read_text())
    assert report['complete'] and report['config_sha256'] == sha(config_path)
    spec = json.loads(Path(config['source_config']).read_text())
    data = {k: report[k] for k in ('version', 'radii_h', 'grids', 'h_mode', 'flows')}
    bins = np.array([0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024,
                     2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, np.inf])
    labels = [f'{int(a):,}' if b == a+1 else f'{int(a):,}–{int(b-1):,}' if np.isfinite(b) else f'≥{int(a):,}' for a, b in zip(bins[:-1], bins[1:])]
    for fi, flow in enumerate(spec['flows']):
        name = flow['name']; entry = data['flows'][name]; count_file = root/f'counts_{name}.npz'
        assert sha(count_file) == entry['count_file_sha256']
        folder = Path(spec['source_output'])/'physical'/name
        assert sha(folder/'metadata.npz') == entry['frozen_files']['metadata.npz']
        with np.load(folder/'metadata.npz') as z: meta = {k: z[k] for k in ('fold', 'label')}
        masks = dict(all=np.ones(len(meta['fold']), bool), **baseline.split_masks(meta, fi, spec))
        with np.load(count_file) as z:
            assert np.array_equal(z['radius_h'], data['radii_h'])
            counts = z['counts']
        for ri, radius in enumerate(data['radii_h']):
            for role, mask in masks.items():
                for label, lm in (('all', np.ones(len(mask), bool)), ('hairpin', meta['label'] == 1), ('non_hairpin', meta['label'] == 0)):
                    row = entry['fixed_radius_counts'][str(float(radius))][role][label]
                    histogram, _ = np.histogram(counts[ri, mask & lm], bins)
                    assert histogram.sum() == row['n']
                    row.update(histogram=histogram.tolist(), bin_labels=labels)
        entry['fixed_radius_counts'] = {format(float(k), 'g'): v for k, v in entry['fixed_radius_counts'].items()}
    template = Path('experiments/templates/task4c_ball_query_counts_2_3.html').read_text(encoding='utf8')
    fragment = template.replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
    assert len(fragment.encode('utf8')) < 1_000_000 and '__DATA__' not in fragment
    out = root/'viewer'; out.mkdir(parents=True, exist_ok=True)
    (out/'fragment.html').write_text(fragment, encoding='utf8')
    if inline: Path(inline).write_text(fragment, encoding='utf8')
    wrapper = '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task4-c Ball query · Channel Δy / TBL Δz</title>
<style>:root{color-scheme:light dark;--foreground:light-dark(#20342e,#e5eee9);--background:light-dark(#fff,#17201c);--border:light-dark(#ccd6d0,#45534b);--viz-series-1:light-dark(#25735e,#78c5a8);--viz-series-2:light-dark(#866331,#dfbd83)}body{font:16px system-ui;color:var(--foreground);background:var(--background);max-width:1120px;margin:24px auto;padding:0 18px}h2,h3{font-weight:500}.viz-controls{display:flex;flex-wrap:wrap;gap:18px;margin-bottom:20px}.form-label{display:block}select{display:block;font:inherit;padding:7px;min-width:150px}.table-responsive{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{text-align:right;padding:10px 8px;border-bottom:1px solid var(--border);white-space:nowrap}th{font-weight:500}th:first-child,td:first-child{text-align:left}.text-small{font-size:14px}.tabular-nums{font-variant-numeric:tabular-nums}a{color:var(--foreground)}</style></head><body>'''
    footer = '<p><a href="../../Verify_Task4C_BallQueryCounts_2.2/viewer/index.html">旧 h 分布（归档）</a></p></body></html>'
    (out/'index.html').write_text(wrapper+fragment+footer, encoding='utf8')
    (out/'manifest.json').write_text(json.dumps(dict(statistics_sha256=sha(stats_path), config_sha256=sha(config_path),
        template_sha256=sha('experiments/templates/task4c_ball_query_counts_2_3.html'), fragment_sha256=sha(out/'fragment.html'),
        fragment_bytes=len(fragment.encode('utf8')), histogram_bins=bins[:-1].tolist()), indent=2)+'\n')
    print(str((out/'index.html').resolve()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/Verify_Task4C_BallQueryCounts_2.3.json')
    parser.add_argument('--inline')
    args = parser.parse_args(); build(args.config, args.inline)
