"""Build a local, offline two-page workbench from existing scientific arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from FMT_Utils.FeatureContrast_1_1 import BLOCKS, explain_partition, p35_schema


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def js(value):
    return json.dumps(plain(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')


def build(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    source = Path(cfg['feature_bundle']).resolve()
    output = Path(cfg['output']).resolve()
    page = output / 'features'
    (page / 'data').mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as z:
        raw, values = z['feature_raw'], z['feature_vectors']
        metadata = json.loads(str(z['metadata_json']))
    info = metadata['feature_analysis']
    prep = info['preprocessing']
    if info['method'] != 'p35_n0_k06' or info['mode'] != 'direct' or prep['normalization'] != 'standard' or prep['pca_effective'] != 0:
        raise ValueError('This version requires direct, standardized p35/n0_k06 features without PCA preprocessing')
    geometry_path = (source.parent / metadata['base_bundle']).resolve()
    if sha(geometry_path) != metadata['base_sha256']:
        raise ValueError('Source geometry hash changed')
    with np.load(geometry_path, allow_pickle=False) as z:
        paths, ids = z['paths'], z['sample_ids']
        geometry_info = json.loads(str(z['metadata_json']))
    if paths.shape != (len(raw), 7, 32, 3) or raw.shape != values.shape or raw.shape[1] != 141:
        raise ValueError('Invalid geometry/feature shape')
    if len(set(ids.tolist())) != len(ids) or len(ids) != len(raw):
        raise ValueError('Invalid sample IDs')
    if 'cylinder' in geometry_info['dataset'].lower() and geometry_info['start_time'] < 7.5:
        raise ValueError('Cylinder start time violates the frozen original-time policy')

    # Verify the source extractor using this repository's frozen implementation.
    import torch
    from FMT_Utils.FMT_P35_NormFrequency_3_1 import encode
    torch.set_num_threads(4)
    local = np.ascontiguousarray(paths - paths[:, :1, :1])
    reconstructed = encode(torch.from_numpy(local), dict(geometry='max_radius', frequencies=6)).numpy()
    encoding_error = float(np.abs(reconstructed - raw).max())
    if not np.allclose(reconstructed, raw, rtol=2e-5, atol=2e-6):
        raise ValueError(f'Frozen p35 encoding mismatch: {encoding_error}')
    mean, scale = np.asarray(prep['mean']), np.asarray(prep['scale'])
    computed_scale = raw.std(0, dtype=np.float64)
    computed_scale[computed_scale < 1e-8] = 1.
    if prep['fit_count'] != len(raw) or not np.allclose(mean, raw.mean(0, dtype=np.float64), atol=1e-12) or not np.allclose(scale, computed_scale, atol=1e-12):
        raise ValueError('Source normalization population differs')
    rebuilt = ((raw - mean) / scale).astype(np.float32)
    if not np.array_equal(rebuilt, values):
        raise ValueError('Stored standardized feature values differ')
    x = values.astype(np.float64)
    with threadpool_limits(limits=4):
        km = KMeans(n_clusters=2, n_init=cfg['n_init'], random_state=cfg['seed'],
                    max_iter=cfg['max_iter'], tol=cfg['tol'], algorithm='lloyd').fit(x)
        order = sorted(range(2), key=lambda c: int(ids[km.labels_ == c].min()))
        label_map = np.argsort(order)
        labels, centers = label_map[km.labels_], km.cluster_centers_[order]
        projection = PCA(n_components=2, svd_solver='full').fit_transform(x)
    result = explain_partition(raw, x, labels, centers)
    schema = p35_schema()
    if not np.all(raw[:, [f['index'] for f in schema if f['theoretical_zero']]] == 0):
        raise ValueError('Theoretical zero slots differ from the source encoder')
    rows = []
    for j, feature in enumerate(schema):
        rows.append(dict(feature, mean_A=result['mean'][0, j], mean_B=result['mean'][1, j],
            raw_mean_A=result['raw_mean'][0, j], raw_mean_B=result['raw_mean'][1, j],
            std_A=result['std'][0, j], std_B=result['std'][1, j],
            raw_std_A=result['raw_std'][0, j], raw_std_B=result['raw_std'][1, j],
            delta=result['delta'][j], squared_gap=result['square'][j], share=result['share'][j],
            constant=bool(np.ptp(raw[:, j]) == 0)))
    groups = []
    for b in BLOCKS:
        sl = slice(b['start'], b['stop'])
        groups.append(dict(b, square_sum=result['square'][sl].sum(),
            square_per_dimension=result['square'][sl].mean(), share=result['share'][sl].sum()))
    chunk_size = cfg['geometry_chunk_size']
    for start in range(0, len(paths), chunk_size):
        chunk = start // chunk_size
        (page / 'data' / f'geometry_{chunk:03d}.js').write_text(
            f'window.registerGeometry({chunk},' + js(paths[start:start + chunk_size]) + ');\n', encoding='utf-8')
    stats = {k: v for k, v in result.items() if k not in ('contribution', 'distance')}
    # Preserve complete source arrays, including raw values, in the downloadable package.
    np.savez_compressed(page / 'analysis_arrays.npz', raw=raw, standardized=values, sample_ids=ids,
        labels=labels, centers=centers, paths=paths, contribution=result['contribution'],
        margin=result['margin'], normalizer_mean=mean, normalizer_scale=scale)
    physical_match = re.search(r'Re\d+', geometry_info.get('display_metadata', {}).get('source_path', ''))
    summary = dict(version=cfg['version'], dataset=geometry_info['dataset'],
        physical_variant=physical_match.group(0) if physical_match else '', samples=len(raw), dimensions=141,
        time=[geometry_info['start_time'], geometry_info['end_time']],
        counts=result['counts'], method='p35/n0_k06', cluster_space='full standardized 141D',
        fit_population='all valid primitives; no display subsampling for fitting',
        settings={k: cfg[k] for k in ('seed', 'n_init', 'max_iter', 'tol')},
        iterations=km.n_iter_, feature_rows=rows, groups=groups,
        center_distance_squared=float(result['square'].sum()),
        source_files=dict(feature_bundle=dict(path=str(source), sha256=sha(source)),
                          geometry_bundle=dict(path=str(geometry_path), sha256=sha(geometry_path))),
        checks=dict(encoder_max_abs_error=encoding_error, normalization_exact=True,
                    all_labels_nearest_center=True, all_centers_equal_cluster_means=True,
                    sample_contribution_max_abs_error=result['decomposition_error'],
                    theoretical_zero_dimensions=sum(f['theoretical_zero'] for f in schema)),
        environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, torch=torch.__version__),
        code_sha256={str(p): sha(p) for p in [Path(__file__), Path('FMT_Utils/FeatureContrast_1_1.py'),
            Path('FMT_Utils/FMT_P35_NormFrequency_3_1.py'), Path('FMT_Utils/DFT_FMT_3D.py'), Path(config_path)]},
        interpretation='descriptive fixed-cluster distance decomposition, not causal importance or a quality metric',
        labels_or_ivd_used=False, source_metadata=metadata, geometry_metadata=geometry_info)
    dump(page / 'analysis_summary.json', summary)
    with (page / 'feature_ranking.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['index', 'name', 'block_name', 'frequency', 'formula', 'raw_mean_A', 'raw_mean_B', 'mean_A', 'mean_B', 'delta', 'squared_gap', 'share', 'constant']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows([plain(rows[j]) for j in result['ranking']])
    payload = dict(summary={k: v for k, v in summary.items() if k not in ('source_metadata', 'geometry_metadata', 'feature_rows')},
        features=rows, groups=groups, ids=ids, labels=labels, centers=centers,
        z=values, raw=raw, projection=projection, center_paths=paths[:, 0], chunk_size=chunk_size, stats=stats)
    (page / 'data.js').write_text('window.FEATURE_CONTRAST=' + js(payload) + ';\n', encoding='utf-8')
    from plotly.offline import get_plotlyjs
    (page / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    templates = Path(__file__).parent / 'templates'
    (page / 'index.html').write_text((templates / 'fmt_feature_contrast.html').read_text(encoding='utf-8'), encoding='utf-8')
    saliency = Path(cfg['saliency_viewer']).resolve()
    if not saliency.is_file(): raise FileNotFoundError(saliency)
    relative = Path(os.path.relpath(saliency, output)).as_posix()
    (output / 'index.html').write_text((templates / 'fmt_analysis_workbench.html').read_text(encoding='utf-8').replace('__SALIENCY_URL__', relative), encoding='utf-8')
    artifact_paths = [output / 'index.html'] + [p for p in page.rglob('*') if p.is_file()]
    audit = dict(version=cfg['version'], numerical_checks=summary['checks'],
        cluster_counts=result['counts'], source_files=summary['source_files'],
        browser_interaction_verified=False, saliency_source=dict(path=str(saliency), sha256=sha(saliency)),
        files={p.relative_to(output).as_posix(): sha(p) for p in artifact_paths})
    dump(output / 'build_audit.json', audit)
    print(json.dumps(plain(dict(samples=len(raw), counts=result['counts'], checks=summary['checks'],
        top_features=[rows[j] for j in result['ranking'][:5]], output=str(output))), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Other_FMT_FeatureContrast_1.1.json')
    args = parser.parse_args()
    build(args.config)
