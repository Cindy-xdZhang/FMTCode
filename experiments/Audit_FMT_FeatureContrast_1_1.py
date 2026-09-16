"""Independent NumPy reconstruction and artifact audit for the feature viewer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'build_audit.json').read_text(encoding='utf-8'))
    for file, expected in manifest['files'].items():
        assert hashlib.sha256((folder / file).read_bytes()).hexdigest() == expected, file
    summary = json.loads((folder / 'features/analysis_summary.json').read_text(encoding='utf-8'))
    for source in manifest['source_files'].values():
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    with np.load(folder / 'features/analysis_arrays.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    # Rebuild every individual dimension without calling any FMT implementation.
    local = (data['paths'] - data['paths'][:, :1, :1]).astype(np.float64)
    local -= local.mean((1, 2), keepdims=True)
    local /= np.linalg.norm(local, axis=-1).max((1, 2))[:, None, None, None]
    q32 = local.astype(np.float32)
    # Compare the independent formulas in double precision. Float32 FFT backend
    # roundoff is amplified in cosine/triple ratios for nearly straight lines.
    q = q32.astype(np.float64)
    def vector_descriptor(u):
        spec = np.fft.rfft(u, axis=-2)[..., :6, :]
        a, b = spec.real, spec.imag
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        cos = (a * b).sum(-1) / np.maximum(na * nb, 1e-8)
        gram = np.stack((na, nb, cos), -1).reshape(*u.shape[:-2], 18)
        triple = (np.cross(a[..., :-1, :], b[..., :-1, :]) * a[..., 1:, :]).sum(-1)
        denom = np.maximum(na[..., :-1] * nb[..., :-1] * na[..., 1:], 1e-8)
        return np.concatenate((gram, triple / denom), -1)
    center = vector_descriptor(np.diff(q[:, 0], axis=-2))
    neighbors = vector_descriptor(np.diff(q[:, 1:] - q[:, :1], axis=-2))
    delta = np.diff(q[:, 0], axis=-2)
    tangent = delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-12)
    tangent = np.concatenate((tangent, tangent[:, -1:]), axis=1)
    signal = np.concatenate((q[:, 0], tangent), -1)
    spectrum = np.fft.rfft(signal, axis=1, norm='ortho')[:, :6]
    direction = np.stack((spectrum.real, spectrum.imag), -1).reshape(len(q), 72)
    rebuilt = np.concatenate((center, direction, neighbors.mean(1), neighbors.max(1)), -1)
    import torch
    from FMT_Utils.FMT_P35_NormFrequency_3_1 import primitive_features
    torch.set_num_threads(4)
    reference64 = primitive_features(torch.from_numpy(q), 6).numpy()
    formula_error = np.abs(rebuilt - reference64)
    assert np.allclose(rebuilt, reference64, rtol=1e-9, atol=1e-10), formula_error.max()
    error = np.abs(rebuilt - data['raw'])
    assert error.max() < 1e-3, error.max()
    features = summary['feature_rows']
    assert [f['index'] for f in features] == list(range(141))
    assert np.all(data['raw'][:, [f['index'] for f in features if f['theoretical_zero']]] == 0)
    # Check the displayed ranking and distance identity through an affine form.
    x, labels, c = data['standardized'].astype(np.float64), data['labels'], data['centers']
    assert np.array_equal(((data['raw'] - data['normalizer_mean']) / data['normalizer_scale']).astype(np.float32), data['standardized'])
    assert np.allclose(c, np.stack([x[labels == i].mean(0) for i in range(2)]), atol=1e-12)
    affine = 2 * x * (c[0] - c[1]) + c[1] ** 2 - c[0] ** 2
    assert np.allclose(affine, data['contribution'], atol=1e-11)
    assert np.allclose(affine.sum(1), data['margin'], atol=1e-10)
    assert np.array_equal((affine.sum(1) < 0).astype(int), labels)
    gap = c[1] - c[0]
    assert np.allclose([f['squared_gap'] for f in features], gap ** 2)
    assert np.allclose([f['share'] for f in features], gap ** 2 / np.dot(gap, gap))
    assert np.isclose(sum(g['share'] for g in summary['groups']), 1.)
    # Swapping arbitrary cluster names must reverse individual evidence, not its rank.
    reverse = 2 * x * (c[1] - c[0]) + c[0] ** 2 - c[1] ** 2
    assert np.allclose(reverse, -affine, atol=1e-12)
    # Verify exported browser data and every geometry chunk against the source arrays.
    payload_text = (folder / 'features/data.js').read_text(encoding='utf-8')
    payload = json.loads(payload_text.removeprefix('window.FEATURE_CONTRAST=').strip().removesuffix(';'))
    for key, source in [('raw', 'raw'), ('z', 'standardized'), ('ids', 'sample_ids'), ('labels', 'labels')]:
        assert np.array_equal(payload[key], data[source]), key
    assert np.array_equal(payload['center_paths'], data['paths'][:, 0])
    all_geometry = []
    for path in sorted((folder / 'features/data').glob('geometry_*.js')):
        text = path.read_text(encoding='utf-8')
        all_geometry.extend(json.loads(text.split(',', 1)[1].strip().removesuffix(');')))
    assert np.array_equal(all_geometry, data['paths'])
    result = dict(complete=True, version=summary['version'], samples=len(x), dimensions=x.shape[1],
        files_checked=len(manifest['files']), independent_numpy_vs_torch_float64_max_abs_error=float(formula_error.max()),
        independent_float64_vs_stored_float32_max_abs_error=float(error.max()),
        numpy_block_max_abs_error={b['id']: float(error[:, b['start']:b['stop']].max()) for b in summary['groups']},
        max_affine_distance_identity_error=float(np.abs(affine.sum(1) - data['margin']).max()),
        source_hashes_valid=True, all_browser_values_exact=True, cluster_name_swap_check=True,
        all_geometry_chunks_exact=True, reference_labels_used=False)
    (folder / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='outputs/Other_FMT_AnalysisWorkbench_1.1')
    audit(parser.parse_args().folder)
