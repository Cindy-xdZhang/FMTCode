"""Test objective relative vectors and their induced Fourier transformation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from FMT_Utils import objective_fmt_nTDO as module


def maximum(a, b):
    return float(torch.max(torch.abs(a-b)))


def main():
    torch.set_num_threads(2)
    out = Path('outputs/Verify_RelativeFourierObjectivity_1.1')
    out.mkdir(parents=True, exist_ok=False)
    length = 32
    offsets = torch.cat((torch.zeros(1, 3), torch.eye(3), -torch.eye(3))).double()
    x = offsets[None, :, None, :].expand(1, 7, length, 3).clone()
    angle = 2*torch.pi*torch.arange(length, dtype=torch.float64)/length
    q = torch.zeros(length, 3, 3, dtype=torch.float64)
    q[:, 0, 0] = q[:, 1, 1] = angle.cos()
    q[:, 0, 1] = -angle.sin(); q[:, 1, 0] = angle.sin(); q[:, 2, 2] = 1.
    eye = torch.eye(3, dtype=torch.float64).expand(length, 3, 3)
    torch.testing.assert_close(q @ q.transpose(-1, -2), eye, atol=1e-14, rtol=1e-14)
    y = torch.einsum('tij,nktj->nkti', q, x)
    a, b = module.build_fourier_inputs(x), module.build_fourier_inputs(y)
    expected_relative = torch.einsum('tij,nktj->nkti', q, a['relative'])
    torch.testing.assert_close(b['relative'], expected_relative, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(a['distances'], b['distances'], atol=1e-14, rtol=1e-14)
    f = torch.fft.rfft(a['relative'], dim=2)
    transformed_f = torch.fft.rfft(b['relative'], dim=2)
    # Full Fourier representation is reversible and has an induced observer law.
    reconstructed = torch.fft.irfft(f, n=length, dim=2)
    induced = torch.fft.rfft(torch.einsum('tij,nktj->nkti', q, reconstructed), dim=2)
    torch.testing.assert_close(transformed_f, induced, atol=1e-13, rtol=1e-13)
    expected_first = torch.tensor([16.+0j, -16j, 0j], dtype=torch.complex128)
    torch.testing.assert_close(transformed_f[0, 0, 1], expected_first, atol=1e-13, rtol=1e-13)
    assert float(torch.linalg.vector_norm(f[0, 0, 1])) == 0.
    assert float(torch.linalg.vector_norm(transformed_f[0, 0, 1])) > 22.
    # Independent nonzero-center, time-dependent translation, multi-primitive check.
    rng = np.random.default_rng(11092026)
    random_x = torch.from_numpy(rng.normal(size=(256, 7, length, 3)))
    c = torch.from_numpy(rng.normal(size=(1, 1, length, 3)))
    random_y = torch.einsum('tij,nktj->nkti', q, random_x) + c
    ra, rb = module.build_fourier_inputs(random_x), module.build_fourier_inputs(random_y)
    expected = torch.einsum('tij,nktj->nkti', q, ra['relative'])
    torch.testing.assert_close(rb['relative'], expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(ra['distances'], rb['distances'], atol=1e-13, rtol=1e-13)
    result = {
        'experiment': 'Verify_RelativeFourierObjectivity_1.1', 'execution_status': 'PASS',
        'relative_vector_objectivity_test': 'PASS', 'distance_objectivity_test': 'PASS',
        'full_fourier_induced_observer_law_test': 'PASS',
        'per_frequency_norm_invariance': 'FAIL',
        'relative_residual_static': maximum(b['relative'], expected_relative),
        'relative_residual_random256': maximum(rb['relative'], expected),
        'distance_residual_static': maximum(a['distances'], b['distances']),
        'distance_residual_random256': maximum(ra['distances'], rb['distances']),
        'full_spectrum_induced_law_residual': maximum(transformed_f, induced),
        'full_spectrum_reconstruction_residual': maximum(a['relative'], reconstructed),
        'center_neighbor_distance_before': float(a['distances'][0, 0, 0]),
        'center_neighbor_distance_after_min': float(b['distances'][0, 0].min()),
        'center_neighbor_distance_after_max': float(b['distances'][0, 0].max()),
        'original_DC_real': f[0, 0, 0].real.tolist(),
        'original_frequency1_real': f[0, 0, 1].real.tolist(),
        'original_frequency1_imag': f[0, 0, 1].imag.tolist(),
        'transformed_frequency1_real': transformed_f[0, 0, 1].real.tolist(),
        'transformed_frequency1_imag': transformed_f[0, 0, 1].imag.tolist(),
        'original_frequency1_norm': float(torch.linalg.vector_norm(f[0, 0, 1])),
        'transformed_frequency1_norm': float(torch.linalg.vector_norm(transformed_f[0, 0, 1])),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'module_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        'dtype': 'float64', 'random_seed': 11092026, 'labels_used': False,
    }
    np.savez_compressed(out/'counterexample.npz', original=x.numpy(), transformed=y.numpy(),
                        Q=q.numpy(), relative=a['relative'].numpy(), relative_star=b['relative'].numpy(),
                        spectrum=f.numpy(), spectrum_star=transformed_f.numpy())
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
