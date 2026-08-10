"""
Equivalence of :func:`pycsldv.demodulate_ods` with the original MATLAB
implementation.

The tests in ``test_demodulate.py`` verify pyCSLDV against its own
simulation, so a shared misconception in ``simulate.py`` and
``demodulate.py`` would pass unnoticed. The reference implementation below
closes that gap: it is a direct, deliberately un-idiomatic transcription of
``ComputeODS.m`` from the original CSLDV suite (Bartlett & Tarazaga,
CC-BY-4.0, DOI 10.5281/zenodo.21301126), kept close to the MATLAB line by
line so that it can be checked against the published source by eye.

Two bookkeeping differences between the implementations are expected and
are handled by the tests:

* The MATLAB code applies the sideband-count normalization implicitly --
  the entries of its polynomial transformation matrix ``T`` are twice the
  Chebyshev-to-monomial coefficients for every order above zero, which
  exactly cancels the ``1/4``, ``1/2`` and ``1`` sideband factors of the
  interior, edge and DC coefficients. pyCSLDV applies those factors
  explicitly in :func:`pycsldv.demodulate_ods` and stays in the Chebyshev
  basis. The reconstructions are identical.
* The global phase is normalized differently: MATLAB subtracts the
  response baseband phase, :func:`pycsldv.align_phase` rotates the dominant
  coefficient onto the real axis. The reconstructions therefore agree up to
  a real scale factor, which the MAC is insensitive to.

MATLAB's ``meshgrid`` uses ``'xy'`` indexing and
:func:`pycsldv.evaluate_ods` uses ``'ij'``, so the reference grid is
transposed before comparison.
"""

import numpy as np
import pytest
from scipy.signal.windows import hann

import pycsldv

FS = 5000.0
N = 50000       # 10 s
FX = 1.4
FY = 8.0
FZ = 500.0
RESOLUTION = 40

# Get_T_Matrix from ComputeODS.m, leading 8x8 block (orders 0..7).
T_MATRIX = np.array([
    [1, 0, -2, 0, 2, 0, -2, 0],
    [0, 2, 0, -6, 0, 10, 0, -14],
    [0, 0, 4, 0, -16, 0, 36, 0],
    [0, 0, 0, 8, 0, -40, 0, 112],
    [0, 0, 0, 0, 16, 0, -96, 0],
    [0, 0, 0, 0, 0, 32, 0, -224],
    [0, 0, 0, 0, 0, 0, 64, 0],
    [0, 0, 0, 0, 0, 0, 0, 128],
], dtype=float)


def _trigonometric_projection(signal, frequency, phase_shift, sample, n_total):
    """Transcription of Compute_Trigonometric_Projection."""
    real = np.sum(signal * np.cos(frequency * sample - phase_shift)) / n_total
    imag = np.sum(signal * np.sin(frequency * sample - phase_shift)) / n_total
    magnitude = np.sqrt(real ** 2 + imag ** 2)
    phase = np.arctan(imag / real)
    if imag < 0 and real >= 0:
        phase = 2 * np.pi + phase
    if real < 0:
        phase = np.pi + phase
    return magnitude, phase


def compute_ods_matlab(x, y, z, fs, n_total, fx, fy, fz, nm, resolution):
    """Transcription of ComputeODS.m (without the ODS rotation).

    :return: ``(x_grid, y_grid, ods)`` with ``'xy'`` grid indexing
    """
    sample = np.arange(n_total)
    window = hann(n_total, sym=False)
    hx, hy, hz = 2 * x * window, 2 * y * window, 2 * z * window

    wx = 2 * np.pi * fx / fs
    wy = 2 * np.pi * fy / fs
    wz = 2 * np.pi * fz / fs

    _, x_base = _trigonometric_projection(hx, wx, 0, sample, n_total)
    _, y_base = _trigonometric_projection(hy, wy, 0, sample, n_total)
    _, z_base = _trigonometric_projection(hz, wz, 0, sample, n_total)

    real_val = np.zeros((nm + 1, nm + 1))
    imag_val = np.zeros((nm + 1, nm + 1))
    signs = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    for m in range(nm + 1):
        for n in range(nm + 1):
            magnitudes = np.zeros(4)
            phases = np.zeros(4)
            for s, (sx, sy) in enumerate(signs):
                s_omega = sx * n * wx + sy * m * wy + wz
                s_phi = sx * n * x_base + sy * m * y_base
                magnitudes[s], phases[s] = _trigonometric_projection(
                    hz, s_omega, s_phi, sample, n_total)
            upper, lower = 0.8 * 2 * np.pi, 0.2 * 2 * np.pi
            for a in range(4):
                others = [phases[b] for b in range(4) if b != a]
                if phases[a] > upper and any(o < lower for o in others):
                    phases[a] -= 2 * np.pi
            phase = np.mean(phases) - z_base
            magnitude = 2 * np.mean(magnitudes)
            real_val[n, m] = magnitude * np.cos(phase)
            imag_val[n, m] = magnitude * np.sin(phase)

    t = T_MATRIX[:nm + 1, :nm + 1]
    real_coeffs = t @ real_val @ t.T
    imag_coeffs = t @ imag_val @ t.T

    points = np.linspace(-1, 1, resolution)
    x_grid, y_grid = np.meshgrid(points, points)
    z_real = np.zeros_like(x_grid)
    z_imag = np.zeros_like(x_grid)
    for m in range(nm + 1):
        for n in range(nm + 1):
            basis = x_grid ** n * y_grid ** m
            z_real = z_real + real_coeffs[n, m] * basis
            z_imag = z_imag + imag_coeffs[n, m] * basis
    return x_grid, y_grid, z_real + 1j * z_imag


def _reconstruct_both(shape, order, **simulate_kwargs):
    """Reconstruct one simulated measurement with both implementations."""
    _, x, y, velocity = pycsldv.simulate_response(
        shape, FZ, FX, FY, FS, N, **simulate_kwargs)
    *_, reference = compute_ods_matlab(
        x, y, velocity, FS, N, FX, FY, FZ, order, RESOLUTION)
    coefficients = pycsldv.demodulate_ods(
        velocity, x, y, FS, FX, FY, FZ, order=order)
    *_, ods = pycsldv.evaluate_ods(pycsldv.align_phase(coefficients),
                                   resolution=RESOLUTION)
    return reference.T, ods       # 'xy' -> 'ij'


class TestMatlabEquivalence:

    def test_matches_reference_implementation(self):
        """With no mirror or response phase offset the two implementations
        agree to numerical precision, not merely in shape: both phase
        normalizations are then the identity."""
        c_true = np.zeros((4, 4))
        c_true[0, 0] = 0.2
        c_true[1, 2] = 1.0
        c_true[3, 1] = -0.7
        c_true[2, 0] = 0.4
        reference, ods = _reconstruct_both(pycsldv.chebyshev_shape(c_true), 3)

        assert np.allclose(ods.real, reference.real, atol=1e-10)

    def test_matches_reference_with_phases(self):
        """Mirror lag and response phase are compensated identically by
        both implementations, up to their differing global-phase
        normalization."""
        c_true = np.zeros((3, 3))
        c_true[1, 1] = 1.0
        c_true[2, 2] = 0.5
        reference, ods = _reconstruct_both(
            pycsldv.chebyshev_shape(c_true), 2,
            phase_x=0.5, phase_y=-0.3, response_phase=0.7)

        assert pycsldv.mac(reference.real, ods.real) == pytest.approx(1.0, abs=1e-9)

    def test_matches_reference_for_plate_mode(self):
        """A smooth shape that is not exactly representable at the chosen
        order is truncated the same way by both implementations."""
        reference, ods = _reconstruct_both(pycsldv.plate_mode(2, 3), 5)

        assert pycsldv.mac(reference.real, ods.real) == pytest.approx(1.0, abs=1e-9)

    def test_matches_reference_with_noise(self):
        """The implementations remain equivalent on a noisy measurement."""
        rng = np.random.default_rng(42)
        reference, ods = _reconstruct_both(
            pycsldv.plate_mode(2, 3), 5, noise_std=0.05, rng=rng)

        assert pycsldv.mac(reference.real, ods.real) > 0.99
