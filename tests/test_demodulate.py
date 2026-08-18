import numpy as np
import pytest

import pycsldv

# Frequencies chosen so that all sidebands fz +/- n*fx +/- m*fy fall on
# distinct DFT bins for a 10 s measurement (see test docstrings).
FS = 5000.0
N = 50000       # 10 s
FX = 1.4
FY = 8.0
FZ = 500.0


class TestRoundTrip:

    def test_exact_chebyshev_shape(self):
        """A shape that is exactly a low-order Chebyshev series is
        recovered to numerical precision from a noise-free simulation."""
        c_true = np.zeros((4, 4))
        c_true[0, 0] = 0.2
        c_true[1, 2] = 1.0
        c_true[3, 1] = -0.7
        c_true[2, 0] = 0.4
        shape = pycsldv.chebyshev_shape(c_true)

        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS)
        c_est = pycsldv.demodulate_ods(v, x, y, FS, FX, FY, FZ, order=3)

        assert np.allclose(c_est.imag, 0.0, atol=1e-8)
        assert np.allclose(c_est.real, c_true, atol=1e-8)

    def test_phases_are_compensated(self):
        """Mirror phases (inertia lag) and the response phase do not
        corrupt the recovered coefficients."""
        c_true = np.zeros((3, 3))
        c_true[1, 1] = 1.0
        c_true[2, 2] = 0.5
        shape = pycsldv.chebyshev_shape(c_true)

        _, x, y = pycsldv.lissajous(FX, FY, N, FS, phase_x=0.5, phase_y=-0.3)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS, response_phase=0.7)
        c_est = pycsldv.demodulate_ods(v, x, y, FS, FX, FY, FZ, order=2)

        aligned = pycsldv.align_phase(c_est)
        assert np.allclose(aligned.imag, 0.0, atol=1e-8)
        assert np.allclose(aligned.real, c_true, atol=1e-8)

    def test_plate_mode_with_noise(self):
        """A smooth analytical mode shape measured with additive noise is
        reconstructed with a high MAC value."""
        shape = pycsldv.plate_mode(2, 3)
        rng = np.random.default_rng(42)
        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS,
                                      noise_std=0.05, rng=rng)
        c_est = pycsldv.demodulate_ods(v, x, y, FS, FX, FY, FZ, order=8)

        x_grid, y_grid, z = pycsldv.evaluate_ods(pycsldv.align_phase(c_est),
                                                 resolution=50)
        reference = shape(x_grid, y_grid)
        assert pycsldv.mac(reference, z.real) > 0.99

    def test_sideband_collision_warning(self):
        """Overlapping sidebands (fy an integer multiple of fx) trigger a
        warning."""
        shape = pycsldv.plate_mode(1, 1)
        _, x, y = pycsldv.lissajous(2.0, 4.0, N, FS)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS)
        with pytest.warns(UserWarning, match="overlaps"):
            pycsldv.demodulate_ods(v, x, y, FS, 2.0, 4.0, FZ, order=2)


class TestEvaluate:

    def test_evaluate_ods_grid(self):
        c = np.zeros((2, 2))
        c[1, 0] = 1.0  # T_1(x) = x
        x_grid, y_grid, z = pycsldv.evaluate_ods(c, resolution=11)
        assert x_grid.shape == y_grid.shape == z.shape == (11, 11)
        assert np.allclose(z, x_grid)

    def test_align_phase(self):
        c = np.array([[1.0, 0.2], [0.1, -0.4]]) * np.exp(1j * 0.9)
        aligned = pycsldv.align_phase(c)
        assert np.allclose(aligned.imag, 0.0, atol=1e-12)
        assert aligned[0, 0] == pytest.approx(1.0)
