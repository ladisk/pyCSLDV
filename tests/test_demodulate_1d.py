"""
The least-squares reconstruction of a line-scan ODS.

When the laser is swept along a single line rather than over an area, the
shape is a one-dimensional Chebyshev series and
:func:`pycsldv.demodulate_ods_1d` fits

.. math::

    v(t) = \\sum_k \\mathrm{Re}\\left\\{ C^{(k)}(x(t))\\,
           e^{\\lambda_k t} \\right\\},
    \\qquad x(t) = \\cos(2 \\pi f_x t + \\phi_x),

in one global linear system. Damping, several modes and several sensors are
handled exactly as in the two-dimensional case; the downstream
:func:`pycsldv.evaluate_ods` and :func:`pycsldv.plot_ods` recognize a line
scan from the shape of the coefficients.
"""

import numpy as np
import pytest
from numpy.polynomial import chebyshev

import pycsldv

FS = 5000.0
N = 50000       # 10 s
FX = 1.4
FZ = 500.0


def line_response(coefficients, frequency, x, t, phase=0.0):
    """Velocity of a line whose deflection shape is a Chebyshev series."""
    shape = chebyshev.chebval(x, coefficients)
    return (shape * np.exp(1j * (2 * np.pi * frequency * t + phase))).real


class TestRoundTrip:

    def test_exact_chebyshev_series(self):
        """A shape that is exactly a low-order Chebyshev series is
        recovered to numerical precision from a noise-free simulation."""
        c_true = np.array([0.2, 1.0, -0.7, 0.4])

        t, x = pycsldv.sinusoidal_scan(FX, N, FS)
        v = line_response(c_true, FZ, x, t)
        c_est = pycsldv.demodulate_ods_1d(v, FS, fn=FZ, fx=FX, order=3)

        assert c_est.shape == (4,)
        assert np.allclose(c_est.imag, 0.0, atol=1e-8)
        assert np.allclose(c_est.real, c_true, atol=1e-8)

    def test_with_noise(self):
        """A smooth shape measured with additive noise is reconstructed
        with a high MAC value."""
        c_true = np.array([0.0, 1.0, 0.0, -0.5, 0.0, 0.2])
        rng = np.random.default_rng(42)

        t, x = pycsldv.sinusoidal_scan(FX, N, FS)
        v = line_response(c_true, FZ, x, t) + rng.normal(0.0, 0.05, N)

        c_est = pycsldv.align_phase(
            pycsldv.demodulate_ods_1d(v, FS, fn=FZ, fx=FX, order=5))
        points, z = pycsldv.evaluate_ods(c_est, resolution=100)
        reference = chebyshev.chebval(points, c_true)
        assert pycsldv.mac(reference, z.real) > 0.99

    def test_the_scan_phase_is_compensated(self):
        """The mirror lag enters as ``phi_x``; the remaining global phase is
        the response phase."""
        c_true = np.array([0.0, 1.0, 0.5])

        t, x = pycsldv.sinusoidal_scan(FX, N, FS, phase=0.5)
        v = line_response(c_true, FZ, x, t, phase=0.7)
        c_est = pycsldv.demodulate_ods_1d(v, FS, fn=FZ, fx=FX, order=2, phi_x=0.5)

        assert np.allclose(np.angle(c_est[1]), 0.7)
        aligned = pycsldv.align_phase(c_est)
        assert np.allclose(aligned.imag, 0.0, atol=1e-8)
        assert np.allclose(aligned.real, c_true, atol=1e-8)


class TestComplexModes:

    def test_damped_mode_from_a_pole(self):
        """A decaying response with a complex shape is recovered from its
        pole."""
        c_true = np.array([0.0, np.exp(1j * 0.9), 0.5 * np.exp(1j * 2.1)])
        pole = -3.0 + 2j * np.pi * FZ

        t, x = pycsldv.sinusoidal_scan(FX, N, FS)
        v = (chebyshev.chebval(x, c_true) * np.exp(pole * t)).real

        c_est = pycsldv.demodulate_ods_1d(v, FS, poles=pole, fx=FX, order=2)
        assert np.allclose(c_est, c_true, atol=1e-8)

    def test_a_zero_frequency_mode_is_real(self):
        """A static component has no quadrature part."""
        c_true = np.array([0.0, 1.0, 0.5])
        _, x = pycsldv.sinusoidal_scan(FX, N, FS)

        c_est = pycsldv.demodulate_ods_1d(chebyshev.chebval(x, c_true), FS,
                                          fn=0.0, fx=FX, order=2)
        assert np.allclose(c_est.imag, 0.0)
        assert np.allclose(c_est.real, c_true, atol=1e-8)


class TestSeveralModes:

    def test_two_closely_spaced_modes(self):
        """Two modes 3 Hz apart are fitted simultaneously."""
        f_1, f_2 = FZ, FZ + 3.0
        c_1 = np.array([0.0, 1.0, 0.0, -0.5])
        c_2 = np.array([0.0, 0.0, 1.0, 0.3])

        t, x = pycsldv.sinusoidal_scan(FX, N, FS)
        v = (line_response(c_1, f_1, x, t)
             + 0.7 * line_response(c_2, f_2, x, t, phase=1.1))

        estimated = pycsldv.demodulate_ods_1d(v, FS, fn=[f_1, f_2], fx=FX, order=3)
        assert len(estimated) == 2
        for c_est, c_true, scale in zip(estimated, (c_1, c_2), (1.0, 0.7)):
            aligned = pycsldv.align_phase(c_est)
            assert np.allclose(aligned.real, scale * c_true, atol=1e-6)

    def test_order_per_mode(self):
        estimated = pycsldv.demodulate_ods_1d(np.zeros(N), FS,
                                              fn=[FZ, FZ + 3.0], fx=FX,
                                              order=[3, 5])
        assert [c.shape for c in estimated] == [(4,), (6,)]


class TestSeveralSensors:

    def test_sensors_share_the_design_matrix(self):
        """A stack of velocity signals returns one coefficient vector per
        location."""
        c_true = np.array([0.0, 1.0, -0.5])
        t, x = pycsldv.sinusoidal_scan(FX, N, FS)
        v = line_response(c_true, FZ, x, t)

        c_est = pycsldv.demodulate_ods_1d(np.vstack([v, 2.0 * v, -0.5 * v]),
                                          FS, fn=FZ, fx=FX, order=2)
        assert c_est.shape == (3, 3)
        for location, scale in enumerate([1.0, 2.0, -0.5]):
            assert np.allclose(c_est[location].real, scale * c_true, atol=1e-8)


class TestArguments:

    def test_a_frequency_or_a_pole_is_required(self):
        with pytest.raises(ValueError, match="fn.*poles|poles.*fn"):
            pycsldv.demodulate_ods_1d(np.zeros(N), FS, fx=FX)

    def test_rejects_an_order_that_does_not_match_the_modes(self):
        with pytest.raises(ValueError, match="same length"):
            pycsldv.demodulate_ods_1d(np.zeros(N), FS, fn=[FZ, FZ + 3.0],
                                      fx=FX, order=[1, 2, 3])


class TestLineScanIsRecognized:

    def test_evaluate_ods_returns_a_line(self):
        """A 1D coefficient vector evaluates to a line, not a grid."""
        points, z = pycsldv.evaluate_ods(np.array([0.0, 1.0, 0.5]), resolution=11)
        assert points.shape == (11,) and z.shape == (11,)
        assert np.allclose(z, chebyshev.chebval(points, [0.0, 1.0, 0.5]))

    def test_plot_ods_draws_a_line(self):
        """:func:`pycsldv.plot_ods` switches to a line plot."""
        import matplotlib
        matplotlib.use("Agg")

        ax = pycsldv.plot_ods(np.array([0.0, 1.0, 0.5]), resolution=11)
        assert len(ax.lines) == 1
        assert ax.get_ylabel() == "deflection [-]"

    def test_plot_ods_rejects_a_bad_part(self):
        import matplotlib
        matplotlib.use("Agg")

        with pytest.raises(ValueError, match="part"):
            pycsldv.plot_ods(np.array([0.0, 1.0]), part="magnitude")
