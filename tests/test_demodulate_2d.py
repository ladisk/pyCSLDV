"""
The least-squares reconstruction of a two-dimensional ODS.

:func:`pycsldv.demodulate_ods_2d` fits

.. math::

    v(t) = \\sum_k \\mathrm{Re}\\left\\{ C^{(k)}(x(t), y(t))\\,
           e^{\\lambda_k t} \\right\\}

to the measured velocity in one global linear system, rather than projecting
onto the individual sidebands as :func:`pycsldv.demodulate_ods` does. That
buys three things the projection cannot give -- complex (damped) modes,
several modes at once even when their sidebands overlap, and several sensors
sharing one design matrix -- and costs one: the scan trajectory is no longer
passed in but rebuilt from ``fx``, ``fy`` and the mirror phases, so those
phases have to be supplied by the caller.
"""

import numpy as np
import pytest

import pycsldv

FS = 5000.0
N = 50000       # 10 s
FX = 1.4
FY = 8.0
FZ = 500.0


def chebyshev_along(coefficients, x, y):
    """Evaluate a Chebyshev series along a path, for coefficients that may
    be complex; :func:`pycsldv.chebyshev_shape` is real-valued."""
    coefficients = np.asarray(coefficients)
    t_x = np.cos(np.arange(coefficients.shape[0])[:, None] * np.arccos(np.clip(x, -1, 1)))
    t_y = np.cos(np.arange(coefficients.shape[1])[:, None] * np.arccos(np.clip(y, -1, 1)))
    return np.einsum("pa,qa,pq->a", t_x, t_y, coefficients)


class TestRoundTrip:

    def test_exact_chebyshev_shape(self):
        """A shape that is exactly a low-order Chebyshev series is
        recovered to numerical precision from a noise-free simulation."""
        c_true = np.zeros((4, 4))
        c_true[0, 0] = 0.2
        c_true[1, 2] = 1.0
        c_true[3, 1] = -0.7
        shape = pycsldv.chebyshev_shape(c_true)

        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS)
        c_est = pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=3)

        assert c_est.shape == (4, 4)
        assert np.allclose(c_est.imag, 0.0, atol=1e-8)
        assert np.allclose(c_est.real, c_true, atol=1e-8)

    def test_plate_mode_with_noise(self):
        """A smooth analytical mode shape measured with additive noise is
        reconstructed with a high MAC value."""
        shape = pycsldv.plate_mode(2, 3)
        rng = np.random.default_rng(42)
        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS, noise_std=0.05, rng=rng)

        c_est = pycsldv.align_phase(
            pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=8))
        x_grid, y_grid, z = pycsldv.evaluate_ods(c_est, resolution=50)
        assert pycsldv.mac(shape(x_grid, y_grid), z.real) > 0.99

    def test_agrees_with_the_sideband_projection(self):
        """On data both methods can handle, the global fit and the
        per-sideband projection give the same shape."""
        shape = pycsldv.plate_mode(2, 3)
        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS)

        projected = pycsldv.align_phase(
            pycsldv.demodulate_ods(v, x, y, FS, FX, FY, FZ, order=6))
        fitted = pycsldv.align_phase(
            pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=6))
        assert np.allclose(projected, fitted, atol=1e-6)


class TestScanPhase:

    def test_mirror_and_response_phases_are_compensated(self):
        """Given the mirror phases, the recovered coefficients differ from
        the true ones only by the global response phase, which
        :func:`pycsldv.align_phase` removes."""
        c_true = np.zeros((3, 3))
        c_true[1, 1] = 1.0
        c_true[2, 2] = 0.5
        shape = pycsldv.chebyshev_shape(c_true)

        _, x, y = pycsldv.lissajous(FX, FY, N, FS, phase_x=0.5, phase_y=-0.3)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS, response_phase=0.7)
        c_est = pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=2,
                                          phi_x=0.5, phi_y=-0.3)

        assert np.allclose(np.angle(c_est[1, 1]), 0.7)
        aligned = pycsldv.align_phase(c_est)
        assert np.allclose(aligned.imag, 0.0, atol=1e-8)
        assert np.allclose(aligned.real, c_true, atol=1e-8)

    def test_the_phases_are_read_off_the_mirror_feedback(self):
        """The trajectory is not passed in, so the mirror phases have to
        come from somewhere; :func:`pycsldv.reference_phase` recovers them
        from the recorded mirror signals."""
        shape = pycsldv.plate_mode(2, 3)
        _, x, y = pycsldv.lissajous(FX, FY, N, FS, phase_x=0.5, phase_y=-0.3)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS)

        phi_x = pycsldv.reference_phase(x, FX, FS)
        phi_y = pycsldv.reference_phase(y, FY, FS)
        assert phi_x == pytest.approx(0.5, abs=1e-3)
        assert phi_y == pytest.approx(-0.3, abs=1e-3)

        c_est = pycsldv.align_phase(
            pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=8,
                                      phi_x=phi_x, phi_y=phi_y))
        x_grid, y_grid, z = pycsldv.evaluate_ods(c_est, resolution=50)
        assert pycsldv.mac(shape(x_grid, y_grid), z.real) > 0.99

    def test_ignoring_the_mirror_phase_corrupts_the_shape(self):
        """A lagging mirror is not compensated by itself: leaving the phase
        at its default when the mirrors do lag gives the wrong shape."""
        shape = pycsldv.plate_mode(2, 3)
        _, x, y = pycsldv.lissajous(FX, FY, N, FS, phase_x=0.5, phase_y=-0.3)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS)

        c_est = pycsldv.align_phase(
            pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=8))
        x_grid, y_grid, z = pycsldv.evaluate_ods(c_est, resolution=50)
        assert pycsldv.mac(shape(x_grid, y_grid), z.real) < 0.9


class TestComplexModes:

    def test_damped_mode_from_a_pole(self):
        """A decaying response with a genuinely complex shape is recovered
        from its pole, magnitude and phase together."""
        c_true = np.zeros((3, 3), dtype=complex)
        c_true[1, 1] = np.exp(1j * 0.9)
        c_true[2, 0] = 0.5 * np.exp(1j * 2.1)
        pole = -3.0 + 2j * np.pi * FZ

        t, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = (chebyshev_along(c_true, x, y) * np.exp(pole * t)).real

        c_est = pycsldv.demodulate_ods_2d(v, FS, FX, FY, poles=pole, order=2)
        assert np.allclose(c_est, c_true, atol=1e-8)

    def test_damping_cannot_be_ignored(self):
        """The same data fitted as an undamped response at the same
        frequency does not give the shape."""
        shape = pycsldv.plate_mode(2, 3)
        pole = -3.0 + 2j * np.pi * FZ
        t, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = (shape(x, y) * np.exp(pole * t)).real

        damped = pycsldv.demodulate_ods_2d(v, FS, FX, FY, poles=pole, order=6)
        undamped = pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=6)
        assert not np.allclose(damped, undamped, atol=1e-3)

        x_grid, y_grid, z = pycsldv.evaluate_ods(pycsldv.align_phase(damped),
                                                 resolution=50)
        assert pycsldv.mac(shape(x_grid, y_grid), z.real) > 0.99

    def test_a_zero_frequency_mode_is_real(self):
        """A static (zero-frequency) component has no quadrature part and
        is fitted with the real block only."""
        c_true = np.zeros((3, 3))
        c_true[1, 1] = 1.0
        shape = pycsldv.chebyshev_shape(c_true)

        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = shape(x, y)

        c_est = pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=0.0, order=2)
        assert np.allclose(c_est.imag, 0.0)
        assert np.allclose(c_est.real, c_true, atol=1e-8)


class TestSeveralModes:

    def test_two_closely_spaced_modes(self):
        """Two modes 3 Hz apart -- far closer than the sidebands could
        separate -- are fitted simultaneously."""
        f_1, f_2 = FZ, FZ + 3.0
        shape_1, shape_2 = pycsldv.plate_mode(2, 3), pycsldv.plate_mode(3, 2)

        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = (pycsldv.simulate_response(shape_1, f_1, x, y, FS)
             + 0.7 * pycsldv.simulate_response(shape_2, f_2, x, y, FS,
                                               response_phase=1.1))

        estimated = pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=[f_1, f_2], order=8)
        assert len(estimated) == 2
        for c_est, shape in zip(estimated, (shape_1, shape_2)):
            x_grid, y_grid, z = pycsldv.evaluate_ods(pycsldv.align_phase(c_est),
                                                     resolution=50)
            assert pycsldv.mac(shape(x_grid, y_grid), z.real) > 0.99

    def test_order_per_mode(self):
        """One order per mode, the same in both directions."""
        estimated = pycsldv.demodulate_ods_2d(np.zeros(N), FS, FX, FY,
                                              fn=[FZ, FZ + 3.0], order=[3, 5])
        assert [c.shape for c in estimated] == [(4, 4), (6, 6)]

    def test_order_per_mode_and_direction(self):
        """An order for each direction of each mode."""
        estimated = pycsldv.demodulate_ods_2d(np.zeros(N), FS, FX, FY,
                                              fn=[FZ, FZ + 3.0],
                                              order=[[3, 5], [5, 3]])
        assert [c.shape for c in estimated] == [(4, 6), (6, 4)]

    def test_a_transposed_order_is_accepted(self):
        """``(2, n_modes)`` is taken as the two directions of each mode.
        Three modes, so the layout is unambiguous."""
        frequencies = [FZ, FZ + 3.0, FZ + 6.0]
        estimated = pycsldv.demodulate_ods_2d(np.zeros(N), FS, FX, FY,
                                              fn=frequencies,
                                              order=[[3, 4, 5], [5, 4, 3]])
        assert [c.shape for c in estimated] == [(4, 6), (5, 5), (6, 4)]

    def test_order_for_a_single_mode_in_two_directions(self):
        """For one mode, a pair of orders is read as the two directions."""
        c_est = pycsldv.demodulate_ods_2d(np.zeros(N), FS, FX, FY,
                                          fn=FZ, order=[3, 5])
        assert c_est.shape == (4, 6)


class TestSeveralSensors:

    def test_sensors_share_the_design_matrix(self):
        """A stack of velocity signals is fitted in one solve and returns
        one coefficient matrix per location."""
        shape = pycsldv.plate_mode(2, 3)
        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = pycsldv.simulate_response(shape, FZ, x, y, FS)

        velocities = np.vstack([v, 2.0 * v, -0.5 * v])
        c_est = pycsldv.demodulate_ods_2d(velocities, FS, FX, FY, fn=FZ, order=6)

        assert c_est.shape == (3, 7, 7)
        single = pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=6)
        for location, scale in enumerate([1.0, 2.0, -0.5]):
            assert np.allclose(c_est[location], scale * single, atol=1e-8)

    def test_several_sensors_and_several_modes(self):
        """The two extensions compose: one matrix per mode, per location."""
        velocities = np.zeros((2, N))
        estimated = pycsldv.demodulate_ods_2d(velocities, FS, FX, FY,
                                              fn=[FZ, FZ + 3.0], order=4)
        assert [c.shape for c in estimated] == [(2, 5, 5), (2, 5, 5)]


class TestArguments:

    def test_a_frequency_or_a_pole_is_required(self):
        with pytest.raises(ValueError, match="fn.*poles|poles.*fn"):
            pycsldv.demodulate_ods_2d(np.zeros(N), FS, FX, FY)

    def test_rejects_an_order_that_does_not_match_the_modes(self):
        with pytest.raises(ValueError, match="order"):
            pycsldv.demodulate_ods_2d(np.zeros(N), FS, FX, FY,
                                      fn=[FZ, FZ + 3.0], order=[1, 2, 3])

    def test_rejects_a_misshapen_order(self):
        with pytest.raises(ValueError, match="order"):
            pycsldv.demodulate_ods_2d(np.zeros(N), FS, FX, FY,
                                      fn=[FZ, FZ + 3.0],
                                      order=[[1, 2, 3], [4, 5, 6]])

    def test_rejects_a_negative_order(self):
        with pytest.raises(ValueError, match="non-negative"):
            pycsldv.demodulate_ods_2d(np.zeros(N), FS, FX, FY, fn=FZ, order=-1)


class TestWithRotation:

    def test_a_fitted_shape_can_be_turned_onto_the_specimen_axes(self):
        """The two halves of the package meet: a specimen mounted askew is
        reconstructed by the global fit in the frame of the scan, and
        :func:`pycsldv.rotate_ods` turns the result onto its own axes."""
        angle = np.deg2rad(7.0)
        upright = pycsldv.plate_mode(2, 3)

        def askew(x, y):
            return upright(x * np.cos(angle) + y * np.sin(angle),
                           -x * np.sin(angle) + y * np.cos(angle))

        _, x, y = pycsldv.lissajous(FX, FY, N, FS)
        v = pycsldv.simulate_response(askew, FZ, x, y, FS)
        c_est = pycsldv.align_phase(
            pycsldv.demodulate_ods_2d(v, FS, FX, FY, fn=FZ, order=8))

        x_grid, y_grid, z = pycsldv.evaluate_ods(c_est, resolution=60)
        assert pycsldv.mac(upright(x_grid, y_grid), z.real) < 0.95

        rotated = pycsldv.rotate_ods(c_est, angle)
        x_grid, y_grid, z = pycsldv.evaluate_ods(rotated, resolution=60)
        # away from the corners, which the rotated series extrapolates
        inside = np.hypot(x_grid, y_grid) < 0.7
        assert pycsldv.mac(upright(x_grid, y_grid)[inside], z.real[inside]) > 0.99
