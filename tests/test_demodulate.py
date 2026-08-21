import warnings

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval2d

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


class TestRotate:
    """Expressing a reconstructed shape in a rotated frame."""

    C = np.array([[0.2, 0.0, 0.5, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.3, 0.0, 0.0, 0.0],
                  [0.0, -0.7, 0.0, 0.0]])

    def test_zero_angle_is_the_identity(self):
        rotated = pycsldv.rotate_ods(self.C, 0.0)
        assert np.allclose(rotated[:4, :4], self.C, atol=1e-12)
        assert np.allclose(rotated[4:, :], 0.0) and np.allclose(rotated[:, 4:], 0.0)

    def test_matches_direct_substitution(self):
        """The rotated coefficients describe the same shape, evaluated at
        rotated coordinates. Checked well inside the domain, where no
        extrapolation is involved."""
        angle = np.radians(17.0)
        points = np.linspace(-0.6, 0.6, 25)
        u, v = np.meshgrid(points, points, indexing="ij")
        direct = chebval2d(u * np.cos(angle) - v * np.sin(angle),
                           u * np.sin(angle) + v * np.cos(angle), self.C)

        rotated = chebval2d(u, v, pycsldv.rotate_ods(self.C, angle))
        assert np.allclose(direct, rotated, atol=1e-12)

    def test_aspect_stretches_the_rotation(self):
        """On a scan that is longer than it is wide, a rotation of the
        physical surface is not a rotation of the normalized domain: it
        tilts the shape by the aspect ratio more."""
        angle, aspect = np.radians(2.0), 5.0
        points = np.linspace(-0.6, 0.6, 25)
        u, v = np.meshgrid(points, points, indexing="ij")
        direct = chebval2d(u * np.cos(angle) - v * np.sin(angle) / aspect,
                           u * np.sin(angle) * aspect + v * np.cos(angle), self.C)

        rotated = chebval2d(u, v, pycsldv.rotate_ods(self.C, angle, aspect=aspect))
        assert np.allclose(direct, rotated, atol=1e-12)
        # and it is a different shape from the isotropic rotation
        assert not np.allclose(rotated, chebval2d(u, v, pycsldv.rotate_ods(self.C, angle)))

    def test_round_trip(self):
        angle = np.radians(23.0)
        there = pycsldv.rotate_ods(self.C, angle)
        back = pycsldv.rotate_ods(there, -angle)
        assert np.allclose(back[:4, :4], self.C, atol=1e-10)
        assert np.abs(back[4:, :]).max() < 1e-10

    def test_complex_coefficients_are_preserved(self):
        """The reconstruction is complex, and a rotation acts on the shape,
        not on the phase."""
        coefficients = self.C * np.exp(1j * 0.8)
        rotated = pycsldv.rotate_ods(coefficients, np.radians(11.0))
        assert np.iscomplexobj(rotated)
        assert np.allclose(rotated, pycsldv.rotate_ods(self.C, np.radians(11.0))
                           * np.exp(1j * 0.8), atol=1e-12)

    def test_degree_grows(self):
        """A rotation mixes the directions, so the tensor-product degree
        grows even though the total degree does not."""
        assert pycsldv.rotate_ods(np.zeros((5, 3)), 0.4).shape == (7, 7)

    def test_warns_when_the_domain_is_left(self):
        """A slender scan turned through a large angle pushes the corners
        far outside the domain the series was fitted on."""
        with pytest.warns(UserWarning, match="extrapolates"):
            pycsldv.rotate_ods(self.C, np.radians(10.0), aspect=5.0)

    def test_rejects_a_shape_it_cannot_rotate(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            pycsldv.rotate_ods([1.0, 2.0, 3.0], 0.3)
        with pytest.raises(ValueError, match="positive extent ratio"):
            pycsldv.rotate_ods(self.C, 0.3, aspect=0.0)

    def test_no_warning_for_a_small_rotation(self):
        warnings.simplefilter("error")
        try:
            pycsldv.rotate_ods(self.C, np.radians(2.0))
        finally:
            warnings.resetwarnings()
