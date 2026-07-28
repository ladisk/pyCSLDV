"""
Virtual continuous scanning LDV experiment.

Generates the velocity signal an LDV would measure while its laser spot
follows a Lissajous trajectory over a harmonically vibrating surface, for
validating the reconstruction chain without hardware.
"""

import numpy as np

from .scan import lissajous

__all__ = ["simulate_response", "plate_mode", "chebyshev_shape"]


def simulate_response(shape, fz, fx, fy, fs, n_samples,
                      phase_x=0.0, phase_y=0.0, response_phase=0.0,
                      noise_std=0.0, rng=None):
    """
    Simulate a CSLDV velocity measurement.

    :param shape: callable ``shape(x, y)`` returning the deflection-shape
        amplitude on the normalized domain ``[-1, 1] x [-1, 1]``
    :param fz: response (excitation) frequency [Hz]
    :param fx: x scan frequency [Hz]
    :param fy: y scan frequency [Hz]
    :param fs: sampling frequency [Hz]
    :param n_samples: number of samples
    :param phase_x: x mirror phase [rad]
    :param phase_y: y mirror phase [rad]
    :param response_phase: phase of the harmonic response [rad]
    :param noise_std: standard deviation of additive Gaussian noise
    :param rng: optional :class:`numpy.random.Generator`
    :return: ``(t, x, y, velocity)``
    """
    t, x, y = lissajous(fx, fy, n_samples, fs, phase_x, phase_y)
    velocity = shape(x, y) * np.cos(2 * np.pi * fz * t + response_phase)
    if noise_std:
        if rng is None:
            rng = np.random.default_rng()
        velocity = velocity + rng.normal(0.0, noise_std, n_samples)
    return t, x, y, velocity


def plate_mode(p, q):
    """
    Analytical mode shape of a simply supported rectangular plate.

    Returns the shape ``sin(p pi (x+1)/2) sin(q pi (y+1)/2)`` on the
    normalized domain, useful as a reference deflection shape.

    :param p: number of half-waves in the x direction
    :param q: number of half-waves in the y direction
    :return: callable ``shape(x, y)``
    """
    def shape(x, y):
        return (np.sin(p * np.pi * (np.asarray(x) + 1) / 2)
                * np.sin(q * np.pi * (np.asarray(y) + 1) / 2))
    return shape


def chebyshev_shape(coefficients):
    """
    Deflection shape defined by a (real) Chebyshev coefficient matrix.

    :param coefficients: array ``C`` where ``C[n, m]`` multiplies
        ``T_n(x) T_m(y)``
    :return: callable ``shape(x, y)``
    """
    from numpy.polynomial import chebyshev

    def shape(x, y):
        return chebyshev.chebval2d(np.asarray(x), np.asarray(y), coefficients)
    return shape
