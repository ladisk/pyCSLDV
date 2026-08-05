"""
Virtual continuous scanning LDV experiment.

Generates the velocity signal an LDV would measure while its laser spot
follows a Lissajous trajectory over a harmonically vibrating surface, for
validating the reconstruction chain without hardware.
"""

import numpy as np

from .scan import lissajous

__all__ = ["simulate_response", "plate_mode", "plate_frequency", "chebyshev_shape"]


def simulate_response(shape, fz, path_x, path_y, fs,
                      phase_x=0.0, phase_y=0.0, response_phase=0.0,
                      noise_std=0.0, rng=None):
    """
    Simulate a CSLDV velocity measurement.

    :param shape: callable ``shape(x, y)`` returning the deflection-shape
        amplitude on the normalized domain ``[-1, 1] x [-1, 1]``
    :param fz: response (excitation) frequency [Hz]
    :param path_x: x position of the laser spot on the normalized domain
    :param path_y: y position of the laser spot on the normalized domain
    :param fs: sampling frequency [Hz]
    :param response_phase: phase of the harmonic response [rad]
    :param noise_std: standard deviation of additive Gaussian noise
    :param rng: optional :class:`numpy.random.Generator`
    :return: ``(t, x, y, velocity)``
    """
    # t, x, y = lissajous(fx, fy, n_samples, fs, phase_x, phase_y)
    n_samples = len(path_x)
    t = np.arange(n_samples) / fs
    velocity = shape(path_x, path_y) * np.cos(2 * np.pi * fz * t + response_phase)
    if noise_std:
        if rng is None:
            rng = np.random.default_rng()
        velocity = velocity + rng.normal(0.0, noise_std, n_samples)
    return velocity


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

def plate_frequency(p, q, Lx, Ly, E, rho, h, nu = 0.3):
    """
    Natural frequency of a simply supported rectangular plate.

    :param p: number of half-waves in the x direction
    :param q: number of half-waves in the y direction
    :param a: plate length in the x direction [m]
    :param b: plate length in the y direction [m]
    :param E: Young's modulus [Pa]
    :param rho: density [kg/m^3]
    :param h: thickness [m]
    :return: natural frequency [Hz]
    """
    D = E * h**3 / (12 * (1 - nu**2))
    return (np.pi / 2) * np.sqrt(D / (rho * h)) * ((p / Lx)**2 + (q / Ly)**2)

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
