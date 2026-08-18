"""
Lissajous scan-trajectory design for continuous scanning LDV measurements.

The scan mirrors are driven with two harmonic signals of frequencies ``fx``
and ``fy``; the laser spot then traces a Lissajous pattern over the
(rectangular) measurement surface. All positions are expressed in
normalized coordinates on the domain ``[-1, 1] x [-1, 1]``.
"""

from fractions import Fraction
from math import gcd

import numpy as np
from scipy.signal.windows import hann

from .demodulate import _projection

__all__ = ["lissajous", "sinusoidal_scan", "scan_period", "drive_signals",
           "normalize_scan", "dominant_frequency"]


def lissajous(fx, fy, n_samples, fs, phase_x=0.0, phase_y=0.0, amplitude=(1.0, 1.0)):
    """
    Generate a Lissajous scan trajectory in normalized coordinates.

    :param fx: scan frequency in the x direction [Hz]
    :param fy: scan frequency in the y direction [Hz]
    :param n_samples: number of samples to generate
    :param fs: sampling frequency [Hz]
    :param phase_x: phase of the x mirror-drive signal [rad]
    :param phase_y: phase of the y mirror-drive signal [rad]
    :param amplitude: ``(ax, ay)`` trajectory amplitudes (normalized units)
    :return: ``(t, x, y)`` — time vector [s] and normalized positions
    """
    t = np.arange(n_samples) / fs
    x = amplitude[0] * np.cos(2 * np.pi * fx * t + phase_x)
    y = amplitude[1] * np.cos(2 * np.pi * fy * t + phase_y)
    return t, x, y

def sinusoidal_scan(f, n_samples, fs, phase=0.0, amplitude=1.0):
    """
    Generate a 1D sinusoidal scan trajectory in normalized coordinates.

    :param f: scan frequency [Hz]
    :param n_samples: number of samples to generate
    :param fs: sampling frequency [Hz]
    :param phase: phase of the drive signal [rad]
    :param amplitude: trajectory amplitude (normalized units)
    :return: ``(t, x)`` — time vector [s] and normalized position
    """
    t = np.arange(n_samples) / fs
    x = amplitude * np.cos(2 * np.pi * f * t + phase)
    return t, x

def scan_period(fx, fy, max_denominator=10**6):
    """
    Closure period of the Lissajous pattern.

    The trajectory repeats after ``1 / gcd(fx, fy)`` seconds, where the
    greatest common divisor is evaluated on the rational approximations of
    the two scan frequencies. Measuring for an integer number of closure
    periods ensures uniform surface coverage.

    :param fx: scan frequency in the x direction [Hz]
    :param fy: scan frequency in the y direction [Hz]
    :param max_denominator: rational-approximation limit for the frequencies
    :return: closure period [s]
    """
    rx = Fraction(fx).limit_denominator(max_denominator)
    ry = Fraction(fy).limit_denominator(max_denominator)
    common = Fraction(gcd(rx.numerator, ry.numerator),
                      rx.denominator * ry.denominator // gcd(rx.denominator, ry.denominator))
    return float(1 / common)


def normalize_scan(position, f, fs):
    """
    Map a measured mirror position onto the normalized domain ``[-1, 1]``.

    The reconstruction assumes a scan of the form ``cos(2 pi f t + phi)`` on
    ``[-1, 1]``, whereas a measured mirror feedback signal is an offset
    harmonic in volts or millimetres. The position is centred on its mean and
    scaled by the amplitude of its component at the scan frequency, rather
    than by its extremes, which are sensitive to noise and to cross-axis
    coupling.

    :param position: measured mirror position, in arbitrary units
    :param f: scan frequency of this mirror [Hz]
    :param fs: sampling frequency [Hz]
    :return: ``(normalized, offset, amplitude)`` — the normalized position,
        and the offset and amplitude that were removed, in the units of the
        input, which together give the physical extent of the scan
    """
    position = np.asarray(position, float)
    offset = position.mean()
    amplitude = np.abs(_projection(position - offset, f, fs,
                                   hann(len(position), sym=False)))
    return (position - offset) / amplitude, offset, amplitude


def dominant_frequency(signal, fs):
    """
    Frequency of the largest spectral component of a signal.

    Useful to identify which measured channel scans at which frequency, as
    the reconstruction has to be given the scan frequency belonging to each
    mirror signal.

    :param signal: time-domain signal
    :param fs: sampling frequency [Hz]
    :return: frequency of the largest component of the mean-removed
        spectrum [Hz]
    """
    signal = np.asarray(signal, float)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    return np.fft.rfftfreq(len(signal), 1 / fs)[spectrum.argmax()]


def drive_signals(x, y, scale=(1.0, 1.0), offset=(0.0, 0.0)):
    """
    Map a normalized trajectory to galvo mirror drive voltages.

    A linear mirror characteristic is assumed:
    ``V = scale * position + offset``.

    :param x: normalized x trajectory
    :param y: normalized y trajectory
    :param scale: ``(kx, ky)`` voltage per normalized unit [V]
    :param offset: ``(x0, y0)`` voltage offsets [V]
    :return: ``(vx, vy)`` drive voltage signals [V]
    """
    vx = scale[0] * np.asarray(x) + offset[0]
    vy = scale[1] * np.asarray(y) + offset[1]
    return vx, vy
