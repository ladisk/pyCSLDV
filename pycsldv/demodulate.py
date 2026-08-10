"""
Reconstruction of operating deflection shapes (ODS) from a continuous
scanning LDV measurement using the polynomial (sideband demodulation)
approach.

When the laser spot moves along a Lissajous trajectory
``x(t) = cos(2 pi fx t + phi_x)``, ``y(t) = cos(2 pi fy t + phi_y)`` over a
structure vibrating harmonically at ``fz``, the measured velocity is

.. math::

    v(t) = \\mathrm{Re}\\left\\{ \\sum_{n,m} C_{nm}\\,
           T_n(x(t))\\, T_m(y(t))\\, e^{i 2 \\pi f_z t} \\right\\},

where :math:`T_n` are Chebyshev polynomials of the first kind. Because
:math:`T_n(\\cos\\theta) = \\cos(n\\theta)`, each coefficient maps onto a
set of sidebands at ``fz ± n fx ± m fy``; the complex coefficient matrix
``C`` is therefore recovered by demodulating the velocity signal at the
sideband frequencies. The ODS in normalized coordinates is the Chebyshev
series with coefficients ``C``.
"""

import warnings

import numpy as np
from numpy.polynomial import chebyshev
from scipy.signal.windows import hann

__all__ = ["demodulate_ods", "evaluate_ods", "align_phase", "reference_phase", "demodulate_ods_1d"]


def _projection(signal, f, fs, window = None):
    """
    Estimate the complex single-sided amplitude of a signal at a specified frequency
    using a windowed Fourier projection.

    :param signal: time-domain signal
    :param f: frequency of interest [Hz]
    :param fs: sampling frequency [Hz]
    :param window: window function applied to the signal
    :return: complex amplitude at frequency ``f`` — magnitude gives the signal
             amplitude and phase gives the corresponding phase [rad]
    """
    if window is None:
        window = np.ones_like(signal)
    n = np.arange(len(signal))
    return 2.0 / window.sum() * np.sum(window * signal * np.exp(-2j * np.pi * f / fs * n))


def reference_phase(signal, f, fs, window=None):
    """
    Phase of a harmonic reference signal at frequency ``f``.

    Used to obtain the actual mirror phases from the measured feedback
    signals, which compensates the inertial lag of the scanning mirrors.

    :param signal: measured reference (mirror feedback) signal
    :param f: reference frequency [Hz]
    :param fs: sampling frequency [Hz]
    :param window: optional window; default periodic Hann
    :return: phase [rad]
    """
    if window is None:
        window = hann(len(signal), sym=False)
    return np.angle(_projection(signal, f, fs, window))

def demodulate_ods_1d(
    velocity, fs, fn, fx, order=10, phi_x=0.0
):
    """
    Estimate Chebyshev coefficients of one or more scanned modes
    using linear least squares.

    The measurement model is

        X = Phi @ T

    where ``T`` contains the temporal sideband basis functions and
    ``Phi`` contains the Chebyshev coefficients of each mode.

    For multiple modes, the temporal basis is stacked vertically by
    mode. The resulting coefficient matrix is then split into one
    Chebyshev coefficient array per mode.

    :param velocity: measured velocity signal(s), shape
        ``(locations, time)`` or simply ``(time,)``
    :param fs: sampling frequency [Hz]
    :param fn: response frequency [Hz], scalar or array
    :param fx: scan frequency [Hz]
    :param order: maximum Chebyshev order, scalar or array matching
        ``fn``
    :param phi_x: scan-path phase [rad]
    :return: Chebyshev coefficients. For a single mode, returns an
        array of shape ``(locations, P+1)`` or ``(P+1,)`` for a
        single measurement location. For multiple modes, returns a
        list containing one such array per mode.
    """
    velocity = np.asarray(velocity)

    if velocity.ndim == 1:
        velocity = velocity[None, :]
        unpack_location = True
    else:
        unpack_location = False

    fn = np.atleast_1d(fn)
    order = np.atleast_1d(order)

    if len(fn) != len(order):
        raise ValueError(
            "`fn` and `order` must have the same length."
        )

    n_locations, n_samples = velocity.shape
    t = np.arange(n_samples) / fs

    temporal_blocks = []

    for f_mode, P in zip(fn, order):
        P = int(P)
        p = np.arange(P + 1)

        omega_pos = 2 * np.pi * (f_mode + p * fx)
        omega_neg = 2 * np.pi * (f_mode - p * fx)

        T_mode = 0.5 * (
            np.cos(
                omega_pos[:, None] * t
                + p[:, None] * phi_x
            )
            +
            np.cos(
                omega_neg[:, None] * t
                - p[:, None] * phi_x
            )
        )

        temporal_blocks.append(T_mode)

    # Stack temporal bases vertically by mode:
    #
    # T = [T_mode_1]
    #     [T_mode_2]
    #       ...
    T = np.vstack(temporal_blocks)

    # Solve
    #
    # X.T = T.T @ Phi.T
    #
    # Result:
    # Phi.shape = (locations, sum(P + 1))
    Phi = np.linalg.lstsq(
        T.T,
        velocity.T,
        rcond=None
    )[0].T

    # Split the coefficient matrix into one block per mode.
    Phi_vec = []
    column = 0

    for P in order:
        n_coefficients = int(P) + 1

        phi_mode = Phi[:, column:column + n_coefficients]
        Phi_vec.append(phi_mode)

        column += n_coefficients

    # For a single measurement location, remove the location dimension.
    if unpack_location:
        Phi_vec = [phi[0] for phi in Phi_vec]

    # Preserve the convenient scalar-input behaviour.
    if len(Phi_vec) == 1:
        return Phi_vec[0]

    return Phi_vec


def demodulate_ods(velocity, x, y, fs, fx, fy, fz, order=10):
    """
    Recover the complex Chebyshev coefficient matrix of the ODS.

    :param velocity: measured LDV velocity signal
    :param x: measured (or commanded) normalized x mirror signal
    :param y: measured (or commanded) normalized y mirror signal
    :param fs: sampling frequency [Hz]
    :param fx: x scan frequency [Hz]
    :param fy: y scan frequency [Hz]
    :param fz: response (excitation) frequency [Hz]
    :param order: maximum Chebyshev order in each direction
    :return: complex array ``C`` of shape ``(order+1, order+1)``;
        ``C[n, m]`` multiplies ``T_n(x) T_m(y)``

    .. note::
        The scan and response frequencies must be chosen such that the
        sidebands ``fz ± n fx ± m fy`` are distinct for all coefficient
        pairs up to ``order``; overlapping sidebands are detected and
        reported with a warning.
    """
    velocity = np.asarray(velocity)
    n_samples = len(velocity)
    window = hann(n_samples, sym=False)

    phi_x = reference_phase(x, fx, fs, window)
    phi_y = reference_phase(y, fy, fs, window)

    df = fs / n_samples
    seen = {}
    coefficients = np.zeros((order + 1, order + 1), dtype=complex)
    for n in range(order + 1):
        for m in range(order + 1):
            signs_x = (1, -1) if n else (1,)
            signs_y = (1, -1) if m else (1,)
            count = len(signs_x) * len(signs_y)
            estimates = []
            for sx in signs_x:
                for sy in signs_y:
                    fc = fz + sx * n * fx + sy * m * fy
                    psi = sx * n * phi_x + sy * m * phi_y
                    key = round(abs(fc) / df)
                    if key in seen and seen[key] != (n, m):
                        warnings.warn(
                            f"Sideband of coefficient ({n}, {m}) overlaps with "
                            f"coefficient {seen[key]} at {abs(fc):.3f} Hz; "
                            "choose different scan frequencies.")
                    seen[key] = (n, m)
                    amplitude = _projection(velocity, abs(fc), fs, window)
                    if fc < 0:
                        amplitude = np.conj(amplitude)
                    estimates.append(count * amplitude * np.exp(-1j * psi))
            coefficients[n, m] = np.mean(estimates)
    return coefficients


def evaluate_ods(coefficients, resolution=100):
    """
    Evaluate the ODS Chebyshev series on a regular grid.

    :param coefficients: complex coefficient matrix from :func:`demodulate_ods`
    :param resolution: number of grid points per direction
    :return: ``(x_grid, y_grid, z)`` where ``z`` is the complex deflection
        shape; all arrays have shape ``(resolution, resolution)``
    """
    points = np.linspace(-1, 1, resolution)
    x_grid, y_grid = np.meshgrid(points, points, indexing="ij")
    z = chebyshev.chebgrid2d(points, points, coefficients)
    return x_grid, y_grid, z


def align_phase(coefficients):
    """
    Rotate the complex coefficients so the dominant coefficient is real.

    The demodulated coefficients share an arbitrary global phase (the
    response phase relative to the acquisition start). Removing it makes the
    real part of the reconstruction directly comparable to a real mode shape.

    :param coefficients: complex coefficient matrix
    :return: phase-aligned coefficient matrix
    """
    coefficients = np.asarray(coefficients)
    dominant = coefficients.flat[np.argmax(np.abs(coefficients))]
    return coefficients * np.exp(-1j * np.angle(dominant))
