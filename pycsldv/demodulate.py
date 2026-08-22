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
from math import comb

import numpy as np
from numpy.polynomial import chebyshev
from scipy.signal.windows import hann

__all__ = ["demodulate_ods", "demodulate_ods_1d", "demodulate_ods_2d",
           "evaluate_ods", "align_phase", "reference_phase", "rotate_ods"]


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

#: Condition number above which the fit is reported as poorly determined.
_CONDITION_LIMIT = 100.0


def _solve(design, velocity, fs, scan_frequencies):
    """
    Solve the least-squares system, and say so when it cannot be solved.

    ``numpy.linalg.lstsq`` returns the minimum-norm solution of a
    rank-deficient system rather than refusing it, so a fit that cannot be
    determined by the data comes back as a plausible-looking set of
    coefficients. The usual cause is a record that does not cover the scan:
    until the laser has been everywhere, the basis functions are not linearly
    independent over the samples that exist.

    :param design: design matrix of the system
    :param velocity: right-hand side, shape ``(locations, n_samples)``
    :param fs: sampling frequency [Hz], for the diagnostic only
    :param scan_frequencies: the scan frequencies [Hz], for the diagnostic only
    :return: the solution, shape ``(locations, n_unknowns)``
    """
    from .scan import scan_period      # imported here: scan imports from this module

    solution, _, rank, singular = np.linalg.lstsq(design, velocity.T, rcond=None)
    unknowns = design.shape[1]
    if rank == unknowns and singular[-1] > 0:
        # The singular values come free with the solve. A scan that covers the
        # surface conditions this system at a few units; the threshold sits
        # where poor coverage starts to matter in practice. Measured on a
        # plate mode with 5 % noise: at a condition number of 15 the
        # reconstruction still reaches MAC 0.999, at 624 it falls to 0.52.
        condition = singular[0] / singular[-1]
        if condition > _CONDITION_LIMIT:
            warnings.warn(
                f"the least-squares system is ill-conditioned (condition "
                f"number {condition:.3g}), so noise in the measurement is "
                f"strongly amplified in the coefficients; a scan that covers "
                f"the surface conditions it at a few units. The shape may be "
                f"reconstructed from a noise-free simulation and still be "
                f"unusable from a measurement.")
    if rank < unknowns:
        duration = design.shape[0] / fs
        closure = scan_period(*scan_frequencies) if len(scan_frequencies) > 1 \
            else 1.0 / scan_frequencies[0] if scan_frequencies[0] else None
        hint = ""
        if closure is not None and duration < closure:
            hint = (f" The record is {duration:.3g} s long and the scan closes "
                    f"after {closure:.3g} s, so the laser has not yet been "
                    f"everywhere on the surface.")
        warnings.warn(
            f"the least-squares system is rank-deficient ({rank} of {unknowns} "
            f"unknowns are determined by the data), so the coefficients are the "
            f"minimum-norm solution and not a reconstruction of the shape."
            + hint)
    return solution.T


def demodulate_ods_1d(
    velocity, fs, fn=None, fx=0.0, order=10, phi_x=0.0, poles=None
):
    """
    Estimate complex Chebyshev coefficients using a global linear system,
    incorporating complex continuous-time poles (damping + frequency).

    :param velocity: real-valued velocity signal, shape (n_samples,) or (locations, n_samples)
    :param fs: sampling frequency [Hz]
    :param fn: response frequencies [Hz], scalar or array. Used if poles is None.
    :param fx: scan frequency [Hz]
    :param order: maximum Chebyshev order, scalar or array matching `poles`/`fn`
    :param phi_x: scan phase offset [rad]
    :param poles: complex poles lambda = sigma + j*omega (rad/s), scalar or array-like
    :return: Complex Chebyshev coefficients per mode.
    """
    velocity = np.asarray(velocity)

    if velocity.ndim == 1:
        velocity = velocity[None, :]  # Shape: (1, n_samples)
        unpack_location = True
    else:
        unpack_location = False

    # Standardize input poles / fn
    if poles is None:
        if fn is None:
            raise ValueError("Either `fn` or `poles` must be provided.")
        fn = np.atleast_1d(fn)
        poles = 2j * np.pi * fn
    else:
        poles = np.atleast_1d(poles)

    order = np.atleast_1d(order)

    if len(order) == 1 and len(poles) > 1:
        order = np.full_like(poles, order[0], dtype=int)

    if len(poles) != len(order):
        raise ValueError("`poles` and `order` must have the same length.")

    n_locations, n_samples = velocity.shape
    t = np.arange(n_samples) / fs

    theta_x = 2.0 * np.pi * fx * t + phi_x

    real_blocks = []
    imag_blocks = []
    mode_info = []

    real_col_idx = 0
    imag_col_idx = 0

    for pole, P in zip(poles, order):
        P = int(P)
        p = np.arange(P + 1)[:, None]  # Shape: (P + 1, 1)

        T_p = np.cos(p * theta_x)
        nc = P + 1

        sigma = pole.real
        omega = pole.imag  # Frequency in rad/s

        # Exponential decay/growth envelope
        decay = np.exp(sigma * t)

        if np.isclose(omega, 0.0):
            # Purely real / DC component (omega = 0)
            real_blocks.append(T_p * decay)

            mode_info.append({
                'nc': nc,
                'is_dc': True,
                'real_col': real_col_idx,
                'imag_col': None
            })
            real_col_idx += nc
        else:
            # AC component with damping
            real_blocks.append(T_p * decay * np.cos(omega * t))
            imag_blocks.append(T_p * decay * np.sin(omega * t))

            mode_info.append({
                'nc': nc,
                'is_dc': False,
                'real_col': real_col_idx,
                'imag_col': imag_col_idx
            })
            real_col_idx += nc
            imag_col_idx += nc

    # Stack only active basis blocks
    A_real_all = np.vstack(real_blocks)

    if imag_blocks:
        A_imag_all = np.vstack(imag_blocks)
        # v(t) = Re(A) @ Re(C) - Im(A) @ Im(C)
        A_real_system = np.hstack([A_real_all.T, -A_imag_all.T])
    else:
        A_real_system = A_real_all.T

    # Solve full linear least-squares system
    Theta = _solve(A_real_system, velocity, fs, (fx,))

    # Separate real and imaginary solution blocks
    Re_C_all = Theta[:, :real_col_idx]
    Im_C_all = Theta[:, real_col_idx:] if imag_blocks else None

    # Reconstruct mode-by-mode complex coefficient arrays
    Phi_vec = []
    for info in mode_info:
        nc = info['nc']
        r_idx = info['real_col']

        re_mode = Re_C_all[:, r_idx : r_idx + nc]

        if info['is_dc']:
            im_mode = np.zeros_like(re_mode)
        else:
            i_idx = info['imag_col']
            im_mode = Im_C_all[:, i_idx : i_idx + nc]

        Phi_vec.append(re_mode + 1j * im_mode)

    if unpack_location:
        Phi_vec = [phi[0] for phi in Phi_vec]

    return Phi_vec[0] if len(Phi_vec) == 1 else Phi_vec

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

def demodulate_ods_2d(
    velocity, fs, fx, fy, fn=None, order=10,
    phi_x=0.0, phi_y=0.0, poles=None
):
    """
    Estimate complex 2D Chebyshev coefficients using a global linear
    least-squares system, incorporating complex continuous-time poles
    (damping + frequency).

    :param velocity: real-valued velocity signal, shape
        (n_samples,) or (locations, n_samples)
    :param fs: sampling frequency [Hz]
    :param fx: x scan frequency [Hz]
    :param fy: y scan frequency [Hz]
    :param fn: response frequencies [Hz], scalar or array. Used if
        poles is None.
    :param order: Chebyshev order. Can be:
        - scalar: same x/y order for every mode
        - array of length n_modes: same x/y order per mode
        - array of shape (n_modes, 2): [order_x, order_y] per mode
        - array of shape (2, n_modes): accepted and transposed
    :param phi_x: x scan-path phase [rad]
    :param phi_y: y scan-path phase [rad]
    :param poles: complex poles lambda = sigma + j*omega [rad/s],
        scalar or array-like.
    :return: Complex Chebyshev coefficient matrices per mode.
    """
    velocity = np.asarray(velocity)

    if velocity.ndim == 1:
        velocity = velocity[None, :]
        unpack_location = True
    else:
        unpack_location = False

    # Standardize input poles / fn
    if poles is None:
        if fn is None:
            raise ValueError("Either `fn` or `poles` must be provided.")
        fn = np.atleast_1d(fn)
        poles = 2j * np.pi * fn
    else:
        poles = np.atleast_1d(poles)

    n_modes = len(poles)

    # Standardize order to shape (n_modes, 2)
    order = np.asarray(order)

    if order.ndim == 0:
        order = np.full((n_modes, 2), int(order))

    elif order.ndim == 1:
        if len(order) == 1:
            order = np.full((n_modes, 2), int(order[0]))
        elif len(order) == n_modes:
            order = np.column_stack((order, order))
        elif n_modes == 1 and len(order) == 2:
            order = order[None, :]
        else:
            raise ValueError(
                "`order` must be a scalar, have length n_modes, "
                "or contain two values for a single mode."
            )

    elif order.ndim == 2:
        if order.shape == (n_modes, 2):
            pass
        elif order.shape == (2, n_modes):
            order = order.T
        else:
            raise ValueError(
                "`order` must have shape (n_modes, 2) or (2, n_modes)."
            )

    else:
        raise ValueError("`order` must be scalar, 1D, or 2D.")

    order = order.astype(int)

    if np.any(order < 0):
        raise ValueError("Chebyshev orders must be non-negative.")

    n_locations, n_samples = velocity.shape
    t = np.arange(n_samples) / fs

    theta_x = 2.0 * np.pi * fx * t + phi_x
    theta_y = 2.0 * np.pi * fy * t + phi_y

    real_blocks = []
    imag_blocks = []
    mode_info = []

    real_col_idx = 0
    imag_col_idx = 0

    for pole, (Px, Py) in zip(poles, order):
        Px = int(Px)
        Py = int(Py)

        x_basis = np.cos(np.arange(Px + 1)[:, None] * theta_x[None, :])
        y_basis = np.cos(np.arange(Py + 1)[:, None] * theta_y[None, :])
        basis = (x_basis[:, None, :] * y_basis[None, :, :]).reshape((Px + 1) * (Py + 1), n_samples)

        sigma = pole.real
        omega = pole.imag
        decay = np.exp(sigma * t)

        nc = (Px + 1) * (Py + 1)

        if np.isclose(omega, 0.0):
            real_blocks.append(basis * decay[None, :])

            mode_info.append({
                "shape": (Px, Py),
                "nc": nc,
                "is_dc": True,
                "real_col": real_col_idx,
                "imag_col": None,
            })

            real_col_idx += nc

        else:
            real_blocks.append(basis * (decay * np.cos(omega * t))[None, :])
            imag_blocks.append(basis * (decay * np.sin(omega * t))[None, :])

            mode_info.append({
                "shape": (Px, Py),
                "nc": nc,
                "is_dc": False,
                "real_col": real_col_idx,
                "imag_col": imag_col_idx,
            })

            real_col_idx += nc
            imag_col_idx += nc

    A_real = np.vstack(real_blocks).T

    A_imag = np.zeros((n_samples, imag_col_idx))

    imag_block_idx = 0
    for info in mode_info:
        if not info["is_dc"]:
            nc = info["nc"]
            col = info["imag_col"]
            A_imag[:, col:col + nc] = imag_blocks[imag_block_idx].T
            imag_block_idx += 1

    A = np.hstack([A_real, -A_imag])

    Theta = _solve(A, velocity, fs, (fx, fy))

    Re_C_all = Theta[:, :real_col_idx]
    Im_C_all = Theta[:, real_col_idx:]

    Phi_vec = []

    for info in mode_info:
        Px, Py = info["shape"]
        nc = info["nc"]
        r_idx = info["real_col"]

        re_mode = Re_C_all[:, r_idx:r_idx + nc]

        if info["is_dc"]:
            im_mode = np.zeros_like(re_mode)
        else:
            i_idx = info["imag_col"]
            im_mode = Im_C_all[:, i_idx:i_idx + nc]

        phi_mode = (re_mode + 1j * im_mode).reshape(n_locations, Px + 1, Py + 1)
        Phi_vec.append(phi_mode)

    if unpack_location:
        Phi_vec = [phi[0] for phi in Phi_vec]

    return Phi_vec[0] if len(Phi_vec) == 1 else Phi_vec

def evaluate_ods(coefficients, resolution=100):
    """
    Evaluate the ODS Chebyshev series on a regular grid.

    For a 1D coefficient vector of shape ``(P + 1,)``, returns
    ``(x, z)`` where both arrays have shape ``(resolution,)``.

    For a 2D coefficient matrix of shape ``(Px + 1, Py + 1)``, returns
    ``(x_grid, y_grid, z)`` where all arrays have shape
    ``(resolution, resolution)``.

    A multimodal or multi-sensor reconstruction is a list of matrices or an
    array stacked along a leading axis; each is evaluated in turn and the
    results are returned as a list.

    .. note::
        A line scan measured by several sensors comes back as
        ``(locations, P + 1)``, which is indistinguishable from the
        coefficients of a single surface. Evaluate those one location at a
        time.

    :param coefficients: complex Chebyshev coefficients from
        :func:`demodulate_ods`, :func:`demodulate_ods_1d` or
        :func:`demodulate_ods_2d`
    :param resolution: number of grid points per direction
    """
    if isinstance(coefficients, (list, tuple)):
        return [evaluate_ods(c, resolution) for c in coefficients]

    coefficients = np.asarray(coefficients)
    if coefficients.ndim > 2:
        return [evaluate_ods(c, resolution) for c in coefficients]

    points = np.linspace(-1, 1, resolution)

    if coefficients.ndim == 1:
        z = chebyshev.chebval(points, coefficients)
        return points, z

    if coefficients.ndim == 2:
        x_grid, y_grid = np.meshgrid(points, points, indexing="ij")
        z = chebyshev.chebgrid2d(points, points, coefficients)
        return x_grid, y_grid, z

    raise ValueError(
        "`coefficients` must be a 1D or 2D array."
    )

def align_phase(coefficients):
    """
    Rotate the complex coefficients so the dominant coefficient is real.

    The demodulated coefficients share an arbitrary global phase (the
    response phase relative to the acquisition start). Removing it makes the
    real part of the reconstruction directly comparable to a real mode shape.

    :param coefficients: complex coefficient matrix
    :return: phase-aligned coefficient matrix
    """

    if isinstance(coefficients, (list, tuple)):
        return [align_phase(c) for c in coefficients]

    coefficients = np.asarray(coefficients)
    dominant = coefficients.flat[np.argmax(np.abs(coefficients))]
    return coefficients * np.exp(-1j * np.angle(dominant))


def _map_axis(func, matrix, axis):
    """
    Apply a 1D coefficient transform along one axis of a coefficient matrix.

    :func:`numpy.apply_along_axis` cannot be used because the conversions of
    :mod:`numpy.polynomial` trim trailing zeros and so do not preserve the
    length; the result is padded back instead.
    """
    matrix = np.moveaxis(np.asarray(matrix), axis, 0)
    transformed = np.zeros_like(matrix)
    for column in range(matrix.shape[1]):
        converted = func(matrix[:, column])
        transformed[:len(converted), column] = converted
    return np.moveaxis(transformed, 0, axis)


def rotate_ods(coefficients, angle, aspect=1.0):
    """
    Express a reconstructed ODS in a rotated coordinate frame.

    The scan axes need not line up with the edges of the measured object: if
    it is mounted askew, the reconstruction comes out in the frame of the
    scan and has to be turned onto the object's own axes before it can be
    compared with a model or with a measurement of a differently mounted
    specimen. This rotates the shape itself, by substituting the rotated
    coordinates into the Chebyshev series and expanding again, so the result
    is exact -- no resampling and no interpolation.

    The new frame is the old one rotated by ``angle``, so a shape whose
    features run along the old x axis comes back running at ``angle`` to the
    new one.

    :param coefficients: complex (or real) Chebyshev coefficient matrix, as
        returned by :func:`demodulate_ods`. A list of matrices or an array
        stacked along a leading axis -- a multimodal or multi-sensor
        reconstruction -- has every shape in it turned by the same angle
    :param angle: rotation angle [rad]
    :param aspect: ratio of the x extent of the scanned region to its y
        extent, e.g. the ratio of the two amplitudes returned by
        :func:`pycsldv.normalize_scan`. The default of ``1.0`` rotates the
        normalized domain itself, which is a rotation of the physical surface
        only when the scan is square; pass the true ratio to rotate the
        physical surface, as the normalized coordinates compress the two
        directions differently.
    :return: coefficient matrix in the rotated frame, of shape
        ``(n + m + 1, n + m + 1)`` for an input of shape ``(n + 1, m + 1)``:
        a rotation mixes the two directions, so the tensor-product degree
        grows even though the total degree does not

    .. warning::
        The rotated domain is not the domain the shape was reconstructed on:
        the corners of ``[-1, 1] x [-1, 1]`` turn outside it, where the
        series extrapolates and a high-order one diverges quickly. The
        overhang is ``|sin(angle)|`` in the isotropic case but ``aspect``
        times that across the short direction, so it is the combination of a
        slender scan and a large angle that is unsafe. A warning is issued
        when the rotated domain exceeds the original by more than 5 %.
    """
    if isinstance(coefficients, (list, tuple)):
        return [rotate_ods(c, angle, aspect) for c in coefficients]

    coefficients = np.asarray(coefficients)
    if coefficients.ndim > 2:
        # a multimodal or multi-sensor reconstruction: turn each shape
        return np.stack([rotate_ods(c, angle, aspect) for c in coefficients])
    if coefficients.ndim != 2:
        raise ValueError("a two-dimensional coefficient matrix is required, "
                         f"got {coefficients.ndim} dimension(s); a rotation "
                         "mixes the two directions of the shape")
    if not aspect > 0:
        raise ValueError(f"aspect must be a positive extent ratio, got {aspect}")
    dtype = np.result_type(coefficients.dtype, float)
    cosine, sine = np.cos(angle), np.sin(angle)

    # x = a x' + b y',  y = c x' + d y'  on the normalized domain
    a, b = cosine, -sine / aspect
    c, d = aspect * sine, cosine

    overhang = max(abs(a) + abs(b), abs(c) + abs(d))
    if overhang > 1.05:
        warnings.warn(
            f"the rotated domain extends {100 * (overhang - 1):.0f} % beyond "
            f"[-1, 1], where the Chebyshev series extrapolates; the shape near "
            f"the corners is not to be trusted.")

    # Chebyshev -> monomial, so that the rotated coordinates can be
    # substituted and the powers expanded with the binomial theorem
    powers = _map_axis(chebyshev.cheb2poly, coefficients.astype(dtype), 0)
    powers = _map_axis(chebyshev.cheb2poly, powers, 1)

    n_x, n_y = powers.shape[0] - 1, powers.shape[1] - 1
    rotated = np.zeros((n_x + n_y + 1, n_x + n_y + 1), dtype=dtype)
    binomial = [[comb(n, k) for k in range(n + 1)] for n in range(max(n_x, n_y) + 1)]
    for i in range(n_x + 1):
        for j in range(n_y + 1):
            if powers[i, j] == 0:
                continue
            # (a x' + b y')^i (c x' + d y')^j
            for k in range(i + 1):
                term = binomial[i][k] * a ** k * b ** (i - k) * powers[i, j]
                for l in range(j + 1):
                    rotated[k + l, (i - k) + (j - l)] += \
                        term * binomial[j][l] * c ** l * d ** (j - l)

    rotated = _map_axis(chebyshev.poly2cheb, rotated, 0)
    return _map_axis(chebyshev.poly2cheb, rotated, 1)
