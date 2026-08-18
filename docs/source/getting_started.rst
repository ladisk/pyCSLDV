Getting started
===============

Installation
------------

.. code-block:: console

    $ pip install pyCSLDV

Background
----------

In a continuous scanning LDV (CSLDV) measurement the laser spot sweeps
continuously over the vibrating surface instead of dwelling at discrete
points. With a Lissajous scan the two galvo mirrors are driven
harmonically at frequencies :math:`f_x` and :math:`f_y`, so the
normalized laser position is

.. math::

    x(t) = \cos(2 \pi f_x t + \varphi_x), \qquad
    y(t) = \cos(2 \pi f_y t + \varphi_y).

If the structure responds harmonically at :math:`f_z` with an operating
deflection shape expressed as a two-dimensional Chebyshev series
:math:`\sum_{n,m} C_{nm} T_n(x) T_m(y)`, the measured velocity becomes

.. math::

    v(t) = \mathrm{Re}\left\{ \sum_{n,m} C_{nm}\,
           T_n(x(t))\, T_m(y(t))\, e^{i 2 \pi f_z t} \right\}.

Because :math:`T_n(\cos\theta) = \cos(n\theta)`, every coefficient
appears as a set of sidebands at :math:`f_z \pm n f_x \pm m f_y`.
Demodulating the velocity signal at these frequencies — with the mirror
phases estimated from the measured feedback signals — recovers the
complex coefficients :math:`C_{nm}` and therefore the full-field ODS
from a single-point sensor.

Minimal example
---------------

.. code-block:: python

    import pycsldv

    fs = 25000          # sampling frequency [Hz]
    fx, fy = 1.4, 8.0   # scan frequencies [Hz]
    fz = 2000.0         # response frequency [Hz]
    n = int(10 * fs)    # 10 s measurement

    # Virtual measurement of a plate mode with 5 % noise
    shape = pycsldv.plate_mode(2, 3)
    t, x, y = pycsldv.lissajous(fx, fy, n, fs)
    velocity = pycsldv.simulate_response(shape, fz, x, y, fs, noise_std=0.05)

    # ODS reconstruction by sideband demodulation
    C = pycsldv.demodulate_ods(velocity, x, y, fs, fx, fy, fz, order=8)
    C = pycsldv.align_phase(C)

    # Compare with the exact shape and visualize
    X, Y, Z = pycsldv.evaluate_ods(C, resolution=100)
    print(pycsldv.mac(shape(X, Y), Z.real))
    pycsldv.plot_ods(C)

Choosing the scan frequencies
-----------------------------

Two practical constraints apply:

* the sidebands :math:`f_z \pm n f_x \pm m f_y` must be distinct for all
  coefficients up to the chosen ``order`` (``demodulate_ods`` warns when
  they overlap — avoid, e.g., integer ratios :math:`f_y / f_x`);
* the measurement time should span an integer number of Lissajous
  closure periods (see :func:`pycsldv.scan.scan_period`) and be long
  enough that neighbouring sidebands are separated by at least two
  frequency-resolution bins.
