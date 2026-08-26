pyCSLDV
-------

|Testing| |Python| |License| |Status|

Continuous scanning laser Doppler vibrometry (CSLDV) in Python: Lissajous
scan-trajectory design, virtual CSLDV experiments and reconstruction of
operating deflection shapes (ODS) using the polynomial (sideband
demodulation) approach.

In a CSLDV measurement the laser spot of the vibrometer sweeps
continuously over the vibrating surface. For a harmonic response at
frequency ``fz`` and a Lissajous scan with mirror frequencies ``fx`` and
``fy``, the deflection shape modulates the measured velocity and appears
as sidebands at ``fz ± n fx ± m fy``. Demodulating these sidebands yields
the coefficients of a two-dimensional Chebyshev series that describes the
full-field ODS from a single-point sensor measurement.

The coefficients are recovered either by projecting the velocity onto each
sideband in turn, or by fitting every mode at once as one global linear
least-squares system. The second form carries complex poles instead of
frequencies, so it also covers damped and complex modes, modes whose
sidebands overlap, several sensors, and line scans.

This package is developed on the `SDyPy template
<https://github.com/sdypy/sdypy_template_project>`_ and is part of the
wider `SDyPy <https://github.com/sdypy>`_ ecosystem effort.

Origin and attribution
----------------------

pyCSLDV is an independent Python reimplementation of the *Continuous
Scanning Laser Doppler Vibrometry (CSLDV) Vibration Measurement &
Simulation Suite* (LabVIEW/MATLAB) by Joshua Bartlett and Pablo Tarazaga,
FAST Laboratory, Texas A&M University, published under the CC-BY-4.0
license at `zenodo.org/records/22032252
<https://zenodo.org/records/22032252>`_ (DOI: `10.5281/zenodo.22032252
<https://doi.org/10.5281/zenodo.22032252>`_; the concept DOI
`10.5281/zenodo.21301125 <https://doi.org/10.5281/zenodo.21301125>`_ always
resolves to the latest version).

The measurement principle, the sideband-demodulation processing chain and
the compensation of the mirror inertial lag follow the original suite, and
the reconstruction is verified against a transcription of its
``ComputeODS.m`` (see ``tests/test_matlab_reference.py``). The code is not
a translation of the MATLAB routines:

- the demodulation is formulated in complex arithmetic, which removes the
  explicit magnitude/phase quadrant and unwrapping heuristics;
- the shape is kept in the Chebyshev basis instead of being converted to
  monomial coefficients, so the sideband normalization is applied
  explicitly rather than folded into a hard-coded transformation matrix;
- the global phase is normalized on the dominant coefficient
  (``align_phase``) rather than on the response baseband phase;
- overlapping sidebands are detected and reported;
- besides the per-sideband projection, the coefficients of every mode can be
  fitted in a single linear least-squares system (``demodulate_ods_2d``,
  ``demodulate_ods_1d``), which takes complex poles rather than real
  frequencies and so reconstructs damped and complex modes, resolves modes
  whose sidebands fall on top of each other, and fits several sensors and
  line scans;
- the ODS rotation acts on the Chebyshev coefficients, so turning a shape
  onto another set of axes is exact rather than a resampling of a grid, and
  it is applied to the physical surface or to the normalized domain as the
  caller chooses (``rotate_ods``); the angle can be measured from the mirror
  feedback (``scan_rotation``) instead of being taken from the calibration
  record;
- the scope is narrower: the LabVIEW acquisition and hardware control and
  the Rayleigh-Ritz model of the original suite have no counterpart here.

If you use pyCSLDV in academic work, please cite the original suite as
well; complete citation metadata (with ORCIDs) is provided in the
``CITATION.cff`` file.

Development note
----------------

The initial Python implementation in this repository was written with
Claude (Anthropic), working from the published source of the original
suite. It has not yet been reviewed by the authors of that suite.

The reconstruction chain is covered end to end by the test suite: an exact
Chebyshev shape is recovered from a simulated measurement to within
``1e-8``, mirror and response phases are verified to be compensated, a
plate mode measured with 5 % noise is reconstructed with MAC > 0.99, a
damped complex mode is recovered from its pole, two modes whose sidebands
coincide are separated, and the result is checked against the transcribed
MATLAB reference implementation.

The chain is also run on a real measurement — the 4527 Hz plate scan
published with the original suite — and agrees with the ODS that suite
reconstructed from the same samples at MAC 0.9905, and at 0.9955 once the
shape is turned onto the mirror axes. That is a comparison against one
measurement processed by one other implementation, not a metrological
validation; the package is at an alpha stage and results should be
validated against your own reference data before being relied upon.

Installation
------------

pyCSLDV is not on PyPI yet; install it from the repository:

.. code-block:: console

    $ pip install git+https://github.com/ladisk/pyCSLDV.git

For development, clone the repository and install in editable mode:

.. code-block:: console

    $ uv venv
    $ uv pip install -e ".[dev]"

Usage
-----

A complete virtual experiment — simulate the LDV velocity signal for a
known deflection shape, reconstruct the ODS and quantify the agreement:

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

    # Compare with the exact shape
    X, Y, Z = pycsldv.evaluate_ods(C, resolution=100)
    print(pycsldv.mac(shape(X, Y), Z.real))

    # Visualize
    pycsldv.plot_ods(C)

Several modes, damping, several sensors, line scans
---------------------------------------------------

``demodulate_ods`` above projects the velocity onto each sideband in turn. It
takes the recorded mirror signals, so the mirror phases — the inertial lag of
the galvos — are estimated from the signals themselves.

``demodulate_ods_2d`` instead fits

.. code-block:: text

    v(t) = sum_k Re{ C_k(x(t), y(t)) exp(lambda_k t) }

as one global linear least-squares system. It rebuilds the trajectory from
``fx``, ``fy`` and the mirror phases, which therefore have to be passed as
``phi_x`` and ``phi_y`` (``reference_phase(x, fx, fs)`` measures them from
recorded feedback). In exchange it takes complex poles
``lambda = sigma + i omega`` instead of real frequencies, and any number of
modes and sensors at once:

.. code-block:: python

    # one undamped mode and one ringing down, fitted together,
    # to different orders
    poles = [2j * np.pi * 500.0, -0.5 + 2j * np.pi * 502.8]
    C_first, C_second = pycsldv.demodulate_ods_2d(velocity, fs, fx, fy,
                                                  poles=poles, order=[6, 8])

    # the three heads of a 3D scanning vibrometer: (locations, n_samples) in,
    # one coefficient matrix per location out
    C = pycsldv.demodulate_ods_2d(np.vstack([v_x, v_y, v_z]), fs, fx, fy,
                                  fn=500.0, order=6)

    # a line scan: one Chebyshev variable; evaluate_ods and plot_ods
    # recognize it from the shape of the coefficients
    C = pycsldv.demodulate_ods_1d(velocity, fs, fn=500.0, fx=fx, order=5)
    points, z = pycsldv.evaluate_ods(C, resolution=200)

The results follow the input: several modes come back as a list, several
sensors as an array stacked along a leading axis, and ``evaluate_ods``,
``plot_ods`` and ``rotate_ods`` take either.

A fit that the measurement cannot determine is reported rather than returned
quietly. ``numpy``'s least-squares solver answers a rank-deficient system
with its minimum-norm solution, which looks like a set of coefficients and is
not one, so both the rank and the conditioning are checked: a record that
does not cover the scan, or one that barely does, produces a warning naming
the scan period or the condition number.

Fitting the modes together matters when they are close enough for the
sidebands of one to land on those of another. The spacing that hurts is not
"a few Hz" but a multiple of a scan frequency: a second mode ``2 fx`` away
puts its carrier exactly on the first mode's ``n = 2`` sideband, where the
projection cannot tell the two apart while the global fit still recovers
both.

The design matrix is ``n_samples`` by ``2 sum_k (Px + 1)(Py + 1)``, which for
a long record is larger than everything else in the problem put together, so
it is never held whole: it is built a block of rows at a time and only the
normal equations are accumulated. Those are as wide as the unknowns and no
wider, so the memory does not follow the length of the record. Forming them
squares the condition number of the system, which is one more reason the fit
reports how well the scan determines it.

Measured data
-------------

For measured data, pass the recorded velocity and mirror feedback signals
to ``demodulate_ods`` directly — the mirror phases (inertial lag) are
estimated from the feedback signals themselves. pyCSLDV reads no file
formats of its own; the two steps a measurement does need before the
reconstruction are mapping each mirror signal onto the normalized domain and
telling the channels apart by their scan frequency:

.. code-block:: python

    x, x_centre, x_amplitude = pycsldv.normalize_scan(x_mm, fx, fs)
    pycsldv.dominant_frequency(x_mm, fs)   # which mirror is this?

If the object was mounted askew and the scan was rotated to follow its edges,
the reconstruction already comes out on the object's own axes, because only
the phase of each mirror at its scan frequency is used. The angle is needed
only to step back to the mirror axes, and it can be measured from the
feedback signals rather than trusted to the calibration record:

.. code-block:: python

    angle = pycsldv.scan_rotation(x, y, fx, fy, fs)
    C_lab = pycsldv.rotate_ods(C, -angle)

An adapter for the export format of the original LabVIEW/MATLAB suite is
provided as an example rather than as part of the package, in
``examples/csldv_suite_export.py``.

Examples and notebooks
----------------------

The bundled example can be run from the project base directory with:

.. code-block:: console

    $ python -m examples.simulate_and_reconstruct

For a guided walk-through of the complete workflow — trajectory design,
sideband spectrum, ODS reconstruction and MAC validation — see the
`Showcase notebook <Showcase.ipynb>`_. The same workflow on a real
measurement, compared with the ODS the original suite reconstructed from the
same samples, is in `Showcase_measured.ipynb <Showcase_measured.ipynb>`_.
That dataset is published with the original suite and is not part of this
repository: it is 80 MB, and both the notebook and
``tests/test_measured_reference.py`` download it from Zenodo the first time
they need it, into ``examples/data/``. Set ``PYCSLDV_NO_DOWNLOAD=1`` to
forbid that — the tests then skip instead of fetching 52 MB, which is what a
continuous-integration run wants — and ``PYCSLDV_DATA`` to keep the files
somewhere else. The download is also skipped, rather than failed, when there
is no network.

References
----------

- J. Bartlett, P. Tarazaga: Continuous Scanning Laser Doppler Vibrometry
  (CSLDV) Vibration Measurement & Simulation Suite, Zenodo, 2026,
  DOI: 10.5281/zenodo.21301126.
- A. B. Stanbridge, D. J. Ewins: Modal testing using a scanning laser
  Doppler vibrometer, Mechanical Systems and Signal Processing 13(2),
  1999, 255–270.
- L. Mignanelli, P. Chiariotti, P. Castellini, M. Martarelli: Blind 
  Identification of Operational Deflection Shapes from Continuous Scanning 
  Laser Doppler Vibrometry Data, Sensors and Instrumentation 5, 2016, 
  105–111.
- S. Rothberg et al.: An international review of laser Doppler vibrometry:
  Making light work of vibration measurement, Optics and Lasers in
  Engineering 99, 2017, 11–22.
- D. Di Maio et al.: Continuous Scanning Laser Vibrometry: A raison d’être
 and applications to vibration measurements, Mechanical Systems and Signal 
 Processing 156, 2021, 107573.


.. |Testing| image:: https://github.com/ladisk/pyCSLDV/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/ladisk/pyCSLDV/actions/workflows/python-package.yml
   :alt: Testing

.. |Python| image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10, 3.11, 3.12

.. |License| image:: https://img.shields.io/badge/license-MIT-yellow.svg
   :target: https://github.com/ladisk/pyCSLDV/blob/main/License
   :alt: MIT license

.. |Status| image:: https://img.shields.io/badge/status-alpha-orange.svg
   :target: https://github.com/ladisk/pyCSLDV#origin-and-attribution
   :alt: Alpha
