pyCSLDV
-------

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

This package is developed on the `SDyPy template
<https://github.com/sdypy/sdypy_template_project>`_ and is part of the
wider `SDyPy <https://github.com/sdypy>`_ ecosystem effort.

Origin and attribution
----------------------

pyCSLDV is an independent Python reimplementation of the *Continuous
Scanning Laser Doppler Vibrometry (CSLDV) Vibration Measurement &
Simulation Suite* (LabVIEW/MATLAB) by Joshua Bartlett and Pablo Tarazaga,
FAST Laboratory, Texas A&M University, published under the CC-BY-4.0
license at `zenodo.org/records/21301126
<https://zenodo.org/records/21301126>`_ (DOI: `10.5281/zenodo.21301126
<https://doi.org/10.5281/zenodo.21301126>`_).

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
- the scope is narrower: the LabVIEW acquisition and hardware control, the
  Rayleigh-Ritz model and the ODS rotation of the original suite have no
  counterpart here.

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
plate mode measured with 5 % noise is reconstructed with MAC > 0.99, and
the result is checked against the transcribed MATLAB reference
implementation. Those tests validate the processing chain against the
original method, not against physical measurements; the package is at an
alpha stage and results should be validated against your own reference
data before being relied upon.

Installation
------------

.. code-block:: console

    $ pip install pyCSLDV

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
    t, x, y, velocity = pycsldv.simulate_response(shape, fz, fx, fy, fs, n,
                                                  noise_std=0.05)

    # ODS reconstruction by sideband demodulation
    C = pycsldv.demodulate_ods(velocity, x, y, fs, fx, fy, fz, order=8)
    C = pycsldv.align_phase(C)

    # Compare with the exact shape
    X, Y, Z = pycsldv.evaluate_ods(C, resolution=100)
    print(pycsldv.mac(shape(X, Y), Z.real))

    # Visualize
    pycsldv.plot_ods(C)

For measured data, pass the recorded velocity and mirror feedback signals
to ``demodulate_ods`` directly — the mirror phases (inertial lag) are
estimated from the feedback signals themselves.

The same example can be run from the project base directory with:

.. code-block:: console

    $ python -m examples.simulate_and_reconstruct

For a guided walk-through of the complete workflow — trajectory design,
sideband spectrum, ODS reconstruction and MAC validation — see the
`Showcase notebook <Showcase.ipynb>`_.

References
----------

- J. Bartlett, P. Tarazaga: Continuous Scanning Laser Doppler Vibrometry
  (CSLDV) Vibration Measurement & Simulation Suite, Zenodo, 2026,
  DOI: 10.5281/zenodo.21301126.
- A. B. Stanbridge, D. J. Ewins: Modal testing using a scanning laser
  Doppler vibrometer, Mechanical Systems and Signal Processing 13(2),
  1999, 255–270.
- S. Rothberg et al.: An international review of laser Doppler vibrometry:
  Making light work of vibration measurement, Optics and Lasers in
  Engineering 99, 2017, 11–22.
