"""
The reconstruction chain on a measured CSLDV dataset.

Every other test runs on simulated data or on a transcription of the original
MATLAB routine. These run the whole chain — reading the export, matching the
mirror channels, normalizing the scan, demodulating — on a real measurement,
and compare the result with the ODS the original LabVIEW/MATLAB suite
reconstructed from the same samples.

The dataset is a 10 s measurement of a rectangular plate excited at 4527 Hz,
provided by Joshua Bartlett (FAST Laboratory, Texas A&M University), to be
published on Zenodo. It is tens of MB and is therefore not part of the
repository: place the exported files in ``examples/data/`` to run these
tests. They skip when the data is absent, so they never run in CI.

The acquisition parameters below are not stored in the exported files; they
are read from ``GUISettings_4527.png``, the screenshot of the front panel
that comes with the measurement.
"""

from pathlib import Path

import numpy as np
import pytest

import pycsldv

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
MEASUREMENT = DATA / "TimeResponse_4527.txt"
REFERENCE = Path(__file__).resolve().parent / "reference" / "measured_ods_4527.npz"

pytestmark = pytest.mark.skipif(
    not MEASUREMENT.exists(),
    reason=f"measured dataset not found; place the exported files in {DATA}")

FS = 100_000.0      # sampling rate [S/s]
FX, FY = 1.4, 20.0  # scan frequencies [Hz]
FZ = 4527.0         # excitation frequency [Hz]
ORDER = 12          # spectral sideband quantity
ROTATION = 1.60368  # sample-alignment angle recorded by the calibration [deg]


@pytest.fixture(scope="module")
def measurement():
    from examples.csldv_suite_export import read_time_response
    return read_time_response(MEASUREMENT, fx=FX, fy=FY)


@pytest.fixture(scope="module")
def coefficients(measurement):
    _, x, y, velocity = measurement
    return pycsldv.align_phase(
        pycsldv.demodulate_ods(velocity, x, y, FS, FX, FY, FZ, order=ORDER))


class TestMeasuredOds:

    def test_matches_the_original_suite(self, coefficients):
        """The shape agrees with the one the original suite reconstructed
        from the same measurement."""
        from examples.csldv_suite_export import read_ods_export

        _, _, reference = read_ods_export(DATA / "4527.xlsx")
        points = np.linspace(-1, 1, reference.shape[0])
        _, _, ods = pycsldv.evaluate_ods(coefficients, resolution=len(points))

        # the workbook's first axis is the 20 Hz axis, which is y here
        assert pycsldv.mac(reference.real, ods.real.T) > 0.99

    def test_matches_the_recorded_reconstruction(self, coefficients):
        """The coefficients have not drifted from the recorded ones. The MAC
        above is a loose criterion; this catches changes in the chain that
        leave the shape recognizable."""
        if not REFERENCE.exists():
            pytest.skip(f"no recorded reconstruction at {REFERENCE}")
        recorded = np.load(REFERENCE)["coefficients"]

        assert coefficients.shape == recorded.shape
        assert np.allclose(coefficients, recorded, atol=1e-6 * np.abs(recorded).max())

    def test_shape_is_essentially_real(self, coefficients):
        """A normal mode: after align_phase the imaginary part is small."""
        assert np.abs(coefficients.imag).max() < 0.05 * np.abs(coefficients).max()


class TestMeasuredScan:

    def test_channels_are_matched_by_frequency(self, measurement):
        """The mirror columns of this export do not follow their labels, so
        the reader has to assign them by their measured frequency."""
        _, x, y, _ = measurement
        assert pycsldv.dominant_frequency(x, FS) == pytest.approx(FX)
        assert pycsldv.dominant_frequency(y, FS) == pytest.approx(FY)

    def test_scan_extent(self):
        """The normalized domain corresponds to the scanned surface. The
        exported mm columns give a 71.0 mm x 14.5 mm region against a sample
        that is nominally 75 mm x 15 mm; the few per cent is the vertical
        standoff distance, which was not precisely measured when the setup
        was calibrated. The reconstruction is unaffected, as it works on the
        normalized domain."""
        from examples.csldv_suite_export import read_time_response

        _, x_mm, y_mm, _ = read_time_response(MEASUREMENT, fx=FX, fy=FY,
                                              normalize=False)
        _, _, x_amplitude = pycsldv.normalize_scan(x_mm, FX, FS)
        _, _, y_amplitude = pycsldv.normalize_scan(y_mm, FY, FS)

        assert 2 * x_amplitude == pytest.approx(71.0, abs=0.5)
        assert 2 * y_amplitude == pytest.approx(14.5, abs=0.5)

    def test_rotation_appears_as_cross_axis_coupling(self):
        """The suite rotates the commanded scan by the sample-alignment angle
        recorded during calibration, so that the scan follows the edges of a
        specimen that is not square to the mirror axes. A rotation mixes the
        two axes, and the mixing is what the mirror feedback shows: each
        channel carries a little of the other one's scan frequency.

        The size of each leak, relative to the axis it leaked from, is the
        tangent of the angle, so the two channels give two independent
        estimates of it. They agree to a fifth of a degree and both come out
        above the recorded angle -- the calibration is known to
        over-compensate. This pins the coupling as geometry: it is neither an
        imperfection of the mirrors nor a fault in the reconstruction.

        A pure rotation would give the same number twice. That the two differ
        by 0.2 deg leaves room for a second-order effect on top of it, the
        14.7 mm spacing between the two mirrors being the obvious candidate,
        but it is far too small to account for the coupling itself.
        """
        from examples.csldv_suite_export import read_time_response

        _, x_mm, y_mm, _ = read_time_response(MEASUREMENT, fx=FX, fy=FY,
                                              normalize=False)

        def amplitude(signal, f):
            """Amplitude of the component of a mirror signal at ``f`` [mm]."""
            return pycsldv.normalize_scan(signal, f, FS)[2]

        # x scans slowly over the long edge, y quickly across the short one
        long_axis, long_leak = amplitude(x_mm, FX), amplitude(y_mm, FX)
        short_axis, short_leak = amplitude(y_mm, FY), amplitude(x_mm, FY)

        # each channel is still dominated by its own scan frequency ...
        assert long_leak < 0.05 * long_axis
        assert short_leak < 0.05 * short_axis

        # ... but the long axis, being five times the short one, contaminates
        # the short channel with a sixth of its own amplitude
        assert long_leak / short_axis == pytest.approx(0.17, abs=0.02)

        # the two estimates of the angle agree, and exceed the recorded one
        from_long = np.degrees(np.arctan2(long_leak, long_axis))
        from_short = np.degrees(np.arctan2(short_leak, short_axis))
        assert from_long == pytest.approx(from_short, abs=0.3)
        assert ROTATION < min(from_long, from_short)
        assert max(from_long, from_short) < 2.5

    def test_commanded_pattern_carries_no_rotation(self):
        """The rotation is absent from the ``LissajousPattern`` export, whose
        cross-terms sit at machine precision. That file holds the coordinates
        of the figure the suite plots for illustration, and the figure is
        drawn before the rotation is applied, so it is not the path the laser
        followed -- unlike the feedback signals used above."""
        from examples.csldv_suite_export import read_scan_path

        horizontal, vertical = read_scan_path(DATA / "LissajousPattern_4527.txt")

        for signal, own, other in ((horizontal, FX, FY), (vertical, FY, FX)):
            # also confirms this export shares the sampling rate of the
            # measurement, which it has to for the frequencies to be read off
            assert pycsldv.dominant_frequency(signal, FS) == pytest.approx(own)
            main = pycsldv.normalize_scan(signal, own, FS)[2]
            cross = pycsldv.normalize_scan(signal, other, FS)[2]
            assert cross < 1e-12 * main

    def test_carrier_is_symmetric(self, measurement):
        """The sidebands are symmetric about the excitation frequency, which
        confirms that the frequency read off the front panel is the one the
        structure was driven at."""
        _, _, _, velocity = measurement
        spectrum = 2 * np.abs(np.fft.rfft(velocity)) / len(velocity)
        df = FS / len(velocity)

        for n in (1, 2, 4):
            lower = spectrum[round((FZ - n * FX) / df)]
            upper = spectrum[round((FZ + n * FX) / df)]
            assert lower == pytest.approx(upper, rel=0.02)
