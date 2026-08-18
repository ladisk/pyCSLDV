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
        """The normalized domain corresponds to the scanned surface: a
        71 mm x 14.5 mm region, as set up on the front panel."""
        from examples.csldv_suite_export import read_time_response

        _, x_mm, y_mm, _ = read_time_response(MEASUREMENT, fx=FX, fy=FY,
                                              normalize=False)
        _, _, x_amplitude = pycsldv.normalize_scan(x_mm, FX, FS)
        _, _, y_amplitude = pycsldv.normalize_scan(y_mm, FY, FS)

        assert 2 * x_amplitude == pytest.approx(71.0, abs=0.5)
        assert 2 * y_amplitude == pytest.approx(14.5, abs=0.5)

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
