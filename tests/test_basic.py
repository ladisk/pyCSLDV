import numpy as np
import pytest

import pycsldv


def test_version():
    """pycsldv exposes a version attribute"""
    assert hasattr(pycsldv, "__version__")
    assert isinstance(pycsldv.__version__, str)


class TestScan:

    def test_lissajous(self):
        t, x, y = pycsldv.lissajous(fx=1.4, fy=20.0, n_samples=1000, fs=1000.0)
        assert t.shape == x.shape == y.shape == (1000,)
        assert np.max(np.abs(x)) <= 1.0
        assert np.max(np.abs(y)) <= 1.0
        assert x[0] == pytest.approx(1.0)

    def test_lissajous_amplitude_phase(self):
        _, x, y = pycsldv.lissajous(1.0, 2.0, 100, 100.0,
                                    phase_x=np.pi / 2, amplitude=(0.5, 0.8))
        assert x[0] == pytest.approx(0.0, abs=1e-12)
        assert np.max(np.abs(y)) <= 0.8

    def test_scan_period(self):
        # fx = 7/5 Hz, fy = 20 Hz -> gcd = 1/5 Hz -> period 5 s
        assert pycsldv.scan_period(1.4, 20.0) == pytest.approx(5.0)
        assert pycsldv.scan_period(2.0, 3.0) == pytest.approx(1.0)

    def test_drive_signals(self):
        x = np.array([-1.0, 0.0, 1.0])
        vx, vy = pycsldv.drive_signals(x, x, scale=(2.0, 1.0), offset=(0.5, 0.0))
        assert np.allclose(vx, [-1.5, 0.5, 2.5])
        assert np.allclose(vy, x)


class TestNormalizeScan:
    """Preparing a measured mirror feedback signal for the reconstruction."""

    FS = 1000.0
    N = 5000        # 5 s

    def _mirror(self, f, offset, amplitude, phase=0.3):
        t = np.arange(self.N) / self.FS
        return offset + amplitude * np.cos(2 * np.pi * f * t + phase)

    def test_offset_and_amplitude_are_recovered(self):
        """The offset and amplitude returned are those of the component at
        the scan frequency, so they give the physical extent of the scan."""
        position = self._mirror(1.4, offset=3.0, amplitude=35.5)

        normalized, offset, amplitude = pycsldv.normalize_scan(position, 1.4, self.FS)

        assert offset == pytest.approx(3.0, abs=1e-3)
        assert amplitude == pytest.approx(35.5, rel=1e-3)
        # not exactly 1: the samples do not fall on the turning points
        assert np.max(np.abs(normalized)) == pytest.approx(1.0, abs=0.01)

    def test_ignores_content_away_from_the_scan_frequency(self):
        """Cross-axis coupling and noise, which do inflate the extremes of a
        measured mirror signal, leave the normalization alone."""
        clean = self._mirror(20.0, offset=0.7, amplitude=7.25)
        contaminated = clean + self._mirror(1.4, offset=0.0, amplitude=1.2)

        _, offset, amplitude = pycsldv.normalize_scan(contaminated, 20.0, self.FS)

        assert offset == pytest.approx(0.7, abs=0.05)
        assert amplitude == pytest.approx(7.25, rel=0.02)

    def test_dominant_frequency(self):
        signal = self._mirror(20.0, offset=5.0, amplitude=1.0) \
            + 0.3 * self._mirror(1.4, offset=0.0, amplitude=1.0)
        assert pycsldv.dominant_frequency(signal, self.FS) == pytest.approx(20.0)


class TestMAC:

    def test_identical(self):
        shape = np.random.default_rng(0).normal(size=100)
        assert pycsldv.mac(shape, shape) == pytest.approx(1.0)

    def test_orthogonal(self):
        n = np.arange(100)
        assert pycsldv.mac(np.sin(2 * np.pi * n / 100),
                           np.cos(2 * np.pi * n / 100)) == pytest.approx(0.0, abs=1e-12)

    def test_complex(self):
        shape = np.random.default_rng(1).normal(size=50) + \
            1j * np.random.default_rng(2).normal(size=50)
        assert pycsldv.mac(shape, 1j * shape) == pytest.approx(1.0)
