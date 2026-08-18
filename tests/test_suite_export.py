"""
The ``examples/csldv_suite_export.py`` adapter for the export format of the
original CSLDV suite.

The adapter is not part of the installed package, but it is what the measured
reference test and ``Showcase_measured.ipynb`` read their data with, so it is
covered here. The measured dataset itself is not in the repository, so the
fixtures below write miniature files in the same format: tab-separated text
with a single header line, and a workbook with one numeric column per sheet.
"""

import zipfile

import numpy as np
import pytest

import pycsldv
from examples import csldv_suite_export as suite

FS = 1000.0
N = 5000       # 5 s
FX = 1.4
FY = 20.0


def _write_time_response(path, swap_columns=False, offsets=(3.0, -1.0),
                         amplitudes=(35.5, 7.25)):
    """A TimeResponse_*.txt export of a Lissajous scan, positions in mm."""
    t = np.arange(N) / FS
    x = offsets[0] + amplitudes[0] * np.cos(2 * np.pi * FX * t + 0.3)
    y = offsets[1] + amplitudes[1] * np.cos(2 * np.pi * FY * t - 0.2)
    velocity = (x * y) * np.cos(2 * np.pi * 200.0 * t)
    if swap_columns:
        x, y = y, x
    columns = np.column_stack([t, x, t, y, t, velocity])
    header = ("Time (s) - X-Mirror\tPosition (mm) - X-Mirror\t"
              "Time (s) - Y-Mirror\tPosition (mm) - Y-Mirror\t"
              "Time (s) - LDV\tVelocity (mm/s) - LDV")
    np.savetxt(path, columns, delimiter="\t", header=header, comments="")
    return t, x, y, velocity


def _write_xlsx(path, sheets):
    """A minimal workbook with one numeric column per sheet."""
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    relations = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", f"<workbook {ns} {relations}><sheets>" + "".join(
            f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(sheets, start=1)) + "</sheets></workbook>")
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>'
                      for i in range(1, len(sheets) + 1)) + "</Relationships>")
        for i, values in enumerate(sheets.values(), start=1):
            rows = "".join(f'<row r="{r}"><c r="A{r}"><v>{float(value)!r}</v></c></row>'
                           for r, value in enumerate(values, start=1))
            archive.writestr(f"xl/worksheets/sheet{i}.xml",
                             f"<worksheet {ns}><sheetData>{rows}</sheetData></worksheet>")


class TestTimeResponse:

    def test_reads_channels_and_normalizes(self, tmp_path):
        """The mirror positions are centred and scaled onto [-1, 1] and the
        velocity column is returned unchanged."""
        path = tmp_path / "TimeResponse_200.txt"
        _, x_mm, y_mm, velocity = _write_time_response(path)

        t, x, y, v = suite.read_time_response(path, fx=FX, fy=FY)

        assert 1 / (t[1] - t[0]) == pytest.approx(FS)
        assert np.allclose(v, velocity, atol=1e-6)
        for normalized, physical in ((x, x_mm), (y, y_mm)):
            # not exactly 1: the samples do not fall on the turning points
            assert np.max(np.abs(normalized)) == pytest.approx(1.0, abs=0.01)
            assert np.corrcoef(normalized, physical)[0, 1] == pytest.approx(1.0)

    def test_physical_extent_is_recovered(self, tmp_path):
        """Without normalization the positions stay in mm, and
        normalize_scan reports the offset and amplitude of the scan."""
        path = tmp_path / "TimeResponse_200.txt"
        _write_time_response(path, offsets=(3.0, -1.0), amplitudes=(35.5, 7.25))

        t, x, y, _ = suite.read_time_response(path, fx=FX, fy=FY, normalize=False)
        fs = 1 / (t[1] - t[0])
        for signal, f, offset, amplitude in ((x, FX, 3.0, 35.5), (y, FY, -1.0, 7.25)):
            _, measured_offset, measured_amplitude = pycsldv.normalize_scan(signal, f, fs)
            assert measured_offset == pytest.approx(offset, abs=1e-3)
            assert measured_amplitude == pytest.approx(amplitude, rel=1e-3)

    def test_swapped_columns_are_matched_by_frequency(self, tmp_path):
        """A file whose mirror columns do not follow the given scan
        frequencies is corrected, with a warning."""
        path = tmp_path / "TimeResponse_200.txt"
        _write_time_response(path, swap_columns=True)

        with pytest.warns(UserWarning, match="swapped"):
            t, x, y, _ = suite.read_time_response(path, fx=FX, fy=FY)

        fs = 1 / (t[1] - t[0])
        assert pycsldv.dominant_frequency(x, fs) == pytest.approx(FX, abs=1e-6)
        assert pycsldv.dominant_frequency(y, fs) == pytest.approx(FY, abs=1e-6)

    def test_unexpected_scan_frequency_warns(self, tmp_path):
        """A scan frequency that the measurement does not show is reported
        rather than silently used."""
        path = tmp_path / "TimeResponse_200.txt"
        _write_time_response(path)

        with pytest.warns(UserWarning, match="differs from the given"):
            suite.read_time_response(path, fx=FX, fy=17.0)

    def test_normalization_needs_the_scan_frequencies(self, tmp_path):
        path = tmp_path / "TimeResponse_200.txt"
        _write_time_response(path)

        with pytest.raises(ValueError, match="fx and fy"):
            suite.read_time_response(path)

    def test_wrong_column_count(self, tmp_path):
        path = tmp_path / "TimeResponse_200.txt"
        np.savetxt(path, np.zeros((10, 4)), delimiter="\t", header="a\tb\tc\td",
                   comments="")

        with pytest.raises(ValueError, match="six columns"):
            suite.read_time_response(path, fx=FX, fy=FY)


class TestScanPath:

    def test_reads_two_columns(self, tmp_path):
        path = tmp_path / "LissajousPattern_200.txt"
        volts = np.column_stack([np.linspace(-2.3, 2.3, 50),
                                 np.linspace(-0.44, 0.44, 50)])
        np.savetxt(path, volts, delimiter="\t", comments="",
                   header="Horizontal Position\tVertical Position")

        horizontal, vertical = suite.read_scan_path(path)
        assert np.allclose(horizontal, volts[:, 0])
        assert np.allclose(vertical, volts[:, 1])


class TestOdsExport:

    def test_reads_grid(self, tmp_path):
        """The four ODS columns are reshaped onto the square grid they were
        flattened from, with x varying along the first axis."""
        path = tmp_path / "200.xlsx"
        points = np.linspace(-1, 1, 3)
        x_grid, y_grid = np.meshgrid(points, points, indexing="ij")
        z = x_grid + 2j * y_grid
        _write_xlsx(path, {
            "ODS_X_Data_mm": x_grid.ravel(),
            "ODS_Y_Data_mm": y_grid.ravel(),
            "ODS_Z_Real_Data_mm_s_V": z.real.ravel(),
            "ODS_Z_Imag_Data_mm_s_V": z.imag.ravel(),
            "FRF_Freq_Data_kHz": [0.2],
            "FRF_Mag_Data_dB": [-12.8],
        })

        x, y, values = suite.read_ods_export(path)
        assert x.shape == y.shape == values.shape == (3, 3)
        assert np.allclose(x, x_grid)
        assert np.allclose(y, y_grid)
        assert np.allclose(values, z)

    def test_missing_sheet(self, tmp_path):
        path = tmp_path / "200.xlsx"
        _write_xlsx(path, {"FRF_Freq_Data_kHz": [0.2]})

        with pytest.raises(ValueError, match="ODS_X"):
            suite.read_ods_export(path)
