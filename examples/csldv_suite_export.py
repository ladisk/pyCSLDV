"""
Readers for the measurement files written by the original LabVIEW/MATLAB
CSLDV suite (Bartlett & Tarazaga, DOI 10.5281/zenodo.22032252).

These are an adapter for one specific acquisition front end, not part of the
pyCSLDV package: the package works on plain arrays, and measured data will
reach it in whatever format the user's vibrometer and DAQ produce. The
adapter lives here because it is what ``Showcase_measured.ipynb`` and
``tests/test_measured_reference.py`` need to run pyCSLDV on a real
measurement. Copy it if you have data from the same suite.

The files themselves are published with the suite and are fetched on demand
by :mod:`examples.zenodo_dataset`, so neither the notebook nor the tests
need them to be placed by hand.

A measurement is exported as a set of files sharing the excitation frequency
as their name, e.g. for 4527 Hz:

``TimeResponse_4527.txt``
    Tab-separated, one header line, six columns: time [s] and position [mm]
    of the first mirror, the same for the second mirror, and time [s] and
    velocity [mm/s] of the LDV. This is the measurement itself and the input
    of :func:`pycsldv.demodulate_ods`.
``LissajousPattern_4527.txt``
    Tab-separated, one header line, two columns: the commanded scan path in
    mirror drive volts (horizontal, vertical). This is the *unrotated*
    pattern: the suite exports the coordinates of the figure it plots for
    illustration, and that figure is drawn before the sample-alignment
    rotation is applied (see the note on rotation below). For the path the
    laser actually followed, use the mirror feedback columns of
    ``TimeResponse``.
``Reference_4527.xlsx``
    The processed results of the suite: the reconstructed ODS on a regular
    grid (``ODS_X``, ``ODS_Y``, ``ODS_Z_Real``, ``ODS_Z_Imag``) and the
    averaged FRF (``FRF_Freq``, ``FRF_Mag``), one column per sheet, without
    a header row.
``GUISettings_4527.png``
    Screenshot of the acquisition front panel; the sampling rate, the scan
    frequencies, the excitation frequency and the sideband count have to be
    read from it, as they are not stored in the exported files.

``Reference_4527.mat``
    The contents of the workbook as MATLAB ``table`` objects (MCOS), which
    cannot be read without MATLAB; use the workbook instead.

The last two are named ``4527.xlsx`` and ``4527.mat`` in the copies that
circulated by e-mail before the dataset was published;
:func:`examples.zenodo_dataset.dataset_file` accepts either name.

.. warning::
    The axis naming is not consistent across these files: in the dataset the
    readers were developed against, the column labelled *X-Mirror* in
    ``TimeResponse`` follows the frequency the acquisition GUI calls the
    *Y-Mirror Scanning Frequency*, and the workbook's ``ODS_X`` is the axis
    that ``LissajousPattern`` lists second. The author of the suite
    attributes the first of these to the mirror channels having been swapped
    in the wiring or in the acquisition backend, which makes it a property of
    that measurement rather than of the export format: another dataset may
    well be labelled correctly. Either way, pass the scan frequencies to
    :func:`read_time_response` so that the channels are matched by their
    measured frequency rather than by their label.

.. note::
    The suite rotates the scan by the sample-alignment angle recorded during
    calibration, and exports the ODS in that rotated sample frame. The
    rotation shows up in the ``TimeResponse`` mirror columns as cross-axis
    coupling -- each column carries a little of the other mirror's scan
    frequency -- but not in ``LissajousPattern``, which is exported
    unrotated. It is worth checking against the calibration angle, as the
    calibration is known to over-compensate: in the 4527 Hz dataset the
    coupling implies 1.8 deg to 2.0 deg against a recorded 1.60368 deg.

    pyCSLDV reconstructs in the scan frame, which follows the specimen, and
    the coupling does not disturb it: :func:`pycsldv.normalize_scan` and the
    demodulation both read only the component at the scan frequency. To
    compare with the suite's export, which is on the mirror axes, measure
    the angle with :func:`pycsldv.scan_rotation` and turn the shape with
    :func:`pycsldv.rotate_ods`::

        angle = pycsldv.scan_rotation(x_mm, y_mm, fx, fy, fs)
        on_the_mirror_axes = pycsldv.rotate_ods(coefficients, -angle)

    For the 4527 Hz dataset that lifts the agreement with the suite from
    MAC 0.9905 to 0.9955. The fit wants the rotation on the normalized
    domain rather than on the physical surface (``aspect=1``, the default),
    which suggests the suite rotates its grid without allowing for the two
    axes being scaled differently.
"""

import warnings
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from pycsldv import dominant_frequency, normalize_scan

__all__ = ["read_time_response", "read_scan_path", "read_ods_export"]


def read_time_response(path, fx=None, fy=None, normalize=True):
    """
    Read a ``TimeResponse_*.txt`` measurement export.

    When the scan frequencies are given, the two mirror columns are assigned
    to the x and y axes by their measured dominant frequency, which resolves
    the inconsistent axis labelling of the export (see the module
    documentation); a warning is issued when the assignment does not follow
    the column order of the file.

    :param path: path of the ``TimeResponse_*.txt`` file
    :param fx: scan frequency of the x mirror [Hz], as set in the GUI
    :param fy: scan frequency of the y mirror [Hz], as set in the GUI
    :param normalize: map the mirror positions onto ``[-1, 1]``; requires
        ``fx`` and ``fy``. Set to ``False`` to keep the exported positions
        (in mm) and normalize them with :func:`pycsldv.normalize_scan`
        separately.
    :return: ``(t, x, y, velocity)`` — time [s], the two mirror positions
        (normalized, or in mm) and the LDV velocity [mm/s]. The sampling
        frequency is ``1 / (t[1] - t[0])``.
    """
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(
            f"expected six columns (time and position of two mirrors, time "
            f"and velocity of the LDV), got {data.shape[1] if data.ndim == 2 else 1}")

    t, first, second, velocity = data[:, 0], data[:, 1], data[:, 3], data[:, 5]
    if not (np.allclose(t, data[:, 2]) and np.allclose(t, data[:, 4])):
        warnings.warn("the three time columns of the export differ; "
                      "the mirror and LDV channels may not be aligned")
    fs = 1 / np.mean(np.diff(t))

    x, y = first, second
    if fx is not None and fy is not None:
        f_first, f_second = (dominant_frequency(s, fs) for s in (first, second))
        if abs(f_first - fx) + abs(f_second - fy) > abs(f_first - fy) + abs(f_second - fx):
            x, y = second, first
            warnings.warn(
                f"the mirror columns are swapped with respect to the given scan "
                f"frequencies: the first column scans at {f_first:.4g} Hz and the "
                f"second at {f_second:.4g} Hz; assigning the second column to x "
                f"(fx = {fx:g} Hz) and the first to y (fy = {fy:g} Hz)")
        for axis, signal, f in (("x", x, fx), ("y", y, fy)):
            measured = dominant_frequency(signal, fs)
            if abs(measured - f) > 0.01 * f:
                warnings.warn(f"the {axis} mirror scans at {measured:.4g} Hz, "
                              f"which differs from the given {f:g} Hz")
    elif normalize:
        raise ValueError("normalization requires the scan frequencies fx and fy")

    if normalize:
        x = normalize_scan(x, fx, fs)[0]
        y = normalize_scan(y, fy, fs)[0]
    return t, x, y, velocity


def read_scan_path(path):
    """
    Read a ``LissajousPattern_*.txt`` scan-path export.

    :param path: path of the ``LissajousPattern_*.txt`` file
    :return: ``(horizontal, vertical)`` commanded mirror drive signals [V]
    """
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("expected two columns (horizontal and vertical "
                         "position of the scan path)")
    return data[:, 0], data[:, 1]


def read_ods_export(path):
    """
    Read the ODS the original suite reconstructed, from its ``.xlsx`` export.

    Useful as a reference for the ODS of :func:`pycsldv.demodulate_ods`; the
    two are comparable with :func:`pycsldv.mac`.

    :param path: path of the ``<frequency>.xlsx`` workbook
    :return: ``(x, y, z)`` on a square grid of shape ``(n, n)``; ``x`` varies
        along the first axis and ``z`` is complex [mm/s per excitation volt].
        Which physical axis ``x`` is follows the workbook, not the scan
        frequencies, and the shape is in the rotated sample frame (see the
        module documentation).
    """
    columns = _read_xlsx_columns(path)

    def column(prefix):
        matches = [name for name in columns if name.startswith(prefix)]
        if not matches:
            raise ValueError(f"the workbook has no {prefix}* sheet; "
                             f"it has {sorted(columns)}")
        return columns[matches[0]]

    x, y = column("ODS_X"), column("ODS_Y")
    z = column("ODS_Z_Real") + 1j * column("ODS_Z_Imag")
    n = int(round(len(x) ** 0.5))
    if n * n != len(x) or not (len(y) == len(z) == len(x)):
        raise ValueError(f"expected four ODS columns of equal square length, "
                         f"got {len(x)}, {len(y)}, {len(z)}")
    return x.reshape(n, n), y.reshape(n, n), z.reshape(n, n)


_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RELATION_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _read_xlsx_columns(path):
    """
    Numeric values of every sheet of a workbook, keyed by sheet name.

    A deliberately minimal xlsx reader, so that reading the exported results
    needs no additional dependency. It assumes what the suite writes: a
    single column of numbers per sheet and no header row. Text cells (shared
    or inline strings) are skipped.
    """
    columns = {}
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        targets = {relation.get("Id"): relation.get("Target") for relation
                   in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        for sheet in workbook.iter(f"{_SHEET_NS}sheet"):
            part = targets[sheet.get(f"{_RELATION_NS}id")].lstrip("/")
            if not part.startswith("xl/"):
                part = "xl/" + part
            values = []
            for cell in ET.fromstring(archive.read(part)).iter(f"{_SHEET_NS}c"):
                value = cell.find(f"{_SHEET_NS}v")
                if value is not None and cell.get("t") not in ("s", "str", "inlineStr"):
                    values.append(float(value.text))
            columns[sheet.get("name")] = np.array(values)
    return columns
