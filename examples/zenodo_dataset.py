"""
Retrieval of the measured CSLDV dataset from Zenodo.

The measurement used by ``Showcase_measured.ipynb`` and by
``tests/test_measured_reference.py`` is a 10 s scan of a rectangular plate
excited at 4527 Hz, recorded by Joshua Bartlett (FAST Laboratory, Texas A&M
University) with the original LabVIEW/MATLAB suite. It is 80 MB, so it is not
part of this repository; it is published as part of the suite itself, at

    https://doi.org/10.5281/zenodo.22032252

Everything that needs the data asks for it through :func:`dataset_file` or
:func:`fetch_dataset`, which download and unpack it on first use and do
nothing at all once it is there.

Two details of the published archive are worth knowing.

The five files of the measurement live inside a 52 MB zip together with the
whole software suite, and there is no way to fetch one of them alone, so the
first call downloads the archive, extracts the five and discards the rest.

The suite exports the processed results as ``Reference_4527.xlsx`` and
``Reference_4527.mat``, but the copies that circulated by e-mail before the
release were named ``4527.xlsx`` and ``4527.mat``. Both names are accepted:
a file already present under either name is used as it is, and nothing is
downloaded to replace it.

The dataset is licensed CC-BY-4.0 by Joshua Bartlett and Pablo Tarazaga; see
``CITATION.cff`` for how to cite it.

Set ``PYCSLDV_DATA`` to keep the data somewhere other than ``examples/data``,
and ``PYCSLDV_NO_DOWNLOAD=1`` to forbid the download entirely -- with that
set, anything missing raises :class:`DatasetUnavailable` instead of being
fetched, which is how a continuous-integration run avoids pulling 52 MB.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

__all__ = ["DatasetUnavailable", "data_directory", "dataset_file",
           "fetch_dataset", "DOI", "RECORD_URL"]

DOI = "10.5281/zenodo.22032252"
RECORD_URL = "https://zenodo.org/records/22032252"
ARCHIVE_URL = ("https://zenodo.org/api/records/22032252/files/"
               "CSLDV_Software_Suite.zip/content")
ARCHIVE_SIZE = 52_429_470

#: Directory inside the archive that holds the measurement.
ARCHIVE_DIRECTORY = ("CSLDV_Software_Suite/MATLAB_Files/"
                     "Experimental_DataProcessing_Example/"
                     "Experimental_Dataset_Unprocessed")

#: Published name -> (MD5 of the published file, names it is also known by).
#: The checksums pin the exact revision of the archive the reconstruction was
#: verified against: the release was re-uploaded once, on 2026-08-21, to add
#: ``LissajousPattern_4527.txt`` to it, so the DOI alone does not identify the
#: contents.
DATASET = {
    "TimeResponse_4527.txt": ("846c9c504b96861183b174685e8f3889", ()),
    "LissajousPattern_4527.txt": ("3a0c5ca57c2d4fb826567555d8c0ddaa", ()),
    "Reference_4527.xlsx": ("3658118cb2dac4e5a10cf28f0dd58ca8", ("4527.xlsx",)),
    "Reference_4527.mat": ("8bdc5a1b2cf737b2199d4379b7fe69c2", ("4527.mat",)),
    "GUISettings_4527.png": ("d18bc94a7f7fac6813be9416841a88f5", ()),
}


class DatasetUnavailable(RuntimeError):
    """The measurement is neither on disk nor retrievable."""


def data_directory():
    """
    Where the measurement is kept.

    ``examples/data`` beside this file, unless ``PYCSLDV_DATA`` says
    otherwise.
    """
    override = os.environ.get("PYCSLDV_DATA")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "data"


def _downloads_allowed():
    """Downloads are allowed unless PYCSLDV_NO_DOWNLOAD is set to something
    other than an empty string or ``0``."""
    return os.environ.get("PYCSLDV_NO_DOWNLOAD", "").strip() in ("", "0")


def _checksum(path, chunk=1 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _names(name):
    """The published name of a file and the names it is also known by."""
    if name not in DATASET:
        known = ", ".join(sorted(DATASET))
        raise KeyError(f"{name!r} is not part of the dataset; it holds {known}")
    return (name,) + DATASET[name][1]


def _locate(name, directory):
    for candidate in _names(name):
        path = directory / candidate
        if path.exists():
            return path
    return None


def _report(message, quiet):
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def _download(url, destination, quiet):
    """Stream a URL to a file, reporting progress on stderr."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "pyCSLDV (https://github.com/ladisk/pyCSLDV)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or ARCHIVE_SIZE)
        read = 0
        step = 0
        with open(destination, "wb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                read += len(block)
                if total and read * 10 // total > step:
                    step = read * 10 // total
                    _report(f"  {read / 1e6:6.1f} MB of {total / 1e6:.1f} MB", quiet)
    return read


def _extract(archive, missing, directory, quiet):
    """Take the missing files out of the downloaded archive."""
    extracted = {}
    with zipfile.ZipFile(archive) as bundle:
        for name in missing:
            member = f"{ARCHIVE_DIRECTORY}/{name}"
            try:
                source = bundle.open(member)
            except KeyError:
                raise DatasetUnavailable(
                    f"the archive at {RECORD_URL} does not contain {member}; "
                    "the release may have been replaced -- please report this "
                    "at https://github.com/ladisk/pyCSLDV/issues") from None
            target = directory / name
            partial = target.with_suffix(target.suffix + ".part")
            with source, open(partial, "wb") as handle:
                shutil.copyfileobj(source, handle, 1 << 20)

            expected = DATASET[name][0]
            actual = _checksum(partial)
            if actual != expected:
                partial.unlink()
                raise DatasetUnavailable(
                    f"{name} from {RECORD_URL} has MD5 {actual}, expected "
                    f"{expected}; the published archive has changed since this "
                    "was written, so the data may no longer be the revision "
                    "the reconstruction was verified against")
            partial.replace(target)
            extracted[name] = target
            _report(f"  {name}", quiet)
    return extracted


def fetch_dataset(directory=None, quiet=False):
    """
    Make sure the whole measurement is on disk, and say where it is.

    Files already present -- under their published name or under the name the
    e-mailed copies used -- are left alone. If any are missing, the Zenodo
    archive is downloaded once and every missing file is taken out of it.

    :param directory: where to keep the data; defaults to
        :func:`data_directory`
    :param quiet: suppress the progress messages on stderr
    :return: dict mapping the published file name to its path on disk
    :raises DatasetUnavailable: if files are missing and cannot be downloaded,
        either because ``PYCSLDV_NO_DOWNLOAD`` is set, because the download
        fails, or because what arrives is not what was expected
    """
    directory = Path(directory) if directory is not None else data_directory()

    found = {}
    missing = []
    for name in DATASET:
        path = _locate(name, directory) if directory.exists() else None
        if path is None:
            missing.append(name)
        else:
            found[name] = path
    if not missing:
        return found

    if not _downloads_allowed():
        raise DatasetUnavailable(
            f"{', '.join(missing)} missing from {directory} and "
            "PYCSLDV_NO_DOWNLOAD is set; unset it to download the measurement "
            f"from {RECORD_URL}, or place the files there by hand")

    directory.mkdir(parents=True, exist_ok=True)
    _report(f"pyCSLDV: fetching the measured dataset from {RECORD_URL}\n"
            f"  ({', '.join(missing)}; the archive is "
            f"{ARCHIVE_SIZE / 1e6:.0f} MB and is discarded once unpacked)", quiet)

    with tempfile.TemporaryDirectory() as workspace:
        archive = Path(workspace) / "CSLDV_Software_Suite.zip"
        try:
            _download(ARCHIVE_URL, archive, quiet)
        except (urllib.error.URLError, OSError) as error:
            raise DatasetUnavailable(
                f"could not download the measurement from {RECORD_URL}: "
                f"{error}. Download it by hand and put the contents of "
                f"{ARCHIVE_DIRECTORY}/ into {directory}") from error
        try:
            found.update(_extract(archive, missing, directory, quiet))
        except zipfile.BadZipFile as error:
            raise DatasetUnavailable(
                f"what was downloaded from {RECORD_URL} is not a zip archive: "
                f"{error}") from error

    return found


def dataset_file(name, directory=None, quiet=False):
    """
    The path of one file of the measurement, fetching it if it is not there.

    :param name: published name of the file, e.g. ``'TimeResponse_4527.txt'``;
        ``'4527.xlsx'`` and ``'4527.mat'`` are accepted as well
    :param directory: where to keep the data; defaults to
        :func:`data_directory`
    :param quiet: suppress the progress messages on stderr
    :return: :class:`pathlib.Path` of the file
    :raises DatasetUnavailable: if it is missing and cannot be downloaded
    """
    for published, (_, aliases) in DATASET.items():
        if name == published or name in aliases:
            name = published
            break
    else:
        # not a file of this dataset: say so now, rather than after
        # downloading 52 MB to look for something that is not in it
        _names(name)

    directory = Path(directory) if directory is not None else data_directory()
    if directory.exists():
        path = _locate(name, directory)
        if path is not None:
            return path
    return fetch_dataset(directory, quiet=quiet)[name]
