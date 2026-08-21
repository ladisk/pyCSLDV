"""
Retrieval of the measured dataset.

These never touch the network: the archive is a small zip built in a
temporary directory and served over a ``file://`` URL, with the expected
checksums patched to match it. What is being checked is the logic around the
download -- which files are considered missing, that an existing copy under
either of its names is left alone, that a mismatched checksum is refused
rather than silently accepted, and that everything fails as a skip-able
:class:`DatasetUnavailable` rather than as some other error.
"""

import hashlib
import zipfile

import pytest

from examples import zenodo_dataset
from examples.zenodo_dataset import DatasetUnavailable

PAYLOAD = {
    "TimeResponse_4527.txt": b"time\tvelocity\n0\t0\n",
    "LissajousPattern_4527.txt": b"horizontal\tvertical\n0\t0\n",
    "Reference_4527.xlsx": b"not really a workbook",
    "Reference_4527.mat": b"not really a mat file",
    "GUISettings_4527.png": b"not really a png",
}


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A stand-in for the published archive, served from the filesystem."""
    path = tmp_path / "CSLDV_Software_Suite.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in PAYLOAD.items():
            bundle.writestr(f"{zenodo_dataset.ARCHIVE_DIRECTORY}/{name}", content)

    monkeypatch.setattr(zenodo_dataset, "ARCHIVE_URL", path.as_uri())
    monkeypatch.setattr(zenodo_dataset, "DATASET", {
        name: (hashlib.md5(content).hexdigest(),
               zenodo_dataset.DATASET[name][1])
        for name, content in PAYLOAD.items()})
    monkeypatch.delenv("PYCSLDV_NO_DOWNLOAD", raising=False)
    return path


@pytest.fixture
def destination(tmp_path):
    return tmp_path / "data"


class TestFetch:

    def test_downloads_and_unpacks(self, archive, destination):
        found = zenodo_dataset.fetch_dataset(destination, quiet=True)

        assert set(found) == set(PAYLOAD)
        for name, content in PAYLOAD.items():
            assert found[name] == destination / name
            assert found[name].read_bytes() == content

    def test_does_nothing_once_the_files_are_there(self, archive, destination,
                                                   monkeypatch):
        zenodo_dataset.fetch_dataset(destination, quiet=True)

        # any download from here on would fail, so a second call that
        # succeeds proves none was attempted
        monkeypatch.setattr(zenodo_dataset, "ARCHIVE_URL",
                            "file:///nonexistent/archive.zip")
        again = zenodo_dataset.fetch_dataset(destination, quiet=True)
        assert set(again) == set(PAYLOAD)

    def test_a_file_under_its_other_name_is_kept(self, archive, destination,
                                                 monkeypatch):
        """The copies that circulated by e-mail are named 4527.xlsx and
        4527.mat; those are not downloaded again under the published name."""
        destination.mkdir(parents=True)
        (destination / "4527.xlsx").write_bytes(b"the copy already here")
        (destination / "4527.mat").write_bytes(b"and this one")

        found = zenodo_dataset.fetch_dataset(destination, quiet=True)

        assert found["Reference_4527.xlsx"] == destination / "4527.xlsx"
        assert found["Reference_4527.xlsx"].read_bytes() == b"the copy already here"
        assert not (destination / "Reference_4527.xlsx").exists()
        # the rest was still fetched
        assert found["TimeResponse_4527.txt"].read_bytes() == PAYLOAD["TimeResponse_4527.txt"]

    def test_only_what_is_missing_is_taken_from_the_archive(self, archive,
                                                           destination):
        destination.mkdir(parents=True)
        (destination / "GUISettings_4527.png").write_bytes(b"kept as it is")

        found = zenodo_dataset.fetch_dataset(destination, quiet=True)
        assert found["GUISettings_4527.png"].read_bytes() == b"kept as it is"


class TestRefusals:

    def test_a_changed_archive_is_refused(self, archive, destination, monkeypatch):
        """If the published file is not the revision the checksums pin, the
        data is not used -- the release has been re-uploaded once already."""
        patched = dict(zenodo_dataset.DATASET)
        patched["Reference_4527.mat"] = ("0" * 32, ("4527.mat",))
        monkeypatch.setattr(zenodo_dataset, "DATASET", patched)

        with pytest.raises(DatasetUnavailable, match="has MD5"):
            zenodo_dataset.fetch_dataset(destination, quiet=True)

        assert not (destination / "Reference_4527.mat").exists()
        assert not list(destination.glob("*.part"))

    def test_a_missing_member_is_reported(self, tmp_path, destination, monkeypatch):
        empty = tmp_path / "empty.zip"
        with zipfile.ZipFile(empty, "w") as bundle:
            bundle.writestr("CSLDV_Software_Suite/README.md", "no data here")
        monkeypatch.setattr(zenodo_dataset, "ARCHIVE_URL", empty.as_uri())
        monkeypatch.delenv("PYCSLDV_NO_DOWNLOAD", raising=False)

        with pytest.raises(DatasetUnavailable, match="does not contain"):
            zenodo_dataset.fetch_dataset(destination, quiet=True)

    def test_an_unreachable_archive_is_reported(self, destination, monkeypatch):
        monkeypatch.setattr(zenodo_dataset, "ARCHIVE_URL",
                            "file:///nonexistent/archive.zip")
        monkeypatch.delenv("PYCSLDV_NO_DOWNLOAD", raising=False)

        with pytest.raises(DatasetUnavailable, match="could not download"):
            zenodo_dataset.fetch_dataset(destination, quiet=True)

    def test_downloads_can_be_forbidden(self, archive, destination, monkeypatch):
        monkeypatch.setenv("PYCSLDV_NO_DOWNLOAD", "1")

        with pytest.raises(DatasetUnavailable, match="PYCSLDV_NO_DOWNLOAD"):
            zenodo_dataset.fetch_dataset(destination, quiet=True)
        assert not destination.exists()

    @pytest.mark.parametrize("value", ["", "0"])
    def test_an_empty_or_zero_setting_does_not_forbid_them(self, archive,
                                                           destination,
                                                           monkeypatch, value):
        monkeypatch.setenv("PYCSLDV_NO_DOWNLOAD", value)
        assert set(zenodo_dataset.fetch_dataset(destination, quiet=True)) == set(PAYLOAD)


class TestSingleFile:

    def test_a_file_is_fetched_by_its_published_name(self, archive, destination):
        path = zenodo_dataset.dataset_file("TimeResponse_4527.txt", destination,
                                           quiet=True)
        assert path.read_bytes() == PAYLOAD["TimeResponse_4527.txt"]

    def test_a_file_is_fetched_by_its_other_name(self, archive, destination):
        path = zenodo_dataset.dataset_file("4527.xlsx", destination, quiet=True)
        assert path == destination / "Reference_4527.xlsx"

    def test_an_unknown_name_is_rejected(self, archive, destination):
        with pytest.raises(KeyError, match="not part of the dataset"):
            zenodo_dataset.dataset_file("TimeResponse_9999.txt", destination,
                                        quiet=True)


class TestLocation:

    def test_defaults_to_examples_data(self, monkeypatch):
        monkeypatch.delenv("PYCSLDV_DATA", raising=False)
        assert zenodo_dataset.data_directory().parts[-2:] == ("examples", "data")

    def test_the_environment_moves_it(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PYCSLDV_DATA", str(tmp_path / "elsewhere"))
        assert zenodo_dataset.data_directory() == tmp_path / "elsewhere"
