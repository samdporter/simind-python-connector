import hashlib
import importlib.util
import json
import pathlib
import zipfile

import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


pytestmark = pytest.mark.unit

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_y90_bundle.py"
spec = importlib.util.spec_from_file_location("make_y90_bundle", SCRIPT)
make_y90_bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_y90_bundle)


def _write_dicom(path):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "Phantom"
    ds.InstitutionName = "X"
    ds.StationName = "Y"
    ds.add_new(0x00090010, "LO", "VENDOR")
    ds.add_new(0x00091001, "LO", "secret")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path)


def _source(root, peak_in="."):
    root.mkdir()
    for name in ("peak.hs", "peak.s", "umap_zoomed.hv", "umap_zoomed.v"):
        (root / name).write_bytes(b"x")
    _write_dicom(root / peak_in / "peak.dcm")
    _write_dicom(root / "ctac" / "a.dcm")
    _write_dicom(root / "ctac" / "b.dcm")
    return root


@pytest.mark.parametrize("peak_in", [".", "spect_dicom"])
def test_bundle_contents_and_deidentification(tmp_path, capsys, peak_in):
    source = _source(tmp_path / "src", peak_in)
    out = tmp_path / "out"

    make_y90_bundle.main([str(source), str(out)])

    archive = out / "y90_phantom.zip"
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "README.md",
            "peak.dcm",
            "peak.hs",
            "peak.s",
            "umap_zoomed.hv",
            "umap_zoomed.v",
            "ctac/a.dcm",
            "ctac/b.dcm",
        }
        for name in ("peak.dcm", "ctac/a.dcm", "ctac/b.dcm"):
            with bundle.open(name) as member:
                ds = pydicom.dcmread(member)
            assert ds.InstitutionName == ""
            assert ds.StationName == ""
            assert ds.PatientName == "Phantom"
            assert not any(element.tag.is_private for element in ds.iterall())
        assert "136.1 kBq/g" in bundle.read("README.md").decode()
    original = pydicom.dcmread(source / peak_in / "peak.dcm")
    assert original.InstitutionName == "X"
    assert original[0x00091001].value == "secret"

    printed = capsys.readouterr().out
    assert hashlib.sha256(archive.read_bytes()).hexdigest() in printed
    assert "InstitutionName: 3" in printed
    assert "private elements: 6" in printed
    metadata = json.loads((out / "zenodo_metadata.json").read_text())
    assert metadata["license"] == "cc-by-4.0"
