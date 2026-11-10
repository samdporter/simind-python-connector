"""Tests for the real Y-90 CT conversion tutorials."""

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    NuclearMedicineImageStorage,
    generate_uid,
)

from examples import _y90 as y90


ROOT = Path(__file__).resolve().parents[1]
FRAME_UID = "1.3.6.1.4.1.12842.1.1.14.5.20240417.120102.489.3254784391"


def _write_slice(
    path,
    *,
    z,
    pixel=0,
    slope=1.0,
    intercept=0.0,
    series_uid,
    instance_uid=None,
    frame_uid=FRAME_UID,
    image_position=(-10.0, -10.0),
    orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    pixel_spacing=(1.0, 1.0),
    rows=4,
    columns=4,
    table_height=-140.0,
):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = instance_uid or generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = frame_uid
    dataset.Modality = "CT"
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.ImageOrientationPatient = list(orientation)
    dataset.ImagePositionPatient = [image_position[0], image_position[1], z]
    dataset.PixelSpacing = list(pixel_spacing)
    dataset.SliceThickness = 1.0
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.full((rows, columns), pixel, dtype=np.uint16).tobytes()
    if table_height is not None:
        dataset.TableHeight = table_height
    dataset.save_as(path)


def _write_nm(path, *, frame_uid=FRAME_UID):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = NuclearMedicineImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "NM"
    dataset.FrameOfReferenceUID = frame_uid
    detector = Dataset()
    detector.ImagePositionPatient = [268.371064, -134.249991, -301.503906]
    detector.ImageOrientationPatient = [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    detector.RadialPosition = 300.0
    dataset.DetectorInformationSequence = Sequence([detector])
    dataset.save_as(path)


def _synthetic_series(tmp_path, table_height=-140.0, frame_uid=FRAME_UID):
    series_uid = generate_uid()
    for name, z in (("slice_a.dcm", 2.0), ("slice_b.dcm", 0.0), ("slice_c.dcm", 1.0)):
        _write_slice(
            tmp_path / name,
            z=z,
            series_uid=series_uid,
            frame_uid=frame_uid,
            table_height=table_height,
        )
    return y90.read_ct_series(tmp_path)


# %%
# CT loading and coordinate conversion


@pytest.mark.unit
def test_ct_loader_sorts_slices_and_applies_per_slice_rescaling(tmp_path):
    series_uid = generate_uid()
    _write_slice(tmp_path / "slice_a.dcm", z=2.0, pixel=20, series_uid=series_uid)
    _write_slice(
        tmp_path / "slice_b.dcm",
        z=0.0,
        pixel=10,
        slope=2.0,
        intercept=-100.0,
        series_uid=series_uid,
    )
    _write_slice(
        tmp_path / "slice_c.dcm",
        z=1.0,
        pixel=10,
        slope=1.0,
        intercept=5.0,
        series_uid=series_uid,
    )

    series = y90.read_ct_series(tmp_path)

    assert series.hu.shape == (3, 4, 4)
    assert series.hu[0, 0, 0] == pytest.approx(-80.0)
    assert series.hu[1, 0, 0] == pytest.approx(15.0)
    assert series.hu[2, 0, 0] == pytest.approx(20.0)
    assert [entry["position"][2] for entry in series.slices] == [0.0, 1.0, 2.0]
    assert np.allclose(series.affine[:3, 3], [-10.0, -10.0, 0.0])
    assert np.allclose(series.affine[:3, 2], [0.0, 0.0, 1.0])
    assert np.allclose(series.affine @ [1.0, 2.0, 0.0, 1.0], [-9.0, -8.0, 0.0, 1.0])


@pytest.mark.unit
def test_ct_loader_rejects_mixed_series(tmp_path):
    _write_slice(tmp_path / "a.dcm", z=0.0, series_uid=generate_uid())
    _write_slice(tmp_path / "b.dcm", z=1.0, series_uid=generate_uid())
    with pytest.raises(ValueError, match="series"):
        y90.read_ct_series(tmp_path)


@pytest.mark.unit
def test_ct_loader_rejects_duplicate_slices(tmp_path):
    series_uid = generate_uid()
    _write_slice(tmp_path / "a.dcm", z=0.0, series_uid=series_uid)
    _write_slice(tmp_path / "b.dcm", z=0.0, series_uid=series_uid)
    with pytest.raises(ValueError, match="duplicate"):
        y90.read_ct_series(tmp_path)


@pytest.mark.unit
def test_ct_loader_rejects_missing_or_nonuniform_slices(tmp_path):
    series_uid = generate_uid()
    _write_slice(tmp_path / "a.dcm", z=0.0, series_uid=series_uid)
    _write_slice(tmp_path / "b.dcm", z=1.0, series_uid=series_uid)
    _write_slice(tmp_path / "c.dcm", z=3.0, series_uid=series_uid)
    with pytest.raises(ValueError, match="uniform"):
        y90.read_ct_series(tmp_path)


@pytest.mark.unit
def test_ct_loader_rejects_mixed_origins(tmp_path):
    series_uid = generate_uid()
    _write_slice(
        tmp_path / "a.dcm", z=0.0, image_position=(-10.0, -10.0), series_uid=series_uid
    )
    _write_slice(
        tmp_path / "b.dcm", z=1.0, image_position=(-9.0, -10.0), series_uid=series_uid
    )
    with pytest.raises(ValueError, match="axis"):
        y90.read_ct_series(tmp_path)


@pytest.mark.unit
def test_ct_loader_rejects_oblique_geometry(tmp_path):
    series_uid = generate_uid()
    _write_slice(
        tmp_path / "a.dcm",
        z=0.0,
        orientation=(0.866, 0.5, 0.0, -0.5, 0.866, 0.0),
        series_uid=series_uid,
    )
    with pytest.raises(ValueError, match="axial"):
        y90.read_ct_series(tmp_path)


@pytest.mark.unit
def test_lps_to_ras_affine_flips_patient_x_and_y():
    lps = np.diag([1.0, 1.0, 1.0, 1.0])
    lps[:3, 3] = [-10.0, 20.0, 30.0]
    ras = y90.LPS_TO_RAS @ lps
    assert np.allclose(ras[:3, 3], [10.0, -20.0, 30.0])
    assert np.allclose(np.diag(ras)[:3], [-1.0, -1.0, 1.0])


@pytest.mark.unit
def test_patient_to_spect_affine_places_the_grid_centre_on_the_landmark(tmp_path):
    series = _synthetic_series(tmp_path)
    nm_file = tmp_path / "peak.dcm"
    _write_nm(nm_file)

    affine = y90.patient_to_spect_affine(nm_file, series)
    centre = affine @ [63.5, 63.5, 63.5, 1.0]

    assert np.allclose(centre[:3], [-1.999236, -140.0, -571.874206], atol=1e-5)
    assert np.allclose(affine[:3, 0], [4.2578, 0.0, 0.0])
    assert np.allclose(affine[:3, 1], [0.0, 4.2578, 0.0])
    assert np.allclose(affine[:3, 2], [0.0, 0.0, -4.2578])


@pytest.mark.unit
def test_ct_coverage_uses_the_volume_axis_order():
    ct_affine = np.eye(4)
    target_affine = np.eye(4)
    coverage = y90.ct_coverage(ct_affine, target_affine, (10, 20, 30))

    assert coverage[9, 19, 29]
    assert coverage[0, 0, 0]
    assert not coverage[10, 0, 0]
    assert not coverage[0, 20, 0]
    assert not coverage[0, 0, 30]


@pytest.mark.unit
def test_patient_to_spect_affine_rejects_a_different_frame(tmp_path):
    series = _synthetic_series(tmp_path)
    nm_file = tmp_path / "peak.dcm"
    _write_nm(nm_file, frame_uid=generate_uid())
    with pytest.raises(ValueError, match="frames of reference"):
        y90.patient_to_spect_affine(nm_file, series)
