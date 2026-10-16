"""DICOM conversion tests for the acquisition builder (no SIRF/STIR needed)."""

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

from simind_python_connector.builders.acquisition_builder import (
    STIRSPECTAcquisitionDataBuilder,
)


pytestmark = pytest.mark.unit


def _make_dicom(
    path: Path,
    frames: int,
    rows: int = 4,
    cols: int = 5,
    with_rotation: bool = True,
    with_time: bool = True,
    with_energy_windows: bool = True,
    explicit_frames: bool = False,
) -> Path:
    """Write a minimal NM DICOM file exercising the builder's parsing."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "NM"
    ds.AcquisitionDate = "20260101"
    ds.AcquisitionTime = "120000"
    ds.AcquisitionNumber = 1
    ds.Manufacturer = "Test scanner"
    ds.InstitutionName = "Test institution"
    ds.PatientName = "Test^Patient"
    ds.StudyDescription = "SPECT"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = [1.5, 2.0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if frames > 1 or explicit_frames:
        ds.NumberOfFrames = frames

    pixel = np.arange(frames * rows * cols, dtype=np.uint16).reshape(frames, rows, cols)
    ds.PixelData = pixel.tobytes()

    if with_rotation:
        rot = Dataset()
        rot.StartAngle = 180
        rot.ScanArc = 360
        rot.RotationDirection = "CW"
        rot.RadialPosition = 200.0
        if with_time:
            from pydicom.dataelem import DataElement

            # ActualFrameDuration (0018,1242), in ms
            rot[(0x0018, 0x1242)] = DataElement((0x0018, 0x1242), "IS", 500)
        ds.RotationInformationSequence = [rot]

    if with_energy_windows:
        window_range = Dataset()
        window_range.EnergyWindowLowerLimit = 126.0
        window_range.EnergyWindowUpperLimit = 154.0
        window_item = Dataset()
        window_item.EnergyWindowRangeSequence = [window_range]
        ds.EnergyWindowInformationSequence = [window_item]

    ds.save_as(path, write_like_original=False)
    return path


def _new_builder() -> STIRSPECTAcquisitionDataBuilder:
    return STIRSPECTAcquisitionDataBuilder()


def test_single_frame_dicom_populates_header_and_pixels(tmp_path: Path):
    dicom_path = _make_dicom(tmp_path / "single.dcm", frames=1)

    builder = _new_builder()
    with pytest.warns(UserWarning, match="NumberOfFrames"):
        builder.update_header_from_dicom(str(dicom_path))

    assert builder.header["!matrix size [1]"] == "5"
    assert builder.header["!matrix size [2]"] == "4"
    assert builder.header["scaling factor (mm/pixel) [1]"] == "2.0"
    assert builder.header["scaling factor (mm/pixel) [2]"] == "1.5"
    assert builder.pixel_array.ndim == 4
    assert builder.pixel_array.shape == (1, 4, 1, 5)
    # Frames-first promotion feeds the same transpose/rotation as multiframe
    # data: rows keep their order and columns are mirrored.
    source = np.arange(20, dtype=np.uint16).reshape(4, 5)
    expected = source[:, ::-1][np.newaxis, :, np.newaxis, :]
    assert np.array_equal(builder.pixel_array, expected)


def test_single_frame_pixels_match_explicit_one_frame_multiframe(tmp_path: Path):
    single = _new_builder()
    with pytest.warns(UserWarning, match="NumberOfFrames"):
        single.update_header_from_dicom(
            str(_make_dicom(tmp_path / "single.dcm", frames=1))
        )
    multi = _new_builder()
    multi.update_header_from_dicom(
        str(_make_dicom(tmp_path / "multi1.dcm", frames=1, explicit_frames=True))
    )

    assert np.array_equal(single.pixel_array, multi.pixel_array)


def test_multiframe_dicom_sets_projection_count(tmp_path: Path):
    dicom_path = _make_dicom(tmp_path / "multi.dcm", frames=6)

    builder = _new_builder()
    builder.update_header_from_dicom(str(dicom_path))

    assert builder.header["!number of projections"] == "6"
    # 500 ms per projection * 6 projections
    assert builder.header["number of time frames"] == "1"
    assert builder.header["!image duration (sec)[1]"] == "3"


def test_missing_time_per_projection_preserves_other_metadata(tmp_path: Path):
    dicom_path = _make_dicom(tmp_path / "notime.dcm", frames=4, with_time=False)

    builder = _new_builder()
    builder.update_header_from_dicom(str(dicom_path))

    assert float(builder.header["start angle"]) == pytest.approx(180.0)
    assert builder.header["!direction of rotation"] in {"CW", "CCW"}
    assert float(builder.header["!extent of rotation"]) == pytest.approx(360.0)
    assert "!image duration (sec)[1]" not in builder.header
    assert "!time per projection (sec)[1]" not in builder.header


def test_missing_rotation_sequence_warns_but_keeps_pixel_data(tmp_path: Path):
    dicom_path = _make_dicom(tmp_path / "norot.dcm", frames=2, with_rotation=False)

    builder = _new_builder()
    with pytest.warns(UserWarning, match="Rotation Information"):
        builder.update_header_from_dicom(str(dicom_path))
    with_rotation = _new_builder()
    with_rotation.update_header_from_dicom(
        str(_make_dicom(tmp_path / "rot.dcm", frames=2))
    )

    assert np.array_equal(builder.pixel_array, with_rotation.pixel_array)


def test_missing_energy_windows_warns(tmp_path: Path):
    dicom_path = _make_dicom(
        tmp_path / "nowin.dcm", frames=2, with_energy_windows=False
    )

    builder = _new_builder()
    with pytest.warns(UserWarning, match="Energy Window"):
        builder.update_header_from_dicom(str(dicom_path))

    assert not hasattr(builder, "energy_windows") or not builder.energy_windows


def test_dicom_energy_window_vector_is_read(tmp_path: Path):
    dicom_path = _make_dicom(tmp_path / "vector.dcm", frames=4)
    ds = pydicom.dcmread(dicom_path)
    ds.EnergyWindowVector = [1, 2, 1, 2]
    ds.save_as(dicom_path)

    builder = _new_builder()
    builder.update_header_from_dicom(str(dicom_path))

    assert builder.energy_window_vector == [1, 2, 1, 2]
