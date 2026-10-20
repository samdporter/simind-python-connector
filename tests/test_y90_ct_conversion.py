"""Tests for the real Y-90 CT conversion tutorials."""

import importlib.util
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
from scipy.ndimage import shift as shift_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "examples" / "real_data"
FRAME_UID = "1.3.6.1.4.1.12842.1.1.14.5.20240417.120102.489.3254784391"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stir_script = _load("plot_16a_stir_ct_conversion")
sirf_script = _load("plot_16b_sirf_ct_conversion")
pytomography_script = _load("plot_16c_pytomography_ct_conversion")


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
    return stir_script.read_ct_series(tmp_path)


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

    series = stir_script.read_ct_series(tmp_path)

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
        stir_script.read_ct_series(tmp_path)


@pytest.mark.unit
def test_ct_loader_rejects_duplicate_slices(tmp_path):
    series_uid = generate_uid()
    _write_slice(tmp_path / "a.dcm", z=0.0, series_uid=series_uid)
    _write_slice(tmp_path / "b.dcm", z=0.0, series_uid=series_uid)
    with pytest.raises(ValueError, match="duplicate"):
        stir_script.read_ct_series(tmp_path)


@pytest.mark.unit
def test_ct_loader_rejects_missing_or_nonuniform_slices(tmp_path):
    series_uid = generate_uid()
    _write_slice(tmp_path / "a.dcm", z=0.0, series_uid=series_uid)
    _write_slice(tmp_path / "b.dcm", z=1.0, series_uid=series_uid)
    _write_slice(tmp_path / "c.dcm", z=3.0, series_uid=series_uid)
    with pytest.raises(ValueError, match="uniform"):
        stir_script.read_ct_series(tmp_path)


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
        stir_script.read_ct_series(tmp_path)


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
        stir_script.read_ct_series(tmp_path)


@pytest.mark.unit
def test_lps_to_ras_affine_flips_patient_x_and_y():
    lps = np.diag([1.0, 1.0, 1.0, 1.0])
    lps[:3, 3] = [-10.0, 20.0, 30.0]
    ras = sirf_script.LPS_TO_RAS @ lps
    assert np.allclose(ras[:3, 3], [10.0, -20.0, 30.0])
    assert np.allclose(np.diag(ras)[:3], [-1.0, -1.0, 1.0])


@pytest.mark.unit
def test_patient_to_spect_affine_places_the_grid_centre_on_the_landmark(tmp_path):
    series = _synthetic_series(tmp_path)
    nm_file = tmp_path / "peak.dcm"
    _write_nm(nm_file)

    affine = stir_script.patient_to_spect_affine(nm_file, series)
    centre = affine @ [63.5, 63.5, 63.5, 1.0]

    assert np.allclose(centre[:3], [-1.999236, -140.0, -571.874206], atol=1e-5)
    assert np.allclose(affine[:3, 0], [4.2578, 0.0, 0.0])
    assert np.allclose(affine[:3, 1], [0.0, 4.2578, 0.0])
    assert np.allclose(affine[:3, 2], [0.0, 0.0, -4.2578])


@pytest.mark.unit
def test_ct_coverage_uses_the_volume_axis_order():
    ct_affine = np.eye(4)
    target_affine = np.eye(4)
    coverage = sirf_script.ct_coverage(ct_affine, target_affine, (10, 20, 30))

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
        stir_script.patient_to_spect_affine(nm_file, series)


# %%
# Registration metric and gate


def _asymmetric_blob(size=48):
    axes = np.indices((size, size, size), dtype=float)
    centre = (size - 1) / 2.0
    main = np.exp(
        -(
            (axes[0] - centre) ** 2
            + (axes[1] - centre + 4.0) ** 2
            + (axes[2] - centre - 3.0) ** 2
        )
        / (2 * 6.0**2)
    )
    feature = 0.6 * np.exp(
        -(
            (axes[0] - centre - 6.0) ** 2
            + (axes[1] - centre - 5.0) ** 2
            + (axes[2] - centre + 5.0) ** 2
        )
        / (2 * 3.0**2)
    )
    return main + feature


@pytest.mark.unit
@pytest.mark.parametrize(
    "offset,status",
    [
        ((0.0, 0.0, 0.0), "pass"),
        ((0.5, 0.0, 0.0), "pass"),
        ((1.0, 0.0, 0.0), "pass"),
        ((2.0, 0.0, 0.0), "fail"),
    ],
)
def test_registration_gate_accepts_one_voxel_and_rejects_two(offset, status):
    reference = _asymmetric_blob()
    moved = shift_image(reference, offset, order=3)
    report = stir_script.registration_report(moved, reference, reference > 0.05)

    assert report["status"] == status
    assert np.allclose(report["offset_zyx"], -np.array(offset), atol=0.2)
    assert report["zero_pearson"] > 0.8


@pytest.mark.unit
def test_registration_rejects_empty_and_flat_regions():
    flat = np.ones((32, 32, 32), dtype=np.float32)
    empty = stir_script.registration_report(flat, flat, np.zeros_like(flat, bool))
    assert empty["status"] == "fail"
    assert "empty" in empty["reason"]

    region = np.ones_like(flat, bool)
    flat_report = stir_script.registration_report(flat, flat, region)
    assert flat_report["status"] == "fail"
    assert "flat" in flat_report["reason"]
    with pytest.raises(RuntimeError, match="registration failed"):
        stir_script.accept_registration(flat_report)


@pytest.mark.unit
def test_near_pass_is_reported_separately_from_a_strict_pass():
    reference = _asymmetric_blob()
    moved = shift_image(reference, (1.05, 0.0, 0.0), order=3)
    report = stir_script.registration_report(
        moved, reference, reference > 0.05, near_pass_voxels=1.1
    )

    assert report["passed"] is False
    assert report["status"] == "near_pass"
    stir_script.accept_registration(report, allow_near_pass=True)
    with pytest.raises(RuntimeError):
        stir_script.accept_registration(report)


# %%
# Live checks on the measured data


@pytest.fixture(scope="session")
def y90_data():
    from simind_python_connector.datasets import fetch_y90_phantom

    try:
        return fetch_y90_phantom()
    except Exception as exc:  # noqa: BLE001 - the local dataset is optional
        pytest.skip(f"Y-90 local data unavailable: {exc}")


@pytest.fixture(scope="session")
def real_ct(y90_data):
    return stir_script.read_ct_series(y90_data / "ctac")


@pytest.fixture(scope="session")
def target_affine(y90_data, real_ct):
    return stir_script.patient_to_spect_affine(y90_data / "peak.dcm", real_ct)


def _landmark_checks(hu_grid):
    assert hu_grid[0, 0, 0] <= -900.0
    assert hu_grid[55, 64, 65] > -400.0
    assert -400.0 < hu_grid[63, 63, 63] < 200.0


@pytest.mark.requires_sirf
def test_sirf_identity_route_passes_the_native_grid_gate(
    y90_data, real_ct, target_affine, tmp_path
):
    hu_grid, _ = sirf_script.resample_to_spect(
        real_ct, target_affine, y90_data / "umap_zoomed.hv", tmp_path
    )
    coverage = sirf_script.ct_coverage(real_ct.affine, target_affine, real_ct.hu.shape)
    hu_grid[~coverage] = -1000.0
    attenuation = sirf_script.hu_to_attenuation(hu_grid, 150.0)
    report = sirf_script.registration_report(
        attenuation,
        sirf_script.read_umap(y90_data),
        (hu_grid > -400) & coverage,
    )

    assert report["status"] == "pass"
    assert report["euclidean_voxels"] <= 1.0
    assert report["zero_pearson"] > 0.3
    assert report["region_voxels"] > 100_000
    _landmark_checks(hu_grid)


@pytest.mark.requires_stir
def test_stir_zoom_image_route_is_an_about_one_voxel_near_pass(
    y90_data, real_ct, target_affine, tmp_path
):
    hu_grid, coverage = stir_script.resample_to_spect(
        real_ct, target_affine, y90_data / "umap_zoomed.hv", tmp_path
    )
    attenuation = stir_script.hu_to_attenuation(hu_grid, 150.0)
    report = stir_script.registration_report(
        attenuation,
        stir_script.read_umap(y90_data),
        (hu_grid > -400) & coverage,
        near_pass_voxels=1.1,
    )

    assert report["passed"] is False
    assert report["status"] == "near_pass"
    assert 1.0 < report["euclidean_voxels"] <= 1.1
    assert report["zero_pearson"] > 0.5
    _landmark_checks(hu_grid)


@pytest.mark.requires_pytomography
def test_pytomography_corrected_affine_passes_and_conversion_is_compared(
    y90_data, real_ct, target_affine, tmp_path
):
    files = sorted((y90_data / "ctac").glob("*.dcm"))
    hu_grid, mu_grid, mu_ct, ct_xyz = pytomography_script.resample_to_spect(
        real_ct, target_affine, files, y90_data / "peak.dcm", tmp_path
    )
    coverage = pytomography_script.ct_coverage(
        real_ct.affine, target_affine, real_ct.hu.shape
    )
    hu_grid[~coverage] = -1000.0
    mu_grid[~coverage] = 0.0
    report = pytomography_script.registration_report(
        mu_grid,
        pytomography_script.read_umap(y90_data),
        coverage & (mu_grid > 0),
    )

    assert report["status"] == "pass"
    assert report["euclidean_voxels"] <= 1.0
    assert report["zero_pearson"] > 0.8
    assert np.count_nonzero(mu_grid) > 100_000
    _landmark_checks(hu_grid)

    conversion = pytomography_script.conversion_comparison(ct_xyz, mu_ct)
    assert 0.9 < conversion["median_ratio"] < 1.05
    assert conversion["samples"] > 100_000
