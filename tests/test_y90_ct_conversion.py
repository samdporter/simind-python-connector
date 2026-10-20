"""Tests for the real Y-90 CT conversion tutorials."""

import importlib.util
import json
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


def _independent_ncc(first, second):
    first = first - first.mean()
    second = second - second.mean()
    return float((first * second).sum() / np.sqrt((first**2).sum() * (second**2).sum()))


def _fixed_mask_search(ct_map, reference, mask, max_shift=5):
    """Brute-force the integer NCC search without any mask movement."""
    best = (-np.inf, (0, 0, 0))
    for dz in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                moved = shift_image(
                    ct_map, (dz, dy, dx), order=1, mode="constant", cval=0.0
                )
                if not np.all(np.isfinite(moved[mask])):
                    continue
                score = _independent_ncc(moved[mask], reference[mask])
                if np.isfinite(score) and score > best[0]:
                    best = (score, (dz, dy, dx))
    return best


@pytest.fixture(
    params=[stir_script, sirf_script, pytomography_script],
    ids=["stir", "sirf", "pytomography"],
)
def script(request):
    return request.param


@pytest.mark.unit
def test_registration_scores_every_candidate_on_one_fixed_mask(script):
    reference = _asymmetric_blob()
    moved = shift_image(reference, (2.0, -1.0, 1.0), order=3)
    mask = (reference > 0.05) & (moved > 0.05)
    mask[:, :, :10] = False
    frozen = mask.copy()

    report = script.registration_report(moved, reference, mask)

    assert np.array_equal(mask, frozen)
    best_score, best_shift = _fixed_mask_search(moved, reference, frozen)
    assert report["best_integer_zyx"] == list(best_shift)
    assert report["best_ncc"] == pytest.approx(best_score, abs=1e-6)
    assert report["region_voxels"] == int(frozen.sum())
    assert report["best_ncc_definition"] == "integer-search NCC"


@pytest.mark.unit
@pytest.mark.parametrize(
    "offset",
    [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.75, -0.5)],
)
def test_refinement_recovers_integer_and_fractional_offsets(script, offset):
    reference = _asymmetric_blob()
    moved = shift_image(reference, offset, order=3)

    report = script.registration_report(moved, reference, reference > 0.05)

    # A 0.15-voxel estimator tolerance is a test property, not a gate allowance.
    assert np.allclose(report["offset_zyx"], -np.array(offset), atol=0.15)
    assert report["zero_pearson"] > 0.8


@pytest.mark.unit
@pytest.mark.parametrize(
    "distance,expected", [(1.0, True), (1.02, False), (1.05, False)]
)
def test_gate_accepts_exactly_one_voxel(script, distance, expected):
    assert script.passes_gate(distance) is expected


@pytest.mark.unit
def test_registration_rejects_empty_and_flat_regions(script):
    flat = np.ones((32, 32, 32), dtype=np.float32)
    empty = script.registration_report(flat, flat, np.zeros_like(flat, bool))
    assert empty["status"] == "fail"
    assert empty["passed"] is False
    assert "empty" in empty["reason"]

    region = np.ones_like(flat, bool)
    flat_report = script.registration_report(flat, flat, region)
    assert flat_report["status"] == "fail"
    assert flat_report["passed"] is False
    assert "flat" in flat_report["reason"]
    with pytest.raises(RuntimeError, match="registration failed"):
        sirf_script.accept_registration(flat_report)


@pytest.mark.unit
def test_nonfinite_candidate_samples_invalidate_only_that_candidate(script):
    reference = _asymmetric_blob()
    mask = reference > 0.05
    outside = (~mask) & (reference > 1e-6)
    # Find a voxel just below the mask in z, so a +1 z shift would pull it in.
    candidates = np.argwhere(outside)
    voxel = next(
        tuple(point)
        for point in candidates
        if point[0] + 1 < reference.shape[0] and mask[point[0] + 1, point[1], point[2]]
    )
    moved = reference.copy()
    moved[voxel] = np.nan
    frozen = mask.copy()

    report = script.registration_report(moved, reference, mask)

    assert np.array_equal(mask, frozen)
    assert report["best_integer_zyx"] != [1, 0, 0]
    assert np.isfinite(report["best_ncc"])
    assert report["region_voxels"] == int(frozen.sum())


@pytest.mark.unit
def test_registration_rejects_an_optimum_at_the_search_boundary(script):
    reference = _asymmetric_blob()
    moved = shift_image(reference, (7.0, 0.0, 0.0), order=3)

    report = script.registration_report(moved, reference, reference > 0.05, max_shift=5)

    assert report["passed"] is False
    assert report["status"] == "fail"
    assert "boundary" in report["reason"]
    assert max(abs(value) for value in report["best_integer_zyx"]) == 5


@pytest.mark.unit
def test_quadratic_peak_rejects_invalid_fits(script):
    assert script.quadratic_peak({}, (0, 0, 0)) is None

    minimum = {
        (dz, dy, dx): dz**2 + dy**2 + dx**2
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    }
    assert script.quadratic_peak(minimum, (0, 0, 0)) is None

    unbounded = {
        (dz, dy, dx): -(dz**2 + dy**2 + dx**2) + 10.0 * dz
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    }
    assert script.quadratic_peak(unbounded, (0, 0, 0)) is None

    peak = {
        (dz, dy, dx): -((dz - 0.3) ** 2 + dy**2 + dx**2)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    }
    fitted = script.quadratic_peak(peak, (0, 0, 0))
    assert fitted is not None
    assert np.allclose(fitted, [0.3, 0.0, 0.0], atol=0.01)


@pytest.mark.unit
def test_failure_records_an_unaccepted_summary_and_clears_stale_maps(tmp_path):
    (tmp_path / "maps.png").write_bytes(b"stale")
    (tmp_path / "summary.json").write_text('{"status": "pass"}')
    (tmp_path / "UNACCEPTED.txt").write_text("stale")
    stir_script.clear_stale_outputs(tmp_path)
    assert not (tmp_path / "maps.png").exists()
    assert not (tmp_path / "summary.json").exists()

    report = {
        "status": "fail",
        "passed": False,
        "reason": "offset 1.450 voxels exceeds the 1.0-voxel gate",
        "euclidean_voxels": 1.45,
        "euclidean_mm": 6.17,
    }
    assert stir_script.accept_or_report(report, tmp_path) is False

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "UNACCEPTED"
    assert summary["euclidean_voxels"] == pytest.approx(1.45)
    assert summary["reason"] == report["reason"]
    assert "one-voxel gate" in (tmp_path / "UNACCEPTED.txt").read_text()
    assert not (tmp_path / "maps.png").exists()


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
def test_sirf_identity_route_fails_the_fixed_mask_gate(
    y90_data, real_ct, target_affine, tmp_path
):
    hu_grid, _ = sirf_script.resample_to_spect(
        real_ct, target_affine, y90_data / "umap_zoomed.hv", tmp_path
    )
    coverage = sirf_script.ct_coverage(real_ct.affine, target_affine, real_ct.hu.shape)
    hu_grid[~coverage] = -1000.0
    attenuation = sirf_script.hu_to_attenuation(hu_grid, 150.0)
    umap = sirf_script.read_umap(y90_data)
    region = sirf_script.registration_mask(coverage, hu_grid, umap, attenuation)
    report = sirf_script.registration_report(attenuation, umap, region)

    assert report["passed"] is False
    assert report["status"] == "fail"
    assert report["euclidean_voxels"] > 1.0
    assert report["zero_pearson"] > 0.3
    assert report["region_voxels"] > 100_000
    _landmark_checks(hu_grid)


@pytest.mark.requires_stir
def test_stir_zoom_image_route_fails_the_native_grid_gate(
    y90_data, real_ct, target_affine, tmp_path
):
    hu_grid, coverage = stir_script.resample_to_spect(
        real_ct, target_affine, y90_data / "umap_zoomed.hv", tmp_path
    )
    attenuation = stir_script.hu_to_attenuation(hu_grid, 150.0)
    umap = stir_script.read_umap(y90_data)
    region = stir_script.registration_mask(coverage, hu_grid, umap, attenuation)
    report = stir_script.registration_report(attenuation, umap, region)

    assert report["passed"] is False
    assert report["status"] == "fail"
    assert report["euclidean_voxels"] > 1.0
    assert report["zero_pearson"] > 0.5
    _landmark_checks(hu_grid)


@pytest.mark.requires_pytomography
def test_pytomography_corrected_affine_fails_the_fixed_mask_gate(
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
    umap = pytomography_script.read_umap(y90_data)
    region = pytomography_script.registration_mask(coverage, hu_grid, umap, mu_grid)
    report = pytomography_script.registration_report(mu_grid, umap, region)

    assert report["passed"] is False
    assert report["status"] == "fail"
    assert report["euclidean_voxels"] > 1.0
    assert report["zero_pearson"] > 0.3
    assert report["region_voxels"] > 100_000
    assert np.count_nonzero(mu_grid) > 100_000
    _landmark_checks(hu_grid)

    conversion = pytomography_script.conversion_comparison(ct_xyz, mu_ct)
    assert 0.9 < conversion["median_ratio"] < 1.05
    assert conversion["samples"] > 100_000
