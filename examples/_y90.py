"""Shared CT preparation for the restricted Y-90 documentation pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


VOXEL_SIZE_MM = 4.2578
MATRIX_SIZE = 128
GRID_CENTRE = (MATRIX_SIZE - 1) / 2.0
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass
class CTSeries:
    """One DICOM CT series: HU volume, LPS affine and sorted slice metadata."""

    hu: np.ndarray
    affine: np.ndarray
    slices: list
    series_uid: str
    frame_uid: str


def _is_axial(row_direction, column_direction, normal):
    axes = np.abs(np.vstack([row_direction, column_direction, normal]))
    cardinal = np.isclose(axes.max(axis=1), 1.0, atol=1e-4) & np.isclose(
        axes.sum(axis=1), 1.0, atol=1e-4
    )
    return bool(cardinal.all() and abs(normal[2]) > 0.999)


def read_ct_series(ct_directory):
    """Read one CT series into a sorted HU volume and a voxel-centre affine."""
    import pydicom

    paths = sorted(Path(ct_directory).glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in {ct_directory}")

    headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in paths]
    series_uids = {str(header.SeriesInstanceUID) for header in headers}
    frames = {str(getattr(header, "FrameOfReferenceUID", "")) for header in headers}
    if len(series_uids) != 1:
        raise ValueError("CT directory contains more than one series")
    if len(frames) != 1:
        raise ValueError("CT slices use different frames of reference")
    if len({str(header.SOPInstanceUID) for header in headers}) != len(headers):
        raise ValueError("CT series repeats a SOP Instance UID")

    rows = {int(header.Rows) for header in headers}
    columns = {int(header.Columns) for header in headers}
    orientations = {
        tuple(round(float(value), 6) for value in header.ImageOrientationPatient)
        for header in headers
    }
    spacings = {
        tuple(round(float(value), 6) for value in header.PixelSpacing)
        for header in headers
    }
    if len(rows) != 1 or len(columns) != 1:
        raise ValueError("CT slices have different matrix sizes")
    if len(orientations) != 1:
        raise ValueError("CT slices have different orientations")
    if len(spacings) != 1:
        raise ValueError("CT slices have different pixel spacing")

    rows = rows.pop()
    columns = columns.pop()
    orientation = np.array(orientations.pop(), float)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    normal = np.cross(row_direction, column_direction)
    if not _is_axial(row_direction, column_direction, normal):
        raise ValueError("Only axial, axis-aligned CT geometry is supported")
    pixel_spacing = spacings.pop()

    positions = np.array([header.ImagePositionPatient for header in headers], float)
    projections = positions @ normal
    order = np.argsort(projections)
    paths = [paths[index] for index in order]
    headers = [headers[index] for index in order]
    positions = positions[order]
    projections = projections[order]

    steps = np.diff(projections)
    if np.any(steps <= 1e-6):
        raise ValueError("CT series contains duplicate slice positions")
    slice_spacing = float(np.median(steps))
    if not np.allclose(steps, slice_spacing, atol=1e-3):
        raise ValueError("CT slice spacing is not uniform (missing slices?)")
    expected = positions[0] + np.outer(projections - projections[0], normal)
    if not np.allclose(positions, expected, atol=1e-3):
        raise ValueError("CT slices do not share a common axis (mixed origins)")

    hu = np.empty((len(paths), rows, columns), dtype=np.float32)
    slices = []
    for index, path in enumerate(paths):
        dataset = pydicom.dcmread(path)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        hu[index] = dataset.pixel_array.astype(np.float32) * slope + intercept
        slices.append(
            {
                "path": str(path),
                "position": [float(value) for value in dataset.ImagePositionPatient],
                "slope": slope,
                "intercept": intercept,
                "table_height": float(getattr(dataset, "TableHeight", np.nan)),
            }
        )
        del dataset

    affine = np.eye(4)
    affine[:3, 0] = row_direction * pixel_spacing[1]
    affine[:3, 1] = column_direction * pixel_spacing[0]
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = slices[0]["position"]
    return CTSeries(
        hu=hu,
        affine=affine,
        slices=slices,
        series_uid=series_uids.pop(),
        frame_uid=frames.pop(),
    )


def patient_to_spect_affine(nm_file, ct_series):
    """Return the SPECT grid ``(i, j, k)`` to patient LPS affine."""
    import pydicom

    nm = pydicom.dcmread(nm_file, stop_before_pixels=True)
    if str(nm.FrameOfReferenceUID) != ct_series.frame_uid:
        raise ValueError("The NM and CT files use different frames of reference")
    detector = nm.DetectorInformationSequence[0]
    position = np.array(detector.ImagePositionPatient, float)
    orientation = np.array(detector.ImageOrientationPatient, float)
    centre = (
        position + (orientation[:3] + orientation[3:]) * GRID_CENTRE * VOXEL_SIZE_MM
    )
    table_height = ct_series.slices[0]["table_height"]
    if not np.isfinite(table_height):
        raise ValueError("The CT series has no TableHeight")

    affine = np.eye(4)
    affine[0, 0] = VOXEL_SIZE_MM
    affine[1, 1] = VOXEL_SIZE_MM
    affine[2, 2] = -VOXEL_SIZE_MM
    affine[0, 3] = centre[0] - GRID_CENTRE * VOXEL_SIZE_MM
    affine[1, 3] = table_height - GRID_CENTRE * VOXEL_SIZE_MM
    affine[2, 3] = centre[2] + GRID_CENTRE * VOXEL_SIZE_MM
    return affine


def read_umap(data_directory):
    """Read the vendor attenuation map on the SPECT grid, in ``(k, j, i)``."""
    values = np.fromfile(Path(data_directory) / "umap_zoomed.v", dtype="<f4")
    return values.astype(np.float32).reshape(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE)


def ct_coverage(ct_affine, target_affine, ct_shape):
    """Return which target voxel centres fall inside the CT field of view."""
    axes = [np.arange(MATRIX_SIZE) for _ in range(3)]
    indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    patient = indices[..., ::-1] @ target_affine[:3, :3].T + target_affine[:3, 3]
    inverse = np.linalg.inv(ct_affine)
    voxel = inverse[:3, :3] @ patient[..., None]
    voxel = voxel[..., 0] + inverse[:3, 3]
    # ``voxel`` is (i, j, k); the volume shape is (k, j, i).
    limits = np.asarray(ct_shape[::-1], float) - 0.5
    return np.all((voxel >= -0.5) & (voxel <= limits), axis=-1)


def resample_ct(ct_series, target_affine, umap_header, work_directory):
    """Resample CT HU onto the SPECT grid with SIRF NiftyResample."""
    import nibabel as nib
    import sirf.Reg as Reg
    import sirf.STIR as STIR

    from simind_python_connector.utils import get_array

    work_directory = Path(work_directory)
    work_directory.mkdir(parents=True, exist_ok=True)
    ct_file = work_directory / "ct_hu.nii"
    reference_file = work_directory / "spect_reference.nii"
    ct_xyz = np.ascontiguousarray(ct_series.hu.transpose(2, 1, 0))
    positive_hu = np.clip(ct_xyz, -1000.0, 3000.0) + 1000.0
    nib.save(
        nib.Nifti1Image(positive_hu.astype(np.float32), LPS_TO_RAS @ ct_series.affine),
        str(ct_file),
    )
    nib.save(
        nib.Nifti1Image(
            np.zeros((MATRIX_SIZE,) * 3, np.float32),
            LPS_TO_RAS @ target_affine,
        ),
        str(reference_file),
    )

    ct_image = Reg.NiftiImageData3D(str(ct_file))
    reference = Reg.NiftiImageData3D(str(reference_file))
    resampler = Reg.NiftyResample()
    resampler.set_reference_image(reference)
    resampler.set_floating_image(ct_image)
    resampler.set_interpolation_type_to_linear()
    resampler.set_padding_value(0)
    resampler.add_transformation(Reg.AffineTransformation(np.eye(4)))
    resampled = resampler.forward(ct_image)
    hu_grid = (
        np.asarray(resampled.as_array(), dtype=np.float32).transpose(2, 1, 0) - 1000.0
    )

    grid = STIR.ImageData(str(umap_header)).get_uniform_copy(0.0)
    grid.fill(np.ascontiguousarray(hu_grid, dtype=np.float32))
    if not np.allclose(get_array(grid), hu_grid, atol=1e-5):
        raise RuntimeError("The resampled HU values did not survive the STIR transfer")
    return hu_grid, grid
