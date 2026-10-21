"""
Y-90 reconstruction with Monte Carlo scatter
============================================

Reconstruct a measured Y-90 phantom acquisition with SIRF OSEM and a SIMIND
Monte Carlo scatter estimate, and compare five ways of putting that estimate on
the scale of the data plus an uncorrected baseline.

The data are the measured 120-view acquisition (``peak.hs``, 75-225 keV,
4.2578 mm bins) of a liver phantom with a hot sphere. The scatter model is
SIMIND PENETRATE with the Y-90 bremsstrahlung spectrum, the measured
non-circular orbit, the MA-MLEGP collimator and the density map derived from
the CT. The reconstruction model is SPECTUB with the 150-keV CT attenuation
map, twelve subsets and 48 subiterations per pass.

The five corrections are:

1. **Known activity.** SIMIND is given 160.8 MBq and 40 s per view (Index 25),
   so its output is in counts and no factor is applied. This illustrates
   uncalibrated SIMIND sensitivity; it is not a quantitative route.
2. **Reference, sum.** SIMIND's geometric primary is scaled to the fast
   model's forward projection of the same image, with the ratio of sums.
3. **Reference, trimmed.** The same with a trimmed mean of the bin ratios.
4. **Dewaraja, global.** One factor scales SIMIND's total to the measured
   data (Dewaraja et al., Med Phys 44:6364, 2017).
5. **Dewaraja, pixel-wise.** SIMIND's scatter fraction in each bin times the
   measured data, as in the same paper, smoothed over 3 bins including the
   view axis.

The baseline reconstructs the same data with a zero additive term and no
refresh.

Schedule: one common zero-additive pass, one shared first-refresh SIMIND run
from that image, one strategy-specific second-refresh SIMIND run for each of
the five corrected strategies, and a baseline to the same total budget. The
refreshes are at one third and two thirds of the unchanged 144-subiteration
budget, on complete iteration boundaries. Damping is 0.5: the first estimate is
halved and the second is a damped average. The five second-refresh runs are
executed in parallel, each with its own output directory.

.. warning::
   This example takes several hours. Nearly all of it is in six SIMIND runs at
   120 views, two more for the independent-seed check and two for the
   history-doubling check. A source map is quantized at a scale that keeps the
   activity lost to integer rounding below 1%, so the run time is set by the
   statistics rather than by the reconstruction.

The Y-90 phantom data are not published for download yet. Set
``SIMIND_CONNECTOR_Y90_DATA`` to a directory holding ``peak.dcm``,
``peak.hs``, ``peak.s``, ``umap_zoomed.hv``, ``umap_zoomed.v`` and ``ctac/``.
"""

# %%
# Setup
# -----

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.ndimage import shift as shift_image

from simind_python_connector.converters.attenuation import (
    hu_to_attenuation,
    hu_to_density_schneider,
)
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    MAX_SOURCE,
    ScoringRoutine,
)
from simind_python_connector.datasets import fetch_y90_phantom
from simind_python_connector.normalisation import scale_to_reference
from simind_python_connector.recon import (
    AdditiveUpdater,
    SimindProjector,
    UpdateSchedule,
)


VOXEL_MM = 4.2578
MATRIX_SIZE = 128
GRID_CENTRE = (MATRIX_SIZE - 1) / 2.0
ENERGY_KEV = 150.0
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

TOTAL_MBQ = 160.8
TIME_PER_VIEW_S = 40.0
LIVER_MBQ = 156.4
SPHERE_MBQ = 4.40
LIVER_MASS_G = 1149.49
SPHERE_MASS_G = 8.082
LIVER_KBQ_G = 136.1
SPHERE_KBQ_G = 544.4
TRUTH_RATIO = 4.0

# Fractional VOI volumes from the accepted CT segmentation and the fixed sphere.
SPHERE_VOLUME_ML = 8.082
SPHERE_RADIUS_MM = (3.0 * SPHERE_VOLUME_ML / (4.0 * np.pi)) ** (1.0 / 3.0) * 10.0
SPHERE_CENTRE_MM = (-77.43, -181.69, -576.20)
LIVER_CT_VOLUME_ML = 978.17
LIVER_CT_VOLUME_WITH_SPHERE_ML = 985.98
SPHERE_VOXEL_TOLERANCE = 2.0

NUM_SUBSETS = 12
SUBITERATIONS_PER_PASS = 48
PASSES = 3
NUM_UPDATES = 2
DAMPING = 0.5
TOTAL_SUBITERATIONS = SUBITERATIONS_PER_PASS * PASSES
REFERENCE_SUBITERATIONS = 12

# Starting source scale. Each run is raised in 1.25x steps only when more than
# 1% of its source activity would be lost to integer quantization; the measured
# scale and loss are reported for every run.
QUANTIZATION_SCALE = 0.75 * DEFAULT_QUANTIZATION_SCALE
MAX_QUANTIZATION_LOSS = 0.01
SMOOTHING_FWHM_BINS = 3.0

TABLE_ROW = 94
GATE_VOXELS = 1.0

STRATEGIES = (
    ("Known activity", "known"),
    ("Reference, sum", "reference_sum"),
    ("Reference, trimmed", "reference_trimmed"),
    ("Dewaraja, global", "dewaraja_global"),
    ("Dewaraja, pixel-wise", "pixelwise"),
)
SLUGS = {
    "Known activity": "known",
    "Reference, sum": "reference_sum",
    "Reference, trimmed": "reference_trimmed",
    "Dewaraja, global": "dewaraja_global",
    "Dewaraja, pixel-wise": "pixelwise",
}
BASELINE = "Baseline"


# %%
# CT series and patient-to-SPECT frame
# ------------------------------------
# The CT slices are not stored in spatial order and are sorted by
# ``ImagePositionPatient``. Only axial, axis-aligned geometry is accepted. The
# patient-to-SPECT affine uses the NM detector centre in x and z and the CT
# table height in y; no origin is re-derived from the images.


@dataclass
class CTSeries:
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
    """Return the grid ``(i, j, k)`` -> patient LPS affine for the SPECT frame."""
    nm = pydicom.dcmread(nm_file, stop_before_pixels=True)
    if str(nm.FrameOfReferenceUID) != ct_series.frame_uid:
        raise ValueError("The NM and CT files use different frames of reference")
    detector = nm.DetectorInformationSequence[0]
    position = np.array(detector.ImagePositionPatient, float)
    orientation = np.array(detector.ImageOrientationPatient, float)
    centre = position + (orientation[:3] + orientation[3:]) * GRID_CENTRE * VOXEL_MM
    table_height = ct_series.slices[0]["table_height"]
    if not np.isfinite(table_height):
        raise ValueError("The CT series has no TableHeight")
    affine = np.eye(4)
    affine[0, 0] = VOXEL_MM
    affine[1, 1] = VOXEL_MM
    affine[2, 2] = -VOXEL_MM
    affine[0, 3] = centre[0] - GRID_CENTRE * VOXEL_MM
    affine[1, 3] = table_height - GRID_CENTRE * VOXEL_MM
    affine[2, 3] = centre[2] + GRID_CENTRE * VOXEL_MM
    return affine


def read_umap(data_directory):
    """Read the vendor attenuation map on the SPECT grid, in ``(k, j, i)`` order."""
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
    limits = np.asarray(ct_shape[::-1], float) - 0.5
    return np.all((voxel >= -0.5) & (voxel <= limits), axis=-1)


# %%
# Registration
# ------------
# The metric is the normalized cross-correlation on the native 4.2578 mm grid.
# The fixed region is CT coverage, CT HU > -400, positive vendor attenuation and
# finite values, eroded once to keep the body/table boundary out of the edge
# correlation. The acceptance criterion is a Euclidean offset of at most one
# voxel. The measured offset is reported, not applied to the maps.

MASK_DEFINITION = (
    "CT coverage, CT HU > -400, positive vendor attenuation and finite values"
)
MASK_EROSION = "one 3-D connectivity-1 binary erosion, one voxel, border_value=0"


def _intersection_mask(coverage, hu_grid, umap, ct_map):
    return (
        coverage
        & (hu_grid > -400)
        & (umap > 0)
        & np.isfinite(ct_map)
        & np.isfinite(umap)
    )


def registration_mask(coverage, hu_grid, umap, ct_map):
    """Return the fixed body region used for every candidate translation."""
    return binary_erosion(
        _intersection_mask(coverage, hu_grid, umap, ct_map), iterations=1
    )


def _ncc(first, second, mask):
    first = first[mask] - first[mask].mean()
    second = second[mask] - second[mask].mean()
    denominator = np.sqrt((first * first).sum() * (second * second).sum())
    return float((first * second).sum() / denominator) if denominator > 0 else np.nan


def passes_gate(euclidean_voxels):
    """Return whether a measured offset meets the one-voxel gate."""
    return euclidean_voxels <= GATE_VOXELS


def quadratic_peak(scores, centre):
    """Return the subvoxel maximum around ``centre``, or None if the fit fails."""
    rows, values = [], []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                key = (centre[0] + dz, centre[1] + dy, centre[2] + dx)
                if key not in scores:
                    continue
                rows.append(
                    [
                        1,
                        dz,
                        dy,
                        dx,
                        dz * dz,
                        dy * dy,
                        dx * dx,
                        dz * dy,
                        dz * dx,
                        dy * dx,
                    ]
                )
                values.append(scores[key])
    if len(values) < 10:
        return None
    coefficients, *_ = np.linalg.lstsq(
        np.array(rows, float), np.array(values, float), rcond=None
    )
    if not np.all(np.isfinite(coefficients)):
        return None
    hessian = np.array(
        [
            [2 * coefficients[4], coefficients[7], coefficients[8]],
            [coefficients[7], 2 * coefficients[5], coefficients[9]],
            [coefficients[8], coefficients[9], 2 * coefficients[6]],
        ]
    )
    if not np.all(np.linalg.eigvalsh(hessian) < 0):
        return None
    gradient = np.array([coefficients[1], coefficients[2], coefficients[3]])
    try:
        subvoxel = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(subvoxel)) or np.max(np.abs(subvoxel)) > 1.0:
        return None
    return subvoxel


def registration_report(
    ct_map, reference, mask, voxel_size_mm=VOXEL_MM, max_shift=5, min_voxels=1000
):
    """Measure the CT-to-SPECT offset on a fixed mask and apply the gate."""
    ct_map = np.asarray(ct_map, float)
    reference = np.asarray(reference, float)
    mask = np.asarray(mask, bool) & np.isfinite(ct_map) & np.isfinite(reference)
    report = {
        "region_voxels": int(mask.sum()),
        "zero_pearson": None,
        "best_integer_zyx": None,
        "best_ncc": None,
        "best_ncc_definition": "integer-search NCC",
        "offset_zyx": None,
        "offset_mm": None,
        "euclidean_voxels": None,
        "euclidean_mm": None,
        "refined_by": "integer",
        "passed": False,
        "status": "fail",
        "reason": "",
    }
    if mask.sum() < min_voxels:
        report["reason"] = "empty registration region"
        return report
    if np.std(ct_map[mask]) == 0 or np.std(reference[mask]) == 0:
        report["reason"] = "flat registration region"
        return report
    report["zero_pearson"] = float(np.corrcoef(ct_map[mask], reference[mask])[0, 1])

    occupied = np.argwhere(mask)
    low = np.maximum(occupied.min(axis=0) - max_shift - 2, 0)
    high = np.minimum(occupied.max(axis=0) + max_shift + 3, mask.shape)
    crop = tuple(slice(start, stop) for start, stop in zip(low, high))
    ct_crop = ct_map[crop]
    reference_crop = reference[crop]
    mask_crop = mask[crop]

    scores = {}
    best = (-np.inf, (0, 0, 0))
    for dz in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                moved = shift_image(
                    ct_crop, (dz, dy, dx), order=1, mode="constant", cval=0.0
                )
                if not np.all(np.isfinite(moved[mask_crop])):
                    continue
                score = _ncc(moved, reference_crop, mask_crop)
                if not np.isfinite(score):
                    continue
                scores[(dz, dy, dx)] = score
                if score > best[0]:
                    best = (score, (dz, dy, dx))
    if not scores:
        report["reason"] = "no overlap survives the translation search"
        return report

    best_score, best_shift = best
    report["best_integer_zyx"] = [int(value) for value in best_shift]
    report["best_ncc"] = float(best_score)
    if max(abs(value) for value in best_shift) == max_shift:
        report["reason"] = "best integer shift reaches the search boundary"
        return report

    refined = np.array(best_shift, float)
    subvoxel = quadratic_peak(scores, best_shift)
    if subvoxel is not None:
        refined += subvoxel
        report["refined_by"] = "quadratic"
    distance = float(np.linalg.norm(refined))
    report.update(
        {
            "offset_zyx": refined.tolist(),
            "offset_mm": (refined * voxel_size_mm).tolist(),
            "euclidean_voxels": distance,
            "euclidean_mm": distance * voxel_size_mm,
            "passed": passes_gate(distance),
        }
    )
    if report["passed"]:
        report["status"] = "pass"
    else:
        report["reason"] = (
            f"offset {distance:.3f} voxels exceeds the {GATE_VOXELS:.1f}-voxel gate"
        )
    return report


def accept_registration(report):
    """Stop the tutorial with diagnostics when the gate fails."""
    if not report["passed"]:
        raise RuntimeError(
            "CT-to-SPECT registration failed: "
            f"{report['reason'] or report['status']}. Diagnostics were written "
            "before any map was produced."
        )


# %%
# SIRF resampling
# ---------------
# The CT is written as ``HU + 1000`` NIfTI in RAS coordinates so the
# resampler's zero padding becomes -1000 HU. The reference is an empty image
# with the target SPECT affine and the identity transform is deliberate. The
# resampled array is transferred to a STIR image on the vendor grid.


def resample_to_spect(ct_series, target_affine, umap_file, work_directory):
    """Resample CT HU onto the SPECT grid and return HU and a STIR template."""
    import nibabel as nib
    import sirf.Reg as Reg
    import sirf.STIR as STIR

    from simind_python_connector.utils import get_array

    work_directory = Path(work_directory)
    work_directory.mkdir(parents=True, exist_ok=True)
    ct_file = work_directory / "ct_hu.nii"
    reference_file = work_directory / "spect_reference.nii"

    ct_xyz = np.ascontiguousarray(ct_series.hu.transpose(2, 1, 0))
    ct_positive = np.clip(ct_xyz, -1000.0, 3000.0) + 1000.0
    nib.save(
        nib.Nifti1Image(ct_positive.astype(np.float32), LPS_TO_RAS @ ct_series.affine),
        str(ct_file),
    )
    nib.save(
        nib.Nifti1Image(
            np.zeros((MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE), np.float32),
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

    clone = STIR.ImageData(str(umap_file)).get_uniform_copy(0.0)
    clone.fill(np.ascontiguousarray(hu_grid, dtype=np.float32))
    if not np.allclose(get_array(clone), hu_grid, atol=1e-5):
        raise RuntimeError("The resampled HU values did not survive the STIR transfer")
    return hu_grid, clone


# %%
# Liver VOI
# ---------
# The liver insert wall is a closed high-HU contour. The interior is the largest
# closed component of the wall on each CT slice, excluding the wall itself, air,
# the lung domes and other compartments. The mask is generated once from the CT,
# before any reconstruction, and is never changed by a strategy.


def liver_interior_from_ct(series):
    """Return the CT liver interior on the native CT grid, excluding walls."""
    hu = series.hu
    z0 = series.slices[0]["position"][2]
    dz = series.affine[2, 2]
    body = hu > -400
    interior = np.zeros(hu.shape, dtype=bool)
    for k in range(hu.shape[0]):
        z = z0 + k * dz
        if not (-640.0 <= z <= -530.0):
            continue
        smooth = ndi.uniform_filter(hu[k], 5)
        outer = ndi.binary_fill_holes(ndi.binary_opening(body[k], iterations=2))
        inner_body = ndi.binary_erosion(outer, iterations=25)
        wall = (smooth > 60) & inner_body
        wall = ndi.binary_closing(wall, iterations=6)
        filled = ndi.binary_fill_holes(wall)
        candidate = ndi.binary_opening(filled & ~wall, iterations=3)
        labels, count = ndi.label(candidate)
        if count:
            sizes = ndi.sum(candidate, labels, range(1, count + 1))
            largest = int(np.argmax(sizes)) + 1
            if sizes[largest - 1] > 3000:
                interior[k] = labels == largest
    return interior


def resample_mask(mask, ct_affine, target_affine, supersample=4):
    """Resample a CT mask to fractional weights on the SPECT grid."""
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    axes = [np.arange(MATRIX_SIZE) for _ in range(3)]
    k, j, i = np.meshgrid(*axes, indexing="ij")
    inverse = np.linalg.inv(ct_affine)
    fraction = np.zeros((MATRIX_SIZE,) * 3, np.float32)
    for sz in offsets:
        for sy in offsets:
            for sx in offsets:
                z = target_affine[2, 2] * (k + sz) + target_affine[2, 3]
                y = target_affine[1, 1] * (j + sy) + target_affine[1, 3]
                x = target_affine[0, 0] * (i + sx) + target_affine[0, 3]
                ci = inverse[0, 0] * x + inverse[0, 3]
                cj = inverse[1, 1] * y + inverse[1, 3]
                ck = inverse[2, 2] * z + inverse[2, 3]
                fraction += ndi.map_coordinates(
                    mask.astype(np.float32), [ck, cj, ci], order=0, cval=0
                )
    return fraction / supersample**3


def sphere_fraction(centre_mm, radius_mm, target_affine, supersample=4):
    """Fractional occupancy of a sphere centred at a fixed patient position."""
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    axes = [np.arange(MATRIX_SIZE) for _ in range(3)]
    k, j, i = np.meshgrid(*axes, indexing="ij")
    fraction = np.zeros((MATRIX_SIZE,) * 3, np.float32)
    for sz in offsets:
        for sy in offsets:
            for sx in offsets:
                z = target_affine[2, 2] * (k + sz) + target_affine[2, 3]
                y = target_affine[1, 1] * (j + sy) + target_affine[1, 3]
                x = target_affine[0, 0] * (i + sx) + target_affine[0, 3]
                distance = np.sqrt(
                    (x - centre_mm[0]) ** 2
                    + (y - centre_mm[1]) ** 2
                    + (z - centre_mm[2]) ** 2
                )
                fraction += (distance <= radius_mm).astype(np.float32)
    return fraction / supersample**3


def liver_excluding_sphere(liver, sphere):
    """Remove the sphere occupancy from the liver weights."""
    return np.clip(np.asarray(liver, float) - np.asarray(sphere, float), 0.0, 1.0)


def hot_sphere_centroid(reference, liver, target_affine):
    """Weighted centroid within 8 voxels of the hottest liver voxel."""
    smooth = gaussian_filter(np.asarray(reference, float), 1.0)
    mask = np.asarray(liver, float) > 0.5
    peak = np.unravel_index(np.argmax(smooth), smooth.shape)
    axes = np.ogrid[:MATRIX_SIZE, :MATRIX_SIZE, :MATRIX_SIZE]
    distance2 = (
        (axes[0] - peak[0]) ** 2 + (axes[1] - peak[1]) ** 2 + (axes[2] - peak[2]) ** 2
    )
    background = np.median(
        reference[(distance2 >= 15**2) & (distance2 <= 25**2) & mask]
    )
    selected = (distance2 <= 8**2) & mask
    weights = np.maximum(np.asarray(reference, float) - background, 0.0) * selected
    voxels = np.array(np.nonzero(weights)).T
    values = weights[weights > 0]
    centroid = (voxels * values[:, None]).sum(axis=0) / values.sum()
    centre = np.array(
        [
            target_affine[0, 0] * centroid[2] + target_affine[0, 3],
            target_affine[1, 1] * centroid[1] + target_affine[1, 3],
            target_affine[2, 2] * centroid[0] + target_affine[2, 3],
        ]
    )
    return centre, np.asarray(centroid, float), tuple(int(v) for v in peak)


def mask_volume_ml(weights):
    """Volume represented by fractional voxel weights."""
    return float(np.asarray(weights, float).sum() * VOXEL_MM**3 / 1000.0)


def mask_centroid_mm(weights, target_affine):
    """Patient-coordinate centroid of fractional voxel weights."""
    weights = np.asarray(weights, float)
    axes = [np.arange(MATRIX_SIZE) for _ in range(3)]
    k, j, i = np.meshgrid(*axes, indexing="ij")
    total = weights.sum()
    x = (weights * (target_affine[0, 0] * i + target_affine[0, 3])).sum() / total
    y = (weights * (target_affine[1, 1] * j + target_affine[1, 3])).sum() / total
    z = (weights * (target_affine[2, 2] * k + target_affine[2, 3])).sum() / total
    return (float(x), float(y), float(z))


# %%
# Metrics
# -------
# The primary result is the weighted mean activity concentration in the sphere
# divided by the weighted mean in the liver VOI with the sphere removed. The
# voxel volume cancels, so the ratio needs no calibration. The truth is 4.0
# under the equal water-equivalent density approximation.
#
# The secondary result scales the image so the total in the fixed CT body
# support is 160.8 MBq and reports compartment activities and concentrations.
# That calibration is circular for the total, not for the split between liver
# and sphere. It is affected by recovery, spillover and the VOI uncertainty.
# The activities are those at the scan; no decay correction is applied, and
# +-2% is the supplied activity uncertainty only.


def weighted_mean(image, weights):
    """Weighted mean of an image over fractional voxel weights."""
    image = np.asarray(image, float)
    weights = np.asarray(weights, float)
    if image.shape != weights.shape:
        raise ValueError("image and weights must have the same shape")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("weights sum to zero")
    return float((image * weights).sum() / total)


def voi_ratio(image, sphere, liver):
    """Sphere-to-liver weighted concentration ratio; the voxel volume cancels."""
    image = np.asarray(image, float)
    numerator = weighted_mean(image, sphere)
    denominator = weighted_mean(image, liver)
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("nonpositive sphere or liver mean")
    return numerator / denominator


def calibrate_to_total(image, support, total_mbq=TOTAL_MBQ):
    """Scale a per-voxel image so the total in ``support`` is ``total_mbq``."""
    image = np.asarray(image, float)
    support = np.asarray(support, bool)
    activity = float(image[support].sum() * VOXEL_MM**3 / 1000.0)
    if activity <= 0.0:
        raise ValueError("nonpositive activity in the calibration support")
    scale = float(total_mbq) / activity
    return image * scale, scale


def compartment_metrics(image, sphere, liver, support):
    """Relative-calibration activities and concentrations."""
    scaled, scale = calibrate_to_total(image, support)
    voxel_ml = VOXEL_MM**3 / 1000.0
    sphere_ml = mask_volume_ml(sphere)
    liver_ml = mask_volume_ml(liver)
    sphere_mbq = float((scaled * sphere).sum() * voxel_ml)
    liver_mbq = float((scaled * liver).sum() * voxel_ml)
    return {
        "scale": scale,
        "body_total_mbq": float(scaled[support].sum() * voxel_ml),
        "sphere_ml": sphere_ml,
        "liver_ml": liver_ml,
        "sphere_mbq": sphere_mbq,
        "liver_mbq": liver_mbq,
        "sphere_kbq_g": sphere_mbq / sphere_ml * 1000.0,
        "liver_kbq_g": liver_mbq / liver_ml * 1000.0,
    }


def ratio_with_truth_mass(image, sphere, liver, truth_mass_g=LIVER_MASS_G):
    """The old ratio that divides liver activity by the whole-organ truth mass.

    The frozen liver VOI is about 978 mL against the 1149.49 g truth, so this
    denominator inflates the sphere-to-liver ratio by about 17.5%.
    """
    image = np.asarray(image, float)
    sphere_mean = weighted_mean(image, sphere)
    voxel_ml = VOXEL_MM**3 / 1000.0
    liver_activity = float((image * liver).sum() * voxel_ml)
    if sphere_mean <= 0.0 or liver_activity <= 0.0:
        raise ValueError("nonpositive sphere mean or liver activity")
    return sphere_mean / (liver_activity / truth_mass_g)


def liver_cov(image, liver):
    """Weighted spatial coefficient of variation in the liver VOI."""
    image = np.asarray(image, float)
    weights = np.asarray(liver, float)
    mean = weighted_mean(image, weights)
    if mean <= 0.0:
        raise ValueError("nonpositive liver mean")
    variance = float((weights * (image - mean) ** 2).sum() / weights.sum())
    return float(np.sqrt(variance) / mean)


def quantization_loss(source, quantization_scale, max_source=MAX_SOURCE):
    """Fraction of source activity in voxels rounded to zero histories."""
    source = np.asarray(source, float)
    total = float(source.sum())
    if total <= 0.0:
        return 0.0
    maximum = float(source.max())
    scaled = source / maximum * (max_source * quantization_scale)
    integer = np.clip(np.round(scaled), 0, max_source).astype(np.uint16)
    return float(source[integer == 0].sum() / total)


def source_quantization_scale(source):
    """Smallest preset scale that keeps the source quantization loss below 1%."""
    scale = QUANTIZATION_SCALE
    loss = quantization_loss(source, scale)
    while loss > MAX_QUANTIZATION_LOSS and scale < 1.0:
        scale = min(1.0, scale * 1.25)
        loss = quantization_loss(source, scale)
    return scale, loss


def smooth_scatter(array, fwhm_bins=SMOOTHING_FWHM_BINS, across_views=True):
    """Gaussian-smooth a projection along axial, bin and optionally view axes.

    The total is preserved. With ``across_views`` the view axis wraps, which
    assumes the views cover 360 degrees.
    """
    values = np.asarray(array, dtype=np.float64)
    sigma = [0.0] * values.ndim
    mode = ["reflect"] * values.ndim
    sigma[-3] = sigma[-1] = fwhm_bins / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if across_views:
        sigma[-2] = sigma[-1]
        mode[-2] = "wrap"
    smoothed = gaussian_filter(values, sigma, mode=mode)
    total = smoothed.sum()
    if total != 0.0:
        smoothed *= values.sum() / total
    return smoothed.astype(np.float32)


def select_views(array, step):
    """Select every ``step``-th view from a (tof, axial, view, bin) array."""
    array = np.asarray(array)
    if array.ndim != 4 or step < 1:
        raise ValueError("expected a 4-D projection array and step >= 1")
    if array.shape[2] % step != 0:
        raise ValueError("view count must be divisible by step")
    return array[:, :, ::step, :]


# %%
# Scatter strategies
# ------------------


@dataclass
class Additive:
    """One strategy's additive term and the numbers behind it."""

    values: np.ndarray
    scale: float | None
    zero_denominator_fraction: float


def build_additive(
    raw,
    measured_array,
    kind,
    reference_projection=None,
    smoothing_fwhm_bins=SMOOTHING_FWHM_BINS,
):
    """Build one strategy's additive term from raw SIMIND components.

    ``raw`` holds the unmodified ``all_interactions`` and
    ``geom_coll_primary`` arrays on the common physical count scale. The
    returned additive is a new array; the raw components are not modified.
    """
    total = np.asarray(raw["all_interactions"], dtype=np.float64)
    primary = np.asarray(raw["geom_coll_primary"], dtype=np.float64)
    scatter = total - primary
    scale = 1.0
    zero_fraction = 0.0

    if kind == "known":
        additive = scatter
    elif kind in ("reference_sum", "reference_trimmed"):
        if reference_projection is None:
            raise ValueError("reference strategies need a reference projection")
        method = "sum" if kind == "reference_sum" else "trimmed"
        scale = scale_to_reference(primary, reference_projection, method=method)
        additive = scatter * scale
    elif kind == "dewaraja_global":
        total_sum = float(total.sum())
        if total_sum <= 0.0:
            raise ValueError("SIMIND total is zero")
        scale = float(np.asarray(measured_array, float).sum() / total_sum)
        additive = scatter * scale
    elif kind == "pixelwise":
        denominator = total > 0.0
        zero_fraction = float((~denominator).mean())
        fraction = np.divide(
            scatter,
            total,
            out=np.zeros_like(scatter),
            where=denominator,
        )
        additive = np.asarray(measured_array, float) * fraction
        scale = None
    else:
        raise ValueError(f"unknown strategy {kind!r}")

    additive = smooth_scatter(additive, smoothing_fwhm_bins)
    return Additive(
        values=additive.astype(np.float32),
        scale=scale,
        zero_denominator_fraction=zero_fraction,
    )


def damped_additive(previous, estimate, damping=DAMPING, floor=1e-5):
    """Return a damped additive update; a zero previous term gives a half-strength first estimate when damping is 0.5."""  # noqa: E501
    previous = np.asarray(previous, float)
    estimate = np.asarray(estimate, float)
    if previous.shape != estimate.shape:
        raise ValueError("previous and estimate shapes differ")
    return np.maximum(previous * (1.0 - damping) + estimate * damping, floor)


# %%
# Reconstruction schedule
# -----------------------


@dataclass
class ScheduleResult:
    common_image: np.ndarray
    first_images: dict
    final_images: dict
    first_additives: dict
    final_additives: dict
    first_estimates: dict
    second_estimates: dict
    shared_raw: dict
    second_raws: dict
    ratios_first: dict
    ratios_final: dict
    simulate_calls: list = field(default_factory=list)
    update_counts: dict = field(default_factory=dict)
    budgets: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)


def run_schedule(
    initial_image,
    zero_additive,
    simulate,
    simulate_many,
    reconstruct,
    estimate,
    ratio,
    pass_subiterations=SUBITERATIONS_PER_PASS,
    damping=DAMPING,
):
    """Run the shared-refresh schedule.

    ``simulate(image, label)`` returns one raw-component dict. ``simulate_many``
    receives ``[(name, image), ...]`` for the divergent second refreshes and
    returns ``{name: raw}``. ``reconstruct(image, additive, subiterations)``
    returns a new image. ``estimate(name, raw, image, pass_index)`` returns the
    strategy's undamped additive. ``ratio(image)`` returns the primary metric.
    """
    common = reconstruct(initial_image, zero_additive, pass_subiterations)
    simulate_calls = ["shared"]
    shared_raw = simulate(common, "shared")

    first_images, first_additives, first_estimates = {}, {}, {}
    for name, _ in STRATEGIES:
        estimate_values = estimate(name, shared_raw, common, 0)
        first_estimates[name] = estimate_values
        first_additives[name] = damped_additive(
            np.zeros_like(estimate_values.values), estimate_values.values, damping
        )
        first_images[name] = reconstruct(
            common, first_additives[name], pass_subiterations
        )

    jobs = [(name, first_images[name]) for name, _ in STRATEGIES]
    second_raws = simulate_many(jobs)
    simulate_calls.extend(name for name, _ in STRATEGIES)

    final_images, final_additives, second_estimates = {}, {}, {}
    for name, _ in STRATEGIES:
        estimate_values = estimate(
            name, second_raws[name], first_images[name], len(simulate_calls)
        )
        second_estimates[name] = estimate_values
        final_additives[name] = damped_additive(
            first_additives[name], estimate_values.values, damping
        )
        final_images[name] = reconstruct(
            first_images[name], final_additives[name], pass_subiterations
        )

    baseline = reconstruct(common, zero_additive, 2 * pass_subiterations)
    final_images[BASELINE] = baseline
    final_additives[BASELINE] = np.asarray(zero_additive, float)

    update_counts = {name: NUM_UPDATES for name, _ in STRATEGIES}
    update_counts[BASELINE] = 0
    budgets = {name: pass_subiterations * (NUM_UPDATES + 1) for name, _ in STRATEGIES}
    budgets[BASELINE] = pass_subiterations * (NUM_UPDATES + 1)
    settings = {
        "num_subsets": NUM_SUBSETS,
        "pass_subiterations": pass_subiterations,
        "total_subiterations": pass_subiterations * (NUM_UPDATES + 1),
        "num_updates": NUM_UPDATES,
        "damping": damping,
    }
    return ScheduleResult(
        common_image=np.asarray(common, float),
        first_images=first_images,
        final_images=final_images,
        first_additives={name: value for name, value in first_additives.items()},
        final_additives={name: value for name, value in final_additives.items()},
        first_estimates=first_estimates,
        second_estimates=second_estimates,
        shared_raw=shared_raw,
        second_raws=second_raws,
        ratios_first={name: ratio(image) for name, image in first_images.items()},
        ratios_final={name: ratio(image) for name, image in final_images.items()},
        simulate_calls=simulate_calls,
        update_counts=update_counts,
        budgets=budgets,
        settings=settings,
    )


# %%
# SIRF and SIMIND run-time helpers
# --------------------------------


def _sirf():
    import sirf.STIR as sirf

    return sirf


def _image_from_array(array, reference=None):
    """Build a SIRF image holding ``array`` in (z, y, x) order."""
    if reference is not None:
        image = reference.get_uniform_copy(0.0)
    else:
        from simind_python_connector.builders import STIRSPECTImageDataBuilder

        shape = np.asarray(array).shape
        builder = STIRSPECTImageDataBuilder(
            {
                "!matrix size [1]": str(shape[2]),
                "!matrix size [2]": str(shape[1]),
                "!matrix size [3]": str(shape[0]),
                "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
                "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
                "scaling factor (mm/pixel) [3]": str(VOXEL_MM),
            },
            backend="sirf",
        )
        builder.set_pixel_array(np.asarray(array, dtype=np.float32))
        return builder.build()
    image.fill(np.ascontiguousarray(np.asarray(array, dtype=np.float32)))
    return image


def _acquisition_from_array(array, reference):
    """Build a SIRF AcquisitionData holding ``array`` on ``reference``."""
    additive = reference.get_uniform_copy(0.0)
    additive.fill(np.ascontiguousarray(np.asarray(array, dtype=np.float32)))
    return additive


def _make_model(acquisition_data, attenuation_image):
    """Return a fresh SPECTUB model with the same settings on every call."""
    sirf = _sirf()
    matrix = sirf.SPECTUBMatrix()
    matrix.set_attenuation_image(attenuation_image)
    matrix.set_resolution_model(0.0, 0.0, False)
    model = sirf.AcquisitionModelUsingMatrix(matrix)
    return model


def _run_osem(acquisition_data, model_factory, image, additive, subiterations):
    """One OSEM pass with a fixed additive term.

    A zero additive uses plain OSEM: installing an all-zero (or floored)
    additive term makes STIR take a much slower path on the full data.
    """
    from simind_python_connector.utils import get_array

    if np.allclose(np.asarray(additive), 0.0):
        sirf = _sirf()
        model = model_factory()
        objective = sirf.make_Poisson_loglikelihood(acquisition_data)
        objective.set_acquisition_model(model)
        objective.set_num_subsets(NUM_SUBSETS)
        reconstructor = sirf.OSMAPOSLReconstructor()
        reconstructor.set_objective_function(objective)
        reconstructor.set_num_subsets(NUM_SUBSETS)
        reconstructor.set_num_subiterations(subiterations)
        reconstructor.set_input(acquisition_data)
        reconstructor.set_up(image)
        reconstructor.reconstruct(image)
        return np.asarray(get_array(image), dtype=np.float32)

    from simind_python_connector.recon.osem import run_osem_with_corrections

    if not hasattr(additive, "clone"):
        additive = _acquisition_from_array(additive, acquisition_data)
    updater = AdditiveUpdater(None, UpdateSchedule(every=1), additive, damping=1.0)
    result = run_osem_with_corrections(
        acquisition_data,
        model_factory,
        image,
        updater,
        NUM_SUBSETS,
        subiterations,
        num_updates=0,
    )
    return np.asarray(get_array(result), dtype=np.float32)


# %%
# SIMIND job
# ----------


def _adaptor_settings(adaptor):
    """Apply the Y-90 PENETRATE acquisition settings to an adaptor."""
    adaptor.add_config_value(1, -150.0)
    adaptor.add_config_value(20, 225.0)
    adaptor.add_config_value(21, 75.0)
    adaptor.add_config_value(19, 2)
    adaptor.add_config_value(53, 1)
    adaptor.add_runtime_switch("FI", "y90")
    adaptor.add_runtime_switch("CC", "ma-mlegp")


def simulate_job(job):
    """Run one SIMIND simulation and save its raw components.

    This is a top-level function so a process pool can run several independent
    second-refresh realizations in parallel. Every job has its own output
    directory and output prefix.
    """
    from simind_python_connector import SirfSimindAdaptor, configs

    source = np.load(job["source"])
    density = np.load(job["density"])
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = _image_from_array(source)
    density_image = _image_from_array(density)
    adaptor = SirfSimindAdaptor(
        configs.get("AnyScan.yaml"),
        str(output_dir),
        job["prefix"],
        scoring_routine=ScoringRoutine.PENETRATE,
        quantization_scale=job["quantization_scale"],
        mu_map_type="density",
        photon_multiplier=job.get("nn", 1),
    )
    adaptor.set_mpi(None)
    _adaptor_settings(adaptor)
    projector = SimindProjector(
        adaptor,
        job["template"],
        density_image,
        activity_mbq=lambda image: job["activity_mbq"],
        time_per_projection_s=job["time_per_projection_s"],
        seed=job.get("seed"),
    )
    start = time.perf_counter()
    outputs = projector.project(source_image)
    seconds = time.perf_counter() - start
    loss = quantization_loss(source, job["quantization_scale"])
    sums = {}
    for key, value in outputs.items():
        sums[key] = float(value.sum())
        if key in ("all_interactions", "geom_coll_primary"):
            np.save(
                output_dir / f"{key}.npy",
                np.asarray(value.as_array(), dtype=np.float32),
            )
    return {
        "name": job["name"],
        "dir": str(output_dir),
        "seconds": seconds,
        "sums": sums,
        "source_sum": float(source.sum()),
        "quantization_loss": loss,
        "quantization_scale": job["quantization_scale"],
    }


def _load_raw(directory):
    """Load the raw components a job saved."""
    directory = Path(directory)
    raw = {
        "all_interactions": np.load(directory / "all_interactions.npy"),
        "geom_coll_primary": np.load(directory / "geom_coll_primary.npy"),
    }
    with open(directory / "components.json") as stream:
        raw["sums"] = json.load(stream)
    return raw


def _write_raw_sums(result):
    """Save the component totals next to the arrays."""
    with open(Path(result["dir"]) / "components.json", "w") as stream:
        json.dump(result["sums"], stream, indent=1)


def _job_from_image(name, image, template, density_file, output_root, **overrides):
    """Write a source map and return a job description."""
    prefix = overrides.get("prefix", name.lower())
    output_dir = Path(output_root) / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = output_dir / "source.npy"
    source = np.asarray(image, dtype=np.float32)
    np.save(source_file, source)
    scale, loss = source_quantization_scale(source)
    job = {
        "name": name,
        "source": str(source_file),
        "density": str(density_file),
        "template": str(template),
        "output_dir": str(output_dir),
        "prefix": prefix,
        "quantization_scale": scale,
        "quantization_loss": loss,
        "source_total": float(source.sum()),
        "activity_mbq": TOTAL_MBQ,
        "time_per_projection_s": TIME_PER_VIEW_S,
        "nn": 1,
        "seed": None,
    }
    job.update(overrides)
    return job


def simulate_single(name, image, template, density_file, output_root, **overrides):
    """Run one SIMIND realization in this process."""
    job = _job_from_image(name, image, template, density_file, output_root, **overrides)
    if job["quantization_loss"] > MAX_QUANTIZATION_LOSS:
        raise RuntimeError(
            f"{name}: {100 * job['quantization_loss']:.2f}% of the source activity "
            "is lost to quantization even at the maximum scale"
        )
    result = simulate_job(job)
    _write_raw_sums(result)
    raw = _load_raw(result["dir"])
    raw["result"] = result
    raw["source_sum"] = result["source_sum"]
    raw["quantization_loss"] = job["quantization_loss"]
    raw["quantization_scale"] = job["quantization_scale"]
    return raw


def simulate_many(jobs, workers=5, minimum_total=0.0):
    """Run several SIMIND realizations in parallel, one job per strategy.

    A source whose total is below ``minimum_total`` is treated as a collapsed
    image: its quantization loss is reported but not enforced, because it has
    no meaningful activity to quantize.
    """
    if len(jobs) == 1:
        name, image, template, density_file, output_root, overrides = jobs[0]
        return {
            name: simulate_single(
                name, image, template, density_file, output_root, **overrides
            )
        }
    prepared = []
    losses = {}
    for name, image, template, density_file, output_root, overrides in jobs:
        job = _job_from_image(
            name, image, template, density_file, output_root, **overrides
        )
        if (
            job["source_total"] >= minimum_total
            and job["quantization_loss"] > MAX_QUANTIZATION_LOSS
        ):
            raise RuntimeError(
                f"{name}: {100 * job['quantization_loss']:.2f}% of the source "
                "activity is lost to quantization even at the maximum scale"
            )
        prepared.append(job)
        losses[name] = job["quantization_loss"]
    try:
        context = get_context("fork")
    except ValueError:  # pragma: no cover - not available on every platform
        context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(prepared)),
        mp_context=context,
    ) as executor:
        futures = [executor.submit(simulate_job, job) for job in prepared]
        results = [future.result() for future in futures]
    outputs = {}
    for result in results:
        _write_raw_sums(result)
        raw = _load_raw(result["dir"])
        raw["result"] = result
        raw["source_sum"] = result["source_sum"]
        raw["quantization_loss"] = losses[result["name"]]
        raw["quantization_scale"] = result["quantization_scale"]
        outputs[result["name"]] = raw
    return outputs


# %%
# Templates and measured data
# ---------------------------


def reduced_template(source_header, destination, step):
    """Write a header with every ``step``-th view from the measured template."""
    lines = Path(source_header).read_text().splitlines()
    values = {}
    for line in lines:
        if ":=" in line:
            key, value = line.split(":=", 1)
            values[key.strip()] = value.strip()
    radii = [value.strip() for value in values["radii"].strip("{} ").split(",")]
    selected = radii[::step]
    if len(selected) * step < len(radii):
        raise ValueError("view selection does not cover the measured orbit")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = []
    for line in lines:
        if ":=" not in line:
            output.append(line)
            continue
        key, value = line.split(":=", 1)
        key = key.strip()
        if key == "!number of projections":
            value = str(len(selected))
        elif key == "radii":
            value = "{" + ",".join(selected) + "}"
        elif key == "!image duration (sec)[1]":
            value = str(len(selected) * TIME_PER_VIEW_S)
        elif key == "name of data file":
            value = destination.with_suffix(".s").name
        elif key == "orbit":
            value = "non-circular" if len(set(selected)) > 1 else "circular"
        elif key == "Radius":
            continue
        output.append(f"{key} := {value.strip()}")
    destination.write_text("\n".join(output) + "\n")
    return destination


def measured_subset(measured, template_header, step, destination):
    """Write every ``step``-th measured view and load it as SIRF data."""
    sirf = _sirf()
    array = select_views(measured.as_array(), step)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reduced_template(template_header, destination.with_suffix(".hs"), step)
    data = np.ascontiguousarray(array[0].transpose(1, 0, 2), dtype=np.float32)
    data.tofile(destination.with_suffix(".s"))
    return sirf.AcquisitionData(str(destination.with_suffix(".hs")))


# %%
# Figures


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _normalise(image, low, high):
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def save_registration_overlay(path, ct_map, umap, report):
    plt = _pyplot()
    ct = _normalise(ct_map, 0.0, 0.25)
    reference = _normalise(umap, 0.0, 0.25)
    planes = [
        ("axial k=64", ct[64], reference[64]),
        ("coronal j=64", ct[:, 64, :], reference[:, 64, :]),
        ("sagittal i=64", ct[:, :, 64], reference[:, :, 64]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (title, ct_plane, reference_plane) in zip(axes, planes):
        axis.imshow(np.dstack([ct_plane, reference_plane, np.zeros_like(ct_plane)]))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        "CT attenuation (red) and vendor umap (green): "
        f"{report['euclidean_voxels']:.3f} voxels, "
        f"{report['euclidean_mm']:.2f} mm ({report['status']})"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def save_voi_overlay(path, reference, liver, sphere):
    plt = _pyplot()
    contours = [(liver, "lime", "liver"), (sphere, "red", "sphere")]
    planes = [
        ("axial k=64", 0, 64),
        ("coronal j=54", 1, 54),
        ("sagittal i=46", 2, 46),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (title, orientation, index) in zip(axes, planes):
        if orientation == 0:
            image = reference[64]
            lines = [(mask[64], colour) for mask, colour, _ in contours]
        elif orientation == 1:
            image = reference[:, 54, :]
            lines = [(mask[:, 54, :], colour) for mask, colour, _ in contours]
        else:
            image = reference[:, :, 46]
            lines = [(mask[:, :, 46], colour) for mask, colour, _ in contours]
        axis.imshow(image, cmap="gray")
        for mask, colour in lines:
            axis.contour(mask > 0.5, levels=[0.5], colors=[colour], linewidths=1.2)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Frozen liver (green) and sphere (red) VOIs")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def save_reconstructions(path, images, titles, slice_index):
    plt = _pyplot()
    values = [image[slice_index] for image in images]
    high = max(float(np.percentile(value[value > 0], 99.5)) for value in values)
    columns = 3
    rows = int(np.ceil(len(values) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    for axis, title, value in zip(np.ravel(axes), titles, values):
        shown = axis.imshow(value, cmap="magma", vmin=0.0, vmax=high)
        axis.set_title(title)
        axis.axis("off")
    for axis in np.ravel(axes)[len(values) :]:
        axis.axis("off")
    figure.colorbar(shown, ax=np.ravel(axes).tolist(), shrink=0.7, label="activity")
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def save_ratios(path, ratios):
    plt = _pyplot()
    names = list(ratios)
    values = [ratios[name] for name in names]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(range(len(names)), values, color="C0")
    axis.axhline(TRUTH_RATIO, color="k", linestyle="--", label="truth 4.0")
    axis.set_xticks(range(len(names)), names, rotation=20, ha="right")
    axis.set_ylabel("sphere / liver concentration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def save_scatter(path, additives):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for index, (name, additive) in enumerate(additives.items()):
        per_view = np.asarray(additive)[0].sum(axis=(0, 2))
        axis.plot(per_view, color=f"C{index}", label=name, linewidth=1.0)
    axis.set_xlabel("view")
    axis.set_ylabel("scatter counts per view")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def save_profiles(path, images, centre_voxel):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    row = int(round(float(centre_voxel[1])))
    column = int(round(float(centre_voxel[2])))
    for index, (name, image) in enumerate(images.items()):
        axis.plot(image[:, row, column], color=f"C{index}", label=name)
    axis.set_xlabel("z voxel")
    axis.set_ylabel("activity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


# %%
# Run the tutorial
# ----------------


def clear_stale_outputs(output_directory):
    for name in (
        "summary.json",
        "registration.json",
        "registration.png",
        "vois.png",
        "reconstructions.png",
        "ratios.png",
        "scatter.png",
        "profiles.png",
        "UNACCEPTED.txt",
    ):
        (Path(output_directory) / name).unlink(missing_ok=True)


def _save_json(path, payload):
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=1)


def run():
    started = time.perf_counter()
    data_directory = fetch_y90_phantom()
    output_directory = (
        Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
        / "real_data"
        / "y90_mc_scatter"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    work_directory = output_directory / "work"
    simulations = output_directory / "simulations"
    clear_stale_outputs(output_directory)

    print("Reading the CT and building the patient-to-SPECT affine")
    ct_series = read_ct_series(data_directory / "ctac")
    target_affine = patient_to_spect_affine(data_directory / "peak.dcm", ct_series)
    print(
        f"CT volume: {ct_series.hu.shape}, HU {ct_series.hu.min():.0f} to "
        f"{ct_series.hu.max():.0f}"
    )
    print(f"SPECT LPS affine:\n{np.round(target_affine, 4)}")

    hu_grid, template_clone = resample_to_spect(
        ct_series, target_affine, data_directory / "umap_zoomed.hv", work_directory
    )
    coverage = ct_coverage(ct_series.affine, target_affine, ct_series.hu.shape)
    hu_grid[~coverage] = -1000.0
    umap = read_umap(data_directory)
    attenuation = hu_to_attenuation(hu_grid, ENERGY_KEV).astype(np.float32)
    region = registration_mask(coverage, hu_grid, umap, attenuation)
    registration = registration_report(attenuation, umap, region)
    registration["coverage_fraction"] = float(coverage.mean())
    registration["mask"] = {
        "definition": MASK_DEFINITION,
        "erosion": MASK_EROSION,
        "region_voxels": registration["region_voxels"],
    }
    _save_json(output_directory / "registration.json", registration)
    save_registration_overlay(
        output_directory / "registration.png", attenuation, umap, registration
    )
    print(
        "Registration: zero-offset Pearson "
        f"{registration['zero_pearson']:.4f}, best NCC "
        f"{registration['best_ncc']:.4f} at {registration['best_integer_zyx']}"
    )
    print(
        f"Euclidean offset: {registration['euclidean_voxels']:.3f} voxels = "
        f"{registration['euclidean_mm']:.2f} mm -> "
        f"{registration['status'].upper()}"
    )
    if not registration["passed"]:
        _save_json(output_directory / "summary.json", {"registration": registration})
        (output_directory / "UNACCEPTED.txt").write_text(
            "Registration did not meet the one-voxel gate.\n"
        )
    accept_registration(registration)

    density = hu_to_density_schneider(hu_grid).astype(np.float32)
    body_support = coverage & (hu_grid > -400)
    rows = np.indices((MATRIX_SIZE,) * 3)[1]
    body_support &= rows < TABLE_ROW
    print(
        f"Fixed body support: {int(body_support.sum())} voxels, "
        f"{mask_volume_ml(body_support):.1f} mL"
    )
    density_file = work_directory / "density.npy"
    np.save(density_file, density)

    print("Segmenting the liver from the CT")
    liver_ct = liver_interior_from_ct(ct_series)
    liver_fraction = resample_mask(liver_ct, ct_series.affine, target_affine)
    ct_liver_ml = float(
        liver_ct.sum() * abs(np.linalg.det(ct_series.affine[:3, :3])) / 1000.0
    )
    print(f"CT liver interior: {int(liver_ct.sum())} voxels, {ct_liver_ml:.1f} mL")
    print(f"Liver on the SPECT grid: {mask_volume_ml(liver_fraction):.1f} mL")
    # The phantom activity is confined to the liver insert. Restricting the
    # SIMIND source to the fixed CT liver VOI keeps the simulated scatter
    # representative and avoids spending histories on reconstruction noise in
    # the surrounding air. The secondary calibration still uses the full body.
    activity_support = liver_fraction > 0
    print(
        f"SIMIND source support: {int(activity_support.sum())} voxels, "
        f"{mask_volume_ml(activity_support):.1f} mL"
    )

    sirf = _sirf()
    sirf.AcquisitionData.set_storage_scheme("memory")
    measured = sirf.AcquisitionData(str(data_directory / "peak.hs"))
    zero_additive = np.zeros(measured.as_array().shape, dtype=np.float32)
    print(f"Measured projections: {measured.dimensions()}, sum {measured.sum():.1f}")

    umap_image = _image_from_array(umap[..., ::-1], reference=template_clone)
    ct_mu_image = _image_from_array(attenuation[..., ::-1], reference=template_clone)

    print("Generating the independent attenuation-only reference reconstruction")
    reference_start = time.perf_counter()
    reference_image = umap_image.get_uniform_copy(1.0)
    reference = _run_osem(
        measured,
        lambda: _make_model(measured, umap_image),
        reference_image,
        zero_additive,
        REFERENCE_SUBITERATIONS,
    )
    reference_seconds = time.perf_counter() - reference_start
    np.save(output_directory / "reference_recon.npy", reference)
    print(
        f"Reference: {reference_seconds:.1f} s, max {reference.max():.3f}, "
        f"sum {reference.sum():.1f}"
    )

    centre, centre_voxel, peak = hot_sphere_centroid(
        reference, liver_fraction, target_affine
    )
    distance = float(np.linalg.norm(centre - np.array(SPHERE_CENTRE_MM)))
    print(
        f"Reference hot-sphere centroid: {np.round(centre, 2)} mm at voxel "
        f"{np.round(centre_voxel, 2)} (peak {peak}); accepted centre "
        f"{SPHERE_CENTRE_MM}, difference {distance:.2f} mm"
    )
    if distance > SPHERE_VOXEL_TOLERANCE * VOXEL_MM:
        raise RuntimeError(
            "The reference reconstruction does not confirm the frozen sphere "
            f"centre ({distance:.1f} mm from the accepted position)"
        )

    sphere = sphere_fraction(SPHERE_CENTRE_MM, SPHERE_RADIUS_MM, target_affine)
    liver = liver_excluding_sphere(liver_fraction, sphere)
    sphere_ml = mask_volume_ml(sphere)
    liver_ml = mask_volume_ml(liver)
    print(
        f"Sphere: {sphere_ml:.3f} mL, centroid "
        f"{np.round(mask_centroid_mm(sphere, target_affine), 2)} mm"
    )
    print(
        f"Liver ex sphere: {liver_ml:.2f} mL, centroid "
        f"{np.round(mask_centroid_mm(liver, target_affine), 2)} mm"
    )
    if abs(liver_ml - LIVER_CT_VOLUME_ML) > 1.0:
        raise RuntimeError(
            f"the live liver VOI ({liver_ml:.2f} mL) differs from the frozen "
            f"value ({LIVER_CT_VOLUME_ML:.2f} mL)"
        )
    if abs(sphere_ml - SPHERE_VOLUME_ML) > 0.05:
        raise RuntimeError(
            f"the live sphere VOI ({sphere_ml:.3f} mL) differs from the fixed "
            f"volume ({SPHERE_VOLUME_ML:.3f} mL)"
        )
    print(
        f"True liver mass {LIVER_MASS_G:.2f} g; the CT VOI is "
        f"{100 * (LIVER_MASS_G - liver_ml) / LIVER_MASS_G:.1f}% smaller "
        "under the 1 g/mL approximation"
    )
    print(
        "Dividing sampled liver activity by 1149.49 g instead of the VOI volume "
        f"would inflate the ratio by {LIVER_MASS_G / liver_ml:.3f}x "
        f"(4.0 -> {TRUTH_RATIO * LIVER_MASS_G / liver_ml:.2f})"
    )
    save_voi_overlay(output_directory / "vois.png", reference, liver, sphere)
    _save_json(
        output_directory / "vois.json",
        {
            "liver_ct_volume_ml": float(
                liver_ct.sum() * abs(np.linalg.det(ct_series.affine[:3, :3])) / 1000.0
            ),
            "liver_spect_volume_ml": liver_ml,
            "liver_spect_volume_with_sphere_ml": mask_volume_ml(liver_fraction),
            "liver_centroid_mm": mask_centroid_mm(liver, target_affine),
            "sphere_volume_ml": sphere_ml,
            "sphere_radius_mm": SPHERE_RADIUS_MM,
            "sphere_centroid_mm": mask_centroid_mm(sphere, target_affine),
            "sphere_centre_accepted_mm": SPHERE_CENTRE_MM,
            "sphere_centre_reference_mm": centre.tolist(),
            "sphere_centre_difference_mm": distance,
            "sphere_provenance": (
                "Fixed 8.08 mL sphere at the centre from the independent "
                "attenuation-only reference reconstruction; not CT-visible"
            ),
            "liver_provenance": (
                "CT liver insert interior, walls and other compartments excluded"
            ),
        },
    )

    print("Building the reconstruction model and running the common pass")
    template = str(data_directory / "peak.hs")
    initial_image = np.ones((MATRIX_SIZE,) * 3, dtype=np.float32)

    def ratio(image):
        return voi_ratio(image, sphere, liver)

    full_model = _make_model(measured, ct_mu_image)
    full_model.set_up(measured, _image_from_array(initial_image, template_clone))

    schedule_dir = simulations / "second_refresh"

    shared_source_total = [0.0]

    def simulate(image, label):
        source = np.asarray(image, float) * activity_support
        shared_source_total[0] = float(source.sum())
        raw = simulate_single(
            "shared_refresh", source, template, density_file, simulations
        )
        return raw

    def simulate_second(jobs):
        prepared = []
        for name, image in jobs:
            source = np.asarray(image, float) * activity_support
            prepared.append(
                (
                    name,
                    source,
                    template,
                    density_file,
                    schedule_dir,
                    {"prefix": SLUGS[name]},
                )
            )
        return simulate_many(
            prepared,
            workers=5,
            minimum_total=0.01 * shared_source_total[0],
        )

    def estimate(name, raw, image, _pass_index):
        source = np.asarray(image, float) * activity_support
        reference_projection = None
        kind = dict(STRATEGIES)[name]
        if kind in ("reference_sum", "reference_trimmed"):
            source_image = _image_from_array(source, reference=template_clone)
            reference_projection = np.asarray(
                full_model.direct(source_image).as_array(), dtype=float
            )
        return build_additive(
            raw,
            measured.as_array(),
            kind,
            reference_projection=reference_projection,
        )

    def model_factory():
        return _make_model(measured, ct_mu_image)

    def reconstruct(image, additive, subiterations):
        additive_image = _acquisition_from_array(additive, measured)
        return _run_osem(
            measured,
            model_factory,
            _image_from_array(image, reference=template_clone),
            additive_image,
            subiterations,
        )

    result = run_schedule(
        initial_image,
        zero_additive,
        simulate,
        simulate_second,
        reconstruct,
        estimate,
        ratio,
    )
    common_image = result.common_image
    final_images = result.final_images
    first_additives = result.first_additives
    final_additives = result.final_additives
    ratios_first = result.ratios_first
    ratios_final = result.ratios_final
    first_refresh_seconds = result.shared_raw["result"]["seconds"]
    print(
        f"Shared first-refresh SIMIND run: {first_refresh_seconds:.0f} s, "
        f"quantization loss "
        f"{100 * result.shared_raw['quantization_loss']:.2f}%"
    )
    second_seconds = {
        name: result.second_raws[name]["result"]["seconds"] for name, _ in STRATEGIES
    }
    print("Second-refresh SIMIND runs (parallel):")
    for name, seconds in second_seconds.items():
        print(f"  {name}: {seconds:.0f} s")
    print(
        "Schedule: six SIMIND runs, two updates per corrected strategy, none "
        "for the baseline, 144 subiterations each"
    )

    # %%
    # Metrics

    strategy_metrics = {}
    for name, _ in STRATEGIES:
        metrics = compartment_metrics(final_images[name], sphere, liver, body_support)
        first_estimate = result.first_estimates[name]
        second_estimate = result.second_estimates[name]
        metrics.update(
            {
                "ratio": ratios_final[name],
                "ratio_first_refresh": ratios_first[name],
                "stability": abs(ratios_final[name] - ratios_first[name])
                / ratios_first[name],
                "liver_cov": liver_cov(final_images[name], liver),
                "scatter_fraction_first": float(
                    first_additives[name].sum() / measured.sum()
                ),
                "scatter_fraction_final": float(
                    final_additives[name].sum() / measured.sum()
                ),
                "normalisation_scale_first": first_estimate.scale,
                "normalisation_scale_final": second_estimate.scale,
                "zero_denominator_fraction": second_estimate.zero_denominator_fraction,
                "source_sum": result.second_raws[name]["source_sum"],
                "quantization_loss": result.second_raws[name]["quantization_loss"],
                "quantization_scale": result.second_raws[name].get(
                    "quantization_scale", QUANTIZATION_SCALE
                ),
                "component_sums": result.second_raws[name]["sums"],
                "simind_seconds": second_seconds[name],
            }
        )
        strategy_metrics[name] = metrics

    baseline_metrics = compartment_metrics(
        final_images[BASELINE], sphere, liver, body_support
    )
    baseline_metrics.update(
        {
            "ratio": ratios_final[BASELINE],
            "liver_cov": liver_cov(final_images[BASELINE], liver),
            "scatter_fraction_final": 0.0,
        }
    )
    strategy_metrics[BASELINE] = baseline_metrics

    print("Primary metric (sphere/liver concentration, truth 4.0):")
    for name, values in strategy_metrics.items():
        print(
            f"  {name}: {values['ratio']:.3f} "
            f"(first refresh {values.get('ratio_first_refresh', float('nan')):.3f})"
        )
    old_ratio = ratio_with_truth_mass(final_images["Reference, sum"], sphere, liver)
    print(
        "  Reference, sum with the old whole-organ 1149.49 g denominator: "
        f"{old_ratio:.2f}"
    )

    # %%
    # Secondary relative calibration
    print("Relative calibration (body total set to 160.8 MBq):")
    for name, values in strategy_metrics.items():
        print(
            f"  {name}: sphere {values['sphere_mbq']:.3f} MBq "
            f"({values['sphere_kbq_g']:.1f} kBq/g), liver "
            f"{values['liver_mbq']:.2f} MBq ({values['liver_kbq_g']:.1f} kBq/g), "
            f"total {values['body_total_mbq']:.2f} MBq"
        )
    print(
        f"  truth: sphere {SPHERE_MBQ:.2f} MBq ({SPHERE_KBQ_G:.1f} kBq/g), "
        f"liver {LIVER_MBQ:.2f} MBq ({LIVER_KBQ_G:.1f} kBq/g), "
        f"total {TOTAL_MBQ:.1f} MBq (+-2% activity uncertainty)"
    )

    # %%
    # Statistical adequacy
    print("Statistical adequacy")
    minimum_source = 0.01 * result.shared_raw["source_sum"]
    valid_names = [
        name
        for name, _ in STRATEGIES
        if strategy_metrics[name]["source_sum"] >= minimum_source
    ]
    collapsed_names = [name for name, _ in STRATEGIES if name not in valid_names]
    quantization_values = [result.shared_raw["quantization_loss"]]
    quantization_values.extend(
        strategy_metrics[name]["quantization_loss"] for name in valid_names
    )
    max_quantization = max(quantization_values)
    if collapsed_names:
        print(
            "  collapsed source(s), quantization gate not applicable: "
            + ", ".join(collapsed_names)
        )
    print(
        f"  maximum source quantization loss: {100 * max_quantization:.2f}% "
        f"(gate < {100 * MAX_QUANTIZATION_LOSS:.0f}%)"
    )

    probe_step = 2
    probe_template = reduced_template(
        data_directory / "peak.hs",
        simulations / "probes" / "probe_template.hs",
        probe_step,
    )
    probe_measured = measured_subset(
        measured,
        data_directory / "peak.hs",
        probe_step,
        simulations / "probes" / "probe_measured",
    )
    probe_model = _make_model(probe_measured, ct_mu_image)
    probe_model.set_up(probe_measured, _image_from_array(initial_image, template_clone))

    frozen_source = common_image * activity_support
    print("  independent-seed and history-doubling probes on the frozen source")
    seed_jobs = [
        (
            f"seed_{seed}",
            frozen_source,
            template,
            density_file,
            simulations / "probes",
            {"seed": seed},
        )
        for seed in (101, 202)
    ]
    nn_jobs = [
        (
            "nn_1",
            frozen_source,
            probe_template,
            density_file,
            simulations / "probes",
            {"nn": 1},
        ),
        (
            "nn_2",
            frozen_source,
            probe_template,
            density_file,
            simulations / "probes",
            {"nn": 2},
        ),
    ]
    probe_raws = simulate_many(seed_jobs + nn_jobs, workers=4)
    seed_raws = {
        name: raw for name, raw in probe_raws.items() if name.startswith("seed_")
    }
    nn_raws = {name: raw for name, raw in probe_raws.items() if name.startswith("nn_")}
    seed_scatter = {}
    for name, raw in seed_raws.items():
        seed_scatter[name] = smooth_scatter(
            np.asarray(raw["all_interactions"], float)
            - np.asarray(raw["geom_coll_primary"], float)
        )
    seed_names = sorted(seed_scatter)
    first_seed, second_seed = seed_names
    scatter_one = seed_scatter[first_seed]
    scatter_two = seed_scatter[second_seed]
    seed_total_difference = abs(float(scatter_one.sum() - scatter_two.sum())) / float(
        0.5 * (scatter_one.sum() + scatter_two.sum())
    )
    seed_l1 = float(
        np.abs(scatter_one - scatter_two).sum()
        / (0.5 * (scatter_one.sum() + scatter_two.sum()))
    )
    print(
        f"  seed scatter total difference: {100 * seed_total_difference:.2f}% "
        "(gate <= 2%)"
    )
    print(f"  seed smoothed-scatter relative L1: {100 * seed_l1:.2f}% (gate <= 5%)")
    nn_ratios = {1: {}, 2: {}}
    for nn, raw in ((1, nn_raws["nn_1"]), (2, nn_raws["nn_2"])):
        for name, kind in STRATEGIES:
            source = np.asarray(common_image, float) * activity_support
            reference_projection = None
            if kind in ("reference_sum", "reference_trimmed"):
                reference_projection = np.asarray(
                    probe_model.direct(
                        _image_from_array(source, reference=template_clone)
                    ).as_array(),
                    dtype=float,
                )
            additive = build_additive(
                raw,
                probe_measured.as_array(),
                kind,
                reference_projection=reference_projection,
            )
            image = _run_osem(
                probe_measured,
                lambda: _make_model(probe_measured, ct_mu_image),
                _image_from_array(common_image, reference=template_clone),
                _acquisition_from_array(additive.values, probe_measured),
                2 * SUBITERATIONS_PER_PASS,
            )
            nn_ratios[nn][name] = ratio(image)
    nn_changes = {
        name: abs(nn_ratios[2][name] - nn_ratios[1][name]) / nn_ratios[1][name]
        for name, _ in STRATEGIES
    }
    print(
        "  NN=1 / NN=2 final-pass ratios: "
        + ", ".join(
            f"{name} {nn_ratios[1][name]:.3f}/{nn_ratios[2][name]:.3f}"
            for name, _ in STRATEGIES
        )
    )
    print(
        f"  largest NN ratio change: {100 * max(nn_changes.values()):.2f}% (gate <= 2%)"
    )

    print("  iteration-extension probe (two full iterations with the final term)")
    iteration_changes = {}
    for name, _ in STRATEGIES:
        extended = reconstruct(
            final_images[name], final_additives[name], 2 * NUM_SUBSETS
        )
        iteration_changes[name] = (
            abs(ratio(extended) - ratios_final[name]) / ratios_final[name]
        )
    print(
        "  largest ratio change after two more iterations: "
        f"{100 * max(iteration_changes.values()):.2f}% (gate <= 2%)"
    )

    print("  two-refresh stability (first versus second refresh):")
    for name, _ in STRATEGIES:
        print(
            f"    {name}: {100 * strategy_metrics[name]['stability']:.2f}% (gate <= 5%)"
        )

    # %%
    # Scatter fractions and inter-strategy differences
    print("Scatter fractions of the measured counts:")
    for name, _ in STRATEGIES:
        first = strategy_metrics[name]["scatter_fraction_first"]
        final = strategy_metrics[name]["scatter_fraction_final"]
        print(f"  {name}: first {100 * first:.1f}%, final {100 * final:.1f}%")
    differences = {}
    for first, _ in STRATEGIES:
        row = {}
        first_values = final_additives[first]
        first_norm = first_values / first_values.sum()
        for second, _ in STRATEGIES:
            second_values = final_additives[second]
            second_norm = second_values / second_values.sum()
            row[second] = float(np.abs(first_norm - second_norm).sum())
        differences[first] = row
    print("Normalised final scatter-estimate differences (relative L1):")
    header = "  " + " ".join(f"{name[:10]:>10}" for name, _ in STRATEGIES)
    print(header)
    for first, _ in STRATEGIES:
        print(
            f"  {first[:18]:<18} "
            + " ".join(
                f"{differences[first][second]:10.3f}" for second, _ in STRATEGIES
            )
        )

    # %%
    # Figures
    save_reconstructions(
        output_directory / "reconstructions.png",
        [final_images[name] for name, _ in STRATEGIES] + [final_images[BASELINE]],
        [name for name, _ in STRATEGIES] + [BASELINE],
        64,
    )
    save_ratios(
        output_directory / "ratios.png",
        {name: strategy_metrics[name]["ratio"] for name, _ in STRATEGIES}
        | {BASELINE: ratios_final[BASELINE]},
    )
    save_scatter(
        output_directory / "scatter.png",
        {name: final_additives[name] for name, _ in STRATEGIES},
    )
    save_profiles(
        output_directory / "profiles.png",
        {name: final_images[name] for name, _ in STRATEGIES}
        | {BASELINE: final_images[BASELINE]},
        centre_voxel,
    )

    # %%
    # Gate summary and summary file
    stability_gate = max(strategy_metrics[name]["stability"] for name in valid_names)
    nn_gate = max(nn_changes[name] for name in valid_names)
    iteration_gate = max(iteration_changes[name] for name in valid_names)
    gates = {
        "registration": registration["passed"],
        "registration_voxels": registration["euclidean_voxels"],
        "max_quantization_loss": max_quantization,
        "quantization_passed": max_quantization < MAX_QUANTIZATION_LOSS,
        "seed_scatter_total_difference": seed_total_difference,
        "seed_scatter_l1": seed_l1,
        "seed_passed": seed_total_difference <= 0.02 and seed_l1 <= 0.05,
        "nn_max_ratio_change": nn_gate,
        "nn_passed": nn_gate <= 0.02,
        "iteration_max_ratio_change": iteration_gate,
        "iteration_passed": iteration_gate <= 0.02,
        "stability_max_ratio_change": stability_gate,
        "stability_passed": stability_gate <= 0.05,
        "valid_strategies": valid_names,
        "collapsed_strategies": collapsed_names,
    }
    all_passed = all(
        gates[key]
        for key in (
            "registration",
            "quantization_passed",
            "seed_passed",
            "nn_passed",
            "iteration_passed",
            "stability_passed",
        )
    )

    elapsed = time.perf_counter() - started
    summary = {
        "status": "accepted" if all_passed else "quality_not_established",
        "elapsed_s": elapsed,
        "quantization_scale": QUANTIZATION_SCALE,
        "registration": registration,
        "vois": json.load(open(output_directory / "vois.json")),
        "schedule": {
            "passes": PASSES,
            "num_updates": NUM_UPDATES,
            "num_subsets": NUM_SUBSETS,
            "subiterations_per_pass": SUBITERATIONS_PER_PASS,
            "total_subiterations": TOTAL_SUBITERATIONS,
            "damping": DAMPING,
            "simind_runs": 6,
            "simind_probe_runs": 4,
            "shared_refresh_seconds": first_refresh_seconds,
            "shared_quantization_scale": result.shared_raw.get(
                "quantization_scale", QUANTIZATION_SCALE
            ),
            "second_refresh_seconds": second_seconds,
        },
        "strategies": {
            name: {
                **{key: value for key, value in values.items() if key not in ("scale",)}
            }
            for name, values in strategy_metrics.items()
        },
        "truth": {
            "total_mbq": TOTAL_MBQ,
            "liver_mbq": LIVER_MBQ,
            "sphere_mbq": SPHERE_MBQ,
            "liver_mass_g": LIVER_MASS_G,
            "sphere_mass_g": SPHERE_MASS_G,
            "liver_kbq_g": LIVER_KBQ_G,
            "sphere_kbq_g": SPHERE_KBQ_G,
            "ratio": TRUTH_RATIO,
            "activity_uncertainty": 0.02,
            "no_decay_correction": True,
        },
        "gates": gates,
        "valid_strategies": valid_names,
        "collapsed_strategies": collapsed_names,
        "scatter_differences": differences,
        "nn_ratios": {"nn_1": nn_ratios[1], "nn_2": nn_ratios[2]},
        "iteration_ratio_changes": iteration_changes,
        "seed_scatter_totals": {
            name: float(seed_scatter[name].sum()) for name in seed_names
        },
    }
    _save_json(output_directory / "summary.json", summary)
    if not all_passed:
        (output_directory / "UNACCEPTED.txt").write_text(
            "Quality not established: one or more statistical gates failed.\n"
            + json.dumps(gates, indent=1)
            + "\n"
        )
    print(f"Total wall time: {elapsed / 60.0:.1f} min")
    print(f"Summary written to {output_directory}")
    if not all_passed:
        raise RuntimeError(
            "Statistical adequacy was not established; see "
            f"{output_directory / 'UNACCEPTED.txt'}"
        )
    _pyplot().show()


if __name__ == "__main__":
    run()
