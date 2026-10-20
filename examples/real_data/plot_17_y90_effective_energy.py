"""
Y-90 effective attenuation energy
=================================

Y-90 is a pure beta emitter: the 75-225 keV imaging window contains a broad
bremsstrahlung and scatter continuum, not a photopeak. A single "150 keV"
attenuation map is therefore an approximation. This tutorial compares it with
a map weighted by the simulated detected spectrum.

The tutorial fetches the measured Y-90 phantom, resamples the CT onto the
SPECT grid with SIRF, reads the installed Y-90 source spectrum from SIMIND,
runs a short list-mode acquisition for the detected spectrum, and reconstructs
the measured projection data twice with the same settings and two attenuation
maps.

The detected spectrum comes from SIMIND's list-mode output. It contains the
crystal deposited energy, including the detector energy resolution (a 140.5 keV
monoenergetic control has a 20.5 keV FWHM), and it is not separated into the
PENETRATE interaction components. It is a simulated spectrum, not measured
spectroscopy.

The spectrum-weighted attenuation map is a global effective-map approximation:
one map, used for every projection, that reproduces the average attenuation of
the detected spectrum. Both reconstructions omit scatter and are presented to
show the effect of the attenuation map only, not as an accuracy ranking.
"""

# %%
# Setup
# -----
# The tutorial needs the Y-90 phantom data (set
# ``SIMIND_CONNECTOR_Y90_DATA``), SIMIND, SIRF, NumPy, matplotlib, pydicom and
# nibabel. Outputs go to ``$SIMIND_EXAMPLES_OUTPUT`` or ``output/``.

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simind_python_connector import configs
from simind_python_connector.connectors import SimindPythonConnector
from simind_python_connector.converters.attenuation import (
    hu_to_attenuation,
    hu_to_density_schneider,
)
from simind_python_connector.datasets import fetch_y90_phantom
from simind_python_connector.utils.interfile import ProjectionGeometry


VOXEL_SIZE_MM = 4.2578
MATRIX_SIZE = 128
GRID_CENTRE = (MATRIX_SIZE - 1) / 2.0
REFERENCE_ENERGY_KEV = 150.0
WINDOW_LOW_KEV = 75.0
WINDOW_HIGH_KEV = 225.0
SPECTRUM_VIEWS = 24
CHANNEL_KEV = 4.0
# The hot sphere is not visible on CT. This centre was established once from an
# attenuation-only reconstruction of the measured data, before any comparison.
HOT_SPHERE_CENTRE_MM = (-77.43, -181.69, -576.20)
HOT_SPHERE_RADIUS_MM = 12.449
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

# %%
# Isotope spectra
# ---------------
# SIMIND reads source spectra from ``.isd`` files in its ``smc_dir``. A
# ``CONTINUOUS`` header declares a continuum sampled on an energy grid; without
# it the file lists discrete gamma lines. The ``y90_frey`` model names a
# companion ``y90.cpd`` file that is not part of the SIMIND installation, and
# the NUDAT file is a single 2186.2 keV line outside the imaging window.


@dataclass
class IsotopeSpectrum:
    """One ``.isd`` spectrum: energies in keV, relative yields and headers."""

    energies_kev: np.ndarray
    yields: np.ndarray
    continuous: bool
    continuous_scale: float | None
    bremsstrahlung: str | None
    half_life: float | None

    @property
    def kind(self) -> str:
        return "continuum" if self.continuous else "discrete"


def read_isotope_spectrum(path):
    """Read a SIMIND ``.isd`` file into energies, yields and its header keys."""
    energies, yields = [], []
    header = {}
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip().lower()] = value.strip()
            continue
        if ":" in line and not line[0].isdigit() and line[0] not in "+-.":
            key, value = line.split(":", 1)
            header[key.strip().lower()] = value.strip()
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected two columns, got {line!r}")
        try:
            energy, value = (float(field) for field in fields)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        energies.append(energy)
        yields.append(value)

    if not energies:
        raise ValueError(f"{path}: no spectrum points found")
    energies = np.asarray(energies, dtype=float)
    yields = np.asarray(yields, dtype=float)
    if not np.isfinite(energies).all() or not np.isfinite(yields).all():
        raise ValueError(f"{path}: spectrum contains non-finite values")
    if (energies < 0).any() or (yields < 0).any():
        raise ValueError(f"{path}: energies and yields must be non-negative")
    if np.any(np.diff(energies) <= 0):
        raise ValueError(f"{path}: energies must be strictly increasing")

    continuous = "continuous" in header
    scale = header.get("continuous")
    if scale is not None:
        try:
            scale = float(scale)
        except ValueError as error:
            raise ValueError(f"{path}: invalid CONTINUOUS value {scale!r}") from error
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{path}: CONTINUOUS must be positive, got {scale!r}")
    half_life = header.get("halflife")
    return IsotopeSpectrum(
        energies_kev=energies,
        yields=yields,
        continuous=continuous,
        continuous_scale=scale,
        bremsstrahlung=header.get("bremsstrahlung"),
        half_life=float(half_life) if half_life is not None else None,
    )


def find_isotope_file(name):
    """Locate an isotope file in the installed SIMIND data directories."""
    directories = []
    for variable in ("SIMIND_SMC_DIR", "SMC_DIR", "SIMIND_DATA_DIR"):
        value = os.environ.get(variable)
        if value:
            directories.append(Path(value).expanduser())
    root = os.environ.get("SIMIND_ROOT")
    if root:
        directories.extend([Path(root) / "smc_dir", Path(root) / "tutorial"])
    binary = os.environ.get("SIMIND_BIN") or "simind"
    resolved = shutil.which(binary) or binary
    if resolved:
        base = Path(resolved).expanduser().resolve().parent
        directories.extend([base / "smc_dir", base / "tutorial"])
    for directory in directories:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {name} in SIMIND's data directories. Set SIMIND_ROOT "
        "or SIMIND_SMC_DIR to the installed SIMIND tree."
    )


# %%
# List-mode records
# -----------------
# SIMIND writes the ``/OU:11`` list-mode file as headerless 16-byte
# little-endian records: three crystal coordinates in cm x 100, the crystal
# energy in keV x 10, and the event weight. A negative weight marks the first
# event of a new projection. The weight magnitude is kept.


LIST_MODE_DTYPE = np.dtype(
    [
        ("x", "<i2"),
        ("y", "<i2"),
        ("z", "<i2"),
        ("energy", "<i2"),
        ("weight", "<f8"),
    ]
)


@dataclass
class ListModeEvents:
    """Decoded list-mode events: energy, weight, projection and crystal position."""

    energies_kev: np.ndarray
    weights: np.ndarray
    projections: np.ndarray
    crystal_cm: np.ndarray


def read_list_mode(path):
    """Read a ``/OU:11`` list-mode file written by SIMIND."""
    path = Path(path)
    size = path.stat().st_size
    if size == 0 or size % LIST_MODE_DTYPE.itemsize:
        raise ValueError(
            f"{path}: size {size} is not a whole number of "
            f"{LIST_MODE_DTYPE.itemsize}-byte records"
        )
    records = np.fromfile(path, dtype=LIST_MODE_DTYPE)
    weight = records["weight"].astype(float)
    projections = np.cumsum(weight < 0)
    # The first record belongs to projection 0 whether or not it carries the
    # negative marker.
    projections = projections - projections[0]
    return ListModeEvents(
        energies_kev=records["energy"].astype(float) / 10.0,
        weights=np.abs(weight),
        projections=projections,
        crystal_cm=np.column_stack(
            [records[name].astype(float) for name in ("x", "y", "z")]
        )
        / 100.0,
    )


def histogram_spectrum(energies_kev, weights, bin_width_kev=CHANNEL_KEV):
    """Return the weighted histogram and its bin edges over 0-2048 keV."""
    edges = np.arange(0.0, 2048.0 + bin_width_kev, bin_width_kev)
    counts, _ = np.histogram(energies_kev, bins=edges, weights=weights)
    return counts, edges


def window_weight(energies_kev, weights, lower=WINDOW_LOW_KEV, upper=WINDOW_HIGH_KEV):
    """Return the weight of the events inside the energy window."""
    inside = (energies_kev >= lower) & (energies_kev <= upper)
    return float(np.sum(weights[inside]))


def window_channel_weights(weights, edges, lower=WINDOW_LOW_KEV, upper=WINDOW_HIGH_KEV):
    """Clip channel weights to a window, splitting channels that cross an edge.

    A channel that only partly overlaps the window contributes the overlapping
    fraction of its weight, at the midpoint of the overlap. The alternative,
    counting the whole boundary channel, would bias the low-energy end of the
    window.
    """
    weights = np.asarray(weights, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if weights.shape != (edges.size - 1,):
        raise ValueError("weights must have one value per bin edge interval")
    if np.any(weights < 0):
        raise ValueError("spectrum weights must be non-negative")
    width = np.diff(edges)
    if np.any(width <= 0):
        raise ValueError("bin edges must be strictly increasing")

    low = np.maximum(edges[:-1], lower)
    high = np.minimum(edges[1:], upper)
    overlap = np.clip(high - low, 0.0, None)
    fraction = overlap / width
    clipped = weights * fraction
    keep = fraction > 0
    return 0.5 * (low[keep] + high[keep]), clipped[keep]


def weighted_mean_energy(energies_kev, weights):
    """Return the weight-averaged energy in keV."""
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        raise ValueError("cannot average energies with zero total weight")
    return float(np.average(energies_kev, weights=weights))


def weighted_attenuation(hu, energies_kev, weights, converter=hu_to_attenuation):
    """Return ``sum_j w_j mu(HU, E_j) / sum_j w_j`` for one energy per channel."""
    hu = np.asarray(hu, dtype=float)
    energies_kev = np.asarray(energies_kev, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if energies_kev.shape != weights.shape:
        raise ValueError("energies and weights must have the same shape")
    if np.any(weights < 0):
        raise ValueError("spectrum weights must be non-negative")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cannot weight attenuation with zero total weight")

    result = np.zeros_like(hu)
    for energy, weight in zip(energies_kev, weights):
        if weight:
            result += weight * converter(hu, float(energy))
    return result / total


# %%
# CT preparation
# --------------
# The CT slice files are not in spatial order, so they are sorted by
# ``ImagePositionPatient``. The volume is written as a NIfTI image in RAS
# coordinates. The reconstruction grid is the vendor ``umap_zoomed`` grid,
# whose attenuation map has no recorded photon energy and is used only as the
# target geometry. SIRF's resampler pads with zero, so the CT is written as
# ``HU + 1000`` and 1000 is subtracted afterwards: padding becomes -1000 HU.


@dataclass
class CTSeries:
    """One axial CT series: HU volume, LPS affine, table height and frame."""

    hu: np.ndarray
    affine: np.ndarray
    table_height: float
    frame_uid: str


def read_ct_series(directory):
    """Read and spatially sort one axial CT series."""
    import pydicom

    paths = sorted(Path(directory).glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in {directory}")
    headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in paths]
    if len({str(header.SeriesInstanceUID) for header in headers}) != 1:
        raise ValueError("The CT directory contains more than one series")
    if len({str(header.ImageOrientationPatient) for header in headers}) != 1:
        raise ValueError("The CT slices have different orientations")

    orientation = np.array(headers[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    positions = np.array(
        [header.ImagePositionPatient for header in headers], dtype=float
    )
    order = np.argsort(positions @ normal)
    paths = [paths[index] for index in order]
    headers = [headers[index] for index in order]
    positions = positions[order]

    spacing = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    slice_spacing = float(np.median(spacing))
    if np.any(spacing <= 1e-6) or not np.allclose(spacing, slice_spacing, atol=1e-3):
        raise ValueError("The CT slice spacing is not uniform")

    pixel_spacing = np.array(headers[0].PixelSpacing, dtype=float)
    hu = np.empty(
        (len(paths), int(headers[0].Rows), int(headers[0].Columns)),
        dtype=np.float32,
    )
    for index, path in enumerate(paths):
        dataset = pydicom.dcmread(path)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        hu[index] = dataset.pixel_array.astype(np.float32) * slope + intercept
        del dataset

    affine = np.eye(4)
    affine[:3, 0] = orientation[:3] * pixel_spacing[1]
    affine[:3, 1] = orientation[3:] * pixel_spacing[0]
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = positions[0]
    return CTSeries(
        hu=hu,
        affine=affine,
        table_height=float(getattr(headers[0], "TableHeight", np.nan)),
        frame_uid=str(headers[0].FrameOfReferenceUID),
    )


def patient_to_spect_affine(nm_file, table_height, frame_uid=None):
    """Return the reconstruction grid ``(i, j, k)`` to patient LPS affine."""
    import pydicom

    nm = pydicom.dcmread(nm_file, stop_before_pixels=True)
    if frame_uid is not None and str(nm.FrameOfReferenceUID) != frame_uid:
        raise ValueError("The NM and CT files use different frames of reference")
    detector = nm.DetectorInformationSequence[0]
    position = np.array(detector.ImagePositionPatient, dtype=float)
    orientation = np.array(detector.ImageOrientationPatient, dtype=float)
    centre = (
        position + (orientation[:3] + orientation[3:]) * GRID_CENTRE * VOXEL_SIZE_MM
    )
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


def resample_ct(ct_series, target_affine, umap_header, work_directory):
    """Resample CT HU onto the SPECT grid and transfer it to the STIR grid."""
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

    grid = STIR.ImageData(str(umap_header)).get_uniform_copy(0.0)
    grid.fill(np.ascontiguousarray(hu_grid, dtype=np.float32))
    if not np.allclose(get_array(grid), hu_grid, atol=1e-5):
        raise RuntimeError("The resampled HU values did not survive the STIR transfer")
    return hu_grid, grid


def y90_source(hu_grid, target_affine, centre_mm=HOT_SPHERE_CENTRE_MM):
    """Return a Y-90 source with liver background and a fourfold hot sphere.

    The hot sphere is not visible on CT. The frozen centre localises it, the
    liver-like CT voxels within 40 mm supply the background, and the sphere
    carries four times the liver concentration, matching the phantom's 4:1
    sphere-to-liver activity concentration.
    """
    indices = np.indices(hu_grid.shape).reshape(3, -1)
    patient = target_affine[:3, :3] @ indices[::-1] + target_affine[:3, 3, None]
    distance = np.linalg.norm(patient - np.asarray(centre_mm)[:, None], axis=0)
    distance = distance.reshape(hu_grid.shape)

    region = distance <= 40.0
    liver = (hu_grid >= -200.0) & (hu_grid <= 80.0) & region
    source = liver.astype(np.float32)
    source[distance <= HOT_SPHERE_RADIUS_MM] = 4.0
    return source


# %%
# Detected spectrum acquisition
# -----------------------------
# The connector does not run the list-mode scoring routine, so the validated
# source, density, collimator and geometry are configured through the normal
# connector and the engine is invoked script-locally with ``/OU:11``. The short
# 24-view acquisition gives ample statistics for the spectrum shape; it is a
# spectrum acquisition, not a substitute for the projection simulation. The
# records include the detector energy broadening and the total detected
# spectrum, without separation into PENETRATE components.


def acquire_detected_spectrum(
    source,
    density,
    output_directory,
    *,
    views=SPECTRUM_VIEWS,
    quantization_scale=0.05,
    num_bins=MATRIX_SIZE,
    num_axial=MATRIX_SIZE,
    prefix="y90_spectrum",
    timeout=3600.0,
):
    """Run the list-mode spectrum acquisition and return its output prefix."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    geometry = ProjectionGeometry(
        num_projections=views,
        extent_deg=360.0,
        direction="CCW",
        start_angle_deg=360.0,
        radius_mm=300.0,
        radii_mm=None,
        num_bins=num_bins,
        num_axial=num_axial,
        bin_size_mm=VOXEL_SIZE_MM,
        axial_size_mm=VOXEL_SIZE_MM,
        image_duration_s=views * 40.0,
    )
    connector = SimindPythonConnector(
        configs.get("AnyScan.yaml"),
        output_dir=output_directory,
        output_prefix=prefix,
        quantization_scale=quantization_scale,
    )
    connector.configure_acquisition(geometry)
    connector.configure_voxel_phantom(
        source=source,
        mu_map=density,
        voxel_size_mm=VOXEL_SIZE_MM,
        scoring_routine=4,
        mu_map_type="density",
    )
    connector.add_config_value(1, -REFERENCE_ENERGY_KEV)
    connector.add_config_value(20, WINDOW_HIGH_KEV)
    connector.add_config_value(21, WINDOW_LOW_KEV)
    connector.add_config_value(19, 2)
    connector.add_config_value(53, 1)
    connector.add_config_value(80, 512)
    connector.add_config_value(27, CHANNEL_KEV)
    connector.add_runtime_switch("FI", "y90")
    connector.add_runtime_switch("CC", "ma-mlegp")
    connector.set_activity(160.8, 40.0)

    config = connector.get_config()
    config.set_value(84, 2)
    config.set_flag(3, False)
    config.set_flag(5, True)
    config.set_flag(14, True)
    config.save_file(output_directory / prefix)

    for suffix in (".lmf", ".a00", ".res", ".log"):
        (output_directory / f"{prefix}{suffix}").unlink(missing_ok=True)

    binary = os.environ.get("SIMIND_BIN") or shutil.which("simind")
    if not binary:
        raise RuntimeError(
            "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
            "Install SIMIND before running this tutorial."
        )
    if os.sep in binary:
        binary = str(Path(binary).expanduser().resolve())
    command = [
        binary,
        prefix,
        prefix,
        f"/FI:y90/CC:ma-mlegp/PX:{VOXEL_SIZE_MM / 10.0:.6f}/OU:11",
    ]
    print("SIMIND command:", " ".join(command))
    log_file = output_directory / f"{prefix}.log"
    with open(log_file, "w") as log:
        subprocess.run(
            command,
            cwd=output_directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    if "Simulation stopped" not in log_file.read_text():
        raise RuntimeError(f"SIMIND did not finish the spectrum run; see {log_file}")
    if not (output_directory / f"{prefix}.lmf").is_file():
        raise RuntimeError(f"SIMIND wrote no {prefix}.lmf; see {log_file}")
    return output_directory / prefix


# %%
# Count check
# -----------
# The list-mode file records every crystal energy; the window is applied by the
# reader. The engine's ``.a00`` projection is already windowed and restricted to
# the projection matrix, and ``.res`` gives the per-projection count rate. Their
# agreement checks the record interpretation and the window.


def read_count_metadata(res_file):
    """Read the projection count fields from a SIMIND ``.res`` file."""
    text = Path(res_file).read_text()
    views = int(re.search(r"Nr of Projections\.*:\s*(\d+)", text).group(1))
    window_rate = float(
        re.search(r"CountRate E-Window\.*:\s*([0-9Ee+.\-]+)", text).group(1)
    )
    detector_hits = re.search(r"DetectorHits\.*:\s*(\d+)", text)
    elapsed = re.search(r"Elapsed time\.*:\s*(.+)", text)
    return {
        "views": views,
        "count_rate_window": window_rate,
        "window_total": window_rate * views,
        "detector_hits": int(detector_hits.group(1)) if detector_hits else None,
        "elapsed": elapsed.group(1).strip() if elapsed else None,
    }


def check_window_counts(prefix, events, lower=WINDOW_LOW_KEV, upper=WINDOW_HIGH_KEV):
    """Compare the list-mode window weight with the engine count outputs."""
    prefix = Path(prefix)
    metadata = read_count_metadata(prefix.with_suffix(".res"))
    projection = np.fromfile(prefix.with_suffix(".a00"), dtype="<f4")
    if projection.size % metadata["views"]:
        raise ValueError("The .a00 size is not a whole number of projections")
    projection_total = float(projection.reshape(metadata["views"], -1).sum())
    listed = window_weight(events.energies_kev, events.weights, lower, upper)
    return {
        **metadata,
        "list_mode_window": listed,
        "projection_total": projection_total,
        "ratio_to_projection": listed / projection_total,
        "ratio_to_count_rate": listed / metadata["window_total"],
    }


# %%
# Weighted attenuation map
# ------------------------


def spectub_image(template, values):
    """Copy values into a STIR image, mirroring x for the SPECTUB convention."""
    image = template.get_uniform_copy(0.0)
    image.fill(np.ascontiguousarray(np.asarray(values)[..., ::-1], dtype=np.float32))
    return image


def reconstruct(acquisition, attenuation, initial, num_subsets, num_iterations):
    """One OSEM reconstruction with a SPECTUB attenuation model and no scatter."""
    import sirf.STIR as sirf

    matrix = sirf.SPECTUBMatrix()
    matrix.set_attenuation_image(attenuation)
    matrix.set_keep_all_views_in_cache(False)
    matrix.set_resolution_model(0.0, 0.0, False)
    model = sirf.AcquisitionModelUsingMatrix(matrix)
    objective = sirf.make_Poisson_loglikelihood(acquisition)
    objective.set_acquisition_model(model)
    objective.set_num_subsets(num_subsets)

    reconstructor = sirf.OSMAPOSLReconstructor()
    reconstructor.set_objective_function(objective)
    reconstructor.set_num_subsets(num_subsets)
    reconstructor.set_num_subiterations(num_iterations)
    reconstructor.set_input(acquisition)
    image = initial.clone()
    reconstructor.set_up(image)
    reconstructor.reconstruct(image)
    return image


# %%
# Paired reconstruction
# ---------------------
# The measured ``peak.hs`` acquisition is the template: 120 views over 360
# degrees with a non-circular orbit. Both arms use the same measured data, the
# same 12 subsets and 12 iterations, the same unit body support as the initial
# image, and zero additive scatter. Only the attenuation map changes. The
# attenuation image is x-mirrored at the SPECTUB boundary because STIR's
# SPECTUB implementation mirrors x relative to the image; the activity image
# and the support are not mirrored. The two images are not a ranking: neither
# models scatter, and the weighted map is a global approximation of the
# energy-dependent attenuation.


def paired_reconstruction(
    acquisition,
    grid_template,
    hu_grid,
    attenuation_150,
    attenuation_weighted,
    *,
    num_subsets=12,
    num_iterations=12,
):
    """Reconstruct the same data twice, changing only the attenuation map."""
    support = np.ascontiguousarray((hu_grid > -400.0).astype(np.float32))
    initial = grid_template.get_uniform_copy(0.0)
    initial.fill(support)
    images = {}
    for name, attenuation in (
        ("150 keV", attenuation_150),
        ("spectrum weighted", attenuation_weighted),
    ):
        attenuation_image = spectub_image(grid_template, attenuation)
        images[name] = reconstruct(
            acquisition, attenuation_image, initial, num_subsets, num_iterations
        )
    return images


# %%
# Figures
# -------


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def save_isotope_spectra(path, spectra):
    """Plot the installed isotope files and mark the imaging window."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, spectrum in spectra.items():
        if spectrum.continuous:
            axis.plot(
                spectrum.energies_kev,
                spectrum.yields,
                label=f"{name} ({spectrum.kind})",
                linewidth=2 if name == "y90.isd" else 1,
            )
        else:
            axis.vlines(
                spectrum.energies_kev,
                0.0,
                spectrum.yields,
                label=f"{name} ({spectrum.kind})",
                linewidth=2,
            )
    axis.axvspan(WINDOW_LOW_KEV, WINDOW_HIGH_KEV, color="0.85", zorder=0)
    axis.set_yscale("log")
    axis.set_xlim(0.0, 2200.0)
    axis.set_xlabel("Energy (keV)")
    axis.set_ylabel("Tabulated relative yield")
    axis.set_title("Y-90 source spectra in the SIMIND installation")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)


def save_detected_spectrum(path, counts, edges, events, check):
    """Plot the weighted detected spectrum with the window shaded."""
    plt = _pyplot()
    centres = 0.5 * (edges[:-1] + edges[1:])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, xlim in zip(axes, ((0.0, 2100.0), (0.0, 350.0))):
        axis.step(centres, counts, where="mid", color="black", linewidth=1)
        axis.axvspan(WINDOW_LOW_KEV, WINDOW_HIGH_KEV, color="tab:orange", alpha=0.2)
        axis.set_xlim(*xlim)
        axis.set_xlabel("Energy (keV)")
        axis.set_ylabel("Monte Carlo event weight per 4 keV")
    axes[0].set_title(
        f"LIST_MODE detected spectrum, {check['views']} views, "
        f"{events.energies_kev.size} records"
    )
    axes[1].set_title("Imaging window (75-225 keV)")
    axes[1].text(
        0.98,
        0.95,
        f"window weight {check['list_mode_window']:.4g}\n"
        f"ratio to .a00 {check['ratio_to_projection']:.4f}\n"
        "detector broadening 20.5 keV FWHM at 140.5 keV\n"
        "no PENETRATE component separation",
        transform=axes[1].transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    figure.savefig(path, dpi=140)


def save_attenuation_maps(path, attenuation_150, attenuation_weighted):
    """Show the two attenuation maps and their difference on one axial slice."""
    plt = _pyplot()
    slice_index = int(round(63.5))
    difference = attenuation_weighted[slice_index] - attenuation_150[slice_index]
    maps = [
        (attenuation_150[slice_index], "Attenuation at 150 keV", "magma", 0.0, 0.25),
        (
            attenuation_weighted[slice_index],
            "Spectrum-weighted attenuation",
            "magma",
            0.0,
            0.25,
        ),
        (difference, "Weighted minus 150 keV", "magma", 0.0, 0.03),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (image, title, cmap, low, high) in zip(axes, maps):
        shown = axis.imshow(image, cmap=cmap, vmin=low, vmax=high)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, shrink=0.7, label="cm$^{-1}$")
    figure.tight_layout()
    figure.savefig(path, dpi=140)


def save_reconstruction_comparison(
    path, images, hu_grid, target_affine, centre_mm=HOT_SPHERE_CENTRE_MM
):
    """Show both reconstructions, their difference and a profile through the sphere."""
    plt = _pyplot()
    recon_150 = np.asarray(images["150 keV"].as_array(), dtype=np.float32)
    recon_weighted = np.asarray(
        images["spectrum weighted"].as_array(), dtype=np.float32
    )
    difference = recon_weighted - recon_150
    centre_index = np.linalg.solve(
        target_affine, np.append(np.asarray(centre_mm, dtype=float), 1.0)
    )
    k = int(round(centre_index[2]))
    j = int(round(centre_index[1]))
    limit = float(np.percentile(np.maximum(recon_150, recon_weighted), 99.5))

    figure = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    panels = [
        (recon_150[k], "150 keV attenuation", "viridis", 0.0, limit),
        (recon_weighted[k], "Spectrum-weighted attenuation", "viridis", 0.0, limit),
        (difference[k], "Weighted minus 150 keV", "coolwarm", -limit / 4, limit / 4),
    ]
    for column, (image, title, cmap, low, high) in enumerate(panels):
        axis = figure.add_subplot(grid[0, column])
        shown = axis.imshow(image, cmap=cmap, vmin=low, vmax=high)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, shrink=0.75)

    profile_axis = figure.add_subplot(grid[1, :])
    profile_axis.plot(recon_150[k, j], label="150 keV attenuation")
    profile_axis.plot(recon_weighted[k, j], label="Spectrum-weighted attenuation")
    profile_axis.set_xlabel("x (voxel)")
    profile_axis.set_ylabel("Reconstructed value")
    profile_axis.set_title("Profile through the hot-sphere centre; both omit scatter")
    profile_axis.legend()
    figure.savefig(path, dpi=140)


def save_json(path, payload):
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=1)


# %%
# Run the tutorial
# ----------------


def run():
    start = time.perf_counter()
    data_directory = fetch_y90_phantom()
    output_directory = (
        Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
        / "real_data"
        / "effective_energy"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Reading the CT series")
    ct_series = read_ct_series(data_directory / "ctac")
    target_affine = patient_to_spect_affine(
        data_directory / "peak.dcm", ct_series.table_height, ct_series.frame_uid
    )
    hu_grid, grid = resample_ct(
        ct_series,
        target_affine,
        data_directory / "umap_zoomed.hv",
        output_directory / "work",
    )
    attenuation_150 = hu_to_attenuation(hu_grid, REFERENCE_ENERGY_KEV).astype(
        np.float32
    )
    density = hu_to_density_schneider(hu_grid).astype(np.float32)
    print(
        f"CT: {ct_series.hu.shape}, HU {ct_series.hu.min():.0f} to "
        f"{ct_series.hu.max():.0f}; grid {hu_grid.shape}"
    )
    print(f"SPECT grid affine (LPS):\n{np.round(target_affine, 4)}")

    print("Reading the installed isotope spectra")
    spectra = {
        "y90.isd": read_isotope_spectrum(find_isotope_file("y90.isd")),
        "y90_frey.isd": read_isotope_spectrum(find_isotope_file("y90_frey.isd")),
        "y90_nudat.isd": read_isotope_spectrum(find_isotope_file("y90_nudat.isd")),
        "y90_tissue.isd": read_isotope_spectrum(find_isotope_file("y90_tissue.isd")),
    }
    source_spectrum = spectra["y90.isd"]
    print(
        f"y90.isd: {source_spectrum.kind}, {source_spectrum.energies_kev.size} "
        f"points from {source_spectrum.energies_kev.min():.0f} to "
        f"{source_spectrum.energies_kev.max():.0f} keV"
    )
    save_isotope_spectra(output_directory / "isotope_spectra.png", spectra)

    source = y90_source(hu_grid, target_affine)
    print(
        f"Y-90 source: {np.count_nonzero(source)} voxels, sum {source.sum():.1f}, "
        f"maximum {source.max():.1f}"
    )

    print("Running the list-mode spectrum acquisition")
    spectrum_start = time.perf_counter()
    spectrum_prefix = acquire_detected_spectrum(
        source, density, output_directory / "spectrum"
    )
    spectrum_seconds = time.perf_counter() - spectrum_start
    events = read_list_mode(spectrum_prefix.with_suffix(".lmf"))
    counts, edges = histogram_spectrum(events.energies_kev, events.weights)
    check = check_window_counts(spectrum_prefix, events)
    print(
        f"Spectrum geometry: {check['views']} views, 360 deg, 300 mm circular, "
        f"{MATRIX_SIZE} x {MATRIX_SIZE} bins of {VOXEL_SIZE_MM} mm, "
        f"{WINDOW_LOW_KEV:.0f}-{WINDOW_HIGH_KEV:.0f} keV"
    )
    print(
        f"Spectrum: {events.energies_kev.size} records, "
        f"energy {events.energies_kev.min():.1f} to "
        f"{events.energies_kev.max():.1f} keV, {spectrum_seconds / 60.0:.1f} min; "
        f"detector hits {check['detector_hits']}; engine {check['elapsed']}"
    )
    print(
        f"Window weight {check['list_mode_window']:.1f}; .a00 "
        f"{check['projection_total']:.1f}; ratio "
        f"{check['ratio_to_projection']:.5f}; .res count-rate ratio "
        f"{check['ratio_to_count_rate']:.5f}"
    )
    save_detected_spectrum(
        output_directory / "detected_spectrum.png", counts, edges, events, check
    )

    channel_energies, channel_weights = window_channel_weights(counts, edges)
    mean_energy = weighted_mean_energy(channel_energies, channel_weights)
    attenuation_weighted = weighted_attenuation(
        hu_grid, channel_energies, channel_weights
    ).astype(np.float32)
    body = hu_grid > -400.0
    difference = attenuation_weighted - attenuation_150
    ratio = np.median(attenuation_weighted[body] / attenuation_150[body])
    print(
        f"Weighted mean energy: {mean_energy:.1f} keV; attenuation ratio to "
        f"150 keV: median {ratio:.3f}"
    )
    print(
        f"Weighted map difference over the body: mean {difference[body].mean():.5f} "
        f"cm^-1, max {np.abs(difference[body]).max():.5f} cm^-1"
    )
    save_attenuation_maps(
        output_directory / "attenuation_maps.png",
        attenuation_150,
        attenuation_weighted,
    )

    import sirf.STIR as sirf

    sirf.AcquisitionData.set_storage_scheme("memory")
    acquisition = sirf.AcquisitionData(str(data_directory / "peak.hs"))
    print(f"Measured acquisition shape: {acquisition.dimensions()}")

    recon_start = time.perf_counter()
    images = paired_reconstruction(
        acquisition, grid, hu_grid, attenuation_150, attenuation_weighted
    )
    recon_seconds = time.perf_counter() - recon_start
    save_reconstruction_comparison(
        output_directory / "reconstruction.png", images, hu_grid, target_affine
    )
    print(f"Paired reconstruction: {recon_seconds / 60.0:.1f} min")

    summary = {
        "source_spectrum": {
            "file": "y90.isd",
            "kind": source_spectrum.kind,
            "points": int(source_spectrum.energies_kev.size),
            "energy_range_kev": [
                float(source_spectrum.energies_kev.min()),
                float(source_spectrum.energies_kev.max()),
            ],
        },
        "detected_spectrum": {
            "views": check["views"],
            "records": int(events.energies_kev.size),
            "energy_range_kev": [
                float(events.energies_kev.min()),
                float(events.energies_kev.max()),
            ],
            "window_weight": check["list_mode_window"],
            "projection_total": check["projection_total"],
            "ratio_to_projection": check["ratio_to_projection"],
            "ratio_to_count_rate": check["ratio_to_count_rate"],
            "detector_hits": check["detector_hits"],
            "engine_elapsed": check["elapsed"],
            "acquisition_seconds": spectrum_seconds,
            "detector_broadening_fwhm_kev_at_140_5": 20.5,
        },
        "weighted_attenuation": {
            "weighted_mean_energy_kev": mean_energy,
            "body_mean_difference_cm1": float(difference[body].mean()),
            "body_max_absolute_difference_cm1": float(np.abs(difference[body]).max()),
        },
        "reconstruction": {
            "subsets": 12,
            "iterations": 12,
            "initial": "unit body support (HU > -400)",
            "additive_scatter": "zero in both",
            "seconds": recon_seconds,
        },
        "total_seconds": time.perf_counter() - start,
    }
    save_json(output_directory / "summary.json", summary)
    print(f"Total runtime: {summary['total_seconds'] / 60.0:.1f} min")
    print(f"Outputs written to {output_directory}")
    _pyplot().show()


if __name__ == "__main__":
    run()
