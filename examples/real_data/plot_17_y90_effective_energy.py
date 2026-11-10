"""Y-90 effective attenuation energy.
Y-90 is a pure beta emitter: the 75-225 keV window is a bremsstrahlung and
scatter continuum, not a photopeak. A single 150 keV map is an approximation.
This tutorial compares it with a map weighted by the simulated detected
spectrum, which includes detector resolution (20.5 keV FWHM at 140.5 keV) and
is not split into PENETRATE components. The spectrum is simulated and the
weighted map is a global effective-map approximation; neither is an accuracy
reference.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from examples import _y90_spectrum as spectrum
from examples._y90 import (
    MATRIX_SIZE,
    VOXEL_SIZE_MM,
    patient_to_spect_affine,
    read_ct_series,
    resample_ct,
)
from examples._y90_spectrum import (
    CHANNEL_KEV,
    WINDOW_HIGH_KEV,
    WINDOW_LOW_KEV,
    find_isotope_file,
    histogram_spectrum,
    read_isotope_spectrum,
    read_list_mode,
    weighted_mean_energy,
    window_channel_weights,
    window_weight,
)
from simind_python_connector import configs
from simind_python_connector.connectors import SimindPythonConnector
from simind_python_connector.converters.attenuation import (
    hu_to_attenuation,
    hu_to_density_schneider,
)
from simind_python_connector.datasets import fetch_y90_phantom
from simind_python_connector.utils.interfile import ProjectionGeometry


REFERENCE_ENERGY_KEV = 150.0
LIST_MODE_DTYPE = spectrum.LIST_MODE_DTYPE
SPECTRUM_VIEWS = 24
HOT_SPHERE_CENTRE_MM = (-77.43, -181.69, -576.20)
HOT_SPHERE_RADIUS_MM = 12.449


def weighted_attenuation(hu, energies_kev, weights, converter=hu_to_attenuation):
    """Return ``sum_j w_j mu(HU, E_j) / sum_j w_j``."""
    hu = np.asarray(hu, float)
    energies_kev = np.asarray(energies_kev, float)
    weights = np.asarray(weights, float)
    if energies_kev.shape != weights.shape or np.any(weights < 0):
        raise ValueError("invalid spectrum weights")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cannot weight attenuation with zero total weight")
    result = np.zeros_like(hu)
    for energy, weight in zip(energies_kev, weights):
        if weight:
            result += weight * converter(hu, float(energy))
    return result / total


def y90_source(hu_grid, target_affine, centre_mm=HOT_SPHERE_CENTRE_MM):
    """Liver background with a fourfold hot sphere at the frozen centre."""
    indices = np.indices(hu_grid.shape).reshape(3, -1)
    patient = target_affine[:3, :3] @ indices[::-1] + target_affine[:3, 3, None]
    distance = np.linalg.norm(patient - np.asarray(centre_mm)[:, None], axis=0)
    distance = distance.reshape(hu_grid.shape)
    region = distance <= 40.0
    source = ((hu_grid >= -200.0) & (hu_grid <= 80.0) & region).astype(np.float32)
    source[distance <= HOT_SPHERE_RADIUS_MM] = 4.0
    return source


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
    """Run the list-mode spectrum acquisition and return its prefix."""
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
    for index, value in (
        (1, -REFERENCE_ENERGY_KEV),
        (20, WINDOW_HIGH_KEV),
        (21, WINDOW_LOW_KEV),
        (19, 2),
        (53, 1),
        (80, 512),
        (27, CHANNEL_KEV),
    ):
        connector.add_config_value(index, value)
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
            "SIMIND executable not found ($SIMIND_BIN or simind on PATH)."
        )
    binary = str(Path(binary).expanduser().resolve()) if os.sep in binary else binary
    command = [
        binary,
        prefix,
        prefix,
        f"/FI:y90/CC:ma-mlegp/PX:{VOXEL_SIZE_MM / 10.0:.6f}/OU:11",
    ]
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


def read_count_metadata(res_file):
    text = Path(res_file).read_text()
    views = int(re.search(r"Nr of Projections\.*:\s*(\d+)", text).group(1))
    rate = float(re.search(r"CountRate E-Window\.*:\s*([0-9Ee+.\-]+)", text).group(1))
    return {"views": views, "count_rate_window": rate, "window_total": rate * views}


def check_window_counts(prefix, events, lower=WINDOW_LOW_KEV, upper=WINDOW_HIGH_KEV):
    """Compare the list-mode window weight with the engine count outputs."""
    prefix = Path(prefix)
    metadata = read_count_metadata(prefix.with_suffix(".res"))
    projection = np.fromfile(prefix.with_suffix(".a00"), dtype="<f4")
    if projection.size % metadata["views"]:
        raise ValueError("The .a00 size is not a whole number of projections")
    total = float(projection.reshape(metadata["views"], -1).sum())
    listed = window_weight(events.energies_kev, events.weights, lower, upper)
    return {
        **metadata,
        "list_mode_window": listed,
        "projection_total": total,
        "ratio_to_projection": listed / total,
        "ratio_to_count_rate": listed / metadata["window_total"],
    }


def save_detected_spectrum(path, counts, edges, events, check):
    import matplotlib.pyplot as plt

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
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_attenuation_maps(path, attenuation_150, attenuation_weighted):
    import matplotlib.pyplot as plt

    k = int(round(63.5))
    difference = attenuation_weighted[k] - attenuation_150[k]
    limit = float(np.max(np.abs(difference)))
    maps = (
        (attenuation_150[k], "Attenuation at 150 keV", "magma", 0.0, 0.25),
        (attenuation_weighted[k], "Spectrum-weighted attenuation", "magma", 0.0, 0.25),
        (difference, "Weighted minus 150 keV", "coolwarm", -limit, limit),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (image, title, cmap, low, high) in zip(axes, maps):
        shown = axis.imshow(image, cmap=cmap, vmin=low, vmax=high)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, shrink=0.7, label="cm$^{-1}$")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run():
    import sirf.STIR  # noqa: F401 - lazy SIRF dependency of the resampling helper

    data_directory = fetch_y90_phantom()
    output_directory = (
        Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
        / "real_data"
        / "effective_energy"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    ct_series = read_ct_series(data_directory / "ctac")
    target_affine = patient_to_spect_affine(data_directory / "peak.dcm", ct_series)
    hu_grid, _ = resample_ct(
        ct_series,
        target_affine,
        data_directory / "umap_zoomed.hv",
        output_directory / "work",
    )
    attenuation_150 = hu_to_attenuation(hu_grid, REFERENCE_ENERGY_KEV).astype(
        np.float32
    )
    density = hu_to_density_schneider(hu_grid).astype(np.float32)
    print(f"CT: {ct_series.hu.shape}; SPECT grid: {hu_grid.shape}")
    source_spectrum = read_isotope_spectrum(find_isotope_file("y90.isd"))
    print(
        f"y90.isd: {source_spectrum.kind}, {source_spectrum.energies_kev.size} points"
    )
    source = y90_source(hu_grid, target_affine)
    spectrum_prefix = acquire_detected_spectrum(
        source, density, output_directory / "spectrum"
    )
    events = read_list_mode(spectrum_prefix.with_suffix(".lmf"))
    counts, edges = histogram_spectrum(events.energies_kev, events.weights)
    check = check_window_counts(spectrum_prefix, events)
    save_detected_spectrum(
        output_directory / "detected_spectrum.png", counts, edges, events, check
    )
    print(
        f"Spectrum: {events.energies_kev.size} records, window weight "
        f"{check['list_mode_window']:.4g}, ratio to .a00 "
        f"{check['ratio_to_projection']:.5f}, ratio to .res "
        f"{check['ratio_to_count_rate']:.5f}"
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
        f"Weighted mean energy: {mean_energy:.1f} keV; attenuation ratio {ratio:.3f}; "
        f"mean difference {difference[body].mean():.5f} cm^-1"
    )
    save_attenuation_maps(
        output_directory / "attenuation_maps.png",
        attenuation_150,
        attenuation_weighted,
    )
    print(f"Outputs written to {output_directory}")


if __name__ == "__main__":
    run()
