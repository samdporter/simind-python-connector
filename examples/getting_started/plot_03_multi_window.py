"""
Multi-window simulation
=======================

This example runs a multi-window (TEW-style) simulation of Lu-177 with the
pure Python connector and builds a triple-energy-window scatter correction.
It uses the default source quantization, a history multiplier of 10 and four
360-degree views of 32 x 32 bins at 4 mm.

TEW assumes that the scatter spectrum is linear across the three windows.
That is an approximation: the estimate is not a measured scatter component and
can be negative after subtraction where the side windows are noisy.
"""

# %%
# Setup
# -----
# Check that SIMIND is installed and choose where the outputs are written.

import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE
from simind_python_connector.utils.interfile import ProjectionGeometry
from simind_python_connector.utils.projection_metrics import (
    projection_view0,
    require_output_keys,
    source_weight_sum,
    tew_correction,
    zero_history_fraction,
)


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "multi_window"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the phantom
# -----------------
# A cylindrical body with a hot sphere, on a 32 x 32 x 32 grid (z, y, x).

dims = (32, 32, 32)
z, y, x = np.indices(dims)

body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

source = np.zeros(dims, dtype=np.float32)
source[body] = 0.8
source[hot] = 2.0

mu_map = np.zeros_like(source)
mu_map[body] = 0.15

# %%
# Configure three energy windows
# ------------------------------
# Lu-177 TEW around the 208 keV peak: lower scatter 166-187 keV, photopeak
# 187-229 keV and upper scatter 229-250 keV. SIMIND writes one projection per
# window (``tot_w1`` to ``tot_w3``).

PHOTON_MULTIPLIER = 10
SEED = 12345

connector = SimindPythonConnector(
    config_source=configs.get("Example.yaml"),
    output_dir=output_dir,
    output_prefix="tew_case01",
    quantization_scale=DEFAULT_QUANTIZATION_SCALE,
)
connector.configure_voxel_phantom(
    source=source,
    mu_map=mu_map,
    voxel_size_mm=4.0,
    scoring_routine=1,
)
connector.configure_acquisition(
    ProjectionGeometry(
        num_projections=4,
        extent_deg=360.0,
        direction="CW",
        start_angle_deg=180.0,
        radius_mm=250.0,
        radii_mm=None,
        num_bins=32,
        num_axial=32,
        bin_size_mm=4.0,
        axial_size_mm=4.0,
        image_duration_s=None,
    )
)
connector.add_config_value(19, 2)
connector.add_config_value(53, 0)

lowers = [166.0, 187.0, 229.0]
uppers = [187.0, 229.0, 250.0]
connector.set_energy_windows(lowers, uppers, [0, 0, 0])
connector.add_config_value(1, 208.0)
connector.add_runtime_switch("FI", "lu177")
connector.add_runtime_switch("CC", "ma-megp")
connector.add_runtime_switch("NN", PHOTON_MULTIPLIER)
connector.add_runtime_switch("RR", SEED)

started = time.perf_counter()
outputs = connector.run()
elapsed_s = time.perf_counter() - started

windows = require_output_keys(outputs, "tot_w1", "tot_w2", "tot_w3")
lower_scatter = windows["tot_w1"].projection
photopeak = windows["tot_w2"].projection
upper_scatter = windows["tot_w3"].projection

# %%
# Apply the TEW correction
# ------------------------
# Each side window is scaled to the photopeak width and half of the total is
# assigned to the peak. The result is subtracted and clipped at zero, because a
# negative activity is unphysical. The clipping fraction says how much of the
# photopeak the estimate removed at zero; a large fraction means the side
# windows are too noisy for a quantitative correction.

widths = [upper - lower for lower, upper in zip(lowers, uppers)]
scatter_estimate, corrected, clipped_counts, clipped_pixels = tew_correction(
    photopeak,
    lower_scatter,
    upper_scatter,
    lower_width_kev=widths[0],
    photopeak_width_kev=widths[1],
    upper_width_kev=widths[2],
)

np.save(output_dir / "tew_lower_window.npy", lower_scatter)
np.save(output_dir / "tew_photopeak.npy", photopeak)
np.save(output_dir / "tew_upper_window.npy", upper_scatter)
np.save(output_dir / "tew_scatter_estimate.npy", scatter_estimate)
np.save(output_dir / "tew_corrected.npy", corrected)

print(
    "Histories per view: "
    f"{source_weight_sum(source, DEFAULT_QUANTIZATION_SCALE) * PHOTON_MULTIPLIER}"
)
loss = zero_history_fraction(source, DEFAULT_QUANTIZATION_SCALE)
print(f"Quantization loss: {loss:.4%}")
print(f"Photopeak projection sum: {float(photopeak.sum()):.3f}")
print(f"TEW scatter estimate projection sum: {float(scatter_estimate.sum()):.3f}")
print(f"TEW corrected projection sum: {float(corrected.sum()):.3f}")
print(f"Clipped photopeak counts: {clipped_counts:.3%}")
print(f"Clipped pixels: {clipped_pixels:.3%}")
print(f"Elapsed: {elapsed_s:.0f} s")

# %%
# Plot the result
# ---------------
# The lower and upper windows and the TEW estimate share one colour scale,
# because they are all scatter-scale images. The photopeak and corrected image
# share a second scale. One scale for all five would hide the side windows.


lower_view0 = projection_view0(lower_scatter)
photopeak_view0 = projection_view0(photopeak)
upper_view0 = projection_view0(upper_scatter)
scatter_view0 = projection_view0(scatter_estimate)
corrected_view0 = projection_view0(corrected)
views = [
    lower_view0,
    photopeak_view0,
    upper_view0,
    scatter_view0,
    corrected_view0,
]
scatter_vmax = float(max(lower_view0.max(), upper_view0.max(), scatter_view0.max()))
peak_vmax = float(max(photopeak_view0.max(), corrected_view0.max()))

fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
axes = axes.ravel()
axes[0].imshow(source[source.shape[0] // 2, :, :], cmap="viridis")
axes[0].set_title("Input source")
for ax, view, title, vmax in zip(
    axes[1:],
    views,
    (
        "Lower scatter 166-187 keV",
        "Photopeak 187-229 keV",
        "Upper scatter 229-250 keV",
        "TEW scatter estimate",
        "TEW corrected",
    ),
    (scatter_vmax, peak_vmax, scatter_vmax, scatter_vmax, peak_vmax),
):
    image = ax.imshow(view, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.7)
for ax in axes:
    ax.axis("off")
fig.savefig(output_dir / "multi_window_summary.png", dpi=140)
plt.show()
