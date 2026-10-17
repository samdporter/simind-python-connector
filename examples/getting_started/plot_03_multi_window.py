"""
Multi-window simulation
=======================

This example runs a multi-window (TEW-style) simulation of Lu-177 with the
pure Python connector and builds a triple-energy-window scatter correction.
"""

# %%
# Setup
# -----
# Check that SIMIND is installed and choose where the outputs are written.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE


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

connector = SimindPythonConnector(
    config_source=configs.get("Example.yaml"),
    output_dir=output_dir,
    output_prefix="tew_case01",
    quantization_scale=0.05 * DEFAULT_QUANTIZATION_SCALE,
)
connector.configure_voxel_phantom(
    source=source,
    mu_map=mu_map,
    voxel_size_mm=4.0,
    scoring_routine=1,
)
connector.add_config_value(19, 2)
connector.add_config_value(53, 0)

lowers = [166.0, 187.0, 229.0]
uppers = [187.0, 229.0, 250.0]
connector.set_energy_windows(lowers, uppers, [0, 0, 0])
connector.add_config_value(1, 208.0)
connector.add_runtime_switch("FI", "lu177")
connector.add_runtime_switch("CC", "ma-megp")
connector.add_runtime_switch("NN", 1)
connector.add_runtime_switch("RR", 12345)

outputs = connector.run()

lower_scatter = outputs["tot_w1"].projection
photopeak = outputs["tot_w2"].projection
upper_scatter = outputs["tot_w3"].projection

# %%
# Apply the TEW correction
# ------------------------
# The scatter in the photopeak is estimated from the count densities in the
# two neighbouring windows, then subtracted (clipped at zero).

width1 = uppers[0] - lowers[0]
width2 = uppers[1] - lowers[1]
width3 = uppers[2] - lowers[2]
scatter_estimate = lower_scatter * (width2 / (2.0 * width1)) + upper_scatter * (
    width2 / (2.0 * width3)
)
corrected = np.clip(photopeak - scatter_estimate, a_min=0.0, a_max=None)

np.save(output_dir / "tew_lower_window.npy", lower_scatter)
np.save(output_dir / "tew_photopeak.npy", photopeak)
np.save(output_dir / "tew_upper_window.npy", upper_scatter)
np.save(output_dir / "tew_scatter_estimate.npy", scatter_estimate)
np.save(output_dir / "tew_corrected.npy", corrected)

print(f"Photopeak counts: {float(photopeak.sum()):.3f}")
print(f"TEW scatter estimate counts: {float(scatter_estimate.sum()):.3f}")
print(f"TEW corrected counts: {float(corrected.sum()):.3f}")

# %%
# Plot the result
# ---------------
# The three projection panels share one colour scale, so the scatter estimate
# can be compared directly with the photopeak and the corrected image.


def view0(projection: np.ndarray) -> np.ndarray:
    return projection[0, :, 0, :] if projection.ndim == 4 else projection[0, :, :]


photopeak_view0 = view0(photopeak)
scatter_view0 = view0(scatter_estimate)
corrected_view0 = view0(corrected)
lower_view0 = view0(lower_scatter)
views = [photopeak_view0, scatter_view0, corrected_view0, lower_view0]
proj_vmin = float(min(v.min() for v in views))
proj_vmax = float(max(v.max() for v in views))

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(source[source.shape[0] // 2, :, :], cmap="viridis")
axes[0].set_title("Input source")
axes[1].imshow(photopeak_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
axes[1].set_title("Photopeak (view 0)")
axes[2].imshow(scatter_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
axes[2].set_title("TEW scatter est.")
axes[3].imshow(corrected_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
axes[3].set_title("TEW corrected")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_dir / "multi_window_summary.png", dpi=140)
plt.show()
