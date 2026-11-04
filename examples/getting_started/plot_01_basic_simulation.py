"""
Basic simulation
================

This example runs SIMIND with voxelized NumPy inputs and reads projections
back as NumPy arrays via ``SimindPythonConnector``. It uses the default
source quantization and a history multiplier large enough for the body and the
hot sphere to be resolved in the projection, rather than a smoke-test run.

The acquisition is a four-view, 360-degree orbit with 32 x 32 bins of 4 mm.
Four views keep the runtime near an hour on a single emulated x86 core while
giving each view the full history budget. Increase ``photon_multiplier`` (the
SIMIND ``NN`` switch) for more statistics; the cost is linear in it.
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
    zero_history_fraction,
)


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "basic_simulation"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the phantom
# -----------------
# A cylindrical body with a hot sphere, on a 32 x 32 x 32 grid (z, y, x).
# The source map holds relative activity and the attenuation map holds
# ``mu`` in 1/cm.

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
# Configure and run
# -----------------
# Start from the bundled ``Example.yaml``. The default ``quantization_scale``
# gives the hottest voxel 500 photon histories; ``NN`` multiplies that budget
# without changing the projection scale. The geometry is declared explicitly
# so the number of views, bin size and radius are not inherited by accident.

PHOTON_MULTIPLIER = 8
SEED = 12345

connector = SimindPythonConnector(
    config_source=configs.get("Example.yaml"),
    output_dir=output_dir,
    output_prefix="basic_case01",
    quantization_scale=DEFAULT_QUANTIZATION_SCALE,
)
source_path, density_path = connector.configure_voxel_phantom(
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
connector.add_config_value(1, 140.0)
connector.add_config_value(53, 0)
connector.set_energy_windows([126.0], [154.0], [0])
connector.add_runtime_switch("CA", 0)
connector.add_runtime_switch("FI", "tc99m")
connector.add_runtime_switch("CC", "ma-lehr")
connector.add_runtime_switch("NN", PHOTON_MULTIPLIER)
connector.add_runtime_switch("RR", SEED)

started = time.perf_counter()
outputs = connector.run()
elapsed_s = time.perf_counter() - started
total = require_output_keys(outputs, "tot_w1")["tot_w1"].projection
np.save(output_dir / "total.npy", total)

print(f"Source file: {source_path}")
print(f"Density file: {density_path}")
print(
    "Histories per view: "
    f"{source_weight_sum(source, DEFAULT_QUANTIZATION_SCALE) * PHOTON_MULTIPLIER}"
)
loss = zero_history_fraction(source, DEFAULT_QUANTIZATION_SCALE)
print(f"Quantization loss: {loss:.4%}")
print(f"Total projection sum (activity-scaled): {float(total.sum()):.3f}")
print(f"Elapsed: {elapsed_s:.0f} s")

# %%
# Plot the result
# ---------------
# The left panel is the central slice of the input source. The middle panel is
# the first projection view of the total (scatter plus primary) window, and the
# right panel is a profile through its centre row. The hot sphere is the bright
# disc on the right of the body and the peak on the profile.

source_slice = source[source.shape[0] // 2, :, :]
total_view0 = projection_view0(total)
profile_row = total_view0[total_view0.shape[0] // 2]

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
axes[0].imshow(source_slice, cmap="viridis")
axes[0].set_title("Input source")
total_image = axes[1].imshow(total_view0, cmap="magma")
axes[1].set_title("Total (view 0)")
fig.colorbar(total_image, ax=axes[1], shrink=0.8)
axes[2].plot(profile_row, color="black")
axes[2].axvline(22, color="tab:red", linestyle="--", label="Hot sphere")
axes[2].set_title("Centre-row profile")
axes[2].set_xlabel("Bin")
axes[2].set_ylabel("Counts")
axes[2].legend()
for ax in axes[:2]:
    ax.axis("off")
fig.savefig(output_dir / "basic_simulation_summary.png", dpi=140)
plt.show()
