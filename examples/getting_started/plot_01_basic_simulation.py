"""
Basic simulation
================

This example runs SIMIND with voxelized NumPy inputs and reads projections
back as NumPy arrays via ``SimindPythonConnector``.
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
# Start from the bundled ``Example.yaml``, attach the voxel phantom, use a
# single 126-154 keV window for Tc-99m and run SIMIND.

connector = SimindPythonConnector(
    config_source=configs.get("Example.yaml"),
    output_dir=output_dir,
    output_prefix="basic_case01",
    quantization_scale=0.01 * DEFAULT_QUANTIZATION_SCALE,
)
source_path, density_path = connector.configure_voxel_phantom(
    source=source,
    mu_map=mu_map,
    voxel_size_mm=4.0,
    scoring_routine=1,
)
connector.add_config_value(1, 140.0)
connector.set_energy_windows([126.0], [154.0], [0])
connector.add_runtime_switch("CA", 0)
connector.add_runtime_switch("FI", "tc99m")
connector.add_runtime_switch("CC", "ma-lehr")
connector.add_runtime_switch("NN", 1)
connector.add_runtime_switch("RR", 12345)

outputs = connector.run()
total = outputs["tot_w1"].projection
np.save(output_dir / "total.npy", total)

print(f"Source file: {source_path}")
print(f"Density file: {density_path}")
print(f"Total counts: {float(total.sum()):.3f}")

# %%
# Plot the result
# ---------------
# The left panel is the central slice of the input source and the right panel
# is the first projection view of the total (scatter plus primary) window.

source_slice = source[source.shape[0] // 2, :, :]
total_view0 = total[0, :, 0, :] if total.ndim == 4 else total[0, :, :]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(source_slice, cmap="viridis")
axes[0].set_title("Input source")
axes[1].imshow(total_view0, cmap="magma")
axes[1].set_title("Total (view 0)")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_dir / "basic_simulation_summary.png", dpi=140)
plt.show()
