"""
Seed reproducibility
====================

This example runs the same simulation twice with two independent random seeds
and compares the projections. The only runtime switch that differs is ``RR``,
so the difference isolates Monte Carlo noise, not a change of physics.

Both runs use the default source quantization, four 360-degree views of
32 x 32 bins at 4 mm, and a history multiplier of 3, which gives 3,464,700
histories per view. That keeps both runs inside a two-hour budget together,
while each view still holds enough counts for a stable repeatability measure.
The figure uses a shared count scale for the two projections and a zero-centred
symmetric scale for their difference.
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
    normalised_rms_difference,
    projection_view0,
    require_output_keys,
    symmetric_limits,
    zero_history_fraction,
)


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
    / "runtime_switch_comparison"
)
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

PHOTON_MULTIPLIER = 3


# %%
# Define one simulation case
# --------------------------
# Both cases share every setting except the random seed (runtime switch
# ``RR``).


def run_case(case_dir: Path, prefix: str, seed: int) -> tuple[np.ndarray, float]:
    connector = SimindPythonConnector(
        config_source=configs.get("Example.yaml"),
        output_dir=case_dir,
        output_prefix=prefix,
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
    connector.add_config_value(1, 140.0)
    connector.add_config_value(53, 0)
    connector.set_energy_windows([126.0], [154.0], [0])
    connector.add_runtime_switch("CA", 0)
    connector.add_runtime_switch("FI", "tc99m")
    connector.add_runtime_switch("CC", "ma-lehr")
    connector.add_runtime_switch("NN", PHOTON_MULTIPLIER)
    connector.add_runtime_switch("RR", seed)
    started = time.perf_counter()
    outputs = connector.run()
    elapsed_s = time.perf_counter() - started
    projection = require_output_keys(outputs, "tot_w1")["tot_w1"].projection
    return projection, elapsed_s


# %%
# Run both seeds
# --------------

base_projection, base_seconds = run_case(output_dir / "seed_12345", "seed_12345", 12345)
alt_projection, alt_seconds = run_case(output_dir / "seed_54321", "seed_54321", 54321)

np.save(output_dir / "projection_seed_12345.npy", base_projection)
np.save(output_dir / "projection_seed_54321.npy", alt_projection)

# %%
# Compare the projections
# -----------------------
# The two projections share one count scale. The difference panel is centred
# on zero, so positive and negative Monte Carlo fluctuations are symmetric.
# The relative RMS difference is a repeatability measure: zero for identical
# runs, and about the inverse square root of the counts for two independent
# Monte Carlo runs of the same expectation.

base_view = projection_view0(base_projection)
alt_view = projection_view0(alt_projection)
difference = alt_view - base_view
repeatability = normalised_rms_difference(alt_view, base_view)

projection_vmax = float(max(base_view.max(), alt_view.max()))
difference_limits = symmetric_limits(difference)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
axes[0].imshow(source[source.shape[0] // 2, :, :], cmap="viridis")
axes[0].set_title("Input source")
base_image = axes[1].imshow(base_view, cmap="magma", vmin=0.0, vmax=projection_vmax)
axes[1].set_title("Seed 12345 (view 0)")
fig.colorbar(base_image, ax=axes[1], shrink=0.8)
axes[2].imshow(alt_view, cmap="magma", vmin=0.0, vmax=projection_vmax)
axes[2].set_title("Seed 54321 (view 0)")
difference_image = axes[3].imshow(
    difference, cmap="coolwarm", vmin=difference_limits[0], vmax=difference_limits[1]
)
axes[3].set_title("Seed 54321 - seed 12345")
fig.colorbar(difference_image, ax=axes[3], shrink=0.8)
for ax in axes:
    ax.axis("off")
fig.savefig(output_dir / "runtime_switch_comparison.png", dpi=140)
plt.show()

print(f"Seed 12345 projection sum: {float(base_projection.sum()):.3f}")
print(f"Seed 54321 projection sum: {float(alt_projection.sum()):.3f}")
print(f"Difference sum (view 0): {float(difference.sum()):.3f}")
print(f"Relative RMS difference: {repeatability:.3%}")
loss = zero_history_fraction(source, DEFAULT_QUANTIZATION_SCALE)
print(f"Quantization loss: {loss:.4%}")
print(f"Elapsed: {base_seconds:.0f} s and {alt_seconds:.0f} s")
