"""
Runtime-switch comparison
=========================

This example runs two SIMIND cases with different runtime switch settings and
compares the resulting projection view. It is a quick low-history run, so the
seed-to-seed noise is large.
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


# %%
# Define one simulation case
# --------------------------
# Both cases share every setting except the random seed (runtime switch
# ``RR``).


def run_case(case_dir: Path, prefix: str, seed: int) -> np.ndarray:
    connector = SimindPythonConnector(
        config_source=configs.get("Example.yaml"),
        output_dir=case_dir,
        output_prefix=prefix,
        quantization_scale=0.005 * DEFAULT_QUANTIZATION_SCALE,
    )
    connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        voxel_size_mm=4.0,
        scoring_routine=1,
    )
    connector.add_config_value(1, 140.0)
    connector.add_config_value(19, 2)
    connector.add_config_value(53, 0)
    connector.set_energy_windows([75.0], [225.0], [0])
    connector.add_runtime_switch("FI", "y90_tissue")
    connector.add_runtime_switch("CC", "ma-megp")
    connector.add_runtime_switch("NN", 1)
    connector.add_runtime_switch("RR", seed)
    outputs = connector.run()
    return outputs["tot_w1"].projection


# %%
# Run both seeds
# --------------

base_projection = run_case(output_dir / "seed_12345", "seed_12345", seed=12345)
alt_projection = run_case(output_dir / "seed_54321", "seed_54321", seed=54321)

np.save(output_dir / "projection_seed_12345.npy", base_projection)
np.save(output_dir / "projection_seed_54321.npy", alt_projection)

# %%
# Compare the projections
# -----------------------
# The difference panel shows the Monte Carlo noise that the seed alone
# introduces.


def view0(projection: np.ndarray) -> np.ndarray:
    return projection[0, :, 0, :] if projection.ndim == 4 else projection[0, :, :]


base_view = view0(base_projection)
alt_view = view0(alt_projection)
diff_view = alt_view - base_view

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(source[source.shape[0] // 2, :, :], cmap="viridis")
axes[0].set_title("Input source")
axes[1].imshow(base_view, cmap="magma")
axes[1].set_title("Seed 12345")
axes[2].imshow(alt_view, cmap="magma")
axes[2].set_title("Seed 54321")
axes[3].imshow(diff_view, cmap="coolwarm")
axes[3].set_title("Difference")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_dir / "runtime_switch_comparison.png", dpi=140)
plt.show()

print(f"Seed 12345 counts: {float(base_projection.sum()):.3f}")
print(f"Seed 54321 counts: {float(alt_projection.sum()):.3f}")
print(f"Difference sum: {float(diff_view.sum()):.3f}")
