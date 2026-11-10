"""Basic simulation with a seed check.

Run SIMIND from a voxelized source and attenuation map, read the total
projection back as a NumPy array, and compare two random seeds. The same
``RR`` seed reproduces a run exactly; a different seed changes only the Monte
Carlo noise. Geometry is four 360-degree views of 32 x 32 bins at 4 mm, with
the default source quantization and an eightfold history multiplier.
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE
from simind_python_connector.utils.interfile import ProjectionGeometry
from simind_python_connector.utils.projection_metrics import (
    projection_view0,
    require_output_keys,
)


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "basic_simulation"
)
output_dir.mkdir(parents=True, exist_ok=True)

dims = (32, 32, 32)
z, y, x = np.indices(dims)
body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

source = np.zeros(dims, dtype=np.float32)
source[body] = 0.8
source[hot] = 2.0
mu_map = np.zeros_like(source)
mu_map[body] = 0.15

PHOTON_MULTIPLIER = 8


def run_case(seed: int, prefix: str) -> np.ndarray:
    """Run the same simulation with one random seed and return the total."""
    connector = SimindPythonConnector(
        config_source=configs.get("Example.yaml"),
        output_dir=output_dir,
        output_prefix=prefix,
        quantization_scale=DEFAULT_QUANTIZATION_SCALE,
    )
    connector.configure_voxel_phantom(
        source=source, mu_map=mu_map, voxel_size_mm=4.0, scoring_routine=1
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
    outputs = connector.run()
    return require_output_keys(outputs, "tot_w1")["tot_w1"].projection


first = run_case(12345, "basic_seed_12345")
second = run_case(54321, "basic_seed_54321")
print(f"First projection shape: {first.shape}, sum: {float(first.sum()):.3f}")
print(f"Second projection sum: {float(second.sum()):.3f}")
print(f"Mean absolute difference: {float(np.mean(np.abs(second - first))):.4f}")

view = projection_view0(first)
profile = view[view.shape[0] // 2]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
axes[0].imshow(source[source.shape[0] // 2], cmap="viridis")
axes[0].set_title("Input source")
image = axes[1].imshow(view, cmap="magma")
axes[1].set_title("Total (view 0)")
fig.colorbar(image, ax=axes[1], shrink=0.8)
axes[2].plot(profile, color="black")
axes[2].axvline(22, color="tab:red", linestyle="--", label="Hot sphere")
axes[2].set_title("Centre-row profile")
axes[2].set_xlabel("Bin")
axes[2].set_ylabel("Counts")
axes[2].legend()
for axis in axes[:2]:
    axis.axis("off")
fig.savefig(output_dir / "basic_simulation_summary.png", dpi=140)
plt.show()
