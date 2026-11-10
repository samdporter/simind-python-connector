"""Multi-window simulation and TEW scatter correction.

Run Lu-177 with three energy windows and apply a triple-energy-window (TEW)
estimate. The side windows are scaled to the photopeak width and half their
total is subtracted. TEW assumes scatter is linear across the windows; that is
an approximation, and a noisy side window can make the estimate negative, so
the correction is clipped at zero. The clipping fraction reports how much
photopeak activity was removed at zero.
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
    tew_correction,
)


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "multi_window"
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

lowers = [166.0, 187.0, 229.0]
uppers = [187.0, 229.0, 250.0]
connector = SimindPythonConnector(
    config_source=configs.get("Example.yaml"),
    output_dir=output_dir,
    output_prefix="tew_case01",
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
connector.add_config_value(19, 2)
connector.add_config_value(53, 0)
connector.add_config_value(1, 208.0)
connector.set_energy_windows(lowers, uppers, [0, 0, 0])
connector.add_runtime_switch("FI", "lu177")
connector.add_runtime_switch("CC", "ma-megp")
connector.add_runtime_switch("NN", 10)
connector.add_runtime_switch("RR", 12345)

windows = require_output_keys(connector.run(), "tot_w1", "tot_w2", "tot_w3")
lower_scatter = windows["tot_w1"].projection
photopeak = windows["tot_w2"].projection
upper_scatter = windows["tot_w3"].projection
widths = [upper - lower for lower, upper in zip(lowers, uppers)]
scatter_estimate, corrected, clipped_counts, clipped_pixels = tew_correction(
    photopeak,
    lower_scatter,
    upper_scatter,
    lower_width_kev=widths[0],
    photopeak_width_kev=widths[1],
    upper_width_kev=widths[2],
)

print(f"Photopeak projection sum: {float(photopeak.sum()):.3f}")
print(f"TEW scatter estimate sum: {float(scatter_estimate.sum()):.3f}")
print(f"TEW corrected projection sum: {float(corrected.sum()):.3f}")
print(f"Clipped photopeak counts: {clipped_counts:.3%}")
print(f"Clipped pixels: {clipped_pixels:.3%}")

views = [
    projection_view0(lower_scatter),
    projection_view0(photopeak),
    projection_view0(upper_scatter),
    projection_view0(scatter_estimate),
    projection_view0(corrected),
]
scatter_vmax = float(max(views[0].max(), views[2].max(), views[3].max()))
peak_vmax = float(max(views[1].max(), views[4].max()))
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
axes = axes.ravel()
axes[0].imshow(source[source.shape[0] // 2], cmap="viridis")
axes[0].set_title("Input source")
for axis, view, title, vmax in zip(
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
    image = axis.imshow(view, cmap="magma", vmin=0.0, vmax=vmax)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, shrink=0.7)
for axis in axes:
    axis.axis("off")
fig.savefig(output_dir / "multi_window_summary.png", dpi=140)
plt.show()
