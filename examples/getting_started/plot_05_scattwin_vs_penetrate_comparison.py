"""
SCATTWIN vs PENETRATE comparison
================================

This example compares the SCATTWIN and PENETRATE scoring routines of SIMIND on
the same phantom, with the same geometry, histories and detector acceptance.
It replaces the earlier 16 x 16 preview with a 32 x 32 grid of 4 mm bins and a
history budget large enough for the component decomposition.

The physical settings are shared: the same hot-sphere source, attenuation map,
one 75-225 keV view, ``NN`` history multiplier and seed. SIMIND needs
``CA=2`` to split SCATTWIN into primary and scatter, while PENETRATE writes its
component projections only with ``CA=0``; that difference is part of each
routine, not a change of phantom or acquisition. Both runs keep full detector
acceptance (Index 19 = -90).
"""

# %%
# Setup
# -----
# Check that SIMIND is installed and choose where the outputs are written.

import os
import shutil
import time
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE
from simind_python_connector.utils.projection_metrics import (
    count_preservation_error,
    projection_view0,
    require_output_keys,
    source_weight_sum,
    symmetric_limits,
    zero_history_fraction,
)


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "routine_comparison"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the phantom
# -----------------
# A cylindrical attenuating body on a 32 x 32 x 32 grid (z, y, x) with the
# activity confined to a hot sphere inside it. Zero voxels emit no histories,
# so the whole history budget goes to the sphere. The projection is one view of
# 32 x 32 bins of 4 mm.

dims = (32, 32, 32)
z, y, x = np.indices(dims)

body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

source = np.zeros(dims, dtype=np.float32)
source[hot] = 2.0

mu_map = np.zeros_like(source)
mu_map[body] = 0.15

PHOTON_MULTIPLIER = 120
SEED = 12345
PENETRATE_COMPONENTS = (
    "geom_coll_primary",
    "septal_pen_primary",
    "coll_scatter_primary",
    "coll_xray_primary",
    "geom_coll_primary_back",
    "septal_pen_primary_back",
    "coll_scatter_primary_back",
    "coll_xray_primary_back",
    "geom_coll_scattered",
    "septal_pen_scattered",
    "coll_scatter_scattered",
    "coll_xray_scattered",
    "geom_coll_scattered_back",
    "septal_pen_scattered_back",
    "coll_scatter_scattered_back",
    "coll_xray_scattered_back",
)


# %%
# Define one simulation case
# --------------------------
# Both cases use the same phantom, energy window, history budget and seed.
# Only the scoring routine and its required ``CA`` switch differ (1 is
# SCATTWIN, 4 is PENETRATE).


def run_case(
    case_dir: Path, prefix: str, scoring_routine: int
) -> tuple[dict[str, object], float]:
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
        scoring_routine=scoring_routine,
    )
    if scoring_routine == 1:
        connector.add_runtime_switch("CA", 2)
    else:
        connector.add_runtime_switch("CA", 0)
    connector.add_config_value(1, 140.0)
    # Full detector-hit acceptance for both routines.
    connector.add_config_value(19, -90)
    connector.add_config_value(28, 0.4)
    connector.add_config_value(29, 1)
    connector.add_config_value(76, 32)
    connector.add_config_value(77, 32)
    connector.add_config_value(53, 1)
    connector.set_energy_windows([75.0], [225.0], [0])
    connector.add_runtime_switch("FI", "y90_tissue")
    connector.add_runtime_switch("CC", "ma-megp")
    connector.add_runtime_switch("NN", PHOTON_MULTIPLIER)
    connector.add_runtime_switch("RR", SEED)

    started = time.perf_counter()
    results = connector.run()
    elapsed_s = time.perf_counter() - started
    return results, elapsed_s


# %%
# Run both routines
# -----------------

scatt_dir = output_dir / "scattwin"
pen_dir = output_dir / "penetrate"
scatt_dir.mkdir(parents=True, exist_ok=True)
pen_dir.mkdir(parents=True, exist_ok=True)

scatt_outputs, scatt_seconds = run_case(scatt_dir, "scatt_case01", scoring_routine=1)
pen_outputs, pen_seconds = run_case(pen_dir, "pen_case01", scoring_routine=4)

scatt = require_output_keys(scatt_outputs, "tot_w1", "pri_w1", "sca_w1")
scatt_total = scatt["tot_w1"].projection
scatt_primary = scatt["pri_w1"].projection
scatt_scatter = scatt["sca_w1"].projection

pen = require_output_keys(pen_outputs, "all_interactions", *PENETRATE_COMPONENTS)
pen_total = pen["all_interactions"].projection
pen_primary = sum(
    pen[key].projection
    for key in (
        "geom_coll_primary",
        "septal_pen_primary",
        "coll_scatter_primary",
        "coll_xray_primary",
    )
)

np.save(output_dir / "scatt_total.npy", scatt_total)
np.save(output_dir / "scatt_scatter.npy", scatt_scatter)
np.save(output_dir / "scatt_primary.npy", scatt_primary)
np.save(output_dir / "penetrate_total.npy", pen_total)
np.save(output_dir / "penetrate_primary.npy", pen_primary)

print(
    "Histories per view: "
    f"{source_weight_sum(source, DEFAULT_QUANTIZATION_SCALE) * PHOTON_MULTIPLIER}"
)
loss = zero_history_fraction(source, DEFAULT_QUANTIZATION_SCALE)
print(f"Quantization loss: {loss:.4%}")
print(f"SCATTWIN total projection sum: {float(scatt_total.sum()):.3f}")
print(f"SCATTWIN scatter projection sum: {float(scatt_scatter.sum()):.3f}")
print(f"SCATTWIN primary projection sum: {float(scatt_primary.sum()):.3f}")
print(f"PENETRATE total projection sum: {float(pen_total.sum()):.3f}")
print(f"PENETRATE primary projection sum: {float(pen_primary.sum()):.3f}")
preservation = count_preservation_error(scatt_outputs, "tot_w1", "pri_w1", "sca_w1")
pen_preservation = count_preservation_error(
    pen_outputs, "all_interactions", *PENETRATE_COMPONENTS
)
print(f"SCATTWIN count preservation: {preservation:.2%}")
print(f"PENETRATE count preservation: {pen_preservation:.2%}")
print(f"Routine total difference: {float((pen_total - scatt_total).sum()):.3f}")
print(f"Elapsed: {scatt_seconds:.0f} s and {pen_seconds:.0f} s")

# %%
# Compare the projections
# -----------------------
# The total panels share one colour scale. The difference panel is centred on
# zero, so a small difference means the two routines scored the same hits on
# this one view.

scatt_view0 = projection_view0(scatt_total)
pen_view0 = projection_view0(pen_total)
difference_view0 = pen_view0 - scatt_view0
projection_vmax = float(max(scatt_view0.max(), pen_view0.max()))
difference_limits = symmetric_limits(difference_view0)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
axes[0].imshow(source[source.shape[0] // 2, :, :], cmap="viridis")
axes[0].set_title("Input source")
total_image = axes[1].imshow(scatt_view0, cmap="magma", vmin=0.0, vmax=projection_vmax)
axes[1].set_title("SCATTWIN total")
fig.colorbar(total_image, ax=axes[1], shrink=0.8)
axes[2].imshow(pen_view0, cmap="magma", vmin=0.0, vmax=projection_vmax)
axes[2].set_title("PENETRATE total")
difference_image = axes[3].imshow(
    difference_view0,
    cmap="coolwarm",
    vmin=difference_limits[0],
    vmax=difference_limits[1],
)
axes[3].set_title("PENETRATE - SCATTWIN")
fig.colorbar(difference_image, ax=axes[3], shrink=0.8)
for ax in axes:
    ax.axis("off")
fig.savefig(output_dir / "routine_comparison.png", dpi=140)

# %%
# Component decomposition
# -----------------------
# The left panel stacks the two routines' primary and scattered counts. The
# right panel lists the same numbers, with each PENETRATE group as a share of
# its total. The groups are the geometric-collimated, septal-penetration,
# collimator-scatter and collimator-X-ray components, plus their backscatter
# partners.

component_groups = {
    "Geom. coll. primary": ("geom_coll_primary",),
    "Septal penetration primary": ("septal_pen_primary",),
    "Collimator scatter primary": ("coll_scatter_primary",),
    "Collimator X-ray primary": ("coll_xray_primary",),
    "Primary backscatter": (
        "geom_coll_primary_back",
        "septal_pen_primary_back",
        "coll_scatter_primary_back",
        "coll_xray_primary_back",
    ),
    "Geom. coll. scattered": ("geom_coll_scattered",),
    "Septal penetration scattered": ("septal_pen_scattered",),
    "Collimator scatter scattered": ("coll_scatter_scattered",),
    "Collimator X-ray scattered": ("coll_xray_scattered",),
    "Scatter backscatter": (
        "geom_coll_scattered_back",
        "septal_pen_scattered_back",
        "coll_scatter_scattered_back",
        "coll_xray_scattered_back",
    ),
}
pen_group_sums = {
    label: float(sum(pen[key].projection.sum() for key in keys))
    for label, keys in component_groups.items()
}
pen_total_counts = float(pen_total.sum())
scatt_primary_counts = float(scatt_primary.sum())
scatt_scatter_counts = float(scatt_scatter.sum())

fig, (bar_ax, table_ax) = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    gridspec_kw={"width_ratios": (1, 1.5)},
    constrained_layout=True,
)
bar_ax.bar(
    "SCATTWIN",
    scatt_primary_counts,
    color=cm.Greens(0.6),
    label="Primary",
    width=0.55,
)
bar_ax.bar(
    "SCATTWIN",
    scatt_scatter_counts,
    bottom=scatt_primary_counts,
    color=cm.Reds(0.6),
    label="Scatter",
    width=0.55,
)
bottom = 0.0
for index, (label, value) in enumerate(pen_group_sums.items()):
    color = (
        cm.Greens(0.75 - 0.1 * (index % 4))
        if index < 5
        else cm.Reds(0.75 - 0.1 * (index % 4))
    )
    bar_ax.bar("PENETRATE", value, bottom=bottom, color=color, width=0.55)
    bottom += value
bar_ax.set_ylabel("Counts")
bar_ax.set_title("SCATTWIN vs PENETRATE decomposition")
bar_ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
bar_ax.legend(loc="upper right", fontsize=8)

table_ax.axis("off")
rows = [["Component", "Counts", "Share of PENETRATE total"]]
rows.extend(
    [label, f"{value:.4g}", f"{value / pen_total_counts:.2%}"]
    for label, value in pen_group_sums.items()
)
rows.append(["PENETRATE total", f"{pen_total_counts:.4g}", "100.00%"])
rows.append(["SCATTWIN primary", f"{scatt_primary_counts:.4g}", ""])
rows.append(["SCATTWIN scatter", f"{scatt_scatter_counts:.4g}", ""])
table = table_ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    cellLoc="left",
    colLoc="left",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.25)
fig.savefig(output_dir / "routine_decomposition.png", dpi=140)
plt.show()

with open(output_dir / "routine_decomposition.csv", "w", encoding="utf-8") as handle:
    handle.write("component,counts,share_of_penetrate_total\n")
    for label, value in pen_group_sums.items():
        handle.write(f"{label},{value:.6g},{value / pen_total_counts:.6%}\n")
