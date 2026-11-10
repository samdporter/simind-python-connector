"""Compare SCATTWIN and PENETRATE scoring on one hot-sphere phantom.

Both routines score the same histories with the same phantom, window, history
multiplier and seed. SCATTWIN needs ``CA=2`` to split primary and scatter;
PENETRATE needs ``CA=0`` and writes its components separately. The totals agree
by construction, so the useful comparison is the component decomposition and
the count-preservation check.
"""

import os
import shutil
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE
from simind_python_connector.utils.projection_metrics import (
    count_preservation_error,
    require_output_keys,
)


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "routine_comparison"
)
output_dir.mkdir(parents=True, exist_ok=True)

dims = (32, 32, 32)
z, y, x = np.indices(dims)
body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2
source = np.zeros(dims, dtype=np.float32)
source[hot] = 2.0
mu_map = np.zeros_like(source)
mu_map[body] = 0.15

PHOTON_MULTIPLIER = 120
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


def run_case(prefix: str, scoring_routine: int) -> dict[str, object]:
    connector = SimindPythonConnector(
        config_source=configs.get("Example.yaml"),
        output_dir=output_dir,
        output_prefix=prefix,
        quantization_scale=DEFAULT_QUANTIZATION_SCALE,
    )
    connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        voxel_size_mm=4.0,
        scoring_routine=scoring_routine,
    )
    connector.add_runtime_switch("CA", 2 if scoring_routine == 1 else 0)
    connector.add_config_value(1, 140.0)
    connector.add_config_value(19, -90)  # full detector-hit acceptance
    connector.add_config_value(28, 0.4)
    connector.add_config_value(29, 1)
    connector.add_config_value(76, 32)
    connector.add_config_value(77, 32)
    connector.add_config_value(53, 1)
    connector.set_energy_windows([75.0], [225.0], [0])
    connector.add_runtime_switch("FI", "y90_tissue")
    connector.add_runtime_switch("CC", "ma-megp")
    connector.add_runtime_switch("NN", PHOTON_MULTIPLIER)
    connector.add_runtime_switch("RR", 12345)
    return connector.run()


scatt_outputs = run_case("scatt_case01", scoring_routine=1)
pen_outputs = run_case("pen_case01", scoring_routine=4)

scatt = require_output_keys(scatt_outputs, "tot_w1", "pri_w1", "sca_w1")
pen = require_output_keys(pen_outputs, "all_interactions", *PENETRATE_COMPONENTS)
scatt_primary = scatt["pri_w1"].projection
scatt_scatter = scatt["sca_w1"].projection
pen_total = pen["all_interactions"].projection

scatt_preservation = count_preservation_error(
    scatt_outputs, "tot_w1", "pri_w1", "sca_w1"
)
pen_preservation = count_preservation_error(
    pen_outputs, "all_interactions", *PENETRATE_COMPONENTS
)
print(f"SCATTWIN count preservation: {scatt_preservation:.2%}")
print(f"PENETRATE count preservation: {pen_preservation:.2%}")
print(
    f"SCATTWIN primary and scatter: {float(scatt_primary.sum()):.3f}, "
    f"{float(scatt_scatter.sum()):.3f}"
)

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
    "SCATTWIN", scatt_primary_counts, color=cm.Greens(0.6), label="Primary", width=0.55
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
for index, value in enumerate(pen_group_sums.values()):
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
    cellText=rows[1:], colLabels=rows[0], cellLoc="left", colLoc="left", loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.25)
fig.savefig(output_dir / "routine_decomposition.png", dpi=140)
plt.show()
