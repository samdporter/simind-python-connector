"""
SCATTWIN vs PENETRATE comparison
================================

This example compares the SCATTWIN and PENETRATE scoring routines of SIMIND on
the same phantom using the pure Python connector. It is a quick low-history
run, so the component counts are noisy.
"""

# %%
# Setup
# -----
# Check that SIMIND is installed and choose where the outputs are written.

import os
import shutil
from pathlib import Path

import matplotlib.cm as cm
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
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "routine_comparison"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the phantom
# -----------------
# A cylindrical attenuating body on a 32 x 32 x 32 grid (z, y, x) with the
# activity confined to a hot sphere inside it. Zero voxels emit no histories,
# so all of the history budget goes to the sphere. The run is also reduced to
# one view of 16 x 16 pixels (8.8 mm), so that the few histories fill the image.

dims = (32, 32, 32)
z, y, x = np.indices(dims)

body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

source = np.zeros(dims, dtype=np.float32)
source[hot] = 2.0

mu_map = np.zeros_like(source)
mu_map[body] = 0.15


# %%
# Define one simulation case
# --------------------------
# Both cases use the same phantom and settings; only the scoring routine
# differs (1 is SCATTWIN, 4 is PENETRATE).


def run_case(
    case_dir: Path, prefix: str, scoring_routine: int
) -> dict[str, np.ndarray]:
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
    connector.add_config_value(1, 140.0)
    # Use full detector-hit acceptance for both SCATTWIN and PENETRATE in this
    # comparison example.
    connector.add_config_value(19, -90)
    connector.add_config_value(28, 0.88)
    connector.add_config_value(29, 1)
    connector.add_config_value(76, 16)
    connector.add_config_value(77, 16)
    connector.add_config_value(53, 1)
    connector.set_energy_windows([75.0], [225.0], [0])
    connector.add_runtime_switch("FI", "y90_tissue")
    connector.add_runtime_switch("CC", "ma-megp")
    connector.add_runtime_switch("NN", 1)
    connector.add_runtime_switch("RR", 12345)

    results = connector.run()
    return {key: value.projection for key, value in results.items()}


# %%
# Run both routines
# -----------------

scatt_dir = output_dir / "scattwin"
pen_dir = output_dir / "penetrate"
scatt_dir.mkdir(parents=True, exist_ok=True)
pen_dir.mkdir(parents=True, exist_ok=True)

scatt_outputs = run_case(scatt_dir, "scatt_case01", scoring_routine=1)
pen_outputs = run_case(pen_dir, "pen_case01", scoring_routine=4)

scatt_total = scatt_outputs["tot_w1"]
scatt_scatter = scatt_outputs["sca_w1"]
scatt_primary = scatt_outputs["pri_w1"]
scatt_primary_sum = float(scatt_primary.sum())
scatt_scatter_sum = float(scatt_scatter.sum())

# %%
# Pick a PENETRATE projection to preview
# --------------------------------------
# PENETRATE writes many component projections. The preview is the first of a
# preferred list that is present, falling back to the first key alphabetically.

preferred = ("all_interactions_w1", "all_interactions", "tot_w1", "b01")
pen_key = next((key for key in preferred if key in pen_outputs), None)
if pen_key is None:
    pen_key = sorted(pen_outputs)[0]
pen_preview = pen_outputs[pen_key]

np.save(output_dir / "scatt_total.npy", scatt_total)
np.save(output_dir / "scatt_scatter.npy", scatt_scatter)
np.save(output_dir / "scatt_primary.npy", scatt_primary)
np.save(output_dir / f"penetrate_{pen_key}.npy", pen_preview)

with open(output_dir / "penetrate_components.txt", "w", encoding="utf-8") as handle:
    for key in sorted(pen_outputs):
        handle.write(f"{key}\t{float(pen_outputs[key].sum()):.6f}\n")

print(f"SCATTWIN total counts: {float(scatt_total.sum()):.3f}")
print(f"SCATTWIN scatter counts: {float(scatt_scatter.sum()):.3f}")
print(f"SCATTWIN primary counts: {float(scatt_primary.sum()):.3f}")
print(f"PENETRATE preview key: {pen_key}")
print(f"PENETRATE preview counts: {float(pen_preview.sum()):.3f}")

# %%
# Compare the projections
# -----------------------
# All projection panels share one colour scale.


def view0(projection: np.ndarray) -> np.ndarray:
    return projection[0, :, 0, :] if projection.ndim == 4 else projection[0, :, :]


scatt_total_view0 = view0(scatt_total)
scatt_primary_view0 = view0(scatt_primary)
pen_preview_view0 = view0(pen_preview)
views = [scatt_total_view0, scatt_primary_view0, pen_preview_view0]
proj_vmin = float(min(v.min() for v in views))
proj_vmax = float(max(v.max() for v in views))

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(source[source.shape[0] // 2, :, :], cmap="viridis")
axes[0].set_title("Input source")
axes[1].imshow(scatt_total_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
axes[1].set_title("SCATTWIN total")
axes[2].imshow(scatt_primary_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
axes[2].set_title("SCATTWIN primary")
axes[3].imshow(pen_preview_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
axes[3].set_title(f"PENETRATE {pen_key}")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_dir / "routine_comparison.png", dpi=140)

# %%
# Define the component bar plots
# ------------------------------
# Both plots stack the PENETRATE components (primary in green, scattered in
# red) next to the SCATTWIN primary and scatter counts.


def save_total_bar_comparison(
    scatt_primary_sum: float,
    scatt_scatter_sum: float,
    pen_outputs: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    scatt_primary_color = cm.Greens(0.6)
    scatt_scatter_color = cm.Reds(0.6)

    # SCATTWIN total bar: primary + scatter.
    ax.bar(
        0,
        scatt_primary_sum,
        color=scatt_primary_color,
        alpha=0.9,
        width=0.7,
        label="SCATTWIN Primary",
    )
    ax.bar(
        0,
        scatt_scatter_sum,
        bottom=scatt_primary_sum,
        color=scatt_scatter_color,
        alpha=0.9,
        width=0.7,
        label="SCATTWIN Scatter",
    )

    # PENETRATE total bar: stacked detailed components.
    primary_components = [
        ("geom_coll_primary", scatt_primary_color, "Primary: Geom Coll"),
        ("septal_pen_primary", cm.Greens(0.85), "Primary: Septal Pen"),
        ("coll_scatter_primary", cm.Greens(0.45), "Primary: Coll Scatter"),
        ("coll_xray_primary", cm.Greens(0.30), "Primary: Coll X-ray"),
    ]
    scatter_components = [
        ("geom_coll_scattered", scatt_scatter_color, "Scatter: Geom Coll"),
        ("septal_pen_scattered", cm.Reds(0.85), "Scatter: Septal Pen"),
        ("coll_scatter_scattered", cm.Reds(0.45), "Scatter: Coll Scatter"),
        ("coll_xray_scattered", cm.Reds(0.30), "Scatter: Coll X-ray"),
    ]
    primary_backscatter_components = [
        "geom_coll_primary_back",
        "septal_pen_primary_back",
        "coll_scatter_primary_back",
        "coll_xray_primary_back",
    ]
    scatter_backscatter_components = [
        "geom_coll_scattered_back",
        "septal_pen_scattered_back",
        "coll_scatter_scattered_back",
        "coll_xray_scattered_back",
    ]
    primary_backscatter_total = sum(
        float(pen_outputs.get(comp, np.array(0.0)).sum())
        if comp in pen_outputs
        else 0.0
        for comp in primary_backscatter_components
    )
    scatter_backscatter_total = sum(
        float(pen_outputs.get(comp, np.array(0.0)).sum())
        if comp in pen_outputs
        else 0.0
        for comp in scatter_backscatter_components
    )

    pen_x = 1
    pen_bottom = 0.0
    for comp_slug, color, label in primary_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax.bar(
                    pen_x,
                    val,
                    bottom=pen_bottom,
                    color=color,
                    alpha=0.9,
                    width=0.7,
                    label=label,
                )
                pen_bottom += val
    if primary_backscatter_total > 0:
        ax.bar(
            pen_x,
            primary_backscatter_total,
            bottom=pen_bottom,
            color=cm.Greens(0.2),
            alpha=0.9,
            width=0.7,
            label="Primary Backscatter",
        )
        pen_bottom += primary_backscatter_total

    for comp_slug, color, label in scatter_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax.bar(
                    pen_x,
                    val,
                    bottom=pen_bottom,
                    color=color,
                    alpha=0.9,
                    width=0.7,
                    label=label,
                )
                pen_bottom += val
    if scatter_backscatter_total > 0:
        ax.bar(
            pen_x,
            scatter_backscatter_total,
            bottom=pen_bottom,
            color=cm.Reds(0.2),
            alpha=0.9,
            width=0.7,
            label="Scatter Backscatter",
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SCATTWIN total", "PENETRATE total"])
    ax.set_ylabel("Counts")
    ax.set_title("SCATTWIN vs PENETRATE Total Components")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)


def save_penetrate_count_bars(
    scatt_total: np.ndarray,
    scatt_scatter: np.ndarray,
    scatt_primary: np.ndarray,
    pen_outputs: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    scatt_primary_sum = float(scatt_primary.sum())
    scatt_scatter_sum = float(scatt_scatter.sum())

    scatt_primary_color = cm.Greens(0.6)
    scatt_scatter_color = cm.Reds(0.6)

    ax1.bar(
        "Total",
        scatt_primary_sum,
        color=scatt_primary_color,
        alpha=0.9,
        label="Primary",
    )
    ax1.bar(
        "Total",
        scatt_scatter_sum,
        bottom=scatt_primary_sum,
        color=scatt_scatter_color,
        alpha=0.9,
        label="Scatter",
    )
    ax1.bar("Primary", scatt_primary_sum, color=scatt_primary_color, alpha=0.9)
    ax1.bar("Scatter", scatt_scatter_sum, color=scatt_scatter_color, alpha=0.9)
    ax1.set_title("SCATTWIN Count Components")
    ax1.set_ylabel("Counts")
    ax1.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax1.legend()

    # Keep Geom Coll colors aligned with SCATTWIN primary/scatter bars so the
    # two routine panels are directly comparable at a glance.
    primary_components = [
        ("geom_coll_primary", scatt_primary_color, "Geom Coll"),
        ("septal_pen_primary", cm.Greens(0.85), "Septal Pen"),
        ("coll_scatter_primary", cm.Greens(0.45), "Coll Scatter"),
        ("coll_xray_primary", cm.Greens(0.30), "Coll X-ray"),
    ]
    scatter_components = [
        ("geom_coll_scattered", scatt_scatter_color, "Geom Coll"),
        ("septal_pen_scattered", cm.Reds(0.85), "Septal Pen"),
        ("coll_scatter_scattered", cm.Reds(0.45), "Coll Scatter"),
        ("coll_xray_scattered", cm.Reds(0.30), "Coll X-ray"),
    ]

    primary_backscatter_components = [
        "geom_coll_primary_back",
        "septal_pen_primary_back",
        "coll_scatter_primary_back",
        "coll_xray_primary_back",
    ]
    scatter_backscatter_components = [
        "geom_coll_scattered_back",
        "septal_pen_scattered_back",
        "coll_scatter_scattered_back",
        "coll_xray_scattered_back",
    ]
    primary_backscatter_total = sum(
        float(pen_outputs.get(comp, np.array(0.0)).sum())
        if comp in pen_outputs
        else 0.0
        for comp in primary_backscatter_components
    )
    scatter_backscatter_total = sum(
        float(pen_outputs.get(comp, np.array(0.0)).sum())
        if comp in pen_outputs
        else 0.0
        for comp in scatter_backscatter_components
    )

    primary_backscatter_color = cm.Greens(0.3)
    scatter_backscatter_color = cm.Reds(0.3)

    x_pos = 0
    bottom = 0.0
    for comp_slug, color, label in primary_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax2.bar(
                    x_pos,
                    val,
                    bottom=bottom,
                    color=color,
                    alpha=0.9,
                    width=0.8,
                    label=f"Primary: {label}",
                )
                bottom += val
    if primary_backscatter_total > 0:
        ax2.bar(
            x_pos,
            primary_backscatter_total,
            bottom=bottom,
            color=primary_backscatter_color,
            alpha=0.9,
            width=0.8,
            label="Primary Backscatter",
        )
        bottom += primary_backscatter_total

    for comp_slug, color, label in scatter_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax2.bar(
                    x_pos,
                    val,
                    bottom=bottom,
                    color=color,
                    alpha=0.9,
                    width=0.8,
                    label=f"Scatter: {label}",
                )
                bottom += val
    if scatter_backscatter_total > 0:
        ax2.bar(
            x_pos,
            scatter_backscatter_total,
            bottom=bottom,
            color=scatter_backscatter_color,
            alpha=0.9,
            width=0.8,
            label="Scatter Backscatter",
        )
        bottom += scatter_backscatter_total

    x_pos = 1
    bottom = 0.0
    for comp_slug, color, _ in primary_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax2.bar(x_pos, val, bottom=bottom, color=color, alpha=0.9, width=0.8)
                bottom += val
    if primary_backscatter_total > 0:
        ax2.bar(
            x_pos,
            primary_backscatter_total,
            bottom=bottom,
            color=primary_backscatter_color,
            alpha=0.9,
            width=0.8,
        )
        bottom += primary_backscatter_total

    x_pos = 2
    bottom = 0.0
    for comp_slug, color, _ in scatter_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax2.bar(x_pos, val, bottom=bottom, color=color, alpha=0.9, width=0.8)
                bottom += val
    if scatter_backscatter_total > 0:
        ax2.bar(
            x_pos,
            scatter_backscatter_total,
            bottom=bottom,
            color=scatter_backscatter_color,
            alpha=0.9,
            width=0.8,
        )
        bottom += scatter_backscatter_total

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Total", "Primary", "Scatter"])
    ax2.set_title("PENETRATE Count Components (Stacked)")
    ax2.set_ylabel("Counts")
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)


# %%
# Component counts
# ----------------
# SCATTWIN counts on the left, the stacked PENETRATE components on the right.

save_penetrate_count_bars(
    scatt_total,
    scatt_scatter,
    scatt_primary,
    pen_outputs,
    output_dir / "routine_component_counts.png",
)

# %%
# Totals side by side
# -------------------
# The two stacked bars should agree in total if both routines score the same
# detector hits.

save_total_bar_comparison(
    scatt_primary_sum,
    scatt_scatter_sum,
    pen_outputs,
    output_dir / "routine_total_comparison.png",
)
plt.show()
