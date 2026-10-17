"""
Schneider density conversion
============================

This example demonstrates the Schneider2000-based density conversion, which is
more accurate than the traditional bilinear model.

The Schneider model uses 44 tissue segments with specific HU ranges and
densities, providing clinically accurate tissue density mappings for SPECT
simulation.
"""

# %%
# Setup
# -----

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simind_python_connector.converters.attenuation import (
    compare_density_methods,
    get_schneider_tissue_info,
    hu_to_density,
    hu_to_density_schneider,
)


output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "schneider_density"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Compare the conversion methods
# ------------------------------
# Convert HU values covering the full clinical range with the bilinear and
# Schneider models and print how far they differ.

hu_values = np.linspace(-1050, 2000, 1000)
comparison = compare_density_methods(hu_values)

print(
    f"Max difference (interpolated vs bilinear): "
    f"{comparison['max_diff_interp']:.4f} g/cm³"
)
print(
    f"Max difference (piecewise vs bilinear): "
    f"{comparison['max_diff_piecewise']:.4f} g/cm³"
)
print(
    f"Mean absolute difference (interpolated): "
    f"{comparison['mean_diff_interp']:.4f} g/cm³"
)
print(
    f"Mean absolute difference (piecewise): "
    f"{comparison['mean_diff_piecewise']:.4f} g/cm³"
)

# %%
# Look up tissues
# ---------------
# ``get_schneider_tissue_info`` returns the tissue segment for a HU value.

for hu in [-1000, -500, -100, 0, 50, 200, 500, 1000, 1500]:
    tissue_info = get_schneider_tissue_info(hu)
    if tissue_info:
        print(
            f"HU {hu:4d}: {tissue_info['name']:20s} "
            f"(ρ = {tissue_info['density_g_cm3']:.3f} g/cm³)"
        )

# %%
# Plot the density curves
# -----------------------
# The top row shows the two curves over the full and the clinical range; the
# bottom row shows their difference from the bilinear model.

clinical_mask = (hu_values >= -200) & (hu_values <= 1500)
hu_clinical = hu_values[clinical_mask]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(hu_values, comparison["bilinear"], "b-", label="Bilinear", linewidth=2)
axes[0, 0].plot(
    hu_values,
    comparison["schneider_interpolated"],
    "r--",
    label="Schneider Interpolated",
    linewidth=2,
)
axes[0, 0].set_title("Full Range Density Conversion (-1050 to 2000 HU)")

axes[0, 1].plot(
    hu_clinical,
    comparison["bilinear"][clinical_mask],
    "b-",
    label="Bilinear",
    linewidth=2,
)
axes[0, 1].plot(
    hu_clinical,
    comparison["schneider_interpolated"][clinical_mask],
    "r--",
    label="Schneider Interpolated",
    linewidth=2,
)
axes[0, 1].set_title("Clinical Range (-200 to 1500 HU)")

axes[1, 0].plot(
    hu_values,
    comparison["difference_interpolated"],
    "r-",
    label="Schneider Interp - Bilinear",
    linewidth=2,
)
axes[1, 0].plot(
    hu_values,
    comparison["difference_piecewise"],
    "g-",
    label="Schneider Piece - Bilinear",
    linewidth=2,
)
axes[1, 0].set_title("Density Differences vs Bilinear Model")

axes[1, 1].plot(
    hu_clinical,
    comparison["difference_interpolated"][clinical_mask],
    "r-",
    label="Schneider Interp - Bilinear",
    linewidth=2,
)
axes[1, 1].plot(
    hu_clinical,
    comparison["difference_piecewise"][clinical_mask],
    "g-",
    label="Schneider Piece - Bilinear",
    linewidth=2,
)
axes[1, 1].set_title("Clinical Range Differences")

for ax in axes[1]:
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.set_ylabel("Density Difference (g/cm³)")
for ax in axes[0]:
    ax.set_ylabel("Density (g/cm³)")
for ax in axes.flat:
    ax.set_xlabel("Hounsfield Units")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "density_method_comparison.png", dpi=150, bbox_inches="tight")

# %%
# Build a simulated CT slice
# --------------------------
# A body with two lungs, a spine and a metal implant, in HU.

size = 128
phantom_hu = np.full((size, size), -1000.0)  # Background air

center = size // 2
y, x = np.ogrid[:size, :size]


def disc(center_yx: tuple[int, int], radius: int) -> np.ndarray:
    return (x - center_yx[1]) ** 2 + (y - center_yx[0]) ** 2 <= radius**2


phantom_hu[disc((center, center), size // 3)] = 40  # Soft tissue
phantom_hu[disc((center - size // 6, center - size // 8), size // 8)] = -800  # Lung
phantom_hu[disc((center - size // 6, center + size // 8), size // 8)] = -800
phantom_hu[disc((center + size // 4, center), size // 12)] = 800  # Bone
phantom_hu[disc((center, center + size // 6), size // 20)] = 2000  # Metal

# %%
# Convert the slice to density
# ----------------------------
# The difference panel highlights where the two models disagree, mainly in
# lung, bone and metal.

density_bilinear = hu_to_density(phantom_hu)
density_schneider = hu_to_density_schneider(phantom_hu)
difference = density_schneider - density_bilinear

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im1 = axes[0, 0].imshow(phantom_hu, cmap="gray", vmin=-1000, vmax=2000)
axes[0, 0].set_title("Simulated CT Image (HU)")
fig.colorbar(im1, ax=axes[0, 0], label="HU")

im2 = axes[0, 1].imshow(density_bilinear, cmap="viridis", vmin=0, vmax=2.5)
axes[0, 1].set_title("Bilinear Density Conversion")
fig.colorbar(im2, ax=axes[0, 1], label="Density (g/cm³)")

im3 = axes[1, 0].imshow(density_schneider, cmap="viridis", vmin=0, vmax=2.5)
axes[1, 0].set_title("Schneider Density Conversion")
fig.colorbar(im3, ax=axes[1, 0], label="Density (g/cm³)")

im4 = axes[1, 1].imshow(
    difference,
    cmap="RdBu_r",
    vmin=-np.max(np.abs(difference)),
    vmax=np.max(np.abs(difference)),
)
axes[1, 1].set_title("Density Difference\n(Schneider - Bilinear)")
fig.colorbar(im4, ax=axes[1, 1], label="Density Diff (g/cm³)")

for ax in axes.flat:
    ax.axis("off")
fig.tight_layout()
fig.savefig(
    output_dir / "ct_phantom_density_conversion.png", dpi=150, bbox_inches="tight"
)

np.savez(
    output_dir / "phantom_data.npz",
    hu_image=phantom_hu,
    density_bilinear=density_bilinear,
    density_schneider=density_schneider,
    density_difference=difference,
)
plt.show()
