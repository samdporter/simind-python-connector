"""Convert Hounsfield units to density with bilinear and Schneider models.

The bilinear model uses two linear segments. The Schneider2000 model assigns
tissue segments by HU range and interpolates within them. They agree in soft
tissue and diverge in lung, bone and metal; the differences below are model
differences, not an accuracy claim.
"""

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

hu_values = np.linspace(-1050, 2000, 1000)
comparison = compare_density_methods(hu_values)
print(
    "Max difference from bilinear: "
    f"{comparison['max_diff_interp']:.4f} g/cm3 (interpolated), "
    f"{comparison['max_diff_piecewise']:.4f} g/cm3 (piecewise)"
)

for hu in (-1000, -800, 0, 40, 800, 2000):
    tissue = get_schneider_tissue_info(hu)
    if tissue:
        print(f"HU {hu:4d}: {tissue['name']} ({tissue['density_g_cm3']:.3f} g/cm3)")

clinical = (hu_values >= -200) & (hu_values <= 1500)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0, 0].plot(hu_values, comparison["bilinear"], "b-", label="Bilinear")
axes[0, 0].plot(
    hu_values,
    comparison["schneider_interpolated"],
    "r--",
    label="Schneider interpolated",
)
axes[0, 0].set_title("Full range (-1050 to 2000 HU)")
axes[0, 1].plot(
    hu_values[clinical], comparison["bilinear"][clinical], "b-", label="Bilinear"
)
axes[0, 1].plot(
    hu_values[clinical],
    comparison["schneider_interpolated"][clinical],
    "r--",
    label="Schneider interpolated",
)
axes[0, 1].set_title("Clinical range (-200 to 1500 HU)")
axes[1, 0].plot(
    hu_values,
    comparison["difference_interpolated"],
    "r-",
    label="Interpolated - bilinear",
)
axes[1, 0].plot(
    hu_values, comparison["difference_piecewise"], "g-", label="Piecewise - bilinear"
)
axes[1, 0].set_title("Density difference, full range")
axes[1, 1].plot(
    hu_values[clinical],
    comparison["difference_interpolated"][clinical],
    "r-",
    label="Interpolated - bilinear",
)
axes[1, 1].plot(
    hu_values[clinical],
    comparison["difference_piecewise"][clinical],
    "g-",
    label="Piecewise - bilinear",
)
axes[1, 1].set_title("Density difference, clinical range")
for axis in axes[1]:
    axis.axhline(0.0, color="k", alpha=0.3)
    axis.set_ylabel("Density difference (g/cm3)")
for axis in axes[0]:
    axis.set_ylabel("Density (g/cm3)")
for axis in axes.flat:
    axis.set_xlabel("Hounsfield units")
    axis.legend()
    axis.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "density_method_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print(
    "Density at soft tissue, lung, bone and metal: "
    f"{hu_to_density_schneider(np.array([40.0, -800.0, 800.0, 2000.0]))}"
)
print(f"Bilinear comparison: {hu_to_density(np.array([40.0, -800.0, 800.0, 2000.0]))}")
