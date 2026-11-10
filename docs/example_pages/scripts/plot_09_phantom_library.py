"""Build two phantom-library phantoms without running SIMIND.

The analytic multiple-inserts phantom is described with SIMIND shapes and
voxelised to source weights. The NEMA/IEC body phantom comes from phantomgen
and carries activity in MBq per voxel and a water-equivalent density. Both are
shown as central slices; neither is simulated here.
"""

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt

from simind_python_connector.phantoms import (
    AnalyticPhantom,
    HorizontalCylinder,
    Insert,
    MultipleInserts,
    nema_iec_phantom,
    voxelise,
)


output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "phantom_library"
)
output_dir.mkdir(parents=True, exist_ok=True)

GRID = (48, 48, 48)
VOXEL_MM = (7.0, 7.0, 7.0)

water = HorizontalCylinder(10.0, (10.0, 10.0))
radii_cm = (0.5, 0.65, 0.85, 1.1, 1.4, 1.85)
inserts = tuple(
    Insert(
        (radius, radius, radius),
        (0.0, 5.7 * math.cos(math.radians(angle)), 5.7 * math.sin(math.radians(angle))),
        8.0,
    )
    for radius, angle in zip(radii_cm, range(30, 360, 60))
)
jaszczak = AnalyticPhantom(MultipleInserts(water, inserts, background=1.0), water)
jaszczak_source, jaszczak_density = voxelise(jaszczak, GRID, VOXEL_MM, supersample=4)

# The analytic source contains relative weights, not calibrated activity. The
# attenuator is returned as a fraction per voxel; its voxel size is the grid
# voxel size in mm.
print(f"Jaszczak source shape: {jaszczak_source.shape}")
print(f"Jaszczak source range: {jaszczak_source.min():.3f}-{jaszczak_source.max():.3f}")
print(f"Jaszczak source sum: {float(jaszczak_source.sum()):.3f}")
print(f"Jaszczak attenuator is present: {jaszczak_density is not None}")
print(f"Jaszczak voxel size (mm): {VOXEL_MM}")

# The NEMA/IEC phantom carries MBq per voxel and water-equivalent density. The
# masks name the hot spheres, cold spheres and lung insert.
nema = nema_iec_phantom(GRID, VOXEL_MM, preset="pet", supersample=4)
hot_slice = nema.activity_mbq.max(axis=(1, 2)).argmax()

print(f"NEMA activity shape: {nema.activity_mbq.shape}")
print(
    "NEMA activity range: "
    f"{nema.activity_mbq.min():.3f}-{nema.activity_mbq.max():.3f} MBq"
)
print(f"NEMA activity total: {float(nema.activity_mbq.sum()):.3f} MBq")
print(f"NEMA hottest activity: {float(nema.activity_mbq.max()):.3f} MBq")
print(
    "NEMA density range: "
    f"{nema.density_g_cm3.min():.3f}-{nema.density_g_cm3.max():.3f} g/cm3"
)
print(f"NEMA voxel size (mm): {nema.voxel_size_mm}")
print(f"NEMA density at the hottest voxel: {float(nema.density_g_cm3.max()):.3f} g/cm3")
print(f"NEMA masks: {', '.join(sorted(nema.masks))}")
print(f"Jaszczak density shape: {getattr(jaszczak_density, 'shape', None)}")
print(
    f"Jaszczak density range: {jaszczak_density.min():.3f}-{jaszczak_density.max():.3f}"
)
print(f"Jaszczak density is water-equivalent: {jaszczak_density.max() <= 1.0}")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
axes[0].imshow(jaszczak_source[GRID[0] // 2], cmap="gray_r")
axes[0].set_title("Jaszczak source, central slice")
axes[1].imshow(nema.activity_mbq[hot_slice], cmap="gray_r")
axes[1].set_title("NEMA/IEC activity, hottest slice")
for axis in axes:
    axis.axis("off")
fig.savefig(output_dir / "phantom_library_summary.png", dpi=140)
plt.show()
