"""
Phantom library
===============

Simulate phantoms that need no hand-made voxel maps: SIMIND's analytic
multiple-inserts (Jaszczak-like) phantom and the NEMA/IEC body phantom from
phantomgen. Both are Tc-99m runs with 24 views, so the projections are noisy.

.. warning::
   This example takes about five minutes, nearly all of it in the two SIMIND
   runs.

The NEMA/IEC phantom needs phantomgen:
``pip install "phantomgen @ git+https://github.com/varzakis/phantomgen"``.
"""

# %%
# Setup
# -----
# Check that SIMIND is installed and choose where the outputs are written.

import math
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE
from simind_python_connector.phantoms import (
    AnalyticPhantom,
    HorizontalCylinder,
    Insert,
    MultipleInserts,
    nema_iec_phantom,
    voxelise,
)
from simind_python_connector.utils.interfile import ProjectionGeometry


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "phantom_library"
)
output_dir.mkdir(parents=True, exist_ok=True)

GRID = (48, 48, 48)
VOXEL_MM = 7.0
NUM_VIEWS = 24


def simulate(phantom, prefix):
    """Run SIMIND on the phantom: Tc-99m, 126-154 keV, 24 views of 7 mm."""
    connector = SimindPythonConnector(
        configs.get("Example.yaml"),
        output_dir,
        prefix,
        quantization_scale=0.01 * DEFAULT_QUANTIZATION_SCALE,
    )
    # The acquisition geometry is declared explicitly, not taken from either
    # phantom: 48 bins of 7 mm, a full turn and a 250 mm radius.
    connector.configure_acquisition(
        ProjectionGeometry(
            num_projections=NUM_VIEWS,
            extent_deg=360.0,
            direction="CW",
            start_angle_deg=180.0,
            radius_mm=250.0,
            radii_mm=None,
            num_bins=GRID[2],
            num_axial=GRID[0],
            bin_size_mm=VOXEL_MM,
            axial_size_mm=VOXEL_MM,
            image_duration_s=None,
        )
    )
    connector.set_energy_windows([126], [154], [0])
    connector.add_runtime_switch("CC", "ma-lehr")
    connector.add_runtime_switch("RR", 12345)
    # A small history budget keeps both runs within a few minutes while
    # leaving visible structure in the projections.
    connector.add_config_value(26, 0.1)
    connector.configure_phantom(phantom, time_per_projection_s=10.0)
    return connector.run()["tot_w1"].projection


# %%
# Analytic phantom
# ----------------
# A water-filled cylinder with six spheres of increasing size on a circle,
# described with SIMIND's own shapes. ``voxelise`` shows the source as SIMIND
# sees it. The phantom has a known shape but no absolute activity, so the
# source is in relative units.

water = HorizontalCylinder(10.0, (10.0, 10.0))
radii_cm = (0.5, 0.65, 0.85, 1.1, 1.4, 1.85)
inserts = tuple(
    Insert(
        (r, r, r),
        (0.0, 5.7 * math.cos(math.radians(a)), 5.7 * math.sin(math.radians(a))),
        8.0,
    )
    for r, a in zip(radii_cm, range(30, 360, 60))
)
jaszczak = AnalyticPhantom(MultipleInserts(water, inserts, background=1.0), water)

jaszczak_source, _ = voxelise(jaszczak, GRID, (VOXEL_MM,) * 3, supersample=4)

# %%
# NEMA/IEC body phantom
# ---------------------
# phantomgen's ``pet`` preset has 8:1 hot spheres, two cold spheres and a lung
# insert. Its activity is known in MBq per voxel, so the run is normalised by
# the known total activity (Index 25). Building it here, before either
# simulation, means a missing phantomgen stops the example immediately.

nema = nema_iec_phantom(GRID, (VOXEL_MM,) * 3, preset="pet", supersample=4)

# %%
# Simulations
# -----------
# Both runs use the Tc-99m 126-154 keV window and the ma-lehr collimator.

jaszczak_projection = simulate(jaszczak, "jaszczak")
nema_projection = simulate(nema, "nema")

# %%
# Slices and projections
# ----------------------
# The central slice of the Jaszczak source, the slice through the hottest NEMA/IEC
# voxel, and the first projection view of each.

hot_slice = nema.activity_mbq.max(axis=(1, 2)).argmax()

fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
panels = (
    ("Jaszczak source", jaszczak_source[GRID[0] // 2]),
    ("Jaszczak projection", jaszczak_projection[0]),
    ("NEMA/IEC activity", nema.activity_mbq[hot_slice]),
    ("NEMA/IEC projection", nema_projection[0]),
)
for ax, (title, image) in zip(axes.flat, panels):
    ax.imshow(image, cmap="gray_r")
    ax.set_title(title)
    ax.axis("off")
plt.show()
