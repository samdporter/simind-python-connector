"""
Phantom library
===============

Simulate phantoms that need no hand-made voxel maps: SIMIND's analytic
multiple-inserts (Jaszczak-like) phantom and the NEMA/IEC body phantom from
phantomgen. Both are Tc-99m runs with 24 views.

The analytic phantom gets one million photon histories per projection per unit
of Index 26, so Index 26 is raised well above the old 0.1 smoke-test value.
The voxelized NEMA phantom gets its histories from the source map instead: the
default quantization gives the hottest voxel 500 histories and ``NN``
multiplies them. Both projections use enough histories to show the inserts as
integrated structures rather than isolated counts.

.. warning::
   The two runs take roughly three to four hours on a single emulated x86
   core. Nearly all of that is SIMIND; the phantom construction is seconds.

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
import time
from pathlib import Path

import matplotlib.pyplot as plt

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)
from simind_python_connector.phantoms import (
    AnalyticPhantom,
    HorizontalCylinder,
    Insert,
    MultipleInserts,
    VoxelPhantom,
    nema_iec_phantom,
    voxelise,
)
from simind_python_connector.utils.interfile import ProjectionGeometry
from simind_python_connector.utils.projection_metrics import (
    projection_view0,
    source_weight_sum,
    zero_history_fraction,
)


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
ANALYTIC_INDEX26 = 4.0
VOXEL_NN = 1


def simulate(phantom, prefix):
    """Run SIMIND on the phantom: Tc-99m, 126-154 keV, 24 views of 7 mm."""
    connector = SimindPythonConnector(
        configs.get("Example.yaml"),
        output_dir,
        prefix,
        quantization_scale=DEFAULT_QUANTIZATION_SCALE,
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
    # PENETRATE scores all interactions; it takes its energy window from
    # Indexes 20 (upper) and 21 (lower).
    connector.add_config_value(84, ScoringRoutine.PENETRATE.value)
    connector.add_config_value(20, 154.0)
    connector.add_config_value(21, 126.0)
    connector.add_runtime_switch("CC", "ma-lehr")
    connector.add_runtime_switch("RR", 12345)
    if isinstance(phantom, VoxelPhantom):
        connector.add_runtime_switch("NN", VOXEL_NN)
        histories = (
            source_weight_sum(phantom.activity_mbq, DEFAULT_QUANTIZATION_SCALE)
            * VOXEL_NN
        )
        print(f"{prefix}: {histories} histories per view")
    else:
        connector.add_config_value(26, ANALYTIC_INDEX26)
        print(f"{prefix}: {ANALYTIC_INDEX26 * 1e6:.0f} histories per view")
    connector.configure_phantom(phantom, time_per_projection_s=10.0)
    started = time.perf_counter()
    projection = connector.run()["all_interactions"].projection
    elapsed_s = time.perf_counter() - started
    print(
        f"{prefix}: projection sum {float(projection.sum()):.6g} in {elapsed_s:.0f} s"
    )
    return projection


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

print(
    "NEMA quantization loss: "
    f"{zero_history_fraction(nema.activity_mbq, DEFAULT_QUANTIZATION_SCALE):.4%}"
)
jaszczak_projection = simulate(jaszczak, "jaszczak")
nema_projection = simulate(nema, "nema")

# %%
# Slices and projections
# ----------------------
# The central slice of the Jaszczak source, the slice through the hottest NEMA/IEC
# voxel, and the first projection view of each. The projections use independent
# colour scales because the two phantoms have different activity scales.

hot_slice = nema.activity_mbq.max(axis=(1, 2)).argmax()

fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
panels = (
    ("Jaszczak source", jaszczak_source[GRID[0] // 2]),
    ("Jaszczak projection (view 0)", projection_view0(jaszczak_projection)),
    ("NEMA/IEC activity", nema.activity_mbq[hot_slice]),
    ("NEMA/IEC projection (view 0)", projection_view0(nema_projection)),
)
for ax, (title, image) in zip(axes.flat, panels):
    artist = ax.imshow(image, cmap="gray_r")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(artist, ax=ax, shrink=0.7)
plt.show()
