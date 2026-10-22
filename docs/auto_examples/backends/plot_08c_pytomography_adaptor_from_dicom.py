"""
PyTomography projection from DICOM
==================================

Create a PyTomography projection tensor from a DICOM NM projection file,
following PyTomography's DICOM IO API. The data is a Y-90 bremsstrahlung
study: a Mediso AnyScan acquisition of 120 views over 360 degrees, 128 x 128
bins of 4.2578 mm and a 75-225 keV window. The script takes no path argument:
``fetch_y90_phantom`` reads the directory named by
``SIMIND_CONNECTOR_Y90_DATA``.
"""

# %%
# Setup
# -----
# Locate the Y-90 dataset and choose where the outputs are written.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pytomography.io.SPECT import dicom as pytomo_dicom

from simind_python_connector.datasets import fetch_y90_phantom


dicom_file = fetch_y90_phantom() / "peak.dcm"
output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
    / "dicom_projection_objects"
    / "pytomography"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load the projections
# --------------------
# ``get_projections`` returns a tensor ordered (theta, r, z). ``index_peak=0``
# selects the first energy window.

projections = pytomo_dicom.get_projections(str(dicom_file), index_peak=0)
projection = projections.detach().cpu().numpy().astype(np.float32, copy=False)
np.save(output_dir / "pytomography_projection.npy", projection)

print(f"DICOM input: {dicom_file}")
print(f"Projection shape (theta, r, z): {projection.shape}")

# %%
# Plot two views
# --------------
# DICOM frames 0 and 30 of the 120 views, 90 degrees apart, as in 08A and 08B.
# PyTomography orders the views by its own detector angle, (gantry angle + 180)
# mod 360, so its view k is DICOM frame (k + 60) mod 120 and frame 0 is view 60.
# The tensor is transposed and drawn with ``origin="lower"`` so rows are axial
# with the head end up, and columns are radial.

n_views = projection.shape[0]
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, frame in zip(axes, (0, 30)):
    view = (frame - n_views // 2) % n_views
    ax.imshow(projection[view].T, cmap="magma", origin="lower")
    ax.set_title(f"PyTomography (DICOM frame {frame})")
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_dir / "pytomography_from_dicom_summary.png", dpi=140)
plt.show()
