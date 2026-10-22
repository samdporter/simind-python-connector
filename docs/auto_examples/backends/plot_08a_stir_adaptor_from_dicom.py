"""
STIR projection from DICOM
==========================

Create a STIR projection object from a DICOM NM projection file. The data
is a Y-90 bremsstrahlung study: a Mediso
AnyScan acquisition of 120 views over 360 degrees, 128 x 128 bins of
4.2578 mm and a 75-225 keV window. The script takes no path argument:
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
import stir

from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder
from simind_python_connector.datasets import fetch_y90_phantom
from simind_python_connector.utils import get_array


dicom_file = fetch_y90_phantom() / "peak.dcm"
output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
    / "dicom_projection_objects"
    / "stir"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the STIR projection data
# ------------------------------
# The builder fills an Interfile header from the DICOM metadata and writes
# the projection counts next to it. The result is read back as STIR
# ``ProjData``.

builder = STIRSPECTAcquisitionDataBuilder(backend="stir")
builder.update_header_from_dicom(str(dicom_file))

output_base = output_dir / "stir_from_dicom"
acq = builder.build(output_path=output_base)
projection = get_array(acq)

print(f"DICOM input: {dicom_file}")
print(f"Projection is ProjData: {isinstance(acq, stir.ProjData)}")
print(f"Projection shape: {projection.shape}")
print(f"Interfile output base: {output_base}")

# %%
# Plot two views
# --------------
# Views 0 and 30 of the 120 views, 90 degrees apart.

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, view in zip(axes, (0, 30)):
    plane = (
        projection[0, :, view, :] if projection.ndim == 4 else projection[:, view, :]
    )
    ax.imshow(plane, cmap="magma")
    ax.set_title(f"STIR projection (view {view})")
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_dir / "stir_from_dicom_summary.png", dpi=140)
plt.show()
