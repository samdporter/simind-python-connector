"""
SIRF projection from DICOM
==========================

Create a SIRF projection object from a DICOM NM projection file. The data
is a Y-90 bremsstrahlung study: a Mediso
AnyScan acquisition of 120 views over 360 degrees, 128 x 128 bins of
4.2578 mm and a 75-225 keV window. The script takes no path argument:
``fetch_y90_phantom`` reads the directory named by
``SIMIND_CONNECTOR_Y90_DATA``.

These are raw measured projection previews. All three 08 pages use the same
0-16 counts-per-bin display scale so the result is an honest input illustration,
not a reconstruction. See :doc:`../real_data/plot_16b_sirf_ct_conversion` for
the CT conversion and :doc:`../real_data/plot_17_y90_effective_energy` and
:doc:`../real_data/plot_18_y90_mc_scatter` for the reconstruction case studies.
"""

# %%
# Setup
# -----
# Locate the Y-90 dataset and choose where the outputs are written.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import sirf.STIR as sirf

from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder
from simind_python_connector.datasets import fetch_y90_phantom
from simind_python_connector.utils import get_array


PROJECTION_VMIN = 0.0
PROJECTION_VMAX = 16.0


def _shown(path):
    """Show a path relative to the working directory, or just its name."""
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path.name


dicom_file = fetch_y90_phantom() / "peak.dcm"
output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
    / "dicom_projection_objects"
    / "sirf"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the SIRF projection data
# ------------------------------
# The builder fills an Interfile header from the DICOM metadata and writes
# the projection counts next to it. The result is read back as SIRF
# ``AcquisitionData``.

builder = STIRSPECTAcquisitionDataBuilder(backend="sirf")
builder.update_header_from_dicom(str(dicom_file))

output_base = output_dir / "sirf_from_dicom"
acq = builder.build(output_path=output_base)
projection = get_array(acq)

print(f"DICOM input: {_shown(dicom_file)}")
print(f"Projection is AcquisitionData: {isinstance(acq, sirf.AcquisitionData)}")
print(f"Projection shape: {projection.shape}")
print(f"Interfile output base: {_shown(output_base)}")

# %%
# Plot two views
# --------------
# Views 0 and 30 of the 120 views, 90 degrees apart.

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, view in zip(axes, (0, 30)):
    plane = (
        projection[0, :, view, :] if projection.ndim == 4 else projection[:, view, :]
    )
    shown = ax.imshow(plane, cmap="magma", vmin=PROJECTION_VMIN, vmax=PROJECTION_VMAX)
    ax.set_title(f"SIRF projection (view {view})")
    ax.axis("off")
    fig.colorbar(shown, ax=ax, shrink=0.75, label="counts per bin")
fig.tight_layout()
fig.savefig(output_dir / "sirf_from_dicom_summary.png", dpi=140)
plt.show()
