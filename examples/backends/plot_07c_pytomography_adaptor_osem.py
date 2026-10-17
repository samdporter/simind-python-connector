"""
PyTomography adaptor OSEM
=========================

SIMIND simulation followed by PyTomography OSEM reconstruction. This is a
quick low-history run on a tiny phantom, so the projection and the
reconstruction are noisy. Only PyTomography and torch data objects are used
in the adaptor and reconstruction path.
"""

# %%
# Setup
# -----
# Check that SIMIND is installed and choose where the outputs are written.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytomography import algorithms, likelihoods
from pytomography.io.SPECT import simind as pytomo_simind
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import (
    SPECTAttenuationTransform,
    SPECTPSFTransform,
)

from simind_python_connector import PyTomographySimindAdaptor, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
    / "pytomography_adaptor_osem"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the phantom
# -----------------
# A cylindrical body with a hot sphere on a 32 x 32 x 32 grid. PyTomography
# object tensors are ordered (x, y, z).

dims = (32, 32, 32)  # x, y, z
x, y, z = np.indices(dims)
body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

source = np.zeros(dims, dtype=np.float32)
source[body] = 0.8
source[hot] = 2.0

mu_map = np.zeros_like(source)
mu_map[body] = 0.15

source_tensor = torch.from_numpy(source)
mu_tensor = torch.from_numpy(mu_map)

# %%
# Simulate
# --------
# ``PyTomographySimindAdaptor`` takes the tensors directly. The projections
# are requested on the reconstruction grid: 32 x 32 bins of 4 mm (Index 28 is
# in cm), with a single 75-225 keV window for Y-90 bremsstrahlung.

adaptor = PyTomographySimindAdaptor(
    config_source=configs.get("Example.yaml"),
    output_dir=str(output_dir),
    output_prefix="pytomo_case01",
    photon_multiplier=1,
    voxel_size_mm=4.0,
    quantization_scale=0.03 * DEFAULT_QUANTIZATION_SCALE,
)
adaptor.set_source(source_tensor)
adaptor.set_mu_map(mu_tensor)
adaptor.set_energy_windows([75], [225], [0])
adaptor.add_runtime_switch("FI", "y90_tissue")
adaptor.add_runtime_switch("CC", "ma-megp")
adaptor.add_runtime_switch("NN", 1)
adaptor.add_runtime_switch("RR", 12345)
adaptor.add_config_value(1, 140.0)
adaptor.add_config_value(19, 2)
adaptor.add_config_value(53, 0)
adaptor.add_config_value(28, 0.4)
adaptor.add_config_value(76, 32)
adaptor.add_config_value(77, 32)
adaptor.run()

projection_tensor = adaptor.get_total_output(window=1).to(dtype=torch.float32)
if projection_tensor.ndim != 3:
    raise ValueError(
        "Expected projection tensor shape [views, bins, axial], "
        f"got {tuple(projection_tensor.shape)}"
    )
print(f"Projection sum: {float(projection_tensor.sum()):.3f}")

# %%
# Reconstruct with OSEM
# ---------------------
# The system matrix combines attenuation and the collimator PSF read from the
# SIMIND header of the total window (ma-megp: sigma 1.72 mm at the collimator
# face, slope 0.031, a 3-D blur close to the one 07A and 07B pass to SPECTUB).
# OSEM runs 5 iterations of 4 subsets, the same 20 subiterations as 07A and
# 07B: resolution modelling converges more slowly, so more updates are used
# than without it.

total_h00_path = adaptor.get_output_header_path("tot_w1")
object_meta, proj_meta = pytomo_simind.get_metadata(str(total_h00_path))
psf_meta = pytomo_simind.get_psfmeta_from_header(str(total_h00_path))
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[
        SPECTAttenuationTransform(
            attenuation_map=mu_tensor.to(dtype=torch.float32).contiguous()
        ),
        SPECTPSFTransform(psf_meta=psf_meta),
    ],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)
likelihood = likelihoods.PoissonLogLikelihood(
    system_matrix=system_matrix,
    projections=projection_tensor,
)
reconstruction = algorithms.OSEM(likelihood=likelihood)(5, n_subsets=4)

np.save(
    output_dir / "pytomo_total_tensor.npy", projection_tensor.detach().cpu().numpy()
)
np.save(output_dir / "pytomo_osem_recon.npy", reconstruction.detach().cpu().numpy())
print(f"Reconstruction sum: {float(reconstruction.sum()):.3f}")

# %%
# Plot the result
# ---------------
# Central slice of the input source, the first projection view and the
# central slice of the OSEM reconstruction. Object tensors are (x, y, z) and
# projections are [view, radial, axial], so all are transposed to put y or axial
# on the rows. Axial index 0 is on the top row, as in 07A and 07B.

source_arr = source_tensor.detach().cpu().numpy()
proj_arr = projection_tensor.detach().cpu().numpy()
recon_arr = reconstruction.detach().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(source_arr[:, :, source_arr.shape[2] // 2].T, cmap="viridis")
axes[0].set_title("Input source")
axes[1].imshow(proj_arr[0].T, cmap="magma")
axes[1].set_title("Projection (view 0)")
axes[2].imshow(recon_arr[:, :, recon_arr.shape[2] // 2].T, cmap="viridis")
axes[2].set_title("OSEM reconstruction")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_dir / "pytomo_osem_summary.png", dpi=140)
plt.show()
