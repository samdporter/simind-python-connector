"""
PyTomography adaptor OSEM
=========================

SIMIND simulation followed by PyTomography OSEM reconstruction. This is a
quick low-history Tc-99m run on a tiny phantom, so the projection and the
reconstruction are mildly noisy. Only PyTomography and torch data objects are used
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
from pytomography.transforms.SPECT import SPECTAttenuationTransform

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
# in cm), for Tc-99m (140 keV) with a single 126-154 keV window and the
# ma-lehr collimator. SCATTWIN writes the total projection and, separately,
# the scatter in that window (switch ``CA`` 2 asks for the split).

adaptor = PyTomographySimindAdaptor(
    config_source=configs.get("Example.yaml"),
    output_dir=str(output_dir),
    output_prefix="pytomo_case01",
    photon_multiplier=1,
    voxel_size_mm=4.0,
    quantization_scale=0.105 * DEFAULT_QUANTIZATION_SCALE,
)
adaptor.set_source(source_tensor)
adaptor.set_mu_map(mu_tensor)
adaptor.set_energy_windows([126], [154], [0])
adaptor.add_runtime_switch("FI", "tc99m")
adaptor.add_runtime_switch("CA", 2)
adaptor.add_runtime_switch("CC", "ma-lehr")
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
scatter_tensor = adaptor.get_scatter_output(window=1).to(dtype=torch.float32)
print(f"Projection sum: {float(projection_tensor.sum()):.3f}")

# %%
# Reconstruct with OSEM, without and with scatter
# -----------------------------------------------
# The system matrix is the attenuation map alone. PSF modelling is left out
# to keep the example quick, so the reconstruction does not model the
# collimator blur and the images stay blurred. Modelling the PSF
# (``SPECTPSFTransform``, or SPECTUB ``set_resolution_model`` with the
# collimator's sigma) needs about 20 updates and is not used here. All three
# 07 examples run OSEM with 4 subsets and 8 subiterations (2 full
# iterations), so the three backends use the same settings.
#
# The total projection is reconstructed twice. The second reconstruction has
# SIMIND's own scatter estimate for this run as a fixed additive term, floored
# at a small positive value. It is not updated during the reconstruction; the
# MC-scatter-in-the-loop examples refresh it.

total_h00_path = adaptor.get_output_header_path("tot_w1")
object_meta, proj_meta = pytomo_simind.get_metadata(str(total_h00_path))
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[
        SPECTAttenuationTransform(
            attenuation_map=mu_tensor.to(dtype=torch.float32).contiguous()
        ),
    ],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)


def reconstruct(additive_term=None):
    likelihood = likelihoods.PoissonLogLikelihood(
        system_matrix=system_matrix,
        projections=projection_tensor,
        additive_term=additive_term,
    )
    return algorithms.OSEM(likelihood=likelihood)(2, n_subsets=4)


recon_no_scatter = reconstruct()
recon_scatter = reconstruct(scatter_tensor.clamp(min=1e-5))

np.save(
    output_dir / "pytomo_total_tensor.npy", projection_tensor.detach().cpu().numpy()
)
np.save(output_dir / "pytomo_osem_recon.npy", recon_no_scatter.detach().cpu().numpy())
np.save(
    output_dir / "pytomo_osem_recon_scatter.npy", recon_scatter.detach().cpu().numpy()
)

# %%
# Compare with the truth
# ----------------------
# Hot-to-background ratio of the mean reconstructed values in the hot sphere
# and in the body away from it, against the true ratio. In this small phantom
# scatter is a small share of the counts in the window (the printed fraction),
# so a fixed additive term lowers the absolute level by about that share but
# barely changes the ratio: SIMIND's scatter is smooth and follows the body.
# Object tensors are (x, y, z) and projections are [view, radial, axial], so
# images are transposed to put y or axial on the rows.
# Axial index 0 is on the top row, as in 07A and 07B.

source_arr = source_tensor.detach().cpu().numpy()
proj_arr = projection_tensor.detach().cpu().numpy()
recons = {
    "No scatter term": recon_no_scatter.detach().cpu().numpy(),
    "SIMIND scatter term": recon_scatter.detach().cpu().numpy(),
}

scatter_fraction = float(scatter_tensor.sum() / projection_tensor.sum())
background = body & ((x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 > 8**2)
print(f"Scatter fraction of the window: {scatter_fraction:.1%}")
print(f"True hot-to-background ratio: {2.0 / 0.8:.2f}")
for name, recon_arr in recons.items():
    ratio = recon_arr[hot].mean() / recon_arr[background].mean()
    print(f"{name}: hot-to-background ratio {ratio:.2f}")

fig = plt.figure(figsize=(14, 7))
grid = fig.add_gridspec(2, 4)
panels = [
    ("Input source", source_arr[:, :, 16].T, "viridis"),
    ("Projection (view 0)", proj_arr[0].T, "magma"),
    *[
        (f"OSEM, {name.lower()}", arr[:, :, 16].T, "viridis")
        for name, arr in recons.items()
    ],
]
for column, (title, image, cmap) in enumerate(panels):
    ax = fig.add_subplot(grid[0, column])
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

ax = fig.add_subplot(grid[1, :])
ax.plot(source_arr[:, 16, 16] / source_arr[background].mean(), "k", label="Truth")
for name, recon_arr in recons.items():
    ax.plot(recon_arr[:, 16, 16] / recon_arr[background].mean(), label=name)
ax.set_xlabel("x (voxel), profile through the hot sphere")
ax.set_ylabel("Relative to background mean")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "pytomo_osem_summary.png", dpi=140)
plt.show()
