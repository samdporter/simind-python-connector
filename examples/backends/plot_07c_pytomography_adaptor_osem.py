"""PyTomography adaptor: SIMIND simulation followed by PyTomography OSEM.

Build a small hot-sphere phantom as torch tensors, simulate it, and reconstruct
the total projection with an attenuation transform, without and with SIMIND's
scatter estimate as a fixed additive term. Object tensors are ordered (x, y, z);
projections are (view, radial, axial). Only PyTomography and torch data objects
are used in the adaptor and reconstruction path.
"""

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
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output"))
    / "pytomography_adaptor_osem"
)
output_dir.mkdir(parents=True, exist_ok=True)

dims = (32, 32, 32)
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
scatter_tensor = adaptor.get_scatter_output(window=1).to(dtype=torch.float32)
if projection_tensor.ndim != 3:
    raise ValueError(
        f"Expected projection [views, bins, axial], got {projection_tensor.shape}"
    )
print(f"Projection shape (view, radial, axial): {tuple(projection_tensor.shape)}")
print(f"Projection sum: {float(projection_tensor.sum()):.3f}")

# PyTomography reads the SIMIND header to recover the object and projection
# metadata. The attenuation transform uses the (x, y, z) mu tensor directly.

# OSEM uses four subsets and eight subiterations (two full iterations). The
# second reconstruction receives the scatter estimate as a fixed additive
# term; neither reconstruction updates it. No PSF model is used, so both
# images remain blurred by the collimator.
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
# Floor the scatter estimate so every projection bin has a positive additive
# term; PyTomography expects the same (view, radial, axial) order as the data.
# Both reconstructions use the same system matrix and subset schedule.
recon_scatter = reconstruct(scatter_tensor.clamp(min=1e-5))

source_arr = source_tensor.detach().cpu().numpy()
proj_arr = projection_tensor.detach().cpu().numpy()
recons = {
    "No scatter term": recon_no_scatter.detach().cpu().numpy(),
    "SIMIND scatter term": recon_scatter.detach().cpu().numpy(),
}
background = body & ((x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 > 8**2)
scatter_fraction = float(scatter_tensor.sum() / projection_tensor.sum())
print(f"Scatter fraction of the window: {scatter_fraction:.1%}")
print(f"True hot-to-background ratio: {2.0 / 0.8:.2f}")

# Object volumes are (x, y, z); transpose the y-x plane for display. Views are
# the first projection axis, radial the second and axial the third.
for name, recon_arr in recons.items():
    ratio = recon_arr[hot].mean() / recon_arr[background].mean()
    print(f"{name}: hot-to-background {ratio:.2f}")

# The top row shows the source and first projection, then the two
# reconstructions; the bottom row profiles the hot-sphere row relative to the
# background mean. Axial index 0 is on the top row, as in 07A and 07B.
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
    axis = fig.add_subplot(grid[0, column])
    axis.imshow(image, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
axis = fig.add_subplot(grid[1, :])
axis.plot(source_arr[:, 16, 16] / source_arr[background].mean(), "k", label="Truth")
for name, recon_arr in recons.items():
    axis.plot(recon_arr[:, 16, 16] / recon_arr[background].mean(), label=name)
axis.set_xlabel("x (voxel), profile through the hot sphere")
axis.set_ylabel("Relative to background mean")
axis.legend()
fig.tight_layout()
fig.savefig(output_dir / "pytomo_osem_summary.png", dpi=140)
plt.show()
