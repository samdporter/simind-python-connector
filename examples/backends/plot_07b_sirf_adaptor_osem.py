"""
SIRF adaptor OSEM
=================

SIMIND simulation followed by SIRF OSEM reconstruction. This is a quick
low-history run on a tiny phantom with few projection views, so the
projection and the reconstruction are noisy.

Unit note: SIRF/STIR image geometry is defined in mm and SIMIND geometry
parameters are in cm. The adaptor handles the mm to cm conversion.
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
import sirf.STIR as sirf

from simind_python_connector import SirfSimindAdaptor, configs
from simind_python_connector.builders import STIRSPECTImageDataBuilder
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "sirf_adaptor_osem"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the phantom
# -----------------
# A cylindrical body with a hot sphere on a 32 x 32 x 32 grid of 4 mm voxels.
# Both maps are written as STIR Interfile images and read back as SIRF
# images.

dims = (32, 32, 32)  # z, y, x
voxel_mm = (4.0, 4.0, 4.0)

z, y, x = np.indices(dims)
body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

source = np.zeros(dims, dtype=np.float32)
source[body] = 0.8
source[hot] = 2.0

mu_map = np.zeros_like(source)
mu_map[body] = 0.15


def read_sirf_image(array: np.ndarray, base: Path) -> sirf.ImageData:
    builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(dims[2]),
            "!matrix size [2]": str(dims[1]),
            "!matrix size [3]": str(dims[0]),
            "scaling factor (mm/pixel) [1]": str(voxel_mm[2]),
            "scaling factor (mm/pixel) [2]": str(voxel_mm[1]),
            "scaling factor (mm/pixel) [3]": str(voxel_mm[0]),
        }
    )
    builder.set_pixel_array(array)
    builder.build(output_path=base)
    return sirf.ImageData(str(base.with_suffix(".hv")))


source_image = read_sirf_image(source, output_dir / "sirf_source")
mu_image = read_sirf_image(mu_map, output_dir / "sirf_mu")

# %%
# Simulate
# --------
# ``SirfSimindAdaptor`` takes the SIRF images directly. The projections are
# requested on the reconstruction grid: 32 x 32 bins of 4 mm (Index 28 is in
# cm), with a single 75-225 keV window for Y-90 bremsstrahlung. SCATTWIN
# writes the total projection and, separately, the scatter in that window
# (switch ``CA`` 2 asks for the split).

adaptor = SirfSimindAdaptor(
    config_source=configs.get("Example.yaml"),
    output_dir=str(output_dir),
    output_prefix="sirf_case01",
    photon_multiplier=1,
    quantization_scale=0.09 * DEFAULT_QUANTIZATION_SCALE,
)
adaptor.set_source(source_image)
adaptor.set_mu_map(mu_image)
adaptor.set_energy_windows([75], [225], [0])
adaptor.add_runtime_switch("FI", "y90_tissue")
adaptor.add_runtime_switch("CA", 2)
adaptor.add_runtime_switch("CC", "ma-megp")
adaptor.add_runtime_switch("NN", 1)
adaptor.add_runtime_switch("RR", 12345)
adaptor.add_config_value(1, 140.0)
adaptor.add_config_value(19, 2)
adaptor.add_config_value(53, 0)
adaptor.add_config_value(28, 0.4)
adaptor.add_config_value(76, 32)
adaptor.add_config_value(77, 32)
outputs = adaptor.run()

projection = outputs["tot_w1"]
scatter = outputs["sca_w1"]
projection.write(str(output_dir / "sirf_total.hs"))
print(f"Projection sum: {projection.sum():.3f}")
print(f"Scatter sum: {scatter.sum():.3f}")

# %%
# Reconstruct with OSEM, without and with scatter
# -----------------------------------------------
# The system model is SPECTUB with the attenuation map. PSF modelling is left
# out to keep the example quick, so the reconstruction keeps the collimator
# blur that SIMIND simulated, and the three 07 examples agree. Modelling the
# PSF (``set_resolution_model(1.72, 0.029, True)`` for this collimator, or
# PyTomography's ``SPECTPSFTransform``) needs about 20 updates and is not used
# here. All three 07 examples run OSEM with 4 subsets and 8 subiterations
# (2 full iterations).
#
# The total projection is reconstructed twice. The second reconstruction has
# SIMIND's own scatter estimate for this run as a fixed additive term, floored
# at a small positive value. It is not updated during the reconstruction; the
# MC-scatter-in-the-loop examples refresh it.


def reconstruct(additive_term=None):
    reconstruction = source_image.get_uniform_copy(1.0)

    acq_matrix = sirf.SPECTUBMatrix()
    acq_matrix.set_attenuation_image(mu_image)
    acq_matrix.set_keep_all_views_in_cache(True)
    acq_matrix.set_resolution_model(0.0, 0.0, False)
    acq_model = sirf.AcquisitionModelUsingMatrix(acq_matrix)
    if additive_term is not None:
        acq_model.set_additive_term(additive_term)

    objective = sirf.make_Poisson_loglikelihood(projection)
    objective.set_acquisition_model(acq_model)
    objective.set_num_subsets(4)

    recon = sirf.OSMAPOSLReconstructor()
    recon.set_objective_function(objective)
    recon.set_num_subsets(4)
    recon.set_num_subiterations(8)
    recon.set_input(projection)
    recon.set_up(reconstruction)
    recon.reconstruct(reconstruction)
    return reconstruction


recon_no_scatter = reconstruct()
recon_scatter = reconstruct(scatter.maximum(1e-5))

recon_no_scatter.write(str(output_dir / "sirf_osem_recon.hv"))
recon_scatter.write(str(output_dir / "sirf_osem_recon_scatter.hv"))

# %%
# Compare with the truth
# ----------------------
# Hot-to-background ratio of the mean reconstructed values in the hot sphere
# and in the body away from it, against the true ratio.

source_arr = source_image.as_array()
proj_arr = projection.as_array()
recons = {
    "No scatter term": recon_no_scatter.as_array(),
    "SIMIND scatter term": recon_scatter.as_array(),
}

background = body & ((x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 > 8**2)
print(f"True hot-to-background ratio: {2.0 / 0.8:.2f}")
for name, recon_arr in recons.items():
    ratio = recon_arr[hot].mean() / recon_arr[background].mean()
    print(f"{name}: hot-to-background ratio {ratio:.2f}")

proj_view0 = proj_arr[0, :, 0, :] if proj_arr.ndim == 4 else proj_arr[:, 0, :]

fig = plt.figure(figsize=(14, 7))
grid = fig.add_gridspec(2, 4)
panels = [
    ("Input source", source_arr[16], "viridis"),
    ("Projection (view 0)", proj_view0, "magma"),
    *[(f"OSEM, {name.lower()}", arr[16], "viridis") for name, arr in recons.items()],
]
for column, (title, image, cmap) in enumerate(panels):
    ax = fig.add_subplot(grid[0, column])
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

ax = fig.add_subplot(grid[1, :])
ax.plot(source_arr[16, 16, :] / source_arr[background].mean(), "k", label="Truth")
for name, recon_arr in recons.items():
    ax.plot(recon_arr[16, 16, :] / recon_arr[background].mean(), label=name)
ax.set_xlabel("x (voxel), profile through the hot sphere")
ax.set_ylabel("Relative to background mean")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "sirf_osem_summary.png", dpi=140)
plt.show()
