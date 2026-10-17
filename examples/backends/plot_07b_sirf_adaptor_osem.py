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
# cm), with a single 75-225 keV window for Y-90 bremsstrahlung.

adaptor = SirfSimindAdaptor(
    config_source=configs.get("Example.yaml"),
    output_dir=str(output_dir),
    output_prefix="sirf_case01",
    photon_multiplier=1,
    quantization_scale=0.03 * DEFAULT_QUANTIZATION_SCALE,
)
adaptor.set_source(source_image)
adaptor.set_mu_map(mu_image)
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

projection = adaptor.get_total_output(window=1)
projection_path = output_dir / "sirf_total.hs"
projection.write(str(projection_path))
print(f"Projection sum: {projection.sum():.3f}")

# %%
# Reconstruct with OSEM
# ---------------------
# The system model is SPECTUB with the attenuation map and the collimator
# resolution model. Its parameters, a sigma of 1.72 mm at the collimator face
# (sqrt(0.106^2 + 0.136^2) cm) growing by 0.029 mm per mm of distance, are
# the best fit to the collimator-detector response that SIMIND simulates for
# ma-megp (a Gaussian sigma of about 1.57 mm + 0.0277 mm per mm). SIMIND blurs
# in 3-D, so the model blurs along the axial direction too. Example 07C models
# the same response with ``SPECTPSFTransform``. All three 07 examples run OSEM
# with 4 subsets and 20 subiterations (5 full iterations): resolution
# modelling converges more slowly, so more updates are used than without it.

reconstruction = source_image.get_uniform_copy(1.0)

acq_matrix = sirf.SPECTUBMatrix()
acq_matrix.set_attenuation_image(mu_image)
acq_matrix.set_keep_all_views_in_cache(True)
acq_matrix.set_resolution_model(1.72, 0.029, True)
acq_model = sirf.AcquisitionModelUsingMatrix(acq_matrix)

objective = sirf.make_Poisson_loglikelihood(projection)
objective.set_acquisition_model(acq_model)
objective.set_num_subsets(4)

recon = sirf.OSMAPOSLReconstructor()
recon.set_objective_function(objective)
recon.set_num_subsets(4)
recon.set_num_subiterations(20)
recon.set_input(projection)
recon.set_up(reconstruction)
recon.reconstruct(reconstruction)

recon_path = output_dir / "sirf_osem_recon.hv"
reconstruction.write(str(recon_path))
print(f"Reconstruction sum: {reconstruction.sum():.3f}")

# %%
# Plot the result
# ---------------
# Central slice of the input source, the first projection view and the
# central slice of the OSEM reconstruction.

source_arr = source_image.as_array()
proj_arr = projection.as_array()
recon_arr = reconstruction.as_array()

proj_view0 = proj_arr[0, :, 0, :] if proj_arr.ndim == 4 else proj_arr[:, 0, :]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(source_arr[source_arr.shape[0] // 2], cmap="viridis")
axes[0].set_title("Input source")
axes[1].imshow(proj_view0, cmap="magma")
axes[1].set_title("Projection (view 0)")
axes[2].imshow(recon_arr[recon_arr.shape[0] // 2], cmap="viridis")
axes[2].set_title("OSEM reconstruction")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_dir / "sirf_osem_summary.png", dpi=140)
plt.show()
