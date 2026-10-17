"""
STIR adaptor OSEM
=================

SIMIND simulation followed by STIR OSEM reconstruction. This is a quick
low-history run on a tiny phantom with few projection views, so the
projection and the reconstruction are noisy.

Unit note: STIR image geometry is defined in mm and SIMIND geometry
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
import stir

from simind_python_connector import StirSimindAdaptor, configs
from simind_python_connector.builders import STIRSPECTImageDataBuilder
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE
from simind_python_connector.utils import get_array


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "stir_adaptor_osem"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build the phantom
# -----------------
# A cylindrical body with a hot sphere on a 32 x 32 x 32 grid of 4 mm voxels.
# Both maps are written as STIR Interfile images and read back as STIR
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


def read_stir_image(array: np.ndarray, base: Path) -> stir.FloatVoxelsOnCartesianGrid:
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
    return stir.FloatVoxelsOnCartesianGrid.read_from_file(str(base.with_suffix(".hv")))


source_image = read_stir_image(source, output_dir / "stir_source")
mu_image = read_stir_image(mu_map, output_dir / "stir_mu")

# %%
# Simulate
# --------
# ``StirSimindAdaptor`` takes the STIR images directly. The projections are
# requested on the reconstruction grid: 32 x 32 bins of 4 mm (Index 28 is in
# cm), with a single 75-225 keV window for Y-90 bremsstrahlung.

adaptor = StirSimindAdaptor(
    config_source=configs.get("Example.yaml"),
    output_dir=str(output_dir),
    output_prefix="stir_case01",
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
projection_path = output_dir / "stir_total.hs"
projection.write_to_file(str(projection_path))
print(f"Projection sum: {projection.sum():.3f}")

# %%
# Reconstruct with OSEM
# ---------------------
# The system model is SPECTUB with the attenuation map and the collimator
# resolution model. Its parameters, sigma at the collimator face 1.72 mm
# (sqrt(0.106^2 + 0.136^2) cm) and a slope of 0.031, come from the ma-megp
# collimator PSF that SIMIND simulates and that example 07C models with
# ``SPECTPSFTransform``. All three 07 examples run OSEM with 4 subsets and 8
# subiterations (2 full iterations).
#
# With direct STIR bindings, calling ``set_up()`` on the projector or
# objective before the reconstructor can collapse SPECTUB reconstructions to
# zero, so the chain is configured and only the reconstructor is set up.

reconstruction = source_image.get_empty_copy()
reconstruction.fill(1.0)

proj_matrix = stir.ProjMatrixByBinSPECTUB()
proj_matrix.set_attenuation_image_sptr(mu_image)
proj_matrix.set_resolution_model(1.72, 0.031, False)
proj_matrix.set_keep_all_views_in_cache(True)

projector_pair = stir.ProjectorByBinPairUsingProjMatrixByBin()
projector_pair.set_proj_matrix_sptr(proj_matrix)

objective = stir.PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat()
objective.set_projector_pair_sptr(projector_pair)
objective.set_proj_data_sptr(projection)
objective.set_num_subsets(4)

recon = stir.OSMAPOSLReconstruction3DFloat()
recon.set_objective_function(objective)
recon.set_input_data(projection)
recon.set_num_subsets(4)
recon.set_num_subiterations(8)
recon.set_disable_output(True)
recon.set_up(reconstruction)
recon.reconstruct(reconstruction)

recon_path = output_dir / "stir_osem_recon.hv"
reconstruction.write_to_file(str(recon_path))
print(f"Reconstruction sum: {reconstruction.sum():.3f}")

# %%
# Plot the result
# ---------------
# Central slice of the input source, the first projection view and the
# central slice of the OSEM reconstruction.

source_arr = get_array(source_image)
proj_arr = get_array(projection)
recon_arr = get_array(reconstruction)

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
fig.savefig(output_dir / "stir_osem_summary.png", dpi=140)
plt.show()
