"""STIR adaptor: SIMIND simulation followed by STIR OSEM.

Build a small hot-sphere phantom as STIR images, simulate it, and reconstruct
the total projection with SPECTUB attenuation, without and with SIMIND's
scatter estimate as a fixed additive term. STIR geometry is in mm; SIMIND
geometry values are in cm and the adaptor converts them.
"""

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
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "stir_adaptor_osem"
)
output_dir.mkdir(parents=True, exist_ok=True)

dims = (32, 32, 32)
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
# STIR SPECTUB mirrors x relative to the attenuation image (UCL/STIR#452).
spectub_mu_image = read_stir_image(mu_map[..., ::-1], output_dir / "stir_spectub_mu")

adaptor = StirSimindAdaptor(
    config_source=configs.get("Example.yaml"),
    output_dir=str(output_dir),
    output_prefix="stir_case01",
    photon_multiplier=1,
    quantization_scale=0.105 * DEFAULT_QUANTIZATION_SCALE,
)
adaptor.set_source(source_image)
adaptor.set_mu_map(mu_image)
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
projection = adaptor.get_total_output(window=1)
scatter = adaptor.get_scatter_output(window=1)


# Configure the SPECTUB chain without calling set_up() on the projector or
# objective: with direct STIR bindings that can collapse the reconstruction.
def reconstruct(additive_term=None):
    reconstruction = source_image.get_empty_copy()
    reconstruction.fill(1.0)
    proj_matrix = stir.ProjMatrixByBinSPECTUB()
    proj_matrix.set_attenuation_image_sptr(spectub_mu_image)
    proj_matrix.set_resolution_model(0.0, 0.0, False)
    proj_matrix.set_keep_all_views_in_cache(True)
    projector_pair = stir.ProjectorByBinPairUsingProjMatrixByBin()
    projector_pair.set_proj_matrix_sptr(proj_matrix)
    objective = stir.PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat()
    objective.set_projector_pair_sptr(projector_pair)
    objective.set_proj_data_sptr(projection)
    if additive_term is not None:
        objective.set_additive_proj_data_sptr(additive_term)
    objective.set_num_subsets(4)
    recon = stir.OSMAPOSLReconstruction3DFloat()
    recon.set_objective_function(objective)
    recon.set_input_data(projection)
    recon.set_num_subsets(4)
    recon.set_num_subiterations(8)
    recon.set_disable_output(True)
    recon.set_up(reconstruction)
    recon.reconstruct(reconstruction)
    return reconstruction


additive = stir.ProjDataInMemory(
    projection.get_exam_info(), projection.get_proj_data_info()
)
additive.fill(np.maximum(get_array(scatter), 1e-5).astype(np.float32).flat)
recon_no_scatter = reconstruct()
recon_scatter = reconstruct(additive)

source_arr = get_array(source_image)
proj_arr = get_array(projection)
recons = {
    "No scatter term": get_array(recon_no_scatter),
    "SIMIND scatter term": get_array(recon_scatter),
}
background = body & ((x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 > 8**2)
print(f"Scatter fraction of the window: {float(scatter.sum() / projection.sum()):.1%}")
for name, recon_arr in recons.items():
    ratio = recon_arr[hot].mean() / recon_arr[background].mean()
    print(f"{name}: hot-to-background {ratio:.2f}")

proj_view0 = proj_arr[0, :, 0, :] if proj_arr.ndim == 4 else proj_arr[:, 0, :]
fig = plt.figure(figsize=(14, 7))
grid = fig.add_gridspec(2, 4)
panels = [
    ("Input source", source_arr[16], "viridis"),
    ("Projection (view 0)", proj_view0, "magma"),
    *[(f"OSEM, {name.lower()}", arr[16], "viridis") for name, arr in recons.items()],
]
for column, (title, image, cmap) in enumerate(panels):
    axis = fig.add_subplot(grid[0, column])
    axis.imshow(image, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
axis = fig.add_subplot(grid[1, :])
axis.plot(source_arr[16, 16, :] / source_arr[background].mean(), "k", label="Truth")
for name, recon_arr in recons.items():
    axis.plot(recon_arr[16, 16, :] / recon_arr[background].mean(), label=name)
axis.set_xlabel("x (voxel), profile through the hot sphere")
axis.set_ylabel("Relative to background mean")
axis.legend()
fig.tight_layout()
fig.savefig(output_dir / "stir_osem_summary.png", dpi=140)
plt.show()
