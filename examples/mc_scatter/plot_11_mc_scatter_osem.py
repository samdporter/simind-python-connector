"""
Monte Carlo scatter correction in OSEM
======================================

Reconstruct a simulated NEMA/IEC phantom acquisition twice with SIRF OSEM: once
without scatter correction, and once with a SIMIND scatter estimate that is
refreshed from the current image between OSEM passes.

The data are simulated by SIMIND and reconstructed with a fast SPECTUB model
that has no scatter. SIMIND provides the scatter: all interactions minus the
geometric primary, scaled to the fast model and added to its forward
projection.

.. warning::
   This example takes about nine minutes, nearly all of it in the SIMIND runs:
   the simulation of the data and one scatter estimate per update. Every run
   uses 8 MPI processes.

The phantom needs phantomgen:
``pip install "phantomgen @ git+https://github.com/varzakis/phantomgen"``.
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
from simind_python_connector.builders import (
    STIRSPECTAcquisitionDataBuilder,
    STIRSPECTImageDataBuilder,
)
from simind_python_connector.converters.attenuation import get_attenuation_coefficient
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)
from simind_python_connector.phantoms import nema_iec_phantom
from simind_python_connector.recon import (
    AdditiveUpdater,
    ScatterCorrection,
    SimindProjector,
    UpdateSchedule,
    reference_normaliser,
)
from simind_python_connector.recon.osem import run_osem_with_corrections


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "mc_scatter_osem"
)
output_dir.mkdir(parents=True, exist_ok=True)

GRID = (48, 48, 48)
VOXEL_MM = 7.0

# %%
# NEMA/IEC phantom
# ----------------
# phantomgen's ``pet`` preset has an 8:1 sphere-to-background activity ratio.
# Its two smallest spheres, 10 and 13 mm, are cold; the four hot spheres are
# 17, 22, 28 and 37 mm. The activity is in MBq per voxel. The attenuation map is
# the density times the attenuation coefficient of water at 140 keV.

nema = nema_iec_phantom(GRID, (VOXEL_MM,) * 3, preset="pet", supersample=4)
mu = nema.density_g_cm3 * get_attenuation_coefficient("water", 140.0)
slice_z = round(np.argwhere(nema.masks["sphere_6"]).mean(axis=0)[0])

fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
for ax, (title, image) in zip(
    axes,
    (
        ("Activity (MBq per voxel)", nema.activity_mbq),
        ("Density (g/cm³)", nema.density_g_cm3),
    ),
):
    ax.imshow(image[slice_z], cmap="gray_r")
    ax.set_title(title)
    ax.axis("off")
plt.show()


def to_sirf_image(array):
    builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(GRID[2]),
            "!matrix size [2]": str(GRID[1]),
            "!matrix size [3]": str(GRID[0]),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [3]": str(VOXEL_MM),
        },
        backend="sirf",
    )
    builder.set_pixel_array(array.astype(np.float32))
    return builder.build()


activity_image = to_sirf_image(nema.activity_mbq)
mu_image = to_sirf_image(mu)
# STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452)
spectub_mu_image = to_sirf_image(mu[..., ::-1])

# %%
# Template
# --------
# The template sets the projection geometry: 48 x 48 bins of 7 mm, 60 views
# over 360 degrees and a detector radius of 250 mm. Both the simulation and the
# reconstruction take their geometry from it.

sirf.AcquisitionData.set_storage_scheme("memory")
template = STIRSPECTAcquisitionDataBuilder(
    header_overrides={
        "!matrix size [1]": "48",
        "!matrix size [2]": "48",
        "!number of projections": "60",
        "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
        "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
        "!direction of rotation": "CW",
        "start angle": "180",
        "Radius": "250",
    },
    backend="sirf",
).build(output_path=output_dir / "template")

# %%
# "Measured" data
# ---------------
# SIMIND simulates the phantom for Tc-99m with a 126-154 keV window and the
# ma-lehr collimator. The total activity and the time per view are given, so the
# projections come out in counts (Index 25). The 8 MPI processes each simulate
# all views and SIMIND sums them; without an ``RR`` switch their seeds are fixed.
# Poisson noise from a seeded generator is added to the simulated counts.

TOTAL_MBQ = float(nema.activity_mbq.sum())
TIME_PER_VIEW_S = 2.0

adaptor = SirfSimindAdaptor(
    configs.get("Example.yaml"),
    str(output_dir),
    "measured",
    scoring_routine=ScoringRoutine.PENETRATE,
    quantization_scale=0.06 * DEFAULT_QUANTIZATION_SCALE,
)
adaptor.set_template(template)
adaptor.set_source(activity_image)
adaptor.set_mu_map(mu_image)
# PENETRATE takes its energy window from Indexes 20 (upper) and 21 (lower).
adaptor.add_config_value(20, 154.0)
adaptor.add_config_value(21, 126.0)
adaptor.add_runtime_switch("CC", "ma-lehr")
adaptor.set_mpi(8)
adaptor.add_config_value(1, 140.0)
adaptor.set_activity(TOTAL_MBQ, TIME_PER_VIEW_S)
outputs = adaptor.run()
expected = outputs["all_interactions"]

rng = np.random.default_rng(12345)
measured = expected.clone()
measured.fill(rng.poisson(np.maximum(expected.as_array(), 0)).astype(np.float32))
print(f"Measured counts: {measured.sum():.4g}")

fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
counts = measured.as_array()[0]  # axial, view, bin
axes[0].imshow(counts[:, 0, :], cmap="magma")
axes[0].set_title("Projection, view 0")
axes[1].imshow(counts[counts.shape[0] // 2], cmap="magma")
axes[1].set_title("Sinogram, central row")
axes[1].set_xlabel("bin")
axes[1].set_ylabel("view")
plt.show()

# %%
# OSEM without scatter correction
# -------------------------------
# The fast model is SPECTUB with the flipped attenuation map and no PSF. It has
# no collimator blur and no scatter. OSEM runs with 6 subsets of 10 views each.
# ``run_osem_with_corrections`` runs OSEM in passes and calls the updater between
# them; with a zero additive term and no updates it is plain OSEM. Both
# reconstructions use 48 subiterations, 8 full iterations.

NUM_SUBSETS = 6
SUBITERATIONS_PER_UPDATE = 12
NUM_UPDATES = 3
initial_image = activity_image.get_uniform_copy(1.0)
zero_additive = measured.get_uniform_copy(0)


def make_acq_model():
    acq_matrix = sirf.SPECTUBMatrix()
    acq_matrix.set_attenuation_image(spectub_mu_image)
    acq_matrix.set_resolution_model(0.0, 0.0, False)
    return sirf.AcquisitionModelUsingMatrix(acq_matrix)


uncorrected = run_osem_with_corrections(
    measured,
    make_acq_model,
    initial_image,
    AdditiveUpdater(None, UpdateSchedule(every=1), zero_additive),
    NUM_SUBSETS,
    SUBITERATIONS_PER_UPDATE * (NUM_UPDATES + 1),
    num_updates=0,
)

# %%
# OSEM with SIMIND scatter
# ------------------------
# ``SimindProjector`` simulates the current image with the template geometry.
# ``reference_normaliser`` scales the geometric primary of that simulation to
# the fast model's forward projection of the same image, so that the scatter
# comes out on the scale of the data. ``ScatterCorrection`` smooths the
# estimate over 3 bins, across views as well, because a raw Monte Carlo estimate
# is too noisy to subtract. The additive term starts at zero and is refreshed
# after each pass of 12 subiterations, averaging successive estimates
# (``damping=0.5``). The image at the first update is not corrected for
# scatter, so the first estimate is too high.
#
# ``seed=None`` leaves the ``RR`` switch unset, so the 8 MPI processes use
# their fixed seeds.

full_model = make_acq_model()
full_model.set_up(measured, activity_image)

estimate_adaptor = SirfSimindAdaptor(
    configs.get("Example.yaml"),
    str(output_dir),
    "estimate",
    scoring_routine=ScoringRoutine.PENETRATE,
    quantization_scale=0.06 * DEFAULT_QUANTIZATION_SCALE,
)
estimate_adaptor.add_config_value(20, 154.0)
estimate_adaptor.add_config_value(21, 126.0)
estimate_adaptor.add_runtime_switch("CC", "ma-lehr")
estimate_adaptor.set_mpi(8)
estimate_adaptor.add_config_value(1, 140.0)
projector = SimindProjector(
    estimate_adaptor,
    template,
    mu_image,
    normalise=reference_normaliser(full_model, method="sum"),
    seed=None,
)
updater = AdditiveUpdater(
    ScatterCorrection(projector, smoothing_fwhm_bins=3.0, smooth_across_views=True),
    UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
    zero_additive,
    damping=0.5,
)
corrected = run_osem_with_corrections(
    measured,
    make_acq_model,
    initial_image,
    updater,
    NUM_SUBSETS,
    SUBITERATIONS_PER_UPDATE,
    NUM_UPDATES,
)

# %%
# Scatter updates
# ---------------
# For each update, the sum of the additive term and the time of the SIMIND run.
# The data simulation also gives the true scatter: all interactions minus the
# geometric primary, the same quantity that is estimated.

for record in updater.history:
    print(
        f"update at iteration {record.iteration}: "
        f"scatter sum {record.additive_sum:.4g}, {record.seconds:.0f} s"
    )
true_scatter = expected - outputs["geom_coll_primary"]
print(f"True scatter sum: {true_scatter.sum():.4g}")
true_fraction = true_scatter.sum() / measured.sum()
estimated_fraction = updater.current.sum() / measured.sum()
print(
    f"Scatter fraction of the measured counts: true {true_fraction:.0%}, "
    f"estimated {estimated_fraction:.0%}"
)

central_view = measured.as_array()[0].shape[1] // 2
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, (title, data) in zip(
    axes,
    (
        ("Measured", measured),
        ("True scatter", true_scatter),
        ("Estimated scatter", updater.current),
    ),
):
    ax.imshow(data.as_array()[0][:, central_view, :], cmap="magma")
    ax.set_title(f"{title}, view {central_view}")
    ax.axis("off")
plt.show()

# %%
# Reconstructions
# ---------------
# Both images are shown relative to their mean in the background mask, so the
# arbitrary scale of the reconstructions does not matter.

background = nema.masks["background"] > 0


def relative_to_background(array):
    return array / array[background].mean()


truth_array = relative_to_background(nema.activity_mbq)
recon_arrays = {
    "No scatter correction": relative_to_background(uncorrected.as_array()),
    "SIMIND scatter": relative_to_background(corrected.as_array()),
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, (title, image) in zip(axes, (("Truth", truth_array), *recon_arrays.items())):
    shown = ax.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

# %%
# Contrast recovery
# -----------------
# For each hot sphere, the mean in its mask over the mean in the background
# mask, divided by the true ratio of 8. A value of 1 is full recovery. The
# masks include voxels on the sphere edge, which hold less activity, so even
# the truth scores below 1; its score is the ceiling for the reconstruction.
# The 10 and 13 mm spheres are cold and are not scored.

SPHERES = {"sphere_3": 17, "sphere_4": 22, "sphere_5": 28, "sphere_6": 37}
TRUE_RATIO = 8.0


def contrast_recovery(image, sphere_name):
    sphere = nema.masks[sphere_name] > 0
    return image[sphere].mean() / image[background].mean() / TRUE_RATIO


diameters = [str(d) for d in SPHERES.values()]
scores = {"Truth": [contrast_recovery(truth_array, name) for name in SPHERES]}
for title, image in recon_arrays.items():
    scores[title] = [contrast_recovery(image, name) for name in SPHERES]
for title, values in scores.items():
    print(
        f"{title}: " + ", ".join(f"{d} mm {v:.2f}" for d, v in zip(diameters, values))
    )

x = np.arange(len(diameters))
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for offset, (title, values) in zip((-0.27, 0, 0.27), scores.items()):
    ax.bar(x + offset, values, 0.27, label=title)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.legend()
plt.show()

# %%
# Scatter correction raises the recovery at 28 and 37 mm and leaves the 17 and
# 22 mm values essentially unchanged; all four stay below the truth's own score.
