"""
Residual correction with SIMIND as the accurate model
=====================================================

Reconstruct a simulated NEMA/IEC phantom acquisition with SIRF OSEM three
times: with the fast SPECTUB model only, with a SIMIND scatter estimate, and
with SIMIND as the accurate forward model in a Fu & Qi residual correction.

The fast model has attenuation but no collimator blur and no scatter. The
residual correction replaces its forward projection with SIMIND's all
interactions, scaled to the fast model, minus the fast projection of the same
image. Only the forward projection is accurate: the update stays with the fast
model, as in Fu & Qi, Med Phys 37:704, 2010. The scatter arm uses the same
SIMIND projection but removes only the scatter, all interactions minus the
geometric primary.

.. warning::
   This example takes about fifteen minutes, nearly all of it in the SIMIND
   runs: the simulation of the data, and one SIMIND projection per update for
   each of the two corrected reconstructions. Every run uses 8 MPI processes.

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
    ResidualCorrection,
    ScatterCorrection,
    SimindComponent,
    SimindProjector,
    UpdateSchedule,
    effective_objective,
    reference_normaliser,
)
from simind_python_connector.recon.osem import run_osem_with_corrections


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "residual_correction"
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
QUANTIZATION_SCALE = 0.06 * DEFAULT_QUANTIZATION_SCALE


def make_adaptor(name):
    adaptor = SirfSimindAdaptor(
        configs.get("Example.yaml"),
        str(output_dir),
        name,
        scoring_routine=ScoringRoutine.PENETRATE,
        quantization_scale=QUANTIZATION_SCALE,
    )
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_runtime_switch("CC", "ma-lehr")
    adaptor.set_mpi(8)
    adaptor.add_config_value(1, 140.0)
    return adaptor


measured_adaptor = make_adaptor("measured")
measured_adaptor.set_template(template)
measured_adaptor.set_source(activity_image)
measured_adaptor.set_mu_map(mu_image)
measured_adaptor.set_activity(TOTAL_MBQ, TIME_PER_VIEW_S)
expected = measured_adaptor.run()["all_interactions"]

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
# The fast model
# --------------
# The reconstruction model is SPECTUB with the flipped attenuation map and no
# PSF. OSEM runs with 6 subsets of 10 views each. ``run_osem_with_corrections``
# runs OSEM in passes and calls the updater between them. Every arm uses 48
# subiterations, 8 full iterations, and the first pass has a zero additive term.

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


full_model = make_acq_model()
full_model.set_up(measured, activity_image)

fast_only = run_osem_with_corrections(
    measured,
    make_acq_model,
    initial_image,
    AdditiveUpdater(None, UpdateSchedule(every=1), zero_additive),
    NUM_SUBSETS,
    SUBITERATIONS_PER_UPDATE * (NUM_UPDATES + 1),
    num_updates=0,
)

# %%
# SIMIND scatter
# --------------
# ``SimindProjector`` simulates the current image with the template geometry.
# ``reference_normaliser`` scales the geometric primary of that simulation to
# the fast model's forward projection of the same image, so that the estimate
# comes out on the scale of the data. ``ScatterCorrection`` keeps all
# interactions minus the geometric primary and smooths the result over 3 bins,
# across views as well, because a raw Monte Carlo estimate is too noisy to
# subtract. The additive term starts at zero and is refreshed after each pass of
# 12 subiterations, averaging successive estimates (``damping=0.5``). With
# ``seed=None`` the 8 MPI processes use their fixed seeds in every pass.

scatter_projector = SimindProjector(
    make_adaptor("scatter"),
    template,
    mu_image,
    normalise=reference_normaliser(full_model),
    seed=None,
)
scatter_updater = AdditiveUpdater(
    ScatterCorrection(
        scatter_projector, smoothing_fwhm_bins=3.0, smooth_across_views=True
    ),
    UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
    zero_additive,
    damping=0.5,
)
scatter_corrected = run_osem_with_corrections(
    measured,
    make_acq_model,
    initial_image,
    scatter_updater,
    NUM_SUBSETS,
    SUBITERATIONS_PER_UPDATE,
    NUM_UPDATES,
)

# %%
# Full residual
# -------------
# ``ResidualCorrection`` adds ``A_acc(x) - A_fast(x)`` to the fast model, where
# ``A_acc`` is SIMIND's all interactions, scaled to the fast model, and
# ``A_fast`` is the fast projection of the same image. Away from the phantom
# the two agree, so only the difference that the fast model misses remains.
# The same zero start and damping apply, so the term in use is an average of the
# successive estimates, not the last raw one. The effective objective below is
# the accurate model's Poisson data term at each update's image, evaluated
# before that average, so it measures the image passed to SIMIND rather than the
# damped term the next OSEM pass uses.

residual_projector = SimindProjector(
    make_adaptor("residual"),
    template,
    mu_image,
    normalise=reference_normaliser(full_model),
    seed=None,
)


class RecordedResidualCorrection(ResidualCorrection):
    """Records the effective objective after every update."""

    def __init__(self, measured, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measured = measured
        self.objectives = []

    def estimate(self, image):
        additive = super().estimate(image)
        self.objectives.append(effective_objective(self.measured, self))
        return additive


residual = RecordedResidualCorrection(
    measured,
    SimindComponent(residual_projector, "all_interactions"),
    full_model,
    zero_additive,
)
residual_updater = AdditiveUpdater(
    residual,
    UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
    zero_additive,
    damping=0.5,
)
residual_corrected = run_osem_with_corrections(
    measured,
    make_acq_model,
    initial_image,
    residual_updater,
    NUM_SUBSETS,
    SUBITERATIONS_PER_UPDATE,
    NUM_UPDATES,
)

# %%
# Update cost and effective objective
# -----------------------------------
# The sum of each additive term in use and the time of the SIMIND run that
# produced its estimate. The effective objective is the accurate model's data
# term after each residual update; a falling value means the accurate model
# fits the data better, beyond Monte Carlo noise.

for name, updater in (("scatter", scatter_updater), ("residual", residual_updater)):
    for record in updater.history:
        print(
            f"{name} update at iteration {record.iteration}: "
            f"additive sum {record.additive_sum:.4g}, {record.seconds:.0f} s"
        )
print(
    "effective objective at each residual update: "
    + ", ".join(f"{value:.6g}" for value in residual.objectives)
)

# %%
# Reconstructions
# ---------------
# The images are the reconstructions after the fourth pass, so each additive
# term is the damped average described above, not the latest raw estimate.
# Every image is shown relative to its mean in the background mask, so the
# arbitrary scale of the reconstructions does not matter.

background = nema.masks["background"] > 0


def relative_to_background(array):
    return array / array[background].mean()


truth_array = relative_to_background(nema.activity_mbq)
recon_arrays = {
    "Fast only": relative_to_background(fast_only.as_array()),
    "Fast + SIMIND scatter": relative_to_background(scatter_corrected.as_array()),
    "Fast + full residual": relative_to_background(residual_corrected.as_array()),
}

fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
for ax, (title, image) in zip(
    axes.flat, (("Truth", truth_array), *recon_arrays.items())
):
    shown = ax.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

# %%
# Contrast recovery and noise
# ---------------------------
# For each hot sphere, the mean in its mask over the mean in the background
# mask, divided by the true ratio of 8. A value of 1 is full recovery. The
# masks include voxels on the sphere edge, which hold less activity, so even
# the truth scores below 1; its score is the ceiling for the reconstructions.
# The 10 and 13 mm spheres are cold and are not scored. The noise is the
# coefficient of variation in the background mask.

SPHERES = {"sphere_3": 17, "sphere_4": 22, "sphere_5": 28, "sphere_6": 37}
TRUE_RATIO = 8.0


def contrast_recovery(image, sphere_name):
    sphere = nema.masks[sphere_name] > 0
    return image[sphere].mean() / image[background].mean() / TRUE_RATIO


scores = {"Truth": truth_array, **recon_arrays}
diameters = [str(d) for d in SPHERES.values()]
print(f"{'':24}" + "".join(f"{d:>7} mm" for d in diameters) + "    CoV")
for title, image in scores.items():
    recovery = [contrast_recovery(image, name) for name in SPHERES]
    cov = image[background].std() / image[background].mean()
    print(
        f"{title:<24}" + "".join(f"{value:10.2f}" for value in recovery) + f"{cov:7.2f}"
    )

x = np.arange(len(diameters))
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for offset, (title, image) in zip((-0.3, -0.1, 0.1, 0.3), scores.items()):
    values = [contrast_recovery(image, name) for name in SPHERES]
    ax.bar(x + offset, values, 0.2, label=title)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.legend()
plt.show()

# %%
# Scatter correction raises the recovery of every hot sphere above the fast
# model alone. The full residual correction matches it at 17 mm, but its larger
# and noisier additive term leaves the other three spheres lower, and the 22 mm
# sphere below the fast model alone. The effective objective also stops
# improving after the second update. With one noise realisation and three
# damped updates this chart does not rank the two corrections: it shows how
# much of the gap to the truth each one closes at this budget, and that an
# accurate forward model alone does not guarantee a better image than a
# well-tuned scatter estimate.
