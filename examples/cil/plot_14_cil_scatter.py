"""
Monte Carlo scatter correction with CIL
=======================================

Reconstruct a simulated NEMA/IEC phantom acquisition twice with CIL's ISTA:
once without scatter correction, and once with a SIMIND scatter estimate that
is refreshed from the current image during the run.

The data are simulated by SIMIND and reconstructed with a fast SPECTUB model
that has no scatter. SIMIND provides the scatter: all interactions minus the
geometric primary, scaled to the fast model and added to its forward
projection. The subset objectives are CIL Kullback-Leibler functions, one per
group of views, and a ``CorrectionCallback`` swaps in a new scatter estimate
every 12 iterations.

.. warning::
   This example takes about twelve minutes, nearly all of it in the SIMIND runs:
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
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import IndicatorBox, SumFunction
from cil.optimisation.utilities import AdaptiveSensitivity

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
from simind_python_connector.recon.cil import (
    CorrectionCallback,
    build_subset_objectives,
)


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "cil_scatter"
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
outputs = measured_adaptor.run()
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
# Subset objectives
# -----------------
# The model is SPECTUB with the flipped attenuation map and no PSF. Six
# Kullback-Leibler objectives, one per group of 10 views, share the same
# additive term. ``build_subset_objectives`` sets up one acquisition model per
# subset.

NUM_SUBSETS = 6
ITERATIONS = 48
UPDATE_EVERY = 12
initial_image = activity_image.get_uniform_copy(1.0)
zero_additive = measured.get_uniform_copy(0)


def make_model():
    matrix = sirf.SPECTUBMatrix()
    matrix.set_attenuation_image(spectub_mu_image)
    # Separately set-up SPECTUB models share this cache in STIR 6.4.
    matrix.set_keep_all_views_in_cache(False)
    matrix.set_resolution_model(0.0, 0.0, False)
    return sirf.AcquisitionModelUsingMatrix(matrix)


full_model = make_model()
full_model.set_up(measured, activity_image)


def ista(objectives, callbacks):
    algorithm = ISTA(
        initial=initial_image,
        f=SumFunction(*objectives.functions),
        g=IndicatorBox(lower=0),
        step_size=1.0,
        preconditioner=AdaptiveSensitivity(full_model, max_iterations=ITERATIONS + 1),
    )
    algorithm.run(ITERATIONS, callbacks=callbacks)
    return algorithm.solution


# %%
# ISTA without scatter correction
# -------------------------------
# ``AdaptiveSensitivity`` makes a unit step an MLEM update for the data term.
# With a zero additive term this is plain maximum-likelihood ISTA.

baseline_objectives = build_subset_objectives(
    measured, zero_additive, make_model, NUM_SUBSETS, initial_image
)
uncorrected = ista(baseline_objectives, callbacks=[])

# %%
# ISTA with SIMIND scatter
# ------------------------
# ``SimindProjector`` simulates the current image with the template geometry.
# ``reference_normaliser`` scales the geometric primary of that simulation to
# the fast model's forward projection of the same image, so the scatter comes
# out on the scale of the data. ``ScatterCorrection`` smooths the estimate over
# 3 bins, across views as well, because a raw Monte Carlo estimate is too noisy
# to subtract. The additive term starts at zero and averages successive
# estimates (``damping=0.5``); with ``stop_at=48`` the refresh at the final
# callback is skipped, so the last iterations use the estimate from iteration 36.
#
# ``seed=None`` leaves the ``RR`` switch unset, so the 8 MPI processes use their
# fixed seeds.

estimate_adaptor = make_adaptor("estimate")
projector = SimindProjector(
    estimate_adaptor,
    template,
    mu_image,
    normalise=reference_normaliser(full_model),
    seed=None,
)
updater = AdditiveUpdater(
    ScatterCorrection(projector, smoothing_fwhm_bins=3.0, smooth_across_views=True),
    UpdateSchedule(every=UPDATE_EVERY, stop_at=ITERATIONS),
    zero_additive,
    damping=0.5,
)
objectives = build_subset_objectives(
    measured, zero_additive, make_model, NUM_SUBSETS, initial_image
)
corrected = ista(objectives, callbacks=[CorrectionCallback(updater, objectives)])

for record in updater.history:
    print(
        f"update at iteration {record.iteration}: "
        f"scale {record.scale:.4g}, scatter sum {record.additive_sum:.4g}, "
        f"{record.seconds:.0f} s"
    )
true_scatter = expected - outputs["geom_coll_primary"]
print(f"True scatter sum: {true_scatter.sum():.4g}")

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
for offset, (title, image) in zip((-0.2, 0, 0.2), scores.items()):
    values = [contrast_recovery(image, name) for name in SPHERES]
    ax.bar(x + offset, values, 0.2, label=title)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.legend()
plt.show()

# %%
# Scatter correction raises the recovery of all four hot spheres above the
# reconstruction without correction, and it stays below the truth's own score.
# The background noise rises as well, because the scatter estimate is noisier
# than the data term it replaces.
