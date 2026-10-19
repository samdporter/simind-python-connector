"""
RDP-regularised reconstruction with SVRG
========================================

Reconstruct a simulated NEMA/IEC phantom acquisition with a SIRF relative
difference prior (RDP) and CIL's stochastic variance reduced gradient (SVRG),
and compare it with OSEM. Both reconstructions use a SIMIND scatter estimate
that is refreshed from the current image during the run.

The data are simulated by SIMIND and reconstructed with a fast SPECTUB model
that has no scatter. SIMIND provides the scatter: all interactions minus the
geometric primary, scaled to the fast model and added to its forward
projection. The RDP is smooth, so it is part of the gradient-based objective
rather than a proximal term. Its strength was picked by eye at this grid.

.. warning::
   This example takes about seventeen minutes, nearly all of it in the SIMIND runs:
   the simulation of the data and one scatter estimate per update for each of
   the two reconstructions. Every run uses 8 MPI processes.

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
from cil.optimisation.functions import (
    Function,
    IndicatorBox,
    SumFunction,
    SVRGFunction,
)
from cil.optimisation.utilities import AdaptiveSensitivity, Sampler

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
from simind_python_connector.recon.osem import run_osem_with_corrections


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "cil_rdp_svrg"
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
slice_z, profile_row = (
    np.argwhere(nema.masks["sphere_6"]).mean(axis=0).round().astype(int)[:2]
)

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
    # PENETRATE takes its energy window from Indexes 20 (upper) and 21 (lower).
    adaptor.add_config_value(20, 154.0)
    adaptor.add_config_value(21, 126.0)
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
# Model and scatter updates
# -------------------------
# The model is SPECTUB with the flipped attenuation map and no PSF. Six subsets
# of 10 views are used by both reconstructions. ``SimindProjector`` simulates
# the current image with the template geometry, and ``reference_normaliser``
# scales the geometric primary of that simulation to the fast model's forward
# projection of the same image. ``ScatterCorrection`` keeps all interactions
# minus the geometric primary and smooths the estimate over 3 bins, across
# views as well. The additive term starts at zero and averages successive
# estimates (``damping=0.5``). ``seed=None`` leaves the ``RR`` switch unset, so
# the 8 MPI processes use their fixed seeds.

NUM_SUBSETS = 6
OSEM_SUBITERATIONS = 12
OSEM_UPDATES = 3
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


def make_scatter_updater(name, schedule):
    projector = SimindProjector(
        make_adaptor(name),
        template,
        mu_image,
        normalise=reference_normaliser(full_model),
        seed=None,
    )
    return AdditiveUpdater(
        ScatterCorrection(projector, smoothing_fwhm_bins=3.0, smooth_across_views=True),
        schedule,
        zero_additive,
        damping=0.5,
    )


# %%
# OSEM
# ----
# The comparison uses the same recipe as the OSEM scatter tutorial: 6 subsets,
# passes of 12 subiterations and three scatter updates, 8 full iterations in
# all.

osem_updater = make_scatter_updater(
    "osem_estimate", UpdateSchedule(every=OSEM_SUBITERATIONS)
)
osem = run_osem_with_corrections(
    measured,
    make_model,
    initial_image,
    osem_updater,
    NUM_SUBSETS,
    OSEM_SUBITERATIONS,
    OSEM_UPDATES,
)

# %%
# RDP + SVRG
# ----------
# The RDP is smooth, so it belongs in the smooth part of ISTA. Each subset
# objective carries the prior with a sixth of its strength, so the SVRG snapshot
# gradient still sums to the full prior. This is the split used by the MaGeZ
# algorithm submitted to the PETRIC reconstruction challenge, whose late-epoch
# step size is 0.5; with CIL's ``AdaptiveSensitivity`` a unit step overshoots at
# the edge of the field of view, so 0.5 is used throughout. The strength, 0.2 in
# the units of the fast model, was picked by eye at this grid: it visibly
# smooths the background without flattening the spheres.
#
# ``SVRGFunction`` draws one subset per iteration and retakes the full gradient
# every 12 iterations. ``CorrectionCallback`` refreshes the scatter estimate
# every three epochs and then the stored SVRG gradients.

RDP_STRENGTH = 0.2
EPOCHS = 12
UPDATE_EVERY = 3 * NUM_SUBSETS
ITERATIONS = EPOCHS * NUM_SUBSETS


class PriorFunction(Function):
    """CIL view of a SIRF prior, which provides its own value and gradient."""

    def __init__(self, prior):
        super().__init__()
        self.prior = prior

    def __call__(self, image):
        return self.prior.get_value(image)

    def gradient(self, image, out=None):
        return self.prior.get_gradient(image, out=out)


prior = sirf.RelativeDifferencePrior()
prior.set_penalisation_factor(RDP_STRENGTH / NUM_SUBSETS)
prior.set_up(initial_image)
prior_term = PriorFunction(prior)

objectives = build_subset_objectives(
    measured, zero_additive, make_model, NUM_SUBSETS, initial_image
)
subset_functions = [SumFunction(f, prior_term) for f in objectives.functions]
svrg = SVRGFunction(
    subset_functions,
    sampler=Sampler.sequential(NUM_SUBSETS),
    snapshot_update_interval=2 * NUM_SUBSETS,
)
svrg_updater = make_scatter_updater(
    "svrg_estimate", UpdateSchedule(every=UPDATE_EVERY, stop_at=ITERATIONS)
)
algorithm = ISTA(
    initial=initial_image,
    f=svrg,
    g=IndicatorBox(lower=0),
    step_size=0.5,
    preconditioner=AdaptiveSensitivity(full_model, max_iterations=ITERATIONS + 1),
)
algorithm.run(ITERATIONS, callbacks=[CorrectionCallback(svrg_updater, objectives)])
svrg_image = algorithm.solution

for name, updater in (("OSEM", osem_updater), ("SVRG", svrg_updater)):
    for record in updater.history:
        print(
            f"{name} update at iteration {record.iteration}: "
            f"scatter sum {record.additive_sum:.4g}, {record.seconds:.0f} s"
        )
true_scatter = expected - outputs["geom_coll_primary"]
print(f"True scatter sum: {true_scatter.sum():.4g}")
print(
    f"Scatter fraction of the measured counts: "
    f"true {true_scatter.sum() / measured.sum():.0%}, "
    f"OSEM {osem_updater.current.sum() / measured.sum():.0%}, "
    f"SVRG {svrg_updater.current.sum() / measured.sum():.0%}"
)

# %%
# Reconstructions and profile
# ---------------------------
# The images are shown relative to their mean in the background mask. The
# profile runs along x through the centres of the 22 and 37 mm spheres.

background = nema.masks["background"] > 0


def relative_to_background(array):
    return array / array[background].mean()


truth_array = relative_to_background(nema.activity_mbq)
recon_arrays = {
    "OSEM": relative_to_background(osem.as_array()),
    "RDP + SVRG": relative_to_background(svrg_image.as_array()),
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, (title, image) in zip(axes, (("Truth", truth_array), *recon_arrays.items())):
    shown = ax.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for title, image in (("Truth", truth_array), *recon_arrays.items()):
    ax.plot(
        image[slice_z, profile_row], "k--" if title == "Truth" else "-", label=title
    )
ax.set_xlabel("x (voxel)")
ax.set_ylabel("relative to background")
ax.legend()
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
# At this budget RDP + SVRG is visibly smoother than OSEM and recovers less of
# each sphere: it is a different point on the noise-recovery trade-off, not a
# better image. The comparison also changes the algorithm and step size, not
# only the prior, so it does not isolate the prior's effect. With one noise
# realisation the table does not rank the two methods, and the 10 and 13 mm
# spheres remain cold.
