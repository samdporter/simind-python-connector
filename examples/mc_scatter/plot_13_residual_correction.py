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

The expectation is simulated once with four histories per measured count, so
its Monte Carlo variance is a quarter of the Poisson variance. Every arm uses
the same three independent Poisson realisations of it, 6 subsets, three passes
of 12 subiterations (8 full iterations) and the same initial image. The scatter
and residual estimates use a fortieth of the expectation's history budget.

.. warning::
   The first tutorial run simulates the shared expectation and takes about an
   hour with 8 MPI processes. Each of the two corrected arms makes three
   estimates per realisation, at about a minute each; this example takes about
   an hour in all. Later tutorials reuse the cached expectation.

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

from examples import _nema
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
estimate_dir = output_dir / "estimates"
estimate_dir.mkdir(exist_ok=True)

# %%
# NEMA/IEC phantom
# ----------------
# phantomgen's ``pet`` preset has an 8:1 sphere-to-background activity ratio.
# Its two smallest spheres, 10 and 13 mm, are cold; the four hot spheres are
# 17, 22, 28 and 37 mm. The activity is in MBq per voxel. The attenuation map is
# the density times the attenuation coefficient of water at 140 keV.

nema_phantom = _nema.phantom()
nema = nema_phantom.nema
slice_z = nema_phantom.slice_z

activity_image = _nema.to_sirf_image(nema.activity_mbq)
mu_image = _nema.to_sirf_image(nema_phantom.mu)
# STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452)
spectub_mu_image = _nema.to_sirf_image(nema_phantom.mu[..., ::-1])

# %%
# Template
# --------
# The template sets the projection geometry: 48 x 48 bins of 7 mm, 60 views
# over 360 degrees and a detector radius of 250 mm. Both the simulation and the
# reconstruction take their geometry from it.

template = _nema.template(output_dir)

# %%
# "Measured" data
# ---------------
# SIMIND simulates the phantom for Tc-99m with a 126-154 keV window and the
# ma-lehr collimator. The total activity and the time per view are given, so the
# projections come out in counts (Index 25). The 8 MPI processes each simulate
# all views and SIMIND sums them; without an ``RR`` switch their seeds are fixed.
# Poisson noise from a seeded generator is added to the expectation at 2 s per
# view; the three realisations are shared by every arm.

expectation = _nema.simulate_expectation(nema_phantom, template)
realisations = dict(_nema.paired_realisations(expectation, template))
print(
    f"Expectation: {expectation.total:.4g} counts from "
    f"{expectation.histories:.4g} histories, "
    f"quantization loss {expectation.quantization_loss:.2%}"
)
for seed, measured in realisations.items():
    print(f"Seed {seed}: {measured.sum():.4g} measured counts")

# %%
# The fast model
# --------------
# The reconstruction model is SPECTUB with the flipped attenuation map and no
# PSF. ``run_osem_with_corrections`` runs OSEM in passes and calls the updater
# between them. Every arm uses 6 subsets, 12 subiterations per pass and three
# updates (8 full iterations), and the first pass has a zero additive term.

NUM_SUBSETS = _nema.NUM_SUBSETS
SUBITERATIONS_PER_UPDATE = 12
NUM_UPDATES = 3
initial_image = activity_image.get_uniform_copy(1.0)
zero_additive = next(iter(realisations.values())).get_uniform_copy(0)
ESTIMATE_QUANTIZATION_SCALE = 0.1 * _nema.QUANTIZATION_SCALE


def make_acq_model():
    acq_matrix = sirf.SPECTUBMatrix()
    acq_matrix.set_attenuation_image(spectub_mu_image)
    acq_matrix.set_resolution_model(0.0, 0.0, False)
    return sirf.AcquisitionModelUsingMatrix(acq_matrix)


full_model = make_acq_model()
full_model.set_up(next(iter(realisations.values())), activity_image)

fast_only = {
    seed: run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        AdditiveUpdater(None, UpdateSchedule(every=1), zero_additive),
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE * (NUM_UPDATES + 1),
        num_updates=0,
    )
    for seed, measured in realisations.items()
}

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

scatter_corrected = {}
scatter_updaters = {}
for seed, measured in realisations.items():
    projector = SimindProjector(
        _nema.make_adaptor(
            estimate_dir,
            f"scatter_{seed}",
            quantization_scale=ESTIMATE_QUANTIZATION_SCALE,
            histories=1,
        ),
        template,
        mu_image,
        normalise=reference_normaliser(full_model),
        seed=None,
    )
    updater = AdditiveUpdater(
        ScatterCorrection(projector, smoothing_fwhm_bins=3.0, smooth_across_views=True),
        UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
        zero_additive,
        damping=0.5,
    )
    scatter_corrected[seed] = run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        updater,
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE,
        NUM_UPDATES,
    )
    scatter_updaters[seed] = updater

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

residual_corrected = {}
residual_updaters = {}
residual_objectives = {}


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


for seed, measured in realisations.items():
    projector = SimindProjector(
        _nema.make_adaptor(
            estimate_dir,
            f"residual_{seed}",
            quantization_scale=ESTIMATE_QUANTIZATION_SCALE,
            histories=1,
        ),
        template,
        mu_image,
        normalise=reference_normaliser(full_model),
        seed=None,
    )
    residual = RecordedResidualCorrection(
        measured,
        SimindComponent(projector, "all_interactions"),
        full_model,
        zero_additive,
    )
    updater = AdditiveUpdater(
        residual,
        UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
        zero_additive,
        damping=0.5,
    )
    residual_corrected[seed] = run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        updater,
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE,
        NUM_UPDATES,
    )
    residual_updaters[seed] = updater
    residual_objectives[seed] = residual.objectives

# %%
# Update cost and effective objective
# -----------------------------------
# The sum of each additive term in use and the time of the SIMIND run that
# produced its estimate. The effective objective is the accurate model's data
# term after each residual update; a falling value means the accurate model
# fits the data better, beyond Monte Carlo noise.

for seed in realisations:
    for name, updater in (
        ("scatter", scatter_updaters[seed]),
        ("residual", residual_updaters[seed]),
    ):
        for record in updater.history:
            print(
                f"seed {seed} {name} update at iteration {record.iteration}: "
                f"additive sum {record.additive_sum:.4g}, {record.seconds:.0f} s"
            )
    print(
        f"seed {seed} effective objective at each residual update: "
        + ", ".join(f"{value:.6g}" for value in residual_objectives[seed])
    )

# %%
# Reconstructions
# ---------------
# The images are the reconstructions after the fourth pass, so each additive
# term is the damped average described above, not the latest raw estimate.
# Every image is shown relative to its mean in the background mask, so the
# arbitrary scale of the reconstructions does not matter. Each image is the
# mean of the three realisations, and the difference panels are zero-centred
# with a symmetric scale.

background = nema_phantom.background


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


truth_array = nema_phantom.relative_to_background(nema.activity_mbq)
arrays = {
    "Fast only": mean_relative(fast_only.values()),
    "Fast + SIMIND scatter": mean_relative(scatter_corrected.values()),
    "Fast + full residual": mean_relative(residual_corrected.values()),
}

fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
for ax, (title, image) in zip(axes.flat, (("Truth", truth_array), *arrays.items())):
    shown = ax.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

difference = arrays["Fast + full residual"] - arrays["Fast + SIMIND scatter"]
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
limit = np.abs(difference).max()
shown = ax.imshow(difference[slice_z], cmap="coolwarm", vmin=-limit, vmax=limit)
ax.set_title("Full residual − SIMIND scatter")
ax.axis("off")
fig.colorbar(shown, ax=ax, label="difference, relative to background")
plt.show()

# %%
# Contrast recovery and noise
# ---------------------------
# For each hot sphere, the mean in its mask over the mean in the background
# mask, divided by the true ratio of 8. A value of 1 is full recovery. The
# masks include voxels on the sphere edge, which hold less activity, so even
# the truth scores below 1; its score is the ceiling for the reconstructions.
# The 10 and 13 mm spheres are cold and are not scored. The noise is the
# coefficient of variation in the background mask. The error bars are the
# standard error over the three independent realisations.

truth_recovery = _nema.contrast_recovery(nema.activity_mbq, nema_phantom)
truth_cov = _nema.background_cov(nema.activity_mbq, nema_phantom)
scores = {"Truth": (truth_recovery, [0.0] * 4, truth_cov, 0.0)}
for title, images in (
    ("Fast only", fast_only),
    ("Fast + SIMIND scatter", scatter_corrected),
    ("Fast + full residual", residual_corrected),
):
    recovery, recovery_sem = _nema.mean_and_sem(
        [
            _nema.contrast_recovery(image.as_array(), nema_phantom)
            for image in images.values()
        ]
    )
    cov, cov_sem = _nema.mean_and_sem(
        [
            _nema.background_cov(image.as_array(), nema_phantom)
            for image in images.values()
        ]
    )
    scores[title] = (recovery, recovery_sem, float(cov), float(cov_sem))

diameters = [str(d) for d in _nema.SPHERES.values()]
print(f"{'':24}" + "".join(f"{d:>8} mm" for d in diameters) + "     CoV")
for title, (recovery, recovery_sem, cov, cov_sem) in scores.items():
    print(
        f"{title:<24}" + "".join(f"{value:11.2f}" for value in recovery) + f"{cov:8.2f}"
    )

x = np.arange(len(diameters))
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for offset, (title, (recovery, recovery_sem, _, _)) in zip(
    (-0.3, -0.1, 0.1, 0.3), scores.items()
):
    ax.bar(x + offset, recovery, 0.2, yerr=recovery_sem, capsize=2, label=title)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.set_ylim(0, max(truth_recovery) * 1.15)
ax.legend()
plt.show()

# %%
# The printed table and the error bars show where a corrected arm exceeds the
# fast-only recovery by more than the seed-to-seed spread. With three
# realisations the full residual correction does not separate from the scatter
# correction at every sphere, and both stay below the truth's masked score. The
# residual arm trades a higher background CoV for the small 17 mm gain. The
# effective objective printed at each update is the honest check on the
# accurate-model fit: in this run it rises rather than falls after the first
# update, so the damped residual does not keep improving that fit.
