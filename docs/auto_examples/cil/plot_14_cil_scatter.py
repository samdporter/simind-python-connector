"""
Monte Carlo scatter correction with CIL
=======================================

Reconstruct a simulated NEMA/IEC phantom acquisition twice with CIL's ISTA:
once without scatter correction, and once with a SIMIND scatter estimate that
is refreshed from the current image during the run.

The data are simulated by SIMIND with four histories per measured count, so
the Monte Carlo variance of the expectation is a quarter of the Poisson
variance of the acquisition. The reconstruction uses a fast SPECTUB model that
has no scatter. SIMIND provides the scatter: all interactions minus the
geometric primary, scaled to the fast model and added to its forward
projection. The subset objectives are CIL Kullback-Leibler functions, one per
group of views, and a ``CorrectionCallback`` swaps in a new scatter estimate
every 12 iterations.

The pilot below reconstructs one realisation for 24, 48 and 96 iterations. The
tutorial uses 48: the recovery is stable by then and the extra iterations add
noise. The final bars use three independent Poisson realisations.

.. warning::
   The first tutorial run simulates the shared expectation and takes about an
   hour with 8 MPI processes. Each scatter estimate uses a fortieth of that
   history budget and takes about a minute; this example makes nine estimates
   in the final run and eight in the iteration pilot (its 48-iteration point
   is the final run), so it takes about an hour. Later tutorials reuse the
   cached expectation.

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

from examples import _nema
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
# view; the three realisations are shared by both arms.

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
# Subset objectives
# -----------------
# The model is SPECTUB with the flipped attenuation map and no PSF. Six
# Kullback-Leibler objectives, one per group of 10 views, share the same
# additive term. ``build_subset_objectives`` sets up one acquisition model per
# subset.

NUM_SUBSETS = _nema.NUM_SUBSETS
ITERATIONS = 48
UPDATE_EVERY = 12
initial_image = activity_image.get_uniform_copy(1.0)
zero_additive = next(iter(realisations.values())).get_uniform_copy(0)
ESTIMATE_QUANTIZATION_SCALE = 0.1 * _nema.QUANTIZATION_SCALE


def make_model():
    matrix = sirf.SPECTUBMatrix()
    matrix.set_attenuation_image(spectub_mu_image)
    # Separately set-up SPECTUB models share this cache in STIR 6.4.
    matrix.set_keep_all_views_in_cache(False)
    matrix.set_resolution_model(0.0, 0.0, False)
    return sirf.AcquisitionModelUsingMatrix(matrix)


full_model = make_model()
full_model.set_up(next(iter(realisations.values())), activity_image)


def ista(objectives, callbacks, iterations=ITERATIONS):
    algorithm = ISTA(
        initial=initial_image,
        f=SumFunction(*objectives.functions),
        g=IndicatorBox(lower=0),
        step_size=1.0,
        preconditioner=AdaptiveSensitivity(full_model, max_iterations=iterations + 1),
    )
    algorithm.run(iterations, callbacks=callbacks)
    return algorithm.solution


def make_updater(seed, stop_at):
    projector = SimindProjector(
        _nema.make_adaptor(
            estimate_dir,
            f"estimate_{seed}",
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
        UpdateSchedule(every=UPDATE_EVERY, stop_at=stop_at),
        zero_additive,
        damping=0.5,
    )
    objectives = build_subset_objectives(
        realisations[seed], zero_additive, make_model, NUM_SUBSETS, initial_image
    )
    return updater, objectives


# %%
# ISTA without scatter correction
# -------------------------------
# ``AdaptiveSensitivity`` makes a unit step an MLEM update for the data term.
# With a zero additive term this is plain maximum-likelihood ISTA.

uncorrected = {}
for seed, measured in realisations.items():
    objectives = build_subset_objectives(
        measured, zero_additive, make_model, NUM_SUBSETS, initial_image
    )
    uncorrected[seed] = ista(objectives, callbacks=[])

# %%
# ISTA with SIMIND scatter
# ------------------------
# ``SimindProjector`` simulates the current image with the template geometry.
# ``reference_normaliser`` scales the geometric primary of that simulation to
# the fast model's forward projection of the same image, so the scatter comes
# out on the scale of the data. ``ScatterCorrection`` smooths the estimate over
# 3 bins, across views as well. The additive term starts at zero and averages
# successive estimates (``damping=0.5``); with ``stop_at=48`` the refresh at the
# final callback is skipped, so the last iterations use the estimate from
# iteration 36.
#
# ``seed=None`` leaves the ``RR`` switch unset, so the 8 MPI processes use
# their fixed seeds.

corrected = {}
updaters = {}
for seed in realisations:
    updater, objectives = make_updater(seed, ITERATIONS)
    corrected[seed] = ista(
        objectives, callbacks=[CorrectionCallback(updater, objectives)]
    )
    updaters[seed] = updater

for seed, updater in updaters.items():
    for record in updater.history:
        print(
            f"seed {seed} update at iteration {record.iteration}: "
            f"scale {record.scale:.4g}, scatter sum {record.additive_sum:.4g}, "
            f"{record.seconds:.0f} s"
        )

true_scatter = expectation.true_scatter
print(f"True scatter sum: {true_scatter.sum():.4g}")
for seed in realisations:
    print(
        f"Seed {seed} scatter fraction: "
        f"true {true_scatter.sum() / realisations[seed].sum():.0%}, "
        f"estimated {updaters[seed].current.sum() / realisations[seed].sum():.0%}"
    )

# %%
# Iteration pilot
# ---------------
# One realisation at 24, 48 and 96 iterations, with the same data and update
# schedule. The 48-iteration point is the final reconstruction above, so only
# the shorter and longer runs make new estimates. The table and the figure give
# the contrast recovery and background noise.

pilot_seed = next(iter(realisations))
diameters = [str(d) for d in _nema.SPHERES.values()]
pilot = {
    48: (
        corrected[pilot_seed],
        *_nema.contrast_recovery_scores(corrected[pilot_seed].as_array(), nema_phantom),
    )
}
for iterations in (24, 96):
    updater, objectives = make_updater(pilot_seed, iterations)
    image = ista(
        objectives,
        callbacks=[CorrectionCallback(updater, objectives)],
        iterations=iterations,
    )
    pilot[iterations] = (
        image,
        *_nema.contrast_recovery_scores(image.as_array(), nema_phantom),
    )
    print(
        f"{iterations:>3} iterations, {len(updater.history)} updates: "
        + " ".join(f"{value:.2f}" for value in pilot[iterations][1])
        + f"  CoV {pilot[iterations][2]:.2f}"
    )
print(
    " 48 iterations, 3 updates: "
    + " ".join(f"{value:.2f}" for value in pilot[48][1])
    + f"  CoV {pilot[48][2]:.2f}"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for sphere, diameter in enumerate(diameters):
    axes[0].plot(
        list(pilot),
        [pilot[iterations][1][sphere] for iterations in pilot],
        "o-",
        label=f"{diameter} mm",
    )
axes[0].set_xlabel("Iterations")
axes[0].set_ylabel("Contrast recovery")
axes[0].legend()
axes[1].plot(
    list(pilot),
    [pilot[iterations][2] for iterations in pilot],
    "o-",
    color="k",
)
axes[1].set_xlabel("Iterations")
axes[1].set_ylabel("Background CoV")
plt.show()

# %%
# Reconstructions
# ---------------
# Both images are shown relative to their mean in the background mask, so the
# arbitrary scale of the reconstructions does not matter. Each image is the
# mean of the three realisations, and the difference panel is zero-centred.

background = nema_phantom.background


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


truth_array = nema_phantom.relative_to_background(nema.activity_mbq)
arrays = {
    "No scatter correction": mean_relative(uncorrected.values()),
    "SIMIND scatter": mean_relative(corrected.values()),
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, (title, image) in zip(axes, (("Truth", truth_array), *arrays.items())):
    shown = ax.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

difference = arrays["SIMIND scatter"] - arrays["No scatter correction"]
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
limit = np.abs(difference).max()
shown = ax.imshow(difference[slice_z], cmap="coolwarm", vmin=-limit, vmax=limit)
ax.set_title("SIMIND scatter − no correction")
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
    ("No scatter correction", uncorrected),
    ("SIMIND scatter", corrected),
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

print(f"{'':24}" + "".join(f"{d:>8} mm" for d in diameters) + "     CoV")
for title, (recovery, recovery_sem, cov, cov_sem) in scores.items():
    print(
        f"{title:<24}" + "".join(f"{value:11.2f}" for value in recovery) + f"{cov:8.2f}"
    )

x = np.arange(len(diameters))
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for offset, (title, (recovery, recovery_sem, _, _)) in zip(
    (-0.2, 0, 0.2), scores.items()
):
    ax.bar(x + offset, recovery, 0.2, yerr=recovery_sem, capsize=2, label=title)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.set_ylim(0, max(truth_recovery) * 1.15)
ax.legend()
plt.show()

# %%
# The printed table and the error bars show where the scatter correction
# changes recovery by more than the seed-to-seed spread; the 17 mm sphere is
# the least separated because it is the smallest and the most partial-volume
# affected. All arms stay below the truth's own score. The background noise
# rises because the scatter estimate is noisier than the data term it replaces,
# so the correction is a noise-for-recovery trade rather than a free gain.
